#!/usr/bin/env python3
"""
Hybrid Live Paper Trading System

This script uses HYBRID FIXES to make the strategy viable:
- Risk-Reward: 3:1 (TP 0.9%, SL 0.3%) - lowers breakeven from 33% → 25%
- Daily bias filter for trend alignment
- Volatility filter for signal quality
- Target: 30-40% win rate (realistic based on walk-forward validation)
"""

import asyncio
import httpx
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.ml.inference import MLInference
from src.execution.position_tracker import PositionTracker, Position
from src.data.models import DollarBar, SilverBulletSetup
from src.data.auth_v3 import TradeStationAuthV3
from src.data.contract_detector import ContractDetector
from src.data.market_data_validator import MarketDataValidator
from src.detection.pipeline import DetectionPipeline
from src.detection.silver_bullet_detection import detect_silver_bullet_setup
from src.ml.signal_filter import SignalFilter
from src.risk.position_sizer import PositionSizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedPaperTrader:
    """Paper trading system using OPTIMIZED ML model and strategy."""

    def __init__(self, access_token: str):
        """Initialize paper trader with optimized settings.

        Args:
            access_token: Valid TradeStation access token
        """
        self.access_token = access_token
        self.running = False
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        # Initialize V3 auth with auto-refresh (loads refresh token from .env)
        self.auth = TradeStationAuthV3.from_file(".access_token")

        # 🔥 HYBRID MODEL LOADING
        # Use the 30-minute model (will be retrained on 2025-2026 data in Phase 2)
        self.ml_inference = MLInference(
            model_dir="models/xgboost/30_minute/"
        )
        logger.info("✅ Loaded model (Phase 2: Will retrain on 2025-2026 data)")

        # Use optimized probability threshold from sb_params_optimized.json
        self.signal_filter = SignalFilter(threshold=0.65)
        logger.info("✅ Using 0.65 probability threshold")

        # Position sizing with 3:1 risk-reward ratio
        self.position_sizer = PositionSizer(
            risk_reward_ratio=3.0,  # 3:1 risk-reward
            atr_multiplier=0.3,      # Wider stops for 3:1 R:R
        )
        self.position_tracker = PositionTracker()

        # Initialize contract detector and market data validator
        self.contract_detector = ContractDetector(access_token)
        self.validator = MarketDataValidator()

        # Initialize detection components
        self._bar_queue: asyncio.Queue[DollarBar] = asyncio.Queue(maxsize=100)
        self._signal_queue: asyncio.Queue[SilverBulletSetup] = asyncio.Queue(maxsize=50)

        # Initialize detection pipeline
        self.detection_pipeline = DetectionPipeline(
            input_queue=self._bar_queue,
            signal_queue=self._signal_queue,
        )

        # Track bars for ML predictions and detection
        self.recent_bars: list[DollarBar] = []
        self.historical_bars: list[DollarBar] = []  # For daily bias calculation

        # Track detected setups for ML inference
        self.detected_setups: list[SilverBulletSetup] = []

        # 🎯 HYBRID FIXES - 3:1 Risk-Reward (Phase 1)
        # Updated from 2:1 to 3:1 to lower breakeven win rate from 33% → 25%
        self.take_profit_pct = 0.9 / 100  # 0.9% as decimal (was 0.4%)
        self.stop_loss_pct = 0.3 / 100    # 0.3% as decimal (was 0.2%)
        self.max_bars = 50
        self.risk_reward_ratio = 3.0      # 3:1 R:R (was 2:1)

        logger.info(f"✅ Hybrid parameters: TP={self.take_profit_pct*100:.1f}%, SL={self.stop_loss_pct*100:.1f}%, R:R={self.risk_reward_ratio}:1, MaxBars={self.max_bars}")

    async def fetch_live_quotes(self, symbol: str = "MNQH26") -> dict:
        """Fetch live market data for a symbol.

        Args:
            symbol: Futures contract symbol

        Returns:
            Dictionary with quote data
        """
        # Refresh auth token if needed
        try:
            current_token = await self.auth.authenticate()
            self.headers["Authorization"] = f"Bearer {current_token}"
        except Exception as e:
            logger.error(f"Auth refresh failed: {e}")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.tradestation.com/v3/data/quote/{symbol}",
                headers=self.headers,
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return {}

    async def fetch_historical_bars(self, symbol: str = "MNQH26", bars_count: int = 2000) -> list[DollarBar]:
        """Fetch historical bars from local HDF5 files to pre-populate detection context.

        Args:
            symbol: Futures contract symbol (unused, kept for compatibility)
            bars_count: Number of bars to fetch (default: 2000)

        Returns:
            List of DollarBar objects
        """
        import h5py
        from pathlib import Path

        bars = []
        data_dir = Path("data/processed/time_bars/")

        # Load current month and previous month's data
        current_date = datetime.now()
        months_to_load = [
            current_date.strftime('%Y%m'),  # Current month
            (current_date - pd.DateOffset(months=1)).strftime('%Y%m'),  # Previous month
            (current_date - pd.DateOffset(months=2)).strftime('%Y%m'),  # 2 months ago
        ]

        logger.info(f"Loading historical bars from local HDF5 files...")

        for month_str in months_to_load:
            filename = f"MNQ_time_bars_5min_{month_str}.h5"
            file_path = data_dir / filename

            if file_path.exists():
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['dollar_bars'][:]

                    # Convert to DataFrame first
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Convert to DollarBar objects
                    for _, row in df.tail(len(df)).iterrows():
                        bar = DollarBar(
                            timestamp=row['timestamp'].to_pydatetime(),
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(row['volume']),
                            notional_value=float(row['notional_value']),
                        )
                        bars.append(bar)

                    logger.info(f"✅ Loaded {len(df)} bars from {filename}")

                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.debug(f"File not found: {filename}")

        # Sort by timestamp and limit to bars_count
        bars.sort(key=lambda x: x.timestamp)
        if len(bars) > bars_count:
            bars = bars[-bars_count:]

        logger.info(f"✅ Loaded {len(bars)} historical bars from local files")
        return bars

    def create_mock_bar_from_quote(self, quote: dict) -> DollarBar | None:
        """Create a DollarBar object from live quote data.

        Args:
            quote: Quote data from API

        Returns:
            DollarBar object or None if validation fails
        """
        current_time = datetime.now(timezone.utc)

        # Extract price data
        last_price = quote.get("Last", quote.get("Close", 0))
        high_price = quote.get("High", last_price)
        low_price = quote.get("Low", last_price)
        open_price = quote.get("Open", last_price)
        close_price = quote.get("Close", last_price)
        volume = quote.get("Volume", 0)

        notional_value = last_price * 2

        try:
            bar = DollarBar(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=int(volume),
                notional_value=notional_value,
            )

            if not self.validator.validate_bar_for_trading(bar):
                logger.debug("Bar failed business logic validation")
                return None

            return bar

        except Exception as e:
            logger.error(f"Failed to create DollarBar from quote: {e}")
            return None

    async def initialize_detection_context(self, symbol: str = "MNQM26") -> None:
        """Initialize detection pipeline with historical bars for immediate swing detection.

        Downloads fresh historical bars from TradeStation API at runtime (not backtesting data).
        """
        print(f"\n📥 Fetching fresh historical bars for {symbol}...")

        try:
            historical_bars = await self.fetch_historical_bars(symbol, bars_count=2000)

            if not historical_bars:
                print("⚠️  No historical bars available (API limitation)")
                print("   System will build detection context over time from live bars")
                return

            print(f"✅ Downloaded {len(historical_bars)} historical bars")

            # Process historical bars through detection pipeline to build swing state
            print(f"🔧 Building swing point context from historical bars...")

            swing_points_found = 0
            detection_errors = 0

            for i, bar in enumerate(historical_bars):
                try:
                    await self.detection_pipeline.process_bar(bar)
                    swing_points_found += 1
                except TypeError as e:
                    detection_errors += 1
                    if detection_errors <= 3:
                        logger.debug(f"Detection needs more historical data: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Error processing historical bar {i}: {e}")
                    continue

            # Store historical bars for daily bias calculation
            self.historical_bars = historical_bars
            # Store last 50 for ML
            self.recent_bars = historical_bars[-50:]

            print(f"✅ Detection context initialized:")
            print(f"   Processed {swing_points_found} bars through detection")
            print(f"   Detection errors (warmup): {detection_errors}")
            print(f"   Bars available for ML: {len(self.recent_bars)}")
            print(f"   Total historical bars loaded: {len(historical_bars)}")

        except Exception as e:
            print(f"⚠️  Historical bar download failed: {e}")
            import traceback
            traceback.print_exc()
            print("   System will build detection context from live bars")

    def calculate_daily_bias(self) -> bool:
        """Calculate daily trend bias using historical bars.

        Returns True for uptrend (close > SMA50), False for downtrend.
        """
        if len(self.historical_bars) < 50:
            logger.warning("Not enough historical bars for daily bias calculation")
            return True  # Default to uptrend

        # Convert to DataFrame for easier calculation
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'close': bar.close
            }
            for bar in self.historical_bars
        ]).set_index('timestamp')

        # Calculate SMA50
        sma50 = df['close'].rolling(50).mean()
        current_close = df['close'].iloc[-1]
        current_sma50 = sma50.iloc[-1]

        is_uptrend = current_close > current_sma50

        logger.info(f"📊 Daily Bias: {'UPTREAD' if is_uptrend else 'DOWNTREND'} (Close: {current_close:.2f}, SMA50: {current_sma50:.2f})")

        return is_uptrend

    def calculate_volatility_filter(self) -> bool:
        """Calculate volatility filter using ATR%.

        Returns True if volatility is sufficient (ATR% >= 0.3%).
        """
        if len(self.historical_bars) < 14:
            logger.warning("Not enough historical bars for volatility calculation")
            return True  # Default to allow trading

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close
            }
            for bar in self.historical_bars
        ]).set_index('timestamp')

        # Calculate ATR (simplified using True Range)
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low'] - df['close'].shift(1)).abs()
        }).max(axis=1)

        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / df['close'].iloc[-1]

        min_atr_pct = 0.003  # 0.3%
        is_sufficient_vol = atr_pct >= min_atr_pct

        logger.info(f"📊 Volatility Filter: {'PASS' if is_sufficient_vol else 'FAIL'} (ATR%: {atr_pct*100:.2f}% >= {min_atr_pct*100:.1f}%)")

        return is_sufficient_vol

    async def detect_setups_from_bars(self, bars: list[DollarBar]) -> list[SilverBulletSetup]:
        """Detect Silver Bullet setups from accumulated bars.

        Args:
            bars: List of DollarBar objects to analyze

        Returns:
            List of detected SilverBulletSetup objects
        """
        if len(bars) < 20:
            return []

        # Process bars through detection pipeline
        detection_errors = 0
        for bar in bars:
            try:
                await self.detection_pipeline.process_bar(bar)
            except TypeError as e:
                detection_errors += 1
                if detection_errors <= 3:
                    logger.debug(f"Detection needs more historical data: {e}")
                continue
            except Exception as e:
                logger.debug(f"Error processing bar through detection: {e}")
                continue

        # Collect detected setups from signal queue
        setups = []
        while not self._signal_queue.empty():
            try:
                setup = self._signal_queue.get_nowait()
                setups.append(setup)
            except asyncio.QueueEmpty:
                break

        if setups:
            logger.info(f"Detected {len(setups)} Silver Bullet setup(s)")

        return setups

    async def submit_sim_trade(
        self,
        setup: SilverBulletSetup,
        position_result,
        probability: float,
        horizon: int
    ):
        """Submit trade to TradeStation SIM account.

        Args:
            setup: Silver Bullet setup
            position_result: Position sizing result
            probability: ML probability score
            horizon: Time horizon in minutes
        """
        try:
            # Determine order direction
            action = "BUY" if setup.direction == "bullish" else "SELL"

            print(f"\n📝 Submitting {action} order to SIM account...")
            print(f"   Symbol: MNQM26")
            print(f"   Quantity: {position_result.position_size} contracts")
            print(f"   Order Type: Market")

            # For SIM trading, we'll use the TradeStation API v3 order endpoint
            async with httpx.AsyncClient() as client:
                # Get current token
                current_token = await self.auth.authenticate()
                self.headers["Authorization"] = f"Bearer {current_token}"

                # Submit market order to SIM account
                order_payload = {
                    "symbol": "MNQM26",
                    "quantity": position_result.position_size,
                    "orderType": "Market",
                    "side": action.upper(),
                    "duration": "DAY",
                    "account": "SIM",  # Explicitly use SIM account
                }

                response = await client.post(
                    "https://api.tradestation.com/v3/orderexecution/confirmorder",
                    headers=self.headers,
                    json=order_payload,
                )

                if response.status_code == 200:
                    order_result = response.json()

                    # Extract order ID if available
                    order_id = order_result.get("OrderID", order_result.get("Orders", [{}])[0].get("OrderID", "PENDING"))

                    print(f"\n✅ Order SUBMITTED to SIM account!")
                    print(f"   Order ID: {order_id}")
                    print(f"   Status: Pending execution")

                    # Create position in tracker
                    position = Position(
                        order_id=order_id,
                        symbol="MNQM26",
                        direction=setup.direction,
                        quantity=position_result.position_size,
                        entry_price=position_result.entry_price,
                        stop_loss=position_result.stop_loss,
                        probability=probability,
                    )

                    self.position_tracker.add_position(position)
                    print(f"\n💰 Position tracked:")
                    print(f"   ID: {order_id}")
                    print(f"   {setup.direction.upper()} {position_result.position_size} contracts @ ${position_result.entry_price:.2f}")
                    print(f"   Stop Loss: ${position_result.stop_loss:.2f}")
                    print(f"   P(Success): {probability:.2%}")

                else:
                    error_msg = response.text
                    print(f"\n❌ Order submission failed: {response.status_code}")
                    print(f"   Error: {error_msg}")
                    logger.error(f"Order submission failed: {error_msg}")

        except Exception as e:
            logger.error(f"Failed to submit SIM trade: {e}")
            import traceback
            traceback.print_exc()

    async def run_trading_loop(self, symbol: str = "MNQH26"):
        """Run the main paper trading loop with OPTIMIZED strategy.

        Args:
            symbol: Symbol to trade
        """
        self.running = True

        # Detect active contract first
        print(f"\n📋 Detecting active contract for {symbol}...")
        try:
            active_symbol = await self.contract_detector.detect_active_contract(symbol)
            print(f"✅ Active contract: {active_symbol}")
            symbol = active_symbol
        except Exception as e:
            print(f"⚠️  Contract detection failed: {e}")
            print(f"   Using provided symbol: {symbol}")

        # Initialize detection context with historical bars
        await self.initialize_detection_context(symbol)

        print(f"\n🎯 HYBRID Paper Trading Started - Monitoring {symbol}")
        print("="*70)
        print("🔥 HYBRID FIXES ACTIVE (3:1 R:R, Breakeven: 25% Win Rate)")
        print("   Realistic Target: 30-40% Win Rate")
        print("="*70)

        iteration = 0
        while self.running:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            try:
                # Fetch live quote
                quote = await self.fetch_live_quotes(symbol)

                if not quote:
                    print("⚠️  No quote data available")
                    await asyncio.sleep(10)
                    continue

                # Validate market status
                market_status = self.validator.validate_market_status(quote)
                print(f"📊 {symbol} Market Status: {market_status['market_state']}")
                if market_status.get('reason'):
                    print(f"   Reason: {market_status['reason']}")

                # Display market data
                last_price = quote.get("Last", 0)
                bid = quote.get("Bid", 0)
                ask = quote.get("Ask", 0)
                volume = quote.get("Volume", 0)

                print(f"📊 {symbol} Live Data:")
                print(f"   Last: {last_price}")
                print(f"   Bid:  {bid}")
                print(f"   Ask:  {ask}")
                print(f"   Volume: {volume}")

                # Check if market is tradable
                if not market_status.get("is_tradable", False):
                    print(f"⚠️  Market not tradable: {market_status.get('reason', 'Unknown')}")
                    print("   Skipping this iteration")
                    await asyncio.sleep(10)
                    continue

                # Create bar from quote
                current_bar = self.create_mock_bar_from_quote(quote)

                if current_bar is None:
                    print("⚠️  Bar validation failed - no tradable data")
                    await asyncio.sleep(5)
                    continue

                self.recent_bars.append(current_bar)
                self.historical_bars.append(current_bar)

                # Keep only last 50 bars for ML
                if len(self.recent_bars) > 50:
                    self.recent_bars = self.recent_bars[-50:]

                # Keep last 2000 for daily bias
                if len(self.historical_bars) > 2000:
                    self.historical_bars = self.historical_bars[-2000:]

                # Run detection and ML inference when we have 50 bars
                if len(self.recent_bars) >= 50:
                    try:
                        # Step 0: Apply HYBRID filters
                        daily_bias = self.calculate_daily_bias()
                        volatility_ok = self.calculate_volatility_filter()

                        if not daily_bias or not volatility_ok:
                            print(f"\n🔍 HYBRID filters not met - skipping detection")
                            print(f"   Daily Bias: {'✅' if daily_bias else '❌'}")
                            print(f"   Volatility: {'✅' if volatility_ok else '❌'}")
                        else:
                            # Step 1: Detect Silver Bullet setups
                            setups = await self.detect_setups_from_bars(self.recent_bars)

                            if not setups:
                                print(f"\n🔍 No Silver Bullet setups detected in {len(self.recent_bars)} bars")
                            else:
                                print(f"\n🎯 {len(setups)} Silver Bullet setup(s) detected!")
                                for i, setup in enumerate(setups, 1):
                                    print(f"   Setup {i}: {setup.direction} | Priority: {setup.priority} | Confluence: {setup.confluence_count}")

                                    # Additional filter: Check if setup aligns with daily bias
                                    if setup.direction == "bullish" and not daily_bias:
                                        print(f"   ❌ Filtered: Bullish setup in downtrend")
                                        continue
                                    if setup.direction == "bearish" and daily_bias:
                                        print(f"   ❌ Filtered: Bearish setup in uptrend")
                                        continue

                                    # Step 2: Run ML inference
                                    print(f"\n🤖 Running ML inference on setup {i}...")
                                    predictions = self.ml_inference.predict_all_horizons(setup)

                                    # Show predictions for all horizons
                                    for horizon, result in predictions.items():
                                        probability = result.get("probability", 0.0)
                                        print(f"   {horizon}min horizon: P(success) = {probability:.2%}")

                                    # Use 30-minute horizon as default
                                    default_horizon = 30
                                    if default_horizon in predictions:
                                        probability = predictions[default_horizon]["probability"]

                                        # Step 3: Filter by threshold (0.65)
                                        if self.signal_filter.should_allow(probability):
                                            print(f"\n✅ Signal ALLOWED: P({probability:.2%}) >= 0.65 threshold")

                                            # Step 4: Calculate position size with 3:1 R:R
                                            try:
                                                position_result = self.position_sizer.calculate_position(
                                                    signal=setup,
                                                    atr=50.0,  # Default ATR for MNQ
                                                )

                                                if position_result.valid:
                                                    print(f"\n📊 Position Size Calculated (3:1 R:R):")
                                                    print(f"   Entry: ${position_result.entry_price:.2f}")
                                                    print(f"   Stop Loss: ${position_result.stop_loss:.2f} (0.3%)")
                                                    print(f"   Take Profit: ${position_result.take_profit:.2f} (0.9%)")
                                                    print(f"   R:R Ratio: {position_result.risk_reward_ratio}:1")
                                                    print(f"   Contracts: {position_result.position_size}")
                                                    print(f"   Risk: ${position_result.dollar_risk:.2f}")

                                                    # Step 5: Submit trade to SIM account
                                                    await self.submit_sim_trade(setup, position_result, probability, default_horizon)
                                                else:
                                                    print(f"\n⚠️  Position size invalid: {position_result.validation_reason}")

                                            except Exception as e:
                                                logger.error(f"Position sizing failed: {e}")
                                        else:
                                            print(f"\n❌ Signal FILTERED: P({probability:.2%}) < 0.65 threshold")

                    except Exception as e:
                        logger.error(f"Detection/ML inference failed: {e}")
                        import traceback
                        traceback.print_exc()

                # Show positions
                positions = self.position_tracker.get_all_positions()
                if positions:
                    print(f"\n💰 Current Positions: {len(positions)}")
                    for pos in positions:
                        print(f"   {pos.order_id}: {pos.direction} {pos.quantity} contracts")
                        print(f"   Entry: ${pos.entry_price:.2f} | P&L: ${pos.unrealized_pnl:.2f}")

                print(f"\n🤖 ML Status: {len(self.recent_bars)} bars collected")
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

            # Wait before next iteration
            await asyncio.sleep(5)

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        print("\n🛑 Stopping paper trading...")


async def main():
    """Main entry point."""

    print("\n" + "="*70)
    print("🎯 HYBRID Live Paper Trading System")
    print("="*70)
    print("🔥 Using Hybrid Fixes (3:1 R:R, Target: 30-40% Win Rate)")
    print("="*70)
    print()

    # Check authentication
    print("📋 Step 1: Checking authentication...")
    try:
        with open(".access_token", "r") as f:
            access_token = f.read().strip()
        if not access_token or len(access_token) < 50:
            print("❌ No valid access token found!")
            return
        print("✅ Access token found")
    except FileNotFoundError:
        print("❌ No .access_token file found!")
        return

    print()

    # Test connection
    print("📋 Step 2: Testing API connection...")
    trader = OptimizedPaperTrader(access_token)

    # Start automatic token refresh every 10 minutes
    await trader.auth.start_auto_refresh(interval_minutes=10)
    print("✅ Auto-refresh enabled (every 10 minutes)")

    quote = await trader.fetch_live_quotes("MNQH26")

    if not quote:
        print("❌ Failed to connect to TradeStation API")
        print("   Your access token may have expired")
        print("   Run: .venv/bin/python exchange_token_simple.py <code>")
        return

    print("✅ API connection successful")
    print(f"   MNQH26 Last: {quote.get('Last', 'N/A')}")
    print()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        trader.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start trading loop
        await trader.run_trading_loop("MNQH26")

    except KeyboardInterrupt:
        trader.stop()
        # Stop auto-refresh
        await trader.auth.cleanup()
        print("\n👋 Paper trading stopped")

    print("\n" + "="*70)
    print("✅ Session Complete")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

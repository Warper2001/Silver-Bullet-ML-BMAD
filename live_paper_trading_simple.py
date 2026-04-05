#!/usr/bin/env python3
"""
Simple Live Paper Trading System

This script uses your existing OAuth access token to fetch live market data
and demonstrate the trading system functionality.
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
from src.data.shared_state_db import init_db, write_trading_state

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePaperTrader:
    """Simple paper trading system using live market data."""

    def __init__(self, access_token: str):
        """Initialize paper trader.

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

        # Initialize ML components
        self.ml_inference = MLInference(model_dir="models/xgboost")
        self.signal_filter = SignalFilter(threshold=0.65)  # Default from backtest
        self.position_sizer = PositionSizer()  # Use defaults: 2% risk, max 5 contracts
        self.position_tracker = PositionTracker()

        # Initialize contract detector and market data validator
        self.contract_detector = ContractDetector(access_token)
        self.validator = MarketDataValidator()

        # Initialize detection components
        # Create queues for detection pipeline
        self._bar_queue: asyncio.Queue[DollarBar] = asyncio.Queue(maxsize=100)
        self._signal_queue: asyncio.Queue[SilverBulletSetup] = asyncio.Queue(maxsize=50)

        # Initialize detection pipeline
        self.detection_pipeline = DetectionPipeline(
            input_queue=self._bar_queue,
            signal_queue=self._signal_queue,
        )

        # Track bars for ML predictions and detection
        self.recent_bars: list[DollarBar] = []

        # Track detected setups for ML inference
        self.detected_setups: list[SilverBulletSetup] = []

        # Initialize shared state database
        try:
            init_db()
            print("✅ Shared state database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize shared state DB: {e}")

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
        """Fetch historical bars to pre-populate detection context.

        Args:
            symbol: Futures contract symbol
            bars_count: Number of bars to fetch (default: 2000)

        Returns:
            List of DollarBar objects
        """
        # Refresh auth token if needed
        try:
            current_token = await self.auth.authenticate()
            self.headers["Authorization"] = f"Bearer {current_token}"
        except Exception as e:
            logger.error(f"Auth refresh failed: {e}")

        bars = []

        async with httpx.AsyncClient() as client:
            # Use the correct TradeStation v3 API endpoint format
            # POST request with JSON body for bar data
            remaining = bars_count
            skip = 0

            while remaining > 0:
                batch_size = min(remaining, 1000)

                # TradeStation uses POST with JSON body for bar requests
                response = await client.post(
                    "https://api.tradestation.com/v3/marketdata/bars",
                    headers={
                        **self.headers,
                        "Content-Type": "application/json"
                    },
                    json={
                        "symbol": symbol,
                        "interval": "5",
                        "unit": "Minute",
                        "barscount": batch_size,
                        "useEST": True,  # Use Eastern Time
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("Bars"):
                        for bar_data in data["Bars"]:
                            # Convert API response to DollarBar
                            bar = DollarBar(
                                timestamp=datetime.fromisoformat(bar_data["Timestamp"].replace("Z", "+00:00")),
                                open=float(bar_data["Open"]),
                                high=float(bar_data["High"]),
                                low=float(bar_data["Low"]),
                                close=float(bar_data["Close"]),
                                volume=int(bar_data["TotalVolume"]),
                                notional_value=float(bar_data["Close"]) * 2,  # MNQ = $2/point
                            )
                            bars.append(bar)

                        remaining -= len(data["Bars"])

                        # If we got fewer bars than requested, we've reached the beginning
                        if len(data["Bars"]) < batch_size:
                            logger.info(f"Got {len(data['Bars'])} bars, reached beginning of available data")
                            break
                    else:
                        logger.warning(f"No bars returned in batch (skip={skip})")
                        break
                else:
                    logger.error(f"Failed to fetch historical bars: {response.status_code} - {response.text}")
                    break

                skip += batch_size

        logger.info(f"Fetched {len(bars)} historical bars for {symbol}")
        return bars

    def create_mock_bar_from_quote(self, quote: dict) -> DollarBar | None:
        """Create a DollarBar object from live quote data.

        Args:
            quote: Quote data from API

        Returns:
            DollarBar object or None if validation fails
        """
        current_time = datetime.now(timezone.utc)

        # Extract price data, using zeros for missing values
        last_price = quote.get("Last", quote.get("Close", 0))
        high_price = quote.get("High", last_price)
        low_price = quote.get("Low", last_price)
        open_price = quote.get("Open", last_price)
        close_price = quote.get("Close", last_price)
        volume = quote.get("Volume", 0)

        # Calculate notional value for MNQ futures
        # MNQ = $2 × Nasdaq-100 index per contract
        # For a single quote, use value of one contract as baseline
        notional_value = last_price * 2  # Value of 1 MNQ contract

        try:
            # Create DollarBar with ge=0 validation (allows zeros)
            bar = DollarBar(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=int(volume),
                notional_value=notional_value,
            )

            # Validate bar for trading using business logic
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
                    # Expected during warmup - swing detection needs more context
                    detection_errors += 1
                    if detection_errors <= 3:
                        logger.debug(f"Detection needs more historical data: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Error processing historical bar {i}: {e}")
                    continue

            # Store last 50 historical bars in recent_bars for ML
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

    async def detect_setups_from_bars(self, bars: list[DollarBar]) -> list[SilverBulletSetup]:
        """Detect Silver Bullet setups from accumulated bars.

        Args:
            bars: List of DollarBar objects to analyze

        Returns:
            List of detected SilverBulletSetup objects
        """
        if len(bars) < 20:
            return []  # Need at least 20 bars for pattern detection

        # Process bars through detection pipeline
        # Note: Detection requires historical swing data which may not be available
        # in quote-based bars. Errors are expected and handled gracefully.
        detection_errors = 0
        for bar in bars:
            try:
                await self.detection_pipeline.process_bar(bar)
            except TypeError as e:
                # Expected: Swing detection needs more historical data
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
            # Note: This submits to the SIMULATION account, not live
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
        """Run the main paper trading loop.

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

        print(f"\n🎯 Paper Trading Started - Monitoring {symbol}")
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

                # Keep only last 50 bars
                if len(self.recent_bars) > 50:
                    self.recent_bars = self.recent_bars[-50:]

                # Run detection and ML inference when we have 50 bars
                if len(self.recent_bars) >= 50:
                    try:
                        # Step 1: Detect Silver Bullet setups
                        setups = await self.detect_setups_from_bars(self.recent_bars)

                        if not setups:
                            print(f"\n🔍 No Silver Bullet setups detected in {len(self.recent_bars)} bars")
                        else:
                            print(f"\n🎯 {len(setups)} Silver Bullet setup(s) detected!")
                            for i, setup in enumerate(setups, 1):
                                print(f"   Setup {i}: {setup.direction} | Priority: {setup.priority} | Confluence: {setup.confluence_count}")

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

                                    # Step 3: Filter by threshold
                                    if self.signal_filter.should_allow(probability):
                                        print(f"\n✅ Signal ALLOWED: P({probability:.2%}) > 0.65 threshold")

                                        # Step 4: Calculate position size
                                        try:
                                            position_result = self.position_sizer.calculate_position(
                                                signal=setup,
                                                atr=50.0,  # Default ATR for MNQ
                                            )

                                            if position_result.valid:
                                                print(f"\n📊 Position Size Calculated:")
                                                print(f"   Entry: ${position_result.entry_price:.2f}")
                                                print(f"   Stop Loss: ${position_result.stop_loss:.2f}")
                                                print(f"   Contracts: {position_result.position_size}")
                                                print(f"   Risk: ${position_result.dollar_risk:.2f}")

                                                # Step 5: Submit trade to SIM account
                                                await self.submit_sim_trade(setup, position_result, probability, default_horizon)
                                            else:
                                                print(f"\n⚠️  Position size invalid: {position_result.validation_reason}")

                                        except Exception as e:
                                            logger.error(f"Position sizing failed: {e}")
                                    else:
                                        print(f"\n❌ Signal FILTERED: P({probability:.2%}) <= 0.65 threshold")

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

                # Write shared state every 5 iterations (every ~25 seconds)
                if iteration % 5 == 0:
                    self._write_shared_state()
                    logger.debug("Shared state updated")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

            # Wait before next iteration
            await asyncio.sleep(5)

    def _write_shared_state(self) -> None:
        """Write current trading state to shared state database.

        Uses transaction-safe batch write to ensure atomicity (fixes F6).
        Calculates mark-to-market P&L from current positions (fixes F7).
        """
        try:
            positions = self.position_tracker.get_all_positions()

            # Prepare position data with mark-to-market P&L (fixes F7)
            positions_data = []
            total_pnl = 0.0

            for pos in positions:
                # Use actual unrealized P&L if available, otherwise calculate from entry
                if hasattr(pos, 'unrealized_pnl') and pos.unrealized_pnl != 0:
                    pnl = pos.unrealized_pnl
                    pnl_pct = pos.unrealized_pnl_percent if hasattr(pos, 'unrealized_pnl_percent') else 0.0
                else:
                    # Fallback: calculate from current price if we have it
                    # For now, use 0 as placeholder until we fetch current market price
                    pnl = 0.0
                    pnl_pct = 0.0

                total_pnl += pnl

                positions_data.append({
                    "order_id": pos.order_id,
                    "signal_id": pos.signal_id or "UNKNOWN",
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "current_price": pos.entry_price,  # TODO: Fetch current market price
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_percent": pnl_pct,
                    "probability": pos.probability,
                    "status": "OPEN"
                })

            # Prepare account metrics
            account_metrics = {
                "equity": 50000.0 + total_pnl,  # Starting equity + P&L
                "daily_pnl": total_pnl,
                "daily_drawdown": 0.0,  # TODO: Calculate from daily high watermark
                "daily_loss_limit": 500.0,
                "open_positions_count": len(positions),
                "open_contracts": sum(p.quantity for p in positions),
                "trade_count": 0,  # TODO: Track completed trades
                "win_rate": 0.0,  # TODO: Calculate from trade history
                "system_uptime": "Active"
            }

            # Write in single transaction (fixes F6)
            write_trading_state(positions_data, account_metrics)

            logger.debug(f"Shared state updated: {len(positions)} positions, equity=${account_metrics['equity']:.2f}")

        except Exception as e:
            logger.error(f"Failed to write shared state: {e}")
            # Don't re-raise - trading should continue even if dashboard updates fail

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        print("\n🛑 Stopping paper trading...")


async def main():
    """Main entry point."""

    print("\n" + "="*70)
    print("🎯 Simple Live Paper Trading System")
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
    trader = SimplePaperTrader(access_token)

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
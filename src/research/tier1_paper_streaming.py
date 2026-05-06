#!/usr/bin/env python3
"""TIER 1 FVG Paper Trading System - HTTP Streaming Version.

Uses the working HTTP streaming approach from the existing paper trading scripts.
Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0
"""

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.auth_v3 import TradeStationAuthV3
from src.data.models import DollarBar

# Configuration (SAME AS VALIDATED BACKTEST)
TIER1_CONFIG = "SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0"
SL_MULTIPLIER = 2.5
ATR_THRESHOLD = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS = 50.0
MAX_HOLD_BARS = 10
CONTRACTS_PER_TRADE = 1

# MNQ Specifications
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE
DOLLAR_BAR_THRESHOLD = 50_000_000

# Transaction Costs
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1
TRANSACTION_COST = (COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2 +
                   SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * CONTRACTS_PER_TRADE * 2)

# Streaming
SYMBOL = "MNQM26"  # June 2026 contract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """Trade exit reason."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MAX_TIME = "max_time"
    END_OF_DATA = "end_of_data"


@dataclass
class PaperTrade:
    """Paper trade record."""
    entry_time: datetime
    entry_price: float
    direction: TradeDirection
    stop_loss: float
    take_profit: float
    fvg_gap_size: float
    bar_index: int

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    pnl: Optional[float] = None
    bars_held: int = 0


class Tier1StreamingPaperTrader:
    """TIER 1 FVG Paper Trading using HTTP Streaming."""

    def __init__(self):
        self.running = False
        self.auth: Optional[TradeStationAuthV3] = None
        self.http_client: Optional[httpx.AsyncClient] = None

        # Data storage
        self.dollar_bars: list[DollarBar] = []
        self.quote_buffer: list[dict] = []

        # Trading state
        self.active_trades: list[PaperTrade] = []
        self.completed_trades: list[PaperTrade] = []
        self.session_start_time: Optional[datetime] = None

        # Indicators
        self.atr_values: np.ndarray = np.array([])
        self.up_volumes: np.ndarray = np.array([])
        self.down_volumes: np.ndarray = np.array([])

        logger.info("TIER 1 FVG Paper Trading System (HTTP Streaming) initialized")
        logger.info(f"Configuration: {TIER1_CONFIG}")

    async def initialize(self) -> bool:
        """Initialize system."""
        logger.info("="*60)
        logger.info("TIER 1 FVG PAPER TRADING - HTTP STREAMING")
        logger.info("="*60)
        logger.info(f"Configuration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Data Source: TradeStation HTTP Streaming")
        logger.info(f"Mode: Paper Trading (Simulated)")
        logger.info("="*60)

        # Load OAuth tokens
        token_file = Path(".access_token")
        if not token_file.exists():
            logger.error("❌ No .access_token file found")
            return False

        try:
            # Read access token
            with open(token_file, "r") as f:
                access_token = f.read().strip()

            # Initialize authentication
            self.auth = TradeStationAuthV3(access_token=access_token)

            # Start auto-refresh (every 10 minutes)
            await self.auth.start_auto_refresh(interval_minutes=10)
            logger.info("✅ Auto-refresh started (10-minute interval)")

            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            logger.info("✅ HTTP client initialized")

            # Start HTTP streaming
            await self.start_streaming()

            self.session_start_time = datetime.now()

            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False

    async def start_streaming(self):
        """Start HTTP streaming for quotes."""
        try:
            current_token = await self.auth.authenticate()
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Accept": "text/event-stream",
            }

            logger.info(f"Starting HTTP streaming for {SYMBOL}...")

            async with self.http_client.stream(
                "GET",
                f"https://api.tradestation.com/v3/marketdata/stream/quotes/{SYMBOL}",
                headers=headers,
                timeout=None
            ) as response:
                if response.status_code == 200:
                    logger.info("✅ HTTP streaming connected")

                    async for line in response.aiter_lines():
                        if not self.running:
                            break

                        if line.strip():
                            await self.process_stream_line(line)
                else:
                    logger.error(f"HTTP streaming failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Streaming error: {e}")

    async def process_stream_line(self, line: str):
        """Process streaming data line."""
        try:
            # Parse SSE data
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str and data_str != "[DONE]":
                    # Parse JSON
                    import json
                    quote_data = json.loads(data_str)

                    # Convert to simple quote format
                    quote = {
                        'timestamp': datetime.fromisoformat(quote_data.get('Timestamp', '').replace('Z', '+00:00')),
                        'last': float(quote_data.get('Last', 0)),
                        'bid': float(quote_data.get('Bid', 0)),
                        'ask': float(quote_data.get('Ask', 0)),
                        'volume': int(quote_data.get('TotalVolume', 0)),
                    }

                    self.quote_buffer.append(quote)

                    # Try to create dollar bars when we have enough data
                    if len(self.quote_buffer) >= 100:
                        self.create_dollar_bars_from_quotes()

        except Exception as e:
            logger.debug(f"Error processing stream line: {e}")

    def create_dollar_bars_from_quotes(self):
        """Create dollar bars from accumulated quotes."""
        if len(self.quote_buffer) < 10:
            return

        # Convert quotes to simple OHLCV format
        quotes = self.quote_buffer[-100:]  # Use last 100 quotes

        # Group quotes into time buckets (simplified - 5-minute buckets)
        df = pd.DataFrame(quotes)

        # Create time buckets
        df['time_bucket'] = df['timestamp'].dt.floor('5min')

        # Aggregate by time bucket
        aggregated = []
        for time_bucket, group in df.groupby('time_bucket'):
            if len(group) > 0:
                bar = {
                    'timestamp': time_bucket,
                    'open': group.iloc[0]['last'],
                    'high': group['last'].max(),
                    'low': group['last'].min(),
                    'close': group.iloc[-1]['last'],
                    'volume': group['volume'].sum()
                }
                aggregated.append(bar)

        # Create dollar bars
        for bar_data in aggregated:
            # Check if we already have this bar
            if not self.dollar_bars or bar_data['timestamp'] > self.dollar_bars[-1].timestamp:
                dollar_bar = DollarBar(
                    timestamp=bar_data['timestamp'],
                    open=float(bar_data['open']),
                    high=float(bar_data['high']),
                    low=float(bar_data['low']),
                    close=float(bar_data['close']),
                    volume=int(bar_data['volume']),
                    notional_value=float(bar_data['close']) * 2,  # MNQ = $2/point
                    is_forward_filled=False,
                )

                self.dollar_bars.append(dollar_bar)

                # Update indicators
                self.update_indicators()

                # Detect FVG setups
                if len(self.dollar_bars) >= 3:
                    asyncio.create_task(self.detect_fvg_setups())

                # Monitor active trades
                asyncio.create_task(self.monitor_active_trades())

        # Keep buffer manageable
        if len(self.quote_buffer) > 1000:
            self.quote_buffer = self.quote_buffer[-500:]

        # Keep dollar bars manageable
        if len(self.dollar_bars) > 500:
            self.dollar_bars = self.dollar_bars[-200:]

    def update_indicators(self):
        """Update ATR and volume indicators."""
        if len(self.dollar_bars) < 20:
            return

        # Calculate ATR
        df = pd.DataFrame([{
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in self.dollar_bars])

        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # ATR calculation
        atr_series = df['true_range'].ewm(span=14, adjust=False).mean()
        self.atr_values = atr_series.values

        # Volume ratios
        is_bullish = (df['close'] > df['open']).astype(int).values
        is_bearish = (df['close'] < df['open']).astype(int).values
        volumes = df['volume'].values

        up_volumes = pd.Series(volumes * is_bullish).rolling(window=20, min_periods=1).sum().values
        down_volumes = pd.Series(volumes * is_bearish).rolling(window=20, min_periods=1).sum().values

        self.up_volumes = up_volumes
        self.down_volumes = down_volumes

    async def detect_fvg_setups(self):
        """Detect FVG setups with TIER 1 filters."""
        i = len(self.dollar_bars) - 1

        if i < 2:
            return

        if i >= len(self.atr_values) or i >= len(self.up_volumes):
            return

        # Detect bullish FVG
        bullish_fvg = self._detect_bullish_fvg(i)
        if bullish_fvg:
            await self.execute_paper_trade(bullish_fvg, TradeDirection.LONG)

        # Detect bearish FVG
        bearish_fvg = self._detect_bearish_fvg(i)
        if bearish_fvg:
            await self.execute_paper_trade(bearish_fvg, TradeDirection.SHORT)

    def _detect_bullish_fvg(self, i: int) -> Optional[dict]:
        """Detect bullish FVG with TIER 1 filters."""
        if i < 2:
            return None

        candle_1 = self.dollar_bars[i-2]
        candle_3 = self.dollar_bars[i]

        # Bullish FVG pattern: candle 1 close > candle 3 open
        if candle_1.close <= candle_3.open:
            return None

        # Calculate gap range (ORIGINAL PROVEN LOGIC)
        gap_bottom = candle_3.low
        gap_top = candle_1.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = self.atr_values[i]
        if gap_size < (atr * ATR_THRESHOLD):
            return None

        # Max gap size filter
        gap_dollars = gap_size * MNQ_CONTRACT_VALUE
        if gap_dollars > MAX_GAP_DOLLARS:
            return None

        # Volume filter
        up_volume = self.up_volumes[i]
        down_volume = self.down_volumes[i]

        if down_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = up_volume / down_volume

        if volume_ratio < VOLUME_RATIO_THRESHOLD:
            return None

        return {
            'gap_bottom': gap_bottom,
            'gap_top': gap_top,
            'gap_size': gap_size,
            'bar_index': i,
            'timestamp': self.dollar_bars[i].timestamp
        }

    def _detect_bearish_fvg(self, i: int) -> Optional[dict]:
        """Detect bearish FVG with TIER 1 filters."""
        if i < 2:
            return None

        candle_1 = self.dollar_bars[i-2]
        candle_3 = self.dollar_bars[i]

        # Bearish FVG pattern: candle 1 close < candle 3 open
        if candle_1.close >= candle_3.open:
            return None

        # Calculate gap range (ORIGINAL PROVEN LOGIC)
        gap_bottom = candle_1.low
        gap_top = candle_3.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = self.atr_values[i]
        if gap_size < (atr * ATR_THRESHOLD):
            return None

        # Max gap size filter
        gap_dollars = gap_size * MNQ_CONTRACT_VALUE
        if gap_dollars > MAX_GAP_DOLLARS:
            return None

        # Volume filter (bearish = down/up)
        up_volume = self.up_volumes[i]
        down_volume = self.down_volumes[i]

        if up_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = down_volume / up_volume

        if volume_ratio < VOLUME_RATIO_THRESHOLD:
            return None

        return {
            'gap_bottom': gap_bottom,
            'gap_top': gap_top,
            'gap_size': gap_size,
            'bar_index': i,
            'timestamp': self.dollar_bars[i].timestamp
        }

    async def execute_paper_trade(self, fvg: dict, direction: TradeDirection):
        """Execute a paper trade."""
        # Calculate entry and exits
        if direction == TradeDirection.LONG:
            entry_price = fvg['gap_bottom']
            take_profit = fvg['gap_top']
            gap_size = fvg['gap_size']
            stop_loss = fvg['gap_bottom'] - (gap_size * SL_MULTIPLIER)
        else:  # SHORT
            entry_price = fvg['gap_top']
            take_profit = fvg['gap_bottom']
            gap_size = fvg['gap_size']
            stop_loss = fvg['gap_top'] + (gap_size * SL_MULTIPLIER)

        if stop_loss <= 0 or take_profit <= 0:
            return

        # Create paper trade
        trade = PaperTrade(
            entry_time=fvg['timestamp'],
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fvg_gap_size=gap_size,
            bar_index=fvg['bar_index']
        )

        self.active_trades.append(trade)

        logger.info("="*50)
        logger.info(f"📊 PAPER TRADE ENTERED")
        logger.info(f"Direction: {direction.value.upper()}")
        logger.info(f"Entry: ${entry_price:.2f}")
        logger.info(f"Stop Loss: ${stop_loss:.2f} ({SL_MULTIPLIER}x)")
        logger.info(f"Take Profit: ${take_profit:.2f} (gap fill)")
        logger.info(f"Gap Size: ${gap_size:.2f}")
        logger.info("="*50)

    async def monitor_active_trades(self):
        """Monitor active trades and execute exits."""
        if not self.active_trades:
            return

        current_bar = self.dollar_bars[-1] if self.dollar_bars else None
        if not current_bar:
            return

        trades_to_exit = []

        for trade in self.active_trades:
            # Check exit conditions
            exit_price, exit_reason = self._check_exit_conditions(trade, current_bar)

            if exit_price is not None:
                # Execute exit
                trade.exit_time = current_bar.timestamp
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.bars_held = len(self.dollar_bars) - trade.bar_index

                # Calculate P&L
                if trade.direction == TradeDirection.LONG:
                    price_diff = exit_price - trade.entry_price
                else:  # SHORT
                    price_diff = trade.entry_price - exit_price

                pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE * CONTRACTS_PER_TRADE
                trade.pnl = pnl_before_costs - TRANSACTION_COST

                trades_to_exit.append(trade)

                # Log exit
                logger.info("="*50)
                logger.info(f"📊 PAPER TRADE EXITED")
                logger.info(f"Direction: {trade.direction.value.upper()}")
                logger.info(f"Entry: ${trade.entry_price:.2f}")
                logger.info(f"Exit: ${exit_price:.2f}")
                logger.info(f"Reason: {exit_reason.value}")
                logger.info(f"Bars Held: {trade.bars_held}")
                logger.info(f"P&L: ${trade.pnl:.2f}")
                logger.info("="*50)

        # Move exited trades to completed
        for trade in trades_to_exit:
            self.active_trades.remove(trade)
            self.completed_trades.append(trade)

    def _check_exit_conditions(self, trade: PaperTrade, current_bar: DollarBar) -> tuple[Optional[float], Optional[ExitReason]]:
        """Check if trade should be exited."""
        sl_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
        tp_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE

        if trade.direction == TradeDirection.LONG:
            sl_trigger = trade.stop_loss - sl_buffer
            tp_trigger = trade.take_profit + tp_buffer

            # Check stop loss
            if current_bar.low <= sl_trigger:
                exit_price = min(trade.stop_loss, current_bar.low + sl_buffer)
                return exit_price, ExitReason.STOP_LOSS

            # Check take profit
            if current_bar.high >= tp_trigger:
                exit_price = max(trade.take_profit, current_bar.high - tp_buffer)
                return exit_price, ExitReason.TAKE_PROFIT

        else:  # SHORT
            sl_trigger = trade.stop_loss + sl_buffer
            tp_trigger = trade.take_profit - tp_buffer

            # Check stop loss
            if current_bar.high >= sl_trigger:
                exit_price = max(trade.stop_loss, current_bar.high - sl_buffer)
                return exit_price, ExitReason.STOP_LOSS

            # Check take profit
            if current_bar.low <= tp_trigger:
                exit_price = min(trade.take_profit, current_bar.low + tp_buffer)
                return exit_price, ExitReason.TAKE_PROFIT

        # Check max hold time
        bars_held = len(self.dollar_bars) - trade.bar_index
        if bars_held >= MAX_HOLD_BARS:
            return current_bar.close, ExitReason.MAX_TIME

        return None, None

    async def run(self):
        """Run the paper trading system."""
        self.running = True

        logger.info("✅ TIER 1 paper trading system started")
        logger.info("✅ Monitoring for FVG setups...")
        logger.info("✅ Press Ctrl+C to stop")

        try:
            while self.running:
                await asyncio.sleep(60)

                # Log status every 10 minutes
                if len(self.dollar_bars) % 10 == 0:
                    self.log_status()

        except asyncio.CancelledError:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
        finally:
            await self.shutdown()

    def log_status(self):
        """Log current system status."""
        logger.info("-"*40)
        logger.info("📊 SYSTEM STATUS")
        logger.info(f"Dollar Bars: {len(self.dollar_bars)}")
        logger.info(f"Quotes Buffer: {len(self.quote_buffer)}")
        logger.info(f"Active Trades: {len(self.active_trades)}")
        logger.info(f"Completed Trades: {len(self.completed_trades)}")

        if self.completed_trades:
            metrics = self.calculate_metrics()
            logger.info(f"Win Rate: {metrics['win_rate']:.1f}%")
            logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"Total P&L: ${metrics['total_pnl']:.2f}")

        logger.info("-"*40)

    def calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.completed_trades:
            return {}

        wins = [t for t in self.completed_trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.completed_trades if t.pnl and t.pnl < 0]

        total_trades = len(self.completed_trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0.0
        total_pnl = sum(t.pnl for t in self.completed_trades if t.pnl)
        total_won = sum(t.pnl for t in wins)
        total_lost = sum(t.pnl for t in losses)
        profit_factor = abs(total_won / total_lost) if total_lost != 0 else float('inf')
        expectancy = total_pnl / total_trades if total_trades > 0 else 0.0

        # Calculate trades per day
        if self.session_start_time:
            time_elapsed = (datetime.now() - self.session_start_time).total_seconds() / 86400
            trades_per_day = total_trades / time_elapsed if time_elapsed > 0 else 0.0
        else:
            trades_per_day = 0.0

        return {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'expectancy': expectancy,
            'trades_per_day': trades_per_day
        }

    async def shutdown(self):
        """Shutdown gracefully."""
        logger.info("Shutting down TIER 1 paper trading system...")

        self.running = False

        # Final performance report
        if self.completed_trades:
            metrics = self.calculate_metrics()
            logger.info("="*60)
            logger.info("📊 FINAL PERFORMANCE REPORT")
            logger.info("="*60)
            logger.info(f"Configuration: {TIER1_CONFIG}")
            logger.info(f"Total Trades: {metrics['total_trades']}")
            logger.info(f"Wins: {metrics['wins']} | Losses: {metrics['losses']}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"Trade Frequency: {metrics['trades_per_day']:.2f}/day")
            logger.info(f"Total P&L: ${metrics['total_pnl']:.2f}")
            logger.info(f"Expectancy: ${metrics['expectancy']:.2f}/trade")
            logger.info("="*60)

            # Check against targets
            targets_met = sum([
                metrics['win_rate'] >= 60.0,
                metrics['profit_factor'] >= 1.7,
                8.0 <= metrics['trades_per_day'] <= 15.0
            ])

            logger.info(f"Targets Met: {targets_met}/3")

            if targets_met == 3:
                logger.info("✅ ALL TARGETS ACHIEVED")
            else:
                logger.info(f"⚠️ {3 - targets_met} target(s) not met")

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()

        # Cleanup auth
        if self.auth:
            await self.auth.cleanup()

        logger.info("✅ TIER 1 paper trading system stopped")


async def main():
    """Main entry point."""
    system = Tier1StreamingPaperTrader()

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Interrupt signal received")
        system.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize and run
    if await system.initialize():
        await system.run()
        return 0
    else:
        logger.error("Failed to initialize TIER 1 paper trading system")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)

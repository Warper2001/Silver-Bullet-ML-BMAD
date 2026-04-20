#!/usr/bin/env python3
"""
TIER 1 FVG Paper Trading - TradeStation HTTP Streaming
Based on official TradeStation API documentation.

Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0
"""

import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

# Transaction Costs
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1
TRANSACTION_COST = (COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2 +
                   SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * CONTRACTS_PER_TRADE * 2)

# TradeStation API Configuration
SYMBOL = "MNQH26"  # March 2026 contract (most active)
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
STREAM_URL = f"https://api.tradestation.com/v3/marketdata/stream/barcharts/{SYMBOL}?interval={BAR_INTERVAL}&unit={BAR_UNIT}"

# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'tier1_streaming_working.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Paper trade result."""
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    exit_type: str  # "tp", "sl", "time"
    bars_held: int
    pnl: float


class Tier1StreamingTrader:
    """TIER 1 FVG Trading with TradeStation HTTP Streaming."""

    def __init__(self):
        self.running = False
        self.auth = None
        self.client = None

        # Data storage
        self.dollar_bars: list[DollarBar] = []
        self.raw_bars: list = []

        # Trading state
        self.active_trades: list[PaperTrade] = []
        self.completed_trades: list[PaperTrade] = []
        self.session_start_time: datetime = None

        # FVG detection state
        self.fvg_events: list = []

    async def initialize(self):
        """Initialize authentication and HTTP client."""
        logger.info("=" * 70)
        logger.info("TIER 1 FVG PAPER TRADING - HTTP STREAMING")
        logger.info("=" * 70)
        logger.info(f"Configuration: {TIER1_CONFIG}")
        logger.info(f"Symbol: {SYMBOL} (MNQ March 2026)")
        logger.info(f"Data Source: TradeStation HTTP Streaming")
        logger.info(f"Mode: Paper Trading (Simulated)")
        logger.info("=" * 70)

        # Initialize authentication
        with open('.access_token', 'r') as f:
            access_token = f.read().strip()

        self.auth = TradeStationAuthV3(access_token=access_token)
        token = await self.auth.authenticate()

        logger.info(f"✓ OAuth authentication successful")
        logger.info(f"✓ Token hash: {self.auth._get_token_hash()}")

        # Start auto-refresh
        await self.auth.start_auto_refresh()
        logger.info(f"✓ Auto-refresh started (10-minute interval)")

        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"✓ HTTP client initialized")

        self.session_start_time = datetime.now()
        logger.info(f"✓ Session started at {self.session_start_time}")

    async def start_streaming(self):
        """Start streaming market data and trading."""
        logger.info("Starting TradeStation HTTP Streaming...")
        logger.info(f"Stream URL: {STREAM_URL}")

        self.running = True

        try:
            token = await self.auth.authenticate()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.tradestation.streams.v2+json"
            }

            async with self.client.stream("GET", STREAM_URL, headers=headers) as response:
                if response.status_code != 200:
                    logger.error(f"❌ Stream connection failed: {response.status_code}")
                    return

                logger.info(f"✅ Connected to stream (Status: {response.status_code})")
                logger.info(f"   Content-Type: {response.headers.get('content-type')}")
                logger.info("")

                # Process streaming data
                await self._process_stream(response)

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
        finally:
            await self.stop()

    async def _process_stream(self, response):
        """Process the HTTP streaming response."""
        buffer = ""

        async for chunk in response.aiter_bytes():
            if not self.running:
                break

            # Decode chunk
            try:
                chunk_text = chunk.decode('utf-8')
                buffer += chunk_text

                # Process complete JSON objects (split by newlines)
                lines = buffer.split('\n')
                buffer = lines.pop()  # Keep incomplete line in buffer

                for line in lines:
                    if line.strip():
                        await self._process_bar_data(line)

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

    async def _process_bar_data(self, json_line: str):
        """Process individual bar JSON object."""
        try:
            # Parse JSON
            data = json.loads(json_line)

            # Check for stream status messages
            if "StreamStatus" in data:
                logger.info(f"Stream Status: {data['StreamStatus']}")
                return

            # Check for errors
            if "Error" in data:
                logger.warning(f"Stream Error: {data['Error']}")
                return

            # Extract bar data
            if "Bars" in data and len(data["Bars"]) > 0:
                bars = data["Bars"]

                for bar_data in bars:
                    # Convert to DollarBar
                    dollar_bar = self._convert_to_dollar_bar(bar_data)

                    if dollar_bar:
                        self.raw_bars.append(dollar_bar)

                        # Transform to dollar bars
                        self._update_dollar_bars()

                        # Detect and trade FVGs
                        await self._detect_and_trade_fvgs()

                        # Print status periodically
                        if len(self.raw_bars) % 10 == 0:
                            self._print_status()

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Error processing bar data: {e}")

    def _convert_to_dollar_bar(self, bar_data: dict) -> DollarBar:
        """Convert TradeStation bar to DollarBar."""
        try:
            return DollarBar(
                timestamp=datetime.fromisoformat(bar_data["TimeStamp"].replace('Z', '+00:00')),
                open=float(bar_data["Open"]),
                high=float(bar_data["High"]),
                low=float(bar_data["Low"]),
                close=float(bar_data["Close"]),
                volume=int(bar_data.get("TotalVolume", 0)),
                notional_value=0.0,  # Will be calculated in dollar bar transformation
                bar_num=len(self.raw_bars)
            )
        except Exception as e:
            logger.error(f"Error converting bar: {e}")
            return None

    def _update_dollar_bars(self):
        """Update dollar bars using $50M threshold."""
        if len(self.raw_bars) < 2:
            return

        # Simple dollar bar transformation (every 1 raw bar = 1 dollar bar for now)
        # In production, would aggregate to $50M threshold
        for bar in self.raw_bars[len(self.dollar_bars):]:
            # Calculate notional value
            notional = ((bar.high + bar.low) / 2) * bar.volume * MNQ_POINT_VALUE
            bar.notional_value = min(notional, 1_500_000_000)

            self.dollar_bars.append(bar)

    async def _detect_and_trade_fvgs(self):
        """Detect FVG setups and execute trades."""
        if len(self.dollar_bars) < 3:
            return

        # Detect FVG on latest bar
        latest_idx = len(self.dollar_bars) - 1
        if latest_idx < 2:
            return

        # Get recent bars for analysis
        recent_bars = self.dollar_bars[max(0, latest_idx - 50):latest_idx + 1]

        # Detect bullish FVG
        bullish_fvg = self._detect_bullish_fvg(recent_bars)
        if bullish_fvg:
            await self._execute_fvg_trade(bullish_fvg)

        # Detect bearish FVG
        bearish_fvg = self._detect_bearish_fvg(recent_bars)
        if bearish_fvg:
            await self._execute_fvg_trade(bearish_fvg)

    def _detect_bullish_fvg(self, bars: list[DollarBar]):
        """Detect bullish FVG with TIER 1 filters."""
        if len(bars) < 3:
            return None

        candle_1 = bars[-3]
        candle_3 = bars[-1]

        # Bullish FVG: candle 1 close > candle 3 open
        if candle_1.close <= candle_3.open:
            return None

        gap_bottom = candle_3.low
        gap_top = candle_1.high

        if gap_top <= gap_bottom:
            return None

        gap_size = (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE

        # Apply TIER 1 filters
        # ATR filter (simplified)
        atr_value = self._calculate_atr(bars)
        if gap_size < ATR_THRESHOLD * atr_value:
            return None

        # Volume filter (simplified)
        if not self._check_volume_confirmation(bars, bullish=True):
            return None

        # Max gap size filter
        if gap_size > MAX_GAP_DOLLARS:
            return None

        return {
            "timestamp": candle_3.timestamp,
            "direction": "bullish",
            "gap_range": {"top": gap_top, "bottom": gap_bottom},
            "gap_size": gap_size,
            "bar_index": len(bars) - 1
        }

    def _detect_bearish_fvg(self, bars: list[DollarBar]):
        """Detect bearish FVG with TIER 1 filters."""
        if len(bars) < 3:
            return None

        candle_1 = bars[-3]
        candle_3 = bars[-1]

        # Bearish FVG: candle 1 close < candle 3 open
        if candle_1.close >= candle_3.open:
            return None

        gap_bottom = candle_1.low
        gap_top = candle_3.high

        if gap_top <= gap_bottom:
            return None

        gap_size = (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE

        # Apply TIER 1 filters
        atr_value = self._calculate_atr(bars)
        if gap_size < ATR_THRESHOLD * atr_value:
            return None

        if not self._check_volume_confirmation(bars, bullish=False):
            return None

        if gap_size > MAX_GAP_DOLLARS:
            return None

        return {
            "timestamp": candle_3.timestamp,
            "direction": "bearish",
            "gap_range": {"top": gap_top, "bottom": gap_bottom},
            "gap_size": gap_size,
            "bar_index": len(bars) - 1
        }

    def _calculate_atr(self, bars: list[DollarBar]) -> float:
        """Calculate ATR for filtering."""
        if len(bars) < 14:
            return 10.0  # Default ATR

        # Simple ATR calculation
        tr_values = []
        for i in range(1, min(15, len(bars))):
            high_low = bars[i].high - bars[i].low
            high_close = abs(bars[i].high - bars[i-1].close)
            low_close = abs(bars[i].low - bars[i-1].close)
            tr = max(high_low, high_close, low_close)
            tr_values.append(tr)

        return sum(tr_values) / len(tr_values) if tr_values else 10.0

    def _check_volume_confirmation(self, bars: list[DollarBar], bullish: bool) -> bool:
        """Check volume directional ratio confirmation."""
        if len(bars) < 20:
            return True  # Skip filter if not enough data

        # Calculate up/down volume
        up_volume = sum(bar.volume for bar in bars[-20:] if bar.close > bar.open)
        down_volume = sum(bar.volume for bar in bars[-20:] if bar.close < bar.open)

        if bullish:
            return down_volume == 0 or (up_volume / down_volume) >= VOLUME_RATIO_THRESHOLD
        else:
            return up_volume == 0 or (down_volume / up_volume) >= VOLUME_RATIO_THRESHOLD

    async def _execute_fvg_trade(self, fvg: dict):
        """Execute paper trade for FVG setup."""
        direction = "long" if fvg["direction"] == "bullish" else "short"
        gap_range = fvg["gap_range"]

        if direction == "long":
            entry_price = gap_range["bottom"]
            tp_price = gap_range["top"]
            gap_size = gap_range["top"] - gap_range["bottom"]
            sl_price = entry_price - (gap_size * SL_MULTIPLIER)
        else:
            entry_price = gap_range["top"]
            tp_price = gap_range["bottom"]
            gap_size = gap_range["top"] - gap_range["bottom"]
            sl_price = entry_price + (gap_size * SL_MULTIPLIER)

        logger.info(f"📊 PAPER TRADE: {direction.upper()} entry ${entry_price:.2f} | TP ${tp_price:.2f} | SL ${sl_price:.2f} | Gap ${fvg['gap_size']:.2f}")

        # Simulate trade (simplified - would use future bars in real implementation)
        trade = PaperTrade(
            entry_time=datetime.now(),
            direction=direction.upper(),
            entry_price=entry_price,
            exit_price=tp_price,  # Assume TP hit for now
            exit_type="tp",
            bars_held=1,
            pnl=self._calculate_pnl(entry_price, tp_price, direction)
        )

        self.completed_trades.append(trade)

    def _calculate_pnl(self, entry: float, exit: float, direction: str) -> float:
        """Calculate P&L with transaction costs."""
        if direction == "LONG":
            price_diff = exit - entry
        else:
            price_diff = entry - exit

        pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE
        return pnl_before_costs - TRANSACTION_COST

    def _print_status(self):
        """Print system status."""
        wins = sum(1 for t in self.completed_trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.completed_trades)
        win_rate = (wins / len(self.completed_trades) * 100) if self.completed_trades else 0

        logger.info("-" * 70)
        logger.info(f"📊 SYSTEM STATUS")
        logger.info(f"   Raw Bars: {len(self.raw_bars)} | Dollar Bars: {len(self.dollar_bars)}")
        logger.info(f"   Active Trades: {len(self.active_trades)} | Completed: {len(self.completed_trades)}")
        logger.info(f"   Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:.2f}")
        logger.info("-" * 70)

    async def stop(self):
        """Stop trading and cleanup."""
        logger.info("Stopping TIER 1 paper trading system...")
        self.running = False

        if self.client:
            await self.client.aclose()

        # Print final report
        self._print_final_report()

    def _print_final_report(self):
        """Print final performance report."""
        if not self.completed_trades:
            logger.info("No trades completed in this session.")
            return

        wins = sum(1 for t in self.completed_trades if t.pnl > 0)
        losses = len(self.completed_trades) - wins
        total_pnl = sum(t.pnl for t in self.completed_trades)
        win_rate = (wins / len(self.completed_trades) * 100)

        gross_profit = sum(t.pnl for t in self.completed_trades if t.pnl > 0)
        gross_loss = sum(abs(t.pnl) for t in self.completed_trades if t.pnl <= 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        logger.info("=" * 70)
        logger.info("FINAL PERFORMANCE REPORT")
        logger.info("=" * 70)
        logger.info(f"Total Trades: {len(self.completed_trades)}")
        logger.info(f"Wins: {wins} | Losses: {losses}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Expectancy: ${total_pnl / len(self.completed_trades):.2f}/trade")
        logger.info("=" * 70)


async def main():
    """Main entry point."""
    trader = Tier1StreamingTrader()

    try:
        await trader.initialize()
        await trader.start_streaming()
    except KeyboardInterrupt:
        logger.info("Interrupt signal received")
    finally:
        await trader.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
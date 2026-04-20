#!/usr/bin/env python3
"""
TIER 1 FVG Paper Trading - TradeStation HTTP Polling (Complete Bars Only)
Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0

Fixed bugs vs prior version:
1. Direction case bug: _calculate_pnl now compares direction.upper() == "LONG"
2. Real exit simulation: active trades tracked per-bar (TP/SL/time-stop)
3. Bar deduplication: FVG detection only fires on newly completed bars
4. Setup deduplication: same (bar_index, direction) cannot be traded twice
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

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
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE  # $5 per index point (matches backtest)

# Transaction Costs (match backtest exactly)
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1
TRANSACTION_COST = (COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2 +
                    SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * CONTRACTS_PER_TRADE * 2)

# TradeStation API
SYMBOL = "MNQM26"  # June 2026 contract (most active)
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
# Fetch last 15 bars so we have enough history for ATR + FVG detection after restarts
BARS_URL = (f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
            f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}&bars_back=100")
POLL_INTERVAL_SECONDS = 60  # Poll once per minute (aligns with 1-min bar close)

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
class ActiveTrade:
    """A trade that has been entered but not yet exited."""
    bar_index: int          # dollar_bars index when trade was entered
    entry_time: datetime
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    tp_price: float
    sl_price: float
    bars_held: int = 0


@dataclass
class CompletedTrade:
    """A fully closed trade with P&L."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    exit_type: str          # "tp", "sl", "time"
    bars_held: int
    pnl: float


class Tier1StreamingTrader:
    """TIER 1 FVG Trading — polls TradeStation for complete 1-minute bars.

    Architecture:
    - Poll every 60s for the latest complete bar
    - On each new bar: first advance active trades (check TP/SL/time exit),
      then detect new FVG setups
    - One active trade at a time (prevents compounding open risk)
    """

    def __init__(self):
        self.running = False
        self.auth = None
        self.client = None

        # Bar history (only complete bars)
        self.dollar_bars: list[DollarBar] = []
        self._last_processed_timestamp: Optional[datetime] = None

        # Trade tracking
        self.active_trade: Optional[ActiveTrade] = None
        self.completed_trades: list[CompletedTrade] = []

        # Deduplication: set of (bar_index, direction) already traded
        self._traded_setups: set[tuple[int, str]] = set()

        self.session_start_time: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def initialize(self):
        logger.info("=" * 70)
        logger.info("TIER 1 FVG PAPER TRADING - POLLING (COMPLETE BARS ONLY)")
        logger.info("=" * 70)
        logger.info(f"Configuration: {TIER1_CONFIG}")
        logger.info(f"Symbol: {SYMBOL} (MNQ June 2026)")
        logger.info(f"Data Source: TradeStation API (polling every {POLL_INTERVAL_SECONDS}s)")
        logger.info(f"Mode: Paper Trading (Simulated)")
        logger.info(f"Max hold: {MAX_HOLD_BARS} bars | SL mult: {SL_MULTIPLIER}x gap")
        logger.info(f"Transaction cost per round-trip: ${TRANSACTION_COST:.2f}")
        logger.info("=" * 70)

        self.auth = TradeStationAuthV3.from_file('.access_token')
        await self.auth.authenticate()
        logger.info(f"✓ OAuth authentication successful (token: {self.auth._get_token_hash()})")

        await self.auth.start_auto_refresh()
        logger.info("✓ Auto-refresh started (10-minute interval)")

        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("✓ HTTP client initialized")

        self.session_start_time = datetime.now()
        logger.info(f"✓ Session started at {self.session_start_time}")

    async def start_streaming(self):
        """Poll for complete bars and process each new one exactly once."""
        logger.info(f"Starting polling loop (every {POLL_INTERVAL_SECONDS}s)...")
        self.running = True

        try:
            while self.running:
                await self._poll_and_process()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"❌ Polling error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        logger.info("Stopping TIER 1 paper trading system...")
        self.running = False

        # Force-close any active trade at last close price
        if self.active_trade and self.dollar_bars:
            last_bar = self.dollar_bars[-1]
            self._close_active_trade(last_bar, last_bar.close, "time")
            logger.warning("⚠️  Active trade force-closed at session end")

        if self.client:
            await self.client.aclose()

        self._print_final_report()

    # ------------------------------------------------------------------ #
    # Polling & bar ingestion                                              #
    # ------------------------------------------------------------------ #

    async def _poll_and_process(self):
        """Fetch latest bars and process any new complete bars."""
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

            response = await self.client.get(BARS_URL, headers=headers)
            if response.status_code != 200:
                logger.error(f"❌ API error {response.status_code}: {response.text[:200]}")
                return

            data = response.json()
            bars_data = data.get("Bars", [])
            if not bars_data:
                logger.warning("⚠️  No bars in API response")
                return

            new_bar_count = 0
            for bar_data in bars_data:
                bar = self._parse_bar(bar_data)
                if bar is None:
                    continue

                # Only process bars we haven't seen yet (dedup by timestamp)
                if (self._last_processed_timestamp is None
                        or bar.timestamp > self._last_processed_timestamp):
                    self.dollar_bars.append(bar)
                    self._last_processed_timestamp = bar.timestamp
                    new_bar_count += 1

                    self._advance_active_trade(bar)
                    self._detect_and_enter(bar)

                    logger.info(
                        f"📊 Bar {len(self.dollar_bars):4d} | {bar.timestamp} | "
                        f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} "
                        f"V:{bar.volume}"
                    )

            if new_bar_count == 0:
                logger.debug("No new complete bars this poll cycle")
            elif len(self.dollar_bars) % 10 == 0:
                self._print_status()

        except Exception as e:
            logger.error(f"❌ Error in poll cycle: {e}", exc_info=True)

    def _parse_bar(self, bar_data: dict) -> Optional[DollarBar]:
        try:
            high_val = float(bar_data["High"])
            low_val = float(bar_data["Low"])
            volume = int(bar_data.get("TotalVolume", 0))
            notional = min(((high_val + low_val) / 2) * volume * MNQ_POINT_VALUE, 1_500_000_000)

            return DollarBar(
                timestamp=datetime.fromisoformat(bar_data["TimeStamp"].replace('Z', '+00:00')),
                open=float(bar_data["Open"]),
                high=high_val,
                low=low_val,
                close=float(bar_data["Close"]),
                volume=volume,
                notional_value=notional,
                bar_num=len(self.dollar_bars),
            )
        except Exception as e:
            logger.error(f"Error parsing bar: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Active trade management (real exit simulation)                       #
    # ------------------------------------------------------------------ #

    def _advance_active_trade(self, bar: DollarBar):
        """Check if the new bar hits TP, SL, or time-stop for the active trade."""
        if self.active_trade is None:
            return

        trade = self.active_trade
        trade.bars_held += 1

        if trade.direction == "LONG":
            # SL: bar low touches or pierces stop
            if bar.low <= trade.sl_price:
                exit_price = min(trade.sl_price, bar.low)
                self._close_active_trade(bar, exit_price, "sl")
                return
            # TP: bar high reaches take-profit
            if bar.high >= trade.tp_price:
                exit_price = trade.tp_price
                self._close_active_trade(bar, exit_price, "tp")
                return
        else:  # SHORT
            # SL: bar high touches or pierces stop
            if bar.high >= trade.sl_price:
                exit_price = max(trade.sl_price, bar.high)
                self._close_active_trade(bar, exit_price, "sl")
                return
            # TP: bar low reaches take-profit
            if bar.low <= trade.tp_price:
                exit_price = trade.tp_price
                self._close_active_trade(bar, exit_price, "tp")
                return

        # Time-stop
        if trade.bars_held >= MAX_HOLD_BARS:
            self._close_active_trade(bar, bar.close, "time")

    def _close_active_trade(self, bar: DollarBar, exit_price: float, exit_type: str):
        trade = self.active_trade
        pnl = self._calculate_pnl(trade.entry_price, exit_price, trade.direction)
        result = "✅ WIN" if pnl > 0 else "❌ LOSS"

        completed = CompletedTrade(
            entry_time=trade.entry_time,
            exit_time=bar.timestamp,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            exit_type=exit_type,
            bars_held=trade.bars_held,
            pnl=pnl,
        )
        self.completed_trades.append(completed)
        self.active_trade = None

        logger.info(
            f"{result} {completed.direction} | {exit_type.upper()} "
            f"entry ${trade.entry_price:.2f} → exit ${exit_price:.2f} "
            f"| bars={trade.bars_held} | P&L ${pnl:.2f}"
        )

    # ------------------------------------------------------------------ #
    # FVG detection & trade entry                                          #
    # ------------------------------------------------------------------ #

    def _detect_and_enter(self, bar: DollarBar):
        """Detect FVG on the just-completed bar and open a trade if valid."""
        # Don't stack trades
        if self.active_trade is not None:
            return

        bars = self.dollar_bars
        if len(bars) < 3:
            return

        bar_index = len(bars) - 1

        # Try bullish FVG (LONG trade)
        setup_key_long = (bar_index, "LONG")
        if setup_key_long not in self._traded_setups:
            fvg = self._detect_bullish_fvg(bars)
            if fvg:
                self._traded_setups.add(setup_key_long)
                self._enter_trade(fvg, bar, bar_index)
                return

        # Try bearish FVG (SHORT trade) — only if no LONG was taken
        setup_key_short = (bar_index, "SHORT")
        if setup_key_short not in self._traded_setups:
            fvg = self._detect_bearish_fvg(bars)
            if fvg:
                self._traded_setups.add(setup_key_short)
                self._enter_trade(fvg, bar, bar_index)

    def _enter_trade(self, fvg: dict, bar: DollarBar, bar_index: int):
        direction = "LONG" if fvg["direction"] == "bullish" else "SHORT"
        gap_range = fvg["gap_range"]
        gap_size = gap_range["top"] - gap_range["bottom"]

        if direction == "LONG":
            entry_price = gap_range["bottom"]
            tp_price = gap_range["top"]
            sl_price = entry_price - gap_size * SL_MULTIPLIER
        else:
            entry_price = gap_range["top"]
            tp_price = gap_range["bottom"]
            sl_price = entry_price + gap_size * SL_MULTIPLIER

        self.active_trade = ActiveTrade(
            bar_index=bar_index,
            entry_time=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )

        logger.info(
            f"🔔 ENTER {direction} | entry ${entry_price:.2f} "
            f"TP ${tp_price:.2f} SL ${sl_price:.2f} | gap ${fvg['gap_size']:.2f}"
        )

    # ------------------------------------------------------------------ #
    # FVG detection (identical conditions to backtest fvg_detection.py)   #
    # ------------------------------------------------------------------ #

    def _detect_bullish_fvg(self, bars: list[DollarBar]) -> Optional[dict]:
        """Bullish FVG: candle_1.close > candle_3.open, gap between c1.high and c3.low."""
        if len(bars) < 3:
            return None

        c1, c3 = bars[-3], bars[-1]
        if c1.close <= c3.open:
            return None

        gap_top = c1.high
        gap_bottom = c3.low
        if gap_top <= gap_bottom:
            return None

        gap_size = (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE

        if gap_size < ATR_THRESHOLD * self._calculate_atr(bars):
            return None
        if not self._check_volume(bars, bullish=True):
            return None
        if gap_size > MAX_GAP_DOLLARS:
            return None

        return {
            "direction": "bullish",
            "gap_range": {"top": gap_top, "bottom": gap_bottom},
            "gap_size": gap_size,
        }

    def _detect_bearish_fvg(self, bars: list[DollarBar]) -> Optional[dict]:
        """Bearish FVG: candle_1.close < candle_3.open, gap between c3.high and c1.low."""
        if len(bars) < 3:
            return None

        c1, c3 = bars[-3], bars[-1]
        if c1.close >= c3.open:
            return None

        gap_top = c3.high
        gap_bottom = c1.low
        if gap_top <= gap_bottom:
            return None

        gap_size = (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE

        if gap_size < ATR_THRESHOLD * self._calculate_atr(bars):
            return None
        if not self._check_volume(bars, bullish=False):
            return None
        if gap_size > MAX_GAP_DOLLARS:
            return None

        return {
            "direction": "bearish",
            "gap_range": {"top": gap_top, "bottom": gap_bottom},
            "gap_size": gap_size,
        }

    def _calculate_atr(self, bars: list[DollarBar]) -> float:
        if len(bars) < 14:
            return 10.0
        tr_values = []
        for i in range(1, min(15, len(bars))):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            tr_values.append(tr)
        return sum(tr_values) / len(tr_values) if tr_values else 10.0

    def _check_volume(self, bars: list[DollarBar], bullish: bool) -> bool:
        if len(bars) < 20:
            return True
        up_vol = sum(b.volume for b in bars[-20:] if b.close > b.open)
        dn_vol = sum(b.volume for b in bars[-20:] if b.close < b.open)
        if bullish:
            return dn_vol == 0 or (up_vol / dn_vol) >= VOLUME_RATIO_THRESHOLD
        else:
            return up_vol == 0 or (dn_vol / up_vol) >= VOLUME_RATIO_THRESHOLD

    # ------------------------------------------------------------------ #
    # P&L calculation (FIX: use direction.upper() for comparison)         #
    # ------------------------------------------------------------------ #

    def _calculate_pnl(self, entry: float, exit_price: float, direction: str) -> float:
        """Calculate realized P&L after transaction costs.

        Uses MNQ_CONTRACT_VALUE ($5/point) to match backtest convention exactly.
        direction is case-insensitive ("long" or "LONG" both work).
        """
        if direction.upper() == "LONG":
            price_diff = exit_price - entry
        else:
            price_diff = entry - exit_price

        return price_diff * MNQ_CONTRACT_VALUE * CONTRACTS_PER_TRADE - TRANSACTION_COST

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def _print_status(self):
        wins = sum(1 for t in self.completed_trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.completed_trades)
        win_rate = (wins / len(self.completed_trades) * 100) if self.completed_trades else 0

        logger.info("-" * 70)
        logger.info(f"📊 STATUS | Bars: {len(self.dollar_bars)} | "
                    f"Trades: {len(self.completed_trades)} | "
                    f"Active: {'YES' if self.active_trade else 'no'}")
        logger.info(f"   Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:.2f}")
        logger.info("-" * 70)

    def _print_final_report(self):
        if not self.completed_trades:
            logger.info("No trades completed in this session.")
            return

        wins = [t for t in self.completed_trades if t.pnl > 0]
        losses = [t for t in self.completed_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.completed_trades)
        win_rate = len(wins) / len(self.completed_trades) * 100

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = sum(abs(t.pnl) for t in losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        tp_exits = sum(1 for t in self.completed_trades if t.exit_type == "tp")
        sl_exits = sum(1 for t in self.completed_trades if t.exit_type == "sl")
        time_exits = sum(1 for t in self.completed_trades if t.exit_type == "time")

        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0

        logger.info("=" * 70)
        logger.info("FINAL PERFORMANCE REPORT")
        logger.info("=" * 70)
        logger.info(f"Total Trades:    {len(self.completed_trades)}")
        logger.info(f"Wins / Losses:   {len(wins)} / {len(losses)}")
        logger.info(f"Win Rate:        {win_rate:.2f}%  (backtest target: 74.45%)")
        logger.info(f"Profit Factor:   {profit_factor:.2f}  (backtest target: 1.75)")
        logger.info(f"Total P&L:       ${total_pnl:.2f}")
        logger.info(f"Expectancy:      ${total_pnl / len(self.completed_trades):.2f}/trade")
        logger.info(f"Avg Win:         ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        logger.info(f"Exit breakdown:  TP={tp_exits}  SL={sl_exits}  Time={time_exits}")
        logger.info(f"Bars processed:  {len(self.dollar_bars)}")
        logger.info("=" * 70)


async def main():
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

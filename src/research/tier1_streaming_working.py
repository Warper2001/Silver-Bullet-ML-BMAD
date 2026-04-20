#!/usr/bin/env python3
"""
TIER 1 FVG Paper Trading - TradeStation HTTP Polling + SIM Order Placement
Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0

Entry fires a bracket order on SIM account (entry + TP limit + SL stop).
The SIM account manages TP/SL fills. Local per-bar simulation is the
authoritative P&L record and handles the time-stop (cancel bracket + flat close).
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

# TradeStation market data API
SYMBOL = "MNQM26"  # June 2026 contract (most active)
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
BARS_URL = (f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
            f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}&bars_back=100")
POLL_INTERVAL_SECONDS = 60

# TradeStation SIM order placement
SIM_ACCOUNT_ID = "SIM279251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orders"

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
    sim_entry_order_id: Optional[str] = None
    sim_tp_order_id: Optional[str] = None
    sim_sl_order_id: Optional[str] = None


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
    sim_order_id: Optional[str] = None


class Tier1StreamingTrader:
    """TIER 1 FVG Trading — polls TradeStation for complete 1-minute bars.

    Architecture:
    - Poll every 60s for the latest complete bar
    - On each new bar: first advance active trades (check TP/SL/time exit),
      then detect new FVG setups
    - One active trade at a time (prevents compounding open risk)
    - Entry fires a SIM bracket order (entry + TP limit + SL stop)
    - Time-stop cancels bracket legs and sends a flat market close order
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
        logger.info("TIER 1 FVG PAPER TRADING - SIM ORDER PLACEMENT")
        logger.info("=" * 70)
        logger.info(f"Configuration: {TIER1_CONFIG}")
        logger.info(f"Symbol: {SYMBOL} (MNQ June 2026)")
        logger.info(f"SIM Account: {SIM_ACCOUNT_ID}")
        logger.info(f"Data Source: TradeStation API (polling every {POLL_INTERVAL_SECONDS}s)")
        logger.info(f"Mode: Bracket orders on SIM (entry + TP limit + SL stop)")
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
            await self._close_active_trade(last_bar, last_bar.close, "time")
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

                    await self._advance_active_trade(bar)
                    await self._detect_and_enter(bar)

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

    async def _advance_active_trade(self, bar: DollarBar):
        """Check if the new bar hits TP, SL, or time-stop for the active trade."""
        if self.active_trade is None:
            return

        trade = self.active_trade
        trade.bars_held += 1

        if trade.direction == "LONG":
            if bar.low <= trade.sl_price:
                exit_price = min(trade.sl_price, bar.low)
                await self._close_active_trade(bar, exit_price, "sl")
                return
            if bar.high >= trade.tp_price:
                await self._close_active_trade(bar, trade.tp_price, "tp")
                return
        else:  # SHORT
            if bar.high >= trade.sl_price:
                exit_price = max(trade.sl_price, bar.high)
                await self._close_active_trade(bar, exit_price, "sl")
                return
            if bar.low <= trade.tp_price:
                await self._close_active_trade(bar, trade.tp_price, "tp")
                return

        if trade.bars_held >= MAX_HOLD_BARS:
            await self._close_active_trade(bar, bar.close, "time")

    async def _close_active_trade(self, bar: DollarBar, exit_price: float, exit_type: str):
        trade = self.active_trade
        pnl = self._calculate_pnl(trade.entry_price, exit_price, trade.direction)
        result = "✅ WIN" if pnl > 0 else "❌ LOSS"

        # Time-stop: bracket legs still open on SIM → cancel them + send flat close
        if exit_type == "time":
            if trade.sim_tp_order_id:
                await self._cancel_sim_order(trade.sim_tp_order_id)
            if trade.sim_sl_order_id:
                await self._cancel_sim_order(trade.sim_sl_order_id)
            await self._submit_close_order(trade.direction)
            sim_note = "SIM bracket cancelled + flat close sent"
        elif exit_type in ("tp", "sl"):
            # Bracket already filled on SIM automatically — no additional order needed
            sim_note = "SIM closed (bracket)"
        else:
            sim_note = ""

        completed = CompletedTrade(
            entry_time=trade.entry_time,
            exit_time=bar.timestamp,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            exit_type=exit_type,
            bars_held=trade.bars_held,
            pnl=pnl,
            sim_order_id=trade.sim_entry_order_id,
        )
        self.completed_trades.append(completed)
        self.active_trade = None

        logger.info(
            f"{result} {completed.direction} | {exit_type.upper()} "
            f"entry ${trade.entry_price:.2f} → exit ${exit_price:.2f} "
            f"| bars={trade.bars_held} | P&L ${pnl:.2f}"
            + (f" | {sim_note}" if sim_note else "")
        )

    # ------------------------------------------------------------------ #
    # FVG detection & trade entry                                          #
    # ------------------------------------------------------------------ #

    async def _detect_and_enter(self, bar: DollarBar):
        """Detect FVG on the just-completed bar and open a trade if valid."""
        if self.active_trade is not None:
            return

        bars = self.dollar_bars
        if len(bars) < 3:
            return

        bar_index = len(bars) - 1

        setup_key_long = (bar_index, "LONG")
        if setup_key_long not in self._traded_setups:
            fvg = self._detect_bullish_fvg(bars)
            if fvg:
                self._traded_setups.add(setup_key_long)
                await self._enter_trade(fvg, bar, bar_index)
                return

        setup_key_short = (bar_index, "SHORT")
        if setup_key_short not in self._traded_setups:
            fvg = self._detect_bearish_fvg(bars)
            if fvg:
                self._traded_setups.add(setup_key_short)
                await self._enter_trade(fvg, bar, bar_index)

    async def _enter_trade(self, fvg: dict, bar: DollarBar, bar_index: int):
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

        # Submit bracket order to SIM
        entry_id, tp_id, sl_id = await self._submit_bracket_order(
            direction, entry_price, tp_price, sl_price
        )
        self.active_trade.sim_entry_order_id = entry_id
        self.active_trade.sim_tp_order_id = tp_id
        self.active_trade.sim_sl_order_id = sl_id

        sim_tag = f" | SIM order #{entry_id}" if entry_id else " | SIM order FAILED"
        logger.info(
            f"🔔 ENTER {direction} | entry ${entry_price:.2f} "
            f"TP ${tp_price:.2f} SL ${sl_price:.2f} | gap ${fvg['gap_size']:.2f}"
            + sim_tag
        )

    # ------------------------------------------------------------------ #
    # SIM order placement                                                  #
    # ------------------------------------------------------------------ #

    async def _get_auth_headers(self) -> dict:
        token = await self.auth.authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _submit_bracket_order(
        self, direction: str, entry_price: float, tp_price: float, sl_price: float
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Submit a bracket order (entry + TP limit + SL stop) to SIM.

        Returns (entry_order_id, tp_order_id, sl_order_id).
        Returns (None, None, None) on any error — local simulation continues unaffected.
        """
        if direction == "LONG":
            entry_action = "BUY"
            exit_action = "SELL"
        else:
            entry_action = "SELLSHORT"
            exit_action = "BUYTOCOVER"

        payload = {
            "AccountID": SIM_ACCOUNT_ID,
            "Symbol": SYMBOL,
            "Quantity": str(CONTRACTS_PER_TRADE),
            "OrderType": "Market",
            "TradeAction": entry_action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
            "OSOs": [{
                "Type": "BRK",
                "Orders": [
                    {
                        "AccountID": SIM_ACCOUNT_ID,
                        "Symbol": SYMBOL,
                        "Quantity": str(CONTRACTS_PER_TRADE),
                        "OrderType": "Limit",
                        "TradeAction": exit_action,
                        "TimeInForce": {"Duration": "GTC"},
                        "LimitPrice": str(tp_price),
                    },
                    {
                        "AccountID": SIM_ACCOUNT_ID,
                        "Symbol": SYMBOL,
                        "Quantity": str(CONTRACTS_PER_TRADE),
                        "OrderType": "StopMarket",
                        "TradeAction": exit_action,
                        "TimeInForce": {"Duration": "GTC"},
                        "StopPrice": str(sl_price),
                    },
                ],
            }],
        }

        try:
            headers = await self._get_auth_headers()
            response = await self.client.post(SIM_ORDERS_URL, headers=headers, json=payload)
            logger.debug(f"SIM bracket response {response.status_code}: {response.text[:500]}")

            if response.status_code not in (200, 201):
                logger.warning(
                    f"⚠️  SIM bracket order failed HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None, None, None

            data = response.json()
            orders = data.get("Orders", [])
            entry_id = orders[0].get("OrderID") if len(orders) > 0 else None
            tp_id = orders[1].get("OrderID") if len(orders) > 1 else None
            sl_id = orders[2].get("OrderID") if len(orders) > 2 else None

            logger.info(
                f"✓ SIM bracket submitted | entry #{entry_id} | "
                f"TP #{tp_id} | SL #{sl_id}"
            )
            return entry_id, tp_id, sl_id

        except Exception as e:
            logger.warning(f"⚠️  SIM bracket order exception: {e}")
            return None, None, None

    async def _cancel_sim_order(self, order_id: str):
        """Cancel an open SIM order (used to cancel bracket legs on time-stop)."""
        try:
            headers = await self._get_auth_headers()
            url = f"https://sim-api.tradestation.com/v3/orders/{order_id}"
            response = await self.client.delete(url, headers=headers)
            if response.status_code in (200, 204, 404):
                logger.info(f"✓ SIM order #{order_id} cancelled (HTTP {response.status_code})")
            else:
                logger.warning(
                    f"⚠️  Cancel order #{order_id} returned HTTP {response.status_code}: "
                    f"{response.text[:100]}"
                )
        except Exception as e:
            logger.warning(f"⚠️  Cancel order #{order_id} exception: {e}")

    async def _submit_close_order(self, direction: str):
        """Submit a flat market order to close the SIM position (time-stop only)."""
        close_action = "SELL" if direction == "LONG" else "BUYTOCOVER"
        payload = {
            "AccountID": SIM_ACCOUNT_ID,
            "Symbol": SYMBOL,
            "Quantity": str(CONTRACTS_PER_TRADE),
            "OrderType": "Market",
            "TradeAction": close_action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }

        try:
            headers = await self._get_auth_headers()
            response = await self.client.post(SIM_ORDERS_URL, headers=headers, json=payload)
            if response.status_code in (200, 201):
                data = response.json()
                orders = data.get("Orders", [])
                order_id = orders[0].get("OrderID") if orders else "?"
                logger.info(f"✓ SIM flat close order #{order_id} submitted")
            else:
                logger.warning(
                    f"⚠️  SIM close order failed HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
        except Exception as e:
            logger.warning(f"⚠️  SIM close order exception: {e}")

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
    # P&L calculation                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_pnl(self, entry: float, exit_price: float, direction: str) -> float:
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

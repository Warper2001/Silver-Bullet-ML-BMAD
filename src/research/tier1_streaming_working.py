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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3
from src.data.models import DollarBar

# Configuration (SAME AS VALIDATED BACKTEST)
TIER1_CONFIG = "SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0_TODfilter"
SL_MULTIPLIER = 2.5
ATR_THRESHOLD = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS = 50.0
MAX_HOLD_BARS = 10
CONTRACTS_PER_TRADE = 1

# TOD filter — skip signal detection during these Eastern Time hours
# Source: backtest_tier1_tod_filter_report.txt "Block WR<75%" config
# Blocked: 01:00, 06:00, 08:00, 16:00, 17:00, 22:00, 23:00 ET
# Effect:  76.33% → 80.15% WR | PF 1.20 → 1.57 | +$2,531 over 5-month baseline
TOD_BLOCKED_HOURS_ET: frozenset[int] = frozenset({1, 6, 8, 16, 17, 22, 23})

# MNQ Specifications
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE  # $5/pt — backtest scaling for gap/ATR filters only
MNQ_DOLLAR_VALUE = 2.0  # real MNQ: $2 per index point per contract

# Transaction Costs — commission only; actual SIM fill prices capture real slippage
COMMISSION_PER_CONTRACT = 0.40  # actual TradeStation SIM rate per contract per side
TRANSACTION_COST = COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2  # $0.80/roundtrip

# TradeStation market data API
SYMBOL = "MNQM26"  # June 2026 contract (most active)
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
BARS_BASE_URL = (f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
                 f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}")
HISTORY_HOURS = 8    # hours of history on cold start (~480 bars)
POLL_INTERVAL_SECONDS = 60

# TradeStation SIM order placement
SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"

# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

_handlers: list = [logging.FileHandler(log_dir / 'tier1_streaming_working.log')]
if sys.stdout.isatty():
    _handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=_handlers,
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
    sim_entry_fill: Optional[float] = None  # actual fill price from SIM market order


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

        # True during the initial HISTORY_HOURS backfill — suppresses SIM orders
        self._is_backfill: bool = True

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
        logger.info(f"TOD filter: blocked ET hours {sorted(TOD_BLOCKED_HOURS_ET)} (WR<75% config)")
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
        if self._is_market_open():
            logger.info("🟢 Market is currently OPEN")
        else:
            logger.info(f"🔴 Market is currently CLOSED — will wake at {self._next_open_cst()}")

    async def start_streaming(self):
        """Poll for complete bars and process each new one exactly once."""
        logger.info(f"Starting polling loop (every {POLL_INTERVAL_SECONDS}s)...")
        self.running = True

        try:
            while self.running:
                if not self._is_market_open():
                    wait_secs = self._seconds_until_open()
                    h, m = divmod(wait_secs // 60, 60)
                    logger.info(
                        f"🔴 Market closed — sleeping {h}h {m}m until next open "
                        f"({self._next_open_cst()})"
                    )
                    # Sleep in 30-min chunks so we can recheck (handles DST, etc.)
                    await asyncio.sleep(min(wait_secs, 1800))
                    continue

                await self._poll_and_process()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"❌ Polling error: {e}", exc_info=True)
        finally:
            await self.stop()

    # ------------------------------------------------------------------ #
    # Market hours (CME MNQ, all times UTC; CST = UTC-6)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_market_open() -> bool:
        """Return True if CME MNQ is currently trading.

        Schedule (CST = UTC-6):
        - Mon-Thu: open 5 PM CST prior day → 4 PM CST (daily break 4-5 PM CST)
        - Friday:  open until 4 PM CST, then weekend begins
        - Saturday: closed
        - Sunday:  opens 5 PM CST
        In UTC: daily break 22:00-23:00; weekend Fri 22:00 → Sun 23:00.
        """
        now = datetime.now(timezone.utc)
        wd = now.weekday()   # 0=Mon … 6=Sun
        h  = now.hour

        if wd == 5:          # Saturday — always closed
            return False
        if wd == 6:          # Sunday — open only from 23:00 UTC
            return h >= 23
        if wd == 4:          # Friday — closes at 22:00 UTC, weekend begins
            return h < 22
        # Mon-Thu: closed only during the 22:00 hour (22:00-22:59 UTC)
        return h != 22

    @staticmethod
    def _seconds_until_open() -> int:
        """Seconds until the next market open window."""
        now = datetime.now(timezone.utc)
        wd  = now.weekday()
        h   = now.hour

        # Build the candidate next-open datetime in UTC
        today = now.replace(minute=0, second=0, microsecond=0)

        if wd == 5:  # Saturday → Sunday 23:00 UTC
            days_ahead = 1
            next_open = today.replace(hour=23) + timedelta(days=days_ahead)
        elif wd == 6 and h < 23:  # Sunday before open → today 23:00 UTC
            next_open = today.replace(hour=23)
        elif wd == 4 and h >= 22:  # Friday post-close → Sunday 23:00 UTC
            next_open = today.replace(hour=23) + timedelta(days=2)
        else:  # Mon-Thu daily break (hour == 22) → today 23:00 UTC
            next_open = today.replace(hour=23)

        return max(1, int((next_open - now).total_seconds()))

    @staticmethod
    def _next_open_cst() -> str:
        """Human-readable next open time in CST."""
        now = datetime.now(timezone.utc)
        secs = Tier1StreamingTrader._seconds_until_open()
        opens_utc = now + timedelta(seconds=secs)
        opens_cst = opens_utc - timedelta(hours=6)
        return opens_cst.strftime("%a %Y-%m-%d %H:%M CST")

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

    def _build_bars_url(self) -> str:
        """Build date-range URL. On cold start, go back HISTORY_HOURS; otherwise
        since the last processed bar (dedup handles overlap)."""
        if self._last_processed_timestamp is None:
            since = datetime.now(timezone.utc) - timedelta(hours=HISTORY_HOURS)
        else:
            since = self._last_processed_timestamp
        return f"{BARS_BASE_URL}&firstdate={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    async def _poll_and_process(self):
        """Fetch latest bars and process any new complete bars."""
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

            response = await self.client.get(self._build_bars_url(), headers=headers)
            if response.status_code != 200:
                logger.error(f"❌ API error {response.status_code}: {response.text[:200]}")
                return

            data = response.json()
            bars_data = data.get("Bars", [])
            if not bars_data:
                logger.warning("⚠️  No bars in API response")
                return

            new_bar_count = 0
            now_utc = datetime.now(timezone.utc)
            for bar_data in bars_data:
                bar = self._parse_bar(bar_data)
                if bar is None:
                    continue

                # Skip bars whose close time hasn't arrived yet (in-progress bar)
                if bar.timestamp > now_utc:
                    continue

                # Only process bars we haven't seen yet (dedup by timestamp)
                if (self._last_processed_timestamp is None
                        or bar.timestamp > self._last_processed_timestamp):
                    self.dollar_bars.append(bar)
                    self._last_processed_timestamp = bar.timestamp
                    new_bar_count += 1

                    await self._advance_active_trade(bar)
                    await self._detect_and_enter(bar, is_backfill=self._is_backfill)

                    logger.info(
                        f"📊 Bar {len(self.dollar_bars):4d} | {bar.timestamp} | "
                        f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} "
                        f"V:{bar.volume}"
                    )

            if new_bar_count == 0:
                logger.debug("No new complete bars this poll cycle")
            elif len(self.dollar_bars) % 10 == 0:
                self._print_status()

            # First poll is now complete — all subsequent bars are live
            if self._is_backfill and new_bar_count > 0:
                self._is_backfill = False
                logger.info(f"✅ Backfill complete ({len(self.dollar_bars)} bars) — live SIM orders now active")

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
        sim_exit_fill: Optional[float] = None

        # Time-stop: cancel bracket legs, then only flat-close if SIM still holds
        # the position.  If a TP/SL bracket leg already filled while we were
        # waiting for the next bar, sending a second close order would flip us
        # into an unintended ghost position on the other side.
        if exit_type == "time":
            tp_was_open = False
            sl_was_open = False
            if trade.sim_tp_order_id:
                tp_was_open = await self._cancel_sim_order(trade.sim_tp_order_id)
            if trade.sim_sl_order_id:
                sl_was_open = await self._cancel_sim_order(trade.sim_sl_order_id)

            bracket_already_filled = not tp_was_open and not sl_was_open
            if bracket_already_filled or not await self._sim_has_open_position(trade.direction):
                sim_note = "SIM bracket already filled — skipped flat close"
                logger.info(f"⏭  Time-stop: bracket already closed on SIM, no flat close needed")
            else:
                close_order_id = await self._submit_close_order(trade.direction)
                sim_note = "SIM bracket cancelled + flat close sent"
                if close_order_id:
                    sim_exit_fill = await self._fetch_order_fill(close_order_id)

        elif exit_type == "tp":
            sim_note = "SIM closed (bracket)"
            if trade.sim_tp_order_id:
                sim_exit_fill = await self._fetch_order_fill(trade.sim_tp_order_id)
        elif exit_type == "sl":
            sim_note = "SIM closed (bracket)"
            if trade.sim_sl_order_id:
                sim_exit_fill = await self._fetch_order_fill(trade.sim_sl_order_id)
        else:
            sim_note = ""

        # Use actual SIM fill prices when available; fall back to theoretical
        actual_entry = trade.sim_entry_fill if trade.sim_entry_fill is not None else trade.entry_price
        actual_exit = sim_exit_fill if sim_exit_fill is not None else exit_price
        pnl = self._calculate_pnl(actual_entry, actual_exit, trade.direction)
        result = "✅ WIN" if pnl > 0 else "❌ LOSS"

        # Build price annotation — show actual vs theoretical if they differ
        entry_str = f"${actual_entry:.2f}"
        exit_str = f"${actual_exit:.2f}"
        if trade.sim_entry_fill is None and trade.sim_entry_order_id:
            entry_str += " (theoretical)"
        if sim_exit_fill is None and trade.sim_entry_order_id:
            exit_str += " (theoretical)"

        completed = CompletedTrade(
            entry_time=trade.entry_time,
            exit_time=bar.timestamp,
            direction=trade.direction,
            entry_price=actual_entry,
            exit_price=actual_exit,
            exit_type=exit_type,
            bars_held=trade.bars_held,
            pnl=pnl,
            sim_order_id=trade.sim_entry_order_id,
        )
        self.completed_trades.append(completed)
        self.active_trade = None

        logger.info(
            f"{result} {completed.direction} | {exit_type.upper()} "
            f"entry {entry_str} → exit {exit_str} "
            f"| bars={trade.bars_held} | P&L ${pnl:.2f}"
            + (f" | {sim_note}" if sim_note else "")
        )

    # ------------------------------------------------------------------ #
    # FVG detection & trade entry                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bar_et_hour(ts: datetime) -> int:
        """UTC timestamp → Eastern hour (EDT = UTC-4 for months ≤10, EST = UTC-5 otherwise)."""
        offset = -4 if ts.month <= 10 else -5
        return (ts.hour + offset) % 24

    async def _detect_and_enter(self, bar: DollarBar, is_backfill: bool = False):
        """Detect FVG on the just-completed bar and open a trade if valid."""
        if self.active_trade is not None:
            return

        et_hour = self._bar_et_hour(bar.timestamp)
        if et_hour in TOD_BLOCKED_HOURS_ET:
            logger.debug(f"⏭  TOD filter: {et_hour:02d}:00 ET blocked")
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
                await self._enter_trade(fvg, bar, bar_index, is_backfill=is_backfill)
                return

        setup_key_short = (bar_index, "SHORT")
        if setup_key_short not in self._traded_setups:
            fvg = self._detect_bearish_fvg(bars)
            if fvg:
                self._traded_setups.add(setup_key_short)
                await self._enter_trade(fvg, bar, bar_index, is_backfill=is_backfill)

    async def _enter_trade(self, fvg: dict, bar: DollarBar, bar_index: int, is_backfill: bool = False):
        direction = "LONG" if fvg["direction"] == "bullish" else "SHORT"
        gap_range = fvg["gap_range"]
        gap_size = gap_range["top"] - gap_range["bottom"]

        if direction == "LONG":
            entry_price = gap_range["bottom"]
            tp_price = gap_range["top"]
            sl_price = round(round((entry_price - gap_size * SL_MULTIPLIER) / MNQ_TICK_SIZE) * MNQ_TICK_SIZE, 2)
        else:
            entry_price = gap_range["top"]
            tp_price = gap_range["bottom"]
            sl_price = round(round((entry_price + gap_size * SL_MULTIPLIER) / MNQ_TICK_SIZE) * MNQ_TICK_SIZE, 2)

        self.active_trade = ActiveTrade(
            bar_index=bar_index,
            entry_time=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )

        # Submit bracket order to SIM (suppressed during backfill)
        if not is_backfill:
            entry_id, tp_id, sl_id = await self._submit_bracket_order(
                direction, entry_price, tp_price, sl_price
            )
            self.active_trade.sim_entry_order_id = entry_id
            self.active_trade.sim_tp_order_id = tp_id
            self.active_trade.sim_sl_order_id = sl_id

            if entry_id:
                actual_fill = await self._fetch_order_fill(entry_id)
                if actual_fill is not None:
                    self.active_trade.sim_entry_fill = actual_fill
                    slip = actual_fill - entry_price if direction == "LONG" else entry_price - actual_fill
                    slip_str = f"+{slip:.2f}" if slip >= 0 else f"{slip:.2f}"
                    logger.info(
                        f"  → Entry fill: ${actual_fill:.2f} (theoretical ${entry_price:.2f}, "
                        f"slippage {slip_str} pts)"
                    )
                sim_tag = f" | SIM order #{entry_id}"
            else:
                sim_tag = " | SIM order FAILED"
        else:
            sim_tag = " | SIM order suppressed (backfill)"
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
            # Futures use BUY/SELL only (not SELLSHORT/BUYTOCOVER which are equities-only)
            entry_action = "SELL"
            exit_action = "BUY"

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
            # Submission response only has {"Message": "...", "OrderID": "..."} — no
            # OrderType field.  Identify each leg from the message text:
            #   "@ ... Stop Market" → SL, "@ ... Limit" → TP, else → entry (Market).
            entry_id = tp_id = sl_id = None
            for order in orders:
                msg = order.get("Message", "")
                oid = order.get("OrderID")
                if "Stop Market" in msg:
                    sl_id = oid
                elif "Limit" in msg:
                    tp_id = oid
                else:
                    entry_id = oid

            logger.info(
                f"✓ SIM bracket submitted | entry #{entry_id} | "
                f"TP #{tp_id} | SL #{sl_id}"
            )
            return entry_id, tp_id, sl_id

        except Exception as e:
            logger.warning(f"⚠️  SIM bracket order exception: {e}")
            return None, None, None

    async def _cancel_sim_order(self, order_id: str) -> bool:
        """Cancel an open SIM order.  Returns True if the order was still open
        (cancel accepted), False if it was already filled / not found."""
        try:
            headers = await self._get_auth_headers()
            url = f"https://sim-api.tradestation.com/v3/orderexecution/orders/{order_id}"
            response = await self.client.delete(url, headers=headers)
            if response.status_code in (200, 204):
                logger.info(f"✓ SIM order #{order_id} cancelled (HTTP {response.status_code})")
                return True
            elif response.status_code == 404:
                # Order already gone — filled or expired
                logger.info(f"⚠️  SIM order #{order_id} already gone (404) — likely already filled")
                return False
            else:
                logger.warning(
                    f"⚠️  Cancel order #{order_id} returned HTTP {response.status_code}: "
                    f"{response.text[:100]}"
                )
                return False
        except Exception as e:
            logger.warning(f"⚠️  Cancel order #{order_id} exception: {e}")
            return False

    async def _sim_has_open_position(self, direction: str) -> bool:
        """Return True if the SIM account holds an open position in the expected direction.

        Queried immediately before sending a time-stop flat close to avoid opening
        a ghost position when the bracket TP/SL already filled on the SIM.
        Defaults to True on any API error so the flat close is still sent."""
        try:
            headers = await self._get_auth_headers()
            r = await self.client.get(
                f"https://sim-api.tradestation.com/v3/brokerage/accounts/{SIM_ACCOUNT_ID}/positions",
                headers=headers,
            )
            if r.status_code != 200:
                logger.warning(f"⚠️  Position check HTTP {r.status_code} — assuming position open")
                return True
            for pos in r.json().get("Positions", []):
                if pos.get("Symbol") != SYMBOL:
                    continue
                qty = int(float(pos.get("Quantity", 0)))
                if direction == "LONG" and qty > 0:
                    return True
                if direction == "SHORT" and qty < 0:
                    return True
            return False
        except Exception as e:
            logger.warning(f"⚠️  Position check exception: {e} — assuming position open")
            return True

    async def _submit_close_order(self, direction: str) -> Optional[str]:
        """Submit a flat market order to close the SIM position (time-stop only).

        Returns the order ID on success, None on failure.
        """
        close_action = "SELL" if direction == "LONG" else "BUY"
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
                order_id = orders[0].get("OrderID") if orders else None
                logger.info(f"✓ SIM flat close order #{order_id} submitted")
                return order_id
            else:
                logger.warning(
                    f"⚠️  SIM close order failed HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None
        except Exception as e:
            logger.warning(f"⚠️  SIM close order exception: {e}")
            return None

    async def _fetch_order_fill(self, order_id: str, max_attempts: int = 3) -> Optional[float]:
        """Fetch actual fill price for a completed SIM order.

        Retries up to max_attempts times (1s apart) to allow market orders time to fill.
        Returns None on failure so callers can fall back to theoretical prices.
        """
        url = (f"https://sim-api.tradestation.com/v3/brokerage/accounts/"
               f"{SIM_ACCOUNT_ID}/orders/{order_id}")
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    await asyncio.sleep(1)
                headers = await self._get_auth_headers()
                r = await self.client.get(url, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    # Response may be {"Orders": [...]} or a single order object
                    orders = data.get("Orders", [data] if "FilledPrice" in data else [])
                    for o in orders:
                        if o.get("Status") == "FLL":
                            fill = float(o.get("FilledPrice", 0))
                            if fill > 0:
                                return fill
            except Exception as e:
                logger.warning(f"⚠️  Fill fetch attempt {attempt + 1} for #{order_id}: {e}")
        return None

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
        if len(bars) < 5:
            return 10.0
        window = bars[-20:]  # rolling 20-bar window, matches backtest methodology
        tr_values = []
        for i in range(1, len(window)):
            tr = max(
                window[i].high - window[i].low,
                abs(window[i].high - window[i - 1].close),
                abs(window[i].low - window[i - 1].close),
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
        return price_diff * MNQ_DOLLAR_VALUE * CONTRACTS_PER_TRADE - TRANSACTION_COST

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
        logger.info(f"Win Rate:        {win_rate:.2f}%  (backtest target: 80.15%)")
        logger.info(f"Profit Factor:   {profit_factor:.2f}  (backtest target: 1.57)")
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

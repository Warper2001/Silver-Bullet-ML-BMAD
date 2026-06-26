#!/usr/bin/env python3
"""
gap_fade_live.py — GAP-1 Panic-Open Mean-Reversion Fade — internal simulation

Pre-registration: _bmad-output/preregistration_gap_fade_panic_open.md (seal 32da5d5)

Strategy (FROZEN — no parameter changes without a new pre-registration):
  - Gap threshold: abs(rth_open − prior_rth_close) / prior_rth_close >= 0.5%
  - Direction: fade (gap up → SHORT, gap down → LONG)
  - Entry: open of the first RTH 1-min bar (09:30 ET)
  - Target (TP): prior RTH close
  - Stop (SL): entry ± 2.0 × gap_abs beyond open
  - Time-stop: close at the open of the first bar at/after 13:00 ET
  - Exclude Fridays (weekday == 4)
  - 1 MNQ contract, $2/point, max 1 trade/day

INTERNAL SIMULATION ONLY — no broker orders are placed. Fills are simulated
against bar OHLC using the same logic as backtest_gap_fade.py to faithfully
measure prospective OOS performance from 2026-06-26 onward.

OOS decision rule (per pre-reg, from first live trade):
  N >= 30 AND >= 30 calendar days:
    PF > 1.20  → scale to 2ct
    PF 1.00-1.20 → continue 1ct, re-evaluate at N=60
    PF < 1.00  → STOP — archive strategy

Usage:
  .venv/bin/python src/research/gap_fade_live.py
  .venv/bin/python src/research/gap_fade_live.py --replay data/processed/dollar_bars/1_minute/mnq_1min_2025.csv

Logs:
  data/gap_fade/trades.csv    — hash-chained; one row per closed sim trade
  data/gap_fade/decisions.csv — one row per qualifying gap-check (entered or skipped)
  data/gap_fade/state.json    — crash-recovery: current open sim position
  data/trades.db              — canonical cross-bot SQLite (trader_id='trader-gap-fade')
"""
import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.auth_v3 import TradeStationAuthV3
from src.monitoring.trade_db import TradeDatabase

# ─────────────────────────────────────────────────────────────────────────────
# Frozen strategy parameters — pre-registered 32da5d5; DO NOT change
# ─────────────────────────────────────────────────────────────────────────────
GAP_MIN_PCT    = 0.005   # 0.5% minimum overnight gap
STOP_MULT      = 2.0     # SL = entry ± STOP_MULT × gap_abs beyond open
TIME_STOP_HOUR = 13      # close at the open of the first bar at/after 13:00 ET
EXCLUDE_DOW    = {4}     # 4 = Friday (0=Mon … 4=Fri)
MIN_RTH_BARS   = 300     # minimum prior-session RTH bars to trust prior close
MNQ_PV         = 2.0     # $2.00 per point per MNQ contract
CONTRACTS      = 1

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TRADER_ID = "trader-gap-fade"
ET        = pytz.timezone("America/New_York")
BARSBACK  = 3000    # ~50h of 1-min bars — covers 2 full prior RTH sessions

# No auto-roll: update GAP_FADE_SYMBOL in trader-gap-fade.service at each quarterly roll.
# Next roll: ~2026-09-11 → MNQZ26
SYMBOL = os.environ.get("GAP_FADE_SYMBOL", "MNQU26")

DATA_DIR = BASE_DIR / "data" / "gap_fade"
LOG_DIR  = BASE_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("gap_fade")

TS_BARS_BASE = "https://api.tradestation.com/v3/marketdata/barcharts"
STATE_PATH   = DATA_DIR / "state.json"
DOW_NAMES    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ─────────────────────────────────────────────────────────────────────────────
# Hash-chained append-only CSV (same pattern as mim_nb_live / thursday_short)
# ─────────────────────────────────────────────────────────────────────────────
class ChainedCsv:
    """Tamper-evident append-only CSV with SHA-256 hash chaining."""

    def __init__(self, path: Path, fields: list):
        self.path   = path
        self.fields = list(fields) + ["chain"]
        self.head   = "GENESIS"
        if path.exists():
            try:
                with open(path) as f:
                    for row in csv.DictReader(f):
                        self.head = row.get("chain", self.head)
            except Exception:
                logger.warning("chain reload failed for %s — restarting chain", path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row: dict):
        payload   = "|".join(str(row.get(k, "")) for k in self.fields if k != "chain")
        self.head = hashlib.sha256((self.head + "|" + payload).encode()).hexdigest()[:16]
        row       = dict(row)
        row["chain"] = self.head
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Bar helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bar_et(bar: dict) -> datetime:
    """Parse TS bar TimeStamp → ET-aware datetime."""
    ts = datetime.fromisoformat(bar["TimeStamp"].replace("Z", "+00:00"))
    return ts.astimezone(ET)


def _is_rth(ts_et: datetime) -> bool:
    """True for 09:30 ≤ bar < 16:00 ET — matches backtest_gap_fade.py is_rth()."""
    h, m = ts_et.hour, ts_et.minute
    after_open   = (h == 9 and m >= 30) or h > 9
    before_close = h < 16   # 16:00 bar excluded (RTH_END exclusive in backtest)
    return after_open and before_close


# ─────────────────────────────────────────────────────────────────────────────
# Crash-recovery state file
# ─────────────────────────────────────────────────────────────────────────────
def _save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _load_state() -> Optional[dict]:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception as e:
        logger.warning("state.json read failed (%s) — treating as flat", e)
        return None


def _clear_state():
    STATE_PATH.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Parity replay (--replay mode: verify live path matches backtest)
# ─────────────────────────────────────────────────────────────────────────────
def _run_replay(csv_path: str) -> None:
    """Feed a historical 1-min bar CSV through the frozen strategy logic.

    Should reproduce the sealed Gate-0 result for mnq_1min_2025.csv:
      N=78, WR≈62.8%, PF≈1.760, Net≈$6,462

    A mismatch means the live path has drifted from the pre-registered spec.
    """
    import numpy as np
    import pandas as pd

    print(f"\n=== GAP-1 PARITY REPLAY: {csv_path} ===")
    df     = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ts_col = df["timestamp"]
    ts_col = ts_col.dt.tz_localize("UTC") if ts_col.dt.tz is None else ts_col.dt.tz_convert("UTC")
    df["timestamp"] = ts_col.dt.tz_convert(ET)
    df = df.set_index("timestamp")

    rth        = df[df.index.map(lambda t: _is_rth(t))].copy()
    rth["date_et"] = rth.index.date
    by         = rth.groupby("date_et")
    rth_closes = by["close"].last()
    rth_opens  = by["open"].first()
    rth_counts = by["close"].count()
    rth_dows   = by.apply(lambda g: g.index[0].weekday(), include_groups=False)

    trades = []
    dates  = sorted(rth_closes.index)
    for i in range(1, len(dates)):
        today = dates[i]; yest = dates[i - 1]
        if rth_counts[yest] < MIN_RTH_BARS: continue
        if rth_dows[today] in EXCLUDE_DOW:  continue
        pc  = rth_closes[yest]; ro = rth_opens[today]
        gap = ro - pc; gap_abs = abs(gap); gap_pct = gap_abs / pc
        if gap_pct < GAP_MIN_PCT: continue
        direction = -1 if gap > 0 else 1
        entry = ro; target = pc
        stop  = (entry + STOP_MULT * gap_abs) if direction == -1 else (entry - STOP_MULT * gap_abs)
        # Simulate bar-by-bar — skip the 09:30 opening bar (same as backtest)
        day_bars    = rth[rth["date_et"] == today].iloc[1:]
        outcome, exit_px = "eod", (day_bars["close"].iloc[-1] if len(day_bars) else entry)
        for ts_b, bar in day_bars.iterrows():
            if ts_b.hour >= TIME_STOP_HOUR:
                outcome = "time"; exit_px = bar["open"]; break
            if direction == -1:
                if bar["low"]  <= target: outcome = "fill"; exit_px = target; break
                if bar["high"] >= stop:   outcome = "stop"; exit_px = stop;   break
            else:
                if bar["high"] >= target: outcome = "fill"; exit_px = target; break
                if bar["low"]  <= stop:   outcome = "stop"; exit_px = stop;   break
        pnl_pts = direction * (exit_px - entry)
        trades.append({"date": today, "dir": "long" if direction == 1 else "short",
                       "entry": entry, "exit_px": exit_px, "outcome": outcome,
                       "pnl_pts": pnl_pts, "pnl_usd": pnl_pts * MNQ_PV})

    if not trades:
        print("No trades generated — check CSV format / date range"); return

    pnls = np.array([t["pnl_usd"] for t in trades])
    wins = pnls > 0; losses = pnls < 0
    gw   = pnls[wins].sum(); gl = abs(pnls[losses].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    print(f"N={len(trades)}, WR={wins.mean()*100:.1f}%, PF={pf:.3f}, Net=${pnls.sum():.0f}")
    by_month: dict = {}
    for t in trades:
        m = str(t["date"])[:7]
        by_month[m] = by_month.get(m, 0) + t["pnl_usd"]
    print("Monthly P&L:")
    for m in sorted(by_month):
        print(f"  {m}: ${by_month[m]:+.0f}")
    by_out: dict = {}
    for t in trades: by_out[t["outcome"]] = by_out.get(t["outcome"], 0) + 1
    print(f"Exits: {by_out}")
    print("\nExpected (2025 IS sealed result): N=78, WR=62.8%, PF=1.760, Net=$6,462")


# ─────────────────────────────────────────────────────────────────────────────
# Main trader class
# ─────────────────────────────────────────────────────────────────────────────
class GapFadeTrader:
    """Internal-simulation gap-fade bot.

    Polls TradeStation 1-min bars, detects qualifying overnight gaps at the RTH
    open, and simulates fills against bar OHLC — no broker orders. Logs completed
    trades to data/gap_fade/trades.csv (hash-chained) and data/trades.db.
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol  = symbol
        self.running = False
        self.auth:   Optional[TradeStationAuthV3] = None
        self.http:   Optional[httpx.AsyncClient]  = None
        self._db     = TradeDatabase()
        self._trades_log:    Optional[ChainedCsv] = None
        self._decisions_log: Optional[ChainedCsv] = None

        # Per-session state (reset at each new RTH day)
        self._today_et:        Optional[str]   = None   # "2026-06-26"
        self._prior_rth_close: Optional[float] = None
        self._today_setup_done = False   # True once gap detection has run today

        # Sim trade state (persisted to state.json)
        self._trade_open  = False
        self._direction:  Optional[int]      = None   # +1 long / -1 short
        self._entry:      Optional[float]    = None
        self._target:     Optional[float]    = None
        self._stop:       Optional[float]    = None
        self._gap_pct:    Optional[float]    = None
        self._gap_abs:    Optional[float]    = None
        self._last_bar_ts: Optional[datetime] = None  # last bar checked for exit

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self):
        self.auth = TradeStationAuthV3.from_file(".access_token")
        await self.auth.authenticate()
        await self.auth.start_auto_refresh()   # critical — tokens expire ~20 min (lesson d4c0c39)
        self.http = httpx.AsyncClient(timeout=30)
        self._trades_log = ChainedCsv(
            DATA_DIR / "trades.csv",
            ["date_et", "dir", "gap_pct", "gap_abs_pts",
             "entry", "exit_px", "target", "stop",
             "outcome", "pnl_pts", "pnl_usd"],
        )
        self._decisions_log = ChainedCsv(
            DATA_DIR / "decisions.csv",
            ["date_et", "dow", "gap_pct", "gap_abs_pts",
             "prior_close", "rth_open", "action", "detail"],
        )
        # Crash recovery: reload open sim position from disk
        st = _load_state()
        if st and st.get("trade_open"):
            logger.info("Crash recovery: reloading open sim trade → %s", st)
            self._today_et         = st["today_et"]
            self._prior_rth_close  = st["prior_rth_close"]
            self._direction        = st["direction"]
            self._entry            = st["entry"]
            self._target           = st["target"]
            self._stop             = st["stop"]
            self._gap_pct          = st["gap_pct"]
            self._gap_abs          = st["gap_abs"]
            lbt = st.get("last_bar_ts")
            self._last_bar_ts      = datetime.fromisoformat(lbt).astimezone(ET) if lbt else None
            self._trade_open       = True
            self._today_setup_done = True   # don't re-detect gap after crash recovery

        logger.info("=" * 65)
        logger.info("GAP-1 PANIC-OPEN FADE — internal simulation — %s ×%d ct", self.symbol, CONTRACTS)
        logger.info("Pre-reg 32da5d5 | OOS: N≥30+30d | PF>1.20→2ct | PF<1.00→archive")
        logger.info("Logging → data/gap_fade/ + data/trades.db (%s)", TRADER_ID)
        logger.info("=" * 65)

    async def stop(self):
        logger.info("GAP-1 trader stopping")
        self.running = False
        if self.http:
            await self.http.aclose()

    # ── Bar fetching ──────────────────────────────────────────────────────────

    async def _fetch_bars(self) -> list:
        token = await self.auth.authenticate()
        url   = f"{TS_BARS_BASE}/{self.symbol}?interval=1&unit=Minute&barsback={BARSBACK}"
        r     = await self.http.get(url, headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 200:
            logger.warning("Bars API HTTP %s — skipping tick", r.status_code)
            return []
        return r.json().get("Bars", [])

    # ── Session map ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_session_map(bars: list) -> dict:
        """Group RTH bars by ET date.
        Returns {date_iso: {"bars": [(et, bar), ...], "n_bars": int, "open": float, "close": float}}.
        """
        sessions: dict = {}
        for bar in bars:
            et = _bar_et(bar)
            if not _is_rth(et):
                continue
            d = et.date().isoformat()
            if d not in sessions:
                sessions[d] = {"bars": [], "open": float(bar["Open"]), "close": None}
            sessions[d]["bars"].append((et, bar))
            sessions[d]["close"] = float(bar["Close"])
        for sess in sessions.values():
            sess["n_bars"] = len(sess["bars"])
        return sessions

    # ── Sim trade lifecycle ───────────────────────────────────────────────────

    def _open_trade(self, direction: int, entry: float, target: float, stop: float,
                    gap_pct: float, gap_abs: float,
                    today_et: str, dow: int, entry_bar_et: datetime):
        dir_str = "long" if direction == 1 else "short"
        logger.info(
            "🟢 SIM ENTRY [%s] | gap=%.1f pts (%.2f%%) | entry=%.2f target=%.2f stop=%.2f | %s",
            dir_str, gap_abs, gap_pct * 100, entry, target, stop, today_et,
        )
        self._trade_open       = True
        self._today_setup_done = True
        self._direction        = direction
        self._entry            = entry
        self._target           = target
        self._stop             = stop
        self._gap_pct          = gap_pct
        self._gap_abs          = gap_abs
        self._last_bar_ts      = entry_bar_et   # exit checking starts from the NEXT bar

        self._decisions_log.append({
            "date_et":     today_et,
            "dow":         DOW_NAMES[dow],
            "gap_pct":     round(gap_pct * 100, 3),
            "gap_abs_pts": round(gap_abs, 2),
            "prior_close": round(self._prior_rth_close, 2),
            "rth_open":    round(entry, 2),
            "action":      "ENTERED",
            "detail":      dir_str,
        })
        _save_state({
            "trade_open":      True,
            "today_et":        today_et,
            "prior_rth_close": self._prior_rth_close,
            "direction":       direction,
            "entry":           entry,
            "target":          target,
            "stop":            stop,
            "gap_pct":         gap_pct,
            "gap_abs":         gap_abs,
            "last_bar_ts":     entry_bar_et.isoformat(),
        })

    def _close_trade(self, exit_price: float, reason: str, exit_ts: datetime):
        pnl_pts = self._direction * (exit_price - self._entry)
        pnl_usd = pnl_pts * MNQ_PV * CONTRACTS
        dir_str = "L" if self._direction == 1 else "S"
        icon    = "✅" if pnl_usd >= 0 else "🔴"
        logger.info(
            "%s SIM EXIT [%s] | entry=%.2f exit=%.2f | %+.1f pts | %+.2f USD | %s",
            icon, reason, self._entry, exit_price, pnl_pts, pnl_usd, self._today_et,
        )
        # Canonical cross-bot SQLite (idempotent INSERT OR IGNORE)
        self._db.log_trade(
            trader_id   = TRADER_ID,
            timestamp   = exit_ts.astimezone(timezone.utc).isoformat(),
            pnl         = round(pnl_usd, 2),
            symbol      = self.symbol,
            direction   = dir_str,
            entry_price = round(self._entry, 4),
            exit_price  = round(exit_price, 4),
            exit_reason = reason,
            metadata    = {
                "gap_pct":     round(self._gap_pct * 100, 3),
                "gap_abs_pts": round(self._gap_abs, 2),
                "target":      round(self._target, 2),
                "stop":        round(self._stop, 2),
                "contracts":   CONTRACTS,
                "simulated":   True,
            },
        )
        # Hash-chained CSV
        self._trades_log.append({
            "date_et":     self._today_et,
            "dir":         dir_str,
            "gap_pct":     round(self._gap_pct * 100, 3),
            "gap_abs_pts": round(self._gap_abs, 2),
            "entry":       round(self._entry, 2),
            "exit_px":     round(exit_price, 2),
            "target":      round(self._target, 2),
            "stop":        round(self._stop, 2),
            "outcome":     reason,
            "pnl_pts":     round(pnl_pts, 2),
            "pnl_usd":     round(pnl_usd, 2),
        })
        # Reset sim trade state
        self._trade_open  = False
        self._direction   = None
        self._entry       = None
        self._target      = None
        self._stop        = None
        self._gap_pct     = None
        self._gap_abs     = None
        self._last_bar_ts = None
        _clear_state()

    def _check_exit_on_bar(self, bar_et: datetime, bar: dict) -> Optional[tuple]:
        """Return (exit_price, reason) if this bar triggers an exit, else None.

        Priority order matches simulate_day() in backtest_gap_fade.py:
          1. Time-stop (bar at/after 13:00 ET) → exit at bar open
          2. TP hit → exit at target
          3. SL hit → exit at stop
        """
        if bar_et.hour >= TIME_STOP_HOUR:
            return float(bar["Open"]), "time"
        lo = float(bar["Low"]); hi = float(bar["High"])
        if self._direction == -1:   # short: target below, stop above
            if lo <= self._target: return self._target, "fill"
            if hi >= self._stop:   return self._stop,   "stop"
        else:                       # long: target above, stop below
            if hi >= self._target: return self._target, "fill"
            if lo <= self._stop:   return self._stop,   "stop"
        return None

    # ── Main processing tick ──────────────────────────────────────────────────

    async def _process_tick(self):
        bars = await self._fetch_bars()
        if not bars:
            return
        sessions = self._build_session_map(bars)
        if not sessions:
            return

        now_et   = datetime.now(ET)
        today_et = now_et.date().isoformat()
        if today_et not in sessions:
            return   # pre-market or weekend — no RTH bars yet

        today_sess   = sessions[today_et]
        sorted_dates = sorted(sessions.keys())
        prior_dates  = [d for d in sorted_dates if d < today_et]

        # ── New RTH session detected ──────────────────────────────────────────
        if self._today_et != today_et:
            if not prior_dates:
                logger.debug("No prior RTH session in bar window — skipping session init")
                return
            prior_date = prior_dates[-1]
            prior_sess = sessions[prior_date]
            if prior_sess["n_bars"] < MIN_RTH_BARS:
                logger.info("Prior session %s: %d RTH bars < %d required — skipping",
                            prior_date, prior_sess["n_bars"], MIN_RTH_BARS)
                return

            # Carryover: EOD-close any open sim trade the time-stop should have caught
            if self._trade_open:
                prior_last_et = prior_sess["bars"][-1][0]
                logger.warning(
                    "Session boundary with open sim trade from %s — EOD carryover close at %.2f",
                    self._today_et, prior_sess["close"],
                )
                self._close_trade(
                    exit_price = prior_sess["close"],
                    reason     = "eod_carryover",
                    exit_ts    = prior_last_et,
                )

            # Reset for new session
            self._today_et         = today_et
            self._prior_rth_close  = prior_sess["close"]
            self._today_setup_done = False
            logger.info("New session %s | prior RTH close = %.2f", today_et, self._prior_rth_close)

        # ── Gap detection (once per session) ──────────────────────────────────
        if not self._trade_open and not self._today_setup_done:
            today_bars = today_sess["bars"]
            if not today_bars:
                return
            first_bar_et, first_bar = today_bars[0]
            rth_open = float(first_bar["Open"])
            gap      = rth_open - self._prior_rth_close
            gap_abs  = abs(gap)
            gap_pct  = gap_abs / self._prior_rth_close
            dow      = first_bar_et.weekday()
            dow_name = DOW_NAMES[dow] if dow < 7 else "?"

            if dow in EXCLUDE_DOW:
                action = "SKIPPED_FRIDAY"
            elif gap_pct < GAP_MIN_PCT:
                action = "NO_SETUP"
            else:
                direction = -1 if gap > 0 else 1
                target    = self._prior_rth_close
                stop      = (rth_open + STOP_MULT * gap_abs) if direction == -1 \
                            else (rth_open - STOP_MULT * gap_abs)
                self._open_trade(
                    direction=direction, entry=rth_open,
                    target=target, stop=stop,
                    gap_pct=gap_pct, gap_abs=gap_abs,
                    today_et=today_et, dow=dow, entry_bar_et=first_bar_et,
                )
                action = "ENTERED"

            self._today_setup_done = True
            if action != "ENTERED":
                self._decisions_log.append({
                    "date_et":     today_et,
                    "dow":         dow_name,
                    "gap_pct":     round(gap_pct * 100, 3),
                    "gap_abs_pts": round(gap_abs, 2),
                    "prior_close": round(self._prior_rth_close, 2),
                    "rth_open":    round(rth_open, 2),
                    "action":      action,
                    "detail":      "",
                })
                logger.info("Gap check: %s | gap=%.2f%% | %s | %s",
                            action, gap_pct * 100, dow_name, today_et)

        # ── Exit checking for open sim trade ───────────────────────────────────
        if self._trade_open:
            today_bars = today_sess["bars"]
            for bar_et, bar in today_bars:
                if self._last_bar_ts is not None and bar_et <= self._last_bar_ts:
                    continue   # already processed
                self._last_bar_ts = bar_et
                # Keep state file current so crash recovery catches the latest progress
                _save_state({
                    "trade_open":      True,
                    "today_et":        self._today_et,
                    "prior_rth_close": self._prior_rth_close,
                    "direction":       self._direction,
                    "entry":           self._entry,
                    "target":          self._target,
                    "stop":            self._stop,
                    "gap_pct":         self._gap_pct,
                    "gap_abs":         self._gap_abs,
                    "last_bar_ts":     bar_et.isoformat(),
                })
                result = self._check_exit_on_bar(bar_et, bar)
                if result:
                    exit_price, reason = result
                    self._close_trade(exit_price, reason, bar_et)
                    return   # trade closed this tick

            # Still open — log live unrealized P&L
            if today_bars:
                last_et, last_bar = today_bars[-1]
                cur    = float(last_bar["Close"])
                unreal = self._direction * (cur - self._entry) * MNQ_PV
                logger.info(
                    "SIM OPEN [%s] | entry=%.2f cur=%.2f | unreal %+.2f USD | %s %s ET",
                    "L" if self._direction == 1 else "S",
                    self._entry, cur, unreal, today_et,
                    last_et.strftime("%H:%M"),
                )

    # ── Run loop ──────────────────────────────────────────────────────────────

    async def run(self):
        self.running = True
        consecutive_failures = 0
        logger.info("Poll loop starting — 60s during RTH, 300s outside")
        while self.running:
            try:
                await self._process_tick()
                consecutive_failures = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error("Tick error #%d: %s", consecutive_failures, e, exc_info=True)
                if consecutive_failures >= 10:
                    logger.error("10 consecutive tick failures — stopping")
                    self.running = False
                    break

            now_et = datetime.now(ET)
            in_active_window = (
                now_et.weekday() < 5
                and (now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 25))
                and now_et.hour < 16
            )
            await asyncio.sleep(60 if (in_active_window or self._trade_open) else 300)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
async def _main_async():
    trader = GapFadeTrader(symbol=SYMBOL)
    await trader.initialize()
    loop = asyncio.get_running_loop()

    def _stop():
        asyncio.ensure_future(trader.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)
    try:
        await trader.run()
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=True)
    finally:
        await trader.stop()


def main():
    parser = argparse.ArgumentParser(
        description="GAP-1 Panic-Open Mean-Reversion Fade — internal simulation"
    )
    parser.add_argument(
        "--replay", metavar="CSV",
        help=(
            "Parity check: replay a 1-min bar CSV through the frozen strategy logic. "
            "Should reproduce N=78, WR≈62.8%%, PF≈1.760 on mnq_1min_2025.csv."
        ),
    )
    args = parser.parse_args()
    if args.replay:
        _run_replay(args.replay)
    else:
        asyncio.run(_main_async())


if __name__ == "__main__":
    main()

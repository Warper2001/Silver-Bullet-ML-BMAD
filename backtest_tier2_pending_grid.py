"""
Grid search over MAX_PENDING_BARS × MAX_HOLD_BARS using full 2025 1-min history.

Finds the optimal limit-entry fill timeout and active hold timeout independently.
No API calls — uses local data/processed/dollar_bars/1_minute/mnq_1min_2025.csv.

Usage:
  .venv/bin/python backtest_tier2_pending_grid.py
"""
import asyncio
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import logging

import numpy as np
import pytz

sys.path.insert(0, str(Path(__file__).parent))
import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

# Suppress all logging output during grid search — the logger writes ~130k lines/run
# to the shared log file, making each combination 100× slower than it needs to be.
logging.disable(logging.CRITICAL)

LOCAL_CSV = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
ET_TZ = pytz.timezone("US/Eastern")

# Refined search: Phase 1 (hold sensitivity at pending=180) +
#                 Phase 2 (pending sensitivity at hold=60)
GRID_COMBOS = [
    # Only the missing last row
    (240, 60),
]
# Deduplicate while preserving order (180,60 appears in both phases)
seen = set()
GRID_COMBOS = [(p, h) for p, h in GRID_COMBOS if not (p, h) in seen and not seen.add((p, h))]


class RunResult(NamedTuple):
    pending_bars: int
    hold_bars: int
    trades: int
    wins: int
    gross_profit: float
    gross_loss: float
    net_pnl: float
    max_drawdown: float  # largest peak-to-trough decline in cumulative P&L

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / abs(self.gross_loss) if self.gross_loss < 0 else float("inf")


def load_bars() -> list[DollarBar]:
    bars = []
    with open(LOCAL_CSV) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bars.append(DollarBar.model_construct(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(float(row["volume"])),
                notional_value=float(row["notional"]),
                is_forward_filled=False,
            ))
    return bars


async def run_one(bars: list[DollarBar], pending_bars: int, hold_bars: int) -> RunResult:
    tier2_mod.MAX_PENDING_BARS = pending_bars
    tier2_mod.MAX_HOLD_BARS = hold_bars

    trader = Tier2StreamingTrader()

    async def _no_bracket(*a, **kw): return (None, None, None)
    async def _no_cancel(*a, **kw): return True
    async def _no_close(*a, **kw): return None

    trader._submit_bracket_order = _no_bracket
    trader._cancel_sim_order     = _no_cancel
    trader._submit_close_order   = _no_close
    trader.ml_filter._log_decision = lambda *a, **kw: None

    last_h1_ts = None  # track which H1 bar we last updated structure for

    for bar in bars:
        trader.dollar_bars.append(bar)
        trader._last_processed_timestamp = bar.timestamp

        bar_et = bar.timestamp.astimezone(ET_TZ)
        if trader._current_day != bar_et.date():
            if trader._current_day is not None:
                trader._daily_ranges.append(trader._session_high - trader._session_low)
                if len(trader._daily_ranges) > 20:
                    trader._daily_ranges.pop(0)
            trader._current_day = bar_et.date()
            trader._session_open_price = np.nan
            trader._session_high, trader._session_low = bar.high, bar.low
        else:
            trader._session_high = max(trader._session_high, bar.high)
            trader._session_low  = min(trader._session_low,  bar.low)
        if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
            trader._session_open_price = bar.open

        # Only recompute H1 structure at H1 boundaries (60× speedup vs every 1m bar).
        # Detection result is identical: sweeps are flagged on completed H1 bars and
        # stay active for SWEEP_WINDOW_HOURS regardless of per-minute calls.
        h1_ts = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        if h1_ts != last_h1_ts:
            trader._update_h1_structure()
            last_h1_ts = h1_ts

        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(bar, is_backfill=False)

    trades = trader.completed_trades
    wins         = sum(1 for t in trades if t.pnl > 0)
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss   = sum(t.pnl for t in trades if t.pnl < 0)
    net_pnl      = gross_profit + gross_loss

    # Max drawdown: largest peak-to-trough drop in cumulative P&L
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum_pnl += t.pnl
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd

    return RunResult(pending_bars, hold_bars, len(trades), wins, gross_profit, gross_loss, net_pnl, max_dd)


async def main() -> None:
    print(f"Loading bars from {LOCAL_CSV} …", flush=True)
    bars = load_bars()
    print(f"Loaded {len(bars):,} bars  ({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})\n")

    total = len(GRID_COMBOS)
    results: list[RunResult] = []

    for n, (pending, hold) in enumerate(GRID_COMBOS, 1):
        print(f"  [{n:2d}/{total}]  pending={pending:3d}  hold={hold:3d} …", end="  ", flush=True)
        r = await run_one(bars, pending, hold)
        results.append(r)
        pf_str = f"{r.profit_factor:.3f}" if r.profit_factor != float("inf") else "  inf"
        print(f"trades={r.trades:3d}  WR={r.win_rate:.1%}  PF={pf_str}  net=${r.net_pnl:+,.0f}  maxDD=${r.max_drawdown:,.0f}")

    # ── Ranked results ─────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("RANKED BY PROFIT FACTOR  (descending)")
    print("=" * 75)
    header = f"  {'PENDING':>7}  {'HOLD':>4}  {'TRADES':>6}  {'WIN%':>5}  {'PF':>6}  {'NET_PNL':>10}  {'MAX_DD':>9}"
    print(header)
    print("  " + "-" * 82)

    ranked = sorted(results, key=lambda r: r.profit_factor if r.profit_factor != float("inf") else 999, reverse=True)
    for r in ranked:
        pf_str = f"{r.profit_factor:6.3f}" if r.profit_factor != float("inf") else "   inf"
        print(f"  {r.pending_bars:>7}  {r.hold_bars:>4}  {r.trades:>6}  {r.win_rate:>5.1%}  {pf_str}  ${r.net_pnl:>+10,.0f}  ${r.max_drawdown:>8,.0f}")

    # ── Phase summaries ────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("PHASE 1 — hold sensitivity  (pending=180, hold varies)")
    print("=" * 75)
    p1 = [r for r in results if r.pending_bars == 180]
    for r in sorted(p1, key=lambda x: x.hold_bars):
        pf_str = f"{r.profit_factor:6.3f}" if r.profit_factor != float("inf") else "   inf"
        print(f"  hold={r.hold_bars:>3}  trades={r.trades:3d}  WR={r.win_rate:.1%}  PF={pf_str}  net=${r.net_pnl:>+9,.0f}  maxDD=${r.max_drawdown:>8,.0f}")

    print()
    print("=" * 75)
    print("PHASE 2 — pending sensitivity  (hold=60, pending varies)")
    print("=" * 75)
    p2 = [r for r in results if r.hold_bars == 60]
    for r in sorted(p2, key=lambda x: x.pending_bars):
        pf_str = f"{r.profit_factor:6.3f}" if r.profit_factor != float("inf") else "   inf"
        print(f"  pending={r.pending_bars:>3}  trades={r.trades:3d}  WR={r.win_rate:.1%}  PF={pf_str}  net=${r.net_pnl:>+9,.0f}  maxDD=${r.max_drawdown:>8,.0f}")

    # ── Best configuration ─────────────────────────────────────────────────────
    best = ranked[0]
    print()
    print("=" * 75)
    print(f"BEST CONFIG:  MAX_PENDING_BARS={best.pending_bars}  MAX_HOLD_BARS={best.hold_bars}")
    print(f"              PF={best.profit_factor:.3f}  WR={best.win_rate:.1%}  "
          f"trades={best.trades}  net=${best.net_pnl:+,.0f}  maxDD=${best.max_drawdown:,.0f}")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())

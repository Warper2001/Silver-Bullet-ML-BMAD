"""
Oct/Nov Seasonal Edge Check — 2023, 2024, 2025

Feeds Sep as warmup (to build vol regime history and H1 structure),
then evaluates Oct and Nov trades only. Uses the current live strategy config.

Usage:
  .venv/bin/python backtest_octnov_seasonal.py
"""
import asyncio
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytz

sys.path.insert(0, str(Path(__file__).parent))
import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

logging.disable(logging.CRITICAL)

ET_TZ = pytz.timezone("US/Eastern")

DATASETS = [
    {
        "year": 2023,
        "csv":  Path("data/processed/dollar_bars/1_minute/mnq_1min_2023_sepnov.csv"),
        "warmup_end": datetime(2023, 9, 30, 23, 59, 59, tzinfo=timezone.utc),
    },
    {
        "year": 2024,
        "csv":  Path("data/processed/dollar_bars/1_minute/mnq_1min_2024_sepnov.csv"),
        "warmup_end": datetime(2024, 9, 30, 23, 59, 59, tzinfo=timezone.utc),
    },
    {
        "year": 2025,
        "csv":  Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"),
        "warmup_end": datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc),
    },
]


def load_bars(csv_path: Path) -> list[DollarBar]:
    bars = []
    with open(csv_path) as f:
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


def profit_factor(pnl: list[float]) -> str:
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    if gl == 0:
        return "  inf"
    return f"{gp/gl:5.3f}"


async def run_year(ds: dict) -> dict:
    bars = load_bars(ds["csv"])
    warmup_end = ds["warmup_end"]

    # Filter to Sep 1 → Nov 30 for 2025 (full CSV is the whole year)
    if ds["year"] == 2025:
        bars = [b for b in bars if b.timestamp >= datetime(2025, 9, 1, tzinfo=timezone.utc)
                and b.timestamp <= datetime(2025, 11, 30, 23, 59, 59, tzinfo=timezone.utc)]

    trader = Tier2StreamingTrader()

    async def _no_bracket(*a, **kw): return (None, None, None)
    async def _no_cancel(*a, **kw): return True
    async def _no_close(*a, **kw): return None

    trader._submit_bracket_order = _no_bracket
    trader._cancel_sim_order     = _no_cancel
    trader._submit_close_order   = _no_close
    trader.ml_filter._log_decision = lambda *a, **kw: None

    last_h1_ts = None

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

        h1_ts = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        if h1_ts != last_h1_ts:
            trader._update_h1_structure()
            last_h1_ts = h1_ts

        await trader._advance_active_trade(bar)

        # Only enter trades after warmup (Oct 1 onward)
        if bar.timestamp > warmup_end:
            await trader._detect_and_enter(bar, is_backfill=False)

    return trader.completed_trades


async def main():
    print(f"Oct/Nov Seasonal Edge Check")
    print(f"Config: SL={tier2_mod.SL_MULTIPLIER}x  TP={tier2_mod.TP_MULTIPLIER}x  "
          f"PENDING={tier2_mod.MAX_PENDING_BARS}  HOLD={tier2_mod.MAX_HOLD_BARS}")
    print(f"Vol gate: ATR>{tier2_mod.VOL_REGIME_THRESHOLD:.0%}pct  "
          f"MinGap/ATR={tier2_mod.MIN_GAP_ATR_RATIO}")
    print("=" * 70)

    all_results = {}

    for ds in DATASETS:
        year = ds["year"]
        print(f"\nRunning {year} (Sep warmup → Oct/Nov evaluation) …", flush=True)
        trades = await run_year(ds)
        print(f"  {len(trades)} trades completed")
        all_results[year] = trades

    # ── Per-year, per-month report ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS — Oct and Nov by year")
    print(f"{'=' * 70}")
    print(f"  {'YEAR-MONTH':>10}  {'TRADES':>6}  {'WIN%':>5}  {'PF':>6}  {'NET_PNL':>10}  {'AVG_WIN':>8}  {'AVG_LOSS':>9}")
    print(f"  {'-' * 65}")

    oct_nov_summary: dict[int, dict] = {}

    for year in [2023, 2024, 2025]:
        trades = all_results[year]
        by_month: dict[str, list] = defaultdict(list)
        for t in trades:
            m = t.entry_time.astimezone(ET_TZ).strftime("%Y-%m")
            by_month[m].append(t)

        year_oct_nov = []
        for month in sorted(by_month):
            mo = int(month.split("-")[1])
            if mo not in (10, 11):
                continue
            mp = [t.pnl for t in by_month[month]]
            year_oct_nov.extend(mp)
            wins = [p for p in mp if p > 0]
            losses = [p for p in mp if p < 0]
            wr = len(wins) / len(mp) if mp else 0
            pf_str = profit_factor(mp)
            net = sum(mp)
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            print(f"  {month:>10}  {len(mp):>6}  {wr:>5.1%}  {pf_str}  ${net:>+9,.0f}  ${avg_w:>+7,.0f}  ${avg_l:>+8,.0f}")

        oct_nov_summary[year] = year_oct_nov

    # ── Combined Oct/Nov summary per year ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("OCT+NOV COMBINED SUMMARY BY YEAR")
    print(f"{'=' * 70}")
    print(f"  {'YEAR':>6}  {'TRADES':>6}  {'WIN%':>5}  {'PF':>6}  {'NET_PNL':>10}  {'AVG_TRADE':>10}")
    print(f"  {'-' * 55}")

    for year in [2023, 2024, 2025]:
        mp = oct_nov_summary[year]
        if not mp:
            print(f"  {year:>6}  {'—':>6}")
            continue
        wins = [p for p in mp if p > 0]
        losses = [p for p in mp if p < 0]
        wr = len(wins) / len(mp)
        pf_str = profit_factor(mp)
        net = sum(mp)
        avg = np.mean(mp)
        print(f"  {year:>6}  {len(mp):>6}  {wr:>5.1%}  {pf_str}  ${net:>+9,.0f}  ${avg:>+9,.2f}")

    print(f"\n  Reference — full 2025 year: 191 trades, 50.3% WR, PF=1.033, net=+$1,768")
    print(f"  Reference — Oct 2025 alone: 25 trades, 64.0% WR, PF=2.300, net=+$4,890")
    print(f"  Reference — Nov 2025 alone: 12 trades, 58.3% WR, PF=2.120, net=+$2,674")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())

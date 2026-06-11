"""study_orb_reversion_rate.py — SORM Gate 0: ORB Extension Reversion Rate Study.

Answers the load-bearing question BEFORE the full backtest:
    What % of 09:30–09:44 ET opening-range extensions (≥ 0.5 × ORB_size)
    revert to the ORB midpoint before 11:30 ET?

Gate 0 thresholds:
    ≥ 55%  → proceed to full backtest
    50–54% → marginal, reconsider parameters
    < 50%  → stop, no live code

No entry filters are applied here (no RSI, no stop-size check).
We measure the raw reversion tendency of the market.

In-sample data: 2025-01-01 → 2026-02-28 UTC
Pre-registration commit: ffb60f5

Usage:
    .venv/bin/python study_orb_reversion_rate.py
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.research.sorm_core import (
    SORMConfig,
    build_opening_range,
    check_reversion_to_mid,
    detect_extension,
    load_bars_et,
)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = SORMConfig()
UTC = timezone.utc

# In-sample range: 2025-01-01 → 2026-02-28 (DO NOT touch holdout ≥ 2026-03-01)
START_UTC = datetime(2025, 1, 1, tzinfo=UTC)
END_UTC = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")


def main() -> None:
    print("=" * 70)
    print("SORM Gate 0: ORB Extension Reversion Rate Study")
    print(f"In-sample: {START_UTC.date()} → {END_UTC.date()}")
    print("=" * 70)

    # ── Load bars ─────────────────────────────────────────────────────────────
    print("\nLoading 1-min bars…", end=" ", flush=True)
    df = load_bars_et([CSV_2025, CSV_2026], START_UTC, END_UTC)
    if df.empty:
        print("ERROR: no bars loaded — check CSV paths")
        sys.exit(1)
    print(f"{len(df):,} bars loaded ({df.index[0].date()} → {df.index[-1].date()})")

    # ── Session iteration ─────────────────────────────────────────────────────
    # Group by ET date
    df["_date"] = df.index.date
    sessions = df.groupby("_date")

    counters = {
        "sessions_total": 0,
        "sessions_with_orb": 0,
        "sessions_with_extension": 0,
        "extensions_reverted": 0,
        "bearish_extensions": 0,
        "bearish_reverted": 0,
        "bullish_extensions": 0,
        "bullish_reverted": 0,
    }

    monthly: dict[str, dict[str, int]] = defaultdict(lambda: {"ext": 0, "rev": 0})

    for date_et, sess_df in sessions:
        sess_df = sess_df.drop(columns=["_date"])
        counters["sessions_total"] += 1

        # Skip weekends and very sparse sessions
        if date_et.weekday() >= 5:
            continue

        # Build ORB
        orb = build_opening_range(sess_df, CFG)
        if orb is None:
            continue
        counters["sessions_with_orb"] += 1

        # Detect extension
        ext = detect_extension(sess_df, orb, CFG)
        if ext is None:
            continue
        counters["sessions_with_extension"] += 1

        month_key = f"{date_et.year}-{date_et.month:02d}"
        monthly[month_key]["ext"] += 1

        if ext.direction.value == "BEARISH":
            counters["bearish_extensions"] += 1
        else:
            counters["bullish_extensions"] += 1

        # Post-extension bars until 11:30 ET (after detection bar)
        hard_close_str = CFG.hard_close_et.strftime("%H:%M")
        ext_ts = ext.detection_bar_ts
        post_df = sess_df.loc[sess_df.index > ext_ts]
        post_df = post_df.between_time("00:00", hard_close_str, inclusive="left")

        # Check reversion to mid
        reverted = check_reversion_to_mid(post_df, orb, ext)
        if reverted:
            counters["extensions_reverted"] += 1
            monthly[month_key]["rev"] += 1
            if ext.direction.value == "BEARISH":
                counters["bearish_reverted"] += 1
            else:
                counters["bullish_reverted"] += 1

    # ── Report ────────────────────────────────────────────────────────────────
    n_ext = counters["sessions_with_extension"]
    n_rev = counters["extensions_reverted"]
    rate = n_rev / n_ext if n_ext > 0 else 0.0

    print()
    print("─" * 70)
    print("RESULTS SUMMARY")
    print("─" * 70)
    print(f"  Sessions total:              {counters['sessions_total']:>5}")
    print(f"  Sessions with valid ORB:     {counters['sessions_with_orb']:>5}")
    print(f"  Sessions with extension:     {n_ext:>5}")
    print(f"  Extensions that reverted:    {n_rev:>5}")
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  REVERSION RATE: {rate*100:.1f}%  ({n_rev}/{n_ext})               │")
    print(f"  └─────────────────────────────────────────────┘")
    print()

    # Direction breakdown
    n_bear = counters["bearish_extensions"]
    n_bull = counters["bullish_extensions"]
    n_bear_rev = counters["bearish_reverted"]
    n_bull_rev = counters["bullish_reverted"]

    if n_bear > 0:
        print(f"  Bearish extensions:  {n_bear_rev}/{n_bear} = {n_bear_rev/n_bear*100:.1f}%  reverted to mid")
    if n_bull > 0:
        print(f"  Bullish extensions:  {n_bull_rev}/{n_bull} = {n_bull_rev/n_bull*100:.1f}%  reverted to mid")

    # Gate verdict
    print()
    print("─" * 70)
    if rate >= 0.55:
        verdict = f"✅ GATE 0 PASS ({rate*100:.1f}% ≥ 55%)  — proceed to full backtest"
    elif rate >= 0.50:
        verdict = f"⚠️  GATE 0 MARGINAL ({rate*100:.1f}%, 50–55%) — reconsider parameters"
    else:
        verdict = f"🔴 GATE 0 FAIL ({rate*100:.1f}% < 50%)  — stop, do not build live system"
    print(f"  {verdict}")
    print("─" * 70)

    # Monthly breakdown
    print()
    print("BY-MONTH BREAKDOWN")
    print(f"  {'Month':<10}  {'Extensions':>10}  {'Reverted':>9}  {'Rate':>7}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*7}")
    for ym in sorted(monthly):
        m = monthly[ym]
        r = m["rev"] / m["ext"] if m["ext"] > 0 else 0.0
        print(f"  {ym:<10}  {m['ext']:>10}  {m['rev']:>9}  {r*100:>6.1f}%")

    print()
    print("=" * 70)
    print(f"Pre-registration commit: ffb60f5")
    print(f"In-sample data: {START_UTC.date()} → {END_UTC.date()} (holdout ≥ 2026-03-01 sealed)")
    print("=" * 70)


if __name__ == "__main__":
    main()

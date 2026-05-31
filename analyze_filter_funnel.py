#!/usr/bin/env python3
"""
Quick filter funnel report from tier2_bar_decisions.csv.
Shows exactly which filter is blocking trades and at what rate.

Usage: .venv/bin/python analyze_filter_funnel.py [--days N]
"""
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=7, help="Look back N days (default 7)")
args = parser.parse_args()

log = Path("logs/tier2_bar_decisions.csv")
if not log.exists():
    print("No tier2_bar_decisions.csv found.")
    raise SystemExit(1)

df = pd.read_csv(log, parse_dates=["bar_timestamp"])
df["bar_timestamp"] = pd.to_datetime(df["bar_timestamp"], utc=True)

cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
df = df[df["bar_timestamp"] >= cutoff].copy()

if df.empty:
    print(f"No data in the last {args.days} days.")
    raise SystemExit(0)

total = len(df)
print(f"\n=== S25 Filter Funnel — last {args.days} days ({total:,} bars) ===\n")

counts = df["action"].value_counts()
print("Action breakdown:")
for action, n in counts.items():
    pct = 100 * n / total
    bar = "█" * int(pct / 2)
    print(f"  {action:<25} {n:>6,}  ({pct:5.1f}%)  {bar}")

# Funnel: how many bars pass each gate
print("\n--- Funnel (bars reaching each stage) ---")
held = (df["action"] == "HOLD").sum()
tuesday = (df["action"] == "SKIP:TUESDAY").sum()
circuit = (df["action"] == "SKIP:CIRCUIT_BREAKER").sum()
seasonal = (df["action"] == "SKIP:SEASONAL").sum()
vol = (df["action"] == "SKIP:VOL_REGIME").sum()
no_sweep = (df["action"] == "SKIP:NO_SWEEP").sum()
no_choch = (df["action"] == "SKIP:NO_CHOCH").sum()
no_fvg = (df["action"] == "SKIP:NO_FVG").sum()
lr = (df["action"] == "SKIP:LR_REGIME").sum()
ml = (df["action"] == "SKIP:ML_FILTER").sum()
entered = (df["action"] == "ENTER").sum()

# Bars that were live candidates (not HOLD)
live = total - held
print(f"  Total bars processed:    {total:>6,}")
print(f"  Active trade (HOLD):     {held:>6,}  ({100*held/total:.1f}%)")
print(f"  Live candidate bars:     {live:>6,}")
if live > 0:
    print(f"  → Blocked: Tuesday:      {tuesday:>6,}  ({100*tuesday/live:.1f}% of live)")
    print(f"  → Blocked: Circuit:      {circuit:>6,}  ({100*circuit/live:.1f}% of live)")
    print(f"  → Blocked: Seasonal:     {seasonal:>6,}  ({100*seasonal/live:.1f}% of live)")
    print(f"  → Blocked: Vol regime:   {vol:>6,}  ({100*vol/live:.1f}% of live)")
    remaining = live - tuesday - circuit - seasonal - vol
    if remaining > 0:
        print(f"  Passed vol gate:         {remaining:>6,}")
        print(f"  → No H1 sweep:           {no_sweep:>6,}  ({100*no_sweep/remaining:.1f}% of remaining)")
        print(f"  → H1 sweep, no CHoCH:    {no_choch:>6,}  ({100*no_choch/remaining:.1f}% of remaining)")
        print(f"  → H1+CHoCH, no FVG:      {no_fvg:>6,}  ({100*no_fvg/remaining:.1f}% of remaining)")
        print(f"  → FVG hit, LR blocked:   {lr:>6,}  ({100*lr/max(1,remaining):.1f}% of remaining)")
        print(f"  → FVG hit, ML blocked:   {ml:>6,}  ({100*ml/max(1,remaining):.1f}% of remaining)")
        print(f"  → ENTERED:               {entered:>6,}  ({100*entered/remaining:.1f}% of remaining)")

print(f"\nPrimary bottleneck: ", end="")
skip_counts = {
    "NO_SWEEP": no_sweep, "NO_CHOCH": no_choch, "VOL_REGIME": vol,
    "NO_FVG": no_fvg, "TUESDAY": tuesday, "LR_REGIME": lr, "ML_FILTER": ml,
}
if skip_counts:
    top = max(skip_counts, key=skip_counts.get)
    print(f"SKIP:{top} ({skip_counts[top]:,} bars)")
else:
    print("insufficient data")

print()

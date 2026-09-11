#!/usr/bin/env python3
"""
Filter funnel report from logs/tier2_bar_decisions.csv.
Shows exactly which gate is blocking trades, and at what rate.

Rewritten 2026-09-11. The previous version grouped on the `action` column, matching
strings like "SKIP:TUESDAY". Only tier2_streaming_working.py ever wrote those suffixes —
YANK wrote a bare "SKIP" — so every counter in this report read ZERO for the one bot
that was actually live. That is a large part of why diagnosing YANK's 24-day silence in
September 2026 needed a 275 MB text log instead of this tool.

It now groups on the `rejection_reason` column (see src/research/decision_log.REASONS),
which all three writers populate, and can filter by `trader_id`, which they now stamp.

Usage:
    .venv/bin/python analyze_filter_funnel.py [--days N] [--trader trader-yank]
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.research.decision_log import REASONS  # noqa: E402

# Human labels, in gate-chain order. Keys are decision_log.REASONS.
STAGES = [
    ("data_stale",          "Feed stale"),
    ("warmup",              "Warm-up (<20 bars)"),
    ("flatten_window",      "Topstep flatten window"),
    ("tuesday",             "Tuesday exclusion"),
    ("daily_breaker",       "Daily loss breaker"),
    ("seasonality",         "Seasonality block"),
    ("vol_regime",          "Volatility regime"),
    ("no_sweep_or_choch",   "No H1 sweep / no M15 CHoCH"),
    ("no_fvg",              "Sweep+CHoCH, no FVG"),
    ("fvg_wrong_direction", "FVG wrong direction (near-miss)"),
    ("lr_regime",           "FVG hit, LR regime blocked"),
    ("ml_threshold",        "FVG hit, ML threshold blocked"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=7, help="Look back N days (default 7)")
parser.add_argument("--trader", default=None,
                    help="Only this trader_id (e.g. trader-yank). Default: all, "
                         "with a per-trader breakdown.")
args = parser.parse_args()

log = Path(__file__).resolve().parent / "logs" / "tier2_bar_decisions.csv"
if not log.exists():
    print("No logs/tier2_bar_decisions.csv found.")
    raise SystemExit(1)

df = pd.read_csv(log)
if "rejection_reason" not in df.columns:
    print("This file predates the 2026-09-11 schema (no rejection_reason column).\n"
          "Older archives are under logs/archive/. Nothing to report from it here.")
    raise SystemExit(1)

df["bar_timestamp"] = pd.to_datetime(df["bar_timestamp"], format="ISO8601", utc=True,
                                     errors="coerce")
df = df.dropna(subset=["bar_timestamp"])
df["rejection_reason"] = df["rejection_reason"].fillna("")

cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
df = df[df["bar_timestamp"] >= cutoff].copy()

if args.trader:
    df = df[df["trader_id"] == args.trader].copy()

if df.empty:
    who = f" for {args.trader}" if args.trader else ""
    print(f"No data in the last {args.days} days{who}.")
    raise SystemExit(0)

total = len(df)
scope = args.trader or "all traders"
span = f"{df['bar_timestamp'].min():%Y-%m-%d %H:%M} → {df['bar_timestamp'].max():%Y-%m-%d %H:%M} UTC"
print(f"\n=== Filter funnel — last {args.days}d, {scope} ({total:,} bars) ===")
print(f"    {span}\n")

if not args.trader and "trader_id" in df.columns:
    print("Rows by trader:")
    for tid, n in df["trader_id"].value_counts().items():
        print(f"  {str(tid):<22} {n:>7,}")
    print()

held = int((df["action"] == "HOLD").sum())
entered = int((df["action"] == "ENTER").sum())
live = total - held

print(f"  Total bars processed:    {total:>7,}")
print(f"  Active trade (HOLD):     {held:>7,}  ({100*held/total:.1f}%)")
print(f"  Live candidate bars:     {live:>7,}")
print()

counts = df["rejection_reason"].value_counts()
print("--- Rejections by gate (chain order) ---")
for key, label in STAGES:
    n = int(counts.get(key, 0))
    pct = 100 * n / live if live else 0.0
    bar = "█" * int(pct / 2)
    print(f"  {label:<34} {n:>7,}  ({pct:5.1f}% of live)  {bar}")
print(f"  {'ENTERED':<34} {entered:>7,}  ({100*entered/live if live else 0:5.1f}% of live)")

unknown = counts[~counts.index.isin(list(REASONS) + [""])]
if len(unknown):
    print("\n  ⚠️ reason codes not in decision_log.REASONS (writer/reader drift):")
    for k, n in unknown.items():
        print(f"     {k!r}: {n:,}")
blank = int(counts.get("", 0))
if blank:
    print(f"\n  note: {blank:,} rows have a blank reason — a writer that has not yet "
          f"been given the full vocabulary (btc_combine_streaming writes coarse rows).")

# Volatility-regime percentile: the value behind the block, previously discarded.
if "vol_regime_pct" in df.columns:
    pct_col = pd.to_numeric(df["vol_regime_pct"], errors="coerce").dropna()
    if len(pct_col):
        blocked = df["rejection_reason"] == "vol_regime"
        print(f"\n--- Volatility regime percentile ---")
        print(f"  median {pct_col.median():.3f} | p90 {pct_col.quantile(0.9):.3f} "
              f"| max {pct_col.max():.3f}   (gate fires above the configured threshold)")
        if blocked.any():
            b = pd.to_numeric(df.loc[blocked, "vol_regime_pct"], errors="coerce").dropna()
            if len(b):
                print(f"  on blocked bars: median {b.median():.3f} | min {b.min():.3f}")

ranked = [(k, int(counts.get(k, 0))) for k, _ in STAGES]
ranked.sort(key=lambda kv: kv[1], reverse=True)
top_key, top_n = ranked[0]
top_label = dict(STAGES)[top_key]
print(f"\nPrimary bottleneck: {top_label} ({top_n:,} bars)")
if entered == 0:
    print("No entries in this window. That is the expected reading when the funnel "
          "narrows to a handful of candidates — check the last two gates above before "
          "concluding the bot is broken.")
print()

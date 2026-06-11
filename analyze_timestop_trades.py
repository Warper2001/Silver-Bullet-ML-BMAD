"""
Time-Stop Trade Cluster Analysis

Loads the most recent 1-year backtest trade CSV and compares the feature profile
of time-stop exits vs TP/SL exits. Goal: find 2-3 features that reliably
distinguish time-stop trades (no-resolution entries) from resolved trades.

Usage:
  .venv/bin/python analyze_timestop_trades.py
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

import pytz

ET_TZ = pytz.timezone("US/Eastern")

# Use the most recent backtest CSV
REPORTS_DIR = Path("data/reports")


def load_trades(csv_path: Path) -> list[dict]:
    trades = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            et = datetime.fromisoformat(row["entry_time"]).astimezone(ET_TZ)
            trades.append({
                "entry_time":  datetime.fromisoformat(row["entry_time"]),
                "exit_time":   datetime.fromisoformat(row["exit_time"]),
                "direction":   row["direction"],
                "entry_price": float(row["entry_price"]),
                "exit_price":  float(row["exit_price"]),
                "exit_type":   row["exit_type"],
                "bars_held":   int(row["bars_held"]),
                "pnl":         float(row["pnl"]),
                "month":       et.strftime("%Y-%m"),
                "hour_et":     et.hour,
                "weekday":     et.strftime("%A"),
                "gap_pts":     abs(float(row["exit_price"]) - float(row["entry_price"])),
            })
    return trades


def stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"mean={mean(values):.2f}  med={median(values):.2f}  sd={stdev(values):.2f}" if len(values) > 1 else f"mean={mean(values):.2f}"


def pct_dist(items: list, key_fn) -> list[tuple]:
    counts: dict = defaultdict(int)
    for x in items:
        counts[key_fn(x)] += 1
    total = len(items)
    return sorted(counts.items(), key=lambda kv: -kv[1])


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def compare_groups(by_type: dict[str, list[dict]], feature_fn, label: str, top_n: int = 6):
    print(f"\n  {label}")
    print(f"  {'-' * 60}")
    for exit_type in ["time", "tp", "sl"]:
        group = by_type.get(exit_type, [])
        if not group:
            continue
        dist = pct_dist(group, feature_fn)
        total = len(group)
        top = dist[:top_n]
        parts = [f"{k}: {v} ({v/total:.0%})" for k, v in top]
        print(f"  {exit_type.upper():>4} ({total:3d}): {' | '.join(parts)}")


def main():
    # Accept explicit path or find most recent
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csvs = sorted(REPORTS_DIR.glob("backtest_1year_*.csv"), reverse=True)
        if not csvs:
            print("No backtest CSV found in data/reports/ — run backtest_tier2_1year_validation.py first")
            sys.exit(1)
        csv_path = csvs[0]
    print(f"Analyzing: {csv_path.name}")
    trades = load_trades(csv_path)
    print(f"Total trades: {len(trades)}")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_type[t["exit_type"]].append(t)

    for et, grp in sorted(by_type.items()):
        print(f"  {et.upper():>4}: {len(grp)} trades ({len(grp)/len(trades):.1%})")

    # ── Monthly distribution ──────────────────────────────────────────────────
    print_section("MONTHLY DISTRIBUTION")
    compare_groups(by_type, lambda t: t["month"], "Month (YYYY-MM)", top_n=13)

    # ── Hour of day (ET) ─────────────────────────────────────────────────────
    print_section("HOUR-OF-DAY DISTRIBUTION (ET)")
    compare_groups(by_type, lambda t: f"{t['hour_et']:02d}:xx", "Hour (ET)", top_n=10)

    # ── Day of week ───────────────────────────────────────────────────────────
    print_section("DAY-OF-WEEK DISTRIBUTION")
    compare_groups(by_type, lambda t: t["weekday"], "Weekday", top_n=6)

    # ── Bars held ────────────────────────────────────────────────────────────
    print_section("BARS HELD")
    print(f"\n  {'Exit':>4}  {'Count':>5}  {'Mean':>6}  {'Median':>6}  {'Min':>4}  {'Max':>4}")
    print(f"  {'-' * 40}")
    for et in ["time", "tp", "sl"]:
        grp = by_type.get(et, [])
        if not grp:
            continue
        vals = [t["bars_held"] for t in grp]
        print(f"  {et.upper():>4}  {len(grp):>5}  {mean(vals):>6.1f}  {median(vals):>6.1f}  {min(vals):>4}  {max(vals):>4}")

    # ── P&L distribution ─────────────────────────────────────────────────────
    print_section("P&L DISTRIBUTION")
    print(f"\n  {'Exit':>4}  {'Count':>5}  {'Mean $':>8}  {'Median $':>9}  {'Min $':>9}  {'Max $':>8}")
    print(f"  {'-' * 55}")
    for et in ["time", "tp", "sl"]:
        grp = by_type.get(et, [])
        if not grp:
            continue
        vals = [t["pnl"] for t in grp]
        print(f"  {et.upper():>4}  {len(grp):>5}  {mean(vals):>+8.2f}  {median(vals):>+9.2f}  {min(vals):>+9.2f}  {max(vals):>+8.2f}")

    # ── Time-stop P&L detail: how many are wins vs losses ────────────────────
    print_section("TIME-STOP DETAILED BREAKDOWN")
    time_trades = by_type.get("time", [])
    if time_trades:
        time_wins = [t for t in time_trades if t["pnl"] > 0]
        time_losses = [t for t in time_trades if t["pnl"] < 0]
        time_flat = [t for t in time_trades if t["pnl"] == 0]
        print(f"\n  Time-stop wins   : {len(time_wins):>3}  ({len(time_wins)/len(time_trades):.1%})  avg ${mean(t['pnl'] for t in time_wins):+.2f}" if time_wins else "\n  Time-stop wins   : 0")
        print(f"  Time-stop losses : {len(time_losses):>3}  ({len(time_losses)/len(time_trades):.1%})  avg ${mean(t['pnl'] for t in time_losses):+.2f}" if time_losses else "  Time-stop losses : 0")
        print(f"  Time-stop flat   : {len(time_flat):>3}")

        # Monthly breakdown of time-stops
        print(f"\n  Month-by-month time-stop count:")
        month_ts: dict[str, int] = defaultdict(int)
        month_all: dict[str, int] = defaultdict(int)
        for t in trades:
            month_all[t["month"]] += 1
        for t in time_trades:
            month_ts[t["month"]] += 1
        for month in sorted(month_all):
            ts_ct = month_ts.get(month, 0)
            all_ct = month_all[month]
            bar = "█" * ts_ct
            print(f"    {month}: {ts_ct:>2}/{all_ct:>2} ({ts_ct/all_ct:.0%}) {bar}")

    # ── Direction confirmation ────────────────────────────────────────────────
    print_section("DIRECTION BREAKDOWN (sanity check)")
    compare_groups(by_type, lambda t: t["direction"], "Direction", top_n=3)

    print(f"\n{'=' * 70}")
    print("SUMMARY — look for features where TIME differs most from TP/SL")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

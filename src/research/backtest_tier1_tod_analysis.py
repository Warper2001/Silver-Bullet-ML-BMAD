#!/usr/bin/env python3
"""TIER 1 — Time-of-Day Win Rate Analysis (Aug–Dec 2025)

Breaks down the 1,559 baseline trades by Eastern hour to find session segments
where win rate / profitability is strongest or weakest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config (matches validated paper trading setup) ──────────────────────── #
DATA_PATH              = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
START_DATE             = "2025-08-01"
END_DATE               = "2025-12-31"

SL_MULTIPLIER          = 2.5
ATR_THRESHOLD          = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS        = 50.0
MAX_HOLD_BARS          = 10
MNQ_CONTRACT_VALUE     = 5.0
TRANSACTION_COST       = 10.90

REPORT_PATH = Path("backtest_tier1_tod_analysis_report.txt")


def load_data() -> pd.DataFrame:
    print(f"Loading {START_DATE} → {END_DATE} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df[
        (df["timestamp"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(END_DATE + " 23:59", tz="UTC"))
    ]
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} bars")
    return df


def simulate_trade(fvg_direction, entry, tp, sl, df, start_idx):
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_idx + j
        if idx >= len(df):
            break
        h, l = highs[idx], lows[idx]
        if fvg_direction == "bullish":
            if l <= sl:
                ep = min(sl, l)
                return {"exit_type": "sl", "bars_held": j,
                        "pnl": (ep - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
            if h >= tp:
                return {"exit_type": "tp", "bars_held": j,
                        "pnl": (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
        else:
            if h >= sl:
                ep = max(sl, h)
                return {"exit_type": "sl", "bars_held": j,
                        "pnl": (entry - ep) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
            if l <= tp:
                return {"exit_type": "tp", "bars_held": j,
                        "pnl": (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
    idx = min(start_idx + MAX_HOLD_BARS, len(df) - 1)
    ep = closes[idx]
    pnl = ((ep - entry) if fvg_direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    return {"exit_type": "time", "bars_held": MAX_HOLD_BARS, "pnl": pnl}


def run_baseline(df: pd.DataFrame) -> list[dict]:
    print("Running baseline backtest ...")
    n = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values
    vols   = df["volume"].values
    timestamps = df["timestamp"]

    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_close),
                               np.abs(lows - prev_close)))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values

    is_bull = (closes > opens).astype(float)
    is_bear = (closes < opens).astype(float)
    up_vol = pd.Series(vols * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(vols * is_bear).rolling(20, min_periods=1).sum().values

    trades = []
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        bar_ts = timestamps.iloc[i]
        c1_close, c1_high = closes[i - 2], highs[i - 2]
        c3_open, c3_low, c3_high = opens[i], lows[i], highs[i]

        for direction in ("bullish", "bearish"):
            if direction == "bullish":
                if c1_close <= c3_open:
                    continue
                gap_top, gap_bottom = c1_high, c3_low
                entry = gap_bottom
                tp    = gap_top
                sl    = entry - (gap_top - gap_bottom) * SL_MULTIPLIER
            else:
                if c1_close >= c3_open:
                    continue
                gap_top    = c3_high
                gap_bottom = lows[i - 2]
                entry = gap_top
                tp    = gap_bottom
                sl    = entry + (gap_top - gap_bottom) * SL_MULTIPLIER

            if gap_top <= gap_bottom:
                continue
            gap_size = gap_top - gap_bottom
            if gap_size < atr[i] * ATR_THRESHOLD:
                continue
            if gap_size * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS:
                continue

            uv, dv = up_vol[i], dn_vol[i]
            ratio = (uv / dv if dv > 0 else float("inf")) if direction == "bullish" \
                    else (dv / uv if uv > 0 else float("inf"))
            if ratio < VOLUME_RATIO_THRESHOLD:
                continue

            result = simulate_trade(direction, entry, tp, sl, df, i)
            result["direction"] = direction
            result["bar_ts"]    = bar_ts
            result["pnl"]       = result["pnl"]
            trades.append(result)
            next_entry_bar = i + result["bars_held"] + 1
            break

    print(f"  {len(trades)} trades")
    return trades


def analyze_tod(trades: list[dict]) -> str:
    """Break down trades by Eastern hour (UTC-4 summer / UTC-5 winter — use UTC-4 proxy)."""
    rows = []
    for t in trades:
        # Convert UTC → Eastern (approximate: UTC-5 for Nov-Dec, UTC-4 Aug-Oct)
        ts = t["bar_ts"]
        month = ts.month
        offset = -4 if month <= 10 else -5
        et_hour = (ts.hour + offset) % 24
        rows.append({"hour_et": et_hour, "pnl": t["pnl"], "win": t["pnl"] > 0})

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("hour_et")
        .agg(
            trades=("pnl", "count"),
            wins=("win", "sum"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        )
        .reset_index()
    )
    summary["win_rate"] = summary["wins"] / summary["trades"] * 100
    summary = summary.sort_values("hour_et")

    lines = []
    lines.append("=" * 78)
    lines.append("TIER 1 — TIME-OF-DAY WIN RATE ANALYSIS (Aug–Dec 2025, Eastern Time)")
    lines.append("=" * 78)
    lines.append(f"{'Hour (ET)':<12} {'Trades':>7} {'WR%':>7} {'Total P&L':>12} {'Avg P&L':>10}  Session")
    lines.append("-" * 78)

    session_map = {
        8:  "Pre-market",  9:  "NY Open",   10: "NY Morning",
        11: "NY Morning", 12: "NY Midday",  13: "NY Midday",
        14: "NY Afternoon", 15: "NY Close", 16: "After Hours",
    }

    total_trades = 0
    total_wins   = 0
    total_pnl    = 0.0

    for _, row in summary.iterrows():
        h = int(row["hour_et"])
        session = session_map.get(h, "Off-hours")
        marker = " ★" if row["win_rate"] >= 80 else (" ✗" if row["win_rate"] < 70 else "")
        lines.append(
            f"{h:02d}:00–{h+1:02d}:00   "
            f"{int(row['trades']):>7}  "
            f"{row['win_rate']:>6.1f}%  "
            f"${row['total_pnl']:>10.2f}  "
            f"${row['avg_pnl']:>8.2f}"
            f"  {session}{marker}"
        )
        total_trades += row["trades"]
        total_wins   += row["wins"]
        total_pnl    += row["total_pnl"]

    lines.append("-" * 78)
    lines.append(
        f"{'TOTAL':<12} {total_trades:>7}  {total_wins/total_trades*100:>6.1f}%  "
        f"${total_pnl:>10.2f}"
    )
    lines.append("")
    lines.append("★ = hour WR ≥ 80%   ✗ = hour WR < 70%")
    lines.append("")

    # Direction breakdown
    dir_rows = []
    for t in trades:
        dir_rows.append({"direction": t["direction"], "pnl": t["pnl"], "win": t["pnl"] > 0})
    dir_df = pd.DataFrame(dir_rows)
    dir_summary = dir_df.groupby("direction").agg(
        trades=("pnl", "count"),
        wins=("win", "sum"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    dir_summary["win_rate"] = dir_summary["wins"] / dir_summary["trades"] * 100

    lines.append("DIRECTION BREAKDOWN:")
    lines.append(f"  {'Direction':<12} {'Trades':>7} {'WR%':>7} {'Total P&L':>12}")
    lines.append("  " + "-" * 44)
    for _, row in dir_summary.iterrows():
        lines.append(
            f"  {row['direction']:<12} {int(row['trades']):>7}  "
            f"{row['win_rate']:>6.1f}%  ${row['total_pnl']:>10.2f}"
        )

    # Best/worst consecutive hours
    lines.append("")
    wr_by_hour = {int(r["hour_et"]): r["win_rate"] for _, r in summary.iterrows()}
    best_hours = [(h, wr) for h, wr in wr_by_hour.items() if wr >= 78]
    worst_hours = [(h, wr) for h, wr in wr_by_hour.items() if wr < 72]
    if best_hours:
        lines.append("HIGH-EDGE HOURS (WR ≥ 78%): " +
                     ", ".join(f"{h:02d}:00 ({wr:.1f}%)" for h, wr in sorted(best_hours)))
    if worst_hours:
        lines.append("LOW-EDGE HOURS  (WR < 72%): " +
                     ", ".join(f"{h:02d}:00 ({wr:.1f}%)" for h, wr in sorted(worst_hours)))

    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_data()
    trades = run_baseline(df)
    report = analyze_tod(trades)
    print("\n" + report)
    REPORT_PATH.write_text(report)
    print(f"\nSaved → {REPORT_PATH}")

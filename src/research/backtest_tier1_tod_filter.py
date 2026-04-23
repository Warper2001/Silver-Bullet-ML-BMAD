#!/usr/bin/env python3
"""TIER 1 — Time-of-Day Filter Backtest (Aug–Dec 2025)

Systematically tests hour-exclusion sets derived from the TOD analysis.
Uses WR threshold cutoffs: block all ET hours whose historical WR falls
below the threshold. Also tests named session configs.

Baseline TOD summary (Eastern Time):
  Best:  09:00 92%, 15:00 90%, 03:00 87%, 04:00 87%, 10:00 88%
  Worst: 17:00  0%, 22:00 64%, 16:00 65%, 01:00 67%, 23:00 70%

The filter is applied at signal-detection time — blocked hours produce no
entry, and next_entry_bar does not advance (no "phantom hold" penalty).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──────────────────────────────────────────────────────────────────── #
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

MIN_TRADES_PER_DAY = 3.0

REPORT_PATH = Path("backtest_tier1_tod_filter_report.txt")

BASELINE = {
    "total_trades": 1559,
    "win_rate":     76.33,
    "profit_factor": 1.20,
    "avg_tpd":      14.77,
    "total_pnl":    3694.40,
}

# WR by ET hour from backtest_tier1_tod_analysis_report.txt
# {et_hour: (trades, win_rate_pct)}
HOUR_WR: dict[int, tuple[int, float]] = {
     0: (84,  75.0),  1: (81,  66.7),  2: (100, 80.0),  3: (71,  87.3),
     4: (96,  86.5),  5: (82,  76.8),  6: (75,  70.7),  7: (87,  78.2),
     8: (70,  72.9),  9: (25,  92.0), 10: (8,   87.5), 11: (28,  85.7),
    12: (41,  80.5), 13: (42,  78.6), 14: (32,  78.1), 15: (40,  90.0),
    16: (83,  65.1), 17: (2,    0.0), 18: (85,  78.8), 19: (84,  76.2),
    20: (98,  75.5), 21: (79,  78.5), 22: (77,  63.6), 23: (89,  69.7),
}


def build_configs() -> list[dict]:
    """Define all hour-filter configurations to test."""
    configs = []

    # WR-threshold cutoffs: block hours below threshold
    thresholds = [64, 67, 70, 72, 75, 77, 79, 81]
    for thr in thresholds:
        blocked = frozenset(h for h, (_, wr) in HOUR_WR.items() if wr < thr)
        admitted_trades = sum(n for h, (n, _) in HOUR_WR.items() if h not in blocked)
        configs.append({
            "label":   f"Block WR<{thr}%",
            "blocked": blocked,
            "expected_trades": admitted_trades,
        })

    # Named session configs
    configs.append({
        "label":   "NY Session (09–15 ET)",
        "blocked": frozenset(h for h in range(24) if h not in range(9, 16)),
        "expected_trades": sum(n for h, (n, _) in HOUR_WR.items() if 9 <= h <= 15),
    })
    configs.append({
        "label":   "No Late Night (22–01 ET)",
        "blocked": frozenset({22, 23, 0, 1}),
        "expected_trades": sum(n for h, (n, _) in HOUR_WR.items() if h not in {22, 23, 0, 1}),
    })
    configs.append({
        "label":   "No Post-Mkt (16–17 ET)",
        "blocked": frozenset({16, 17}),
        "expected_trades": sum(n for h, (n, _) in HOUR_WR.items() if h not in {16, 17}),
    })
    configs.append({
        "label":   "Top-Bad 4 (1,16,17,22)",
        "blocked": frozenset({1, 16, 17, 22}),
        "expected_trades": sum(n for h, (n, _) in HOUR_WR.items() if h not in {1, 16, 17, 22}),
    })

    return configs


# ── Data loading ─────────────────────────────────────────────────────────────── #

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


def precompute_et_hours(timestamps: np.ndarray) -> np.ndarray:
    """Vectorised UTC → Eastern hour (DST approx: UTC-4 for months ≤10, else UTC-5)."""
    dt  = pd.to_datetime(timestamps)
    off = np.where(dt.month <= 10, -4, -5)
    return (dt.hour + off) % 24


# ── Exit simulation ───────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, start_idx, n):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_idx + j
        if idx >= n:
            break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl:
                return {"exit_type": "sl", "bars_held": j,
                        "pnl": (min(sl, l) - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
            if h >= tp:
                return {"exit_type": "tp", "bars_held": j,
                        "pnl": (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
        else:
            if h >= sl:
                return {"exit_type": "sl", "bars_held": j,
                        "pnl": (entry - max(sl, h)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
            if l <= tp:
                return {"exit_type": "tp", "bars_held": j,
                        "pnl": (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
    ep = closes[min(start_idx + MAX_HOLD_BARS, n - 1)]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    return {"exit_type": "time", "bars_held": MAX_HOLD_BARS, "pnl": pnl}


# ── Backtest engine ───────────────────────────────────────────────────────────── #

def run_backtest(df: pd.DataFrame, et_hours: np.ndarray,
                 blocked_hours: frozenset) -> dict:
    n          = len(df)
    highs      = df["high"].values
    lows       = df["low"].values
    opens      = df["open"].values
    closes     = df["close"].values
    vols       = df["volume"].values

    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values

    is_bull = (closes > opens).astype(float)
    is_bear = (closes < opens).astype(float)
    up_vol = pd.Series(vols * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(vols * is_bear).rolling(20, min_periods=1).sum().values

    trades         = []
    tod_skipped    = 0
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        if et_hours[i] in blocked_hours:
            tod_skipped += 1
            continue

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

            result = simulate_trade(direction, entry, tp, sl, highs, lows, closes, i, n)
            result["direction"] = direction
            trades.append(result)
            next_entry_bar = i + result["bars_held"] + 1
            break

    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_tpd": 0.0, "total_pnl": 0.0, "filtered_pct": 100.0,
                "tod_skipped": tod_skipped}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    days   = (pd.Timestamp(df["timestamp"].iloc[-1]) -
               pd.Timestamp(df["timestamp"].iloc[0])).total_seconds() / 86400
    tdays  = days * (252 / 365)
    total  = len(trades)
    filt   = tod_skipped / (total + tod_skipped) * 100 if (total + tod_skipped) > 0 else 0.0

    return {
        "total_trades":  total,
        "win_rate":      len(wins) / total * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_tpd":       total / tdays if tdays > 0 else 0.0,
        "total_pnl":     sum(t["pnl"] for t in trades),
        "filtered_pct":  filt,
        "tod_skipped":   tod_skipped,
    }


# ── Report ────────────────────────────────────────────────────────────────────── #

def build_report(results: list[dict]) -> str:
    b = BASELINE

    lines = []
    lines.append("=" * 100)
    lines.append("TIER 1 — TIME-OF-DAY FILTER BACKTEST — AUG–DEC 2025")
    lines.append("=" * 100)
    lines.append(f"Base config: SL{SL_MULTIPLIER}x | ATR{ATR_THRESHOLD} | "
                 f"Vol{VOLUME_RATIO_THRESHOLD} | MaxGap${MAX_GAP_DOLLARS} | Hold{MAX_HOLD_BARS}")
    lines.append(f"Filter: skip signal if ET hour is in blocked set (no phantom hold penalty)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        f"BASELINE (no filter): {b['total_trades']} trades | WR {b['win_rate']:.2f}% | "
        f"PF {b['profit_factor']:.2f} | TPD {b['avg_tpd']:.2f} | P&L ${b['total_pnl']:.2f}"
    )
    lines.append("")
    lines.append(f"  {'Config':<28} {'Trades':>7}  {'WR%':>7}  {'ΔWR':>6}  "
                 f"{'PF':>5}  {'TPD':>6}  {'P&L':>11}  {'Filt%':>6}  {'ΔP&L':>10}")
    lines.append("-" * 100)

    star_count = 0
    for r in results:
        wr_delta  = r["win_rate"] - b["win_rate"]
        pnl_delta = r["total_pnl"] - b["total_pnl"]
        is_star   = (wr_delta >= 1.0) and (r["avg_tpd"] >= MIN_TRADES_PER_DAY)
        star      = "★" if is_star else " "
        if is_star:
            star_count += 1
        sign_wr  = "+" if wr_delta >= 0 else ""
        sign_pnl = "+" if pnl_delta >= 0 else ""
        lines.append(
            f" {star} {r['label']:<28} {r['total_trades']:>7}  "
            f"{r['win_rate']:>6.2f}%  {sign_wr}{wr_delta:>4.2f}pp  "
            f"{r['profit_factor']:>5.2f}  {r['avg_tpd']:>6.2f}  "
            f"${r['total_pnl']:>10.2f}  {r['filtered_pct']:>5.1f}%  "
            f"{sign_pnl}${abs(pnl_delta):>8.2f}"
        )

    lines.append("-" * 100)
    lines.append("")
    lines.append(f"★ = WR improves ≥1pp over baseline AND TPD ≥ {MIN_TRADES_PER_DAY}")
    lines.append(f"★ configs found: {star_count}")
    lines.append("")

    # Blocked-hours detail for each threshold config
    lines.append("BLOCKED HOURS BY CONFIG:")
    lines.append("")
    for r in results:
        blocked_sorted = sorted(r["blocked"])
        blocked_str    = ", ".join(f"{h:02d}:00" for h in blocked_sorted) if blocked_sorted else "none"
        lines.append(f"  {r['label']:<28}  blocked: {blocked_str}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("")

    # Verdict
    star_results = [r for r in results if (r["win_rate"] - b["win_rate"]) >= 1.0
                    and r["avg_tpd"] >= MIN_TRADES_PER_DAY]

    lines.append("VERDICT:")
    if star_results:
        # Best by WR among viable
        best = max(star_results, key=lambda x: x["win_rate"])
        lines.append(f"  DEPLOY filter: '{best['label']}'")
        lines.append(f"  WR: {b['win_rate']:.2f}% → {best['win_rate']:.2f}% "
                     f"(+{best['win_rate']-b['win_rate']:.2f}pp)")
        lines.append(f"  TPD: {b['avg_tpd']:.1f} → {best['avg_tpd']:.1f}  |  "
                     f"P&L: ${b['total_pnl']:.2f} → ${best['total_pnl']:.2f}")
        lines.append(f"  Blocked hours: {sorted(best['blocked'])}")
    else:
        # Find closest to threshold
        best_wr = max(results, key=lambda x: x["win_rate"])
        lines.append(f"  No config meets ★ criteria (WR +1pp AND TPD ≥ {MIN_TRADES_PER_DAY}).")
        lines.append(f"  Closest: '{best_wr['label']}' — "
                     f"WR {best_wr['win_rate']:.2f}% (+{best_wr['win_rate']-b['win_rate']:.2f}pp), "
                     f"TPD {best_wr['avg_tpd']:.2f}, P&L ${best_wr['total_pnl']:.2f}")
        lines.append(f"  Consider: best WR/TPD balance may still be worth deploying "
                     f"if P&L improves.")

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


if __name__ == "__main__":
    df       = load_data()
    et_hours = precompute_et_hours(df["timestamp"].values)

    configs = build_configs()
    print(f"\nRunning {len(configs)} TOD filter configurations ...")

    results = []
    for cfg in configs:
        r = run_backtest(df, et_hours, cfg["blocked"])
        r["label"]   = cfg["label"]
        r["blocked"] = cfg["blocked"]
        results.append(r)
        wr_delta = r["win_rate"] - BASELINE["win_rate"]
        sign     = "+" if wr_delta >= 0 else ""
        print(f"  {cfg['label']:<28} → {r['total_trades']:4d} trades  "
              f"{r['win_rate']:5.2f}% WR ({sign}{wr_delta:.2f}pp)  "
              f"PF {r['profit_factor']:.2f}  TPD {r['avg_tpd']:.1f}  "
              f"P&L ${r['total_pnl']:.2f}")

    report = build_report(results)
    print("\n" + report)
    REPORT_PATH.write_text(report)
    print(f"Saved → {REPORT_PATH}")

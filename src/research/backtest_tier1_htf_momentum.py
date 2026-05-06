#!/usr/bin/env python3
"""TIER 1 — HTF Momentum Alignment Grid Search (Aug–Dec 2025)

Unlike directional presence (does an FVG exist on the HTF?), tests whether the
HTF is actively TRENDING in the trade direction at signal time.

Momentum = N-bar close-over-close sign on the resampled parent TF.
  bullish signal → HTF N-bar return > 0
  bearish signal → HTF N-bar return < 0

Hypothesis: choppy off-hours produce flat/reversed HTF momentum → the filter
naturally screens bad-hour trades without knowing the clock time.

Grid: [5,21,89]-min × [3,5,10] bars × OR/AND = 27 combos.
Includes TOD variance analysis for top combos to test the hypothesis.
Baseline: 1,559 trades | 76.33% WR | PF 1.20 | TPD 14.77 | P&L $3,694.40
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

PARENT_TFS    = [5, 21, 89]
LOOKBACK_BARS = [3, 5, 10]
LOGIC_OPTIONS = ["OR", "AND"]

MIN_TRADES_PER_DAY = 3.0
STAR_WR_DELTA      = 1.0

REPORT_PATH = Path("backtest_tier1_htf_momentum_report.txt")

BASELINE = {
    "total_trades": 1559,
    "win_rate":     76.33,
    "profit_factor": 1.20,
    "avg_tpd":      14.77,
    "total_pnl":    3694.40,
}

# From backtest_tier1_tod_analysis_report.txt — used for comparison
BASELINE_TOD: dict[int, tuple[int, float]] = {
     0: (84,  75.0),  1: (81,  66.7),  2: (100, 80.0),  3: (71,  87.3),
     4: (96,  86.5),  5: (82,  76.8),  6: (75,  70.7),  7: (87,  78.2),
     8: (70,  72.9),  9: (25,  92.0), 10: (8,   87.5), 11: (28,  85.7),
    12: (41,  80.5), 13: (42,  78.6), 14: (32,  78.1), 15: (40,  90.0),
    16: (83,  65.1), 17: (2,    0.0), 18: (85,  78.8), 19: (84,  76.2),
    20: (98,  75.5), 21: (79,  78.5), 22: (77,  63.6), 23: (89,  69.7),
}


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


# ── HTF momentum ──────────────────────────────────────────────────────────────── #

def resample_htf(df: pd.DataFrame, tf_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample to tf_minutes bars; return (timestamps, closes) numpy arrays."""
    rs = (
        df.set_index("timestamp")
        .resample(f"{tf_minutes}min", closed="right", label="right")
        .agg({"close": "last"})
        .dropna()
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")
    print(f"  {tf_minutes:3d}-min: {len(rs)} HTF bars")
    return rs["timestamp"].values, rs["close"].values


def build_momentum_signal(closes: np.ndarray, lookback: int) -> np.ndarray:
    """+1 if close > close[i-lookback], -1 if lower, 0 if equal or insufficient history."""
    n = len(closes)
    signal = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        diff = closes[i] - closes[i - lookback]
        if diff > 0:
            signal[i] = 1
        elif diff < 0:
            signal[i] = -1
    return signal


def lookup_momentum(direction: str, bar_ts: np.datetime64,
                    htf_ts: np.ndarray, htf_signal: np.ndarray) -> bool:
    """True if the most recent HTF signal (at or before bar_ts) aligns with direction."""
    idx = int(np.searchsorted(htf_ts, bar_ts, side="right")) - 1
    if idx < 0:
        return False
    sig = int(htf_signal[idx])
    if sig == 0:
        return False
    return sig == (1 if direction == "bullish" else -1)


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

def run_backtest(df: pd.DataFrame,
                 signals: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]],
                 tfs: list[int], lookback: int, logic: str) -> dict:
    n          = len(df)
    highs      = df["high"].values
    lows       = df["low"].values
    opens      = df["open"].values
    closes     = df["close"].values
    vols       = df["volume"].values
    timestamps = df["timestamp"].values

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
    mtf_skipped    = 0
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        bar_ts = timestamps[i]
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

            checks = [
                lookup_momentum(direction, bar_ts,
                                signals[tf][lookback][0], signals[tf][lookback][1])
                for tf in tfs
            ]
            passes = any(checks) if logic == "OR" else all(checks)

            if not passes:
                mtf_skipped += 1
                continue

            result = simulate_trade(direction, entry, tp, sl, highs, lows, closes, i, n)
            result["direction"] = direction
            result["bar_ts"]    = bar_ts
            trades.append(result)
            next_entry_bar = i + result["bars_held"] + 1
            break

    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_tpd": 0.0, "total_pnl": 0.0, "filtered_pct": 100.0,
                "mtf_skipped": mtf_skipped, "trades": []}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    days   = (pd.Timestamp(df["timestamp"].iloc[-1]) -
               pd.Timestamp(df["timestamp"].iloc[0])).total_seconds() / 86400
    tdays  = days * (252 / 365)
    total  = len(trades)
    filt   = mtf_skipped / (total + mtf_skipped) * 100 if (total + mtf_skipped) > 0 else 0.0

    return {
        "total_trades":  total,
        "win_rate":      len(wins) / total * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_tpd":       total / tdays if tdays > 0 else 0.0,
        "total_pnl":     sum(t["pnl"] for t in trades),
        "filtered_pct":  filt,
        "mtf_skipped":   mtf_skipped,
        "trades":        trades,
    }


# ── Grid search ───────────────────────────────────────────────────────────────── #

def run_grid(df: pd.DataFrame) -> list[dict]:
    print("\nPre-computing HTF resamples ...")
    htf_raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tf in PARENT_TFS:
        htf_raw[tf] = resample_htf(df, tf)

    print("Building momentum signals ...")
    signals: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for tf in PARENT_TFS:
        signals[tf] = {}
        ts_arr, cl_arr = htf_raw[tf]
        for lb in LOOKBACK_BARS:
            sig = build_momentum_signal(cl_arr, lb)
            signals[tf][lb] = (ts_arr, sig)
            n_total = len(sig)
            up_pct  = np.sum(sig == 1) / n_total * 100
            dn_pct  = np.sum(sig == -1) / n_total * 100
            print(f"  {tf:3d}m LB{lb:2d}: {up_pct:.0f}% up / {dn_pct:.0f}% dn / "
                  f"{100 - up_pct - dn_pct:.0f}% flat")

    all_combos = []

    for tf in PARENT_TFS:
        for lb in LOOKBACK_BARS:
            all_combos.append({
                "tfs": [tf], "lookback": lb, "logic": "OR",
                "label": f"MOM{tf:3d}m LB{lb:2d} OR  "
            })

    for lb in LOOKBACK_BARS:
        for lg in LOGIC_OPTIONS:
            all_combos.append({
                "tfs": PARENT_TFS, "lookback": lb, "logic": lg,
                "label": f"MOM5+21+89m LB{lb:2d} {lg}"
            })

    for pair in [[5, 21], [21, 89]]:
        tf_str = "+".join(str(t) for t in pair)
        for lb in LOOKBACK_BARS:
            for lg in LOGIC_OPTIONS:
                all_combos.append({
                    "tfs": pair, "lookback": lb, "logic": lg,
                    "label": f"MOM{tf_str}m  LB{lb:2d} {lg} "
                })

    total = len(all_combos)
    print(f"\nRunning {total} combinations ...")
    results = []

    for k, combo in enumerate(all_combos):
        r = run_backtest(df, signals, combo["tfs"], combo["lookback"], combo["logic"])
        r["label"]    = combo["label"]
        r["tfs"]      = combo["tfs"]
        r["lookback"] = combo["lookback"]
        r["logic"]    = combo["logic"]
        results.append(r)
        print(f"  [{k+1:3d}/{total}] {combo['label']} → "
              f"{r['total_trades']:4d} trades  "
              f"{r['win_rate']:5.2f}% WR  "
              f"PF {r['profit_factor']:.2f}  "
              f"Filt {r['filtered_pct']:.1f}%")

    return results


# ── TOD analysis ─────────────────────────────────────────────────────────────── #

def tod_wr_by_hour(trades: list[dict]) -> dict[int, tuple[int, float]]:
    rows = []
    for t in trades:
        ts = pd.Timestamp(t["bar_ts"])
        offset  = -4 if ts.month <= 10 else -5
        et_hour = (ts.hour + offset) % 24
        rows.append({"hour": et_hour, "win": t["pnl"] > 0})
    if not rows:
        return {}
    dfr = pd.DataFrame(rows)
    return {
        int(h): (len(grp), float(grp["win"].mean() * 100))
        for h, grp in dfr.groupby("hour")
    }


def tod_std(tod: dict[int, tuple[int, float]], min_trades: int = 5) -> float:
    wrs = [wr for n, wr in tod.values() if n >= min_trades]
    return float(np.std(wrs)) if len(wrs) >= 2 else 0.0


# ── Report ────────────────────────────────────────────────────────────────────── #

def build_report(results: list[dict]) -> str:
    b       = BASELINE
    sorted_r = sorted(results, key=lambda x: x["win_rate"], reverse=True)

    lines = []
    lines.append("=" * 94)
    lines.append("TIER 1 HTF MOMENTUM ALIGNMENT GRID SEARCH — AUG–DEC 2025")
    lines.append("=" * 94)
    lines.append(f"Base config: SL{SL_MULTIPLIER}x | ATR{ATR_THRESHOLD} | "
                 f"Vol{VOLUME_RATIO_THRESHOLD} | MaxGap${MAX_GAP_DOLLARS} | Hold{MAX_HOLD_BARS}")
    lines.append(f"Filter:  N-bar close-over-close sign on resampled HTF")
    lines.append(f"Parent TFs: {PARENT_TFS} min  |  Lookbacks: {LOOKBACK_BARS} bars  |  Logic: OR / AND")
    lines.append("=" * 94)
    lines.append("")
    lines.append(
        f"BASELINE (no MTF): {b['total_trades']} trades | WR {b['win_rate']:.2f}% | "
        f"PF {b['profit_factor']:.2f} | TPD {b['avg_tpd']:.2f} | P&L ${b['total_pnl']:.2f}"
    )
    lines.append("")
    lines.append("TOP 20 (sorted by win rate ↓):")
    lines.append("")
    lines.append(f"  {'#':>3}  {'Label':<24} {'Trades':>7}  {'WR%':>7}  {'PF':>5}  "
                 f"{'TPD':>6}  {'P&L':>11}  {'Filt%':>6}")
    lines.append("-" * 94)

    star_count = 0
    for rank, r in enumerate(sorted_r[:20], 1):
        wr_delta = r["win_rate"] - b["win_rate"]
        is_star  = (wr_delta >= STAR_WR_DELTA) and (r["avg_tpd"] >= MIN_TRADES_PER_DAY)
        star     = "★" if is_star else " "
        if is_star:
            star_count += 1
        lines.append(
            f" {star} {rank:>3}  {r['label']:<24} {r['total_trades']:>7}  "
            f"{r['win_rate']:>6.2f}%  {r['profit_factor']:>5.2f}  "
            f"{r['avg_tpd']:>6.2f}  ${r['total_pnl']:>10.2f}  {r['filtered_pct']:>5.1f}%"
        )

    lines.append("-" * 94)
    lines.append("")
    lines.append(f"★ = WR improves >{STAR_WR_DELTA}% over baseline AND TPD ≥ {MIN_TRADES_PER_DAY}")
    lines.append(f"★ combos found: {star_count}")
    lines.append("")
    lines.append("BOTTOM 5 (worst win rate):")
    lines.append("")
    for r in sorted_r[-5:]:
        lines.append(
            f"     {r['label']:<24} {r['total_trades']:>7}  {r['win_rate']:>6.2f}%  "
            f"PF {r['profit_factor']:.2f}  Filt {r['filtered_pct']:.1f}%"
        )
    lines.append("")
    lines.append("=" * 94)

    # ── TOD variance section ──────────────────────────────────────────────────── #
    lines.append("")
    lines.append("TIME-OF-DAY VARIANCE — TOP 5 COMBOS vs BASELINE")
    lines.append("=" * 94)
    b_std = tod_std(BASELINE_TOD)
    lines.append(f"Baseline TOD WR std: {b_std:.1f}pp  (lower = more session-independent)")
    lines.append("")
    lines.append(f"  {'Label':<24}  {'Trades':>7}  {'WR%':>7}  {'TOD std':>9}  {'Δ std':>7}  {'Filt%':>6}")
    lines.append("  " + "-" * 68)

    for r in sorted_r[:5]:
        tod = tod_wr_by_hour(r["trades"])
        r_std = tod_std(tod)
        delta_std = r_std - b_std
        sign = "+" if delta_std >= 0 else ""
        lines.append(
            f"  {r['label']:<24}  {r['total_trades']:>7}  {r['win_rate']:>6.2f}%  "
            f"{r_std:>8.1f}pp  {sign}{delta_std:>5.1f}pp  {r['filtered_pct']:>5.1f}%"
        )

    lines.append("")

    # Full TOD breakdown for the top combo
    best     = sorted_r[0]
    best_tod = tod_wr_by_hour(best["trades"])
    b_std_best = tod_std(best_tod)

    lines.append(f"FULL TOD BREAKDOWN — Best combo: {best['label'].strip()}")
    lines.append(f"  ({best['total_trades']} trades | {best['win_rate']:.2f}% WR | "
                 f"TOD std {b_std_best:.1f}pp vs baseline {b_std:.1f}pp)")
    lines.append("")
    lines.append(f"  {'Hour (ET)':<12}  {'Baseline':>12}  {'Best Combo':>12}  "
                 f"{'Δ WR':>8}  {'Δ Trades':>9}")
    lines.append("  " + "-" * 60)

    for h in range(24):
        b_n, b_wr = BASELINE_TOD.get(h, (0, 0.0))
        if b_n == 0:
            continue
        b_str = f"{b_wr:5.1f}%({b_n:3d})"
        if h in best_tod:
            c_n, c_wr  = best_tod[h]
            delta_wr   = c_wr - b_wr
            sign       = "+" if delta_wr >= 0 else ""
            c_str      = f"{c_wr:5.1f}%({c_n:3d})"
            d_wr_str   = f"{sign}{delta_wr:4.1f}pp"
            d_n_str    = f"{c_n - b_n:+d}"
        else:
            c_str    = "  filtered   "
            d_wr_str = "       "
            d_n_str  = f"-{b_n}"
        lines.append(
            f"  {h:02d}:00–{h+1:02d}:00   {b_str:>12}  {c_str:>12}  "
            f"{d_wr_str:>8}  {d_n_str:>9}"
        )

    lines.append("")
    lines.append("★ = hours where filtered combo WR ≥ 80%   ✗ = WR < 70%")
    lines.append("=" * 94)
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_data()
    results = run_grid(df)
    report  = build_report(results)
    print("\n" + report)
    REPORT_PATH.write_text(report)
    print(f"\nSaved → {REPORT_PATH}")

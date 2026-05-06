#!/usr/bin/env python3
"""TIER 1 — HTF FVG Spatial + Momentum Combined Filter Grid Search (Aug–Dec 2025)

Combines two orthogonal filters per parent TF:
  1. SPATIAL  — entry_price falls inside a same-direction parent FVG gap
                (gap_bottom <= entry <= gap_top, FVG formed within lookback window)
  2. MOMENTUM — parent TF is trending in trade direction over the same N bars
                (close[t] > close[t-N] for bullish; < for bearish)

Both must pass for a given TF before that TF counts as confirming.
OR/AND logic then applies to the per-TF combined check across multiple TFs.

Grid: [5,21,89]-min × [3,5,10] bars × OR/AND = 27 combos.
Baseline: 1,559 trades | 76.33% WR | PF 1.20 | TPD 14.77 | P&L $3,694.40
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──────────────────────────────────────────────────────────────── #
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

REPORT_PATH = Path("backtest_tier1_fvg_combined_filter_report.txt")

BASELINE = {
    "total_trades":  1559,
    "win_rate":      76.33,
    "profit_factor": 1.20,
    "avg_tpd":       14.77,
    "total_pnl":     3694.40,
}


# ── Data loading ─────────────────────────────────────────────────────────── #

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


# ── Spatial index ────────────────────────────────────────────────────────── #

def build_parent_fvg_spatial_index(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample to tf_minutes; return DataFrame [close_time, direction, gap_top, gap_bottom].
    Same candle-role convention as baseline: bullish gap_top=c1_high, gap_bottom=c3_low.
    """
    rs = (
        df.set_index("timestamp")
        .resample(f"{tf_minutes}min", closed="right", label="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open"])
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")

    fvgs = []
    for i in range(2, len(rs)):
        c1, c3 = rs.iloc[i - 2], rs.iloc[i]
        ts = rs.iloc[i]["timestamp"]
        if c1["close"] > c3["open"] and c1["high"] > c3["low"]:
            fvgs.append({"close_time": ts, "direction": "bullish",
                         "gap_top": c1["high"], "gap_bottom": c3["low"]})
        if c1["close"] < c3["open"] and c3["high"] > c1["low"]:
            fvgs.append({"close_time": ts, "direction": "bearish",
                         "gap_top": c3["high"], "gap_bottom": c1["low"]})

    cols = ["close_time", "direction", "gap_top", "gap_bottom"]
    result = pd.DataFrame(fvgs) if fvgs else pd.DataFrame(columns=cols)
    print(f"  {tf_minutes:3d}-min spatial FVGs: {len(result)}")
    return result


def is_entry_inside_parent_fvg(entry_price: float, direction: str, bar_ts,
                                parent_df: pd.DataFrame,
                                lookback_bars: int, tf_minutes: int) -> bool:
    if parent_df.empty:
        return False
    window_start = bar_ts - pd.Timedelta(minutes=lookback_bars * tf_minutes)
    # close_time <= bar_ts: FVG must have closed before/at signal bar — no look-ahead
    candidates = parent_df[
        (parent_df["direction"] == direction)
        & (parent_df["close_time"] > window_start)
        & (parent_df["close_time"] <= bar_ts)
    ]
    if candidates.empty:
        return False
    return bool(
        ((candidates["gap_bottom"] <= entry_price) & (entry_price <= candidates["gap_top"])).any()
    )


# ── Momentum index ───────────────────────────────────────────────────────── #

def build_momentum_index(df: pd.DataFrame, tf_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample to tf_minutes; return (timestamps, closes) numpy arrays."""
    rs = (
        df.set_index("timestamp")
        .resample(f"{tf_minutes}min", closed="right", label="right")
        .agg({"close": "last"})
        .dropna()
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")
    print(f"  {tf_minutes:3d}-min momentum bars:  {len(rs)}")
    return rs["timestamp"].values, rs["close"].values


def build_momentum_signal(closes: np.ndarray, lookback: int) -> np.ndarray:
    """+1 if trending bullish over lookback bars, -1 bearish, 0 insufficient history."""
    n = len(closes)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        diff = closes[i] - closes[i - lookback]
        if diff > 0:
            sig[i] = 1
        elif diff < 0:
            sig[i] = -1
    return sig


def lookup_momentum(direction: str, bar_ts_np: np.datetime64,
                    htf_ts: np.ndarray, htf_signal: np.ndarray) -> bool:
    idx = int(np.searchsorted(htf_ts, bar_ts_np, side="right")) - 1
    if idx < 0:
        return False
    s = int(htf_signal[idx])
    if s == 0:
        return False
    return s == (1 if direction == "bullish" else -1)


# ── Exit simulation ──────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, start_idx, n):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_idx + j
        if idx >= n:
            break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
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
    ep  = closes[min(start_idx + MAX_HOLD_BARS, n - 1)]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    return {"exit_type": "time", "bars_held": MAX_HOLD_BARS, "pnl": pnl}


# ── Backtest engine ──────────────────────────────────────────────────────── #

def run_backtest(df: pd.DataFrame,
                 spatial_indices: dict[int, pd.DataFrame],
                 momentum_signals: dict[int, dict[int, np.ndarray]],
                 htf_timestamps: dict[int, np.ndarray],
                 tfs: list[int], lookback: int, logic: str) -> dict:
    """Both spatial AND momentum must pass per TF; OR/AND applies across TFs."""
    n          = len(df)
    highs      = df["high"].values
    lows       = df["low"].values
    opens      = df["open"].values
    closes     = df["close"].values
    vols       = df["volume"].values
    timestamps = df["timestamp"]          # Pandas Series — for spatial (needs Timedelta)
    ts_np      = df["timestamp"].values   # numpy array  — for momentum (needs searchsorted)

    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values

    is_bull = (closes > opens).astype(float)
    is_bear = (closes < opens).astype(float)
    up_vol  = pd.Series(vols * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol  = pd.Series(vols * is_bear).rolling(20, min_periods=1).sum().values

    trades         = []
    mtf_skipped    = 0
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        bar_ts    = timestamps.iloc[i]   # Pandas Timestamp
        bar_ts_np = ts_np[i]             # numpy datetime64

        c1_close = closes[i - 2]
        c1_high  = highs[i - 2]
        c3_open  = opens[i]
        c3_low   = lows[i]
        c3_high  = highs[i]

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

            # Combined check: spatial AND momentum both required per TF
            checks = []
            for tf in tfs:
                spatial_ok = is_entry_inside_parent_fvg(
                    entry, direction, bar_ts,
                    spatial_indices[tf], lookback, tf
                )
                mom_ok = lookup_momentum(
                    direction, bar_ts_np,
                    htf_timestamps[tf], momentum_signals[tf][lookback]
                )
                checks.append(spatial_ok and mom_ok)

            passes = any(checks) if logic == "OR" else all(checks)

            if not passes:
                mtf_skipped += 1
                continue

            result = simulate_trade(direction, entry, tp, sl, highs, lows, closes, i, n)
            result["direction"] = direction
            trades.append(result)
            next_entry_bar = i + result["bars_held"] + 1
            break

    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_tpd": 0.0, "total_pnl": 0.0, "filtered_pct": 100.0,
                "mtf_skipped": mtf_skipped}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))

    days  = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    tdays = days * (252 / 365)
    total = len(trades)
    filt  = mtf_skipped / (total + mtf_skipped) * 100 if (total + mtf_skipped) > 0 else 0.0

    return {
        "total_trades":  total,
        "win_rate":      len(wins) / total * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_tpd":       total / tdays if tdays > 0 else 0.0,
        "total_pnl":     sum(t["pnl"] for t in trades),
        "filtered_pct":  filt,
        "mtf_skipped":   mtf_skipped,
    }


# ── Grid search ──────────────────────────────────────────────────────────── #

def run_grid(df: pd.DataFrame) -> list[dict]:
    print("\nPre-computing spatial FVG indices ...")
    spatial_indices = {tf: build_parent_fvg_spatial_index(df, tf) for tf in PARENT_TFS}

    print("Pre-computing momentum indices ...")
    htf_timestamps: dict[int, np.ndarray] = {}
    raw_closes:     dict[int, np.ndarray] = {}
    for tf in PARENT_TFS:
        ts_arr, cl_arr       = build_momentum_index(df, tf)
        htf_timestamps[tf]   = ts_arr
        raw_closes[tf]       = cl_arr

    print("Building momentum signals ...")
    momentum_signals: dict[int, dict[int, np.ndarray]] = {tf: {} for tf in PARENT_TFS}
    for tf in PARENT_TFS:
        for lb in LOOKBACK_BARS:
            momentum_signals[tf][lb] = build_momentum_signal(raw_closes[tf], lb)

    all_combos = []

    # Single-TF combos (OR = AND for one TF)
    for tf in PARENT_TFS:
        for lb in LOOKBACK_BARS:
            all_combos.append({"tfs": [tf], "lookback": lb, "logic": "OR",
                               "label": f"COMB {tf:3d}m  LB{lb:2d} OR "})

    # Triple [5,21,89] OR and AND
    for lb, lg in product(LOOKBACK_BARS, LOGIC_OPTIONS):
        tf_str = "+".join(str(t) for t in PARENT_TFS)
        all_combos.append({"tfs": PARENT_TFS, "lookback": lb, "logic": lg,
                           "label": f"FIB{tf_str}m LB{lb:2d} {lg}"})

    # Fibonacci pairs [5,21] and [21,89]
    for fib_pair in [[5, 21], [21, 89]]:
        available = [t for t in fib_pair if t in spatial_indices]
        if len(available) < 2:
            continue
        tf_str = "+".join(str(t) for t in available)
        for lb in LOOKBACK_BARS:
            for lg in LOGIC_OPTIONS:
                all_combos.append({"tfs": available, "lookback": lb, "logic": lg,
                                   "label": f"FIB{tf_str}m LB{lb:2d} {lg}"})

    total = len(all_combos)
    print(f"\nRunning {total} combinations ...")

    results = []
    for k, combo in enumerate(all_combos):
        tfs   = combo["tfs"]
        lb    = combo["lookback"]
        lg    = combo["logic"]
        label = combo["label"]
        r = run_backtest(df, spatial_indices, momentum_signals, htf_timestamps, tfs, lb, lg)
        r["label"]    = label
        r["tfs"]      = tfs
        r["lookback"] = lb
        r["logic"]    = lg
        results.append(r)
        print(f"  [{k+1:3d}/{total}] {label} → "
              f"{r['total_trades']:4d} trades  "
              f"{r['win_rate']:5.2f}% WR  "
              f"PF {r['profit_factor']:.2f}  "
              f"TPD {r['avg_tpd']:.1f}  "
              f"Filt {r['filtered_pct']:.1f}%")

    return results


# ── Report ─────────────────────────────────────────────────────────────── #

def build_report(results: list[dict]) -> str:
    b              = BASELINE
    sorted_results = sorted(results, key=lambda x: x["win_rate"], reverse=True)

    lines = []
    lines.append("=" * 94)
    lines.append("TIER 1 HTF FVG SPATIAL + MOMENTUM COMBINED FILTER — AUG–DEC 2025")
    lines.append("=" * 94)
    lines.append(f"Base config: SL{SL_MULTIPLIER}x | ATR{ATR_THRESHOLD} | Vol{VOLUME_RATIO_THRESHOLD} | MaxGap${MAX_GAP_DOLLARS} | Hold{MAX_HOLD_BARS}")
    lines.append("Filter: BOTH (entry inside parent FVG gap) AND (parent TF trending in trade direction)")
    lines.append(f"Parent TFs: {PARENT_TFS} min  |  Lookbacks: {LOOKBACK_BARS} bars  |  Logic: OR / AND")
    lines.append("=" * 94)
    lines.append("")
    lines.append(f"BASELINE (no MTF): {b['total_trades']} trades | WR {b['win_rate']:.2f}% | PF {b['profit_factor']:.2f} | TPD {b['avg_tpd']:.2f} | P&L ${b['total_pnl']:.2f}")
    lines.append("")
    lines.append("TOP 20 (sorted by win rate ↓):")
    lines.append("")
    lines.append(f"  {'#':>3}  {'Label':<28} {'Trades':>7}  {'WR%':>7}  {'PF':>5}  {'TPD':>6}  {'P&L':>11}  {'Filt%':>6}")
    lines.append("-" * 94)

    star_count = 0
    for rank, r in enumerate(sorted_results[:20], 1):
        wr_delta = r["win_rate"] - b["win_rate"]
        is_star  = (wr_delta >= STAR_WR_DELTA) and (r["avg_tpd"] >= MIN_TRADES_PER_DAY)
        star     = "★" if is_star else " "
        if is_star:
            star_count += 1
        lines.append(
            f" {star} {rank:>3}  {r['label']:<28} {r['total_trades']:>7}  "
            f"{r['win_rate']:>6.2f}%  {r['profit_factor']:>5.2f}  "
            f"{r['avg_tpd']:>6.2f}  ${r['total_pnl']:>10.2f}  {r['filtered_pct']:>5.1f}%"
        )

    lines.append("-" * 94)
    lines.append("")
    lines.append(f"★ = WR improves >{STAR_WR_DELTA}% over baseline AND TPD ≥ {MIN_TRADES_PER_DAY}")
    lines.append(f"★ combos found: {star_count}")
    lines.append("")

    # Sort by PF for a secondary view
    by_pf = sorted(results, key=lambda x: x["profit_factor"], reverse=True)
    lines.append("TOP 10 BY PROFIT FACTOR:")
    lines.append("")
    lines.append(f"  {'#':>3}  {'Label':<28} {'Trades':>7}  {'WR%':>7}  {'PF':>5}  {'TPD':>6}  {'Filt%':>6}")
    lines.append("-" * 80)
    for rank, r in enumerate(by_pf[:10], 1):
        lines.append(
            f"      {rank:>3}  {r['label']:<28} {r['total_trades']:>7}  "
            f"{r['win_rate']:>6.2f}%  {r['profit_factor']:>5.2f}  "
            f"{r['avg_tpd']:>6.2f}  {r['filtered_pct']:>5.1f}%"
        )
    lines.append("")

    lines.append("BOTTOM 5 (worst win rate):")
    lines.append("")
    for r in sorted_results[-5:]:
        lines.append(f"     {r['label']:<28} {r['total_trades']:>7}  {r['win_rate']:>6.2f}%  PF {r['profit_factor']:.2f}  Filt {r['filtered_pct']:.1f}%")

    lines.append("")
    lines.append("=" * 94)
    return "\n".join(lines)


if __name__ == "__main__":
    df      = load_data()
    results = run_grid(df)
    report  = build_report(results)
    print("\n" + report)
    REPORT_PATH.write_text(report)
    print(f"\nSaved → {REPORT_PATH}")

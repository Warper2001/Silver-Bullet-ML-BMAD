#!/usr/bin/env python3
"""TIER 1 FVG — MTF Parent Filter Grid Search (Aug–Dec 2025)

The baseline MTF filter (no quality filter on parent FVGs) barely works:
9,663 15-min parents → only 7% of 1-min signals filtered.

This script grid-searches parent FVG filter parameters to find combinations
that meaningfully raise win rate without collapsing trade frequency.

Grid dimensions:
  - Parent ATR threshold: gap must be >= ATR × threshold (5 values)
  - Parent min gap ($):   gap in dollars (4 values)
  - MTF logic:            OR (any parent TF confirms) | AND (both must confirm) (2 values)

Total: 5 × 4 × 2 = 40 combinations
Data: MNQ Aug 1 – Dec 31 2025 (118k 1-min bars)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ────────────────────────────────────────────────────────────────── #
DATA_PATH              = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
START_DATE             = "2025-08-01"
END_DATE               = "2025-12-31"
REPORT_PATH            = Path("backtest_tier1_mtf_grid_search_report.txt")

# TIER 1 entry filters (fixed — same as validated config)
SL_MULTIPLIER          = 2.5
ATR_THRESHOLD          = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS        = 50.0
MAX_HOLD_BARS          = 10
MNQ_TICK_SIZE          = 0.25
MNQ_CONTRACT_VALUE     = 5.0
TRANSACTION_COST       = 10.90

# Grid dimensions
PARENT_ATR_THRESHOLDS  = [0.0, 0.3, 0.5, 0.7, 1.0]
PARENT_MIN_GAP_DOLLARS = [0.0, 20.0, 50.0, 100.0]
MTF_LOGIC_OPTIONS      = ["OR", "AND"]   # OR = any TF confirms; AND = both must confirm
MTF_TIMEFRAMES         = [15, 240]       # 15-min, 4-hour

# ── Data loading ────────────────────────────────────────────────────────────── #

def load_data() -> pd.DataFrame:
    print(f"Loading MNQ 1-min data {START_DATE} → {END_DATE} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df[
        (df["timestamp"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(END_DATE + " 23:59", tz="UTC"))
    ].sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} bars  ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")
    return df


# ── Parent FVG builder (with quality metrics) ──────────────────────────────── #

def _ewm_atr(highs, lows, closes, span=14):
    prev = np.roll(closes, 1)
    prev[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))
    return pd.Series(tr).ewm(span=span, adjust=False).mean().values


def build_parent_fvgs_with_metrics(df: pd.DataFrame, tf_minutes: int) -> list[dict]:
    """Resample to tf_minutes, detect ALL raw FVGs, attach ATR ratio + gap dollars.

    Returned list is unfiltered — filtered at grid-search time for speed.
    Each item: {direction, gap_top, gap_bottom, close_time, gap_dollars, atr_ratio}
    """
    freq = f"{tf_minutes}min"
    rs = (
        df.set_index("timestamp")
        .resample(freq, closed="right", label="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open"])
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")

    h = rs["high"].values
    l = rs["low"].values
    o = rs["open"].values
    c = rs["close"].values
    atr = _ewm_atr(h, l, c)

    fvgs = []
    for i in range(2, len(rs)):
        close_ts = rs["timestamp"].iloc[i]
        c1_close, c1_high, c1_low = c[i - 2], h[i - 2], l[i - 2]
        c3_open,  c3_high, c3_low = o[i],     h[i],     l[i]
        bar_atr = atr[i] if atr[i] > 0 else 1e-9

        # Bullish
        if c1_close > c3_open:
            top, bot = c1_high, c3_low
            if top > bot:
                gs = (top - bot) * MNQ_CONTRACT_VALUE
                fvgs.append({
                    "direction": "bullish", "gap_top": top, "gap_bottom": bot,
                    "close_time": close_ts,
                    "gap_dollars": gs,
                    "atr_ratio": (top - bot) / bar_atr,
                })

        # Bearish
        if c1_close < c3_open:
            top, bot = c3_high, c1_low
            if top > bot:
                gs = (top - bot) * MNQ_CONTRACT_VALUE
                fvgs.append({
                    "direction": "bearish", "gap_top": top, "gap_bottom": bot,
                    "close_time": close_ts,
                    "gap_dollars": gs,
                    "atr_ratio": (top - bot) / bar_atr,
                })

    return fvgs


def filter_parents(all_fvgs: list[dict],
                   min_atr_ratio: float,
                   min_gap_dollars: float) -> list[dict]:
    return [f for f in all_fvgs
            if f["atr_ratio"] >= min_atr_ratio
            and f["gap_dollars"] >= min_gap_dollars]


# ── MTF nesting check ──────────────────────────────────────────────────────── #

def is_nested_or(direction, gap_top, gap_bottom, bar_ts,
                 filtered_parents_per_tf: list[list[dict]]) -> bool:
    """True if 1-min FVG is contained within ANY qualifying parent across any TF."""
    for parent_list in filtered_parents_per_tf:
        for pf in parent_list:
            if pf["direction"] != direction:
                continue
            if pf["close_time"] > bar_ts:
                continue
            if gap_bottom >= pf["gap_bottom"] and gap_top <= pf["gap_top"]:
                return True
    return False


def is_nested_and(direction, gap_top, gap_bottom, bar_ts,
                  filtered_parents_per_tf: list[list[dict]]) -> bool:
    """True if 1-min FVG is nested in at least one qualifying parent on EACH TF."""
    for parent_list in filtered_parents_per_tf:
        found_in_this_tf = False
        for pf in parent_list:
            if pf["direction"] != direction:
                continue
            if pf["close_time"] > bar_ts:
                continue
            if gap_bottom >= pf["gap_bottom"] and gap_top <= pf["gap_top"]:
                found_in_this_tf = True
                break
        if not found_in_this_tf:
            return False
    return True


# ── Trade exit simulation ─────────────────────────────────────────────────── #

def simulate_exit(direction, entry, tp, sl, highs, lows, closes, start_i, n):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_i + j
        if idx >= n:
            break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl:
                ep = min(sl, l)
                return ("sl", ep, j, (ep - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST)
            if h >= tp:
                return ("tp", tp, j, (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST)
        else:
            if h >= sl:
                ep = max(sl, h)
                return ("sl", ep, j, (entry - ep) * MNQ_CONTRACT_VALUE - TRANSACTION_COST)
            if l <= tp:
                return ("tp", tp, j, (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST)
    idx = min(start_i + MAX_HOLD_BARS, n - 1)
    ep = closes[idx]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    return ("time", ep, MAX_HOLD_BARS, pnl)


# ── Backtest engine ───────────────────────────────────────────────────────── #

def run_backtest(df: pd.DataFrame,
                 filtered_parents_per_tf: list[list[dict]],
                 mtf_logic: str) -> dict:
    n = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values
    vols   = df["volume"].values
    ts_series = df["timestamp"]

    # Rolling 20-bar ATR + volume (matches live paper trader methodology)
    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    is_bull = (closes > opens).astype(float)
    is_bear = (closes < opens).astype(float)
    up_vol = pd.Series(vols * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(vols * is_bear).rolling(20, min_periods=1).sum().values

    check = is_nested_or if mtf_logic == "OR" else is_nested_and

    trades         = []
    mtf_skipped    = 0
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        bar_ts = ts_series.iloc[i]
        c1_close, c1_high = closes[i - 2], highs[i - 2]
        c3_open, c3_low, c3_high = opens[i], lows[i], highs[i]

        for direction in ("bullish", "bearish"):
            if direction == "bullish":
                if c1_close <= c3_open: continue
                gap_top, gap_bottom = c1_high, c3_low
                entry, tp = gap_bottom, gap_top
                sl = entry - (gap_top - gap_bottom) * SL_MULTIPLIER
                uv, dv = up_vol[i], dn_vol[i]
                ratio = uv / dv if dv > 0 else float("inf")
            else:
                if c1_close >= c3_open: continue
                gap_top    = c3_high
                gap_bottom = lows[i - 2]
                entry, tp = gap_top, gap_bottom
                sl = entry + (gap_top - gap_bottom) * SL_MULTIPLIER
                uv, dv = up_vol[i], dn_vol[i]
                ratio = dv / uv if uv > 0 else float("inf")

            if gap_top <= gap_bottom: continue
            if (gap_top - gap_bottom) < atr[i] * ATR_THRESHOLD: continue
            if (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS: continue
            if ratio < VOLUME_RATIO_THRESHOLD: continue

            if not check(direction, gap_top, gap_bottom, bar_ts, filtered_parents_per_tf):
                mtf_skipped += 1
                continue

            exit_type, exit_price, bars_held, pnl = simulate_exit(
                direction, entry, tp, sl, highs, lows, closes, i, n)
            trades.append({"direction": direction, "pnl": pnl,
                           "exit_type": exit_type, "bars_held": bars_held})
            next_entry_bar = i + bars_held + 1
            break   # one signal per bar

    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_tpd": 0.0, "total_pnl": 0.0, "mtf_skipped": mtf_skipped,
                "tp": 0, "sl": 0, "time": 0}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    tpd  = len(trades) / (days * 252 / 365) if days > 0 else 0

    return {
        "total_trades":  len(trades),
        "win_rate":      len(wins) / len(trades) * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_tpd":       tpd,
        "total_pnl":     sum(t["pnl"] for t in trades),
        "mtf_skipped":   mtf_skipped,
        "tp": sum(1 for t in trades if t["exit_type"] == "tp"),
        "sl": sum(1 for t in trades if t["exit_type"] == "sl"),
        "time": sum(1 for t in trades if t["exit_type"] == "time"),
    }


# ── Grid search + reporting ───────────────────────────────────────────────── #

def run_grid(df, raw_parents_per_tf):
    results = []
    total = len(PARENT_ATR_THRESHOLDS) * len(PARENT_MIN_GAP_DOLLARS) * len(MTF_LOGIC_OPTIONS)
    done = 0

    for atr_thresh in PARENT_ATR_THRESHOLDS:
        for min_gap in PARENT_MIN_GAP_DOLLARS:
            # Filter parent lists once per (atr_thresh, min_gap) pair
            filtered = [filter_parents(raw, atr_thresh, min_gap)
                        for raw in raw_parents_per_tf]
            counts = [len(f) for f in filtered]

            for logic in MTF_LOGIC_OPTIONS:
                done += 1
                label = f"ATR≥{atr_thresh} Gap≥${min_gap:.0f} {logic}"
                print(f"  [{done:2d}/{total}] {label} | parents: {counts} ...", end="\r")
                m = run_backtest(df, filtered, logic)
                m["atr_thresh"]  = atr_thresh
                m["min_gap"]     = min_gap
                m["logic"]       = logic
                m["label"]       = label
                m["parent_counts"] = counts
                results.append(m)

    print()
    return results


def format_report(baseline: dict, results: list[dict]) -> str:
    # Sort by win rate desc, then profit factor desc
    ranked = sorted(results, key=lambda r: (r["win_rate"], r["profit_factor"]), reverse=True)

    lines = [
        "=" * 90,
        "TIER 1 MTF PARENT FILTER GRID SEARCH — AUG–DEC 2025",
        "=" * 90,
        f"Base config: SL{SL_MULTIPLIER}x | ATR{ATR_THRESHOLD} | Vol{VOLUME_RATIO_THRESHOLD} | MaxGap${MAX_GAP_DOLLARS} | Hold{MAX_HOLD_BARS}",
        f"MTF TFs: 15-min and 240-min (4-hour) parent FVGs",
        f"Parent filter grid: ATR∈{PARENT_ATR_THRESHOLDS}  Gap∈{PARENT_MIN_GAP_DOLLARS}  Logic∈{MTF_LOGIC_OPTIONS}",
        "=" * 90,
        "",
        f"BASELINE (no MTF filter): {baseline['total_trades']} trades | "
        f"WR {baseline['win_rate']:.2f}% | PF {baseline['profit_factor']:.2f} | "
        f"TPD {baseline['avg_tpd']:.2f} | P&L ${baseline['total_pnl']:.2f}",
        "",
        "GRID RESULTS (sorted by win rate ↓):",
        "",
        f"{'#':>3}  {'Label':<32} {'Trades':>7} {'WR%':>7} {'PF':>6} "
        f"{'TPD':>6} {'P&L':>10} {'Filt%':>7} {'Parents 15/240':>16}",
        "-" * 90,
    ]

    for rank, r in enumerate(ranked[:20], 1):   # top-20
        if baseline["total_trades"] > 0:
            filt_pct = r["mtf_skipped"] / (r["total_trades"] + r["mtf_skipped"]) * 100
        else:
            filt_pct = 0
        wr_delta = r["win_rate"] - baseline["win_rate"]
        marker = " ★" if wr_delta > 1.0 and r["avg_tpd"] >= 3 else ""
        lines.append(
            f"{rank:>3}  {r['label']:<32} {r['total_trades']:>7} "
            f"{r['win_rate']:>6.2f}% {r['profit_factor']:>6.2f} "
            f"{r['avg_tpd']:>6.2f} ${r['total_pnl']:>9.2f} "
            f"{filt_pct:>6.1f}% "
            f"  {r['parent_counts'][0]:>5}/{r['parent_counts'][1]:<5}"
            + marker
        )

    lines += [
        "-" * 90,
        "",
        "★ = win rate improves >1% over baseline AND avg trades/day ≥ 3",
        "",
        "BOTTOM 5 (worst win rate):",
        "",
    ]
    for r in ranked[-5:]:
        filt_pct = r["mtf_skipped"] / max(r["total_trades"] + r["mtf_skipped"], 1) * 100
        lines.append(
            f"     {r['label']:<32} {r['total_trades']:>7} "
            f"{r['win_rate']:>6.2f}% {r['profit_factor']:>6.2f} "
            f"{r['avg_tpd']:>6.2f} ${r['total_pnl']:>9.2f} {filt_pct:>6.1f}%"
        )

    lines += ["", "=" * 90]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    df = load_data()

    print("\nBuilding raw parent FVG lists (unfiltered) ...")
    raw_parents = [build_parent_fvgs_with_metrics(df, tf) for tf in MTF_TIMEFRAMES]
    for tf, rp in zip(MTF_TIMEFRAMES, raw_parents):
        print(f"  {tf:3d}-min: {len(rp):,} raw parent FVGs")

    # Baseline (no MTF filter) — run once
    print("\nRunning baseline (no MTF) ...")
    baseline = run_backtest(df, [[], []], "OR")  # empty parents → no filtering
    # Actually run with no filter at all by passing empty lists won't skip anything
    # Re-run properly: use a flag
    baseline = _run_no_mtf(df)
    print(f"  Baseline: {baseline['total_trades']} trades | WR {baseline['win_rate']:.2f}% | "
          f"PF {baseline['profit_factor']:.2f} | TPD {baseline['avg_tpd']:.2f} | "
          f"P&L ${baseline['total_pnl']:.2f}")

    print(f"\nRunning grid search ({len(PARENT_ATR_THRESHOLDS)} × "
          f"{len(PARENT_MIN_GAP_DOLLARS)} × {len(MTF_LOGIC_OPTIONS)} = "
          f"{len(PARENT_ATR_THRESHOLDS)*len(PARENT_MIN_GAP_DOLLARS)*len(MTF_LOGIC_OPTIONS)} combos) ...")
    results = run_grid(df, raw_parents)

    report = format_report(baseline, results)
    print("\n" + report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved → {REPORT_PATH}")


def _run_no_mtf(df: pd.DataFrame) -> dict:
    """Baseline backtest with no MTF filter."""
    n = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values
    vols   = df["volume"].values

    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    is_bull = (closes > opens).astype(float)
    is_bear = (closes < opens).astype(float)
    up_vol = pd.Series(vols * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(vols * is_bear).rolling(20, min_periods=1).sum().values

    trades         = []
    next_entry_bar = 0
    for i in range(2, n):
        if i < next_entry_bar:
            continue

        c1_close, c1_high = closes[i - 2], highs[i - 2]
        c3_open, c3_low, c3_high = opens[i], lows[i], highs[i]

        for direction in ("bullish", "bearish"):
            if direction == "bullish":
                if c1_close <= c3_open: continue
                gap_top, gap_bottom = c1_high, c3_low
                entry, tp = gap_bottom, gap_top
                sl = entry - (gap_top - gap_bottom) * SL_MULTIPLIER
                uv, dv = up_vol[i], dn_vol[i]
                ratio = uv / dv if dv > 0 else float("inf")
            else:
                if c1_close >= c3_open: continue
                gap_top, gap_bottom = c3_high, lows[i - 2]
                entry, tp = gap_top, gap_bottom
                sl = entry + (gap_top - gap_bottom) * SL_MULTIPLIER
                uv, dv = up_vol[i], dn_vol[i]
                ratio = dv / uv if uv > 0 else float("inf")

            if gap_top <= gap_bottom: continue
            if (gap_top - gap_bottom) < atr[i] * ATR_THRESHOLD: continue
            if (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS: continue
            if ratio < VOLUME_RATIO_THRESHOLD: continue

            exit_type, exit_price, bars_held, pnl = simulate_exit(
                direction, entry, tp, sl, highs, lows, closes, i, n)
            trades.append({"pnl": pnl, "exit_type": exit_type})
            next_entry_bar = i + bars_held + 1
            break

    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_tpd": 0.0, "total_pnl": 0.0}
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    tpd  = len(trades) / (days * 252 / 365) if days > 0 else 0
    return {
        "total_trades": len(trades),
        "win_rate":     len(wins) / len(trades) * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_tpd":      tpd,
        "total_pnl":    sum(t["pnl"] for t in trades),
    }


if __name__ == "__main__":
    main()

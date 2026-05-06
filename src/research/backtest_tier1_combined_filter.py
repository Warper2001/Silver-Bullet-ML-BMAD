#!/usr/bin/env python3
"""TIER 1 — Combined MOM5+21m LB10 AND + TOD Filter Backtest (Aug–Dec 2025)

Stacks two orthogonal filters:
  1. HTF momentum alignment: both 5-min AND 21-min must trend in trade direction
     over 10-bar lookback (MOM5+21m LB10 AND — best standalone PF: 1.34)
  2. TOD hour exclusion: block ET hours below WR threshold
     (Block WR<75% — best standalone P&L: +$2,531 over baseline)

Reports reference rows (baseline, MOM-only, TOD-only sweet spot) then the
full combined grid across TOD thresholds to find the stacked optimum.
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

REPORT_PATH = Path("backtest_tier1_combined_filter_report.txt")

BASELINE = {
    "total_trades": 1559, "win_rate": 76.33, "profit_factor": 1.20,
    "avg_tpd": 14.77, "total_pnl": 3694.40,
}

HOUR_WR: dict[int, tuple[int, float]] = {
     0: (84,  75.0),  1: (81,  66.7),  2: (100, 80.0),  3: (71,  87.3),
     4: (96,  86.5),  5: (82,  76.8),  6: (75,  70.7),  7: (87,  78.2),
     8: (70,  72.9),  9: (25,  92.0), 10: (8,   87.5), 11: (28,  85.7),
    12: (41,  80.5), 13: (42,  78.6), 14: (32,  78.1), 15: (40,  90.0),
    16: (83,  65.1), 17: (2,    0.0), 18: (85,  78.8), 19: (84,  76.2),
    20: (98,  75.5), 21: (79,  78.5), 22: (77,  63.6), 23: (89,  69.7),
}

TOD_THRESHOLDS = [64, 67, 70, 72, 75, 77, 79]


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
    dt  = pd.to_datetime(timestamps)
    off = np.where(dt.month <= 10, -4, -5)
    return (dt.hour + off) % 24


# ── HTF momentum ──────────────────────────────────────────────────────────────── #

def resample_htf(df: pd.DataFrame, tf_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    rs = (
        df.set_index("timestamp")
        .resample(f"{tf_minutes}min", closed="right", label="right")
        .agg({"close": "last"})
        .dropna()
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")
    return rs["timestamp"].values, rs["close"].values


def build_momentum_signal(closes: np.ndarray, lookback: int) -> np.ndarray:
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
    idx = int(np.searchsorted(htf_ts, bar_ts, side="right")) - 1
    if idx < 0:
        return False
    sig = int(htf_signal[idx])
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
                 et_hours: np.ndarray,
                 blocked_hours: frozenset,
                 mom_signals: list[tuple[np.ndarray, np.ndarray]] | None) -> dict:
    """
    mom_signals: list of (htf_ts, htf_signal) pairs — ALL must align (AND logic).
                 Pass None to disable momentum filter.
    blocked_hours: ET hours to skip entirely. Pass frozenset() to disable TOD filter.
    """
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
    tod_skipped    = 0
    mom_skipped    = 0
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        if et_hours[i] in blocked_hours:
            tod_skipped += 1
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

            if mom_signals is not None:
                if not all(lookup_momentum(direction, bar_ts, ts, sig)
                           for ts, sig in mom_signals):
                    mom_skipped += 1
                    continue

            result = simulate_trade(direction, entry, tp, sl, highs, lows, closes, i, n)
            result["direction"] = direction
            trades.append(result)
            next_entry_bar = i + result["bars_held"] + 1
            break

    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_tpd": 0.0, "total_pnl": 0.0,
                "tod_skipped": tod_skipped, "mom_skipped": mom_skipped}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    days   = (pd.Timestamp(df["timestamp"].iloc[-1]) -
               pd.Timestamp(df["timestamp"].iloc[0])).total_seconds() / 86400
    tdays  = days * (252 / 365)
    total  = len(trades)

    return {
        "total_trades":  total,
        "win_rate":      len(wins) / total * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_tpd":       total / tdays if tdays > 0 else 0.0,
        "total_pnl":     sum(t["pnl"] for t in trades),
        "tod_skipped":   tod_skipped,
        "mom_skipped":   mom_skipped,
    }


# ── Report ────────────────────────────────────────────────────────────────────── #

def fmt_row(label, r, b, star=False) -> str:
    wr_d   = r["win_rate"] - b["win_rate"]
    pnl_d  = r["total_pnl"] - b["total_pnl"]
    s_wr   = "+" if wr_d >= 0 else ""
    s_pnl  = "+" if pnl_d >= 0 else ""
    mk     = "★" if star else " "
    return (
        f" {mk} {label:<32} {r['total_trades']:>6}  "
        f"{r['win_rate']:>6.2f}%  {s_wr}{wr_d:>5.2f}pp  "
        f"{r['profit_factor']:>5.2f}  {r['avg_tpd']:>5.1f}  "
        f"${r['total_pnl']:>10.2f}  {s_pnl}${abs(pnl_d):>8.2f}"
    )


def build_report(ref_rows: list[tuple[str, dict]],
                 combined_rows: list[tuple[str, dict]]) -> str:
    b = BASELINE
    hdr = (f"  {'Config':<32} {'Trades':>6}  {'WR%':>7}  {'ΔWR':>8}  "
           f"{'PF':>5}  {'TPD':>5}  {'P&L':>11}  {'ΔP&L':>10}")
    div = "-" * 100

    lines = []
    lines.append("=" * 100)
    lines.append("TIER 1 — COMBINED FILTER BACKTEST (MOM5+21m LB10 AND + TOD) — AUG–DEC 2025")
    lines.append("=" * 100)
    lines.append(f"MOM filter: 5-min AND 21-min must align over 10-bar lookback (both directions)")
    lines.append(f"TOD filter: skip ET hours below historical WR threshold")
    lines.append(f"★ = WR ≥ baseline +1pp AND TPD ≥ {MIN_TRADES_PER_DAY}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        f"  {'BASELINE':<32} {b['total_trades']:>6}  {b['win_rate']:>6.2f}%  "
        f"{'---':>8}  {b['profit_factor']:>5.2f}  {b['avg_tpd']:>5.1f}  "
        f"${b['total_pnl']:>10.2f}  {'---':>10}"
    )
    lines.append("")
    lines.append("REFERENCE — single filters:")
    lines.append(hdr)
    lines.append(div)
    for label, r in ref_rows:
        wr_d  = r["win_rate"] - b["win_rate"]
        star  = wr_d >= 1.0 and r["avg_tpd"] >= MIN_TRADES_PER_DAY
        lines.append(fmt_row(label, r, b, star))

    lines.append("")
    lines.append("COMBINED — MOM5+21m LB10 AND stacked on TOD threshold:")
    lines.append(hdr)
    lines.append(div)

    star_count = 0
    for label, r in combined_rows:
        wr_d  = r["win_rate"] - b["win_rate"]
        star  = wr_d >= 1.0 and r["avg_tpd"] >= MIN_TRADES_PER_DAY
        if star:
            star_count += 1
        lines.append(fmt_row(label, r, b, star))

    lines.append(div)
    lines.append(f"  ★ combined configs: {star_count}")
    lines.append("")

    all_rows = ref_rows + combined_rows
    viable   = [(lbl, r) for lbl, r in all_rows
                if (r["win_rate"] - b["win_rate"]) >= 1.0 and r["avg_tpd"] >= MIN_TRADES_PER_DAY]

    lines.append("=" * 100)
    lines.append("VERDICT:")
    if viable:
        best_pnl = max(viable, key=lambda x: x[1]["total_pnl"])
        best_wr  = max(viable, key=lambda x: x[1]["win_rate"])
        lines.append(f"  Best P&L:  '{best_pnl[0]}' → "
                     f"WR {best_pnl[1]['win_rate']:.2f}%  "
                     f"PF {best_pnl[1]['profit_factor']:.2f}  "
                     f"TPD {best_pnl[1]['avg_tpd']:.1f}  "
                     f"P&L ${best_pnl[1]['total_pnl']:.2f} "
                     f"(+${best_pnl[1]['total_pnl']-b['total_pnl']:.2f})")
        lines.append(f"  Best WR:   '{best_wr[0]}' → "
                     f"WR {best_wr[1]['win_rate']:.2f}%  "
                     f"PF {best_wr[1]['profit_factor']:.2f}  "
                     f"TPD {best_wr[1]['avg_tpd']:.1f}  "
                     f"P&L ${best_wr[1]['total_pnl']:.2f} "
                     f"(+${best_wr[1]['total_pnl']-b['total_pnl']:.2f})")
    else:
        lines.append("  No viable combined config found.")
    lines.append("=" * 100)
    return "\n".join(lines)


if __name__ == "__main__":
    df       = load_data()
    et_hours = precompute_et_hours(df["timestamp"].values)

    print("\nBuilding MOM5+21m LB10 signals ...")
    ts5,  cl5  = resample_htf(df, 5)
    ts21, cl21 = resample_htf(df, 21)
    sig5  = build_momentum_signal(cl5,  10)
    sig21 = build_momentum_signal(cl21, 10)
    mom_signals = [(ts5, sig5), (ts21, sig21)]
    print(f"  5m: {np.sum(sig5==1)} up / {np.sum(sig5==-1)} dn bars")
    print(f"  21m: {np.sum(sig21==1)} up / {np.sum(sig21==-1)} dn bars")

    no_block = frozenset()

    # Reference rows
    print("\nRunning reference configs ...")
    ref_rows = []

    r_mom = run_backtest(df, et_hours, no_block, mom_signals)
    ref_rows.append(("MOM5+21m LB10 AND (no TOD)", r_mom))
    print(f"  MOM only            → {r_mom['total_trades']} trades  "
          f"{r_mom['win_rate']:.2f}% WR  PF {r_mom['profit_factor']:.2f}")

    best_tod_block = frozenset(h for h, (_, wr) in HOUR_WR.items() if wr < 75)
    r_tod75 = run_backtest(df, et_hours, best_tod_block, None)
    ref_rows.append(("Block WR<75% (no MOM)", r_tod75))
    print(f"  TOD<75% only        → {r_tod75['total_trades']} trades  "
          f"{r_tod75['win_rate']:.2f}% WR  PF {r_tod75['profit_factor']:.2f}")

    # Combined grid
    print("\nRunning combined configs ...")
    combined_rows = []
    for thr in TOD_THRESHOLDS:
        blocked = frozenset(h for h, (_, wr) in HOUR_WR.items() if wr < thr)
        r = run_backtest(df, et_hours, blocked, mom_signals)
        label = f"MOM + Block WR<{thr}%"
        combined_rows.append((label, r))
        wr_d = r["win_rate"] - BASELINE["win_rate"]
        sign = "+" if wr_d >= 0 else ""
        print(f"  {label:<28} → {r['total_trades']:4d} trades  "
              f"{r['win_rate']:.2f}% WR ({sign}{wr_d:.2f}pp)  "
              f"PF {r['profit_factor']:.2f}  TPD {r['avg_tpd']:.1f}  "
              f"P&L ${r['total_pnl']:.2f}")

    report = build_report(ref_rows, combined_rows)
    print("\n" + report)
    REPORT_PATH.write_text(report)
    print(f"Saved → {REPORT_PATH}")

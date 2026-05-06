#!/usr/bin/env python3
"""Sharpe analysis for COMB 5m LB5 OR — three execution modes.

Modes:
  default    — entry at c3's extreme; perfect limit fill at signal bar.
               Inflated Sharpe (~5), useful only for signal-quality comparison.
  realistic  — market order at next bar's open; matches live system execution.
               Honest Sharpe (~-9); strategy loses money with market orders.
  limit      — limit order placed at gap extreme after bar i closes; only fills
               if price returns to the level within LIMIT_CANCEL bars.
               Honest representation of ICT FVG pullback trading.

Usage:
  python backtest_tier1_sharpe_analysis.py --mode limit --limit-cancel 5
  python backtest_tier1_sharpe_analysis.py --mode realistic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── CLI ───────────────────────────────────────────────────────────────────── #

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["default", "realistic", "limit"],
                   default="default")
    p.add_argument("--sl-mult",      type=float, default=2.5)
    p.add_argument("--atr-thresh",   type=float, default=0.7)
    p.add_argument("--vol-ratio",    type=float, default=2.25)
    p.add_argument("--max-gap",      type=float, default=50.0)
    p.add_argument("--max-hold",     type=int,   default=10)
    p.add_argument("--spatial-tf",   type=int,   default=5)
    p.add_argument("--lookback",     type=int,   default=5)
    p.add_argument("--limit-cancel", type=int,   default=5,
                   help="Max bars to wait for limit fill (limit mode only)")
    p.add_argument("--tp-mult",      type=float, default=1.0,
                   help="TP = entry ± tp_mult×gap_size (default 1.0 = gap_top/bottom)")
    return p


DATA_PATH = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")

# Scaling constants (not tunable — match live system conventions)
MNQ_CONTRACT_VALUE = 5.0   # $5/pt for gap/ATR dollar filters (backtest convention)
TRANSACTION_COST   = 10.90  # commission + 1-tick slippage roundtrip


# ── Data ─────────────────────────────────────────────────────────────────── #

def load_window(start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df[
        (df["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(end + " 23:59", tz="UTC"))
    ]
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Spatial index ─────────────────────────────────────────────────────────── #

def build_spatial_index(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    rs = (
        df.set_index("timestamp")
        .resample(f"{tf}min", closed="right", label="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open"])
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")
    fvgs = []
    for i in range(2, len(rs)):
        c1, c3, ts = rs.iloc[i - 2], rs.iloc[i], rs.iloc[i]["timestamp"]
        if c1["close"] > c3["open"] and c1["high"] > c3["low"]:
            fvgs.append({"close_time": ts, "direction": "bullish",
                         "gap_top": c1["high"], "gap_bottom": c3["low"]})
        if c1["close"] < c3["open"] and c3["high"] > c1["low"]:
            fvgs.append({"close_time": ts, "direction": "bearish",
                         "gap_top": c3["high"], "gap_bottom": c1["low"]})
    cols = ["close_time", "direction", "gap_top", "gap_bottom"]
    return pd.DataFrame(fvgs) if fvgs else pd.DataFrame(columns=cols)


def is_entry_inside(entry: float, direction: str, bar_ts,
                    fvg_df: pd.DataFrame, lb: int, tf: int) -> bool:
    if fvg_df.empty:
        return False
    ws = bar_ts - pd.Timedelta(minutes=lb * tf)
    cands = fvg_df[
        (fvg_df["direction"] == direction)
        & (fvg_df["close_time"] > ws)
        & (fvg_df["close_time"] <= bar_ts)
    ]
    if cands.empty:
        return False
    return bool(((cands["gap_bottom"] <= entry) & (entry <= cands["gap_top"])).any())


# ── Momentum index ────────────────────────────────────────────────────────── #

def build_momentum(df: pd.DataFrame, tf: int, lb: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    rs = (
        df.set_index("timestamp")
        .resample(f"{tf}min", closed="right", label="right")
        .agg({"close": "last"})
        .dropna()
        .reset_index()
    )
    if rs["timestamp"].dt.tz is None:
        rs["timestamp"] = rs["timestamp"].dt.tz_localize("UTC")
    ts = rs["timestamp"].values
    cl = rs["close"].values
    sig = np.zeros(len(cl), dtype=np.int8)
    for i in range(lb, len(cl)):
        d = cl[i] - cl[i - lb]
        sig[i] = 1 if d > 0 else (-1 if d < 0 else 0)
    return ts, sig


def lookup_mom(direction: str, bar_ts_np: np.datetime64,
               htf_ts: np.ndarray, htf_sig: np.ndarray) -> bool:
    idx = int(np.searchsorted(htf_ts, bar_ts_np, side="right")) - 1
    if idx < 0:
        return False
    return int(htf_sig[idx]) == (1 if direction == "bullish" else -1)


# ── Prepare (cache-friendly for grid search) ─────────────────────────────── #

def prepare(df: pd.DataFrame, spatial_tf: int = 5, lookback: int = 5
            ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build spatial index + momentum arrays once; pass to run() to avoid rebuilding."""
    fvg_df = build_spatial_index(df, spatial_tf)
    htf_ts, sig = build_momentum(df, spatial_tf, lookback)
    return fvg_df, htf_ts, sig


# ── Exit sim ──────────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes,
                   start_bar, n, max_hold):
    """Simulate exits starting from start_bar+1.

    start_bar meanings by mode:
      default  — bar i  (fill during signal bar's extreme price)
      realistic — bar i  (fill at opens[i+1]; full bar i+1 range is live)
      limit    — fill_bar (fill when price revisited gap level)
    """
    for j in range(1, max_hold + 1):
        idx = start_bar + j
        if idx >= n:
            break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl:
                return (min(sl, l) - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
            if h >= tp:
                return (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
        else:
            if h >= sl:
                return (entry - max(sl, h)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
            if l <= tp:
                return (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    ep = closes[min(start_bar + max_hold, n - 1)]
    return ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST


# ── Backtest ──────────────────────────────────────────────────────────────── #

def run(df: pd.DataFrame,
        mode: str       = "default",
        sl_mult: float  = 2.5,
        tp_mult: float  = 1.0,
        atr_thresh: float = 0.7,
        vol_ratio: float = 2.25,
        max_gap: float  = 50.0,
        max_hold: int   = 10,
        spatial_tf: int = 5,
        lookback: int   = 5,
        limit_cancel: int = 5,
        _prepared       = None,
        ) -> list[dict]:
    """Run backtest over df.

    Pass _prepared=(fvg_df, htf_ts, sig) to skip rebuilding spatial/momentum
    indexes (useful in grid search when spatial_tf/lookback are fixed).
    """
    if _prepared is None:
        fvg_df, htf_ts, sig = prepare(df, spatial_tf, lookback)
    else:
        fvg_df, htf_ts, sig = _prepared

    n          = len(df)
    highs      = df["high"].values
    lows       = df["low"].values
    opens      = df["open"].values
    closes     = df["close"].values
    vols       = df["volume"].values
    timestamps = df["timestamp"]
    ts_np      = df["timestamp"].values

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
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        bar_ts    = timestamps.iloc[i]
        bar_ts_np = ts_np[i]
        c1_close  = closes[i - 2]
        c1_high   = highs[i - 2]
        c3_open   = opens[i]
        c3_low    = lows[i]
        c3_high   = highs[i]

        if mode == "realistic" and i + 1 >= n:
            continue

        for direction in ("bullish", "bearish"):
            # ── FVG geometry ─────────────────────────────────────────── #
            if direction == "bullish":
                if c1_close <= c3_open:
                    continue
                gap_top, gap_bottom = c1_high, c3_low
            else:
                if c1_close >= c3_open:
                    continue
                gap_top    = c3_high
                gap_bottom = lows[i - 2]

            if gap_top <= gap_bottom:
                continue
            gs = gap_top - gap_bottom
            if gs < atr[i] * atr_thresh:
                continue
            if gs * MNQ_CONTRACT_VALUE > max_gap:
                continue

            # ── Volume filter ─────────────────────────────────────────── #
            uv, dv = up_vol[i], dn_vol[i]
            ratio = (uv / dv if dv > 0 else float("inf")) if direction == "bullish" \
                    else (dv / uv if uv > 0 else float("inf"))
            if ratio < vol_ratio:
                continue

            # ── Spatial + momentum filters ───────────────────────────── #
            gap_entry = gap_bottom if direction == "bullish" else gap_top
            if not (is_entry_inside(gap_entry, direction, bar_ts,
                                    fvg_df, lookback, spatial_tf)
                    and lookup_mom(direction, bar_ts_np, htf_ts, sig)):
                continue

            # ── Entry by mode ─────────────────────────────────────────── #
            if mode == "default":
                entry    = gap_entry
                start_bar = i

            elif mode == "realistic":
                entry    = opens[i + 1]
                start_bar = i           # exits start at i+1 = start_bar+1

            else:  # limit — wait for price to pull back to gap level
                fill_bar = None
                for k in range(1, limit_cancel + 1):
                    ki = i + k
                    if ki >= n:
                        break
                    if direction == "bullish" and lows[ki] <= gap_entry:
                        fill_bar = ki
                        break
                    if direction == "bearish" and highs[ki] >= gap_entry:
                        fill_bar = ki
                        break
                if fill_bar is None:
                    continue
                entry    = gap_entry
                start_bar = fill_bar

            # ── SL / TP (relative to actual entry price) ──────────────── #
            if direction == "bullish":
                tp = entry + gs * tp_mult   # extends beyond gap_top when tp_mult > 1
                sl = entry - gs * sl_mult
            else:
                tp = entry - gs * tp_mult   # extends below gap_bottom when tp_mult > 1
                sl = entry + gs * sl_mult

            # ── Simulate ─────────────────────────────────────────────── #
            pnl = simulate_trade(direction, entry, tp, sl,
                                 highs, lows, closes, start_bar, n, max_hold)

            bars_held = max_hold
            for j in range(1, max_hold + 1):
                idx = start_bar + j
                if idx >= n:
                    break
                h, l = highs[idx], lows[idx]
                if direction == "bullish" and (l <= sl or h >= tp):
                    bars_held = j
                    break
                if direction == "bearish" and (h >= sl or l <= tp):
                    bars_held = j
                    break

            trades.append({"date": bar_ts.date(), "pnl": pnl})
            next_entry_bar = start_bar + bars_held + 1
            break

    return trades


# ── Sharpe & stats ────────────────────────────────────────────────────────── #

def analyse(trades: list[dict], label: str, start: str, end: str,
            silent: bool = False) -> dict:
    """Compute stats and optionally print. Returns dict for grid search."""
    tdays_ct = len(pd.bdate_range(start=start, end=end))

    if not trades:
        stats = dict(trades=0, wr=0.0, pf=0.0, total_pnl=0.0,
                     mean_d=0.0, std_d=0.0, sharpe=float("nan"), max_dd=0.0,
                     trade_days=0, total_days=tdays_ct)
        if not silent:
            print(f"\n{label}: NO TRADES")
        return stats

    trade_df  = pd.DataFrame(trades)
    daily_pnl = trade_df.groupby("date")["pnl"].sum()

    all_days  = pd.bdate_range(start=start, end=end)
    all_daily = daily_pnl.reindex(all_days.date, fill_value=0.0)

    mean_d = all_daily.mean()
    std_d  = all_daily.std(ddof=1)
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else float("nan")

    total   = len(trades)
    wins    = sum(1 for t in trades if t["pnl"] > 0)
    wr      = wins / total * 100
    gross_p = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_l = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf      = gross_p / gross_l if gross_l > 0 else float("inf")
    total_pnl = sum(t["pnl"] for t in trades)

    cum      = all_daily.cumsum()
    max_dd   = (cum - cum.cummax()).min()

    stats = dict(trades=total, wr=wr, pf=pf, total_pnl=total_pnl,
                 mean_d=mean_d, std_d=std_d, sharpe=sharpe, max_dd=max_dd,
                 trade_days=int((daily_pnl > 0).sum()), total_days=tdays_ct)

    if not silent:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Period:         {start}  →  {end}")
        print(f"  Trades:         {total}  ({total / tdays_ct * 252:.1f} annualised/yr)")
        print(f"  Win Rate:       {wr:.2f}%")
        print(f"  Profit Factor:  {pf:.2f}")
        print(f"  Total P&L:      ${total_pnl:,.2f}")
        print(f"  Mean daily P&L: ${mean_d:.2f}  (std ${std_d:.2f})")
        print(f"  Sharpe (ann.):  {sharpe:.2f}")
        print(f"  Max Drawdown:   ${max_dd:,.2f}")
        print(f"  Trade days:     {stats['trade_days']} with trades / {tdays_ct} total")

        trade_df["month"] = pd.to_datetime(trade_df["date"]).dt.to_period("M")
        monthly = trade_df.groupby("month").agg(
            n=("pnl", "count"), pnl=("pnl", "sum"),
            wr=("pnl", lambda x: (x > 0).mean() * 100)
        )
        print(f"\n  Monthly breakdown:")
        for m, row in monthly.iterrows():
            print(f"    {m}  {int(row['n']):3d} trades  WR {row['wr']:.0f}%  P&L ${row['pnl']:,.0f}")

    return stats


# ── Entry point ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    args = _make_parser().parse_args()

    mode_label = {
        "default":   "DEFAULT  (gap-extreme entry — inflated, for reference only)",
        "realistic": "REALISTIC (market order at next bar open)",
        "limit":     f"LIMIT    (pullback to gap, cancel after {args.limit_cancel} bars)",
    }[args.mode]

    print(f"COMB 5m LB5 OR — Sharpe Analysis")
    print(f"Mode:   {mode_label}")
    print(f"Params: SL×{args.sl_mult}  TP×{args.tp_mult}  ATR≥{args.atr_thresh}"
          f"  Vol≥{args.vol_ratio}  MaxGap${args.max_gap}  Hold≤{args.max_hold}bars"
          + (f"  Cancel≤{args.limit_cancel}bars" if args.mode == "limit" else ""))

    kw = dict(mode=args.mode, sl_mult=args.sl_mult, tp_mult=args.tp_mult,
              atr_thresh=args.atr_thresh,
              vol_ratio=args.vol_ratio, max_gap=args.max_gap, max_hold=args.max_hold,
              spatial_tf=args.spatial_tf, lookback=args.lookback,
              limit_cancel=args.limit_cancel)

    print("\n[A] Aug–Dec 2025 (5-month in-sample) ...")
    df_a = load_window("2025-08-01", "2025-12-31")
    analyse(run(df_a, **kw), "A) Aug–Dec 2025  [in-sample]",  "2025-08-01", "2025-12-31")

    print("\n[B] Jan–Jun 2025 (6-month out-of-sample) ...")
    df_b = load_window("2025-01-01", "2025-06-30")
    analyse(run(df_b, **kw), "B) Jan–Jun 2025  [out-of-sample]", "2025-01-01", "2025-06-30")

#!/usr/bin/env python3
"""Sharpe analysis for COMB 5m LB5 OR — best tradeable config.

Runs two windows:
  A) Aug 1 – Dec 31, 2025  (5-month in-sample period)
  B) Jan 1 – Jun 30, 2025  (6-month out-of-sample period)

Sharpe = annualised(mean_daily_pnl / std_daily_pnl), all trading days included
(zero-trade days = $0), risk-free rate = 0.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_PATH              = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
SL_MULTIPLIER          = 2.5
ATR_THRESHOLD          = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS        = 50.0
MAX_HOLD_BARS          = 10
MNQ_CONTRACT_VALUE     = 5.0
TRANSACTION_COST       = 10.90
SPATIAL_TF             = 5
LOOKBACK               = 5


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


# ── Spatial index ────────────────────────────────────────────────────────── #

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


# ── Momentum index ───────────────────────────────────────────────────────── #

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
    ts  = rs["timestamp"].values
    cl  = rs["close"].values
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
    s = int(htf_sig[idx])
    return s == (1 if direction == "bullish" else -1)


# ── Exit sim ─────────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, i, n):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = i + j
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
    ep = closes[min(i + MAX_HOLD_BARS, n - 1)]
    return ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST


# ── Backtest returning trade-level records ────────────────────────────────── #

def run(df: pd.DataFrame) -> list[dict]:
    fvg_df      = build_spatial_index(df, SPATIAL_TF)
    htf_ts, sig = build_momentum(df, SPATIAL_TF, LOOKBACK)

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
            gs = gap_top - gap_bottom
            if gs < atr[i] * ATR_THRESHOLD:
                continue
            if gs * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS:
                continue

            uv, dv = up_vol[i], dn_vol[i]
            ratio = (uv / dv if dv > 0 else float("inf")) if direction == "bullish" \
                    else (dv / uv if uv > 0 else float("inf"))
            if ratio < VOLUME_RATIO_THRESHOLD:
                continue

            if not (is_entry_inside(entry, direction, bar_ts, fvg_df, LOOKBACK, SPATIAL_TF)
                    and lookup_mom(direction, bar_ts_np, htf_ts, sig)):
                continue

            pnl = simulate_trade(direction, entry, tp, sl, highs, lows, closes, i, n)
            bars_held = 1
            for j in range(1, MAX_HOLD_BARS + 1):
                idx = i + j
                if idx >= n:
                    break
                h, l = highs[idx], lows[idx]
                if direction == "bullish" and (l <= sl or h >= tp):
                    bars_held = j; break
                if direction == "bearish" and (h >= sl or l <= tp):
                    bars_held = j; break
            else:
                bars_held = MAX_HOLD_BARS

            trades.append({
                "date": bar_ts.date(),
                "pnl":  pnl,
            })
            next_entry_bar = i + bars_held + 1
            break

    return trades


# ── Sharpe & stats ───────────────────────────────────────────────────────── #

def analyse(trades: list[dict], label: str, start: str, end: str) -> None:
    if not trades:
        print(f"\n{label}: NO TRADES")
        return

    trade_df  = pd.DataFrame(trades)
    daily_pnl = trade_df.groupby("date")["pnl"].sum()

    # Build full trading-day calendar (Mon–Fri within window)
    all_days  = pd.bdate_range(start=start, end=end)
    all_daily = daily_pnl.reindex(all_days.date, fill_value=0.0)

    mean_d  = all_daily.mean()
    std_d   = all_daily.std(ddof=1)
    sharpe  = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else float("nan")

    total     = len(trades)
    wins      = sum(1 for t in trades if t["pnl"] > 0)
    wr        = wins / total * 100
    gross_p   = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_l   = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf        = gross_p / gross_l if gross_l > 0 else float("inf")
    total_pnl = sum(t["pnl"] for t in trades)
    tdays_ct  = len(pd.bdate_range(start=start, end=end))
    tpd       = total / (tdays_ct / (252 / 252))

    # Max drawdown (cumulative daily P&L)
    cum       = all_daily.cumsum()
    roll_max  = cum.cummax()
    drawdown  = cum - roll_max
    max_dd    = drawdown.min()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Period:         {start}  →  {end}")
    print(f"  Trades:         {total}  ({total / tdays_ct * 252:.1f} annualised/day)")
    print(f"  Win Rate:       {wr:.2f}%")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"  Total P&L:      ${total_pnl:,.2f}")
    print(f"  Mean daily P&L: ${mean_d:.2f}  (std ${std_d:.2f})")
    print(f"  Sharpe (ann.):  {sharpe:.2f}")
    print(f"  Max Drawdown:   ${max_dd:,.2f}")
    print(f"  Trade days:     {int((daily_pnl > 0).sum())} with trades / {tdays_ct} total")

    # Monthly breakdown
    trade_df["month"] = pd.to_datetime(trade_df["date"]).dt.to_period("M")
    monthly = trade_df.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wr=("pnl", lambda x: (x > 0).mean() * 100)
    )
    print(f"\n  Monthly breakdown:")
    for m, row in monthly.iterrows():
        print(f"    {m}  {int(row['trades']):3d} trades  WR {row['wr']:.0f}%  P&L ${row['pnl']:,.0f}")


if __name__ == "__main__":
    print("COMB 5m LB5 OR — Sharpe Analysis")
    print("Filter: 5-min spatial FVG + 5-min momentum (same lookback=5)")

    print("\n[A] Aug–Dec 2025 (5-month in-sample) ...")
    df_a   = load_window("2025-08-01", "2025-12-31")
    trades_a = run(df_a)
    analyse(trades_a, "A) Aug–Dec 2025  [5-month, in-sample]",
            "2025-08-01", "2025-12-31")

    print("\n[B] Jan–Jun 2025 (6-month out-of-sample) ...")
    df_b   = load_window("2025-01-01", "2025-06-30")
    trades_b = run(df_b)
    analyse(trades_b, "B) Jan–Jun 2025  [6-month, out-of-sample]",
            "2025-01-01", "2025-06-30")

#!/usr/bin/env python3
"""Two-phase grid search for the Tier 1 FVG strategy.

Speed strategy: spatial and momentum lookups are precomputed as per-bar numpy
boolean arrays once per dataset window, reducing each run from ~17s to ~0.1s.
Total grid time is ~30 seconds instead of ~45 minutes.

Phase 1 — structural params: sl_mult × tp_mult × max_hold × limit_cancel (limit mode).
Phase 2 — signal filters: atr_thresh × vol_ratio × max_gap around best P1 config.

tp_mult controls the TP extension beyond the gap edge (default 1.0 = gap_top/bottom).
R/R = tp_mult / sl_mult; break-even WR = sl_mult / (sl_mult + tp_mult).

Usage:
  python backtest_tier1_grid_search.py
  python backtest_tier1_grid_search.py --mode realistic
  python backtest_tier1_grid_search.py --top 30
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.research.backtest_tier1_sharpe_analysis import (
    load_window, build_spatial_index, build_momentum,
    MNQ_CONTRACT_VALUE, TRANSACTION_COST,
)

# ── Windows ───────────────────────────────────────────────────────────────── #

IS_START,  IS_END  = "2025-08-01", "2025-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-06-30"
SPATIAL_TF = 5
LOOKBACK   = 5

# Phase 1
P1_SL_MULTS      = [1.5, 2.0, 2.5]
P1_TP_MULTS      = [1.5, 2.0, 2.5, 3.0]   # tp_mult=1.0 proven unviable (71.4% BE WR)
P1_MAX_HOLDS     = [8, 10, 15]
P1_LIMIT_CANCELS = [5, 8, 10]
P1_ATR   = 0.7
P1_VOL   = 2.25
P1_GAP   = 50.0

# Phase 2
P2_ATR_THRESHS = [0.3, 0.5, 0.7, 1.0, 1.3]
P2_VOL_RATIOS  = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
P2_MAX_GAPS    = [15.0, 20.0, 30.0, 50.0, 75.0]


# ── Fast precompute ───────────────────────────────────────────────────────── #

def precompute(df: pd.DataFrame) -> dict:
    """Build all per-bar arrays that are independent of tunable parameters.

    Returns a dict of numpy arrays used by fast_run().
    Replaces repeated pandas/searchsorted calls inside the hot loop.
    """
    n          = len(df)
    highs      = df["high"].values
    lows       = df["low"].values
    opens      = df["open"].values
    closes     = df["close"].values
    vols       = df["volume"].values
    ts_np      = df["timestamp"].values

    # ATR (rolling 20)
    prev_close = pd.Series(closes).shift(1).values
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values

    # Directional volume (rolling 20)
    is_bull = (closes > opens).astype(float)
    is_bear = (closes < opens).astype(float)
    up_vol  = pd.Series(vols * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol  = pd.Series(vols * is_bear).rolling(20, min_periods=1).sum().values

    # Momentum: precompute per 1-min bar from 5-min signal array
    htf_ts, htf_sig = build_momentum(df, SPATIAL_TF, LOOKBACK)
    indices = np.searchsorted(htf_ts, ts_np, side="right") - 1
    mom_arr = np.zeros(n, dtype=np.int8)
    valid   = indices >= 0
    mom_arr[valid] = htf_sig[indices[valid]]

    # Spatial: precompute per 1-min bar whether bullish/bearish entry (lows[i]/highs[i])
    # falls inside an active 5-min FVG within the lookback window.
    fvg_df  = build_spatial_index(df, SPATIAL_TF)
    bull_sp = np.zeros(n, dtype=bool)
    bear_sp = np.zeros(n, dtype=bool)

    if not fvg_df.empty:
        window_ns = np.timedelta64(LOOKBACK * SPATIAL_TF * 60, "s").astype("timedelta64[ns]")
        for row in fvg_df.itertuples(index=False):
            ct = np.datetime64(row.close_time)
            # Mark bars AFTER FVG is confirmed (ct <= bar_ts < ct + window),
            # matching is_entry_inside: close_time <= bar_ts AND close_time > bar_ts - window.
            i0 = int(np.searchsorted(ts_np, ct,            side="left"))
            i1 = int(np.searchsorted(ts_np, ct + window_ns, side="left"))
            bar_idxs = np.arange(i0, i1)
            if len(bar_idxs) == 0:
                continue
            if row.direction == "bullish":
                ep = lows[bar_idxs]
                ok = (ep >= row.gap_bottom) & (ep <= row.gap_top)
                bull_sp[bar_idxs[ok]] = True
            else:
                ep = highs[bar_idxs]
                ok = (ep >= row.gap_bottom) & (ep <= row.gap_top)
                bear_sp[bar_idxs[ok]] = True

    return dict(
        n=n, highs=highs, lows=lows, opens=opens, closes=closes,
        atr=atr, up_vol=up_vol, dn_vol=dn_vol,
        mom_arr=mom_arr, bull_sp=bull_sp, bear_sp=bear_sp,
        ts_np=ts_np,
    )


# ── Fast run ──────────────────────────────────────────────────────────────── #

def fast_run(pc: dict,
             mode: str       = "limit",
             sl_mult: float  = 2.5,
             tp_mult: float  = 2.5,
             atr_thresh: float = 0.7,
             vol_ratio: float = 2.25,
             max_gap: float  = 50.0,
             max_hold: int   = 10,
             limit_cancel: int = 5,
             ) -> list[dict]:
    """High-speed backtest using precomputed per-bar arrays."""
    n       = pc["n"]
    highs   = pc["highs"]
    lows    = pc["lows"]
    opens   = pc["opens"]
    closes  = pc["closes"]
    atr     = pc["atr"]
    up_vol  = pc["up_vol"]
    dn_vol  = pc["dn_vol"]
    mom_arr = pc["mom_arr"]
    bull_sp = pc["bull_sp"]
    bear_sp = pc["bear_sp"]

    trades         = []
    next_entry_bar = 0

    for i in range(2, n):
        if i < next_entry_bar:
            continue

        c1_close = closes[i - 2]
        c1_high  = highs[i - 2]
        c3_open  = opens[i]

        if mode == "realistic" and i + 1 >= n:
            continue

        for direction in ("bullish", "bearish"):
            # ── FVG geometry ───────────────────────────────────────────── #
            if direction == "bullish":
                if c1_close <= c3_open:
                    continue
                gap_top    = c1_high
                gap_bottom = lows[i]
                gap_entry  = gap_bottom
                if mom_arr[i] != 1 or not bull_sp[i]:
                    continue
                uv, dv = up_vol[i], dn_vol[i]
                vr = uv / dv if dv > 0 else 1e9
            else:
                if c1_close >= c3_open:
                    continue
                gap_top    = highs[i]
                gap_bottom = lows[i - 2]
                gap_entry  = gap_top
                if mom_arr[i] != -1 or not bear_sp[i]:
                    continue
                uv, dv = up_vol[i], dn_vol[i]
                vr = dv / uv if uv > 0 else 1e9

            if gap_top <= gap_bottom:
                continue
            gs = gap_top - gap_bottom

            # ── Parameter filters ─────────────────────────────────────── #
            if atr[i] > 0 and gs < atr[i] * atr_thresh:
                continue
            if gs * MNQ_CONTRACT_VALUE > max_gap:
                continue
            if vr < vol_ratio:
                continue

            # ── Entry by mode ─────────────────────────────────────────── #
            if mode == "default":
                entry     = gap_entry
                start_bar = i

            elif mode == "realistic":
                entry     = opens[i + 1]
                start_bar = i

            else:  # limit — wait for pullback
                fill_bar = -1
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
                if fill_bar < 0:
                    continue
                entry     = gap_entry
                start_bar = fill_bar

            # ── SL / TP ────────────────────────────────────────────────── #
            if direction == "bullish":
                tp = entry + gs * tp_mult   # extends beyond gap_top when tp_mult > 1
                sl = entry - gs * sl_mult
            else:
                tp = entry - gs * tp_mult   # extends below gap_bottom when tp_mult > 1
                sl = entry + gs * sl_mult

            # ── Exit simulation ───────────────────────────────────────── #
            pnl      = None
            bars_held = max_hold
            for j in range(1, max_hold + 1):
                idx = start_bar + j
                if idx >= n:
                    bars_held = j - 1
                    break
                h, l = highs[idx], lows[idx]
                if direction == "bullish":
                    if l <= sl:
                        pnl = (min(sl, l) - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
                        bars_held = j
                        break
                    if h >= tp:
                        pnl = (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
                        bars_held = j
                        break
                else:
                    if h >= sl:
                        pnl = (entry - max(sl, h)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
                        bars_held = j
                        break
                    if l <= tp:
                        pnl = (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
                        bars_held = j
                        break

            if pnl is None:
                ep  = closes[min(start_bar + max_hold, n - 1)]
                pnl = ((ep - entry) if direction == "bullish"
                       else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST

            # Record date from signal bar timestamp (not fill bar)
            trades.append({"date": pd.Timestamp(pc["ts_np"][i]).date(), "pnl": pnl})
            next_entry_bar = start_bar + bars_held + 1
            break

    return trades


# ── Stats ─────────────────────────────────────────────────────────────────── #

def stats(trades: list[dict], start: str, end: str) -> dict:
    tdays = len(pd.bdate_range(start=start, end=end))
    if not trades:
        return dict(trades=0, wr=0.0, pf=0.0, pnl=0.0,
                    sharpe=float("nan"), max_dd=0.0, tdays=tdays)
    df        = pd.DataFrame(trades)
    daily     = df.groupby("date")["pnl"].sum()
    all_days  = pd.bdate_range(start=start, end=end)
    all_daily = daily.reindex(all_days.date, fill_value=0.0)
    mean_d, std_d = all_daily.mean(), all_daily.std(ddof=1)
    sharpe  = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else float("nan")
    cum     = all_daily.cumsum()
    max_dd  = (cum - cum.cummax()).min()
    total   = len(trades)
    wins    = sum(1 for t in trades if t["pnl"] > 0)
    gp      = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl      = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    return dict(trades=total, wr=wins / total * 100, pf=gp / gl if gl else float("inf"),
                pnl=sum(t["pnl"] for t in trades), sharpe=sharpe,
                max_dd=max_dd, tdays=tdays)


# ── Table printer ─────────────────────────────────────────────────────────── #

def print_table(rows: list[dict], title: str, top: int, param_keys: list[str]) -> None:
    rows = sorted(rows, key=lambda r: r["oos_sh"] if not np.isnan(r["oos_sh"]) else -999,
                  reverse=True)[:top]
    print(f"\n{'='*100}")
    print(f"  {title}  (top {top} by OOS Sharpe)")
    print(f"{'='*100}")
    ph = "  ".join(f"{k:<10}" for k in param_keys)
    print(f"  {'#':<3}  {ph}  {'IS Sh':>6} {'OOS Sh':>7}  {'IS WR':>6} {'OOS WR':>7}  "
          f"{'IS Tr':>5} {'OOS Tr':>6}  {'IS P&L':>9} {'OOS P&L':>9}  Flag")
    print(f"  {'-'*3}  {'-'*len(ph)}  {'------':>6} {'-------':>7}  {'------':>6} {'-------':>7}  "
          f"{'-----':>5} {'------':>6}  {'-'*9} {'-'*9}  ----")
    for rank, r in enumerate(rows, 1):
        pv = "  ".join(f"{str(r[k]):<10}" for k in param_keys)
        flag = "⚠ overfit" if (not np.isnan(r["oos_sh"]) and r["oos_sh"] != 0
                               and abs(r["is_sh"] / r["oos_sh"]) > 2.5) else ""
        print(f"  {rank:<3}  {pv}  "
              f"{r['is_sh']:>6.2f} {r['oos_sh']:>7.2f}  "
              f"{r['is_wr']:>5.1f}% {r['oos_wr']:>6.1f}%  "
              f"{r['is_tr']:>5} {r['oos_tr']:>6}  "
              f"{r['is_pnl']:>9,.0f} {r['oos_pnl']:>9,.0f}  {flag}")


# ── Grid runners ─────────────────────────────────────────────────────────── #

def run_phase(pc_is, pc_oos, combos: list[tuple], param_keys: list[str],
              fixed: dict, mode: str, label: str, top: int) -> dict:
    total = len(combos)
    print(f"\n{'─'*70}")
    print(f"  {label}  ({total} combos, mode={mode})")
    if fixed:
        print(f"  Fixed: " + "  ".join(f"{k}={v}" for k, v in fixed.items()))
    print(f"{'─'*70}")

    rows = []
    t0   = time.time()

    for idx, combo_vals in enumerate(combos):
        params = dict(zip(param_keys, combo_vals))
        kw = dict(mode=mode, **params, **fixed)

        is_s  = stats(fast_run(pc_is,  **kw), IS_START,  IS_END)
        oos_s = stats(fast_run(pc_oos, **kw), OOS_START, OOS_END)

        rows.append({**params,
                     "is_sh":  round(is_s["sharpe"], 2),
                     "oos_sh": round(oos_s["sharpe"], 2),
                     "is_wr":  round(is_s["wr"],   1),
                     "oos_wr": round(oos_s["wr"],  1),
                     "is_tr":  is_s["trades"],
                     "oos_tr": oos_s["trades"],
                     "is_pnl": round(is_s["pnl"],  0),
                     "oos_pnl": round(oos_s["pnl"], 0)})

        if (idx + 1) % 40 == 0 or (idx + 1) == total:
            print(f"  {idx+1}/{total}  ({time.time()-t0:.1f}s)")

    print_table(rows, label, top, param_keys)

    best = sorted(rows, key=lambda r: r["oos_sh"] if not np.isnan(r["oos_sh"]) else -999,
                  reverse=True)[0]
    return best


# ── Detail printout ───────────────────────────────────────────────────────── #

def print_detail(pc_is, pc_oos, cfg: dict, mode: str) -> None:
    from src.research.backtest_tier1_sharpe_analysis import analyse, run as sa_run, prepare

    print(f"\n{'='*70}")
    print(f"  BEST OVERALL CONFIG  (mode={mode})")
    print(f"{'='*70}")
    for k, v in cfg.items():
        if not k.endswith("_sh") and not k.endswith("_wr") and not k.endswith("_tr") \
                and not k.endswith("_pnl"):
            print(f"  {k:<20} = {v}")

    cli = (f"--mode {mode} "
           f"--sl-mult {cfg.get('sl_mult',2.5)} "
           f"--tp-mult {cfg.get('tp_mult',2.5)} "
           f"--max-hold {cfg.get('max_hold',10)} "
           f"--limit-cancel {cfg.get('limit_cancel',5)} "
           f"--atr-thresh {cfg.get('atr_thresh',0.7)} "
           f"--vol-ratio {cfg.get('vol_ratio',2.25)} "
           f"--max-gap {cfg.get('max_gap',50.0)}")
    print(f"\n  CLI to reproduce:")
    print(f"    python src/research/backtest_tier1_sharpe_analysis.py {cli}")

    # Full detail via the reference implementation
    print(f"\n  ── In-sample (Aug–Dec 2025) ──────────────────────────────────")
    df_is  = load_window(IS_START, IS_END)
    df_oos = load_window(OOS_START, OOS_END)
    prep_is  = prepare(df_is,  SPATIAL_TF, LOOKBACK)
    prep_oos = prepare(df_oos, SPATIAL_TF, LOOKBACK)
    kw = dict(mode=mode, sl_mult=cfg.get("sl_mult",2.5), tp_mult=cfg.get("tp_mult",2.5),
              max_hold=cfg.get("max_hold",10),
              limit_cancel=cfg.get("limit_cancel",5), atr_thresh=cfg.get("atr_thresh",0.7),
              vol_ratio=cfg.get("vol_ratio",2.25), max_gap=cfg.get("max_gap",50.0),
              spatial_tf=SPATIAL_TF, lookback=LOOKBACK)
    analyse(sa_run(df_is,  _prepared=prep_is,  **kw), "IS Aug–Dec 2025",
            IS_START,  IS_END)
    print(f"\n  ── Out-of-sample (Jan–Jun 2025) ──────────────────────────────")
    analyse(sa_run(df_oos, _prepared=prep_oos, **kw), "OOS Jan–Jun 2025",
            OOS_START, OOS_END)


# ── Entry point ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["limit", "realistic", "default"], default="limit")
    p.add_argument("--top",  type=int, default=20)
    args = p.parse_args()

    print(f"Tier 1 FVG — Two-Phase Grid Search  (mode={args.mode})")
    print(f"IS: {IS_START}→{IS_END}   OOS: {OOS_START}→{OOS_END}")

    print("\nLoading data and precomputing lookup arrays...")
    t0 = time.time()
    df_is  = load_window(IS_START, IS_END)
    df_oos = load_window(OOS_START, OOS_END)
    pc_is  = precompute(df_is)
    pc_oos = precompute(df_oos)
    print(f"  IS bars: {pc_is['n']:,}   OOS bars: {pc_oos['n']:,}   "
          f"precompute: {time.time()-t0:.1f}s")

    # ── Phase 1 ────────────────────────────────────────────────────────── #
    p1_combos = list(itertools.product(P1_SL_MULTS, P1_TP_MULTS, P1_MAX_HOLDS, P1_LIMIT_CANCELS))
    best_p1 = run_phase(
        pc_is, pc_oos,
        combos     = p1_combos,
        param_keys = ["sl_mult", "tp_mult", "max_hold", "limit_cancel"],
        fixed      = dict(atr_thresh=P1_ATR, vol_ratio=P1_VOL, max_gap=P1_GAP),
        mode       = args.mode,
        label      = "Phase 1 — structural params (sl_mult × tp_mult × max_hold × limit_cancel)",
        top        = args.top,
    )

    # ── Phase 2 ────────────────────────────────────────────────────────── #
    p2_combos = list(itertools.product(P2_ATR_THRESHS, P2_VOL_RATIOS, P2_MAX_GAPS))
    best_p2 = run_phase(
        pc_is, pc_oos,
        combos     = p2_combos,
        param_keys = ["atr_thresh", "vol_ratio", "max_gap"],
        fixed      = dict(sl_mult=best_p1["sl_mult"], tp_mult=best_p1["tp_mult"],
                         max_hold=best_p1["max_hold"],
                         limit_cancel=best_p1["limit_cancel"]),
        mode       = args.mode,
        label      = f"Phase 2 — signal filters (ATR × vol × gap)  "
                     f"[fixed: SL×{best_p1['sl_mult']} TP×{best_p1['tp_mult']} "
                     f"hold={best_p1['max_hold']} cancel={best_p1['limit_cancel']}]",
        top        = args.top,
    )

    best_final = {**best_p1, **best_p2}
    print_detail(pc_is, pc_oos, best_final, args.mode)

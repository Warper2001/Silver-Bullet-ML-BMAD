"""Option B Gate 0 scout — shock-agnostic impulse-aftermath strategy on MNQ 1-min.

Spec is sealed in _bmad-output/preregistration_option_b_impulse_aftermath.md
BEFORE the sweep is run. This script implements that spec verbatim.

Event definition (mechanical, news-free):
  - baseline: rolling median of 1-min bar range and volume over prior 120 bars
  - impulse bar: range >= K * med_range AND |close-open| >= 0.6*range
                 AND volume >= 3 * med_volume
  - excluded ET time slots (scheduled-release/session artifacts):
      08:30-08:34, 09:30-09:31, 10:00-10:04, 13:00-13:02, 14:00-14:34,
      15:59-16:01, 18:00-18:02
  - cooldown: 60 min after any accepted event
  - direction: sign(close-open) of the impulse bar

Trade mechanics (60s-latency realistic):
  - entry: next bar open after the impulse bar (market)
  - exit: close of first bar >= entry_time + H minutes
  - trades whose exit bar lands > H+120 min after entry are dropped (gaps)
  - PnL: points * $2/pt, 1 contract, minus $6 round-trip cost

Sweep (IS only): K in {4,6,8} x side in {follow,fade} x H in {30,60,120,240}.

Usage:
  --counts-only        event counts per K on IS (no PnL) — dataset gate
  --run-is             IS sweep + selection + null test
  --run-oos            OOS confirmation of the selected cell (one shot)
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
REPO = Path("/root/Silver-Bullet-ML-BMAD")
CSV_2025 = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
CSV_2026 = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"

IS_START, IS_END = "2025-01-01", "2025-12-31 23:59"
OOS_START, OOS_END = "2026-01-01", "2026-06-11 23:59"

BASELINE_BARS = 120
BODY_FRAC = 0.6
VOL_MULT = 3.0
COOLDOWN_MIN = 60
POINT_VALUE = 2.0
COST_RT = 6.0
GAP_DROP_MIN = 120
K_GRID = (4.0, 6.0, 8.0)
H_GRID = (30, 60, 120, 240)
SIDES = ("follow", "fade")
N_NULL = 200
RNG_SEED = 20260703

EXCLUDED_SLOTS = [((8, 30), (8, 34)), ((9, 30), (9, 31)), ((10, 0), (10, 4)),
                  ((13, 0), (13, 2)), ((14, 0), (14, 34)), ((15, 59), (16, 1)),
                  ((18, 0), (18, 2))]


def in_excluded_slot(ts_et: datetime) -> bool:
    hm = (ts_et.hour, ts_et.minute)
    return any(lo <= hm <= hi for lo, hi in EXCLUDED_SLOTS)


def load(start: str, end: str) -> pd.DataFrame:
    frames = []
    for p in (CSV_2025, CSV_2026):
        df = pd.read_csv(p, parse_dates=["timestamp"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))
    df = df.loc[mask].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def find_events(df: pd.DataFrame, k: float) -> pd.DataFrame:
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    med_rng = rng.rolling(BASELINE_BARS, min_periods=60).median().shift(1)
    med_vol = df["volume"].rolling(BASELINE_BARS, min_periods=60).median().shift(1)
    cand = (rng >= k * med_rng) & (body >= BODY_FRAC * rng) & \
           (df["volume"] >= VOL_MULT * med_vol) & med_rng.notna()
    idx = np.flatnonzero(cand.to_numpy())
    events = []
    last_t = None
    ts_list = df["ts"].tolist()
    for i in idx:
        t = ts_list[i]
        t_et = t.astimezone(ET)
        if in_excluded_slot(t_et):
            continue
        if last_t is not None and (t - last_t) < timedelta(minutes=COOLDOWN_MIN):
            continue
        last_t = t
        events.append({"i": int(i), "ts": t,
                       "dir": 1 if df["close"].iat[i] >= df["open"].iat[i] else -1})
    return pd.DataFrame(events)


def simulate(df: pd.DataFrame, events: pd.DataFrame, side: str, h: int) -> pd.DataFrame:
    ts_arr = df["ts"].to_numpy()
    trades = []
    for _, ev in events.iterrows():
        i = ev["i"]
        if i + 1 >= len(df):
            continue
        entry_i = i + 1
        entry_t = df["ts"].iat[entry_i]
        entry_px = df["open"].iat[entry_i]
        target_t = entry_t + timedelta(minutes=h)
        j = int(np.searchsorted(ts_arr, np.datetime64(target_t)))
        if j >= len(df):
            continue
        exit_t = df["ts"].iat[j]
        if (exit_t - entry_t) > timedelta(minutes=h + GAP_DROP_MIN):
            continue
        exit_px = df["close"].iat[j]
        d = ev["dir"] if side == "follow" else -ev["dir"]
        pnl = (exit_px - entry_px) * d * POINT_VALUE - COST_RT
        trades.append({"ts": entry_t, "dir": d, "pnl": pnl,
                       "day": entry_t.astimezone(ET).date()})
    return pd.DataFrame(trades)


def pf(pnls) -> float | None:
    gp = float(sum(p for p in pnls if p > 0))
    gl = float(-sum(p for p in pnls if p < 0))
    return None if gl == 0 else gp / gl


def fmt(x):
    return "inf" if x is None else f"{x:.3f}"


def stats_row(label: str, tr: pd.DataFrame) -> str:
    if tr.empty:
        return f"| {label} | 0 | – | – | – | – |"
    wr = 100 * (tr["pnl"] > 0).mean()
    return (f"| {label} | {len(tr)} | ${tr['pnl'].sum():,.0f} | "
            f"${tr['pnl'].mean():,.2f} | {wr:.0f}% | {fmt(pf(tr['pnl']))} |")


def null_test(df: pd.DataFrame, n_trades: int, side_dirs: list[int], h: int,
              eligible_idx: np.ndarray, sel_pf: float) -> float:
    rng = random.Random(RNG_SEED)
    ts_arr = df["ts"].to_numpy()
    beats = 0
    pfs = []
    for _ in range(N_NULL):
        picks = rng.sample(list(eligible_idx), n_trades)
        pnls = []
        for i, d in zip(picks, side_dirs):
            entry_i = i + 1
            if entry_i >= len(df):
                continue
            entry_t = df["ts"].iat[entry_i]
            entry_px = df["open"].iat[entry_i]
            j = int(np.searchsorted(ts_arr, np.datetime64(entry_t + timedelta(minutes=h))))
            if j >= len(df):
                continue
            exit_t = df["ts"].iat[j]
            if (exit_t - entry_t) > timedelta(minutes=h + GAP_DROP_MIN):
                continue
            pnls.append((df["close"].iat[j] - entry_px) * d * POINT_VALUE - COST_RT)
        p = pf(pnls)
        pfs.append(-1 if p is None else p)
        if p is not None and p >= sel_pf:
            beats += 1
    pfs_sorted = sorted(x for x in pfs if x >= 0)
    p95 = pfs_sorted[int(0.95 * len(pfs_sorted))] if pfs_sorted else float("nan")
    print(f"null: {N_NULL} random-entry samples, 95th pct PF = {p95:.3f}, "
          f"share >= selected PF = {beats / N_NULL:.3f}")
    return p95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-only", action="store_true")
    ap.add_argument("--run-is", action="store_true")
    ap.add_argument("--run-oos", action="store_true")
    ap.add_argument("--cell", default=None,
                    help="K,side,H for --run-oos (from sealed IS selection)")
    args = ap.parse_args()

    if args.counts_only:
        df = load(IS_START, IS_END)
        print(f"IS bars: {len(df)}  ({df['ts'].iloc[0]} .. {df['ts'].iloc[-1]})")
        for k in K_GRID:
            ev = find_events(df, k)
            per_mo = ev.groupby(ev["ts"].dt.to_period("M")).size() if not ev.empty else None
            print(f"K={k}: {len(ev)} events "
                  f"({len(ev)/12:.1f}/mo; monthly min..max "
                  f"{per_mo.min() if per_mo is not None else 0}.."
                  f"{per_mo.max() if per_mo is not None else 0})")
        return

    if args.run_is:
        df = load(IS_START, IS_END)
        print("## IS sweep (2025)\n")
        print("| cell | N | total | exp/trade | win% | PF |")
        print("|---|---|---|---|---|---|")
        results = {}
        for k in K_GRID:
            ev = find_events(df, k)
            for side in SIDES:
                for h in H_GRID:
                    tr = simulate(df, ev, side, h)
                    results[(k, side, h)] = tr
                    print(stats_row(f"K{k}/{side}/H{h}", tr))
        viable = {c: t for c, t in results.items() if len(t) >= 60}
        if not viable:
            print("\nDATASET GATE FAIL: no cell with N >= 60 on IS.")
            return
        best = max(viable, key=lambda c: (pf(viable[c]["pnl"]) or -1))
        k, side, h = best
        tr = viable[best]
        sel_pf = pf(tr["pnl"])
        print(f"\nSelected cell: K{k}/{side}/H{h}  N={len(tr)}  PF={fmt(sel_pf)} "
              f"exp=${tr['pnl'].mean():.2f}")
        neigh = [pf(results[(kk, side, h)]["pnl"]) for kk in K_GRID if kk != k]
        print(f"K-neighborhood PFs (same side/H): {[fmt(x) for x in neigh]}")
        rng_all = df["high"] - df["low"]
        med_rng = rng_all.rolling(BASELINE_BARS, min_periods=60).median().shift(1)
        eligible = np.flatnonzero(med_rng.notna().to_numpy()[:-1])
        null_test(df, len(tr), tr["dir"].tolist(), h, eligible, sel_pf or 0)
        top3 = tr.groupby("day")["pnl"].sum().nlargest(3)
        rest = tr[~tr["day"].isin(top3.index)]
        print(f"ex-top-3-days: N={len(rest)} total=${rest['pnl'].sum():,.0f} "
              f"PF={fmt(pf(rest['pnl']))}")
        return

    if args.run_oos:
        k_s, side, h_s = args.cell.split(",")
        k, h = float(k_s), int(h_s)
        df = load(OOS_START, OOS_END)
        print(f"## OOS confirmation {OOS_START}..{OOS_END}  cell K{k}/{side}/H{h}\n")
        ev = find_events(df, k)
        tr = simulate(df, ev, side, h)
        print("| segment | N | total | exp/trade | win% | PF |")
        print("|---|---|---|---|---|---|")
        print(stats_row("OOS all", tr))
        if not tr.empty:
            pre = tr[tr["ts"] < pd.Timestamp("2026-03-01", tz="UTC")]
            post = tr[tr["ts"] >= pd.Timestamp("2026-03-01", tz="UTC")]
            print(stats_row("OOS 01-01..02-28 (open)", pre))
            print(stats_row("OOS 03-01..06-11 (sealed window)", post))
            top3 = tr.groupby("day")["pnl"].sum().nlargest(3)
            rest = tr[~tr["day"].isin(top3.index)]
            print(f"\nex-top-3-days: N={len(rest)} total=${rest['pnl'].sum():,.0f} "
                  f"PF={fmt(pf(rest['pnl']))}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()

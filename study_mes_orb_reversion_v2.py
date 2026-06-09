"""
MES ORB Reversion v2 — Wide-ORB + Opposite ORB Level Target Gate 0 Study
Pre-registration: commit 2350bb9 (2026-06-09)
study_mes_orb_reversion_v2.py written AFTER the pre-registration commit — tamper-evident.

Architectural corrections from v1 (commit cfceffb):
  v1 failure: VWAP centerline target (median 3.2pts from entry → min R/R kills all signals)
              + narrow ORB (median 15pts → 10-pt stop inside intrabar noise)
  v2 fix:     opp_orb_level target (opposite ORB boundary, 96% direction sanity)
              + ORB_MIN = 20 pts (ensures opp_orb ≥ ~17pts from entry → R/R ≥ 1.7:1)

Signal: same false-breakout rejection candle at ORB boundary as v1.
Sensitivity grid: stop_pts × orb_min_size (shows v1→v2 progression).
Primary: stop=10pts, orb_min=20pts, target=opp_orb_level.

In-sample: 2025-05-01 → 2026-02-28. Holdout ≥2026-03-01 SEALED.
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore", message="Converting to Period representation")

ES_DATA = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

IS_START = "2025-05-01"
IS_END   = "2026-02-28"

# ── primary spec (frozen by pre-registration 2350bb9) ────────────────────────
STOP_PRIMARY    = 10
ORB_MIN_PRIMARY = 20.0
TARGET_MODE     = "opp_orb_level"   # fixed: opposite ORB level for all runs

# ── sensitivity grids ─────────────────────────────────────────────────────────
STOP_GRID    = [8, 10, 12]
ORB_MIN_GRID = [10, 15, 20, 25]    # shows v1→v2 progression

# ── opening range parameters ──────────────────────────────────────────────────
ORB_MAX_SIZE  = 40.0     # raised from v1's 30 to avoid cutting valid wide-ORB days
ORB_BARS      = {"09:30", "09:35", "09:40"}

# ── confirmation ──────────────────────────────────────────────────────────────
VOL_MULT     = 1.5
VOL_LOOKBACK = 20
MIN_RR_MULT  = 1.5

# ── session / trade ───────────────────────────────────────────────────────────
RTH_START    = "09:30"
RTH_END      = "15:55"
SESS_CLOSE   = "15:55"
HOLD_MAX     = 12
TIME_WIN     = ("09:45", "11:30")

MES_PV       = 5.0

GATE0_EV_MIN   = 0.0
GATE0_PF_MIN   = 1.20
GATE0_STOP_MAX = 150.0
GATE0_WOMO_PNL = -50.0
GATE0_MIN_N    = 20
COMMISSION     = 4.80


def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


def resample_to_5min(rth_1m: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    h5 = rth_1m.resample("5min", closed="left", label="left").agg(agg).dropna(subset=["close"])
    return h5.between_time(RTH_START, RTH_END)


def build_orb_levels(bars5: pd.DataFrame, orb_min: float) -> pd.DataFrame:
    df = bars5.copy()
    df["time_str"] = df.index.strftime("%H:%M")
    df["date"]     = df.index.date
    orb_bars = df[df["time_str"].isin(ORB_BARS)]
    orb_by_day = orb_bars.groupby("date").agg(
        orb_high=("high",  "max"),
        orb_low=("low",   "min"),
        orb_count=("high", "count"),
    ).copy()
    orb_by_day["orb_size"] = orb_by_day["orb_high"] - orb_by_day["orb_low"]
    orb_by_day["orb_valid"] = (
        (orb_by_day["orb_count"] >= 2) &
        (orb_by_day["orb_size"] >= orb_min) &
        (orb_by_day["orb_size"] <= ORB_MAX_SIZE)
    )
    return orb_by_day


def build_session_vwap_5m(bars5: pd.DataFrame) -> pd.DataFrame:
    df = bars5.copy()
    df["date"] = df.index.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = np.nan
    for d, grp in df.groupby("date"):
        if len(grp) < 2:
            continue
        tp = grp["tp"].values; vol = grp["volume"].values.astype(float)
        cv = np.cumsum(tp * vol); cvol = np.cumsum(vol)
        cvol = np.where(cvol == 0, 1e-9, cvol)
        df.loc[grp.index, "vwap"] = cv / cvol
    df["vol_mean20"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    return df


def run_simulation(bars5: pd.DataFrame, orb_by_day: pd.DataFrame,
                   stop_pts: float, pv: float) -> list:
    """
    Simulate MES ORB Reversion v2.
    Target is always opp_orb_level (opposite ORB boundary).
    """
    stop_usd   = stop_pts * pv
    trades     = []
    active     = None
    hold_count = 0
    prev_date  = None

    orb_dict = orb_by_day.to_dict("index")

    hi_arr   = bars5["high"].values
    lo_arr   = bars5["low"].values
    cl_arr   = bars5["close"].values
    vol_arr  = bars5["volume"].values
    vm20_arr = bars5["vol_mean20"].values
    date_arr = bars5["date"].values
    ts_arr   = bars5.index

    for k in range(len(bars5)):
        ts   = ts_arr[k]
        d    = date_arr[k]
        ts_s = ts.strftime("%H:%M")

        if d != prev_date:
            if active is not None:
                ep  = cl_arr[k-1] if k > 0 else cl_arr[k]
                pnl = (ep - active["entry"]) * active["dir"] * pv - COMMISSION
                trades.append({**active, "exit_p": ep, "pnl": pnl,
                                "win": pnl > 0, "reason": "DAYEND"})
                active = None; hold_count = 0
            prev_date = d

        if active is not None:
            hold_count += 1
            hi_k = hi_arr[k]; lo_k = lo_arr[k]
            at_close = ts_s >= SESS_CLOSE
            day_end  = d != active["date"]

            hit_tp   = ((active["dir"] ==  1 and hi_k >= active["tp"]) or
                        (active["dir"] == -1 and lo_k <= active["tp"]))
            hit_stop = ((active["dir"] ==  1 and lo_k <= active["stop"]) or
                        (active["dir"] == -1 and hi_k >= active["stop"]))

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * active["dir"] * pv - COMMISSION
                trades.append({**active, "exit_p": active["tp"],
                                "pnl": pnl, "win": True, "reason": "TP"})
                active = None; hold_count = 0
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * active["dir"] * pv - COMMISSION
                trades.append({**active, "exit_p": active["stop"],
                                "pnl": pnl, "win": False, "reason": "STOP"})
                active = None; hold_count = 0
            elif at_close or hold_count >= HOLD_MAX or day_end:
                ep  = cl_arr[k]
                pnl = (ep - active["entry"]) * active["dir"] * pv - COMMISSION
                rsn = "CLOSE" if at_close else ("DAYEND" if day_end else "TIME")
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0, "reason": rsn})
                active = None; hold_count = 0
            continue

        orb = orb_dict.get(d)
        if orb is None or not orb["orb_valid"]:
            continue

        if not (TIME_WIN[0] <= ts_s <= TIME_WIN[1]):
            continue

        hi_k = hi_arr[k]; lo_k = lo_arr[k]; cl_k = cl_arr[k]
        orb_high = orb["orb_high"]
        orb_low  = orb["orb_low"]

        short_rej = (hi_k >= orb_high) and (cl_k < orb_high)
        long_rej  = (lo_k <= orb_low)  and (cl_k > orb_low)

        if not (short_rej or long_rej):
            continue
        if short_rej and long_rej:
            continue

        vm20 = vm20_arr[k]
        if np.isnan(vm20) or vm20 <= 0 or vol_arr[k] <= VOL_MULT * vm20:
            continue

        direction = -1 if short_rej else 1
        entry     = cl_k
        stop_p    = entry - direction * stop_pts
        target_p  = orb_low if direction == -1 else orb_high

        # direction sanity
        if direction == -1 and target_p >= entry:
            continue
        if direction ==  1 and target_p <= entry:
            continue

        target_pts = abs(target_p - entry)
        if target_pts < MIN_RR_MULT * stop_pts:
            continue

        active = {
            "dir":        direction,
            "entry":      entry,
            "tp":         target_p,
            "stop":       stop_p,
            "stop_pts":   stop_pts,
            "stop_usd":   stop_usd,
            "target_pts": target_pts,
            "rr":         target_pts / stop_pts,
            "orb_high":   orb_high,
            "orb_low":    orb_low,
            "orb_size":   orb["orb_size"],
            "date":       d,
            "month":      ts.to_period("M"),
            "bar_of_day": ts_s,
        }
        hold_count = 0

    if active:
        pnl = (cl_arr[-1] - active["entry"]) * active["dir"] * pv - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades: list, n_days: int) -> dict:
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, worst_mo_pnl=0.0, pnls=np.array([]),
                    mo={}, mo_pnl={}, exit_tp=0, exit_stop=0, exit_time=0,
                    n_long=0, n_short=0, avg_rr=0.0, med_rr=0.0, avg_be_wr=0.0)
    n     = len(trades)
    wins  = sum(t["win"] for t in trades)
    pnls  = np.array([t["pnl"] for t in trades])
    stops = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_w / max(1e-9, gross_l)
    rrs       = np.array([t["rr"] for t in trades])
    be_wrs    = (stops + COMMISSION) / ((rrs + 1) * stops)
    mo: dict = {}; mo_pnl: dict = {}
    for t in trades:
        m = t["month"]
        mo.setdefault(m, [0, 0])
        mo[m][0 if t["win"] else 1] += 1
        mo_pnl.setdefault(m, [])
        mo_pnl[m].append(t["pnl"])
    worst_mo_pnl = min(float(np.mean(v)) for v in mo_pnl.values()) if mo_pnl else 0.0
    return dict(
        n=n, wr=wins/n, freq=n/n_days, avg_pnl=float(pnls.mean()), pf=pf,
        stop_med=float(np.median(stops)), worst_mo_pnl=worst_mo_pnl,
        pnls=pnls, mo=mo, mo_pnl=mo_pnl,
        exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
        exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
        exit_time=sum(1 for t in trades if t["reason"] in ("TIME", "CLOSE", "DAYEND", "END")),
        n_long=sum(1 for t in trades if t["dir"] ==  1),
        n_short=sum(1 for t in trades if t["dir"] == -1),
        avg_rr=float(rrs.mean()), med_rr=float(np.median(rrs)),
        avg_be_wr=float(be_wrs.mean()),
    )


def path_simulation(daily_pnl_1c: list, n_contracts: int, label: str,
                    target: float = 3000, dd_limit: float = -2000,
                    qual_thresh: float = 150, n_days_horizon: int = 30,
                    n_sims: int = 10_000) -> None:
    if not daily_pnl_1c or len(daily_pnl_1c) < 5:
        print(f"\n  Path simulation: insufficient daily P&L data")
        return
    daily = np.array(daily_pnl_1c) * n_contracts
    rng   = np.random.default_rng(42)
    n_pass = 0; n_ruin = 0
    qual_counts = []; final_pnls = []
    for _ in range(n_sims):
        path = rng.choice(daily, size=n_days_horizon, replace=True)
        cum  = np.cumsum(path)
        hwm  = np.maximum.accumulate(cum)
        ruined = False; passed = False; q = 0
        for i, (c, d) in enumerate(zip(cum, cum - hwm)):
            if d <= dd_limit:
                ruined = True; break
            if path[i] >= qual_thresh:
                q += 1
            if c >= target:
                passed = True; break
        if ruined: n_ruin += 1
        elif passed: n_pass += 1
        qual_counts.append(q)
        final_pnls.append(float(cum[-1]) if not ruined else float(dd_limit))

    p_pass  = n_pass  / n_sims
    p_ruin  = n_ruin  / n_sims
    p_open  = 1 - p_pass - p_ruin
    avg_pnl = float(np.mean(final_pnls))
    med_q   = float(np.median(qual_counts))
    ok      = p_ruin < 0.20 and avg_pnl > 0

    print(f"\n{'='*100}")
    print(f"COMBINE-MATH PATH SIMULATION — {label}  ({n_contracts}c, {n_sims:,} paths)")
    print(f"{'='*100}")
    print(f"  P(reach ${target:,.0f} first):      {p_pass:>7.1%}")
    print(f"  P(ruin by ${abs(dd_limit):,.0f} DD):  {p_ruin:>7.1%}  "
          f"{'✅ <20%' if p_ruin < 0.20 else '❌ >=20%'}")
    print(f"  P(open at day 30):           {p_open:>7.1%}")
    print(f"  Median qualifying days:     {med_q:>6.1f}  (need 5)")
    print(f"  Mean final P&L:             ${avg_pnl:>8,.0f}  "
          f"({'✅ positive' if avg_pnl > 0 else '❌ negative'})")
    if ok:
        print(f"\n  ✅ PATH SIMULATION PASS")
    else:
        fails = []
        if p_ruin >= 0.20: fails.append(f"P(ruin)={p_ruin:.1%}")
        if avg_pnl <= 0:   fails.append(f"E[P&L]=${avg_pnl:,.0f}")
        print(f"\n  ❌ PATH SIMULATION FAIL — {'; '.join(fails)}")


def v(flag: bool, label: str, measured: str) -> str:
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<42} [measured: {measured}]"


def main() -> None:
    print("=" * 100)
    print("MES ORB Reversion v2 — Wide-ORB + Opposite ORB Level Target Gate 0 Study")
    print("Pre-registration: commit 2350bb9 (2026-06-09)")
    print("=" * 100)

    print(f"\nLoading ES 1-min data…")
    raw = load_et(ES_DATA)
    raw = raw[~raw.index.duplicated(keep="first")]
    raw = raw[IS_START:IS_END]
    rth_1m = raw.between_time(RTH_START, RTH_END).copy()
    rth_1m["date"] = rth_1m.index.date

    print(f"Building 5-min bars…")
    bars5 = resample_to_5min(rth_1m)
    bars5["date"] = bars5.index.date
    n_days = len(set(bars5["date"].values))
    print(f"  5-min bars: {len(bars5):,}  (~{len(bars5)/n_days:.0f}/day)  |  {n_days} trading days")
    print(f"  In-sample: {bars5.index[0].date()} → {bars5.index[-1].date()}")

    print(f"Computing session VWAP…")
    bars5 = build_session_vwap_5m(bars5)

    # ── ORB size landscape ────────────────────────────────────────────────
    orb_all = build_orb_levels(bars5, orb_min=0.0)
    orb_all_valid = orb_all[orb_all["orb_count"] >= 2]
    print(f"\nORB size landscape across {n_days} trading days:")
    print(f"  Median={orb_all_valid['orb_size'].median():.1f}pts  "
          f"p25={orb_all_valid['orb_size'].quantile(.25):.1f}  "
          f"p50={orb_all_valid['orb_size'].median():.1f}  "
          f"p75={orb_all_valid['orb_size'].quantile(.75):.1f}  "
          f"p95={orb_all_valid['orb_size'].quantile(.95):.1f}pts")
    for cutoff in [10, 15, 20, 25, 30]:
        n_valid = int((orb_all_valid["orb_size"] >= cutoff).sum())
        pct = n_valid / n_days
        print(f"  ORB >= {cutoff:2d}pts:  {n_valid:>4} days  ({pct:.0%} of trading days)")

    # ── signal funnel at primary spec ─────────────────────────────────────
    orb_primary = build_orb_levels(bars5, ORB_MIN_PRIMARY)
    n_valid_days = int(orb_primary["orb_valid"].sum())
    print(f"\nPrimary spec signal funnel (ORB_MIN={ORB_MIN_PRIMARY}pts, stop={STOP_PRIMARY}):")

    bars5["date_s"]   = bars5["date"]
    bars5["time_str"] = bars5.index.strftime("%H:%M")
    bars_m = bars5.join(orb_primary[["orb_high","orb_low","orb_size","orb_valid"]],
                        on="date_s", how="left")
    bars_m["orb_valid"] = bars_m["orb_valid"].fillna(False)

    in_win = bars_m["time_str"].between(TIME_WIN[0], TIME_WIN[1])
    has_orb = bars_m["orb_valid"]
    vol_mean = bars_m["volume"].rolling(20).mean()
    vol_ok = (bars_m["volume"] > VOL_MULT * vol_mean) & vol_mean.notna()

    short_rej = (bars_m["high"] >= bars_m["orb_high"]) & \
                (bars_m["close"] < bars_m["orb_high"]) & has_orb & in_win
    long_rej  = (bars_m["low"]  <= bars_m["orb_low"])  & \
                (bars_m["close"] > bars_m["orb_low"])  & has_orb & in_win
    n_rej     = int((short_rej | long_rej).sum())
    n_after_vol = int(((short_rej | long_rej) & vol_ok).sum())

    print(f"  Valid ORB days (>={ORB_MIN_PRIMARY}–{ORB_MAX_SIZE}pts): {n_valid_days} / {n_days}  "
          f"({n_valid_days/n_days:.0%})")
    print(f"  Post-ORB AM bars in time window:     "
          f"{int((in_win & has_orb).sum()):>5,}  ({int((in_win & has_orb).sum())/n_days:.2f}/day)")
    print(f"  ORB rejection wicks (raw):           "
          f"{n_rej:>5,}  ({n_rej/n_days:.2f}/day)")
    print(f"  After volume spike filter:           "
          f"{n_after_vol:>5,}  ({n_after_vol/n_days:.2f}/day)")
    print(f"  (direction sanity + min R/R applied in simulation)")

    # ── full grid ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"MES ORB REVERSION v2 GATE 0 GRID  (target=opp_orb_level; grid: STOP × ORB_MIN)")
    print(f"{'='*100}")
    print(f"  {'Stp':>4}  {'ORBmin':>7}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
          f"{'PF':>5}  {'AvgP&L':>8}  {'AvgRR':>7}  {'BEven':>6}  {'WrstMo':>9}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*7}  "
          f"{'-'*5}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*9}")

    grid_res: dict = {}
    for stp in STOP_GRID:
        for orb_min in ORB_MIN_GRID:
            orb_d = build_orb_levels(bars5, orb_min)
            t = run_simulation(bars5, orb_d, stp, MES_PV)
            s = summarise(t, n_days)
            grid_res[(stp, orb_min)] = (t, s)
            is_p = (stp == STOP_PRIMARY and orb_min == ORB_MIN_PRIMARY)
            prim = " ◀ PRIMARY" if is_p else ""
            ev_f = "✅" if s["avg_pnl"] > 0 else "❌"
            pf_f = "✅" if s["pf"] >= GATE0_PF_MIN else "❌"
            print(f"  {stp:>4}  {orb_min:>7.0f}  {s['n']:>5}  "
                  f"{s['freq']:>6.2f}/d  "
                  f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
                  f"${s['avg_pnl']:>6.2f}{ev_f}  "
                  f"{s['avg_rr']:>5.2f}:1  "
                  f"{s['avg_be_wr']:>6.1%}  "
                  f"${s['worst_mo_pnl']:>7.2f}{prim}")

    # ── primary deep dive ─────────────────────────────────────────────────
    pt, ps = grid_res[(STOP_PRIMARY, ORB_MIN_PRIMARY)]

    print(f"\n{'='*100}")
    print(f"PRIMARY SPEC DEEP DIVE  "
          f"(STOP={STOP_PRIMARY}pts, ORB_MIN={ORB_MIN_PRIMARY}pts, TARGET=opp_orb_level, "
          f"HOLD={HOLD_MAX}×5min)")
    print(f"{'='*100}")

    stop_usd = STOP_PRIMARY * MES_PV
    gate_wr  = ps["avg_be_wr"] + 0.05 if ps["n"] > 0 else 0.50

    print(f"\n  Funnel:")
    print(f"    5-min RTH bars:         {len(bars5):,}")
    print(f"    Trading days:           {n_days}")
    print(f"    Valid ORB days:         {n_valid_days}  ({n_valid_days/n_days:.0%})")
    print(f"    Signals taken:          {ps['n']}  ({ps['freq']:.3f}/day)")

    if ps["n"] == 0:
        print(f"\n  ⚠️  N=0 — no signals passed all filters.")
        return

    if ps["n"] < GATE0_MIN_N:
        print(f"\n  ⚠️  N={ps['n']} < {GATE0_MIN_N} — insufficient for reliable Gate 0 read.")

    print(f"    Long / Short:           {ps['n_long']} / {ps['n_short']}  "
          f"({ps['n_long']/ps['n']:.0%} / {ps['n_short']/ps['n']:.0%})")
    print(f"    Exit breakdown:         TP={ps['exit_tp']}  "
          f"STOP={ps['exit_stop']}  TIME/CLOSE={ps['exit_time']}")
    print(f"    Avg realized R/R:       {ps['avg_rr']:.2f}:1  (median {ps['med_rr']:.2f}:1)")
    print(f"    Avg breakeven WR:       {ps['avg_be_wr']:.1%}  (gate ≥ {gate_wr:.1%})")

    rrs = np.array([t["rr"] for t in pt])
    print(f"\n  R/R distribution:")
    for lo, hi in [(0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]:
        cnt = int(((rrs >= lo) & (rrs < hi)).sum())
        lbl = f"{lo:.1f}–{hi:.1f}:1" if hi < 99 else f">{lo:.1f}:1"
        print(f"    {lbl:<12}  {cnt:>4} ({cnt/len(rrs):.0%})")

    orb_sizes = np.array([t["orb_size"] for t in pt])
    print(f"\n  ORB size at signal (pts): mean={orb_sizes.mean():.1f}  "
          f"p25={np.percentile(orb_sizes,25):.1f}  "
          f"p50={np.median(orb_sizes):.1f}  "
          f"p75={np.percentile(orb_sizes,75):.1f}")

    print(f"\n  Performance:")
    print(f"    Win rate:               {ps['wr']:.1%}  (avg be={ps['avg_be_wr']:.1%}; gate≥{gate_wr:.1%})")
    print(f"    Profit factor:          {ps['pf']:.3f}")
    print(f"    Avg net P&L:            ${ps['avg_pnl']:.2f}/trade")
    total_pnl = float(ps["pnls"].sum())
    print(f"    Total P&L (1c):         ${total_pnl:,.0f}  over {n_days} days")
    print(f"    Median stop:            ${ps['stop_med']:.0f}/contract")
    print(f"    Worst-month avg:        ${ps['worst_mo_pnl']:.2f}/trade")

    long_t  = [t for t in pt if t["dir"] ==  1]
    short_t = [t for t in pt if t["dir"] == -1]
    print(f"\n  Long vs Short:")
    for lbl2, subset in [("Long (fade ORB low rejection)", long_t),
                          ("Short (fade ORB high rejection)", short_t)]:
        if subset:
            ns  = len(subset)
            ws  = sum(t["win"] for t in subset)
            ps2 = np.array([t["pnl"] for t in subset])
            gw  = sum(p for p in ps2 if p > 0)
            gl  = abs(sum(p for p in ps2 if p < 0))
            pf2 = gw / max(1e-9, gl)
            rrs2 = np.array([t["rr"] for t in subset])
            print(f"    {lbl2:<32}  N={ns:>3}  WR={ws/ns:>6.1%}  "
                  f"PF={pf2:>5.2f}  AvgRR={rrs2.mean():.2f}:1  Avg=${ps2.mean():>7.2f}")
        else:
            print(f"    {lbl2:<32}  N=0")

    print(f"\n  By month:")
    print(f"  {'Month':<10}  {'N':>4}  {'WR':>7}  {'AvgP&L':>9}  {'AvgRR':>7}  {'freq/d':>7}  Status")
    for m in sorted(ps["mo"]):
        w, l  = ps["mo"][m]
        n_mo  = w + l
        mwr   = w / n_mo if n_mo else 0
        avg   = float(np.mean(ps["mo_pnl"].get(m, [0])))
        mo_b  = bars5[bars5.index.to_period("M") == m]
        mo_d  = len(set(mo_b["date"].values))
        mo_rr = float(np.mean([t["rr"] for t in pt if t["month"] == m]))
        flag  = ("⚠️ N<3" if n_mo < 3 else
                 "❌ avg<-$50" if avg < GATE0_WOMO_PNL else "✅")
        print(f"  {str(m):<10}  {n_mo:>4}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"{mo_rr:>5.2f}:1  {n_mo/max(1, mo_d):>6.2f}/d  {flag}")

    # time-of-day
    tod = Counter(t["bar_of_day"] for t in pt)
    buckets: dict = {}
    for ts_str, cnt in tod.items():
        h, mi = int(ts_str[:2]), int(ts_str[3:])
        b = f"{h:02d}:{(mi//30)*30:02d}"
        buckets[b] = buckets.get(b, 0) + cnt
    tot = sum(buckets.values())
    print(f"\n  Time-of-day (ET):")
    for b in sorted(buckets):
        cnt = buckets[b]
        bar = "█" * int(cnt / max(buckets.values()) * 20)
        print(f"    {b}  {bar:<20}  {cnt:>4} ({cnt/tot:.0%})")

    # rolling-5-day
    print(f"\n  Rolling-5-day variance:")
    day_pnl: dict = {}
    for t in pt:
        day_pnl.setdefault(t["date"], []).append(t["pnl"])
    day_totals = sorted([(d, sum(v)) for d, v in day_pnl.items()])
    if len(day_totals) >= 5:
        vals = [v for _, v in day_totals]
        r5_min = float(min(sum(vals[i:i+5]) for i in range(len(vals)-4)))
        r5_max = float(max(sum(vals[i:i+5]) for i in range(len(vals)-4)))
        print(f"    Worst 5-day: ${r5_min:,.0f}  |  Best 5-day: ${r5_max:,.0f}")
    else:
        print(f"    (fewer than 5 active days)")

    # ── Gate 0 verdict ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"GATE 0 VERDICT — MES ORB REVERSION v2  PRIMARY (STOP={STOP_PRIMARY}, ORB_MIN={ORB_MIN_PRIMARY})")
    print(f"{'='*100}")

    n_ok   = ps["n"] >= GATE0_MIN_N
    ev_ok  = ps["avg_pnl"] > GATE0_EV_MIN
    pf_ok  = ps["pf"] >= GATE0_PF_MIN
    wr_ok  = ps["wr"] >= gate_wr
    stp_ok = stop_usd <= GATE0_STOP_MAX
    wom_ok = ps["worst_mo_pnl"] >= GATE0_WOMO_PNL

    print(v(ev_ok,  "EV > $0",                         f"${ps['avg_pnl']:.2f}/trade"))
    print(v(pf_ok,  "Profit factor ≥ 1.20",            f"{ps['pf']:.3f}"))
    print(v(wr_ok,  f"WR ≥ avg_be_wr+5% (≥{gate_wr:.1%})", f"{ps['wr']:.1%}"))
    print(v(stp_ok, "Median stop ≤ $150/contract",     f"${stop_usd:.0f} (fixed)"))
    print(v(n_ok,   f"N ≥ {GATE0_MIN_N}",               f"N={ps['n']}"))
    print(v(wom_ok, "Worst-month avg ≥ −$50",           f"${ps['worst_mo_pnl']:.2f}/trade"))

    all_pass = all([ev_ok, pf_ok, wr_ok, stp_ok, n_ok, wom_ok])
    print()
    if all_pass:
        print(f"✅ GATE 0 PASS — proceeding to combine-math path simulation.")
    else:
        fails = [f for f, ok in [("EV<0", not ev_ok), (f"PF={ps['pf']:.3f}<1.20", not pf_ok),
                                   (f"WR={ps['wr']:.1%}<{gate_wr:.1%}", not wr_ok),
                                   (f"N={ps['n']}<{GATE0_MIN_N}", not n_ok),
                                   (f"WorstMo=${ps['worst_mo_pnl']:.2f}<-$50", not wom_ok)] if ok]
        print(f"❌ GATE 0 FAIL  ({', '.join(fails)})")

    # ── path sim ──────────────────────────────────────────────────────────
    if ev_ok and pf_ok:
        day_pnl_list = [sum(v) for _, v in day_pnl.items()]
        path_simulation(day_pnl_list, n_contracts=3,
                        label=f"MES ORB Rev v2 (STOP={STOP_PRIMARY}, ORB_MIN={ORB_MIN_PRIMARY})")
    else:
        print(f"\n  Path simulation skipped — EV/PF gates not cleared.")


if __name__ == "__main__":
    main()

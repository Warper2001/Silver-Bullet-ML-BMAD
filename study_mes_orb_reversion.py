"""
MES ORB Reversion — 5-min False-Breakout Fade Gate 0 Study
Pre-registration: commit db42c38 (2026-06-09)
study_mes_orb_reversion.py written AFTER the pre-registration commit — tamper-evident.

Thesis: When ES/MES price breaks the opening range high or low on a 5-min bar but
CLOSES BACK INSIDE the range (rejection candle), the breakout has failed and a
mean-reversion trade back to the session VWAP centerline has positive expectancy.

This applies the same false-breakout architecture as HCVWAP v2 (commit fb8d094) to the
ORB boundary rather than the VWAP σ band. First study targeting MES as primary instrument.

Entry pattern:
  Short (fade ORB high rejection):
    bar.HIGH >= ORB_HIGH  AND  bar.CLOSE < ORB_HIGH
  Long (fade ORB low rejection):
    bar.LOW  <= ORB_LOW   AND  bar.CLOSE > ORB_LOW

Opening range = first 15 min of RTH (09:30–09:44 ET → 3 five-min bars).
ORB size filter: 5 ≤ ORB_SIZE ≤ 30 pts (skip degenerate/gap days).

Instrument: MES ($5/pt, using ES price series).
In-sample: 2025-05-01 → 2026-02-28. Holdout ≥2026-03-01 SEALED.
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore", message="Converting to Period representation")

# ── data paths ────────────────────────────────────────────────────────────────
ES_DATA = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

IS_START = "2025-05-01"
IS_END   = "2026-02-28"

# ── primary spec (frozen by pre-registration db42c38) ────────────────────────
STOP_PRIMARY   = 10
TARGET_PRIMARY = "vwap_centerline"
STOP_GRID      = [8, 10, 12]
TARGET_GRID    = ["vwap_centerline", "opp_orb_level"]

# ── opening range parameters ──────────────────────────────────────────────────
ORB_MINUTES   = 15       # 09:30–09:44 ET → 3 × 5-min bars
ORB_MIN_SIZE  = 5.0      # skip days with ORB < 5 pts (flat open)
ORB_MAX_SIZE  = 30.0     # skip days with ORB > 30 pts (gap/extreme vol)
ORB_BARS      = {"09:30", "09:35", "09:40"}   # 5-min bar timestamps inside ORB

# ── confirmation parameters ───────────────────────────────────────────────────
VOL_MULT      = 1.5
VOL_LOOKBACK  = 20       # 20 × 5-min bars = 100 min rolling window
MIN_RR_MULT   = 1.5      # skip trade if target_pts < 1.5 × stop_pts

# ── session / trade management ────────────────────────────────────────────────
RTH_START     = "09:30"
RTH_END       = "15:55"
SESS_CLOSE    = "15:55"
HOLD_MAX      = 12       # 12 × 5-min = 60 min max hold
TIME_WIN      = ("09:45", "11:30")   # AM window only, post-ORB

# ── instrument ────────────────────────────────────────────────────────────────
MES_PV        = 5.0      # $5 per point, Micro E-mini S&P 500

# ── gate thresholds ───────────────────────────────────────────────────────────
GATE0_EV_MIN   = 0.0
GATE0_PF_MIN   = 1.20
GATE0_STOP_MAX = 150.0
GATE0_WOMO_PNL = -50.0
GATE0_MIN_N    = 20
COMMISSION     = 4.80


def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


def resample_to_5min(rth_1m: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    h5 = rth_1m.resample("5min", closed="left", label="left").agg(agg)
    h5 = h5.dropna(subset=["close"])
    h5 = h5.between_time(RTH_START, RTH_END)
    return h5


def build_orb_levels(bars5: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ORB high/low/size per trading day.
    ORB period = first 15 min of RTH = bars at 09:30, 09:35, 09:40 (on 5-min resolution).
    Returns a DataFrame indexed by date with columns: orb_high, orb_low, orb_size, orb_valid.
    """
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
        (orb_by_day["orb_count"] >= 2) &          # at least 2 of 3 ORB bars present
        (orb_by_day["orb_size"] >= ORB_MIN_SIZE) &
        (orb_by_day["orb_size"] <= ORB_MAX_SIZE)
    )
    return orb_by_day


def build_session_vwap_5m(bars5: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP on 5-min bars, reset daily. Adds: vwap, date columns."""
    df = bars5.copy()
    df["date"] = df.index.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = np.nan

    for d, grp in df.groupby("date"):
        if len(grp) < 2:
            continue
        idx  = grp.index
        tp   = grp["tp"].values
        vol  = grp["volume"].values.astype(float)
        cv   = np.cumsum(tp * vol)
        cvol = np.cumsum(vol)
        cvol = np.where(cvol == 0, 1e-9, cvol)
        df.loc[idx, "vwap"] = cv / cvol

    df["vol_mean20"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    return df


def run_simulation(bars5: pd.DataFrame, orb_by_day: pd.DataFrame,
                   stop_pts: float, target_mode: str, pv: float) -> list:
    """
    Simulate MES ORB Reversion on 5-min bars.

    Signal (all required):
      1. Rejection candle at ORB boundary (false-breakout):
           Short: bar.high >= ORB_HIGH  AND  bar.close < ORB_HIGH
           Long:  bar.low  <= ORB_LOW   AND  bar.close > ORB_LOW
      2. Time window: 09:45–11:30 ET (post-ORB, AM session)
      3. Volume spike: volume > VOL_MULT × 20-bar rolling mean

    Target modes:
      vwap_centerline: session VWAP at entry time (same as HCVWAP v2)
      opp_orb_level:   ORB_LOW for short, ORB_HIGH for long

    Direction sanity: skip if target is on wrong side of entry.
    Min R/R: skip if target_pts < MIN_RR_MULT × stop_pts.
    """
    stop_usd = stop_pts * pv
    trades   = []
    active   = None
    hold_count = 0
    prev_date  = None

    orb_dict = orb_by_day.to_dict("index")

    hi_arr   = bars5["high"].values
    lo_arr   = bars5["low"].values
    cl_arr   = bars5["close"].values
    vw_arr   = bars5["vwap"].values
    vol_arr  = bars5["volume"].values
    vm20_arr = bars5["vol_mean20"].values
    date_arr = bars5["date"].values
    ts_arr   = bars5.index

    for k in range(len(bars5)):
        ts    = ts_arr[k]
        d     = date_arr[k]
        ts_s  = ts.strftime("%H:%M")

        # day boundary
        if d != prev_date:
            if active is not None:
                ep  = cl_arr[k-1] if k > 0 else cl_arr[k]
                pnl = (ep - active["entry"]) * active["dir"] * pv - COMMISSION
                trades.append({**active, "exit_p": ep, "pnl": pnl,
                                "win": pnl > 0, "reason": "DAYEND"})
                active = None; hold_count = 0
            prev_date = d

        # manage active trade
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

        # ── check entry conditions ─────────────────────────────────────────
        # need valid ORB for this day
        orb = orb_dict.get(d)
        if orb is None or not orb["orb_valid"]:
            continue

        orb_high = orb["orb_high"]
        orb_low  = orb["orb_low"]

        # time window: post-ORB AM session only
        if not (TIME_WIN[0] <= ts_s <= TIME_WIN[1]):
            continue

        hi_k = hi_arr[k]; lo_k = lo_arr[k]; cl_k = cl_arr[k]

        # rejection candle at ORB boundary
        short_rej = (hi_k >= orb_high) and (cl_k < orb_high)
        long_rej  = (lo_k <= orb_low)  and (cl_k > orb_low)

        if not (short_rej or long_rej):
            continue
        if short_rej and long_rej:
            continue  # bar pierced both boundaries — degenerate, skip

        # volume spike
        vm20 = vm20_arr[k]
        if np.isnan(vm20) or vm20 <= 0 or vol_arr[k] <= VOL_MULT * vm20:
            continue

        # determine direction and target
        direction = -1 if short_rej else 1
        entry     = cl_k
        stop_p    = entry - direction * stop_pts

        vw = vw_arr[k]
        if target_mode == "vwap_centerline":
            if np.isnan(vw):
                continue
            target_p = vw
        else:  # opp_orb_level
            target_p = orb_low if direction == -1 else orb_high

        # direction sanity: target must be on the correct side of entry
        if direction == -1 and target_p >= entry:
            continue
        if direction ==  1 and target_p <= entry:
            continue

        target_pts = abs(target_p - entry)

        # minimum R/R gate
        if target_pts < MIN_RR_MULT * stop_pts:
            continue

        active = {
            "dir":         direction,
            "entry":       entry,
            "tp":          target_p,
            "stop":        stop_p,
            "stop_pts":    stop_pts,
            "stop_usd":    stop_usd,
            "target_pts":  target_pts,
            "rr":          target_pts / stop_pts,
            "orb_high":    orb_high,
            "orb_low":     orb_low,
            "orb_size":    orb["orb_size"],
            "vwap_at_entry": vw if not np.isnan(vw) else float("nan"),
            "date":        d,
            "month":       ts.to_period("M"),
            "bar_of_day":  ts_s,
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

    rrs    = np.array([t["rr"] for t in trades])
    be_wrs = (stops + COMMISSION) / ((rrs + 1) * stops)
    avg_rr = float(rrs.mean())
    med_rr = float(np.median(rrs))
    avg_be_wr = float(be_wrs.mean())

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
        avg_rr=avg_rr, med_rr=med_rr, avg_be_wr=avg_be_wr,
    )


def path_simulation(daily_pnl_1c: list, n_contracts: int, label: str,
                    target: float = 3000, dd_limit: float = -2000,
                    qual_thresh: float = 150, n_days_horizon: int = 30,
                    n_sims: int = 10_000) -> None:
    if not daily_pnl_1c or len(daily_pnl_1c) < 5:
        print(f"\n  Path simulation ({label}): insufficient daily P&L data")
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
        print(f"\n  ✅ PATH SIMULATION PASS — advance to OOS pre-registration (Gate 2).")
    else:
        fails = []
        if p_ruin >= 0.20: fails.append(f"P(ruin)={p_ruin:.1%}")
        if avg_pnl <= 0:   fails.append(f"E[P&L]=${avg_pnl:,.0f}")
        print(f"\n  ❌ PATH SIMULATION FAIL — {'; '.join(fails)}")


def v(flag: bool, label: str, measured: str) -> str:
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<40} [measured: {measured}]"


def main() -> None:
    print("=" * 100)
    print("MES ORB Reversion — 5-min False-Breakout Fade Gate 0 Study")
    print("Pre-registration: commit db42c38 (2026-06-09)")
    print("=" * 100)

    pv    = MES_PV
    label = "MES ($5/pt, ES price series)"

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

    print(f"Building ORB levels (first 15 min of each RTH session)…")
    orb_by_day = build_orb_levels(bars5)
    n_valid_days = int(orb_by_day["orb_valid"].sum())
    print(f"  Trading days with valid ORB ({ORB_MIN_SIZE}–{ORB_MAX_SIZE} pts): "
          f"{n_valid_days} / {n_days}  ({n_valid_days/n_days:.0%})")
    print(f"  ORB size — mean: {orb_by_day['orb_size'].mean():.1f} pts  "
          f"p25: {orb_by_day['orb_size'].quantile(0.25):.1f}  "
          f"p50: {orb_by_day['orb_size'].quantile(0.50):.1f}  "
          f"p75: {orb_by_day['orb_size'].quantile(0.75):.1f}  "
          f"p95: {orb_by_day['orb_size'].quantile(0.95):.1f} pts")

    # ── signal funnel ─────────────────────────────────────────────────────
    print(f"\nSignal funnel (primary spec, stop={STOP_PRIMARY}, target={TARGET_PRIMARY}):")
    bars5["date_s"] = bars5["date"]
    bars5["time_str"] = bars5.index.strftime("%H:%M")

    # merge ORB levels onto bars5
    bars_orb = bars5.join(orb_by_day[["orb_high", "orb_low", "orb_size", "orb_valid"]],
                          on="date_s", how="left")
    bars_orb["orb_valid"] = bars_orb["orb_valid"].fillna(False)

    in_window = bars_orb["time_str"].between(TIME_WIN[0], TIME_WIN[1])
    has_orb   = bars_orb["orb_valid"]

    short_rej_mask = (bars_orb["high"] >= bars_orb["orb_high"]) & \
                     (bars_orb["close"] < bars_orb["orb_high"]) & has_orb & in_window
    long_rej_mask  = (bars_orb["low"]  <= bars_orb["orb_low"]) & \
                     (bars_orb["close"] > bars_orb["orb_low"]) & has_orb & in_window
    n_raw_rej = int((short_rej_mask | long_rej_mask).sum())

    vm20 = bars5["vol_mean20"]
    vol_ok = (bars5["volume"] > VOL_MULT * vm20) & vm20.notna()
    n_after_vol = int(((short_rej_mask | long_rej_mask) & vol_ok).sum())

    print(f"  Post-ORB AM bars (09:45–11:30) with valid ORB:   "
          f"{int((in_window & has_orb).sum()):>6,}  ({int((in_window & has_orb).sum())/n_days:.2f}/day)")
    print(f"  ORB rejection wicks (raw, pre-vol):               "
          f"{n_raw_rej:>6,}  ({n_raw_rej/n_days:.2f}/day)")
    print(f"  After volume spike filter:                        "
          f"{n_after_vol:>6,}  ({n_after_vol/n_days:.2f}/day)")
    print(f"  (direction sanity + min R/R applied in simulation)")

    # ── grid ─────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"MES ORB REVERSION GATE 0 GRID  (grid: STOP_PTS × TARGET)")
    print(f"{'='*100}")
    print(f"  {'Stp':>4}  {'Target':<18}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
          f"{'PF':>5}  {'AvgP&L':>8}  {'AvgRR':>7}  {'BEven':>6}  {'WrstMo':>9}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*5}  {'-'*8}  {'-'*7}  "
          f"{'-'*5}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*9}")

    grid_res: dict = {}
    for stp in STOP_GRID:
        for tgt in TARGET_GRID:
            t = run_simulation(bars5, orb_by_day, stp, tgt, pv)
            s = summarise(t, n_days)
            grid_res[(stp, tgt)] = (t, s)
            is_p = (stp == STOP_PRIMARY and tgt == TARGET_PRIMARY)
            prim = " ◀ PRIMARY" if is_p else ""
            ev_f = "✅" if s["avg_pnl"] > 0 else "❌"
            pf_f = "✅" if s["pf"] >= GATE0_PF_MIN else "❌"
            print(f"  {stp:>4}  {tgt:<18}  {s['n']:>5}  "
                  f"{s['freq']:>6.2f}/d  "
                  f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
                  f"${s['avg_pnl']:>6.2f}{ev_f}  "
                  f"{s['avg_rr']:>5.2f}:1  "
                  f"{s['avg_be_wr']:>6.1%}  "
                  f"${s['worst_mo_pnl']:>7.2f}{prim}")

    # ── primary deep dive ─────────────────────────────────────────────────
    pt, ps = grid_res[(STOP_PRIMARY, TARGET_PRIMARY)]

    print(f"\n{'='*100}")
    print(f"PRIMARY SPEC DEEP DIVE  "
          f"(STOP={STOP_PRIMARY}pts, TARGET={TARGET_PRIMARY}, "
          f"HOLD={HOLD_MAX}×5min, WIN={TIME_WIN[0]}–{TIME_WIN[1]} ET)")
    print(f"{'='*100}")

    stop_usd = STOP_PRIMARY * pv
    gate_wr  = ps["avg_be_wr"] + 0.05 if ps["n"] > 0 else 0.50

    print(f"\n  Funnel:")
    print(f"    5-min RTH bars:         {len(bars5):,}")
    print(f"    Trading days:           {n_days}")
    print(f"    Valid ORB days:         {n_valid_days}  ({n_valid_days/n_days:.0%})")
    print(f"    Signals taken:          {ps['n']}  ({ps['freq']:.3f}/day)")

    if ps["n"] == 0:
        print(f"\n  ⚠️  N=0 — no trades generated. Check ORB/vol filter or time window.")
        return

    if ps["n"] < GATE0_MIN_N:
        print(f"\n  ⚠️  N={ps['n']} < {GATE0_MIN_N} — insufficient trades for reliable Gate 0 read.")

    print(f"    Long / Short:           {ps['n_long']} / {ps['n_short']}  "
          f"({ps['n_long']/ps['n']:.0%} / {ps['n_short']/ps['n']:.0%})")
    print(f"    Exit breakdown:         TP={ps['exit_tp']}  "
          f"STOP={ps['exit_stop']}  TIME/CLOSE={ps['exit_time']}")
    print(f"    Avg realized R/R:       {ps['avg_rr']:.2f}:1  (median {ps['med_rr']:.2f}:1)")
    print(f"    Avg breakeven WR:       {ps['avg_be_wr']:.1%}  (gate ≥ {gate_wr:.1%})")

    # R/R distribution
    rrs = np.array([t["rr"] for t in pt])
    print(f"\n  R/R distribution:")
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 99)]:
        cnt = int(((rrs >= lo) & (rrs < hi)).sum())
        lbl = f"{lo:.1f}–{hi:.1f}:1" if hi < 99 else f">{lo:.1f}:1"
        print(f"    {lbl:<12}  {cnt:>4} trades  ({cnt/len(rrs):.0%})")

    # ORB size distribution for triggered signals
    if pt:
        orb_sizes = np.array([t["orb_size"] for t in pt])
        print(f"\n  ORB size at signal (pts):")
        print(f"    mean={orb_sizes.mean():.1f}  p25={np.percentile(orb_sizes,25):.1f}  "
              f"p50={np.median(orb_sizes):.1f}  p75={np.percentile(orb_sizes,75):.1f}")

    print(f"\n  Performance:")
    print(f"    Win rate:               {ps['wr']:.1%}  (avg be={ps['avg_be_wr']:.1%}; gate≥{gate_wr:.1%})")
    print(f"    Profit factor:          {ps['pf']:.3f}")
    print(f"    Avg net P&L:            ${ps['avg_pnl']:.2f}/trade")
    total_pnl = float(ps["pnls"].sum())
    print(f"    Total P&L (1c):         ${total_pnl:,.0f}  over {n_days} days")
    print(f"    Median stop:            ${ps['stop_med']:.0f}/contract")
    print(f"    Worst-month avg:        ${ps['worst_mo_pnl']:.2f}/trade")

    # Long vs short breakdown
    long_t  = [t for t in pt if t["dir"] ==  1]
    short_t = [t for t in pt if t["dir"] == -1]
    print(f"\n  Long vs Short breakdown:")
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
            print(f"    {lbl2:<32}  N={ns:>4}  WR={ws/ns:>6.1%}  "
                  f"PF={pf2:>5.2f}  AvgRR={rrs2.mean():.2f}:1  Avg=${ps2.mean():>7.2f}")
        else:
            print(f"    {lbl2:<32}  N=0")

    # By month
    print(f"\n  By month:")
    print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'AvgRR':>7}  {'freq/d':>7}  Status")
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
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"{mo_rr:>5.2f}:1  {n_mo/max(1, mo_d):>6.2f}/d  {flag}")

    # Time-of-day
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

    # Victor's rolling-5-day variance
    print(f"\n  Victor's rolling-5-day variance check:")
    day_pnl: dict = {}
    for t in pt:
        day_pnl.setdefault(t["date"], []).append(t["pnl"])
    day_totals = sorted([(d, sum(v)) for d, v in day_pnl.items()])
    if len(day_totals) >= 5:
        vals = [v for _, v in day_totals]
        r5_min = float(min(sum(vals[i:i+5]) for i in range(len(vals)-4)))
        r5_max = float(max(sum(vals[i:i+5]) for i in range(len(vals)-4)))
        print(f"    Worst 5-day window: ${r5_min:,.0f}  |  Best 5-day: ${r5_max:,.0f}")
    else:
        print(f"    (fewer than 5 active days — rolling window not meaningful)")

    # ── Gate 0 verdict ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"GATE 0 VERDICT — MES ORB REVERSION  PRIMARY (STOP={STOP_PRIMARY}, TARGET={TARGET_PRIMARY})")
    print(f"{'='*100}")

    n_ok   = ps["n"] >= GATE0_MIN_N
    ev_ok  = ps["avg_pnl"] > GATE0_EV_MIN
    pf_ok  = ps["pf"] >= GATE0_PF_MIN
    wr_ok  = ps["wr"] >= gate_wr
    stp_ok = stop_usd <= GATE0_STOP_MAX
    wom_ok = ps["worst_mo_pnl"] >= GATE0_WOMO_PNL

    print(v(ev_ok,  "EV > $0",                        f"${ps['avg_pnl']:.2f}/trade"))
    print(v(pf_ok,  "Profit factor ≥ 1.20",           f"{ps['pf']:.3f}"))
    print(v(wr_ok,  f"WR ≥ avg_be_wr+5% (≥{gate_wr:.1%})",
            f"{ps['wr']:.1%}"))
    print(v(stp_ok, "Median stop ≤ $150/contract",    f"${stop_usd:.0f} (fixed)"))
    print(v(n_ok,   f"N ≥ {GATE0_MIN_N}",              f"N={ps['n']}"))
    print(v(wom_ok, "Worst-month avg ≥ −$50",          f"${ps['worst_mo_pnl']:.2f}/trade"))

    all_pass = all([ev_ok, pf_ok, wr_ok, stp_ok, n_ok, wom_ok])
    print()
    if all_pass:
        print(f"✅ GATE 0 PASS — MES ORB Reversion primary spec clears all gates.")
        print(f"   → Proceeding to combine-math path simulation.")
    else:
        fails = []
        if not ev_ok:  fails.append("EV<$0")
        if not pf_ok:  fails.append(f"PF={ps['pf']:.3f}<1.20")
        if not wr_ok:  fails.append(f"WR={ps['wr']:.1%}<{gate_wr:.1%}")
        if not n_ok:   fails.append(f"N={ps['n']}<{GATE0_MIN_N}")
        if not wom_ok: fails.append(f"WorstMo=${ps['worst_mo_pnl']:.2f}<-$50")
        print(f"❌ GATE 0 FAIL — MES ORB Reversion primary spec  ({', '.join(fails)})")

    # ── path simulation (only if edge gates pass) ─────────────────────────
    if ev_ok and pf_ok:
        day_pnl_list = [sum(v) for _, v in day_pnl.items()]
        path_simulation(day_pnl_list, n_contracts=3,
                        label=f"MES ORB Reversion (STOP={STOP_PRIMARY}, {TARGET_PRIMARY})")
    else:
        print(f"\n  Path simulation skipped — edge gates (EV/PF) not cleared.")


if __name__ == "__main__":
    main()

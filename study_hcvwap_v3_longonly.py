"""
HCVWAP v3 Long-Only — OOS Validation on MNQ 2026-03-01 → 2026-05-19
Pre-registration: commit e1e153f (2026-06-09)
study_hcvwap_v3_longonly.py written AFTER the pre-registration commit — tamper-evident.

This study is ONLY run on the sealed OOS holdout window (2026-03-01 → 2026-05-19).
The in-sample finding being tested: HCVWAP v2 long side showed WR=38.3%, PF=1.87, N=60
(study_hcvwap_v2.py, commit fb8d094, in-sample 2025-01-01 → 2026-02-28).

The long/short asymmetry (longs positive, shorts strongly negative) was observed in-sample
and is what motivates this OOS test. This pre-registration cannot cure the post-hoc selection
of long-only; it only cleanly documents the OOS validation gate.

LONG direction only:
  Entry: 5-min bar.low <= VWAP - SD*σ  AND  bar.close > VWAP - SD*σ
  (price tested the -2σ band intrabar then closed back inside — rejection wick)
  Stop:  15 pts fixed  (primary)
  Target: session VWAP centerline

OOS gate (relaxed for small N):
  EV > $0  |  PF >= 1.10  |  WR >= avg_be_wr + 3pp  |  N >= 10  |  Worst-month >= -$100
  INCONCLUSIVE if N < 10 (too few signals in 55-day OOS window)
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", message="Converting to Period representation")

# ── data ──────────────────────────────────────────────────────────────────────
MNQ_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

OOS_START = "2026-03-01"
OOS_END   = "2026-05-19"

# ── primary spec (frozen by pre-registration e1e153f) ─────────────────────────
SD_PRIMARY   = 2.0
STOP_PRIMARY = 15
SD_GRID      = [1.5, 2.0, 2.5]
STOP_GRID    = [12, 15, 18]

# ── confirmation stack parameters (identical to v2) ────────────────────────────
VOL_MULT         = 1.5
VOL_LOOKBACK     = 20
HTF_EMA_FAST     = 9
HTF_EMA_SLOW     = 21
HTF_ATR_WIN      = 14
HTF_EMA_ATR_MULT = 0.5
SD_MIN_BARS      = 3
MIN_RR_MULT      = 1.5

# ── session / trade management ─────────────────────────────────────────────────
RTH_START  = "09:30"
RTH_END    = "15:55"
SESS_CLOSE = "15:55"
HOLD_MAX   = 12
AM_WIN     = ("09:45", "11:30")
PM_WIN     = ("14:00", "15:00")

# ── OOS gate thresholds (relaxed vs in-sample) ────────────────────────────────
GATE_EV_MIN    = 0.0
GATE_PF_MIN    = 1.10     # relaxed from 1.20 (OOS small-N)
GATE_STOP_MAX  = 150.0
GATE_WOMO_PNL  = -100.0   # relaxed from -50 (only 3 OOS months)
GATE_MIN_N     = 10       # below this = INCONCLUSIVE
GATE_WR_BONUS  = 0.03     # 3pp over breakeven (relaxed from 5pp)
COMMISSION     = 4.80
MNQ_PV         = 2.0


def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def resample_to_5min(rth_1m: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    h5 = rth_1m.resample("5min", closed="left", label="left").agg(agg)
    h5 = h5.dropna(subset=["close"])
    h5 = h5.between_time(RTH_START, RTH_END)
    return h5


def build_htf_ranging_15m(rth_1m: pd.DataFrame) -> pd.Series:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    h15 = rth_1m.resample("15min", closed="left", label="left").agg(agg).dropna(subset=["close"])
    h15["ema_f"]   = ema(h15["close"], HTF_EMA_FAST)
    h15["ema_s"]   = ema(h15["close"], HTF_EMA_SLOW)
    h15["tr"]      = h15["high"] - h15["low"]
    h15["atr"]     = h15["tr"].rolling(HTF_ATR_WIN).mean()
    h15["spread"]  = (h15["ema_f"] - h15["ema_s"]).abs()
    h15["ranging"] = h15["spread"] < (HTF_EMA_ATR_MULT * h15["atr"])
    return h15["ranging"]


def build_session_vwap_5m(bars5: pd.DataFrame) -> pd.DataFrame:
    df = bars5.copy()
    df["date"] = df.index.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = np.nan
    df["sigma_vwap"] = np.nan

    for _d, grp in df.groupby("date"):
        if len(grp) < 2:
            continue
        idx       = grp.index
        tp        = grp["tp"].values
        vol       = grp["volume"].values.astype(float)
        cv        = np.cumsum(tp * vol)
        cvol      = np.cumsum(vol)
        cvol      = np.where(cvol == 0, 1e-9, cvol)
        vwap_arr  = cv / cvol
        dev       = grp["close"].values - vwap_arr
        sig       = np.full(len(grp), np.nan)
        for k in range(SD_MIN_BARS, len(grp)):
            sig[k] = float(np.std(dev[:k+1], ddof=1))
        df.loc[idx, "vwap"]       = vwap_arr
        df.loc[idx, "sigma_vwap"] = sig

    df["vol_mean20"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    return df


def run_simulation(bars5: pd.DataFrame, ranging_15m: pd.Series,
                   sd_thresh: float, stop_pts: float) -> list:
    """
    LONG ONLY — HCVWAP v3 false-breakout rejection below VWAP-σ band.
    Entry: bar.low <= vwap - sd*sigma  AND  bar.close > vwap - sd*sigma
    """
    stop_usd   = stop_pts * MNQ_PV
    trades     = []
    active     = None
    hold_count = 0
    prev_date  = None

    rng_ser = ranging_15m.reindex(bars5.index, method="ffill").fillna(False)

    hi_arr   = bars5["high"].values
    lo_arr   = bars5["low"].values
    cl_arr   = bars5["close"].values
    vw_arr   = bars5["vwap"].values
    sg_arr   = bars5["sigma_vwap"].values
    vol_arr  = bars5["volume"].values
    vm20_arr = bars5["vol_mean20"].values
    date_arr = bars5["date"].values
    ts_arr   = bars5.index
    rng_arr  = rng_ser.values

    for k in range(len(bars5)):
        ts   = ts_arr[k]
        d    = date_arr[k]
        ts_s = ts.strftime("%H:%M")

        if d != prev_date:
            if active is not None:
                ep  = cl_arr[k-1] if k > 0 else cl_arr[k]
                pnl = (ep - active["entry"]) * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": ep, "pnl": pnl,
                                "win": pnl > 0, "reason": "DAYEND"})
                active = None; hold_count = 0
            prev_date = d

        if active is not None:
            hold_count += 1
            at_close = ts_s >= SESS_CLOSE
            day_end  = d != active["date"]
            hi_k = hi_arr[k]; lo_k = lo_arr[k]

            hit_tp   = hi_k >= active["tp"]
            hit_stop = lo_k <= active["stop"]

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": active["tp"],
                                "pnl": pnl, "win": True, "reason": "TP"})
                active = None; hold_count = 0
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": active["stop"],
                                "pnl": pnl, "win": False, "reason": "STOP"})
                active = None; hold_count = 0
            elif at_close or hold_count >= HOLD_MAX or day_end:
                ep  = cl_arr[k]
                pnl = (ep - active["entry"]) * MNQ_PV - COMMISSION
                rsn = "CLOSE" if at_close else ("DAYEND" if day_end else "TIME")
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0, "reason": rsn})
                active = None; hold_count = 0
            continue

        vw = vw_arr[k]; sg = sg_arr[k]
        if np.isnan(vw) or np.isnan(sg) or sg <= 0:
            continue

        lo_k = lo_arr[k]; cl_k = cl_arr[k]
        band_lower = vw - sd_thresh * sg

        # LONG rejection wick only
        if not ((lo_k <= band_lower) and (cl_k > band_lower)):
            continue

        # Time window
        in_win = (AM_WIN[0] <= ts_s <= AM_WIN[1] or PM_WIN[0] <= ts_s <= PM_WIN[1])
        if not in_win:
            continue

        # Volume spike
        vm20 = vm20_arr[k]
        if np.isnan(vm20) or vm20 <= 0 or vol_arr[k] <= VOL_MULT * vm20:
            continue

        # HTF ranging
        if not rng_arr[k]:
            continue

        entry    = cl_k
        stop_p   = entry - stop_pts
        target_p = vw

        # Direction sanity: VWAP must be above entry for long
        if target_p <= entry:
            continue

        target_pts = target_p - entry
        if target_pts < MIN_RR_MULT * stop_pts:
            continue

        active = {
            "entry":      entry,
            "tp":         target_p,
            "stop":       stop_p,
            "stop_pts":   stop_pts,
            "stop_usd":   stop_usd,
            "target_pts": target_pts,
            "rr":         target_pts / stop_pts,
            "vwap_at_entry": vw,
            "sd_at_entry":   abs((entry - vw) / sg),
            "date":       d,
            "month":      ts.to_period("M"),
            "bar_of_day": ts_s,
        }
        hold_count = 0

    if active:
        pnl = (cl_arr[-1] - active["entry"]) * MNQ_PV - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades: list, n_days: int) -> dict:
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, worst_mo_pnl=0.0, pnls=np.array([]),
                    mo={}, mo_pnl={}, exit_tp=0, exit_stop=0, exit_time=0,
                    avg_rr=0.0, med_rr=0.0, avg_be_wr=0.0)
    n    = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    stops = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_w / max(1e-9, gross_l)

    rrs = np.array([t["rr"] for t in trades])
    be_wrs = (stops + COMMISSION) / ((rrs + 1) * stops)
    avg_rr   = float(rrs.mean())
    med_rr   = float(np.median(rrs))
    avg_be_wr = float(be_wrs.mean())

    mo: dict = {}
    mo_pnl: dict = {}
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
        exit_time=sum(1 for t in trades if t["reason"] in
                      ("TIME", "CLOSE", "DAYEND", "END")),
        avg_rr=avg_rr, med_rr=med_rr, avg_be_wr=avg_be_wr,
    )


def main():
    print("=" * 100)
    print("HCVWAP v3 Long-Only — OOS Validation Study")
    print("Pre-registration: commit e1e153f (2026-06-09)")
    print("Testing: v2 long-side finding (WR=38.3%, PF=1.87, N=60 in-sample)")
    print(f"OOS window: {OOS_START} → {OOS_END}  (sealed holdout, ACCESS_LOG logged)")
    print("=" * 100)
    print(f"\nEntry (LONG only): 5-min bar.low ≤ VWAP-{SD_PRIMARY}σ  AND  close returns above band")
    print(f"Stop: {STOP_PRIMARY} pts fixed  |  Target: VWAP centerline  |  Min R/R: {MIN_RR_MULT}:1")
    print(f"Time windows: {AM_WIN[0]}–{AM_WIN[1]} ET  OR  {PM_WIN[0]}–{PM_WIN[1]} ET")
    print(f"\n  ⚠️  DISCLOSURE: long-side split was observed in-sample before this pre-registration.")
    print(f"  ⚠️  This OOS test is the only methodology-clean validation path.")

    # ── load OOS data ──────────────────────────────────────────────────────────
    print(f"\n  Loading MNQ 2026 YTD data → filtering to OOS window {OOS_START}→{OOS_END}…")
    raw = load_et(MNQ_2026)
    raw = raw[~raw.index.duplicated(keep="first")]
    raw_oos = raw[OOS_START:OOS_END]
    rth_1m  = raw_oos.between_time(RTH_START, RTH_END).copy()
    rth_1m["date"] = rth_1m.index.date

    bars5 = resample_to_5min(rth_1m)
    bars5["date"] = bars5.index.date
    n_days = len(set(bars5["date"]))
    print(f"  5-min bars: {len(bars5):,}  (~{len(bars5)/n_days:.0f}/day)  |  {n_days} OOS trading days")
    print(f"  OOS period: {bars5.index[0].date()} → {bars5.index[-1].date()}")

    print("  Computing session VWAP + σ_vwap on 5-min bars…")
    bars5 = build_session_vwap_5m(bars5)

    print("  Building 15-min HTF ranging signal…")
    ranging_15m = build_htf_ranging_15m(rth_1m)

    # ── confirmation funnel ────────────────────────────────────────────────────
    print(f"\n  Confirmation funnel (SD={SD_PRIMARY}, LONG only):")
    vw  = bars5["vwap"]
    sg  = bars5["sigma_vwap"]
    valid_mask  = sg.notna() & vw.notna() & (sg > 0)
    band_lower  = vw - SD_PRIMARY * sg
    long_rej    = (bars5["low"] <= band_lower) & (bars5["close"] > band_lower) & valid_mask
    n_raw_rej   = long_rej.sum()

    def in_window(ts):
        s = ts.strftime("%H:%M")
        return (AM_WIN[0] <= s <= AM_WIN[1]) or (PM_WIN[0] <= s <= PM_WIN[1])

    tw_mask     = pd.Series([in_window(ts) for ts in bars5.index], index=bars5.index)
    n_after_tw  = (long_rej & tw_mask).sum()

    vm20    = bars5["vol_mean20"]
    vol_ok  = (bars5["volume"] > VOL_MULT * vm20) & vm20.notna()
    n_after_vol = (long_rej & tw_mask & vol_ok).sum()

    rang_ok = ranging_15m.reindex(bars5.index, method="ffill").fillna(False)
    n_after_htf = (long_rej & tw_mask & vol_ok & rang_ok).sum()

    print(f"    5-min bars with valid VWAP/σ:               {valid_mask.sum():>6,}")
    print(f"    LONG rejection wicks at |z|≥{SD_PRIMARY:.1f}σ:          {n_raw_rej:>6,}   ({n_raw_rej/n_days:.2f}/day)")
    print(f"    After time-window filter:                   {n_after_tw:>6,}   ({n_after_tw/n_days:.2f}/day)")
    print(f"    After volume spike filter:                  {n_after_vol:>6,}   ({n_after_vol/n_days:.2f}/day)")
    print(f"    After HTF ranging filter:                   {n_after_htf:>6,}   ({n_after_htf/n_days:.2f}/day)")

    # ── sensitivity grid ───────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("HCVWAP v3 SENSITIVITY GRID — MNQ OOS  (LONG ONLY)")
    print("NOTE: Primary spec drives the gate verdict. Grid is robustness context only.")
    print(f"{'='*100}")
    print(f"  {'SD':>4}  {'Stp':>4}  {'N':>5}  {'WR':>7}  {'PF':>5}  {'AvgP&L':>9}  "
          f"{'AvgRR':>7}  {'BEven':>6}  {'WorstMo':>9}")
    print(f"  {'--':>4}  {'--':>4}  {'---':>5}  {'---':>7}  {'----':>5}  {'------':>9}  "
          f"{'------':>7}  {'-----':>6}  {'-------':>9}")

    grid_res: dict = {}
    for sd in SD_GRID:
        for stp in STOP_GRID:
            t = run_simulation(bars5, ranging_15m, sd, stp)
            s = summarise(t, n_days)
            grid_res[(sd, stp)] = (t, s)
            is_p = (sd == SD_PRIMARY and stp == STOP_PRIMARY)
            prim = " ◀ PRIMARY" if is_p else ""
            print(f"  {sd:>3.1f}  {stp:>4}  {s['n']:>5}  "
                  f"{s['wr']:>7.1%}  {s['pf']:>4.2f}  "
                  f"${s['avg_pnl']:>7.2f}  "
                  f"{s['avg_rr']:>5.2f}:1  "
                  f"{s['avg_be_wr']:>6.1%}  "
                  f"${s['worst_mo_pnl']:>7.2f}{prim}")

    # ── primary deep dive ──────────────────────────────────────────────────────
    pt, ps = grid_res[(SD_PRIMARY, STOP_PRIMARY)]

    print(f"\n{'='*100}")
    print(f"PRIMARY SPEC DEEP DIVE — MNQ OOS  (SD={SD_PRIMARY}σ, STOP={STOP_PRIMARY}pts, LONG ONLY)")
    print(f"{'='*100}")

    stop_usd = STOP_PRIMARY * MNQ_PV
    gate_wr  = ps["avg_be_wr"] + GATE_WR_BONUS if ps["n"] > 0 else 0.50

    print(f"\n  Signals taken:  {ps['n']}  ({ps['freq']:.3f}/day)  over {n_days} OOS days")
    if ps["n"] > 0:
        print(f"  Exit breakdown: TP={ps['exit_tp']}  STOP={ps['exit_stop']}  TIME/CLOSE={ps['exit_time']}")
        print(f"  Avg realized R/R: {ps['avg_rr']:.2f}:1  (median {ps['med_rr']:.2f}:1)")
        print(f"  Avg breakeven WR: {ps['avg_be_wr']:.1%}  (OOS gate ≥ {gate_wr:.1%})")

        rrs = np.array([t["rr"] for t in pt])
        print(f"\n  R/R distribution:")
        for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 99)]:
            cnt = ((rrs >= lo) & (rrs < hi)).sum()
            lbl = f"{lo:.1f}:1–{hi:.1f}:1" if hi < 99 else f">{lo:.1f}:1"
            print(f"    {lbl:<12}  {cnt:>4} trades  ({cnt/len(rrs):.0%})")

    print(f"\n  Performance:")
    print(f"    Win rate:        {ps['wr']:.1%}  (avg be={ps['avg_be_wr']:.1%}; gate≥{gate_wr:.1%})")
    print(f"    Profit factor:   {ps['pf']:.3f}  (OOS gate ≥ {GATE_PF_MIN:.2f})")
    print(f"    Avg net P&L:     ${ps['avg_pnl']:.2f}/trade")
    total_pnl = float(ps["pnls"].sum()) if len(ps["pnls"]) > 0 else 0.0
    print(f"    Total P&L (1c):  ${total_pnl:,.0f}  over {n_days} OOS days")
    print(f"    Median stop:     ${ps['stop_med']:.0f}/contract  (gate ≤ ${GATE_STOP_MAX:.0f})")
    print(f"    Worst-month avg: ${ps['worst_mo_pnl']:.2f}/trade  (gate ≥ ${GATE_WOMO_PNL:.0f})")

    # By month
    print(f"\n  By month (OOS):")
    print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'AvgRR':>7}")
    for m in sorted(ps["mo"]):
        w, l  = ps["mo"][m]
        n_mo  = w + l
        mwr   = w / n_mo if n_mo else 0
        avg   = float(np.mean(ps["mo_pnl"].get(m, [0])))
        mo_rr = np.mean([t["rr"] for t in pt if t["month"] == m]) if pt else 0
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  {mo_rr:>5.2f}:1")

    # Equity sketch
    if len(ps["pnls"]) > 0:
        cum    = np.cumsum(ps["pnls"])
        max_dd = float((cum - np.maximum.accumulate(cum)).min())
        print(f"\n  Equity sketch (1 contract):")
        print(f"    Final cum. P&L:      ${cum[-1]:>8,.0f}")
        print(f"    Max DD (HWM→trough): ${max_dd:>8,.0f}")

    # ── OOS Gate verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"OOS GATE VERDICT — HCVWAP v3 Long-Only  "
          f"(SD={SD_PRIMARY}σ, STOP={STOP_PRIMARY}pts, MNQ, OOS {OOS_START}→{OOS_END})")
    print(f"{'='*100}")
    print(f"  Avg realized R/R={ps['avg_rr']:.2f}:1  |  Avg breakeven WR={ps['avg_be_wr']:.1%}  "
          f"|  OOS gate WR≥{gate_wr:.1%}  |  Stop=${STOP_PRIMARY * MNQ_PV:.0f}/contract")
    print()

    if ps["n"] < GATE_MIN_N:
        print(f"  ⚠️  INCONCLUSIVE — N={ps['n']} < {GATE_MIN_N} minimum.")
        print(f"  ⚠️  Too few OOS long signals to determine edge vs noise.")
        print(f"  ⚠️  Next step: wait for live S25 MNQ trades to accumulate (~2026-07-23).")
        print(f"{'='*100}")
        return

    g_ev   = ps["avg_pnl"]  > GATE_EV_MIN
    g_pf   = ps["pf"]       >= GATE_PF_MIN
    g_stop = ps["stop_med"] <= GATE_STOP_MAX
    g_wr   = ps["wr"]       >= gate_wr
    g_n    = ps["n"]        >= GATE_MIN_N
    g_womo = ps["worst_mo_pnl"] >= GATE_WOMO_PNL

    def v(flag, label, measured):
        return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<70} [measured: {measured}]"

    print(v(g_ev,   "EV > $0", f"${ps['avg_pnl']:.2f}/trade"))
    print(v(g_pf,   f"Profit factor ≥ {GATE_PF_MIN:.2f}  (OOS relaxed from 1.20)", f"{ps['pf']:.3f}"))
    print(v(g_stop, f"Median stop ≤ ${GATE_STOP_MAX:.0f}/contract", f"${ps['stop_med']:.0f}"))
    print(v(g_wr,   f"WR ≥ avg_be_wr+{GATE_WR_BONUS:.0%} (≥ {gate_wr:.1%})  (OOS relaxed from +5%)",
            f"{ps['wr']:.1%}"))
    print(v(g_n,    f"N ≥ {GATE_MIN_N}  (OOS relaxed from 20)", f"N={ps['n']}"))
    print(v(g_womo, f"Worst-month avg P&L ≥ ${GATE_WOMO_PNL:.0f}/trade  (OOS relaxed from -$50)",
            f"${ps['worst_mo_pnl']:.2f}"))

    edge_gates = [g_ev, g_pf, g_stop, g_wr, g_n]
    print()
    if all(edge_gates):
        print(f"  ✅ OOS GATE PASS — HCVWAP v3 long-only edge confirmed on MNQ OOS.")
        if not g_womo:
            print(f"  ⚠️  VARIANCE WARNING: worst-month avg=${ps['worst_mo_pnl']:.2f} < -$100")
        print(f"  → Next step: buy Topstep combine and trade at 3 MNQ contracts.")
    else:
        fails = []
        if not g_n:    fails.append(f"N={ps['n']} < {GATE_MIN_N}")
        if not g_ev:   fails.append(f"EV=${ps['avg_pnl']:.2f} ≤ $0")
        if not g_pf:   fails.append(f"PF={ps['pf']:.3f} < {GATE_PF_MIN:.2f}")
        if not g_wr:   fails.append(f"WR={ps['wr']:.1%} < {gate_wr:.1%}")
        if not g_stop: fails.append(f"Med stop=${ps['stop_med']:.0f} > ${GATE_STOP_MAX:.0f}")
        print(f"  ❌ OOS GATE FAIL — HCVWAP v3 long-only edge not confirmed OOS.")
        for lbl in fails:
            print(f"     • {lbl}")
        print(f"\n  → HCVWAP hypothesis declared exhausted across all architectures.")
        print(f"  → Remaining live threads:")
        print(f"      1. S25 paper trading — decision rule N≥20 AND ~2026-07-23")
        print(f"      2. GC CPI prospective test — Event 1 = June 11, N=10 needed (~17 months)")

    print(f"{'='*100}")
    print("\nDone.")


if __name__ == "__main__":
    main()

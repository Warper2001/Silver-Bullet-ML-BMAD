"""
HCVWAP v2 — 5-min False-Breakout Rejection Candle Gate 0 Study
Pre-registration: commit 7b17efe (2026-06-09)
study_hcvwap_v2.py written AFTER the pre-registration commit — tamper-evident.

Architectural fix over v1 (commit 78b6809):
  v1 failure: 1-min bar-close entry, 6-pt stop, fixed 12-pt target → PF=0.813
  v2 fix:     5-min rejection-wick entry, 15-pt stop, VWAP centerline target

Entry pattern (the "false-breakout rejection candle"):
  Short: 5-min bar.HIGH >= VWAP + SD×σ  AND  bar.CLOSE <  VWAP + SD×σ
  Long:  5-min bar.LOW  <= VWAP - SD×σ  AND  bar.CLOSE >  VWAP - SD×σ
  (price tested the 2σ band intrabar but got rejected — wick at the extreme)

Target: session VWAP price at entry (dynamic R/R, typically 2:1 to 6:1+).
Stop:   15 pts fixed (primary); wider than v1 so trade has room to breathe.

Instruments: MNQ ($2/pt), MES ($5/pt using ES price series).
In-sample: 2025-01-01 → 2026-02-28. Holdout >=2026-03-01 SEALED.
"""
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", message="Converting to Period representation")

# ── data paths ────────────────────────────────────────────────────────────────
MNQ_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_DATA  = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

IS_START = "2025-01-01"
IS_END   = "2026-02-28"

# ── primary spec (frozen by pre-registration 7b17efe) ────────────────────────
SD_PRIMARY      = 2.0
STOP_PRIMARY    = 15
SD_GRID         = [1.5, 2.0, 2.5]
STOP_GRID       = [10, 15, 20]

# ── confirmation stack parameters ────────────────────────────────────────────
VOL_MULT        = 1.5
VOL_LOOKBACK    = 20        # 20 × 5-min bars
HTF_EMA_FAST    = 9
HTF_EMA_SLOW    = 21
HTF_ATR_WIN     = 14
HTF_EMA_ATR_MULT = 0.5
SD_MIN_BARS     = 3

MIN_RR_MULT     = 1.5       # skip trade if target_pts < MIN_RR_MULT × stop_pts

# ── session / trade management ────────────────────────────────────────────────
RTH_START   = "09:30"
RTH_END     = "15:55"
SESS_CLOSE  = "15:55"
HOLD_MAX    = 12            # 12 × 5-min bars = 60 min
AM_WIN      = ("09:45", "11:30")
PM_WIN      = ("14:00", "15:00")

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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def resample_to_5min(rth_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min RTH bars to 5-min OHLCV."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    h5 = rth_1m.resample("5min", closed="left", label="left").agg(agg)
    h5 = h5.dropna(subset=["close"])
    # Filter to RTH start-end
    h5 = h5.between_time(RTH_START, RTH_END)
    return h5


def build_htf_ranging_15m(rth_1m: pd.DataFrame) -> pd.Series:
    """15-min HTF ranging signal; returns boolean Series on 5-min index via as-of join."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    h15 = rth_1m.resample("15min", closed="left", label="left").agg(agg).dropna(subset=["close"])
    h15["ema_f"] = ema(h15["close"], HTF_EMA_FAST)
    h15["ema_s"] = ema(h15["close"], HTF_EMA_SLOW)
    h15["tr"]    = h15["high"] - h15["low"]
    h15["atr"]   = h15["tr"].rolling(HTF_ATR_WIN).mean()
    h15["spread"] = (h15["ema_f"] - h15["ema_s"]).abs()
    h15["ranging"] = h15["spread"] < (HTF_EMA_ATR_MULT * h15["atr"])
    return h15["ranging"]


def build_session_vwap_5m(bars5: pd.DataFrame) -> pd.DataFrame:
    """
    Session VWAP and running σ_vwap on 5-min bars; reset daily.
    Adds: vwap, sigma_vwap, date columns.
    """
    df = bars5.copy()
    df["date"] = df.index.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"]  = np.nan
    df["sigma_vwap"] = np.nan

    for d, grp in df.groupby("date"):
        if len(grp) < 2:
            continue
        idx  = grp.index
        tp   = grp["tp"].values
        vol  = grp["volume"].values.astype(float)
        cv   = np.cumsum(tp * vol)
        cvol = np.cumsum(vol)
        cvol = np.where(cvol == 0, 1e-9, cvol)
        vwap_arr = cv / cvol
        close_arr = grp["close"].values
        dev  = close_arr - vwap_arr
        sig  = np.full(len(grp), np.nan)
        for k in range(SD_MIN_BARS, len(grp)):
            sig[k] = float(np.std(dev[:k+1], ddof=1))
        df.loc[idx, "vwap"]       = vwap_arr
        df.loc[idx, "sigma_vwap"] = sig

    df["vol_mean20"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    return df


def run_simulation(bars5: pd.DataFrame, ranging_15m: pd.Series,
                   sd_thresh: float, stop_pts: float, pv: float) -> list:
    """
    Simulate HCVWAP v2 on 5-min bars.

    Signal (all 4 required):
      1. Rejection wick: bar.high >= vwap + sd*sigma (short) or bar.low <= vwap - sd*sigma (long)
         AND close returns inside the band (bar.close < vwap + sd*sigma for short)
      2. Time window: AM_WIN or PM_WIN
      3. Volume spike: volume > VOL_MULT × vol_mean20
      4. HTF ranging: ranging_15m at this timestamp

    Entry: close of signal bar.
    Stop:  stop_pts from entry (fixed).
    Target: vwap price at entry time (dynamic).
    Skip if target < MIN_RR_MULT × stop_pts (low R/R).
    """
    stop_usd  = stop_pts * pv
    trades     = []
    active     = None
    hold_count = 0
    prev_date  = None

    # Join ranging signal to 5-min index (as-of from 15-min)
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
            at_close = ts_s >= SESS_CLOSE
            day_end  = d != active["date"]
            hi_k = hi_arr[k]; lo_k = lo_arr[k]

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
        vw  = vw_arr[k]; sg = sg_arr[k]
        if np.isnan(vw) or np.isnan(sg) or sg <= 0:
            continue

        hi_k = hi_arr[k]; lo_k = lo_arr[k]; cl_k = cl_arr[k]
        band_upper = vw + sd_thresh * sg
        band_lower = vw - sd_thresh * sg

        # False-breakout rejection test
        short_rej = (hi_k >= band_upper) and (cl_k < band_upper)  # wick above, close inside
        long_rej  = (lo_k <= band_lower) and (cl_k > band_lower)  # wick below, close inside

        if not (short_rej or long_rej):
            continue

        # Time window (condition 2)
        in_win = (AM_WIN[0] <= ts_s <= AM_WIN[1] or PM_WIN[0] <= ts_s <= PM_WIN[1])
        if not in_win:
            continue

        # Volume (condition 3)
        vm20 = vm20_arr[k]
        if np.isnan(vm20) or vm20 <= 0 or vol_arr[k] <= VOL_MULT * vm20:
            continue

        # HTF ranging (condition 4)
        if not rng_arr[k]:
            continue

        # Determine direction and R/R
        if short_rej and long_rej:
            # Both fired (bar both touched upper and lower band, very rare) — skip
            continue

        direction = -1 if short_rej else 1
        entry     = cl_k
        stop_p    = entry - direction * stop_pts
        target_p  = vw  # VWAP centerline

        # Direction sanity: VWAP must be on the correct side of entry.
        # If close < VWAP on a short (or close > VWAP on a long), the wick crossed
        # all the way through VWAP — the mean-reversion target is in the wrong direction.
        # These degenerate bars produce immediate "TP" hits with negative P&L; skip them.
        if direction == -1 and target_p >= entry:
            continue
        if direction ==  1 and target_p <= entry:
            continue

        target_pts = abs(target_p - entry)  # always positive after sanity check

        # Minimum R/R gate
        if target_pts < MIN_RR_MULT * stop_pts:
            continue  # VWAP too close — skip degenerate trade

        active = {
            "dir":        direction,
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
        pnl = (cl_arr[-1] - active["entry"]) * active["dir"] * pv - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades: list, n_days: int) -> dict:
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, stop_p75=0.0, worst_mo_wr=0.0, worst_mo_pnl=0.0,
                    pnls=np.array([]), mo={}, mo_pnl={},
                    exit_tp=0, exit_stop=0, exit_time=0,
                    n_long=0, n_short=0,
                    avg_rr=0.0, med_rr=0.0, avg_be_wr=0.0)
    n    = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    stops = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_w / max(1e-9, gross_l)

    # Per-trade breakeven WR (weighted by stop_usd)
    rrs = np.array([t["rr"] for t in trades])
    be_wrs = (stops + COMMISSION) / ((rrs + 1) * stops)
    avg_rr = float(rrs.mean())
    med_rr = float(np.median(rrs))
    avg_be_wr = float(be_wrs.mean())

    mo: dict = {}
    mo_pnl: dict = {}
    for t in trades:
        m = t["month"]
        mo.setdefault(m, [0, 0])
        mo[m][0 if t["win"] else 1] += 1
        mo_pnl.setdefault(m, [])
        mo_pnl[m].append(t["pnl"])
    worst_mo_wr  = min(w/(w+l) if w+l else 0 for w, l in mo.values()) if mo else 0.0
    worst_mo_pnl = min(float(np.mean(v)) for v in mo_pnl.values()) if mo_pnl else 0.0
    return dict(
        n=n, wr=wins/n, freq=n/n_days, avg_pnl=float(pnls.mean()), pf=pf,
        stop_med=float(np.median(stops)), stop_p75=float(np.percentile(stops, 75)),
        worst_mo_wr=worst_mo_wr, worst_mo_pnl=worst_mo_pnl,
        pnls=pnls, mo=mo, mo_pnl=mo_pnl,
        exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
        exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
        exit_time=sum(1 for t in trades if t["reason"] in
                      ("TIME", "CLOSE", "DAYEND", "END")),
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
    print(f"  P(reach ${target:,.0f} first):  {p_pass:>7.1%}")
    print(f"  P(ruin by ${abs(dd_limit):,.0f} DD): {p_ruin:>7.1%}  "
          f"{'✅ <20%' if p_ruin < 0.20 else '❌ >=20%'}")
    print(f"  P(open at day 30):     {p_open:>7.1%}")
    print(f"  Median qualifying days:{med_q:>6.1f}  (need 5)")
    print(f"  Mean final P&L:        ${avg_pnl:>8,.0f}  "
          f"({'✅ positive' if avg_pnl > 0 else '❌ negative'})")
    if ok:
        print(f"\n  ✅ PATH SIMULATION PASS — advance to OOS pre-registration (Gate 2).")
    else:
        fails = []
        if p_ruin >= 0.20: fails.append(f"P(ruin)={p_ruin:.1%}")
        if avg_pnl <= 0:   fails.append(f"E[P&L]=${avg_pnl:,.0f}")
        print(f"\n  ❌ PATH SIMULATION FAIL — {'; '.join(fails)}")


def run_instrument(inst: str) -> None:
    print(f"\n{'#'*100}")
    print(f"  INSTRUMENT: {inst.upper()}")
    print(f"{'#'*100}")

    if inst == "mnq":
        pv    = 2.0
        label = "MNQ ($2/pt)"
        raw   = pd.concat([load_et(MNQ_2025), load_et(MNQ_2026)])
    else:
        pv    = 5.0
        label = "MES ($5/pt, ES price series)"
        raw   = load_et(ES_DATA)

    raw = raw[~raw.index.duplicated(keep="first")]
    raw = raw[IS_START:IS_END]
    rth_1m = raw.between_time(RTH_START, RTH_END).copy()
    rth_1m["date"] = rth_1m.index.date

    # Resample to 5-min
    print(f"\n  Building 5-min bars from 1-min RTH data…")
    bars5 = resample_to_5min(rth_1m)
    bars5["date"] = bars5.index.date
    n_days = len(set(bars5["date"]))
    print(f"  5-min bars: {len(bars5):,}  (~{len(bars5)/n_days:.0f}/day)  |  {n_days} trading days")
    print(f"  In-sample: {bars5.index[0].date()} → {bars5.index[-1].date()}")

    # Session VWAP on 5-min
    print("  Computing session VWAP + σ_vwap on 5-min bars…")
    bars5 = build_session_vwap_5m(bars5)

    # HTF ranging (15-min from 1-min data)
    print("  Building 15-min HTF ranging signal…")
    ranging_15m = build_htf_ranging_15m(rth_1m)

    # ── confirmation funnel ────────────────────────────────────────────────
    print(f"\n  Confirmation funnel (SD={SD_PRIMARY}):")
    vw  = bars5["vwap"]
    sg  = bars5["sigma_vwap"]
    valid_mask = sg.notna() & vw.notna() & (sg > 0)

    band_upper = vw + SD_PRIMARY * sg
    band_lower = vw - SD_PRIMARY * sg

    # Rejection wick (false-breakout): intrabar pierce AND close returns inside
    short_rej = (bars5["high"] >= band_upper) & (bars5["close"] < band_upper) & valid_mask
    long_rej  = (bars5["low"]  <= band_lower) & (bars5["close"] > band_lower) & valid_mask
    n_raw_rej = (short_rej | long_rej).sum()

    # Time window
    def in_window(ts):
        s = ts.strftime("%H:%M")
        return (AM_WIN[0] <= s <= AM_WIN[1]) or (PM_WIN[0] <= s <= PM_WIN[1])

    tw_mask   = pd.Series([in_window(ts) for ts in bars5.index], index=bars5.index)
    n_after_tw = ((short_rej | long_rej) & tw_mask).sum()

    # Volume
    vm20 = bars5["vol_mean20"]
    vol_ok = (bars5["volume"] > VOL_MULT * vm20) & vm20.notna()
    n_after_vol = ((short_rej | long_rej) & tw_mask & vol_ok).sum()

    # HTF ranging
    rang_ok = ranging_15m.reindex(bars5.index, method="ffill").fillna(False)
    n_after_htf = ((short_rej | long_rej) & tw_mask & vol_ok & rang_ok).sum()

    print(f"    5-min bars with valid VWAP/σ:               {valid_mask.sum():>6,}")
    print(f"    Rejection wicks at |z|≥{SD_PRIMARY:.1f}σ (false-breakout): {n_raw_rej:>6,}   ({n_raw_rej/n_days:.2f}/day)")
    print(f"    After time-window filter:                   {n_after_tw:>6,}   ({n_after_tw/n_days:.2f}/day)")
    print(f"    After volume spike filter:                  {n_after_vol:>6,}   ({n_after_vol/n_days:.2f}/day)")
    print(f"    After HTF ranging filter:                   {n_after_htf:>6,}   ({n_after_htf/n_days:.2f}/day)")

    # ── grid ──────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"HCVWAP v2 GATE 0 GRID — {inst.upper()}  "
          f"(target=VWAP centeline, grid: SD × STOP_PTS)")
    print(f"{'='*100}")
    print(f"  {'SD':>4}  {'Stp':>4}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
          f"{'PF':>5}  {'AvgP&L':>8}  {'AvgRR':>7}  {'BEven':>6}  {'WorstMo':>9}")
    print(f"  {'--':>4}  {'--':>4}  {'---':>5}  {'-------':>8}  {'---':>7}  "
          f"{'----':>5}  {'------':>8}  {'------':>7}  {'-----':>6}  {'-------':>9}")

    grid_res: dict = {}
    for sd in SD_GRID:
        for stp in STOP_GRID:
            t = run_simulation(bars5, ranging_15m, sd, stp, pv)
            s = summarise(t, n_days)
            grid_res[(sd, stp)] = (t, s)
            is_p = (sd == SD_PRIMARY and stp == STOP_PRIMARY)
            prim = " ◀ PRIMARY" if is_p else ""
            ev_f = "✅" if s["avg_pnl"] > 0 else "❌"
            pf_f = "✅" if s["pf"] >= GATE0_PF_MIN else "❌"
            print(f"  {sd:>3.1f}  {stp:>4}  {s['n']:>5}  "
                  f"{s['freq']:>6.2f}/d  "
                  f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
                  f"${s['avg_pnl']:>6.2f}{ev_f}  "
                  f"{s['avg_rr']:>5.2f}:1  "
                  f"{s['avg_be_wr']:>6.1%}  "
                  f"${s['worst_mo_pnl']:>7.2f}{prim}")

    # ── primary deep dive ──────────────────────────────────────────────────
    pt, ps = grid_res[(SD_PRIMARY, STOP_PRIMARY)]

    print(f"\n{'='*100}")
    print(f"PRIMARY SPEC DEEP DIVE — {inst.upper()}  "
          f"(SD={SD_PRIMARY}σ, STOP={STOP_PRIMARY}pts, TARGET=VWAP, HOLD={HOLD_MAX}×5m)")
    print(f"{'='*100}")

    stop_usd = STOP_PRIMARY * pv
    gate_wr  = ps["avg_be_wr"] + 0.05 if ps["n"] > 0 else 0.50

    print(f"\n  Funnel:")
    print(f"    5-min RTH bars:     {len(bars5):,}")
    print(f"    Trading days:       {n_days}")
    print(f"    Signals taken:      {ps['n']}  ({ps['freq']:.3f}/day)  [expected 0.1–0.3/day]")
    if ps["n"] > 0:
        print(f"    Long / Short:       {ps['n_long']} / {ps['n_short']}  "
              f"({ps['n_long']/ps['n']:.0%} / {ps['n_short']/ps['n']:.0%})")
        print(f"    Exit breakdown:     TP={ps['exit_tp']}  "
              f"STOP={ps['exit_stop']}  TIME/CLOSE={ps['exit_time']}")
        print(f"    Avg realized R/R:  {ps['avg_rr']:.2f}:1  (median {ps['med_rr']:.2f}:1)")
        print(f"    Avg breakeven WR:  {ps['avg_be_wr']:.1%}  (gate ≥ {gate_wr:.1%})")

        # R/R distribution
        rrs = np.array([t["rr"] for t in pt])
        print(f"\n  R/R distribution of filtered trades:")
        for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 99)]:
            cnt = ((rrs >= lo) & (rrs < hi)).sum()
            lbl = f"{lo:.1f}:1–{hi:.1f}:1" if hi < 99 else f">{lo:.1f}:1"
            print(f"    {lbl:<12}  {cnt:>4} trades  ({cnt/len(rrs):.0%})")

    print(f"\n  Performance:")
    print(f"    Win rate:           {ps['wr']:.1%}  (avg be={ps['avg_be_wr']:.1%}; gate≥{gate_wr:.1%})")
    print(f"    Profit factor:      {ps['pf']:.3f}")
    print(f"    Avg net P&L:        ${ps['avg_pnl']:.2f}/trade")
    total_pnl = float(ps["pnls"].sum()) if len(ps["pnls"]) > 0 else 0.0
    print(f"    Total P&L (1c):     ${total_pnl:,.0f}  over {n_days} days")
    print(f"    Median stop:        ${ps['stop_med']:.0f}/contract")
    print(f"    Worst-month WR:     {ps['worst_mo_wr']:.1%}")
    print(f"    Worst-month avg:    ${ps['worst_mo_pnl']:.2f}/trade")

    # Long vs short
    if pt:
        long_t  = [t for t in pt if t["dir"] ==  1]
        short_t = [t for t in pt if t["dir"] == -1]
        print(f"\n  Long vs Short breakdown:")
        for lbl2, subset in [("Long (fade below VWAP)", long_t),
                              ("Short (fade above VWAP)", short_t)]:
            if subset:
                ns  = len(subset)
                ws  = sum(t["win"] for t in subset)
                ps2 = np.array([t["pnl"] for t in subset])
                gw  = sum(p for p in ps2 if p > 0); gl = abs(sum(p for p in ps2 if p < 0))
                pf2 = gw / max(1e-9, gl)
                rrs2 = np.array([t["rr"] for t in subset])
                print(f"    {lbl2:<28}  N={ns:>4}  WR={ws/ns:>6.1%}  "
                      f"PF={pf2:>5.2f}  AvgRR={rrs2.mean():.2f}:1  Avg=${ps2.mean():>7.2f}")

    # By month
    print(f"\n  By month:")
    print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'AvgRR':>7}  {'freq/d':>7}  Status")
    for m in sorted(ps["mo"]):
        w, l   = ps["mo"][m]
        n_mo   = w + l
        mwr    = w / n_mo if n_mo else 0
        avg    = float(np.mean(ps["mo_pnl"].get(m, [0])))
        mo_b   = bars5[bars5.index.to_period("M") == m]
        mo_d   = len(set(mo_b["date"]))
        mo_rr  = np.mean([t["rr"] for t in pt if t["month"] == m]) if pt else 0
        flag   = ("⚠️ N<3" if n_mo < 3 else
                  "❌ avg<-$50" if avg < GATE0_WOMO_PNL else "✅")
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"{mo_rr:>5.2f}:1  {n_mo/max(1, mo_d):>6.2f}/d  {flag}")

    # Time-of-day distribution
    if pt:
        from collections import Counter
        tod = Counter(t["bar_of_day"] for t in pt)
        buckets: dict = {}
        for ts_str, cnt in tod.items():
            h, mi = int(ts_str[:2]), int(ts_str[3:])
            b = f"{h:02d}:{(mi//30)*30:02d}"
            buckets[b] = buckets.get(b, 0) + cnt
        tot = sum(buckets.values())
        print(f"\n  Time-of-day distribution (ET):")
        for b in sorted(buckets):
            cnt = buckets[b]
            bar = "█" * int(cnt / max(buckets.values()) * 20)
            print(f"    {b}  {bar:<20}  {cnt:>4} ({cnt/tot:.0%})")

    # Victor's rolling-5-day
    print(f"\n  Victor's rolling-5-day variance check:")
    if pt:
        day_pnl: dict = {}
        for t in pt:
            day_pnl[t["date"]] = day_pnl.get(t["date"], 0.0) + t["pnl"]
        all_days  = sorted(set(bars5["date"]))
        daily_ser = [day_pnl.get(d, 0.0) for d in all_days]
        if len(daily_ser) >= 5:
            r5 = [sum(daily_ser[i:i+5]) for i in range(len(daily_ser)-4)]
            print(f"    Worst 5-day P&L:   ${min(r5):,.0f}")
            print(f"    Median 5-day P&L:  ${float(np.median(r5)):,.0f}")
            print(f"    % 5-day < $0:      {sum(1 for x in r5 if x < 0)/len(r5):.0%}")
            print(f"    Worst single-day:  ${min(daily_ser):,.0f}")
            pct_pos = sum(1 for x in daily_ser if x > 0) / len(daily_ser)
            print(f"    Days > $0:         {pct_pos:.0%}  ({len(daily_ser)} days)")
        else:
            print(f"    N<5 trades")
    else:
        print("    No trades")

    # Equity / max-DD
    print(f"\n  Equity / Max-DD sketch (1 contract):")
    if len(ps["pnls"]) > 0:
        cum  = np.cumsum(ps["pnls"])
        max_dd = float((cum - np.maximum.accumulate(cum)).min())
        hwm    = float(np.maximum.accumulate(cum).max())
        print(f"    Final cum. P&L:    ${cum[-1]:>8,.0f}")
        print(f"    Peak equity (HWM): ${hwm:>8,.0f}")
        print(f"    Max DD (HWM→trough): ${max_dd:>8,.0f}  "
              f"({'✅ inside $2k' if max_dd >= -2000 else '❌ EXCEEDS $2k'})")
        if len(ps["pnls"]) >= 20:
            rdds = []
            for i in range(len(ps["pnls"])-19):
                c2 = np.cumsum(ps["pnls"][i:i+20])
                rdds.append(float((c2 - np.maximum.accumulate(c2)).min()))
            print(f"    Worst 20-trade DD: ${min(rdds):>8,.0f}")
    else:
        print("    No trades")

    # ── Gate 0 verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"GATE 0 VERDICT — {inst.upper()} v2  "
          f"(SD={SD_PRIMARY}σ, STOP={STOP_PRIMARY}pts, TARGET=VWAP centeline)")
    print(f"{'='*100}")
    print(f"  Avg realized R/R={ps['avg_rr']:.2f}:1  |  Avg breakeven WR={ps['avg_be_wr']:.1%}  "
          f"|  Gate WR≥{gate_wr:.1%}  |  Stop=${ STOP_PRIMARY * pv:.0f}/contract")
    print()

    g_ev   = ps["avg_pnl"]  > GATE0_EV_MIN
    g_pf   = ps["pf"]       >= GATE0_PF_MIN
    g_stop = ps["stop_med"] <= GATE0_STOP_MAX
    g_wr   = ps["wr"]       >= gate_wr
    g_n    = ps["n"]        >= GATE0_MIN_N
    g_womo = ps["worst_mo_pnl"] >= GATE0_WOMO_PNL

    def v(flag, label, measured):
        return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<60} [measured: {measured}]"

    print(v(g_ev,   "EV > $0 (avg net P&L > $0)", f"${ps['avg_pnl']:.2f}/trade"))
    print(v(g_pf,   f"Profit factor ≥ {GATE0_PF_MIN:.2f}", f"{ps['pf']:.3f}"))
    print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract", f"${ps['stop_med']:.0f}"))
    print(v(g_wr,   f"WR ≥ avg_be_wr+5% (≥ {gate_wr:.1%})", f"{ps['wr']:.1%}"))
    print(v(g_n,    f"N ≥ {GATE0_MIN_N} (min sample)", f"N={ps['n']}"))
    print(v(g_womo, f"Worst-month avg P&L ≥ ${GATE0_WOMO_PNL:.0f}/trade", f"${ps['worst_mo_pnl']:.2f}"))
    print(f"\n  [FREQ] {ps['freq']:.3f}/day  (informational; ~0.1–0.3/day expected for this selective entry)")

    edge_gates = [g_ev, g_pf, g_stop, g_wr, g_n]
    print()
    if all(edge_gates):
        print(f"  ✅ GATE 0 PASS — v2 false-breakout rejection edge confirmed on {inst.upper()}.")
        if not g_womo:
            print(f"  ⚠️  VARIANCE WARNING: worst-month avg=${ps['worst_mo_pnl']:.2f} < -$50")
        else:
            print(f"  ✅ VARIANCE OK")
        if pt:
            day_pnl2: dict = {}
            for t in pt:
                day_pnl2[t["date"]] = day_pnl2.get(t["date"], 0.0) + t["pnl"]
            all_days2 = sorted(set(bars5["date"]))
            daily_1c  = [day_pnl2.get(d, 0.0) for d in all_days2]
            path_simulation(daily_1c, n_contracts=3, label=f"{inst.upper()} v2 Gate0 PASS")
    else:
        fails = []
        if not g_n:
            fails.append(f"N={ps['n']} < {GATE0_MIN_N}")
        if not g_ev:
            fails.append(f"EV=${ps['avg_pnl']:.2f} ≤ $0")
        if not g_pf:
            fails.append(f"PF={ps['pf']:.3f} < {GATE0_PF_MIN:.2f}")
        if not g_wr:
            fails.append(f"WR={ps['wr']:.1%} < {gate_wr:.1%}")
        if not g_stop:
            fails.append(f"Med stop=${ps['stop_med']:.0f} > ${GATE0_STOP_MAX:.0f}")
        print(f"  ❌ GATE 0 FAIL — {inst.upper()} v2 edge not confirmed.")
        for lbl in fails:
            print(f"     • {lbl}")

        if ps["n"] == 0:
            print("     NOTE: Zero trades — the rejection-wick pattern plus 4 conditions NEVER"
                  " fired simultaneously in-sample. The setup may be too rare to evaluate.")
        elif not g_n:
            print(f"     NOTE: N={ps['n']} is too small for a reliable edge read."
                  " This is the 'too rare' verdict — edge may be real but structurally insufficient.")
        elif not g_ev and g_pf:
            print("     NOTE: This should not happen (PF>1.2 with EV<0); check logic.")

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=["mnq", "mes", "both"], default="both")
    args = parser.parse_args()

    print("=" * 100)
    print("HCVWAP v2 — 5-min False-Breakout Rejection Candle Gate 0 Study")
    print("Pre-registration: commit 7b17efe (2026-06-09)")
    print("Architectural fix: 5-min rejection wick + 15-pt stop + VWAP centerline target")
    print("In-sample: 2025-01-01 → 2026-02-28  |  Holdout ≥2026-03-01 SEALED")
    print("=" * 100)
    print(f"\nEntry: 5-min bar.high ≥ VWAP+{SD_PRIMARY}σ AND close returns inside band (short)")
    print(f"       5-min bar.low  ≤ VWAP-{SD_PRIMARY}σ AND close returns inside band (long)")
    print(f"Stop:  {STOP_PRIMARY} pts fixed | Target: VWAP centerline | Min R/R: {MIN_RR_MULT}:1")
    print(f"Time windows: {AM_WIN[0]}–{AM_WIN[1]} ET  OR  {PM_WIN[0]}–{PM_WIN[1]} ET")

    instruments = ["mnq", "mes"] if args.instrument == "both" else [args.instrument]
    for inst in instruments:
        run_instrument(inst)

    print("\n\nDone.")


if __name__ == "__main__":
    main()

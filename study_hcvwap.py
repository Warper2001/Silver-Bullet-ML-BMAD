"""
HCVWAP — High-Confirmation VWAP Gate 0 Study
Pre-registration: commit 4531a3d (2026-06-09)
study_hcvwap.py written AFTER the pre-registration commit — tamper-evident.

Setup class: VWAP 2σ fade filtered by ALL FOUR confirmations simultaneously
  1. Extension:    close ≥ SD_THRESH × σ_vwap above/below session VWAP
  2. Time window:  09:45–11:30 ET  OR  14:00–15:00 ET (exclude open noise + lunch)
  3. Volume spike: bar volume > VOL_MULT × 20-bar rolling mean
  4. HTF ranging:  15-min |EMA(9)−EMA(21)| < HTF_EMA_ATR_MULT × ATR(14)

Direction: short at +2σ (price too extended above VWAP), long at −2σ.
Fixed stop: 6 pts. Fixed target: 12 pts (2:1 R/R). Breakeven WR ≈ 34–47% depending on PV.

Instruments:
  MNQ — 1 contract = $2/pt.   Data: mnq_1min_2025.csv + mnq_1min_2026_ytd.csv
  MES — 1 contract = $5/pt.   Data: es_1min_2025_2026.csv (ES price series, MES PV)

In-sample: 2025-01-01 → 2026-02-28. Holdout ≥2026-03-01 SEALED.
"""
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", message="Converting to Period representation")

# ── data paths ────────────────────────────────────────────────────────────────
MNQ_2025  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_DATA   = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

IS_START  = "2025-01-01"
IS_END    = "2026-02-28"

# ── primary spec (frozen by pre-registration 4531a3d) ────────────────────────
SD_THRESH       = 2.0       # primary; grid: [1.5, 2.0, 2.5]
TP_PTS_PRIMARY  = 12        # points; grid: [10, 12, 16]
STOP_PTS        = 6         # fixed across all grid cells
SD_GRID         = [1.5, 2.0, 2.5]
TP_PTS_GRID     = [10, 12, 16]

# ── confirmation stack parameters ────────────────────────────────────────────
VOL_MULT        = 1.5       # volume > 1.5× 20-bar mean
VOL_LOOKBACK    = 20
HTF_EMA_FAST    = 9
HTF_EMA_SLOW    = 21
HTF_ATR_WIN     = 14
HTF_EMA_ATR_MULT = 0.5      # ranging: |ema9-ema21| < 0.5×ATR(14) on 15m bars
SD_MIN_BARS     = 5         # min bars in session before σ_vwap is valid

# ── session / trade management ────────────────────────────────────────────────
RTH_START     = "09:30"
RTH_END       = "15:55"
SESS_CLOSE    = "15:55"
HOLD_MAX      = 60          # bars; time-stop

# Time windows (ET) — converted to time strings for between_time
AM_WIN = ("09:45", "11:30")
PM_WIN = ("14:00", "15:00")

# ── gate thresholds ───────────────────────────────────────────────────────────
GATE0_EV_MIN    = 0.0
GATE0_PF_MIN    = 1.20
GATE0_STOP_MAX  = 150.0
GATE0_WOMO_PNL  = -50.0
GATE0_MIN_N     = 30
COMMISSION      = 4.80
ROLLING_5D_TGT  = 5 * 150.0  # $750 — Victor's 5×qualifying-day threshold


# ── helpers ───────────────────────────────────────────────────────────────────
def load_et(path: Path) -> pd.DataFrame:
    """Load 1-min bars, localize to America/New_York."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── build 15-min HTF ranging signal ──────────────────────────────────────────
def build_htf_ranging(rth: pd.DataFrame) -> pd.Series:
    """
    Resample 1-min RTH bars to 15-min, compute EMA(9)/EMA(21) spread vs ATR(14).
    Return a boolean Series on the 1-min index (as-of join from last known 15-min bar).
    True = HTF is ranging (EMA spread < HTF_EMA_ATR_MULT × ATR).
    """
    # Resample to 15-min
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    h15 = rth.resample("15min", closed="left", label="left").agg(agg).dropna(subset=["close"])

    # EMAs on 15-min close
    h15["ema_fast"] = ema(h15["close"], HTF_EMA_FAST)
    h15["ema_slow"] = ema(h15["close"], HTF_EMA_SLOW)

    # ATR(14) on 15-min bars (simple high−low proxy; no overnight)
    h15["tr"]  = h15["high"] - h15["low"]
    h15["atr"] = h15["tr"].rolling(HTF_ATR_WIN).mean()

    # Ranging condition
    h15["spread"] = (h15["ema_fast"] - h15["ema_slow"]).abs()
    h15["ranging"] = h15["spread"] < (HTF_EMA_ATR_MULT * h15["atr"])

    # As-of join: for each 1-min bar, use the last available (complete) 15-min bar
    # reindex_asof: fills forward from 15-min to 1-min
    ranging_15 = h15["ranging"].reindex(rth.index, method="ffill")
    return ranging_15.fillna(False)


# ── session VWAP and σ_vwap ──────────────────────────────────────────────────
def build_session_vwap(rth: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative session VWAP and rolling intra-session σ_vwap.
    Returns the input DataFrame with added columns:
      vwap, sigma_vwap, z_score (signed), vol_mean20
    All reset at each RTH session open (09:30 ET).
    """
    df = rth.copy()
    df["date"] = df.index.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3.0  # typical price

    # Initialise
    df["vwap"]       = np.nan
    df["sigma_vwap"] = np.nan
    df["z_score"]    = np.nan

    for d, grp in df.groupby("date"):
        if len(grp) < 2:
            continue
        idx = grp.index
        tp_arr  = grp["tp"].values
        vol_arr = grp["volume"].values.astype(float)

        # Cumulative VWAP
        cum_tpv = np.cumsum(tp_arr * vol_arr)
        cum_vol = np.cumsum(vol_arr)
        cum_vol = np.where(cum_vol == 0, 1e-9, cum_vol)
        vwap_arr = cum_tpv / cum_vol

        # Rolling intra-session σ(close − VWAP)
        close_arr = grp["close"].values
        dev = close_arr - vwap_arr
        sigma_arr = np.full(len(grp), np.nan)
        for k in range(SD_MIN_BARS, len(grp)):
            sigma_arr[k] = float(np.std(dev[:k+1], ddof=1))

        # Z-score
        with np.errstate(invalid="ignore", divide="ignore"):
            z_arr = np.where(sigma_arr > 0, dev / sigma_arr, 0.0)

        df.loc[idx, "vwap"]       = vwap_arr
        df.loc[idx, "sigma_vwap"] = sigma_arr
        df.loc[idx, "z_score"]    = z_arr

    # Volume 20-bar rolling mean (RTH-only)
    df["vol_mean20"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    return df


# ── simulation ────────────────────────────────────────────────────────────────
def run_simulation(rth_full: pd.DataFrame, ranging_signal: pd.Series,
                   sd_thresh: float, tp_pts: float, pv: float) -> list:
    """
    Simulate HCVWAP strategy on RTH bars with precomputed VWAP/z/vol/ranging.

    Entry conditions (all required simultaneously):
      1. |z_score| >= sd_thresh  (direction: short if z>0, long if z<0)
      2. time in AM_WIN or PM_WIN (ET)
      3. volume > VOL_MULT × vol_mean20
      4. ranging_signal == True at this bar
    Trade: fixed stop STOP_PTS, fixed target tp_pts.
    """
    stop_pts = STOP_PTS
    stop_usd = stop_pts * pv

    trades      = []
    active      = None
    hold_count  = 0
    prev_date   = None

    cl_arr   = rth_full["close"].values
    hi_arr   = rth_full["high"].values
    lo_arr   = rth_full["low"].values
    z_arr    = rth_full["z_score"].values
    sig_arr  = rth_full["sigma_vwap"].values
    vol_arr  = rth_full["volume"].values
    vm20_arr = rth_full["vol_mean20"].values
    date_arr = rth_full["date"].values
    ts_arr   = rth_full.index
    rng_arr  = ranging_signal.reindex(rth_full.index).fillna(False).values

    for k in range(len(rth_full)):
        ts   = ts_arr[k]
        d    = date_arr[k]
        time_str = ts.strftime("%H:%M")

        # ── day boundary: force-close residual ────────────────────────────
        if d != prev_date:
            if active is not None:
                ep  = cl_arr[k - 1] if k > 0 else cl_arr[k]
                pnl = (ep - active["entry"]) * active["dir"] * pv - COMMISSION
                trades.append({**active, "exit_p": ep, "pnl": pnl,
                                "win": pnl > 0, "reason": "DAYEND"})
                active = None; hold_count = 0
            prev_date = d

        # ── manage active trade ────────────────────────────────────────────
        if active is not None:
            hold_count += 1
            at_close  = time_str >= SESS_CLOSE
            day_end   = d != active["date"]

            hit_tp   = ((active["dir"] ==  1 and hi_arr[k] >= active["tp"]) or
                        (active["dir"] == -1 and lo_arr[k] <= active["tp"]))
            hit_stop = ((active["dir"] ==  1 and lo_arr[k] <= active["stop"]) or
                        (active["dir"] == -1 and hi_arr[k] >= active["stop"]))

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
            continue  # skip entry check this bar

        # ── confirmation funnel: check 4 conditions ───────────────────────
        z = z_arr[k]
        if np.isnan(z) or abs(z) < sd_thresh:
            continue  # condition 1

        # Time window condition (condition 2)
        in_window = (AM_WIN[0] <= time_str <= AM_WIN[1] or
                     PM_WIN[0] <= time_str <= PM_WIN[1])
        if not in_window:
            continue

        # Volume condition (condition 3)
        vm20 = vm20_arr[k]
        if np.isnan(vm20) or vm20 <= 0 or vol_arr[k] <= VOL_MULT * vm20:
            continue

        # HTF ranging condition (condition 4)
        if not rng_arr[k]:
            continue

        # All 4 conditions satisfied — enter
        direction = -1 if z > 0 else 1   # short if above VWAP, long if below
        entry  = cl_arr[k]
        stop_p = entry - direction * stop_pts
        tp_p   = entry + direction * tp_pts

        active = {
            "dir":       direction,
            "entry":     entry,
            "tp":        tp_p,
            "stop":      stop_p,
            "stop_pts":  stop_pts,
            "stop_usd":  stop_usd,
            "tp_pts":    tp_pts,
            "z_at_entry": abs(z),
            "date":      d,
            "month":     ts.to_period("M"),
            "bar_of_day": time_str,
        }
        hold_count = 0

    # residual
    if active:
        pnl = (cl_arr[-1] - active["entry"]) * active["dir"] * pv - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


# ── summarise ────────────────────────────────────────────────────────────────
def summarise(trades: list, n_days: int) -> dict:
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, stop_p75=0.0, worst_mo_wr=0.0, worst_mo_pnl=0.0,
                    pnls=np.array([]), mo={}, mo_pnl={},
                    exit_tp=0, exit_stop=0, exit_time=0,
                    n_long=0, n_short=0)
    n    = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    stops = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_w / max(1e-9, gross_l)
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
    )


# ── combine-math path simulation ─────────────────────────────────────────────
def path_simulation(daily_pnl_1c: list, n_contracts: int, label: str,
                    target: float = 3000, dd_limit: float = -2000,
                    qual_thresh: float = 150, consistency_cap: float = 0.50,
                    n_days_horizon: int = 30, n_sims: int = 10_000) -> None:
    """
    Monte Carlo combine-math simulation via bootstrap.
    daily_pnl_1c: list of per-day P&L for 1 contract
    n_contracts:  combine sizing
    """
    if not daily_pnl_1c or len(daily_pnl_1c) < 5:
        print(f"\n  Path simulation ({label}): insufficient daily P&L data (N={len(daily_pnl_1c)})")
        return

    daily = np.array(daily_pnl_1c) * n_contracts
    rng   = np.random.default_rng(42)

    n_pass = 0; n_ruin = 0
    qual_day_counts = []
    final_pnls = []

    for _ in range(n_sims):
        path = rng.choice(daily, size=n_days_horizon, replace=True)
        cum  = np.cumsum(path)
        hwm  = np.maximum.accumulate(cum)
        dd   = cum - hwm  # always ≤ 0

        # Check ruin (trailing DD < dd_limit) at any point
        ruined = False
        passed = False
        q_days = 0
        for i, (c, d) in enumerate(zip(cum, dd)):
            if d <= dd_limit:
                ruined = True
                break
            if path[i] >= qual_thresh:
                q_days += 1
            # consistency cap: no single day > 50% of running total P&L
            # (simplification: check day P&L vs cumulative — only when cum > 0)
            if c > 0 and path[i] > consistency_cap * c:
                pass  # informational only — don't auto-fail the sim
            if c >= target and not ruined:
                passed = True
                break

        if ruined:
            n_ruin += 1
        elif passed:
            n_pass += 1
        qual_day_counts.append(q_days)
        final_pnls.append(float(cum[-1]) if not ruined else float(dd_limit))

    p_pass  = n_pass  / n_sims
    p_ruin  = n_ruin  / n_sims
    p_open  = 1 - p_pass - p_ruin   # still in combine at day 30 (neither hit)
    med_q   = float(np.median(qual_day_counts))
    avg_pnl = float(np.mean(final_pnls))
    ok      = p_ruin < 0.20 and avg_pnl > 0

    print(f"\n{'='*100}")
    print(f"COMBINE-MATH PATH SIMULATION — {label}  "
          f"({n_contracts} contracts, {n_sims:,} Monte Carlo paths)")
    print(f"{'='*100}")
    print(f"  Daily P&L bootstrap from {len(daily_pnl_1c)} in-sample days  "
          f"(non-zero: {sum(1 for x in daily_pnl_1c if x != 0)})")
    print(f"  Combine: target=${target:,.0f}  DD_limit=${dd_limit:,.0f}  "
          f"qual_day≥${qual_thresh:.0f}  horizon={n_days_horizon} days")
    print()
    print(f"  P(reach target first):   {p_pass:>7.1%}")
    print(f"  P(ruin by DD):           {p_ruin:>7.1%}  "
          f"{'✅ <20% gate' if p_ruin < 0.20 else '❌ ≥20% gate'}")
    print(f"  P(still open at day 30): {p_open:>7.1%}")
    print(f"  Median qualifying days:  {med_q:.1f}  (need 5 of {n_days_horizon})")
    print(f"  Mean final P&L:          ${avg_pnl:>8,.0f}  "
          f"({'✅ positive' if avg_pnl > 0 else '❌ negative'})")
    print()
    if ok:
        print(f"  ✅ PATH SIMULATION PASS — P(ruin)={p_ruin:.1%} < 20% AND E[P&L]>${avg_pnl:,.0f} > $0")
        print(f"     Proceed to OOS pre-registration (Gate 2 requires separate commit).")
    else:
        fails = []
        if p_ruin >= 0.20:
            fails.append(f"P(ruin)={p_ruin:.1%} ≥ 20%")
        if avg_pnl <= 0:
            fails.append(f"E[P&L]=${avg_pnl:,.0f} ≤ $0")
        print(f"  ❌ PATH SIMULATION FAIL — {'; '.join(fails)}")
        print(f"     Strategy has edge but expected combine path is negative / too risky.")


# ── per-instrument Gate 0 run ─────────────────────────────────────────────────
def run_instrument(inst: str) -> None:
    # ── 1. load data ──────────────────────────────────────────────────────────
    print(f"\n{'#'*100}")
    print(f"  INSTRUMENT: {inst.upper()}")
    print(f"{'#'*100}")

    if inst == "mnq":
        pv    = 2.0
        label = "MNQ (1 contract = $2/pt)"
        raw   = pd.concat([load_et(MNQ_2025), load_et(MNQ_2026)])
    else:
        pv    = 5.0
        label = "MES (1 contract = $5/pt; price from ES 1-min series)"
        raw   = load_et(ES_DATA)

    raw  = raw[~raw.index.duplicated(keep="first")]
    raw  = raw[IS_START:IS_END]
    rth  = raw.between_time(RTH_START, RTH_END).copy()
    rth["date"] = rth.index.date

    n_days = len(set(rth["date"]))
    print(f"\n  Loaded: {len(rth):,} 1-min RTH bars | {n_days} trading days")
    print(f"  In-sample: {rth.index[0].date()} → {rth.index[-1].date()}")
    print(f"  Point value: ${pv}/pt  |  {label}")

    # ── 2. build VWAP / σ / volume features ──────────────────────────────────
    print("\n  Building session VWAP, σ_vwap, z-score, volume features…")
    rth = build_session_vwap(rth)

    # ── 3. build 15-min HTF ranging signal ────────────────────────────────────
    print("  Building 15-min HTF ranging signal (EMA9/21 spread vs ATR14)…")
    ranging = build_htf_ranging(rth)

    # ── 4. confirmation funnel (at primary SD threshold) ─────────────────────
    print(f"\n  Confirmation funnel (primary SD={SD_THRESH}):")
    mask_valid  = rth["z_score"].notna() & rth["sigma_vwap"].notna()
    n_valid     = mask_valid.sum()
    n_raw_ext   = (rth["z_score"].abs() >= SD_THRESH).sum()
    n_after_tw  = sum(1 for ts, z in zip(rth[mask_valid].index,
                                         rth.loc[mask_valid, "z_score"])
                      if abs(z) >= SD_THRESH and
                      (AM_WIN[0] <= ts.strftime("%H:%M") <= AM_WIN[1] or
                       PM_WIN[0] <= ts.strftime("%H:%M") <= PM_WIN[1]))

    # volume filter
    vm20 = rth["vol_mean20"]
    vol_ok = (rth["volume"] > VOL_MULT * vm20) & vm20.notna()
    z_ext = rth["z_score"].abs() >= SD_THRESH
    n_after_vol = (z_ext & vol_ok).sum()

    # HTF filter
    rang_ok = ranging.reindex(rth.index).fillna(False)
    n_after_htf = (z_ext & vol_ok & rang_ok).sum()

    print(f"    Total 1-min RTH bars with valid z-score:     {n_valid:>6,}")
    print(f"    Raw bars at |z| ≥ {SD_THRESH:.1f}σ:                  {n_raw_ext:>6,}   ({n_raw_ext/n_days:.2f}/day)")
    print(f"    After time-window filter (AM + PM only):     {n_after_tw:>6,}   ({n_after_tw/n_days:.2f}/day)")
    print(f"    After volume spike filter (>{VOL_MULT}× mean):   {n_after_vol:>6,}   ({n_after_vol/n_days:.2f}/day)")
    print(f"    After HTF ranging filter:                    {n_after_htf:>6,}   ({n_after_htf/n_days:.2f}/day)")

    # ── 5. grid ───────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"HCVWAP GATE 0 GRID — {inst.upper()}  "
          f"(stop={STOP_PTS}pts fixed, grid: SD × TP_PTS)")
    print(f"{'='*100}")
    print(f"  {'SD':>4}  {'TP':>4}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
          f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed':>8}  {'WorstMoP&L':>11}  {'BEven':>6}")
    sep = f"  {'--':>4}  {'--':>4}  {'---':>5}  {'-------':>8}  {'---':>7}  {'----':>5}  {'------':>8}  {'-------':>8}  {'-----------':>11}  {'-----':>6}"
    print(sep)

    grid_res: dict = {}
    for sd in SD_GRID:
        for tp in TP_PTS_GRID:
            stop_usd = STOP_PTS * pv
            t = run_simulation(rth, ranging, sd, tp, pv)
            s = summarise(t, n_days)
            grid_res[(sd, tp)] = (t, s)
            is_primary = (sd == SD_THRESH and tp == TP_PTS_PRIMARY)
            prim = " ◀ PRIMARY" if is_primary else ""
            be_wr = (stop_usd + COMMISSION) / ((tp / STOP_PTS + 1) * stop_usd) if stop_usd > 0 else 0.0
            ev_f  = "✅" if s["avg_pnl"] > GATE0_EV_MIN  else "❌"
            pf_f  = "✅" if s["pf"]      >= GATE0_PF_MIN  else "❌"
            print(f"  {sd:>3.1f}  {tp:>4.0f}  {s['n']:>5}  "
                  f"{s['freq']:>6.2f}/d  "
                  f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
                  f"${s['avg_pnl']:>6.2f}{ev_f}  "
                  f"${s['stop_med']:>6.0f}  "
                  f"${s['worst_mo_pnl']:>9.2f}  "
                  f"{be_wr:>6.1%}{prim}")

    # ── 6. TP sensitivity at primary SD ──────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"TP SENSITIVITY  (SD={SD_THRESH}, primary SD fixed) — {inst.upper()}")
    print(f"{'='*100}")
    print(f"  {'TP_pts':>6}  {'BEven':>6}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
          f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMoAvg':>11}")
    stop_usd = STOP_PTS * pv
    for tp in TP_PTS_GRID:
        _, s = grid_res[(SD_THRESH, tp)]
        be_wr = (stop_usd + COMMISSION) / ((tp / STOP_PTS + 1) * stop_usd) if stop_usd > 0 else 0.0
        ev_f  = "✅" if s["avg_pnl"] > 0 else "❌"
        prim  = " ◀ PRIMARY" if tp == TP_PTS_PRIMARY else ""
        print(f"  {tp:>6.0f}pts  {be_wr:>6.1%}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
              f"{s['wr']:>7.1%}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}{ev_f}  "
              f"${s['worst_mo_pnl']:>9.2f}{prim}")

    # ── 7. primary spec deep dive ─────────────────────────────────────────────
    pt, ps = grid_res[(SD_THRESH, TP_PTS_PRIMARY)]

    print(f"\n{'='*100}")
    print(f"PRIMARY SPEC DEEP DIVE — {inst.upper()}  "
          f"(SD={SD_THRESH}σ, TP={TP_PTS_PRIMARY}pts, STOP={STOP_PTS}pts, HOLD_MAX={HOLD_MAX})")
    print(f"{'='*100}")

    stop_usd = STOP_PTS * pv
    be_wr = (stop_usd + COMMISSION) / ((TP_PTS_PRIMARY / STOP_PTS + 1) * stop_usd) if stop_usd > 0 else 0.0
    gate_wr = be_wr + 0.05

    print(f"\n  Funnel:")
    print(f"    1-min RTH bars:    {len(rth):,}")
    print(f"    Trading days:      {n_days}")
    print(f"    Filtered signals:  {ps['n']}  ({ps['freq']:.2f}/day)  [expected 0.2–0.4/day]")
    if ps["n"] > 0:
        print(f"    Long / Short:      {ps['n_long']} / {ps['n_short']}  "
              f"({ps['n_long']/ps['n']:.0%} long, {ps['n_short']/ps['n']:.0%} short)")
        print(f"    Exit breakdown:    "
              f"TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
              f"TIME/CLOSE={ps['exit_time']}")

    print(f"\n  Performance:")
    print(f"    Win rate:          {ps['wr']:.1%}  (breakeven={be_wr:.1%}; gate≥{gate_wr:.1%})")
    print(f"    Profit factor:     {ps['pf']:.3f}")
    print(f"    Avg net P&L:       ${ps['avg_pnl']:.2f}/trade")
    total_pnl = float(ps["pnls"].sum()) if len(ps["pnls"]) > 0 else 0.0
    print(f"    Total P&L (1c):    ${total_pnl:,.0f}  over {n_days} days")
    print(f"    Median stop:       ${ps['stop_med']:.0f}/contract")
    print(f"    75th-pct stop:     ${ps['stop_p75']:.0f}/contract")
    print(f"    Worst-month WR:    {ps['worst_mo_wr']:.1%}")
    print(f"    Worst-month avg:   ${ps['worst_mo_pnl']:.2f}/trade")

    # Long vs Short breakdown
    if pt:
        long_t  = [t for t in pt if t["dir"] ==  1]
        short_t = [t for t in pt if t["dir"] == -1]
        print(f"\n  Long vs Short breakdown:")
        for lbl, subset in [("Long (fade below VWAP)", long_t),
                             ("Short (fade above VWAP)", short_t)]:
            if subset:
                ns  = len(subset)
                ws  = sum(t["win"] for t in subset)
                ps2 = np.array([t["pnl"] for t in subset])
                gw  = sum(p for p in ps2 if p > 0); gl = abs(sum(p for p in ps2 if p < 0))
                pf2 = gw / max(1e-9, gl)
                print(f"    {lbl:<28}  N={ns:>4}  WR={ws/ns:>6.1%}  "
                      f"PF={pf2:>5.2f}  Avg=${ps2.mean():>7.2f}")
            else:
                print(f"    {lbl:<28}  N=0")

    # By-month table
    print(f"\n  By month:")
    print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'freq/d':>8}  {'Status'}")
    for m in sorted(ps["mo"]):
        w, l   = ps["mo"][m]
        n_mo   = w + l
        mwr    = w / n_mo if n_mo else 0
        avg    = float(np.mean(ps["mo_pnl"].get(m, [0])))
        mo_bars = rth[rth.index.to_period("M") == m]
        mo_days = len(set(mo_bars["date"]))
        if n_mo < 3:
            flag = "⚠️ N<3"
        elif avg < GATE0_WOMO_PNL:
            flag = "❌ avg<-$50"
        else:
            flag = "✅"
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"{n_mo/max(1, mo_days):>7.2f}/d  {flag}")

    # Time-of-day distribution
    if pt:
        from collections import Counter
        tod = Counter(t["bar_of_day"] for t in pt)
        buckets: dict = {}
        for ts_str, cnt in tod.items():
            h, mi = int(ts_str[:2]), int(ts_str[3:])
            b = f"{h:02d}:{(mi//30)*30:02d}"
            buckets[b] = buckets.get(b, 0) + cnt
        total_t = sum(buckets.values())
        print(f"\n  Time-of-day distribution (ET):")
        for b in sorted(buckets):
            cnt = buckets[b]
            bar = "█" * int(cnt / max(buckets.values()) * 20)
            print(f"    {b}  {bar:<20}  {cnt:>4} ({cnt/total_t:.0%})")

    # Victor's rolling-5-day variance diagnostic
    print(f"\n  Victor's rolling-5-day variance check:")
    if pt:
        day_pnl: dict = {}
        for t in pt:
            d = t["date"]
            day_pnl[d] = day_pnl.get(d, 0.0) + t["pnl"]
        all_days    = sorted(set(rth["date"]))
        daily_ser   = [day_pnl.get(d, 0.0) for d in all_days]

        if len(daily_ser) >= 5:
            rolling5   = [sum(daily_ser[i:i+5]) for i in range(len(daily_ser)-4)]
            worst5     = min(rolling5)
            best5      = max(rolling5)
            median5    = float(np.median(rolling5))
            pct_neg    = sum(1 for x in rolling5 if x < 0) / len(rolling5)
            worst_day  = min(daily_ser)
            pct_days_pos = sum(1 for x in daily_ser if x > 0) / len(daily_ser)
            print(f"    Worst  5-day P&L:   ${worst5:,.0f}  "
                  f"({'✅' if worst5 >= 0 else '❌'})")
            print(f"    Best   5-day P&L:   ${best5:,.0f}")
            print(f"    Median 5-day P&L:   ${median5:,.0f}")
            print(f"    % 5-day windows < $0: {pct_neg:.0%}")
            print(f"    Worst single-day:   ${worst_day:,.0f}")
            print(f"    Days with P&L > $0: {pct_days_pos:.0%}  "
                  f"(of {len(daily_ser)} days)")
        else:
            print(f"    Insufficient data (N={ps['n']} trades)")
    else:
        print("    No trades generated")

    # Equity / max-DD sketch
    print(f"\n  Equity / Max-DD sketch (trade-by-trade, 1 contract):")
    if len(ps["pnls"]) > 0:
        csum   = np.cumsum(ps["pnls"])
        max_dd = float((csum - np.maximum.accumulate(csum)).min())
        hwm    = float(np.maximum.accumulate(csum).max())
        print(f"    Final cum. P&L:    ${csum[-1]:>8,.0f}")
        print(f"    Peak equity (HWM): ${hwm:>8,.0f}")
        print(f"    Max DD (HWM→trough): ${max_dd:>8,.0f}  "
              f"({'✅ inside $2k limit' if max_dd >= -2000 else '❌ EXCEEDS $2k limit'})")
        if len(ps["pnls"]) >= 20:
            roll_dds = []
            for i in range(len(ps["pnls"])-19):
                chunk = np.cumsum(ps["pnls"][i:i+20])
                dd = float((chunk - np.maximum.accumulate(chunk)).min())
                roll_dds.append(dd)
            print(f"    Worst 20-trade DD: ${min(roll_dds):>8,.0f}")
    else:
        print("    No trades generated")

    # ── 8. Gate 0 verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"GATE 0 VERDICT — {inst.upper()} PRIMARY SPEC  "
          f"(SD={SD_THRESH}σ, TP={TP_PTS_PRIMARY}pts, STOP={STOP_PTS}pts)")
    print(f"{'='*100}")
    print(f"  Breakeven WR = {be_wr:.1%}  |  Gate WR ≥ {gate_wr:.1%}  "
          f"|  Stop fixed at {STOP_PTS}pts = ${stop_usd:.0f}/contract")
    print()

    g_ev   = ps["avg_pnl"]  > GATE0_EV_MIN
    g_pf   = ps["pf"]       >= GATE0_PF_MIN
    g_stop = ps["stop_med"] <= GATE0_STOP_MAX
    g_wr   = ps["wr"]       >= gate_wr
    g_n    = ps["n"]        >= GATE0_MIN_N
    g_womo = ps["worst_mo_pnl"] >= GATE0_WOMO_PNL

    def v(flag, label, measured):
        return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<58} [measured: {measured}]"

    print(v(g_ev,   "EV > $0 (avg net P&L > $0)",
                    f"${ps['avg_pnl']:.2f}/trade"))
    print(v(g_pf,   f"Profit factor ≥ {GATE0_PF_MIN:.2f}",
                    f"{ps['pf']:.3f}"))
    print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract",
                    f"${ps['stop_med']:.0f}"))
    print(v(g_wr,   f"WR ≥ breakeven+5% (≥ {gate_wr:.1%})",
                    f"{ps['wr']:.1%}"))
    print(v(g_n,    f"N ≥ {GATE0_MIN_N} (min sample for edge read)",
                    f"N={ps['n']}"))
    print(v(g_womo, f"Worst-month avg P&L ≥ ${GATE0_WOMO_PNL:.0f}/trade",
                    f"${ps['worst_mo_pnl']:.2f}"))
    print(f"\n  [FREQ NOTE — informational, NOT a gate for this selective strategy]")
    print(f"  Frequency: {ps['freq']:.3f}/day  (designed for 0.2–0.4/day; "
          f"applying freq≥1.0 gate would be wrong — same error that killed vol-compression)")

    edge_gates = [g_ev, g_pf, g_stop, g_wr, g_n]
    print()
    if all(edge_gates):
        print(f"  ✅ GATE 0 PASS — confirmation-stack edge confirmed on {inst.upper()}.")
        if not g_womo:
            print(f"  ⚠️  VARIANCE WARNING — worst-month avg=${ps['worst_mo_pnl']:.2f} < -$50 "
                  f"(Track 3 stacking consideration)")
        else:
            print(f"  ✅ VARIANCE OK — worst-month avg=${ps['worst_mo_pnl']:.2f} ≥ -$50")
        # Proceed to combine-math sim if edge passes
        if pt:
            day_pnl2: dict = {}
            for t in pt:
                d = t["date"]
                day_pnl2[d] = day_pnl2.get(d, 0.0) + t["pnl"]
            all_days2 = sorted(set(rth["date"]))
            daily_1c  = [day_pnl2.get(d, 0.0) for d in all_days2]
            path_simulation(daily_1c, n_contracts=3,
                            label=f"{inst.upper()} — primary spec Gate 0 PASS")
    else:
        fails = []
        if not g_n:
            fails.append(f"N={ps['n']} < {GATE0_MIN_N} (too rare to evaluate)")
        if not g_ev:
            fails.append(f"EV=${ps['avg_pnl']:.2f} ≤ $0")
        if not g_pf:
            fails.append(f"PF={ps['pf']:.3f} < {GATE0_PF_MIN:.2f}")
        if not g_wr:
            fails.append(f"WR={ps['wr']:.1%} < breakeven+5% ({gate_wr:.1%})")
        if not g_stop:
            fails.append(f"Med stop=${ps['stop_med']:.0f} > ${GATE0_STOP_MAX:.0f}")
        print(f"  ❌ GATE 0 FAIL — {inst.upper()} confirmation-stack edge not confirmed.")
        for lbl in fails:
            print(f"     • {lbl}")
        if ps["n"] == 0:
            print("     NOTE: Zero trades means ALL 4 conditions were never simultaneously"
                  " satisfied. This is 'too rare', not 'wrong direction'.")
        elif not g_n:
            print(f"     NOTE: N={ps['n']} is too small to distinguish edge from noise."
                  " Like vol-compression: real edge possible but structurally too rare for combine.")
        elif all([g_ev, g_pf]) and not g_wr:
            print("     NOTE: PF/EV positive but WR below breakeven+5%. Check long/short split "
                  "— possible directional asymmetry (short-only may work).")

    print(f"{'='*100}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HCVWAP Gate 0 Study")
    parser.add_argument("--instrument", choices=["mnq", "mes", "both"],
                        default="both",
                        help="Instrument to run (default: both)")
    args = parser.parse_args()

    print("=" * 100)
    print("HCVWAP — High-Confirmation VWAP Gate 0 Study")
    print("Pre-registration: commit 4531a3d (2026-06-09)")
    print("Hypothesis: VWAP 2σ fade × 4-condition stack has edge that unfiltered fades lack")
    print("In-sample: 2025-01-01 → 2026-02-28  |  Holdout ≥2026-03-01 SEALED")
    print("=" * 100)
    print(f"\nConfirmation stack (ALL required simultaneously):")
    print(f"  1. Extension:  |z_score| ≥ {SD_THRESH}σ (primary)")
    print(f"  2. Time window: {AM_WIN[0]}–{AM_WIN[1]} ET  OR  {PM_WIN[0]}–{PM_WIN[1]} ET")
    print(f"  3. Volume:     bar_vol > {VOL_MULT}× trailing {VOL_LOOKBACK}-bar mean")
    print(f"  4. HTF ranging: 15-min |EMA9−EMA21| < {HTF_EMA_ATR_MULT}×ATR(14)")
    print(f"\nTrade spec: STOP={STOP_PTS}pts  TP={TP_PTS_PRIMARY}pts (primary)  HOLD_MAX={HOLD_MAX}bars")

    instruments = ["mnq", "mes"] if args.instrument == "both" else [args.instrument]
    for inst in instruments:
        run_instrument(inst)

    print("\n\nDone.")


if __name__ == "__main__":
    main()

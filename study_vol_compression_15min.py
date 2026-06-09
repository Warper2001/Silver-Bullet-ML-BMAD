"""
Volatility Compression Breakout — 15-Minute Gate 0 Study
Party recommendation 2026-06-08: re-run Carson's "trapped traders release" thesis
at 15-min resolution.

The 1-min version (study_vol_compression_breakout.py) confirmed the edge is REAL
(PF=1.46 at 2:1 R/R) but frequency was only 0.15/day — too rare amid 1-min noise.
At 15-min, three consecutive quiet bars = 45 min of genuine digestion — a real
market-structure event, not microstructure noise. The key open question is whether
frequency clears 1.0/day at a moderate compression threshold.

Two refinements vs. 1-min study (party 2026-06-08):
  1. RTH-only ATR: ATR(20) computed on 15-min RTH bars only.  The 1-min study computed
     ATR on all-session bars, which inflated thresholds slightly via overnight vol.
  2. Victor's rolling-5-day diagnostic: worst rolling 5-day P&L stretch reported as a
     combine-variance check. The $2k trailing DD taxes variance, not just expectancy.

Setup class (same logic, 15-min scale):
  1. Compression zone: MIN_BARS consecutive 15-min RTH bars from the SAME session
     where bar_range ≤ COMP_FRAC × ATR(20) AND zone_width ≤ 2×COMP_FRAC×ATR
  2. Breakout bar: first bar closing above zone_high or below zone_low
  3. Entry: close of the breakout bar
  4. Stop: zone edge ± STOP_BUFFER × ATR  (direction-appropriate)
  5. Target: TP_MULT × stop_distance in breakout direction
  6. Session hard-close at 15:55 ET; HOLD_MAX=4 bars (1h); one trade at a time;
     skip setup if stop > $150/contract

Structural justification (Carson / Market Profile):
  Tight consecutive bars = trapped participants (stops bunched at zone edges).
  Breakout triggers a stop cascade — the trade rides the cascade.
  Direction-agnostic; not momentum-fighting or trend-fading.

Primary spec (frozen before reading results):
  COMP_FRAC=0.50, MIN_BARS=3, TP_MULT=2.0, STOP_BUFFER=0.25
  Rationale: 50% ATR ≈ solid quiet-bar threshold at 15-min scale (vs 30% at 1-min);
  3 bars = 45 min genuine digestion = minimum meaningful cluster at this resolution.

Grid (sensitivity only — NOT cherry-picked post-hoc):
  COMP_FRAC ∈ {0.40, 0.50, 0.60} × MIN_BARS ∈ {2, 3, 4}
TP sensitivity: 1.5, 2.0, 3.0 reported for primary spec only.

In-sample: 2025-01-01 → 2026-02-28 (same window as all prior Gate 0 studies)
Gate 0 thresholds (unchanged from prior studies):
  WR ≥ 50%, freq ≥ 1.0/day, median stop ≤ $150/contract, worst-month WR ≥ 35%
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_1MIN_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_1MIN_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

ATR_WIN        = 20
HOLD_MAX       = 4           # bars after entry (4 × 15min = 1h, same wall-clock as 60 × 1-min)
SESSION_CLOSE  = "15:55"
RTH_START      = "09:30"
RTH_END        = "15:55"     # last 1-min bar in data
MNQ_PV         = 2.0
COMMISSION     = 4.80
STOP_CAP_USD   = 150.0
STOP_BUFFER    = 0.25        # × ATR beyond zone edge

COMP_FRACS     = [0.40, 0.50, 0.60]
MIN_BARS_S     = [2, 3, 4]
TP_MULTS       = [1.5, 2.0, 3.0]

PRIMARY_CF     = 0.50
PRIMARY_MB     = 3
PRIMARY_TP     = 2.0

GATE0_WR_MIN   = 0.50
GATE0_FREQ_MIN = 1.0
GATE0_STOP_MAX = 150.0
GATE0_WORST_MO = 0.35

ROLLING_5D_TARGET = 5 * 150.0   # Victor's threshold: $750 = 5 qualifying days × $150

# ── load 1-min bars and resample to 15-min RTH ────────────────────────────────
def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()

print("Loading 1-min bars…")
bars_1min = pd.concat([load_et(MNQ_1MIN_2025), load_et(MNQ_1MIN_2026)])
bars_1min = bars_1min[~bars_1min.index.duplicated(keep="first")]
bars_1min = bars_1min["2025-01-01":"2026-02-28"]

# Filter to RTH first — this is the Carson ATR fix: no overnight bars means
# ATR will be computed purely on RTH data.
rth_1min = bars_1min.between_time(RTH_START, RTH_END).copy()

# Resample to 15-min. 09:30 aligns naturally with 15-min epoch boundaries
# (570 min from midnight / 15 = 38, exactly divisible), so resample bins land on
# 09:30, 09:45, 10:00, …, 15:45. Empty overnight bins are removed by dropna.
rth = (rth_1min
       .resample("15min")
       .agg({"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"})
       .dropna(subset=["close"]))

# RTH-only ATR: rolling on the compacted (no-overnight) 15-min series
rth["tr"]  = rth["high"] - rth["low"]
rth["atr"] = rth["tr"].rolling(ATR_WIN).mean()
rth["date"] = rth.index.normalize().date

n_days = len(set(rth["date"]))
print(f"  1-min RTH bars:  {len(rth_1min):,}")
print(f"  15-min RTH bars: {len(rth):,}  (~{len(rth)/n_days:.1f}/day)  |  {n_days} trading days")
print(f"  ATR(20) median:  {rth['atr'].median():.2f} pts  "
      f"(≈${rth['atr'].median() * MNQ_PV:.0f}/contract/bar)")


# ── compression detection + simulation ────────────────────────────────────────
def run_simulation(comp_frac: float, min_bars: int, tp_mult: float):
    trades  = []
    active  = None
    hold_count = 0

    hi_arr   = rth["high"].values
    lo_arr   = rth["low"].values
    cl_arr   = rth["close"].values
    atr_arr  = rth["atr"].values
    ts_arr   = rth.index
    date_arr = rth["date"].values

    for k in range(min_bars, len(rth)):
        ts    = ts_arr[k]
        atr_k = atr_arr[k]
        if np.isnan(atr_k) or atr_k <= 0:
            continue

        comp_threshold = comp_frac * atr_k
        stop_buf       = STOP_BUFFER * atr_k

        # ── manage active trade ──────────────────────────────────────────────
        if active is not None:
            hi_k = hi_arr[k]; lo_k = lo_arr[k]
            hit_tp   = (active["dir"] ==  1 and hi_k >= active["tp"]) or \
                       (active["dir"] == -1 and lo_k <= active["tp"])
            hit_stop = (active["dir"] ==  1 and lo_k <= active["stop"]) or \
                       (active["dir"] == -1 and hi_k >= active["stop"])
            at_close   = ts.strftime("%H:%M") >= SESSION_CLOSE
            day_change = date_arr[k] != active["date"]
            hold_count += 1

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": active["tp"],
                                "pnl": pnl, "win": True, "reason": "TP"})
                active = None; hold_count = 0
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": active["stop"],
                                "pnl": pnl, "win": False, "reason": "STOP"})
                active = None; hold_count = 0
            elif at_close or hold_count >= HOLD_MAX or day_change:
                ep  = cl_arr[k]
                pnl = (ep - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                reason = "CLOSE" if at_close else ("DAYEND" if day_change else "TIME")
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0, "reason": reason})
                active = None; hold_count = 0
            continue   # don't look for new entry while in trade

        # ── require compression window within the same session ──────────────
        # Key correctness guard: at 15-min resolution, k-min_bars bars could
        # easily span a session boundary (e.g. k=2 → window includes yesterday's
        # last bar). Reject cross-session windows.
        window_dates = date_arr[k - min_bars:k]
        if not (window_dates == window_dates[0]).all():
            continue

        # ── detect compression zone in the prior min_bars bars ──────────────
        window_hi     = hi_arr[k - min_bars:k]
        window_lo     = lo_arr[k - min_bars:k]
        window_ranges = window_hi - window_lo
        if not (window_ranges <= comp_threshold).all():
            continue   # at least one bar too wide

        zone_high  = window_hi.max()
        zone_low   = window_lo.min()
        zone_width = zone_high - zone_low
        if zone_width > 2.0 * comp_threshold:
            continue   # bars individually quiet but spread too far — not a tight cluster

        # ── check if current bar k breaks out of the zone ───────────────────
        cl_k = cl_arr[k]
        hi_k = hi_arr[k]
        lo_k = lo_arr[k]

        if cl_k > zone_high:
            direction = 1     # long breakout
        elif cl_k < zone_low:
            direction = -1    # short breakout
        else:
            continue          # no clean breakout yet

        entry     = cl_k
        stop_p    = (zone_low  - stop_buf) if direction == 1 else (zone_high + stop_buf)
        stop_dist = abs(entry - stop_p)
        stop_usd  = stop_dist * MNQ_PV
        if stop_usd > STOP_CAP_USD:
            continue   # skip if combine stop cap exceeded

        tp_p = entry + direction * stop_dist * tp_mult

        active = {"dir": direction, "entry": entry,
                  "tp": tp_p, "stop": stop_p,
                  "stop_dist": stop_dist, "stop_usd": stop_usd,
                  "zone_w": zone_width, "atr": atr_k,
                  "date": date_arr[k],
                  "month": ts.to_period("M")}
        hold_count = 0

    if active:
        pnl = (cl_arr[-1] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades):
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, stop_p75=0.0, worst_mo=0.0,
                    pnls=np.array([]), mo={},
                    exit_tp=0, exit_stop=0, exit_time=0)
    n    = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    stops = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_w / max(1e-9, gross_l)
    mo: dict = {}
    for t in trades:
        m = t["month"]
        mo.setdefault(m, [0, 0])
        mo[m][0 if t["win"] else 1] += 1
    worst_mo = min((w/(w+l) if w+l else 0) for w, l in mo.values()) if mo else 0.0
    return dict(n=n, wr=wins/n, freq=n/n_days, avg_pnl=pnls.mean(),
                pf=pf, stop_med=float(np.median(stops)),
                stop_p75=float(np.percentile(stops, 75)),
                worst_mo=worst_mo, pnls=pnls, mo=mo,
                exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
                exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
                exit_time=sum(1 for t in trades if t["reason"]
                               in ("TIME", "CLOSE", "DAYEND", "END")))


# ── main grid (TP=2.0, vary COMP_FRAC × MIN_BARS) ────────────────────────────
print(f"\n{'='*90}")
print(f"15-MIN COMPRESSION BREAKOUT GRID  "
      f"(TP={PRIMARY_TP}×, stop=zone_edge+{STOP_BUFFER}×ATR, cap=${STOP_CAP_USD})")
print(f"{'='*90}")
print(f"  {'CF':>5}  {'MinB':>5}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed$':>9}  {'WorstMo':>8}")
print(f"  {'----':>5}  {'----':>5}  {'---':>5}  {'-------':>8}  {'---':>7}  "
      f"{'----':>5}  {'------':>8}  {'--------':>9}  {'-------':>8}")

grid_res = {}
for cf in COMP_FRACS:
    for mb in MIN_BARS_S:
        t = run_simulation(cf, mb, PRIMARY_TP)
        s = summarise(t)
        grid_res[(cf, mb)] = (t, s)
        prim   = " ◀ PRIMARY" if cf == PRIMARY_CF and mb == PRIMARY_MB else ""
        wr_f   = "✅" if s["wr"]   >= GATE0_WR_MIN   else "❌"
        freq_f = "✅" if s["freq"] >= GATE0_FREQ_MIN  else "❌"
        print(f"  {cf:>4.2f}  {mb:>5}  {s['n']:>5}  "
              f"{s['freq']:>6.2f}/d{freq_f}  "
              f"{s['wr']:>7.1%}{wr_f}  {s['pf']:>5.2f}  "
              f"${s['avg_pnl']:>6.2f}  ${s['stop_med']:>7.0f}  "
              f"{s['worst_mo']:>8.1%}{prim}")

# ── TP sensitivity at primary spec ────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"TP SENSITIVITY  (COMP_FRAC={PRIMARY_CF}, MIN_BARS={PRIMARY_MB})")
print(f"{'='*90}")
print(f"  {'TP×':>4}  {'BEven WR':>9}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>8}")
for tpm in TP_MULTS:
    t = run_simulation(PRIMARY_CF, PRIMARY_MB, tpm)
    s = summarise(t)
    # Breakeven WR: WR × (tpm×S - C) + (1-WR) × (-S - C) = 0  → WR = (S+C)/((tpm+1)×S)
    be_wr = ((s["stop_med"] + COMMISSION) / ((tpm + 1) * s["stop_med"])
             if s["stop_med"] > 0 else 0.0)
    wr_f = "✅" if s["avg_pnl"] > 0 else "❌"
    prim = " ◀ PRIMARY" if tpm == PRIMARY_TP else ""
    print(f"  {tpm:>3.1f}×  {be_wr:>9.1%}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
          f"{s['wr']:>7.1%}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}{wr_f}  "
          f"{s['worst_mo']:>8.1%}{prim}")

# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[(PRIMARY_CF, PRIMARY_MB)]

print(f"\n{'='*90}")
print(f"PRIMARY SPEC DEEP DIVE  "
      f"(COMP_FRAC={PRIMARY_CF}, MIN_BARS={PRIMARY_MB}, TP={PRIMARY_TP}×)")
print(f"{'='*90}")

print(f"\n  Funnel:")
print(f"    15-min RTH bars:          {len(rth):,}")
print(f"    Trades taken:             {ps['n']}  ({ps['freq']:.2f}/day)")
if ps['n'] > 0:
    print(f"    Exit breakdown:           "
          f"TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
          f"TIME/CLOSE/DAYEND={ps['exit_time']}")

print(f"\n  Performance:")
print(f"    Win rate:                 {ps['wr']:.1%}")
print(f"    Profit factor:            {ps['pf']:.2f}")
print(f"    Avg net P&L:              ${ps['avg_pnl']:.2f}/contract")
total_pnl = float(ps["pnls"].sum()) if len(ps["pnls"]) > 0 else 0.0
print(f"    Total P&L (1 MNQ):        ${total_pnl:.0f}  over {n_days} days")
print(f"    Median stop:              ${ps['stop_med']:.0f}/contract")
print(f"    75th-pct stop:            ${ps['stop_p75']:.0f}/contract")
print(f"    Worst-month WR:           {ps['worst_mo']:.1%}")

# ── Victor's rolling-5-day variance diagnostic ────────────────────────────────
print(f"\n  Victor's rolling-5-day variance check (combine DD guard):")
if pt:
    # Build per-day P&L dict; trading days with no trades contribute $0
    day_pnl: dict = {}
    for t in pt:
        d = t["date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + t["pnl"]

    all_trading_days = sorted(set(rth["date"]))
    daily_series = [day_pnl.get(d, 0.0) for d in all_trading_days]

    if len(daily_series) >= 5:
        rolling5  = [sum(daily_series[i:i + 5]) for i in range(len(daily_series) - 4)]
        worst5    = min(rolling5)
        best5     = max(rolling5)
        median5   = float(np.median(rolling5))
        pct_neg   = sum(1 for x in rolling5 if x < 0) / len(rolling5)
        worst_day = min(daily_series)

        print(f"    Worst rolling 5-day P&L:  ${worst5:.0f}  "
              f"({'✅' if worst5 >= 0 else '❌'} {'positive' if worst5 >= 0 else 'negative'})")
        print(f"    Best  rolling 5-day P&L:  ${best5:.0f}")
        print(f"    Median rolling 5-day P&L: ${median5:.0f}")
        print(f"    % of 5-day windows < $0:  {pct_neg:.0%}")
        print(f"    Worst single-day P&L:     ${worst_day:.0f}  "
              f"({'vs $2k DD budget' if worst_day < 0 else 'positive day'})")
        print(f"    (target ≥$0 worst-5d = no losing 5-day stretches)")
    else:
        print(f"    Insufficient sample for 5-day rolling (N={ps['n']} trades)")
else:
    print(f"    No trades generated — cannot compute rolling P&L")

# ── by-month table ────────────────────────────────────────────────────────────
print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'freq/d':>8}")
mo_pnl: dict = {}
for t in pt:
    m = t["month"]
    mo_pnl.setdefault(m, [])
    mo_pnl[m].append(t["pnl"])

for m in sorted(ps["mo"]):
    w, l = ps["mo"][m]
    n_mo = w + l
    mwr  = w / n_mo if n_mo else 0
    avg  = float(np.mean(mo_pnl.get(m, [0])))
    mo_bars = rth[rth.index.to_period("M") == m]
    mo_days = len(set(mo_bars["date"]))
    if n_mo < 5:
        flag = "⚠️ N<5"
    elif mwr < GATE0_WORST_MO:
        flag = "❌"
    else:
        flag = "✅"
    print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
          f"{n_mo/max(1, mo_days):>7.2f}/d  {flag}")

# ── gate 0 verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  "
      f"(CF={PRIMARY_CF}, MB={PRIMARY_MB}, TP={PRIMARY_TP}×)")
print(f"{'='*90}")

g_wr   = ps["wr"]       >= GATE0_WR_MIN
g_freq = ps["freq"]     >= GATE0_FREQ_MIN
g_stop = ps["stop_med"] <= GATE0_STOP_MAX
g_womo = ps["worst_mo"] >= GATE0_WORST_MO


def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<46} [measured: {measured}]"


print(v(g_wr,   f"Win rate ≥ {GATE0_WR_MIN:.0%}",              f"{ps['wr']:.1%}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day",           f"{ps['freq']:.2f}/day"))
print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract", f"${ps['stop_med']:.0f}"))
print(v(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO:.0%}",      f"{ps['worst_mo']:.1%}"))

gate_pass = all([g_wr, g_freq, g_stop, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — positive EV, adequate frequency, stable variance.")
    print("     Proceed to Stage 1: pre-registration before any holdout access.")
else:
    fails = [(label, fail) for label, fail in [
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}",               not g_wr),
        (f"Freq {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}",          not g_freq),
        (f"Med stop ${ps['stop_med']:.0f} > ${GATE0_STOP_MAX:.0f}", not g_stop),
        (f"Worst-mo WR {ps['worst_mo']:.1%} < {GATE0_WORST_MO:.0%}", not g_womo),
    ] if fail]
    print(f"  ❌ GATE 0 FAIL.")
    for label, _ in fails:
        print(f"     • {label}")

print(f"{'='*90}")

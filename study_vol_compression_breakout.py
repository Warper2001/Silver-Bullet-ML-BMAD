"""
Volatility Compression Breakout — Gate 0 Study
Carson's "Trapped Traders Release" thesis (BMAD party 2026-06-08)

Setup class:
  1. Compression detected: N consecutive RTH bars where bar_range ≤ COMP_FRAC × ATR(20)
     AND all bars share a common zone (max_high - min_low ≤ 2× COMP_FRAC × ATR)
  2. Breakout bar: first bar AFTER the compression whose close breaks ABOVE zone_high
     or BELOW zone_low
  3. Entry: close of the breakout bar
  4. Stop: zone edge in the opposite direction ± STOP_BUFFER × ATR
  5. Target: TP_MULT × stop_distance in breakout direction
  6. Session hard-close at 15:55 ET; one trade at a time; skip if stop > $150

Structural justification (Carson / Market Profile):
  Tight consecutive bars = trapped participants (stops bunched at zone edges).
  Breakout triggers a stop cascade — the trade rides the cascade.
  Direction-agnostic (works in trending AND ranging regimes).

Primary spec (frozen before reading results):
  COMP_FRAC=0.30, MIN_BARS=3, TP_MULT=2.0, STOP_BUFFER=0.25
  Rationale: 30% ATR ≈ 4 pts at median vol; 3 bars = minimum meaningful cluster.

Grid: COMP_FRAC ∈ {0.20, 0.30, 0.40} × MIN_BARS ∈ {3, 4, 5}
TP sensitivity: 1.5, 2.0, 3.0 reported for primary COMP/BARS spec only.

In-sample: 2025-01-01 → 2026-02-28 (full window, no ES data dependency)
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_PATH   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

ATR_WIN        = 20
HOLD_MAX       = 60
SESSION_CLOSE  = "15:55"
RTH_START      = "09:30"
RTH_SKIP_MINS  = 5          # ignore first 5 min (09:30–09:34) — opening noise
MNQ_PV         = 2.0
COMMISSION     = 4.80
STOP_CAP_USD   = 150.0
STOP_BUFFER    = 0.25       # × ATR added beyond zone edge

COMP_FRACS     = [0.20, 0.30, 0.40]
MIN_BARS_S     = [3, 4, 5]
TP_MULTS       = [1.5, 2.0, 3.0]

PRIMARY_CF     = 0.30
PRIMARY_MB     = 3
PRIMARY_TP     = 2.0

GATE0_WR_MIN   = 0.50
GATE0_FREQ_MIN = 1.0
GATE0_STOP_MAX = 150.0
GATE0_WORST_MO = 0.35

# ── load ──────────────────────────────────────────────────────────────────────
def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()

print("Loading bars…")
bars = pd.concat([load_et(MNQ_PATH), load_et(MNQ_2026)])
bars = bars[~bars.index.duplicated(keep="first")]
bars = bars["2025-01-01":"2026-02-28"]

# ATR (20-bar mean true range) — computed on full data for warmup
bars["tr"]  = bars["high"] - bars["low"]
bars["atr"] = bars["tr"].rolling(ATR_WIN).mean()

# RTH filter (exclude opening spike window)
rth_full = bars.between_time(RTH_START, SESSION_CLOSE).copy()
rth = rth_full[rth_full.index.time >= pd.to_datetime(
    f"{RTH_START.split(':')[0]}:{int(RTH_START.split(':')[1])+RTH_SKIP_MINS:02d}"
).time()].copy()

n_days  = rth.index.normalize().nunique()
print(f"  RTH bars (post-skip): {len(rth):,}  |  {n_days} days")

# ── compression detection + simulation ───────────────────────────────────────
def run_simulation(comp_frac: float, min_bars: int, tp_mult: float):
    trades = []
    active = None
    hold_count = 0

    hi_arr   = rth["high"].values
    lo_arr   = rth["low"].values
    cl_arr   = rth["close"].values
    atr_arr  = rth["atr"].values
    ts_arr   = rth.index
    date_arr = rth.index.date

    for k in range(min_bars, len(rth)):
        ts    = ts_arr[k]
        atr_k = atr_arr[k]
        if np.isnan(atr_k) or atr_k <= 0:
            continue

        comp_threshold = comp_frac * atr_k
        stop_buf       = STOP_BUFFER * atr_k

        # ── manage active trade ──
        if active is not None:
            hi_k = hi_arr[k]; lo_k = lo_arr[k]
            hit_tp   = (active["dir"] ==  1 and hi_k >= active["tp"]) or \
                       (active["dir"] == -1 and lo_k <= active["tp"])
            hit_stop = (active["dir"] ==  1 and lo_k <= active["stop"]) or \
                       (active["dir"] == -1 and hi_k >= active["stop"])
            at_close  = ts.strftime("%H:%M") >= SESSION_CLOSE
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
            elif at_close or hold_count >= HOLD_MAX:
                ep = cl_arr[k]
                pnl = (ep - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0,
                                "reason": "CLOSE" if at_close else "TIME"})
                active = None; hold_count = 0
            continue   # don't check for new entry while in trade

        # ── detect compression zone in past min_bars bars ──
        # All min_bars consecutive bars (k-min_bars .. k-1) must have range ≤ threshold
        window_hi = hi_arr[k-min_bars:k]
        window_lo = lo_arr[k-min_bars:k]
        window_ranges = window_hi - window_lo
        if not (window_ranges <= comp_threshold).all():
            continue

        zone_high = window_hi.max()
        zone_low  = window_lo.min()
        zone_width = zone_high - zone_low
        # zone must not be wider than 2× threshold (bars must cluster, not spread)
        if zone_width > 2.0 * comp_threshold:
            continue

        # ── check if current bar k breaks out ──
        cl_k = cl_arr[k]
        hi_k = hi_arr[k]
        lo_k = lo_arr[k]

        if cl_k > zone_high:
            direction = 1    # long
        elif cl_k < zone_low:
            direction = -1   # short
        else:
            continue         # no breakout yet

        entry     = cl_k
        stop_p    = (zone_low - stop_buf) if direction == 1 else (zone_high + stop_buf)
        stop_dist = abs(entry - stop_p)
        stop_usd  = stop_dist * MNQ_PV
        if stop_usd > STOP_CAP_USD:
            continue

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
                    stop_med=0.0, stop_p75=0.0, worst_mo=0.0, pnls=np.array([]))
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
                exit_tp=sum(1 for t in trades if t["reason"]=="TP"),
                exit_stop=sum(1 for t in trades if t["reason"]=="STOP"),
                exit_time=sum(1 for t in trades if t["reason"] in ("TIME","CLOSE","END")))


# ── main grid (TP=2.0, vary comp/bars) ───────────────────────────────────────
print(f"\n{'='*84}")
print(f"COMPRESSION BREAKOUT GRID  (TP=2.0×, stop=zone_edge+0.25×ATR, cap=${STOP_CAP_USD})")
print(f"{'='*84}")
print(f"  {'CF':>5}  {'MinB':>5}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed$':>9}  {'WorstMo':>8}")
print(f"  {'----':>5}  {'----':>5}  {'---':>5}  {'------':>7}  {'---':>7}  "
      f"{'----':>5}  {'------':>8}  {'--------':>9}  {'-------':>8}")

grid_res = {}
for cf in COMP_FRACS:
    for mb in MIN_BARS_S:
        t = run_simulation(cf, mb, PRIMARY_TP)
        s = summarise(t)
        grid_res[(cf, mb)] = (t, s)
        prim = " ◀ PRIMARY" if cf == PRIMARY_CF and mb == PRIMARY_MB else ""
        wr_f = "✅" if s["wr"] >= GATE0_WR_MIN else "❌"
        print(f"  {cf:>4.2f}  {mb:>5}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
              f"{s['wr']:>7.1%}{wr_f}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}  "
              f"${s['stop_med']:>7.0f}  {s['worst_mo']:>8.1%}{prim}")

# ── TP sensitivity at primary spec ───────────────────────────────────────────
print(f"\n{'='*84}")
print(f"TP SENSITIVITY  (COMP_FRAC={PRIMARY_CF}, MIN_BARS={PRIMARY_MB})")
print(f"{'='*84}")
print(f"  {'TP×':>4}  {'BEven WR':>9}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>8}")
for tpm in TP_MULTS:
    t = run_simulation(PRIMARY_CF, PRIMARY_MB, tpm)
    s = summarise(t)
    tp_usd   = s["stop_med"] * tpm    # approximate
    be_wr    = (s["stop_med"] + COMMISSION) / (s["stop_med"] * (1 + tpm))
    wr_f = "✅" if s["avg_pnl"] > 0 else "❌"
    prim = " ◀ PRIMARY" if tpm == PRIMARY_TP else ""
    print(f"  {tpm:>3.1f}×  {be_wr:>9.1%}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
          f"{s['wr']:>7.1%}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}{wr_f}  "
          f"{s['worst_mo']:>8.1%}{prim}")

# ── primary spec deep dive ────────────────────────────────────────────────────
pt, ps = grid_res[(PRIMARY_CF, PRIMARY_MB)]

print(f"\n{'='*84}")
print(f"PRIMARY SPEC DEEP DIVE  "
      f"(COMP_FRAC={PRIMARY_CF}, MIN_BARS={PRIMARY_MB}, TP={PRIMARY_TP}×)")
print(f"{'='*84}")
print(f"\n  Funnel:")
print(f"    RTH bars:            {len(rth):,}")
total_comp = sum(1 for k in range(PRIMARY_MB, len(rth))
                 if not rth['atr'].iloc[k] != rth['atr'].iloc[k]   # not nan
                 and (rth['high'].iloc[k-PRIMARY_MB:k] - rth['low'].iloc[k-PRIMARY_MB:k]).max()
                     <= PRIMARY_CF * rth['atr'].iloc[k])
print(f"    Compression zones:   ~{total_comp:,}  (bars following N compressed bars)")
print(f"    Trades taken:        {ps['n']}  ({ps['freq']:.2f}/day)")
print(f"    Exit: TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
      f"TIME/CLOSE={ps['exit_time']}")
print(f"\n  Performance:")
print(f"    Win rate:            {ps['wr']:.1%}")
print(f"    Profit factor:       {ps['pf']:.2f}")
print(f"    Avg net P&L:         ${ps['avg_pnl']:.2f}/contract")
print(f"    Total P&L (1 MNQ):   ${ps['pnls'].sum():.0f}  over {n_days} days")
print(f"    Median stop:         ${ps['stop_med']:.0f}/contract")
print(f"    75th-pct stop:       ${ps['stop_p75']:.0f}/contract")
print(f"    Worst-month WR:      {ps['worst_mo']:.1%}")

print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'freq/d':>8}")
mo_pnl: dict = {}
for t in pt:
    m = t["month"]
    mo_pnl.setdefault(m, [])
    mo_pnl[m].append(t["pnl"])
for m in sorted(ps["mo"]):
    w, l = ps["mo"][m]
    n = w + l
    mwr = w / n if n else 0
    avg = np.mean(mo_pnl.get(m, [0]))
    mo_bars = rth[rth.index.to_period("M") == m]
    mo_days = mo_bars.index.normalize().nunique()
    flag = "❌" if mwr < GATE0_WORST_MO else "✅"
    print(f"  {str(m):<10}  {n:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
          f"{n/max(1, mo_days):>7.2f}/d  {flag}")

# ── gate 0 verdict ────────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print(f"GATE 0 VERDICT — PRIMARY SPEC "
      f"(CF={PRIMARY_CF}, MB={PRIMARY_MB}, TP={PRIMARY_TP}×)")
print(f"{'='*84}")
g_wr   = ps["wr"]       >= GATE0_WR_MIN
g_freq = ps["freq"]     >= GATE0_FREQ_MIN
g_stop = ps["stop_med"] <= GATE0_STOP_MAX
g_womo = ps["worst_mo"] >= GATE0_WORST_MO

def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<42} [measured: {measured}]"

print(v(g_wr,   f"Win rate ≥ {GATE0_WR_MIN:.0%}", f"{ps['wr']:.1%}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day", f"{ps['freq']:.2f}/day"))
print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract", f"${ps['stop_med']:.0f}"))
print(v(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO:.0%}", f"{ps['worst_mo']:.1%}"))

gate_pass = all([g_wr, g_freq, g_stop, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — proceed to pre-registration and full combine backtest.")
else:
    fails = [(l, f) for l, f in [
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}",           not g_wr),
        (f"Freq {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}",      not g_freq),
        (f"Med stop ${ps['stop_med']:.0f} > ${GATE0_STOP_MAX}",not g_stop),
        (f"Worst-mo WR {ps['worst_mo']:.1%} < {GATE0_WORST_MO:.0%}", not g_womo),
    ] if f]
    print(f"  ❌ GATE 0 FAIL.")
    for l, _ in fails:
        print(f"     • {l}")
print(f"{'='*84}")

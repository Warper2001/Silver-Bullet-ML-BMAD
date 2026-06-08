"""
Volume Profile POC / Value Area — Gate 0 Study
Carson's Idea 4: "Treat it like a queuing problem" (BMAD party 2026-06-08)

Setup class:
  Yesterday's RTH volume profile → POC (max-volume price) + Value Area (70% of vol)
  Strategy: fade extensions beyond VAH/VAL back toward POC
    - Long: first bar close below VAL  (price dropped outside value area)
    - Short: first bar close above VAH (price rallied outside value area)
  Entry:  close of the trigger bar
  Stop:   running extreme beyond the VA edge + 0.25×ATR
  Target: yesterday's POC (institutional "fair value" magnet)
  One trade at a time; session close (15:55 ET) forces exit

Structural justification:
  Value area = price range containing 70% of yesterday's traded volume.
  Institutions with inventory from yesterday anchor to those levels.
  Extensions outside the VA represent price discovery away from consensus —
  the POC acts as a gravitational pull back to institutional fair value.

Primary spec (frozen before reading results):
  VALUE_AREA_PCT=0.70, min_vah_val_spread=5pts, TP=POC, STOP_BUFFER=0.25×ATR

Grid:
  VALUE_AREA_PCT ∈ {0.60, 0.70, 0.80}  (loosens/tightens the value area)
  TP variants: fixed POC target + R-multiple alternatives
In-sample: 2025-01-01 → 2026-02-28
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

MNQ_PATH  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

ATR_WIN        = 20
TICK           = 0.25
SESSION_CLOSE  = "15:55"
RTH_START      = "09:30"
MNQ_PV         = 2.0
COMMISSION     = 4.80
STOP_CAP_USD   = 150.0
STOP_BUFFER    = 0.25      # × ATR beyond VA edge
HOLD_MAX       = 60

VA_PCTS        = [0.60, 0.70, 0.80]
PRIMARY_VA     = 0.70
MIN_VA_SPREAD  = 5.0       # minimum pts between VAH and VAL (else skip)

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
bars["tr"]  = bars["high"] - bars["low"]
bars["atr"] = bars["tr"].rolling(ATR_WIN).mean()

rth_all = bars.between_time(RTH_START, SESSION_CLOSE).copy()
rth_all["date"] = rth_all.index.date
n_days = rth_all["date"].nunique()
print(f"  RTH bars: {len(rth_all):,}  |  {n_days} days")

# ── build daily volume profiles ───────────────────────────────────────────────
def build_vp(day_bars, va_pct):
    """Build POC/VAH/VAL from one day's RTH bars.
    Each bar's volume is distributed across its high-low range in TICK increments.
    """
    vp = defaultdict(float)
    for _, row in day_bars.iterrows():
        rng_pts  = row["high"] - row["low"]
        if rng_pts < TICK:
            level = round(row["close"] / TICK) * TICK
            vp[level] += row["volume"]
            continue
        levels = np.arange(row["low"], row["high"] + TICK / 2, TICK)
        vol_per = row["volume"] / len(levels)
        for lv in levels:
            vp[round(lv / TICK) * TICK] += vol_per

    if not vp:
        return None, None, None

    poc = max(vp, key=vp.get)
    total_vol = sum(vp.values())
    va_target = total_vol * va_pct

    sorted_prices = sorted(vp.keys())
    poc_idx = sorted_prices.index(poc)
    up_idx = dn_idx = poc_idx
    va_vol = vp[poc]

    while va_vol < va_target:
        up_ok = up_idx + 1 < len(sorted_prices)
        dn_ok = dn_idx - 1 >= 0
        if not up_ok and not dn_ok:
            break
        up_v = vp[sorted_prices[up_idx + 1]] if up_ok else 0
        dn_v = vp[sorted_prices[dn_idx - 1]] if dn_ok else 0
        if up_v >= dn_v and up_ok:
            up_idx += 1; va_vol += up_v
        elif dn_ok:
            dn_idx -= 1; va_vol += dn_v
        else:
            break

    return poc, sorted_prices[up_idx], sorted_prices[dn_idx]   # poc, vah, val

print("Building volume profiles…")
daily_vp = {}   # date → (poc, vah, val) for va_pct=PRIMARY_VA
dates = sorted(rth_all["date"].unique())
for d in dates:
    day_bars = rth_all[rth_all["date"] == d]
    poc, vah, val = build_vp(day_bars, PRIMARY_VA)
    daily_vp[d] = (poc, vah, val)

# also build for other VA pcts
daily_vp_all = {va: {} for va in VA_PCTS}
for va in VA_PCTS:
    for d in dates:
        day_bars = rth_all[rth_all["date"] == d]
        poc, vah, val = build_vp(day_bars, va)
        daily_vp_all[va][d] = (poc, vah, val)
print(f"  Volume profiles built for {len(dates)} days")

# ── simulation ────────────────────────────────────────────────────────────────
def run_simulation(va_pct, tp_mode="poc", tp_r_mult=None):
    """
    tp_mode: "poc" = target yesterday's POC level
             "r_mult" = target tp_r_mult × stop_distance from entry
    """
    vp_map = daily_vp_all[va_pct]
    trades = []
    active = None
    hold_count = 0

    hi_arr  = rth_all["high"].values
    lo_arr  = rth_all["low"].values
    cl_arr  = rth_all["close"].values
    atr_arr = rth_all["atr"].values
    ts_arr  = rth_all.index
    date_arr = rth_all["date"].values

    prev_date = None
    poc = vah = val = None

    for k in range(len(rth_all)):
        ts     = ts_arr[k]
        d      = date_arr[k]
        cl_k   = cl_arr[k]
        hi_k   = hi_arr[k]
        lo_k   = lo_arr[k]
        atr_k  = atr_arr[k]

        # load previous day's VP levels on day change
        if d != prev_date:
            prev_date = d
            active = None; hold_count = 0
            # find yesterday's date
            d_idx = dates.index(d) if d in dates else -1
            if d_idx > 0:
                prev_d = dates[d_idx - 1]
                lvls = vp_map.get(prev_d, (None, None, None))
                poc, vah, val = lvls
            else:
                poc = vah = val = None

        if poc is None or vah is None or val is None:
            continue
        if np.isnan(atr_k) or atr_k <= 0:
            continue
        if (vah - val) < MIN_VA_SPREAD:
            continue

        stop_buf = STOP_BUFFER * atr_k

        # ── manage active trade ──
        if active is not None:
            hit_tp = hit_stop = False
            if active["dir"] == 1:    # long: bullish fade below VAL
                hit_tp   = lo_k <= active["tp"] if active["tp"] < active["entry"] else hi_k >= active["tp"]
                # POC is above entry for long: price rallied to poc
                hit_tp   = hi_k >= active["tp"]
                hit_stop = lo_k <= active["stop"]
            else:                     # short: bearish fade above VAH
                hit_tp   = lo_k <= active["tp"]
                hit_stop = hi_k >= active["stop"]

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
                ep = cl_k
                pnl = (ep - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0,
                                "reason": "CLOSE" if at_close else "TIME"})
                active = None; hold_count = 0
            continue

        # ── detect new entry ──
        direction = None
        if cl_k > vah:               # price extended above VAH → short fade back to POC
            if poc < cl_k:           # POC must be below entry (otherwise no target)
                direction = -1
        elif cl_k < val:             # price extended below VAL → long fade back to POC
            if poc > cl_k:           # POC must be above entry
                direction = 1

        if direction is None:
            continue

        entry = cl_k
        if direction == -1:          # short above VAH
            # stop = running high from here + buffer
            stop_p  = hi_k + stop_buf
            tp_p    = poc if tp_mode == "poc" else entry - abs(entry - stop_p) * tp_r_mult
        else:                        # long below VAL
            stop_p  = lo_k - stop_buf
            tp_p    = poc if tp_mode == "poc" else entry + abs(entry - stop_p) * tp_r_mult

        stop_dist = abs(entry - stop_p)
        stop_usd  = stop_dist * MNQ_PV
        if stop_usd > STOP_CAP_USD:
            continue
        # sanity: target must be profitable direction
        if direction == 1 and tp_p <= entry:
            continue
        if direction == -1 and tp_p >= entry:
            continue

        active = {"dir": direction, "entry": entry,
                  "tp": tp_p, "stop": stop_p,
                  "stop_dist": stop_dist, "stop_usd": stop_usd,
                  "vah": vah, "val": val, "poc": poc,
                  "date": d, "month": ts.to_period("M")}
        hold_count = 0

    if active:
        pnl = (cl_arr[-1] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades):
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, stop_p75=0.0, worst_mo=0.0, pnls=np.array([]),
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
                exit_tp=sum(1 for t in trades if t["reason"]=="TP"),
                exit_stop=sum(1 for t in trades if t["reason"]=="STOP"),
                exit_time=sum(1 for t in trades if t["reason"] in ("TIME","CLOSE","END")))


# ── va_pct grid (tp=poc) ──────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"VALUE AREA GRID  (TP=yesterday's POC, stop=VA_edge+0.25×ATR)")
print(f"{'='*78}")
print(f"  {'VA%':>5}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed$':>9}  {'WorstMo':>8}")

grid_res = {}
for va in VA_PCTS:
    t = run_simulation(va, tp_mode="poc")
    s = summarise(t)
    grid_res[va] = (t, s)
    prim = " ◀ PRIMARY" if va == PRIMARY_VA else ""
    wr_f = "✅" if s["wr"] >= GATE0_WR_MIN else "❌"
    print(f"  {va:>4.0%}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
          f"{s['wr']:>7.1%}{wr_f}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}  "
          f"${s['stop_med']:>7.0f}  {s['worst_mo']:>8.1%}{prim}")

# ── TP sensitivity at primary VA ─────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"TP SENSITIVITY  (VA={PRIMARY_VA:.0%})")
print(f"{'='*78}")
print(f"  {'TP mode':<14}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>8}")
for tp_mode, tp_r in [("poc (natural)", None), ("1.5R", 1.5), ("2.0R", 2.0), ("3.0R", 3.0)]:
    t = run_simulation(PRIMARY_VA,
                       tp_mode="poc" if tp_mode.startswith("poc") else "r_mult",
                       tp_r_mult=tp_r)
    s = summarise(t)
    wr_f = "✅" if s["avg_pnl"] > 0 else "❌"
    prim = " ◀ PRIMARY" if tp_mode.startswith("poc") else ""
    print(f"  {tp_mode:<14}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
          f"{s['wr']:>7.1%}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}{wr_f}  "
          f"{s['worst_mo']:>8.1%}{prim}")

# ── primary spec deep dive ────────────────────────────────────────────────────
pt, ps = grid_res[PRIMARY_VA]
print(f"\n{'='*78}")
print(f"PRIMARY SPEC DEEP DIVE  (VA={PRIMARY_VA:.0%}, TP=POC)")
print(f"{'='*78}")
print(f"\n  N={ps['n']}  freq={ps['freq']:.2f}/d  WR={ps['wr']:.1%}  "
      f"PF={ps['pf']:.2f}  AvgP&L=${ps['avg_pnl']:.2f}  WorstMo={ps['worst_mo']:.1%}")
print(f"  Exit: TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
      f"TIME/CLOSE={ps['exit_time']}")
print(f"  Med stop: ${ps['stop_med']:.0f}/contract  "
      f"75th-pct stop: ${ps['stop_p75']:.0f}/contract")

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
    mo_bars = rth_all[rth_all.index.to_period("M") == m]
    mo_days = mo_bars["date"].nunique()
    flag = "❌" if mwr < GATE0_WORST_MO else "✅"
    print(f"  {str(m):<10}  {n:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
          f"{n/max(1, mo_days):>7.2f}/d  {flag}")

# ── gate 0 verdict ────────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"GATE 0 VERDICT — PRIMARY SPEC (VA={PRIMARY_VA:.0%}, TP=POC)")
print(f"{'='*78}")
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
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}",               not g_wr),
        (f"Freq {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}",          not g_freq),
        (f"Med stop ${ps['stop_med']:.0f} > ${GATE0_STOP_MAX:.0f}", not g_stop),
        (f"Worst-mo WR {ps['worst_mo']:.1%} < {GATE0_WORST_MO:.0%}", not g_womo),
    ] if f]
    print(f"  ❌ GATE 0 FAIL.")
    for l, _ in fails:
        print(f"     • {l}")
print(f"{'='*78}")

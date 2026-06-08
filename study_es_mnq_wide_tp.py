"""
ES/MNQ Stat Arb — Path 1: Wider TP scan
Same entry/stop logic as study_es_mnq_stat_arb.py, but target is TP_MULT × divergence
rather than 1× (i.e. we're shooting for price to overshoot on the convergence side).

At 2:1 (TP=2×, stop=1×) breakeven WR:
  WR × (2×div×$2 - $4.80) + (1-WR) × (-1×div×$2 - $4.80) > 0
  → at div=10pts: WR > 41.3%   (we measured 51% at 1:1, so margin is there)
  → at div=15pts: WR > 36.2%
  → at div=25pts: WR > 32.2%

PRIMARY SPEC (pre-frozen): THRESH=10pts, TP_MULT=2.0, STOP_MULT=1.0
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_PATH = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_PATH  = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

BETA_WIN      = 60
SPREAD_WIN    = 5
HOLD_MAX      = 60       # extended: 60 bars to reach a 2× target
SESSION_CLOSE = "15:55"
RTH_START     = "09:30"
MNQ_PV        = 2.0
COMMISSION    = 4.80

THRESHOLDS   = [5, 10, 15, 25]
TP_MULTS     = [1.0, 1.5, 2.0, 3.0]
STOP_MULT    = 1.0        # stop fixed at 1× entry divergence

PRIMARY_THRESH  = 10
PRIMARY_TP_MULT = 2.0
GATE0_WR_MIN    = 0.42    # breakeven WR at 2:1 R/R with $4.80 commission is ~41%
                          # set gate at 42% — must clear to have positive EV
GATE0_EV_MIN    = 0.0     # avg net P&L per trade must be positive
GATE0_FREQ_MIN  = 1.0
GATE0_STOP_USD  = 150.0
GATE0_WORST_MO  = 0.35

# ── load ──────────────────────────────────────────────────────────────────────
def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()

print("Loading bars…")
mnq = pd.concat([load_et(MNQ_PATH), load_et(MNQ_2026)])
mnq = mnq[~mnq.index.duplicated(keep="first")]
es  = load_et(ES_PATH)

both = (mnq[["close"]].rename(columns={"close": "mnq"})
        .join(es[["close"]].rename(columns={"close": "es"}), how="inner"))
both = both["2025-05-01":"2026-02-28"]
rth  = both.between_time(RTH_START, SESSION_CLOSE).copy()
print(f"  RTH bars: {len(rth):,}  |  {rth.index.normalize().nunique()} days")

rth["mnq_chg"] = rth["mnq"].diff()
rth["es_chg"]  = rth["es"].diff()
roll_cov = rth["mnq_chg"].rolling(BETA_WIN).cov(rth["es_chg"])
roll_var = rth["es_chg"].rolling(BETA_WIN).var()
rth["beta"] = (roll_cov / roll_var.replace(0, np.nan)).ffill().clip(0, 10)
rth["div"]  = (rth["mnq_chg"].rolling(SPREAD_WIN).sum()
               - rth["beta"] * rth["es_chg"].rolling(SPREAD_WIN).sum())
rth = rth.dropna(subset=["div", "beta"])

mnq_arr = rth["mnq"].values
div_arr = rth["div"].values
ts_arr  = rth.index
n_days  = rth.index.normalize().nunique()
pos_map = {ts: i for i, ts in enumerate(ts_arr)}

# ── simulation ────────────────────────────────────────────────────────────────
def run(thresh, tp_mult):
    trades = []
    active = None
    hold_count = 0

    for k in range(len(rth)):
        ts    = ts_arr[k]
        mnq_k = mnq_arr[k]
        div_k = div_arr[k]

        if active is not None:
            hit_tp   = (active["dir"] ==  1 and mnq_k >= active["tp"]) or \
                       (active["dir"] == -1 and mnq_k <= active["tp"])
            hit_stop = (active["dir"] ==  1 and mnq_k <= active["stop"]) or \
                       (active["dir"] == -1 and mnq_k >= active["stop"])
            at_close  = ts.strftime("%H:%M") >= SESSION_CLOSE
            hold_count += 1

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit": active["tp"], "pnl": pnl,
                                "win": True, "reason": "TP"})
                active = None; hold_count = 0
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit": active["stop"], "pnl": pnl,
                                "win": False, "reason": "STOP"})
                active = None; hold_count = 0
            elif at_close or hold_count >= HOLD_MAX:
                pnl = (mnq_k - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit": mnq_k, "pnl": pnl,
                                "win": pnl > 0, "reason": "CLOSE" if at_close else "TIME"})
                active = None; hold_count = 0
            continue

        if abs(div_k) < thresh:
            continue
        d        = -1 if div_k > 0 else 1
        entry    = mnq_k
        div_abs  = abs(div_k)
        stop_usd = div_abs * STOP_MULT * MNQ_PV
        if stop_usd > GATE0_STOP_USD:
            continue
        active = {"dir": d, "entry": entry, "div": div_abs,
                  "tp": entry + d * div_abs * tp_mult,
                  "stop": entry - d * div_abs * STOP_MULT,
                  "date": ts.date(), "month": ts.to_period("M")}
        hold_count = 0

    if active:
        pnl = (mnq_arr[-1] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
        trades.append({**active, "exit": mnq_arr[-1], "pnl": pnl,
                       "win": pnl > 0, "reason": "END"})
    return trades

def stats(trades):
    if not trades:
        return {}
    n    = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    mo   = {}
    for t in trades:
        m = t["month"]
        mo.setdefault(m, [0, 0])
        mo[m][0 if t["win"] else 1] += 1
    worst_mo = min((w/(w+l) if w+l else 0) for w, l in mo.values())
    gross_w  = sum(p for p in pnls if p > 0)
    gross_l  = abs(sum(p for p in pnls if p < 0))
    pf       = gross_w / max(1e-9, gross_l)
    return {"n": n, "wr": wins/n, "freq": n/n_days,
            "avg_pnl": pnls.mean(), "total_pnl": pnls.sum(),
            "pf": pf, "worst_mo": worst_mo, "pnls": pnls, "mo": mo,
            "tp_exits": sum(1 for t in trades if t["reason"]=="TP"),
            "stop_exits": sum(1 for t in trades if t["reason"]=="STOP"),
            "time_exits": sum(1 for t in trades if t["reason"] in ("TIME","CLOSE","END"))}

# ── grid ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*82}")
print(f"PATH 1 GRID  (stop fixed at 1.0×, TP varies; HOLD_MAX={HOLD_MAX} bars)")
print(f"{'='*82}")
print(f"  {'Thresh':>6}  {'TP×':>4}  {'BEven WR':>9}  {'N':>5}  {'Freq/d':>7}  "
      f"{'WR':>7}  {'PF':>5}  {'Avg P&L':>9}  {'Worst-mo':>9}")
print(f"  {'------':>6}  {'---':>4}  {'---------':>9}  {'---':>5}  {'------':>7}  "
      f"{'---':>7}  {'----':>5}  {'-------':>9}  {'--------':>9}")

all_res = {}
for th in THRESHOLDS:
    for tpm in TP_MULTS:
        # theoretical breakeven WR at this thresh and TP_MULT
        tp_usd   = th * tpm * MNQ_PV
        stop_usd = th * STOP_MULT * MNQ_PV
        be_wr = (stop_usd + COMMISSION) / (tp_usd + stop_usd)

        trades = run(th, tpm)
        s = stats(trades)
        all_res[(th, tpm)] = (trades, s)
        prim = "◀ PRIMARY" if th == PRIMARY_THRESH and tpm == PRIMARY_TP_MULT else ""
        wr_flag = "✅" if s["avg_pnl"] > 0 else "❌"
        print(f"  {th:>5}pt  {tpm:>3.1f}×  {be_wr:>9.1%}  {s['n']:>5}  "
              f"{s['freq']:>7.2f}/d  {s['wr']:>7.1%}  {s['pf']:>5.2f}  "
              f"${s['avg_pnl']:>7.2f}{wr_flag}  {s['worst_mo']:>8.1%}  {prim}")

# ── primary spec deep dive ────────────────────────────────────────────────────
pt, ps = all_res[(PRIMARY_THRESH, PRIMARY_TP_MULT)]

print(f"\n{'='*82}")
print(f"PRIMARY SPEC  THRESH={PRIMARY_THRESH}pts / TP={PRIMARY_TP_MULT}× / STOP=1×  "
      f"(risk ${PRIMARY_THRESH*MNQ_PV:.0f}, target ${PRIMARY_THRESH*PRIMARY_TP_MULT*MNQ_PV:.0f})")
print(f"{'='*82}")
print(f"\n  N={ps['n']}  freq={ps['freq']:.2f}/d  WR={ps['wr']:.1%}  "
      f"PF={ps['pf']:.2f}  AvgP&L=${ps['avg_pnl']:.2f}  "
      f"TotalP&L=${ps['total_pnl']:.0f}  WorstMoWR={ps['worst_mo']:.1%}")
print(f"  Exit: TP={ps['tp_exits']}  STOP={ps['stop_exits']}  "
      f"TIME/CLOSE={ps['time_exits']}")

print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'Avg P&L':>10}  {'freq/d':>8}")
mo_pnl = {}
for t in pt:
    m = t["month"]
    mo_pnl.setdefault(m, [])
    mo_pnl[m].append(t["pnl"])

for m in sorted(ps["mo"]):
    w, l = ps["mo"][m]
    n = w + l
    mwr = w / n if n else 0
    avg = np.mean(mo_pnl.get(m, [0]))
    mo_days = rth[rth.index.to_period("M") == m].index.normalize().nunique()
    flag = "❌" if mwr < GATE0_WORST_MO else "✅"
    print(f"  {str(m):<10}  {n:>5}  {mwr:>7.1%}  ${avg:>8.2f}  "
          f"{n/max(1,mo_days):>7.2f}/d  {flag}")

# ── gate 0 verdict ────────────────────────────────────────────────────────────
print(f"\n{'='*82}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  "
      f"(THRESH={PRIMARY_THRESH}pts, TP={PRIMARY_TP_MULT}×, STOP=1×)")
print(f"{'='*82}")

g_ev   = ps["avg_pnl"] > GATE0_EV_MIN
g_freq = ps["freq"]    >= GATE0_FREQ_MIN
g_womo = ps["worst_mo"] >= GATE0_WORST_MO
g_wr   = ps["wr"]      >= GATE0_WR_MIN

def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<42} [measured: {measured}]"

print(v(g_ev,   "Avg net P&L > $0/trade",
        f"${ps['avg_pnl']:.2f}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day",
        f"{ps['freq']:.2f}/day"))
print(v(g_wr,   f"WR ≥ breakeven {GATE0_WR_MIN:.0%} (2:1 R/R)",
        f"{ps['wr']:.1%}"))
print(v(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO:.0%}",
        f"{ps['worst_mo']:.1%}"))

gate_pass = all([g_ev, g_freq, g_wr, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — positive EV, adequate frequency, stable variance.")
    print("     Proceed to pre-registration and full combine backtest.")
else:
    fails = [(l, f) for l, f in [
        (f"Avg P&L ${ps['avg_pnl']:.2f} ≤ $0", not g_ev),
        (f"Freq {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}", not g_freq),
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}", not g_wr),
        (f"Worst-month WR {ps['worst_mo']:.1%} < {GATE0_WORST_MO:.0%}", not g_womo),
    ] if f]
    print(f"  ❌ GATE 0 FAIL.")
    for l, _ in fails:
        print(f"     • {l}")

# ── best cell summary ─────────────────────────────────────────────────────────
print(f"\n{'='*82}")
print(f"BEST POSITIVE-EV CELLS (avg_pnl > $0 and WR > breakeven WR)")
print(f"{'='*82}")
any_positive = False
for (th, tpm), (trades, s) in sorted(all_res.items()):
    if s["avg_pnl"] > 0 and s["freq"] >= GATE0_FREQ_MIN:
        tp_usd   = th * tpm * MNQ_PV
        stop_usd = th * MNQ_PV
        be_wr    = (stop_usd + COMMISSION) / (tp_usd + stop_usd)
        print(f"  {th}pt / TP={tpm}×  →  freq={s['freq']:.1f}/d  WR={s['wr']:.1%}  "
              f"PF={s['pf']:.2f}  avg=${s['avg_pnl']:.2f}  worst-mo={s['worst_mo']:.1%}  "
              f"(BE WR={be_wr:.1%})")
        any_positive = True
if not any_positive:
    print("  None — no cell achieves positive EV at ≥1.0 setup/day.")
print(f"{'='*82}")

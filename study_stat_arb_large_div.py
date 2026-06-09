"""
ES/MNQ Stat Arb — Large-Divergence Filter (Mary's Suggestion, BMAD party 2026-06-08)

Prior result (study_es_mnq_stat_arb.py): PRIMARY spec THRESH=10pts gave
WR=51.4%, avg P&L=-$3.88/trade, N=5,812 over 302 days. Commission ($4.80)
killed the thin per-trade edge at high frequency (19.6/day).

Mary's insight: The stat-arb dataset (N=5,812) is the largest confirmed-edge
sample we have. The failure was economics, not the absence of edge. Hypothesis:
at large divergences (high-conviction events), the mean-reversion is stronger
AND the per-trade profit is larger (bigger distance to recover = more P&L).
Testing whether filtering to THRESH ≥ 15 or 25 pts produces positive EV
while maintaining adequate frequency.

Additional diagnostic (vs original study):
  - Direction split: long vs short WR and P&L (asymmetry check)
  - Time-of-day filter: does large divergence perform better in specific windows?
  - Sequential trade dependency: does a prior loss change subsequent WR?

Setup (same as study_es_mnq_stat_arb.py — reusing its simulation logic):
  Entry:  5-bar cumulative MNQ divergence from beta-predicted ES > THRESH pts
  Trade:  fade the divergence (expect MNQ to give back the overshoot)
  Target: MNQ recovers the divergence from entry (1× TP)
  Stop:   STOP_MULT × divergence further against us from entry
  State:  one trade at a time; session close (15:55 ET) forces exit
  Beta:   rolling 60-bar OLS on 5-bar cumulative changes

PRIMARY spec (frozen before reading results):
  THRESH=20 pts, STOP_MULT=1.0
  Rationale: 20 pts ($40/contract) is 2× the prior primary spec threshold;
  at 2:1 R/R (TP=1×, stop=1×) with COMMISSION=$4.80 and stop=$40,
  breakeven WR = (40+4.80)/(2×40) = 56.0%. Need WR>56% to be profitable.
  Frequency concern is real: at THRESH=10 → 19.6/day; at THRESH=20 we
  estimate ~3-8/day. Gate 0 requires ≥1.0/day.

Grid (sensitivity only):
  THRESH ∈ {12, 15, 20, 25} pts × STOP_MULT ∈ {1.0, 2.0}
  (Primary: THRESH=20, STOP_MULT=1.0; 12pt added to bridge from prior study)

Gate 0 thresholds (unchanged):
  WR ≥ 50%, freq ≥ 1.0/day, median stop ≤ $150/contract,
  worst-month WR ≥ 35%

In-sample: 2025-05-01 → 2026-02-28 (matches ES data availability)
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_PATH = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_PATH  = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

BETA_WIN      = 60
SPREAD_WIN    = 5
HOLD_MAX      = 30
SESSION_CLOSE = "15:55"
RTH_START     = "09:30"
MNQ_PV        = 2.0
COMMISSION    = 4.80

THRESHOLDS    = [12, 15, 20, 25]
STOP_MULTS    = [1.0, 2.0]

PRIMARY_THRESH    = 20
PRIMARY_STOP_MULT = 1.0

GATE0_WR_MIN      = 0.50
GATE0_FREQ_MIN    = 1.0
GATE0_STOP_USD    = 150.0
GATE0_WORST_MO    = 0.35


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
print(f"  RTH bars: {len(rth):,}  |  {rth.index.normalize().nunique()} days  "
      f"({rth.index[0].date()} → {rth.index[-1].date()})")

# ── rolling beta and 5-bar cumulative divergence ──────────────────────────────
rth["mnq_chg"] = rth["mnq"].diff()
rth["es_chg"]  = rth["es"].diff()
roll_cov = rth["mnq_chg"].rolling(BETA_WIN).cov(rth["es_chg"])
roll_var = rth["es_chg"].rolling(BETA_WIN).var()
rth["beta"] = (roll_cov / roll_var.replace(0, np.nan)).ffill().clip(0, 10)
rth["div"]  = (rth["mnq_chg"].rolling(SPREAD_WIN).sum()
               - rth["beta"] * rth["es_chg"].rolling(SPREAD_WIN).sum())
rth = rth.dropna(subset=["div", "beta"])

n_days  = rth.index.normalize().nunique()
mnq_arr = rth["mnq"].values
div_arr = rth["div"].values
ts_arr  = rth.index


# ── simulation (same logic as study_es_mnq_stat_arb.py) ──────────────────────
def run_simulation(thresh: float, stop_mult: float):
    trades     = []
    active     = None
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
                                "win": pnl > 0,
                                "reason": "CLOSE" if at_close else "TIME"})
                active = None; hold_count = 0
            continue

        if abs(div_k) < thresh:
            continue

        direction  = -1 if div_k > 0 else 1
        entry      = mnq_k
        div_abs    = abs(div_k)
        stop_usd   = div_abs * stop_mult * MNQ_PV
        if stop_usd > GATE0_STOP_USD:
            continue

        active = {"dir": direction, "entry": entry, "div": div_abs,
                  "tp":   entry + direction * div_abs,
                  "stop": entry - direction * div_abs * stop_mult,
                  "date": ts.date(), "month": ts.to_period("M"),
                  "hour": ts.hour}
        hold_count = 0

    if active:
        pnl = (mnq_arr[-1] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
        trades.append({**active, "exit": mnq_arr[-1], "pnl": pnl,
                       "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades):
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    worst_mo=0.0, stop_med=0.0, stop_p75=0.0,
                    pnls=np.array([]), mo={},
                    exit_tp=0, exit_stop=0, exit_time=0)
    n    = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    stops = np.array([t["div"] * t.get("stop_mult", PRIMARY_STOP_MULT) * MNQ_PV
                      for t in trades])
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
                pf=pf, worst_mo=worst_mo,
                stop_med=float(np.median(stops)),
                stop_p75=float(np.percentile(stops, 75)),
                pnls=pnls, mo=mo,
                exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
                exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
                exit_time=sum(1 for t in trades if t["reason"]
                               in ("TIME", "CLOSE", "END")))


# ── main grid ─────────────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print(f"LARGE-DIVERGENCE FILTER GRID  "
      f"(1:1 R/R, stop cap ${GATE0_STOP_USD}, HOLD_MAX={HOLD_MAX})")
print(f"{'='*84}")
print(f"  {'Thresh':>7}  {'Stop×':>6}  {'BE WR':>7}  {'N':>6}  {'Freq/d':>8}  "
      f"{'WR':>7}  {'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>8}")

grid_res = {}
for th in THRESHOLDS:
    for sm in STOP_MULTS:
        t = run_simulation(th, sm)
        s = summarise(t)
        grid_res[(th, sm)] = (t, s)
        # breakeven WR at 1:1 with typical stop = th × sm × MNQ_PV
        stop_est = th * sm * MNQ_PV
        be_wr = (stop_est + COMMISSION) / (2 * stop_est) if stop_est > 0 else 0.5
        prim  = " ◀ PRIMARY" if th == PRIMARY_THRESH and sm == PRIMARY_STOP_MULT else ""
        wr_f  = "✅" if s["wr"] >= GATE0_WR_MIN else "❌"
        freq_f = "✅" if s["freq"] >= GATE0_FREQ_MIN else "❌"
        print(f"  {th:>6}pt  {sm:>5.1f}×  {be_wr:>7.1%}  {s['n']:>6}  "
              f"{s['freq']:>6.2f}/d{freq_f}  {s['wr']:>7.1%}{wr_f}  {s['pf']:>5.2f}  "
              f"${s['avg_pnl']:>6.2f}  {s['worst_mo']:>8.1%}{prim}")

# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[(PRIMARY_THRESH, PRIMARY_STOP_MULT)]

print(f"\n{'='*84}")
print(f"PRIMARY SPEC DEEP DIVE  "
      f"(THRESH={PRIMARY_THRESH}pt, STOP={PRIMARY_STOP_MULT}×, TP=1×div)")
print(f"{'='*84}")

stop_est = PRIMARY_THRESH * PRIMARY_STOP_MULT * MNQ_PV
be_wr    = (stop_est + COMMISSION) / (2 * stop_est)
print(f"  Theoretical stop: ${stop_est:.0f}/contract  |  "
      f"Breakeven WR: {be_wr:.1%}  |  Commission: ${COMMISSION}")

if ps["n"] > 0:
    print(f"\n  N={ps['n']}  freq={ps['freq']:.2f}/d  WR={ps['wr']:.1%}  "
          f"PF={ps['pf']:.2f}  AvgP&L=${ps['avg_pnl']:.2f}")
    print(f"  Exit: TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
          f"TIME/CLOSE={ps['exit_time']}")

    # Direction split
    longs  = [t for t in pt if t["dir"] ==  1]
    shorts = [t for t in pt if t["dir"] == -1]
    if longs:
        l_wr  = sum(t["win"] for t in longs) / len(longs)
        l_avg = float(np.mean([t["pnl"] for t in longs]))
        print(f"\n  Direction split:")
        print(f"    Long  (MNQ underperformed ES): N={len(longs):4}  "
              f"WR={l_wr:.1%}  AvgP&L=${l_avg:.2f}")
    if shorts:
        s_wr  = sum(t["win"] for t in shorts) / len(shorts)
        s_avg = float(np.mean([t["pnl"] for t in shorts]))
        print(f"    Short (MNQ overperformed ES):  N={len(shorts):4}  "
              f"WR={s_wr:.1%}  AvgP&L=${s_avg:.2f}")

    # Divergence size distribution
    div_arr_t = np.array([t["div"] for t in pt])
    print(f"\n  Divergence distribution at entry:")
    print(f"    Median: {np.median(div_arr_t):.1f} pts  "
          f"  p25: {np.percentile(div_arr_t,25):.1f}  "
          f"  p75: {np.percentile(div_arr_t,75):.1f}  "
          f"  p90: {np.percentile(div_arr_t,90):.1f}  "
          f"  max: {div_arr_t.max():.1f}")

    # Time-of-day breakdown
    print(f"\n  Time-of-day WR (ET):")
    hour_data: dict = {}
    for t in pt:
        h = t["hour"]
        hour_data.setdefault(h, [0, 0])
        hour_data[h][0 if t["win"] else 1] += 1
    print(f"  {'Hour':>6}  {'N':>5}  {'WR':>7}  {'%trades':>8}")
    for h in sorted(hour_data):
        w, l = hour_data[h]
        n_h  = w + l
        print(f"  {h:>5}ET  {n_h:>5}  {w/n_h:>7.1%}  {n_h/ps['n']:>7.1%}")

    # By-month table
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
        mo_days = mo_bars.index.normalize().nunique()
        if n_mo < 5:
            flag = "⚠️ N<5"
        elif mwr < GATE0_WORST_MO:
            flag = "❌"
        else:
            flag = "✅"
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"{n_mo/max(1, mo_days):>7.2f}/d  {flag}")

else:
    print(f"  No trades generated.")

# ── gate 0 verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  "
      f"(THRESH={PRIMARY_THRESH}pt, STOP={PRIMARY_STOP_MULT}×)")
print(f"{'='*84}")

g_wr   = ps["wr"]       >= GATE0_WR_MIN
g_freq = ps["freq"]     >= GATE0_FREQ_MIN
g_stop = ps["stop_med"] <= GATE0_STOP_USD
g_womo = ps["worst_mo"] >= GATE0_WORST_MO


def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<46} [measured: {measured}]"


print(v(g_wr,   f"Win rate ≥ {GATE0_WR_MIN:.0%}",               f"{ps['wr']:.1%}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day",            f"{ps['freq']:.2f}/day"))
print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_USD:.0f}/contract", f"${ps['stop_med']:.0f}"))
print(v(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO:.0%}",       f"{ps['worst_mo']:.1%}"))

gate_pass = all([g_wr, g_freq, g_stop, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — proceed to Stage 1 pre-registration.")
else:
    fails = [(label, fail) for label, fail in [
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}",               not g_wr),
        (f"Freq {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}",          not g_freq),
        (f"Med stop ${ps['stop_med']:.0f} > ${GATE0_STOP_USD:.0f}", not g_stop),
        (f"Worst-mo WR {ps['worst_mo']:.1%} < {GATE0_WORST_MO:.0%}", not g_womo),
    ] if fail]
    print(f"  ❌ GATE 0 FAIL.")
    for label, _ in fails:
        print(f"     • {label}")

print(f"{'='*84}")

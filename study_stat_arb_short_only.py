"""
ES/MNQ Stat Arb — Short-Only Direction Diagnostic (2026-06-09)

Finding from study_stat_arb_large_div.py: there is a strong directional
asymmetry in the ES/MNQ fade:

  Long  (MNQ underperformed ES → fade up): WR=47.9%, AvgP&L=-$8.01  NEGATIVE EV
  Short (MNQ overperformed  ES → fade dn): WR=57.4%, AvgP&L=+$5.84  POSITIVE EV

Structural justification for SHORT-ONLY direction:
  When MNQ outperforms ES, it's driven by Nasdaq-specific momentum (tech/AI spikes,
  single-name catalysts, growth rotation). These spikes partially revert as the
  divergence gets arbitraged and order flow normalises. When MNQ underperforms ES,
  the driver is often ES-specific macro strength (defensive rotation, energy, value)
  — MNQ genuinely doesn't participate, so there is no reversion to fade into.

This diagnostic pre-commits SHORT-ONLY before looking at stratified month/threshold
results (methodology guard: no cherry-picking direction after seeing the grid).

Entry:  5-bar cumulative MNQ divergence > +THRESH pts (MNQ overperformed ES → short)
Trade:  short MNQ, expect divergence to revert to zero
Target: MNQ closes the divergence back to entry (1× div, same as prior study)
Stop:   STOP_MULT × divergence beyond entry (same logic)
One trade at a time; 15:55 ET force-exit; HOLD_MAX=30 bars.

Primary spec (frozen before reading stratified results):
  THRESH=20 pts, STOP_MULT=1.0
  Theoretical stop: $40/contract
  Breakeven WR at 1:1 with $4.80 comm: (40+4.80)/(80) = 56.0%

Grid: THRESH ∈ {15, 20, 25, 30} × STOP_MULT ∈ {1.0, 2.0}

Gate 0 (short-only adjusted — WR gate raised to match breakeven):
  WR ≥ 56%  (commission-adjusted breakeven at primary spec; 50% is insufficient)
  Avg net P&L > $0/trade  (explicit EV check)
  Freq ≥ 1.0/day
  Median stop ≤ $150/contract
  Worst-month WR ≥ 35%

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

THRESHOLDS    = [15, 20, 25, 30]
STOP_MULTS    = [1.0, 2.0]

PRIMARY_THRESH    = 20
PRIMARY_STOP_MULT = 1.0

GATE0_WR_MIN   = 0.56   # commission-adjusted breakeven at $40 stop + $4.80 comm
GATE0_EV_MIN   = 0.0    # avg net P&L must be positive
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
mnq = pd.concat([load_et(MNQ_PATH), load_et(MNQ_2026)])
mnq = mnq[~mnq.index.duplicated(keep="first")]
es  = load_et(ES_PATH)

both = (mnq[["close"]].rename(columns={"close": "mnq"})
        .join(es[["close"]].rename(columns={"close": "es"}), how="inner"))
both = both["2025-05-01":"2026-02-28"]
rth  = both.between_time(RTH_START, SESSION_CLOSE).copy()
print(f"  RTH bars: {len(rth):,}  |  {rth.index.normalize().nunique()} days  "
      f"({rth.index[0].date()} → {rth.index[-1].date()})")

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

# ── total available short signals (for funnel context) ────────────────────────
print(f"\n  Short-signal funnel by threshold (MNQ outperformed ES by > THRESH):")
for th in THRESHOLDS:
    n_signals = (rth["div"] > th).sum()
    print(f"    THRESH={th}pt: {n_signals:,} signal-bars  ({n_signals/n_days:.1f}/day before 1-at-a-time filter)")


# ── simulation (SHORT ONLY) ───────────────────────────────────────────────────
def run_simulation(thresh: float, stop_mult: float):
    trades     = []
    active     = None
    hold_count = 0

    for k in range(len(rth)):
        ts    = ts_arr[k]
        mnq_k = mnq_arr[k]
        div_k = div_arr[k]

        if active is not None:
            # active trade is always short (dir=-1)
            hit_tp   = mnq_k <= active["tp"]
            hit_stop = mnq_k >= active["stop"]
            at_close  = ts.strftime("%H:%M") >= SESSION_CLOSE
            hold_count += 1

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * (-1) * MNQ_PV - COMMISSION
                trades.append({**active, "exit": active["tp"],
                                "pnl": pnl, "win": True, "reason": "TP"})
                active = None; hold_count = 0
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * (-1) * MNQ_PV - COMMISSION
                trades.append({**active, "exit": active["stop"],
                                "pnl": pnl, "win": False, "reason": "STOP"})
                active = None; hold_count = 0
            elif at_close or hold_count >= HOLD_MAX:
                pnl = (mnq_k - active["entry"]) * (-1) * MNQ_PV - COMMISSION
                trades.append({**active, "exit": mnq_k, "pnl": pnl,
                                "win": pnl > 0,
                                "reason": "CLOSE" if at_close else "TIME"})
                active = None; hold_count = 0
            continue

        # ── entry: only when div > +thresh (MNQ outperformed ES) ──────────
        if div_k <= thresh:
            continue

        div_abs  = div_k            # positive: MNQ overperformed by div_k pts
        stop_usd = div_abs * stop_mult * MNQ_PV
        if stop_usd > GATE0_STOP_MAX:
            continue

        entry    = mnq_k
        tp_price = entry - div_abs             # short target: give back the div
        sl_price = entry + div_abs * stop_mult  # short stop: divergence widens

        active = {"entry": entry, "div": div_abs,
                  "tp": tp_price, "stop": sl_price,
                  "stop_usd": stop_usd,
                  "date": ts.date(), "month": ts.to_period("M"),
                  "hour": ts.hour}
        hold_count = 0

    if active:
        pnl = (mnq_arr[-1] - active["entry"]) * (-1) * MNQ_PV - COMMISSION
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
                pf=pf, worst_mo=worst_mo,
                stop_med=float(np.median(stops)),
                stop_p75=float(np.percentile(stops, 75)),
                pnls=pnls, mo=mo,
                exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
                exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
                exit_time=sum(1 for t in trades if t["reason"]
                               in ("TIME", "CLOSE", "END")))


# ── grid ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*86}")
print(f"SHORT-ONLY GRID  (fade MNQ outperformance; 1:1 R/R; stop cap ${GATE0_STOP_MAX})")
print(f"{'='*86}")
print(f"  {'Thresh':>7}  {'Stop×':>6}  {'BE WR':>7}  {'N':>6}  {'Freq/d':>8}  "
      f"{'WR':>7}  {'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>8}")

grid_res = {}
for th in THRESHOLDS:
    for sm in STOP_MULTS:
        t = run_simulation(th, sm)
        s = summarise(t)
        grid_res[(th, sm)] = (t, s)
        stop_est = th * sm * MNQ_PV
        be_wr    = (stop_est + COMMISSION) / (2 * stop_est) if stop_est > 0 else 0.5
        prim     = " ◀ PRIMARY" if th == PRIMARY_THRESH and sm == PRIMARY_STOP_MULT else ""
        wr_f     = "✅" if s["wr"] >= GATE0_WR_MIN else "❌"
        ev_f     = "✅" if s["avg_pnl"] > GATE0_EV_MIN else "❌"
        freq_f   = "✅" if s["freq"] >= GATE0_FREQ_MIN else "❌"
        print(f"  {th:>6}pt  {sm:>5.1f}×  {be_wr:>7.1%}  {s['n']:>6}  "
              f"{s['freq']:>6.2f}/d{freq_f}  {s['wr']:>7.1%}{wr_f}  {s['pf']:>5.2f}  "
              f"${s['avg_pnl']:>6.2f}{ev_f}  {s['worst_mo']:>8.1%}{prim}")


# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[(PRIMARY_THRESH, PRIMARY_STOP_MULT)]

print(f"\n{'='*86}")
print(f"PRIMARY SPEC DEEP DIVE  (THRESH={PRIMARY_THRESH}pt, STOP={PRIMARY_STOP_MULT}×, SHORT ONLY)")
print(f"{'='*86}")

stop_est = PRIMARY_THRESH * PRIMARY_STOP_MULT * MNQ_PV
be_wr    = (stop_est + COMMISSION) / (2 * stop_est)
print(f"  Theoretical stop: ${stop_est:.0f}/contract  |  "
      f"Breakeven WR: {be_wr:.1%}  |  Commission: ${COMMISSION}")

if ps["n"] > 0:
    print(f"\n  N={ps['n']}  freq={ps['freq']:.2f}/d  WR={ps['wr']:.1%}  "
          f"PF={ps['pf']:.2f}  AvgP&L=${ps['avg_pnl']:.2f}")
    print(f"  Exit: TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
          f"TIME/CLOSE={ps['exit_time']}")

    # Divergence distribution at entry
    div_arr_t = np.array([t["div"] for t in pt])
    print(f"\n  Divergence at entry (pts — all positive = MNQ outperformed):")
    print(f"    Median: {np.median(div_arr_t):.1f}  "
          f"p25: {np.percentile(div_arr_t,25):.1f}  "
          f"p75: {np.percentile(div_arr_t,75):.1f}  "
          f"p90: {np.percentile(div_arr_t,90):.1f}  "
          f"max: {div_arr_t.max():.1f}")

    # Time-of-day WR
    print(f"\n  Time-of-day WR (ET):")
    hour_data: dict = {}
    for t in pt:
        h = t["hour"]
        hour_data.setdefault(h, [0, 0])
        hour_data[h][0 if t["win"] else 1] += 1
    print(f"  {'Hour':>6}  {'N':>5}  {'WR':>7}  {'AvgStop':>9}  {'%trades':>8}")
    hour_stops: dict = {}
    for t in pt:
        hour_stops.setdefault(t["hour"], [])
        hour_stops[t["hour"]].append(t["stop_usd"])
    for h in sorted(hour_data):
        w, l = hour_data[h]
        n_h = w + l
        avg_stop = float(np.mean(hour_stops.get(h, [0])))
        print(f"  {h:>5}ET  {n_h:>5}  {w/n_h:>7.1%}  ${avg_stop:>7.0f}  "
              f"{n_h/ps['n']:>7.1%}")

    # Victor's rolling-5-day variance check
    print(f"\n  Victor's rolling-5-day variance check:")
    all_trading_days = sorted(set(rth.index.normalize().date))
    day_pnl: dict = {}
    for t in pt:
        d = t["date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + t["pnl"]
    daily_series = [day_pnl.get(d, 0.0) for d in all_trading_days]
    if len(daily_series) >= 5:
        rolling5 = [sum(daily_series[i:i + 5]) for i in range(len(daily_series) - 4)]
        worst5   = min(rolling5)
        best5    = max(rolling5)
        pct_neg  = sum(1 for x in rolling5 if x < 0) / len(rolling5)
        worst_day = min(daily_series)
        print(f"    Worst rolling 5-day P&L:  ${worst5:.0f}  "
              f"({'✅ ≥$0' if worst5 >= 0 else '❌ negative'})")
        print(f"    Best  rolling 5-day P&L:  ${best5:.0f}")
        print(f"    % of 5-day windows < $0:  {pct_neg:.0%}")
        print(f"    Worst single-day P&L:     ${worst_day:.0f}")

    # By-month table
    print(f"\n  By month:")
    print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  "
          f"{'TotalP&L':>10}  {'freq/d':>8}")
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
        tot  = float(np.sum(mo_pnl.get(m, [0])))
        mo_bars = rth[rth.index.to_period("M") == m]
        mo_days = mo_bars.index.normalize().nunique()
        if n_mo < 5:
            flag = "⚠️ N<5"
        elif mwr < GATE0_WORST_MO:
            flag = "❌"
        else:
            flag = "✅"
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"${tot:>8.0f}  {n_mo/max(1, mo_days):>7.2f}/d  {flag}")

    # Equity curve sketch
    pnl_cumsum = np.cumsum(ps["pnls"])
    print(f"\n  Equity curve (1 MNQ contract):")
    print(f"    Start: $0  →  Peak: ${pnl_cumsum.max():.0f}  "
          f"→  Final: ${pnl_cumsum[-1]:.0f}")
    print(f"    Max drawdown from peak: "
          f"${(pnl_cumsum - np.maximum.accumulate(pnl_cumsum)).min():.0f}")

else:
    print(f"  No trades generated.")


# ── gate 0 verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*86}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  "
      f"(THRESH={PRIMARY_THRESH}pt, STOP={PRIMARY_STOP_MULT}×, SHORT ONLY)")
print(f"{'='*86}")

g_wr   = ps["wr"]       >= GATE0_WR_MIN
g_ev   = ps["avg_pnl"]  >  GATE0_EV_MIN
g_freq = ps["freq"]     >= GATE0_FREQ_MIN
g_stop = ps["stop_med"] <= GATE0_STOP_MAX
g_womo = ps["worst_mo"] >= GATE0_WORST_MO


def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<48} [measured: {measured}]"


print(v(g_wr,   f"Win rate ≥ {GATE0_WR_MIN:.0%}  (breakeven at 1:1, ${stop_est:.0f} stop)",
        f"{ps['wr']:.1%}"))
print(v(g_ev,   "Avg net P&L > $0/trade",
        f"${ps['avg_pnl']:.2f}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day",
        f"{ps['freq']:.2f}/day"))
print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract",
        f"${ps['stop_med']:.0f}"))
print(v(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO:.0%}",
        f"{ps['worst_mo']:.1%}"))

gate_pass = all([g_wr, g_ev, g_freq, g_stop, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — positive EV, adequate frequency, stable variance.")
    print("     Proceed to Stage 1 pre-registration before any holdout access.")
else:
    fails = [(label, fail) for label, fail in [
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}", not g_wr),
        (f"Avg P&L ${ps['avg_pnl']:.2f} ≤ $0",      not g_ev),
        (f"Freq {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}", not g_freq),
        (f"Med stop ${ps['stop_med']:.0f} > ${GATE0_STOP_MAX:.0f}", not g_stop),
        (f"Worst-mo WR {ps['worst_mo']:.1%} < {GATE0_WORST_MO:.0%}", not g_womo),
    ] if fail]
    print(f"  ❌ GATE 0 FAIL.")
    for label, _ in fails:
        print(f"     • {label}")

print(f"{'='*86}")

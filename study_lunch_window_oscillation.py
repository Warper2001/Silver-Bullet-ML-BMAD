"""
Lunch-Window Range Oscillation — Gate 0 Study
Carson's new idea (BMAD party 2026-06-08):

Thesis: MNQ at 11:30–13:00 ET is structurally low-momentum (lunch volume drain,
algo participants step back, range compresses). Fading new intra-lunch range
extremes during this window may achieve the WR that all-session fade strategies
cannot (1-min VWAP reversion: 14–37%; session-wide mean-reversion is Wall 1).
By selecting FOR the absence of momentum, this avoids the all-session continuation
bias that killed VWAP reversion, POC fade, and VWAP extension studies.

This is NOT re-testing VWAP reversion — it is selecting a specific intraday regime
(midday volume withdrawal) and testing whether fade WR is higher within that regime.

Setup class:
  Entry: Close of first 1-min bar that makes a new LOOKBACK_N-bar high within today's
         lunch window → short (fade back to range midpoint); new lunch-window low →
         long (fade back). The LOOKBACK_N window is restricted to bars from today's
         11:30 onward (no cross-day, no pre-lunch lookback).
  Stop:  Extreme bar high/low + STOP_BUFFER × ATR(20) beyond the triggered side.
  TP:    Midpoint of the N-bar range at entry:
           short TP = rolling_low + (rolling_high - rolling_low) / 2
           long  TP = rolling_high - (rolling_high - rolling_low) / 2
         i.e., halfway back across the lunch range from the extreme.
  Exit:  TP → win; Stop → loss; 13:00 ET hard close; HOLD_MAX=60 bars (1h, backstop).
  One trade at a time. Skip if stop > $150/contract. New entry only when prior
  trade is resolved.

Primary spec (frozen before reading results):
  LOOKBACK_N=20, STOP_BUFFER=0.25, LUNCH_START="11:30", LUNCH_END="12:50"
  Rationale: 20 bars = 20 min of lunch context; 0.25×ATR = same buffer as compression
  studies; 12:50 hard entry cutoff leaves 10 min for position to resolve before 13:00.

Grid (sensitivity only — NOT cherry-picked post-hoc):
  LOOKBACK_N ∈ {10, 20, 30}
  TP variants: midpoint (primary), 1.0R, 1.5R (at primary N only)

Gate 0 thresholds (unchanged):
  WR ≥ 50%, freq ≥ 1.0/day, median stop ≤ $150/contract, worst-month WR ≥ 35%

In-sample: 2025-01-01 → 2026-02-28
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_PATH  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

ATR_WIN       = 20
HOLD_MAX      = 60
LUNCH_START   = "11:30"
LUNCH_END_SIM = "12:50"     # last bar that can trigger an entry (13:00 hard close)
LUNCH_CLOSE   = "13:00"     # force-exit any open trade at or after this time
MNQ_PV        = 2.0
COMMISSION    = 4.80
STOP_CAP_USD  = 150.0
STOP_BUFFER   = 0.25        # × ATR beyond extreme

LOOKBACK_NS   = [10, 20, 30]
TP_MULTS_ALT  = [1.0, 1.5]   # R-multiple alternatives (vs midpoint primary)

PRIMARY_N     = 20

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

# ATR computed on all bars for warmup, then lunch bars reference it
bars["tr"]  = bars["high"] - bars["low"]
bars["atr"] = bars["tr"].rolling(ATR_WIN).mean()

# Lunch-window bars only — for simulation
lunch = bars.between_time(LUNCH_START, "12:59").copy()
lunch["date"] = lunch.index.date
n_days = lunch["date"].nunique()

print(f"  Total bars:       {len(bars):,}")
print(f"  Lunch bars:       {len(lunch):,}  (~{len(lunch)/n_days:.0f}/day)  |  {n_days} days")
print(f"  Lunch ATR median: {lunch['atr'].median():.2f} pts  "
      f"(≈${lunch['atr'].median() * MNQ_PV:.0f}/contract)")


# ── simulation ────────────────────────────────────────────────────────────────
def run_simulation(lookback_n: int, tp_mode: str = "midpoint", tp_r_mult: float = 1.0):
    """
    tp_mode: "midpoint" = target halfway across the N-bar lunch range from entry
             "r_mult"   = target tp_r_mult × stop_distance from entry
    """
    trades    = []
    active    = None
    hold_count = 0

    hi_arr   = lunch["high"].values
    lo_arr   = lunch["low"].values
    cl_arr   = lunch["close"].values
    atr_arr  = lunch["atr"].values
    ts_arr   = lunch.index
    date_arr = lunch["date"].values

    # Track lunch-window bar indices per day so we can restrict the lookback
    # to same-day bars after 11:30 only.
    # We build a list of per-day start indices as we iterate.
    day_start_k: dict = {}   # date → first bar index in lunch on that day
    for k, d in enumerate(date_arr):
        if d not in day_start_k:
            day_start_k[d] = k

    for k in range(len(lunch)):
        ts    = ts_arr[k]
        atr_k = atr_arr[k]
        d     = date_arr[k]

        if np.isnan(atr_k) or atr_k <= 0:
            continue

        stop_buf = STOP_BUFFER * atr_k

        # ── manage active trade ────────────────────────────────────────────
        if active is not None:
            hi_k = hi_arr[k]; lo_k = lo_arr[k]
            hit_tp   = (active["dir"] ==  1 and hi_k >= active["tp"]) or \
                       (active["dir"] == -1 and lo_k <= active["tp"])
            hit_stop = (active["dir"] ==  1 and lo_k <= active["stop"]) or \
                       (active["dir"] == -1 and hi_k >= active["stop"])
            at_close  = ts.strftime("%H:%M") >= LUNCH_CLOSE
            day_change = d != active["date"]
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

        # ── no entry after LUNCH_END_SIM ──────────────────────────────────
        if ts.strftime("%H:%M") > LUNCH_END_SIM:
            continue

        # ── restrict lookback to same-day lunch bars only ─────────────────
        day_first = day_start_k.get(d, k)
        bars_since_lunch_start = k - day_first + 1
        if bars_since_lunch_start < lookback_n:
            continue   # not enough same-day lunch bars yet

        # Use exactly LOOKBACK_N most recent same-day lunch bars
        w_start = k - lookback_n
        # Safety: ensure window doesn't include prior-day bars
        if w_start < day_first:
            continue

        window_hi = hi_arr[w_start:k]
        window_lo = lo_arr[w_start:k]
        rolling_high = window_hi.max()
        rolling_low  = window_lo.min()
        range_width  = rolling_high - rolling_low

        cl_k = cl_arr[k]
        hi_k = hi_arr[k]
        lo_k = lo_arr[k]

        # ── entry signal: close breaks outside the N-bar window ───────────
        if cl_k > rolling_high:
            direction = -1    # new lunch high → short (fade back down)
        elif cl_k < rolling_low:
            direction = 1     # new lunch low  → long  (fade back up)
        else:
            continue          # price inside the window — no signal

        entry = cl_k

        # Stop: beyond the extreme of the triggering bar
        if direction == -1:
            stop_p = hi_k + stop_buf     # short: stop above this bar's high
        else:
            stop_p = lo_k - stop_buf     # long:  stop below this bar's low

        stop_dist = abs(entry - stop_p)
        stop_usd  = stop_dist * MNQ_PV
        if stop_usd > STOP_CAP_USD:
            continue

        # TP: midpoint of the N-bar range, or R-multiple
        range_mid = (rolling_high + rolling_low) / 2
        if tp_mode == "midpoint":
            tp_p = range_mid
        else:
            tp_p = entry + direction * stop_dist * tp_r_mult

        # Sanity: TP must be profitable
        if direction == -1 and tp_p >= entry:
            continue
        if direction ==  1 and tp_p <= entry:
            continue

        active = {"dir": direction, "entry": entry,
                  "tp": tp_p, "stop": stop_p,
                  "stop_dist": stop_dist, "stop_usd": stop_usd,
                  "range_hi": rolling_high, "range_lo": rolling_low,
                  "range_mid": range_mid, "atr": atr_k,
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


# ── main grid (tp=midpoint, vary N) ───────────────────────────────────────────
print(f"\n{'='*86}")
print(f"LUNCH-WINDOW OSCILLATION GRID  "
      f"(TP=range-midpoint, stop=bar_extreme+{STOP_BUFFER}×ATR, cap=${STOP_CAP_USD})")
print(f"Window: {LUNCH_START}–{LUNCH_CLOSE} ET  |  Entry cutoff: {LUNCH_END_SIM}")
print(f"{'='*86}")
print(f"  {'N':>4}  {'N_tr':>6}  {'Freq/d':>8}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed$':>9}  {'WorstMo':>8}")
print(f"  {'--':>4}  {'----':>6}  {'------':>8}  {'---':>7}  "
      f"{'----':>5}  {'------':>8}  {'--------':>9}  {'-------':>8}")

grid_res = {}
for n in LOOKBACK_NS:
    t = run_simulation(n, tp_mode="midpoint")
    s = summarise(t)
    grid_res[n] = (t, s)
    prim   = " ◀ PRIMARY" if n == PRIMARY_N else ""
    wr_f   = "✅" if s["wr"]   >= GATE0_WR_MIN   else "❌"
    freq_f = "✅" if s["freq"] >= GATE0_FREQ_MIN  else "❌"
    print(f"  {n:>4}  {s['n']:>6}  {s['freq']:>6.2f}/d{freq_f}  "
          f"{s['wr']:>7.1%}{wr_f}  {s['pf']:>5.2f}  "
          f"${s['avg_pnl']:>6.2f}  ${s['stop_med']:>7.0f}  "
          f"{s['worst_mo']:>8.1%}{prim}")

# ── TP sensitivity at primary N ───────────────────────────────────────────────
print(f"\n{'='*86}")
print(f"TP SENSITIVITY  (N={PRIMARY_N} — midpoint vs R-multiple alternatives)")
print(f"{'='*86}")
print(f"  {'TP mode':<14}  {'N_tr':>6}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>8}")

for tp_mode, tp_r in [("midpoint", None)] + [("r_mult", r) for r in TP_MULTS_ALT]:
    label = "midpoint" if tp_mode == "midpoint" else f"{tp_r}R"
    t = run_simulation(PRIMARY_N,
                       tp_mode=tp_mode,
                       tp_r_mult=tp_r if tp_r else 1.0)
    s = summarise(t)
    wr_f = "✅" if s["avg_pnl"] > 0 else "❌"
    prim = " ◀ PRIMARY" if tp_mode == "midpoint" else ""
    print(f"  {label:<14}  {s['n']:>6}  {s['freq']:>7.2f}/d  "
          f"{s['wr']:>7.1%}  {s['pf']:>5.2f}  "
          f"${s['avg_pnl']:>6.2f}{wr_f}  {s['worst_mo']:>8.1%}{prim}")

# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[PRIMARY_N]

print(f"\n{'='*86}")
print(f"PRIMARY SPEC DEEP DIVE  (N={PRIMARY_N}, TP=midpoint)")
print(f"{'='*86}")

print(f"\n  Funnel:")
print(f"    Lunch bars (11:30–12:59):   {len(lunch):,}")
print(f"    Trades taken:               {ps['n']}  ({ps['freq']:.2f}/day)")
if ps["n"] > 0:
    print(f"    Exit breakdown:             "
          f"TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
          f"TIME/CLOSE/DAYEND={ps['exit_time']}")
    print(f"\n  Performance:")
    print(f"    Win rate:                   {ps['wr']:.1%}")
    print(f"    Profit factor:              {ps['pf']:.2f}")
    print(f"    Avg net P&L:                ${ps['avg_pnl']:.2f}/contract")
    total_pnl = float(ps["pnls"].sum())
    print(f"    Total P&L (1 MNQ):          ${total_pnl:.0f}  over {n_days} days")
    print(f"    Median stop:                ${ps['stop_med']:.0f}/contract")
    print(f"    75th-pct stop:              ${ps['stop_p75']:.0f}/contract")
    print(f"    Worst-month WR:             {ps['worst_mo']:.1%}")

    # Victor's rolling-5-day diagnostic
    print(f"\n  Victor's rolling-5-day variance check:")
    all_trading_days = sorted(set(lunch["date"]))
    day_pnl: dict = {}
    for t in pt:
        d = t["date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + t["pnl"]
    daily_series = [day_pnl.get(d, 0.0) for d in all_trading_days]

    if len(daily_series) >= 5:
        rolling5 = [sum(daily_series[i:i + 5]) for i in range(len(daily_series) - 4)]
        worst5   = min(rolling5)
        best5    = max(rolling5)
        print(f"    Worst rolling 5-day P&L:    ${worst5:.0f}  "
              f"({'✅ ≥$0' if worst5 >= 0 else '❌ negative'})")
        print(f"    Best  rolling 5-day P&L:    ${best5:.0f}")
        print(f"    Worst single-day P&L:       ${min(daily_series):.0f}")
    else:
        print(f"    Insufficient sample (N={ps['n']})")

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
        mo_bars = lunch[lunch.index.to_period("M") == m]
        mo_days = len(set(mo_bars["date"]))
        if n_mo < 5:
            flag = "⚠️ N<5"
        elif mwr < GATE0_WORST_MO:
            flag = "❌"
        else:
            flag = "✅"
        print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
              f"{n_mo/max(1, mo_days):>7.2f}/d  {flag}")
else:
    print(f"\n  No trades generated — thesis dead at this window/spec.")

# ── gate 0 verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*86}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  (N={PRIMARY_N}, TP=midpoint)")
print(f"{'='*86}")

g_wr   = ps["wr"]       >= GATE0_WR_MIN
g_freq = ps["freq"]     >= GATE0_FREQ_MIN
g_stop = ps["stop_med"] <= GATE0_STOP_MAX
g_womo = ps["worst_mo"] >= GATE0_WORST_MO


def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<46} [measured: {measured}]"


print(v(g_wr,   f"Win rate ≥ {GATE0_WR_MIN:.0%}",               f"{ps['wr']:.1%}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day",            f"{ps['freq']:.2f}/day"))
print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract", f"${ps['stop_med']:.0f}"))
print(v(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO:.0%}",       f"{ps['worst_mo']:.1%}"))

gate_pass = all([g_wr, g_freq, g_stop, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — proceed to Stage 1 pre-registration.")
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

print(f"{'='*86}")

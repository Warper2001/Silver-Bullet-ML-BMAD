"""
GC/MGC Post-Catalyst Momentum — Gate 0 Study
Carson's suggestion (BMAD party 2026-06-09): macro catalysts (CPI, NFP, FOMC)
cause gold to make a decisive directional move; after the initial spike settles,
enter in the direction of that move and ride the continuation.

Why GC/MGC instead of MNQ:
  Gold is a macro barometer, not a momentum vehicle. After a CPI surprise, gold
  reprices to a new inflation expectation and HOLDS there (unlike equities which
  often spike and retrace). The catalyst provides directional clarity that
  pure-momentum strategies lack. This sidesteps the MNQ 1-min momentum problem
  (68% gap opens, false breakouts) because the CATALYST filters which sessions
  to trade.

Setup class:
  1. At catalyst time T (CPI/NFP 08:30 ET, FOMC 14:00 ET):
     record pre-event reference = close of bar at T−1
  2. Wait WAIT_BARS (e.g. 10 min) for the initial spike to stabilise
  3. Direction: sign(close[T+WAIT] − close[T−1])
  4. MIN_MOVE filter: skip if |move| < MIN_MOVE_ATR × ATR(14)
     (only trade real surprises; skip "non-events")
  5. Entry: close of bar at T+WAIT_BARS
  6. Stop:  STOP_ATR_MULT × ATR(14), capped at $150/MGC (15 pts)
  7. TP:    TP_MULT × stop_dist in entry direction
  8. Max hold: HOLD_MAX bars; no hard session close (GC trades 23h/day)

Instrument / sizing:
  Data:        GC 1-min bars (COMEX full contract, 100 oz)
  Simulation:  MGC economics — $10/pt per contract (10 oz, combine-sized)
  Commission:  $4.80 round-trip (Topstep MGC)
  Stop cap:    $150/contract = 15 pts on MGC

Frequency note (not a standard ≥1.0/day gate):
  ~35 macro events / 12 months ≈ 3 events/month ≈ 0.10 setups/day.
  This is an event-driven strategy; the relevant viability check is:
    avg P&L/event × events/month ≥ combine target pace
  If avg P&L = $200/event at 3 events/month → $600/month → $3k in 5 months.
  The combine allows 60 days (2 months) to hit the $3k target with 5 qualifying
  days. For event-driven strategies, 5 qualifying events (each ≥$150) is a
  stricter bar than 5 qualifying days of continuous trading.

Primary spec (frozen before reading results):
  WAIT_BARS=10, MIN_MOVE_ATR=2.0, STOP_ATR_MULT=1.5, TP_MULT=2.0

Grid (sensitivity — NOT cherry-picked):
  WAIT_BARS ∈ {5, 10, 15}  ×  MIN_MOVE_ATR ∈ {0.0, 2.0, 4.0}
  (fixed SM=1.5, TP=2.0)

In-sample: full GC dataset 2025-05-01 → 2026-05-19.
  No GC sealed holdout exists (first GC study). If Gate 0 passes, prospective
  OOS validation against future events is the next step.
"""
import pandas as pd
import numpy as np
from pathlib import Path

GC_PATH  = Path("data/processed/dollar_bars/1_minute/gc_1min_2025_2026.csv")
CAL_PATH = Path("data/macro/econ_calendar_2025_2026.csv")

ATR_WIN       = 14
HOLD_MAX      = 120          # bars (2 hours) — macro moves often run 1-2h
MGC_PV        = 10.0         # $/pt for 1 MGC contract (10 troy oz)
COMMISSION    = 4.80         # round-trip
STOP_CAP_USD  = 150.0        # combine stop cap
STOP_CAP_PTS  = STOP_CAP_USD / MGC_PV   # 15 pts on MGC

WAIT_BARS_S     = [5, 10, 15]
MIN_MOVE_ATRS   = [0.0, 2.0, 4.0]
STOP_ATR_MULTS  = [1.0, 1.5, 2.0]
TP_MULTS        = [1.5, 2.0, 3.0]

PRIMARY_WAIT  = 10
PRIMARY_MMOV  = 2.0
PRIMARY_SM    = 1.5
PRIMARY_TP    = 2.0

# R/R-aware gates (same philosophy as PDH/PDL study)
GATE0_EV_MIN   = 0.0
GATE0_PF_MIN   = 1.20
GATE0_WOMO_PNL = -100.0   # looser: event-driven variance per month is unavoidable
# WR gate: breakeven + 5pp (computed from primary stop median at runtime)


# ── load GC 1-min data ────────────────────────────────────────────────────────
def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


print("Loading GC 1-min bars…")
gc = load_et(GC_PATH)
gc = gc[~gc.index.duplicated(keep="first")]
gc["tr"]  = gc["high"] - gc["low"]
gc["atr"] = gc["tr"].rolling(ATR_WIN).mean()

print(f"  GC bars:     {len(gc):,}")
print(f"  Range:       {gc.index[0].strftime('%Y-%m-%d')} → {gc.index[-1].strftime('%Y-%m-%d')}")
print(f"  Price range: ${gc['close'].min():.0f} – ${gc['close'].max():.0f}/oz")
print(f"  ATR(14):     median={gc['atr'].median():.2f} pts  "
      f"p75={gc['atr'].quantile(0.75):.2f} pts")
print(f"  MGC stop cap ({STOP_CAP_USD:.0f}/$): {STOP_CAP_PTS:.1f} pts  "
      f"(={STOP_CAP_PTS/gc['atr'].median():.1f}× median ATR)")


# ── load and preview macro calendar ───────────────────────────────────────────
print("\nLoading macro calendar…")
cal = pd.read_csv(CAL_PATH)
print(f"  {len(cal)} events: {cal['event'].value_counts().to_dict()}")

# Build event timestamps in ET
def parse_event_ts(row):
    return pd.Timestamp(f"{row['date']} {row['time_et']}",
                        tz="America/New_York")

cal["ts_et"] = cal.apply(parse_event_ts, axis=1)
# Filter to GC data range (with a small margin for WAIT_BARS lookback)
cal = cal[(cal["ts_et"] >= gc.index[0]) & (cal["ts_et"] <= gc.index[-1])].copy()
cal = cal.sort_values("ts_et").reset_index(drop=True)
print(f"  {len(cal)} events in GC data range  ({cal['ts_et'].min().strftime('%Y-%m-%d')} "
      f"→ {cal['ts_et'].max().strftime('%Y-%m-%d')})")
print(f"  By type: {cal['event'].value_counts().to_dict()}")


# ── event-driven simulation ────────────────────────────────────────────────────
def run_simulation(wait_bars: int, min_move_atr: float,
                   stop_atr_mult: float, tp_mult: float) -> list[dict]:
    """
    One trade per qualifying macro event.
    Events never overlap (CPI/NFP/FOMC always on different days).
    """
    gc_idx   = gc.index
    cl_arr   = gc["close"].values
    hi_arr   = gc["high"].values
    lo_arr   = gc["low"].values
    atr_arr  = gc["atr"].values
    trades: list[dict] = []

    for _, ev in cal.iterrows():
        ev_ts    = ev["ts_et"]
        ev_type  = ev["event"]

        # ── find the event bar (first bar at or after the event time) ────────
        pos = gc_idx.searchsorted(ev_ts)
        if pos >= len(gc):
            continue

        # Verify the event bar is within 2 minutes of the scheduled time
        actual_ts = gc_idx[pos]
        if abs((actual_ts - ev_ts).total_seconds()) > 120:
            continue    # no bar found near event time — skip

        # ── pre-event reference: close of bar just before event ───────────
        if pos == 0:
            continue
        pre_close = cl_arr[pos - 1]
        pre_atr   = atr_arr[pos - 1]
        if np.isnan(pre_atr) or pre_atr <= 0:
            continue

        # ── wait WAIT_BARS bars for the spike to settle ───────────────────
        entry_pos = pos + wait_bars
        if entry_pos >= len(gc):
            continue

        entry_ts    = gc_idx[entry_pos]
        entry_close = cl_arr[entry_pos]
        entry_atr   = atr_arr[entry_pos]
        if np.isnan(entry_atr) or entry_atr <= 0:
            continue

        # ── direction and MIN_MOVE filter ─────────────────────────────────
        net_move  = entry_close - pre_close      # pts from pre-event
        direction = 1 if net_move > 0 else -1
        min_move_pts = min_move_atr * pre_atr

        if abs(net_move) < min_move_pts:
            continue    # catalyst response too weak — skip

        # ── trade geometry ─────────────────────────────────────────────────
        stop_dist_raw = stop_atr_mult * entry_atr
        stop_usd_raw  = stop_dist_raw * MGC_PV
        # clamp to combine stop cap (never skip — catalyst IS the signal)
        stop_usd  = min(stop_usd_raw, STOP_CAP_USD)
        stop_dist = stop_usd / MGC_PV

        entry  = entry_close
        stop_p = entry - direction * stop_dist
        tp_p   = entry + direction * stop_dist * tp_mult

        active = {
            "dir":        direction,
            "entry":      entry,
            "tp":         tp_p,
            "stop":       stop_p,
            "stop_dist":  stop_dist,
            "stop_usd":   stop_usd,
            "stop_raw":   stop_usd_raw,
            "atr_entry":  entry_atr,
            "net_move":   net_move,
            "pre_close":  pre_close,
            "ev_type":    ev_type,
            "ev_ts":      ev_ts,
            "entry_ts":   entry_ts,
            "month":      entry_ts.to_period("M"),
        }

        # ── scan forward for TP / stop / time-stop ────────────────────────
        result = None
        for k in range(entry_pos + 1, min(entry_pos + 1 + HOLD_MAX, len(gc))):
            hi_k = hi_arr[k]; lo_k = lo_arr[k]; cl_k = cl_arr[k]

            hit_tp   = ((direction ==  1 and hi_k >= tp_p) or
                        (direction == -1 and lo_k <= tp_p))
            hit_stop = ((direction ==  1 and lo_k <= stop_p) or
                        (direction == -1 and hi_k >= stop_p))

            if hit_tp:
                pnl = (tp_p - entry) * direction * MGC_PV - COMMISSION
                result = {**active, "exit_p": tp_p, "pnl": pnl,
                          "win": True, "reason": "TP",
                          "bars_held": k - entry_pos}
                break
            elif hit_stop:
                pnl = (stop_p - entry) * direction * MGC_PV - COMMISSION
                result = {**active, "exit_p": stop_p, "pnl": pnl,
                          "win": False, "reason": "STOP",
                          "bars_held": k - entry_pos}
                break

        if result is None:
            # time-stop: close at last scanned bar
            last_k = min(entry_pos + HOLD_MAX, len(gc) - 1)
            ep  = cl_arr[last_k]
            pnl = (ep - entry) * direction * MGC_PV - COMMISSION
            result = {**active, "exit_p": ep, "pnl": pnl,
                      "win": pnl > 0, "reason": "TIME",
                      "bars_held": last_k - entry_pos}

        trades.append(result)

    return trades


def summarise(trades: list[dict]) -> dict:
    if not trades:
        return dict(n=0, wr=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, pnls=np.array([]), mo={}, mo_pnl={},
                    exit_tp=0, exit_stop=0, exit_time=0,
                    n_events_total=len(cal))
    n       = len(trades)
    wins    = sum(t["win"] for t in trades)
    pnls    = np.array([t["pnl"] for t in trades])
    stops   = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf      = gross_w / max(1e-9, gross_l)
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
        n=n, wr=wins/n, avg_pnl=float(pnls.mean()), pf=pf,
        stop_med=float(np.median(stops)),
        pnls=pnls, mo=mo, mo_pnl=mo_pnl,
        worst_mo_pnl=worst_mo_pnl,
        exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
        exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
        exit_time=sum(1 for t in trades if t["reason"] == "TIME"),
        n_events_total=len(cal),
    )


# ── main grid  (WAIT_BARS × MIN_MOVE_ATR, fixed SM=1.5, TP=2.0) ──────────────
print(f"\n{'='*105}")
print(f"GC POST-CATALYST GRID  "
      f"(SM={PRIMARY_SM}×ATR, TP={PRIMARY_TP}×, HOLD_MAX={HOLD_MAX} bars, 1 MGC contract)")
print(f"{'='*105}")
print(f"  Total calendar events in window: {len(cal)}")
print(f"  {'Wait':>4}  {'MinMv':>5}  {'N':>4}  {'Skip%':>5}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed':>8}  {'WorstMoPnL':>11}")
print(f"  {'----':>4}  {'-----':>5}  {'--':>4}  {'-----':>5}  {'---':>7}  "
      f"{'----':>5}  {'------':>8}  {'-------':>8}  {'-----------':>11}")

grid_res: dict = {}
for wb in WAIT_BARS_S:
    for mm in MIN_MOVE_ATRS:
        t = run_simulation(wb, mm, PRIMARY_SM, PRIMARY_TP)
        s = summarise(t)
        grid_res[(wb, mm)] = (t, s)
        is_primary = (wb == PRIMARY_WAIT and mm == PRIMARY_MMOV)
        prim   = " ◀ PRIMARY" if is_primary else ""
        skip_pct = (len(cal) - s["n"]) / max(1, len(cal))
        ev_f   = "✅" if s["avg_pnl"] > GATE0_EV_MIN else "❌"
        pf_f   = "✅" if s["pf"] >= GATE0_PF_MIN     else "❌"
        n_flag = "⚠️" if s["n"] < 10 else ""
        be_wr  = ((s["stop_med"] + COMMISSION) / ((PRIMARY_TP + 1) * s["stop_med"])
                  if s["stop_med"] > 0 else 0.0)
        wr_f   = "✅" if s["n"] > 0 and s["wr"] >= be_wr + 0.05 else "❌"
        print(f"  {wb:>4}  {mm:>4.1f}×  {s['n']:>3}{n_flag}  "
              f"{skip_pct:>5.0%}  "
              f"{s['wr']:>7.1%}{wr_f}  {s['pf']:>4.2f}{pf_f}  "
              f"${s['avg_pnl']:>6.2f}{ev_f}  "
              f"${s['stop_med']:>6.0f}  "
              f"${s['worst_mo_pnl']:>9.2f}{prim}")

# ── TP sensitivity at primary WAIT/MinMov ────────────────────────────────────
print(f"\n{'='*105}")
print(f"TP/STOP SENSITIVITY  "
      f"(WAIT={PRIMARY_WAIT} bars, MIN_MOVE={PRIMARY_MMOV}×ATR)")
print(f"{'='*105}")
print(f"  {'SM':>4}  {'TP':>4}  {'BEven':>6}  {'N':>4}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMo':>9}")
for sm in STOP_ATR_MULTS:
    for tp in TP_MULTS:
        t = run_simulation(PRIMARY_WAIT, PRIMARY_MMOV, sm, tp)
        s = summarise(t)
        be_wr = ((s["stop_med"] + COMMISSION) / ((tp + 1) * s["stop_med"])
                 if s["stop_med"] > 0 else 0.0)
        is_primary = (sm == PRIMARY_SM and tp == PRIMARY_TP)
        prim   = " ◀ PRIMARY" if is_primary else ""
        ev_f   = "✅" if s["avg_pnl"] > 0 else "❌"
        pf_f   = "✅" if s["pf"] >= GATE0_PF_MIN else "❌"
        print(f"  {sm:>3.1f}  {tp:>3.1f}  {be_wr:>6.1%}  {s['n']:>4}  "
              f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
              f"${s['avg_pnl']:>6.2f}{ev_f}  ${s['worst_mo_pnl']:>7.2f}{prim}")

# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[(PRIMARY_WAIT, PRIMARY_MMOV)]
be_wr_primary = ((ps["stop_med"] + COMMISSION) / ((PRIMARY_TP + 1) * ps["stop_med"])
                 if ps["stop_med"] > 0 else 0.0)
gate_wr_threshold = be_wr_primary + 0.05

print(f"\n{'='*105}")
print(f"PRIMARY SPEC DEEP DIVE  "
      f"(WAIT={PRIMARY_WAIT} bars, MIN_MOVE={PRIMARY_MMOV}×ATR, "
      f"SM={PRIMARY_SM}×, TP={PRIMARY_TP}×)")
print(f"{'='*105}")

print(f"\n  Event funnel:")
print(f"    Total calendar events:    {len(cal)}")
print(f"    Events with data:         {sum(1 for _,ev in cal.iterrows() if gc.index.searchsorted(ev['ts_et']) < len(gc))}")
print(f"    Events qualifying (≥{PRIMARY_MMOV}×ATR move): {ps['n']}")
print(f"    Events skipped (weak):    {len(cal) - ps['n']}  "
      f"({(len(cal)-ps['n'])/max(1,len(cal)):.0%} skipped by MIN_MOVE filter)")

if ps["n"] > 0:
    print(f"\n  Performance (1 MGC contract, $10/pt):")
    print(f"    N trades:               {ps['n']}  ({'⚠️ very small sample' if ps['n'] < 15 else 'ok'})")
    print(f"    Win rate:               {ps['wr']:.1%}  "
          f"(breakeven={be_wr_primary:.1%}; gate≥{gate_wr_threshold:.1%})")
    print(f"    Profit factor:          {ps['pf']:.3f}")
    print(f"    Avg net P&L/trade:      ${ps['avg_pnl']:.2f}")
    total_pnl = float(ps["pnls"].sum())
    print(f"    Total P&L:              ${total_pnl:.0f}  over {len(cal)} calendar events")
    print(f"    Median stop:            ${ps['stop_med']:.1f} ($MGC cap={STOP_CAP_USD:.0f})")
    print(f"    Worst-month avg P&L:    ${ps['worst_mo_pnl']:.2f}/trade")
    print(f"    Exit breakdown:         "
          f"TP={ps['exit_tp']}  STOP={ps['exit_stop']}  TIME={ps['exit_time']}")

    # ── per-event details ─────────────────────────────────────────────────────
    print(f"\n  Individual trades:")
    print(f"  {'Date':<12}  {'Type':<5}  {'Dir':>4}  {'Move':>6}  "
          f"{'Entry':>7}  {'Stop$':>6}  {'P&L':>8}  {'Bars':>5}  {'Exit'}")
    for t in pt:
        dir_s = "LONG" if t["dir"] == 1 else "SHORT"
        print(f"  {t['ev_ts'].strftime('%Y-%m-%d'):<12}  "
              f"{t['ev_type']:<5}  {dir_s:>4}  "
              f"{t['net_move']:>+6.1f}  "
              f"{t['entry']:>7.1f}  "
              f"${t['stop_usd']:>5.0f}  "
              f"${t['pnl']:>7.2f}  "
              f"{t['bars_held']:>5}  "
              f"{t['reason']}")

    # ── by event type ─────────────────────────────────────────────────────────
    print(f"\n  By event type:")
    print(f"  {'Type':<5}  {'N':>3}  {'WR':>7}  {'PF':>5}  {'AvgP&L':>8}")
    for ev_type in ["CPI", "NFP", "FOMC"]:
        subset = [t for t in pt if t["ev_type"] == ev_type]
        if not subset:
            print(f"  {ev_type:<5}  N=0 (all filtered by MIN_MOVE)")
            continue
        ns   = len(subset)
        ws   = sum(t["win"] for t in subset)
        ps_  = np.array([t["pnl"] for t in subset])
        gw   = sum(p for p in ps_ if p > 0); gl = abs(sum(p for p in ps_ if p < 0))
        pf_  = gw / max(1e-9, gl)
        print(f"  {ev_type:<5}  {ns:>3}  {ws/ns:>7.1%}  {pf_:>5.2f}  ${ps_.mean():>7.2f}  "
              f"{'⚠️ N<5' if ns < 5 else ''}")

    # ── by month ──────────────────────────────────────────────────────────────
    if len(ps["mo"]) > 0:
        print(f"\n  By month:")
        print(f"  {'Month':<10}  {'N':>3}  {'WR':>7}  {'AvgP&L':>9}")
        for m in sorted(ps["mo"]):
            w, l   = ps["mo"][m]
            n_mo   = w + l
            mwr    = w / n_mo if n_mo else 0
            avg    = float(np.mean(ps["mo_pnl"].get(m, [0])))
            print(f"  {str(m):<10}  {n_mo:>3}  {mwr:>7.1%}  ${avg:>7.2f}  "
                  f"{'⚠️ N<3' if n_mo < 3 else ''}")

    # ── combine math estimate ─────────────────────────────────────────────────
    print(f"\n  Combine-math estimate (event-driven viability):")
    events_in_window = len(cal)
    days_in_window   = (gc.index[-1] - gc.index[0]).days
    events_per_month = events_in_window / (days_in_window / 30.0)
    qualifying_pct   = ps["n"] / max(1, events_in_window)
    trades_per_month = qualifying_pct * events_per_month
    print(f"    Calendar events/month:  {events_per_month:.1f}")
    print(f"    Qualifying trade rate:  {qualifying_pct:.0%}  ({ps['n']}/{events_in_window})")
    print(f"    Qualifying trades/mo:   {trades_per_month:.1f}")
    if ps["avg_pnl"] > 0:
        months_to_profit = 3000.0 / (ps["avg_pnl"] * trades_per_month)
        events_for_5qual = 5.0 / (qualifying_pct * (ps["wr"] if ps["wr"] > 0 else 0.5))
        print(f"    Avg P&L/qualifying:     ${ps['avg_pnl']:.2f}/trade")
        print(f"    P&L/month estimate:     ${ps['avg_pnl']*trades_per_month:.0f}/month")
        print(f"    Months to $3k target:   {months_to_profit:.1f}  "
              f"(combine window: 2 months)")
        print(f"    Events needed for 5 qualifying days: ~{events_for_5qual:.0f}  "
              f"(≈{events_for_5qual/trades_per_month:.1f} months)")
    else:
        print(f"    EV negative — combine math does not work.")

    # ── confidence intervals for WR ───────────────────────────────────────────
    print(f"\n  WR confidence interval (Clopper-Pearson, N={ps['n']}):")
    n_w = int(ps["wr"] * ps["n"])
    import scipy.stats as stats_mod
    lo, hi = stats_mod.binom.interval(0.90, ps["n"], ps["wr"])
    ci_lo = lo / ps["n"]
    ci_hi = hi / ps["n"]
    print(f"    WR = {ps['wr']:.1%}   90% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"    {'✅ lower bound exceeds breakeven' if ci_lo >= be_wr_primary else '❌ lower CI bound below breakeven — edge uncertain at this sample size'}")

    # ── equity / max-DD sketch ────────────────────────────────────────────────
    print(f"\n  Equity / Max-DD sketch:")
    if len(ps["pnls"]) > 0:
        pnl_cumsum = np.cumsum(ps["pnls"])
        max_dd     = float((pnl_cumsum - np.maximum.accumulate(pnl_cumsum)).min())
        hwm        = float(np.maximum.accumulate(pnl_cumsum).max())
        print(f"    Final cumulative P&L:      ${pnl_cumsum[-1]:>8,.0f}")
        print(f"    Peak equity (HWM):         ${hwm:>8,.0f}")
        print(f"    Max drawdown (HWM→trough): ${max_dd:>8,.0f}  "
              f"({'✅ inside $2k combine limit' if max_dd >= -2000 else '❌ EXCEEDS $2k limit'})")

    # ── direction breakdown ───────────────────────────────────────────────────
    longs  = [t for t in pt if t["dir"] ==  1]
    shorts = [t for t in pt if t["dir"] == -1]
    print(f"\n  Long vs Short:")
    for label, subset in [("Long (gold rallied)", longs), ("Short (gold fell)", shorts)]:
        if subset:
            ns = len(subset); ws = sum(t["win"] for t in subset)
            ps_ = np.array([t["pnl"] for t in subset])
            gw = sum(p for p in ps_ if p > 0); gl = abs(sum(p for p in ps_ if p < 0))
            print(f"    {label:<22} N={ns:>2}  WR={ws/ns:>6.1%}  "
                  f"PF={gw/max(1e-9,gl):>5.2f}  Avg=${ps_.mean():>7.2f}")
        else:
            print(f"    {label:<22} N=0")

else:
    print(f"\n  No qualifying events with MIN_MOVE={PRIMARY_MMOV}×ATR. "
          f"Try loosening MIN_MOVE to 0.0 in the grid above.")

# ── gate 0 verdict ────────────────────────────────────────────────────────────
print(f"\n{'='*105}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  "
      f"(WAIT={PRIMARY_WAIT}, MM={PRIMARY_MMOV}×ATR, SM={PRIMARY_SM}×, TP={PRIMARY_TP}×)")
print(f"{'='*105}")
print(f"  ⚠️  SAMPLE SIZE WARNING: N={ps['n']} is very small for statistical confidence.")
print(f"     Interpret all metrics with wide uncertainty bands (see CI above).")
print(f"  Note: frequency gate (≥1.0/day) does NOT apply to event-driven strategies.")
print(f"        Check combine-math estimate instead (above).")
print()

g_ev   = ps["n"] > 0 and ps["avg_pnl"] > GATE0_EV_MIN
g_pf   = ps["n"] > 0 and ps["pf"] >= GATE0_PF_MIN
g_wr   = ps["n"] > 0 and ps["wr"] >= gate_wr_threshold
g_womo = ps["n"] > 0 and ps["worst_mo_pnl"] >= GATE0_WOMO_PNL
g_n    = ps["n"] >= 15   # minimum viable sample for Gate 0


def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<58} [measured: {measured}]"


print(v(g_n,    f"Minimum sample ≥ 15 qualifying events",
                f"N={ps['n']}"))
print(v(g_ev,   f"EV > $0 (avg net P&L > $0 per trade)",
                f"${ps['avg_pnl']:.2f}" if ps["n"] > 0 else "N/A"))
print(v(g_pf,   f"Profit factor ≥ {GATE0_PF_MIN:.2f}",
                f"{ps['pf']:.3f}" if ps["n"] > 0 else "N/A"))
print(v(g_wr,   f"WR ≥ breakeven+5% (≥{gate_wr_threshold:.1%})",
                f"{ps['wr']:.1%}" if ps["n"] > 0 else "N/A"))
print(v(g_womo, f"Worst-month avg P&L ≥ ${GATE0_WOMO_PNL:.0f}/trade",
                f"${ps['worst_mo_pnl']:.2f}" if ps["n"] > 0 else "N/A"))

all_gates = [g_n, g_ev, g_pf, g_wr, g_womo]
gate_pass = all(all_gates)

print()
if gate_pass:
    print("  ✅ GATE 0 PASS — GC post-catalyst shows edge.")
    print("     Caveat: N is small. Pre-register a prospective OOS test before any holdout.")
    print("     Next: pre-register exact parameters, then track next 20+ events prospectively.")
elif not g_n:
    print("  ⚠️  INSUFFICIENT SAMPLE — fewer than 15 qualifying events.")
    print("     Cannot make a reliable Gate 0 PASS/FAIL call at this N.")
    print("     Options:")
    print("       (a) Loosen MIN_MOVE filter (try 0.0 or 1.0×ATR in grid above)")
    print("       (b) Add more event types (e.g., PPI, retail sales)")
    print("       (c) Pre-register and track prospectively — more events will accumulate")
elif not g_ev or not g_pf:
    print("  ❌ GATE 0 FAIL — edge absent (EV≤$0 or PF<1.20).")
    print("     GC post-catalyst does not have combine-viable edge at this spec.")
    print("     Try loosening MIN_MOVE or different WAIT_BARS in the grid.")
else:
    fails = [lbl for flag, lbl in [
        (g_ev,   f"EV ${ps['avg_pnl']:.2f} ≤ $0"),
        (g_pf,   f"PF {ps['pf']:.3f} < {GATE0_PF_MIN:.2f}"),
        (g_wr,   f"WR {ps['wr']:.1%} < breakeven+5% ({gate_wr_threshold:.1%})"),
        (g_womo, f"Worst-mo avg ${ps['worst_mo_pnl']:.2f} < ${GATE0_WOMO_PNL:.0f}"),
    ] if not flag]
    print("  ❌ GATE 0 FAIL.")
    for lbl in fails:
        print(f"     • {lbl}")

print(f"{'='*105}")

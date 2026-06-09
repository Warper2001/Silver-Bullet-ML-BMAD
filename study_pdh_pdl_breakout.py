"""
PDH/PDL Breakout Momentum — Gate 0 Study
Party recommendation 2026-06-09: first momentum-WITH strategy in the entire
Topstep combine search.

All eight prior strategy families were fade/reversion or event-rare:
  VWAP: 84% momentum continuation → wrong direction.  POC fade: WR 14%.
  Lunch-window: WR 16%.  Stat arb: failed OOS regime shift.
The one never-tested quadrant is momentum-WITH — joining the move.

Setup class (PDH/PDL breakout):
  1. Reference levels: prior RTH day's high (PDH) and low (PDL)
  2. Entry (long):  first 1-min close strictly above prior PDH during RTH
  3. Entry (short): first 1-min close strictly below prior PDL during RTH
  4. Stop:  STOP_ATR_MULT × ATR(14), clamped to $150/contract (never skip
            on stop size — the structural level IS the signal)
  5. TP:    TP_MULT × stop_distance in breakout direction
  6. One trade per side per day (≤2 total trades/day); one position at a time
  7. HOLD_MAX=60 bars; 15:55 ET session hard-close

Structural justification (Carson / momentum-WITH):
  PDH/PDL are hard, widely-referenced structural levels (retail and
  institutional alike anchor orders and stops at these levels).  A close
  ABOVE PDH = buyers overwhelmed the prior day's high-water mark;
  continuation (not reversion) is the momentum-biased outcome.
  This is the inverse of every fade strategy that hit the MNQ momentum wall.

R/R note:  At 2:1 reward:risk, breakeven WR ≈ 35–38% (after commission).
  A flat ≥50% WR gate is wrong here.  Gate 0 uses R/R-adjusted gates:
    • EV > $0   • PF ≥ 1.20   • freq ≥ 1.0/day
    • WR ≥ breakeven+5%   • median stop ≤ $150   • worst-month avg P&L ≥ −$50

Primary spec (frozen before reading results):
  STOP_ATR_MULT=1.5, TP_MULT=2.0
  Rationale: 1.5×ATR gives the trade room to breathe past micro-structure noise
  while staying under the $150 combine stop cap for typical MNQ vol.  2:1 R/R
  is the standard momentum R multiple — WR just needs to exceed ~37%.

Grid (sensitivity only — NOT cherry-picked post-hoc):
  STOP_ATR_MULT ∈ {1.0, 1.5, 2.0}  ×  TP_MULT ∈ {1.5, 2.0, 3.0}

In-sample: 2025-01-01 → 2026-02-28  (same window as all prior Gate 0 studies)
Sealed holdout (≥2026-03-01) stays sealed until Gate 2, after pre-registration.
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_1MIN_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_1MIN_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

ATR_WIN       = 14           # 14 × 1-min RTH bars
HOLD_MAX      = 60           # bars after entry (60 min)
SESSION_CLOSE = "15:55"
RTH_START     = "09:30"
RTH_END       = "15:55"
MNQ_PV        = 2.0          # $/point for 1 MNQ contract
COMMISSION    = 4.80         # round-trip commission ($)
STOP_CAP_USD  = 150.0        # combine stop cap — clamp, never skip

STOP_ATR_MULTS = [1.0, 1.5, 2.0]
TP_MULTS       = [1.5, 2.0, 3.0]
PRIMARY_SM     = 1.5
PRIMARY_TP     = 2.0

# R/R-aware Gate 0 thresholds (replace flat ≥50% WR with breakeven-relative gate)
GATE0_EV_MIN    = 0.0          # avg net P&L > $0 (primary gate)
GATE0_PF_MIN    = 1.20         # profit factor
GATE0_FREQ_MIN  = 1.0          # setups/day
GATE0_STOP_MAX  = 150.0        # median stop ≤ $150
GATE0_WOMO_PNL  = -50.0        # worst-month avg P&L ≥ −$50 (variance guard)
# WR gate: breakeven WR + 5pp (computed dynamically from primary stop median)

ROLLING_5D_TARGET = 5 * 150.0  # Victor's threshold: $750 = 5 qualifying days × $150


# ── load 1-min bars (verbatim from study_vol_compression_15min.py) ─────────────
def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


print("Loading 1-min bars…")
bars_all = pd.concat([load_et(MNQ_1MIN_2025), load_et(MNQ_1MIN_2026)])
bars_all  = bars_all[~bars_all.index.duplicated(keep="first")]
bars_all  = bars_all["2025-01-01":"2026-02-28"]

# RTH-only filter — Carson's ATR fix: no overnight bars inflating the ATR
rth = bars_all.between_time(RTH_START, RTH_END).copy()
rth["tr"]   = rth["high"] - rth["low"]          # true range proxy (no overnight gap)
rth["atr"]  = rth["tr"].rolling(ATR_WIN).mean()  # RTH-only ATR
rth["date"] = rth.index.date

n_days = len(set(rth["date"]))
print(f"  1-min RTH bars: {len(rth):,}  (~{len(rth)/n_days:.0f}/day)  |  {n_days} trading days")
print(f"  ATR(14) median: {rth['atr'].median():.2f} pts  "
      f"(≈${rth['atr'].median() * MNQ_PV:.0f}/contract/bar)")
print(f"  Date range:     {rth.index[0].date()} → {rth.index[-1].date()}")


# ── compute prior-day PDH/PDL reference levels ────────────────────────────────
print("\nBuilding prior-day PDH/PDL map…")
dates_list = sorted(set(rth["date"]))
dates_set  = set(dates_list)
dates_to_idx = {d: i for i, d in enumerate(dates_list)}

# daily_hl[date] = (pdh, pdl) for the RTH session of that date
daily_hl: dict = {}
for d in dates_list:
    day_bars = rth[rth["date"] == d]
    if len(day_bars) == 0:
        continue
    daily_hl[d] = (float(day_bars["high"].max()), float(day_bars["low"].min()))

print(f"  PDH/PDL map built for {len(daily_hl)} days")
sample_d = dates_list[1]  # show 2nd day (first day has no prior)
if sample_d in daily_hl:
    prev_d = dates_list[0]
    if prev_d in daily_hl:
        print(f"  Example: on {sample_d}, prior PDH={daily_hl[prev_d][0]:.2f}  "
              f"PDL={daily_hl[prev_d][1]:.2f}")


# ── simulation ─────────────────────────────────────────────────────────────────
def run_simulation(stop_atr_mult: float, tp_mult: float):
    """
    Simulate PDH/PDL breakout strategy.

    Per-day arm management:
      - long_armed=True / short_armed=True reset at each day change
      - Once a side fires, it's disarmed for the rest of that day (≤1/side/day)
      - One active position at a time; entry check skipped while in a trade

    Stop: clamped to $150 (STOP_CAP_USD / MNQ_PV = 75 pts) — never skipped
    """
    trades     = []
    active     = None
    hold_count = 0
    long_armed = False
    short_armed = False
    pdh = pdl = None
    prev_date  = None

    hi_arr   = rth["high"].values
    lo_arr   = rth["low"].values
    cl_arr   = rth["close"].values
    atr_arr  = rth["atr"].values
    ts_arr   = rth.index
    date_arr = rth["date"].values

    for k in range(ATR_WIN, len(rth)):
        ts    = ts_arr[k]
        d     = date_arr[k]
        atr_k = atr_arr[k]

        if np.isnan(atr_k) or atr_k <= 0:
            continue

        # ── day boundary: load prior PDH/PDL, re-arm both sides ─────────────
        if d != prev_date:
            prev_date = d
            # force-close any residual (shouldn't happen given 15:55 close, but safety)
            if active is not None:
                ep  = cl_arr[k - 1]    # close of last bar of prior day
                pnl = (ep - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0, "reason": "DAYEND"})
                active = None; hold_count = 0

            d_idx = dates_to_idx.get(d, -1)
            if d_idx > 0:
                prev_d  = dates_list[d_idx - 1]
                pdh, pdl = daily_hl.get(prev_d, (None, None))
            else:
                pdh = pdl = None

            long_armed  = True
            short_armed = True

        if pdh is None or pdl is None:
            continue

        cl_k = cl_arr[k]
        hi_k = hi_arr[k]
        lo_k = lo_arr[k]

        # ── manage active trade ──────────────────────────────────────────────
        if active is not None:
            hold_count += 1
            at_close   = ts.strftime("%H:%M") >= SESSION_CLOSE
            day_end    = d != active["date"]

            hit_tp   = ((active["dir"] ==  1 and hi_k >= active["tp"]) or
                        (active["dir"] == -1 and lo_k <= active["tp"]))
            hit_stop = ((active["dir"] ==  1 and lo_k <= active["stop"]) or
                        (active["dir"] == -1 and hi_k >= active["stop"]))

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
            elif at_close or hold_count >= HOLD_MAX or day_end:
                ep  = cl_k
                pnl = (ep - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                reason = "CLOSE" if at_close else ("DAYEND" if day_end else "TIME")
                trades.append({**active, "exit_p": ep,
                                "pnl": pnl, "win": pnl > 0, "reason": reason})
                active = None; hold_count = 0
            continue   # skip entry check this bar even if we just exited

        # ── check for new entry (only when no active trade) ──────────────────
        # stop: ATR-based, clamped to combine cap (never skip — the level IS the signal)
        stop_dist_raw = stop_atr_mult * atr_k
        stop_usd_raw  = stop_dist_raw * MNQ_PV
        stop_usd      = min(stop_usd_raw, STOP_CAP_USD)
        stop_dist     = stop_usd / MNQ_PV

        direction = None
        if long_armed and cl_k > pdh:
            direction   = 1
            long_armed  = False
        elif short_armed and cl_k < pdl:
            direction   = -1
            short_armed = False

        if direction is None:
            continue

        entry  = cl_k
        stop_p = entry - direction * stop_dist     # below entry for long, above for short
        tp_p   = entry + direction * stop_dist * tp_mult

        active = {
            "dir":       direction,
            "entry":     entry,
            "tp":        tp_p,
            "stop":      stop_p,
            "stop_dist": stop_dist,
            "stop_usd":  stop_usd,
            "stop_raw":  stop_usd_raw,    # before clamp (to track how often capped)
            "atr":       atr_k,
            "pdh":       pdh,
            "pdl":       pdl,
            "date":      d,
            "month":     ts.to_period("M"),
            "bar_of_day": ts.strftime("%H:%M"),
        }
        hold_count = 0

    # residual at end of data
    if active:
        pnl = (cl_arr[-1] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
        trades.append({**active, "exit_p": cl_arr[-1],
                       "pnl": pnl, "win": pnl > 0, "reason": "END"})
    return trades


# ── summarise ─────────────────────────────────────────────────────────────────
def summarise(trades):
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0,
                    stop_med=0.0, stop_p75=0.0, worst_mo_wr=0.0, worst_mo_pnl=0.0,
                    pnls=np.array([]), mo={},
                    exit_tp=0, exit_stop=0, exit_time=0,
                    n_long=0, n_short=0, n_capped=0)
    n      = len(trades)
    wins   = sum(t["win"] for t in trades)
    pnls   = np.array([t["pnl"] for t in trades])
    stops  = np.array([t["stop_usd"] for t in trades])
    stops_raw = np.array([t["stop_raw"] for t in trades])
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
    n_long  = sum(1 for t in trades if t["dir"] ==  1)
    n_short = sum(1 for t in trades if t["dir"] == -1)
    n_capped = sum(1 for t in trades if t["stop_raw"] > STOP_CAP_USD)
    return dict(
        n=n, wr=wins/n, freq=n/n_days, avg_pnl=float(pnls.mean()), pf=pf,
        stop_med=float(np.median(stops)),
        stop_p75=float(np.percentile(stops, 75)),
        worst_mo_wr=worst_mo_wr, worst_mo_pnl=worst_mo_pnl,
        pnls=pnls, mo=mo, mo_pnl=mo_pnl,
        exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
        exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
        exit_time=sum(1 for t in trades if t["reason"] in
                      ("TIME", "CLOSE", "DAYEND", "END")),
        n_long=n_long, n_short=n_short, n_capped=n_capped,
    )


# ── main grid  (STOP_ATR_MULT × TP_MULT) ──────────────────────────────────────
print(f"\n{'='*100}")
print(f"PDH/PDL BREAKOUT GRID  "
      f"(stop=STOP_MULT×ATR(14) clamped ${STOP_CAP_USD:.0f}, position=1 MNQ)")
print(f"{'='*100}")
print(f"  {'SM':>4}  {'TP':>4}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed$':>9}  {'WorstMoP&L':>11}  {'BEven':>6}")
print(f"  {'--':>4}  {'--':>4}  {'---':>5}  {'-------':>8}  {'---':>7}  "
      f"{'----':>5}  {'------':>8}  {'--------':>9}  {'-----------':>11}  {'-----':>6}")

grid_res: dict = {}
for sm in STOP_ATR_MULTS:
    for tp in TP_MULTS:
        t = run_simulation(sm, tp)
        s = summarise(t)
        grid_res[(sm, tp)] = (t, s)
        is_primary = (sm == PRIMARY_SM and tp == PRIMARY_TP)
        prim  = " ◀ PRIMARY" if is_primary else ""
        be_wr = ((s["stop_med"] + COMMISSION) / ((tp + 1) * s["stop_med"])
                 if s["stop_med"] > 0 else 0.0)
        ev_f  = "✅" if s["avg_pnl"] > GATE0_EV_MIN   else "❌"
        pf_f  = "✅" if s["pf"]      >= GATE0_PF_MIN   else "❌"
        fr_f  = "✅" if s["freq"]    >= GATE0_FREQ_MIN  else "❌"
        print(f"  {sm:>3.1f}  {tp:>3.1f}  {s['n']:>5}  "
              f"{s['freq']:>6.2f}/d{fr_f}  "
              f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
              f"${s['avg_pnl']:>6.2f}{ev_f}  "
              f"${s['stop_med']:>7.0f}  "
              f"${s['worst_mo_pnl']:>9.2f}  "
              f"{be_wr:>6.1%}{prim}")

# ── TP sensitivity at primary stop mult ───────────────────────────────────────
print(f"\n{'='*100}")
print(f"TP SENSITIVITY  (STOP_ATR_MULT={PRIMARY_SM}×ATR, primary stop fixed)")
print(f"{'='*100}")
print(f"  {'TP×':>4}  {'BEven':>6}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'WorstMoAvg':>11}")
for tp in TP_MULTS:
    _, s = grid_res[(PRIMARY_SM, tp)]
    be_wr = ((s["stop_med"] + COMMISSION) / ((tp + 1) * s["stop_med"])
             if s["stop_med"] > 0 else 0.0)
    ev_f = "✅" if s["avg_pnl"] > 0 else "❌"
    prim = " ◀ PRIMARY" if tp == PRIMARY_TP else ""
    print(f"  {tp:>3.1f}×  {be_wr:>6.1%}  {s['n']:>5}  {s['freq']:>7.2f}/d  "
          f"{s['wr']:>7.1%}  {s['pf']:>5.2f}  ${s['avg_pnl']:>6.2f}{ev_f}  "
          f"${s['worst_mo_pnl']:>9.2f}{prim}")

# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[(PRIMARY_SM, PRIMARY_TP)]

print(f"\n{'='*100}")
print(f"PRIMARY SPEC DEEP DIVE  "
      f"(STOP_ATR_MULT={PRIMARY_SM}×, TP={PRIMARY_TP}×, HOLD_MAX={HOLD_MAX} bars)")
print(f"{'='*100}")

print(f"\n  Funnel:")
print(f"    1-min RTH bars:     {len(rth):,}")
print(f"    Trading days:       {n_days}")
print(f"    Trades taken:       {ps['n']}  ({ps['freq']:.2f}/day)")
if ps['n'] > 0:
    print(f"    Long / Short:       {ps['n_long']} / {ps['n_short']}  "
          f"({ps['n_long']/ps['n']:.0%} long, {ps['n_short']/ps['n']:.0%} short)")
    print(f"    Stop capped to $150:{ps['n_capped']}  "
          f"({ps['n_capped']/ps['n']:.0%} of trades; "
          f"native ATR stop would have exceeded cap)")
    print(f"    Exit breakdown:     "
          f"TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
          f"TIME/CLOSE={ps['exit_time']}")

print(f"\n  Performance:")
be_wr_primary = ((ps["stop_med"] + COMMISSION) / ((PRIMARY_TP + 1) * ps["stop_med"])
                 if ps["stop_med"] > 0 else 0.0)
gate_wr_threshold = be_wr_primary + 0.05  # breakeven + 5pp
print(f"    Win rate:           {ps['wr']:.1%}  (breakeven={be_wr_primary:.1%}; gate≥{gate_wr_threshold:.1%})")
print(f"    Profit factor:      {ps['pf']:.3f}")
print(f"    Avg net P&L:        ${ps['avg_pnl']:.2f}/trade")
total_pnl = float(ps["pnls"].sum()) if len(ps["pnls"]) > 0 else 0.0
print(f"    Total P&L (1 MNQ):  ${total_pnl:,.0f}  over {n_days} days")
print(f"    Median stop:        ${ps['stop_med']:.0f}/contract")
print(f"    75th-pct stop:      ${ps['stop_p75']:.0f}/contract")
print(f"    Worst-month WR:     {ps['worst_mo_wr']:.1%}")
print(f"    Worst-month avg:    ${ps['worst_mo_pnl']:.2f}/trade")

# ── long vs short breakdown ───────────────────────────────────────────────────
if pt:
    long_trades  = [t for t in pt if t["dir"] ==  1]
    short_trades = [t for t in pt if t["dir"] == -1]
    print(f"\n  Long vs Short breakdown:")
    for label, subset in [("Long (close > PDH)", long_trades),
                           ("Short (close < PDL)", short_trades)]:
        if subset:
            ns = len(subset)
            ws = sum(t["win"] for t in subset)
            ps_sub = np.array([t["pnl"] for t in subset])
            gw = sum(p for p in ps_sub if p > 0); gl = abs(sum(p for p in ps_sub if p < 0))
            pf_s = gw / max(1e-9, gl)
            print(f"    {label:<25}  N={ns:>4}  WR={ws/ns:>6.1%}  "
                  f"PF={pf_s:>5.2f}  Avg=${ps_sub.mean():>7.2f}")
        else:
            print(f"    {label:<25}  N=0")

# ── by-month table ─────────────────────────────────────────────────────────────
print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'freq/d':>8}  {'Status'}")
for m in sorted(ps["mo"]):
    w, l   = ps["mo"][m]
    n_mo   = w + l
    mwr    = w / n_mo if n_mo else 0
    avg    = float(np.mean(ps["mo_pnl"].get(m, [0])))
    mo_bars = rth[rth.index.to_period("M") == m]
    mo_days = len(set(mo_bars["date"]))
    if n_mo < 5:
        flag = "⚠️ N<5"
    elif avg < GATE0_WOMO_PNL:
        flag = "❌ avg P&L<-$50"
    else:
        flag = "✅"
    print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
          f"{n_mo/max(1, mo_days):>7.2f}/d  {flag}")

# ── time-of-day distribution ──────────────────────────────────────────────────
if pt:
    print(f"\n  Time-of-day distribution (when breakout fires):")
    from collections import Counter
    tod = Counter(t["bar_of_day"] for t in pt)
    # bucket into 30-min windows
    buckets: dict = {}
    for time_str, cnt in tod.items():
        h, mi = int(time_str[:2]), int(time_str[3:])
        bucket = f"{h:02d}:{(mi//30)*30:02d}"
        buckets[bucket] = buckets.get(bucket, 0) + cnt
    total_t = sum(buckets.values())
    for b in sorted(buckets):
        cnt = buckets[b]
        bar = "█" * int(cnt / max(buckets.values()) * 20)
        print(f"    {b}  {bar:<20}  {cnt:>4} ({cnt/total_t:.0%})")

# ── Victor's rolling-5-day variance diagnostic ────────────────────────────────
print(f"\n  Victor's rolling-5-day variance check (combine DD guard):")
if pt:
    day_pnl: dict = {}
    for t in pt:
        d = t["date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + t["pnl"]

    all_trading_days = sorted(set(rth["date"]))
    daily_series     = [day_pnl.get(d, 0.0) for d in all_trading_days]

    if len(daily_series) >= 5:
        rolling5   = [sum(daily_series[i:i + 5]) for i in range(len(daily_series) - 4)]
        worst5     = min(rolling5)
        best5      = max(rolling5)
        median5    = float(np.median(rolling5))
        pct_neg    = sum(1 for x in rolling5 if x < 0) / len(rolling5)
        worst_day  = min(daily_series)
        pct_days_profitable = sum(1 for x in daily_series if x > 0) / len(daily_series)

        print(f"    Worst  rolling 5-day P&L:  ${worst5:,.0f}  "
              f"({'✅ positive' if worst5 >= 0 else '❌ negative'})")
        print(f"    Best   rolling 5-day P&L:  ${best5:,.0f}")
        print(f"    Median rolling 5-day P&L:  ${median5:,.0f}")
        print(f"    % of 5-day windows < $0:   {pct_neg:.0%}")
        print(f"    Worst single-day P&L:       ${worst_day:,.0f}  "
              f"(combine $2k DD budget)")
        print(f"    Days with P&L > $0:         {pct_days_profitable:.0%}  "
              f"(of {len(daily_series)} days; rest are flat-trade days)")
        print(f"    (target: worst 5-day ≥ $0 = no losing 5-day stretches)")
    else:
        print(f"    Insufficient sample (N={ps['n']} trades)")
else:
    print(f"    No trades generated")

# ── equity curve / max-DD sketch ──────────────────────────────────────────────
print(f"\n  Equity / Max-DD sketch (trade-by-trade, 1 MNQ):")
if len(ps["pnls"]) > 0:
    pnl_cumsum = np.cumsum(ps["pnls"])
    max_dd     = float((pnl_cumsum - np.maximum.accumulate(pnl_cumsum)).min())
    final_pnl  = float(pnl_cumsum[-1])
    hwm        = float(np.maximum.accumulate(pnl_cumsum).max())

    print(f"    Final cumulative P&L:   ${final_pnl:>8,.0f}")
    print(f"    Peak equity (HWM):      ${hwm:>8,.0f}")
    print(f"    Max drawdown (HWM→trough): ${max_dd:>8,.0f}  "
          f"({'✅ inside $2k combine limit' if max_dd >= -2000 else '❌ EXCEEDS $2k combine limit'})")

    # rolling max-DD in any 20-trade window (regime stability)
    if len(ps["pnls"]) >= 20:
        roll_dds = []
        for i in range(len(ps["pnls"]) - 19):
            chunk = np.cumsum(ps["pnls"][i:i+20])
            dd = float((chunk - np.maximum.accumulate(chunk)).min())
            roll_dds.append(dd)
        print(f"    Worst rolling-20-trade DD: ${min(roll_dds):>8,.0f}")
else:
    print(f"    No trades — cannot sketch equity curve")

# ── gate 0 verdict — R/R-aware ────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  "
      f"(SM={PRIMARY_SM}×, TP={PRIMARY_TP}×)  [R/R-AWARE GATES]")
print(f"{'='*100}")
print(f"  Note: flat ≥50% WR is WRONG for 2:1 momentum.  "
      f"Breakeven WR = {be_wr_primary:.1%}.  Gate: WR ≥ {gate_wr_threshold:.1%}.")
print()

g_ev   = ps["avg_pnl"] > GATE0_EV_MIN
g_pf   = ps["pf"]      >= GATE0_PF_MIN
g_freq = ps["freq"]    >= GATE0_FREQ_MIN
g_stop = ps["stop_med"] <= GATE0_STOP_MAX
g_wr   = ps["wr"]      >= gate_wr_threshold
g_womo = ps["worst_mo_pnl"] >= GATE0_WOMO_PNL


def v(flag, label, measured):
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<54} [measured: {measured}]"


print(v(g_ev,   f"EV > $0 (avg net P&L > $0)",
                f"${ps['avg_pnl']:.2f}/trade"))
print(v(g_pf,   f"Profit factor ≥ {GATE0_PF_MIN:.2f}",
                f"{ps['pf']:.3f}"))
print(v(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN}/day",
                f"{ps['freq']:.2f}/day"))
print(v(g_stop, f"Median stop ≤ ${GATE0_STOP_MAX:.0f}/contract",
                f"${ps['stop_med']:.0f}"))
print(v(g_wr,   f"WR ≥ breakeven+5% (≥ {gate_wr_threshold:.1%})",
                f"{ps['wr']:.1%}"))
print(v(g_womo, f"Worst-month avg P&L ≥ ${GATE0_WOMO_PNL:.0f}/trade (variance guard)",
                f"${ps['worst_mo_pnl']:.2f}"))

all_gates   = [g_ev, g_pf, g_freq, g_stop, g_wr, g_womo]
edge_gates  = [g_ev, g_pf]          # if these fail → edge absent
var_gates   = [g_womo]              # if edge real but these fail → Track 3 stacking

gate_pass   = all(all_gates)
edge_real   = all(edge_gates) and g_freq and g_stop
variance_ok = g_womo

print()
if gate_pass:
    print("  ✅ GATE 0 PASS — momentum-WITH edge confirmed.")
    print("     Next: pre-register (clone prereg_stat_arb_short_seal.py)")
    print("     → Gate 1 full-combine backtest → Gate 2 OOS holdout.")
elif edge_real and not variance_ok:
    print("  ⚠️  EDGE REAL BUT VARIANCE FAILS — trigger Track 3 (Mary's stacking).")
    print(f"     PF={ps['pf']:.3f} ≥ 1.20, EV>$0, but worst-month avg=${ps['worst_mo_pnl']:.2f} < -$50.")
    print("     Stack with vol-compression-15min to dampen variance; measure pairwise correlation.")
elif not all(edge_gates):
    print("  ❌ GATE 0 FAIL — edge absent (EV≤$0 or PF<1.20).")
    print("     Momentum-WITH quadrant (PDH/PDL) does not have combine-viable edge.")
    print("     Record in memory and proceed to larger combine / GC post-catalyst.")
else:
    fails = [label for flag, label in [
        (g_ev,   f"EV ${ps['avg_pnl']:.2f} ≤ $0"),
        (g_pf,   f"PF {ps['pf']:.3f} < {GATE0_PF_MIN:.2f}"),
        (g_freq, f"Freq {ps['freq']:.2f}/d < {GATE0_FREQ_MIN}"),
        (g_stop, f"Med stop ${ps['stop_med']:.0f} > ${GATE0_STOP_MAX:.0f}"),
        (g_wr,   f"WR {ps['wr']:.1%} < breakeven+5% ({gate_wr_threshold:.1%})"),
        (g_womo, f"Worst-mo avg ${ps['worst_mo_pnl']:.2f} < ${GATE0_WOMO_PNL:.0f}"),
    ] if not flag]
    print("  ❌ GATE 0 FAIL.")
    for lbl in fails:
        print(f"     • {lbl}")

print(f"{'='*100}")

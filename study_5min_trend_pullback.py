"""
5-Min NQ Trend-Pullback — Gate 0 Study
Party recommendation 2026-06-09: Carson's "surf the wave, not fight it" idea.
First momentum-WITH strategy that actually uses momentum as the THESIS.

Strategy class (5-min EMA trend-pullback):
  1. Trend established: after first 30 min of RTH (confirmed at 10:00 ET),
     if 9-period EMA > 21-period EMA by ≥ 0.5 pts → long bias;
     if 9-period EMA < 21-period EMA by ≥ 0.5 pts → short bias; else no-trade day.
  2. Entry: first 5-min bar that (a) touches the fast EMA (bar's LOW ≤ ema_fast for
     long; bar's HIGH ≥ ema_fast for short) AND (b) closes back through it
     (close > ema_fast for long; close < ema_fast for short).
     = "EMA touch-and-recover" pullback. Enter at close of that bar.
  3. Stop: structure stop at the pullback bar's extreme (bar's LOW for long, bar's
     HIGH for short), clamped to $150/contract (combine cap). Never skip.
  4. TP: TP_MULT × stop_distance from entry in trend direction.
  5. Re-arm after each TP/STOP exit if trend still intact (ema_fast still above/
     below ema_slow). One position at a time. HOLD_MAX=24 bars (2 hours).
  6. Force-close all positions at 15:50 ET; no new entries after 15:50.

Why this is structurally different from all 9 prior failures:
  All prior momentum-WITH attempts (PDH/PDL breakout): traded WITH momentum but
  entered at market close after the level broke, with no entry filter. 68% of
  entries fired at 09:30 gap opens that immediately reversed — momentum without
  microstructure timing.
  This setup: enters ONLY at pullbacks into the 9-period EMA after the trend is
  confirmed, with a structure stop from the pullback bar's extreme. Uses the MNQ
  momentum-continuation tendency as the EXPECTED outcome (not fighting it), while
  getting a lower-risk entry point at the EMA touch.

Why path-shape compatible with the combine:
  On trending days: multiple moderate winners compounding into qualifying-day P&L.
  On choppy/flat days: no entry (EMA spread < 0.5 pts) → flat P&L, no bleed.
  Stop = structure from pullback bar low/high → concrete, bounded risk.

Primary spec (frozen before running):
  EMA_FAST=9, EMA_SLOW=21 on 5-min RTH bars.
  TREND_START="10:00", EMA_MIN_SPREAD=0.5 pts.
  Entry: touch-and-recover on 9 EMA. Stop: structure stop at pullback extreme.
  TP_MULT=2.0 (2:1 R/R). HOLD_MAX=24 bars. SESSION_CLOSE="15:50".
  Breakeven WR at 2:1 ≈ 35–38% net of commission (R/R-aware gate).

Grid (sensitivity only — NOT cherry-picked post-hoc):
  TP_MULT ∈ {1.5, 2.0, 3.0}. Primary = 2.0.

In-sample: 2025-01-01 → 2026-02-28 (same window as all prior Gate 0 studies)
Sealed holdout (≥2026-03-01) stays sealed until Gate 2, after pre-registration.
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_1MIN_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_1MIN_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

# ── parameters ─────────────────────────────────────────────────────────────────
EMA_FAST      = 9           # fast EMA period on 5-min bars (9 EMA)
EMA_SLOW      = 21          # slow EMA period on 5-min bars (21 EMA)
EMA_MIN_SPREAD = 0.5        # min |ema_fast - ema_slow| to declare trend (pts)
TREND_START   = "10:00"     # confirm trend at/after this time (first 30 min done)
SESSION_CLOSE = "15:50"     # force-close at this time; no new entries
HOLD_MAX      = 24          # max bars held (24 × 5 min = 2 hours)
MNQ_PV        = 2.0         # $/pt for 1 MNQ contract
COMMISSION    = 4.80        # round-trip commission
STOP_CAP_USD  = 150.0       # combine stop cap — clamp, never skip

TP_MULTS   = [1.5, 2.0, 3.0]
PRIMARY_TP = 2.0

# Gate 0 thresholds (R/R-aware — not flat ≥50% WR)
GATE0_EV_MIN   = 0.0
GATE0_PF_MIN   = 1.20
GATE0_FREQ_MIN = 1.0        # setups/day (combine: need ≥1/day for qualifying days)
GATE0_STOP_MAX = 150.0
GATE0_WOMO_PNL = -50.0      # worst-month avg P&L variance guard
ROLLING_5D_TARGET = 5 * 150.0


# ── load 1-min bars ────────────────────────────────────────────────────────────
def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


print("Loading 1-min bars…")
bars_all = pd.concat([load_et(MNQ_1MIN_2025), load_et(MNQ_1MIN_2026)])
bars_all = bars_all[~bars_all.index.duplicated(keep="first")]
bars_all = bars_all["2025-01-01":"2026-02-28"]

rth_1min = bars_all.between_time("09:30", "15:55").copy()
rth_1min["date"] = rth_1min.index.date

# ── resample 1-min → 5-min OHLCV ──────────────────────────────────────────────
print("Resampling to 5-min bars…")
rth_5min = rth_1min.resample("5min").agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volume=("volume", "sum"),
).dropna(subset=["close"])
rth_5min = rth_5min.between_time("09:30", "15:55")
rth_5min["date"] = rth_5min.index.date

# ── compute EMAs (continuous across the full series) ──────────────────────────
rth_5min["ema_fast"] = rth_5min["close"].ewm(span=EMA_FAST, adjust=False).mean()
rth_5min["ema_slow"] = rth_5min["close"].ewm(span=EMA_SLOW, adjust=False).mean()

n_days = len(set(rth_5min["date"]))
all_trading_days = sorted(set(rth_5min["date"]))
print(f"  5-min RTH bars: {len(rth_5min):,}  (~{len(rth_5min)/n_days:.0f}/day)  |  {n_days} trading days")
print(f"  Date range:     {rth_5min.index[0].date()} → {rth_5min.index[-1].date()}")
print(f"  EMA({EMA_FAST}/{EMA_SLOW}) spread (median absolute, post-10:00): ", end="")
post10 = rth_5min[rth_5min.index.strftime("%H:%M") >= TREND_START]
spread_abs = (post10["ema_fast"] - post10["ema_slow"]).abs()
print(f"{spread_abs.median():.2f} pts  (threshold: ≥{EMA_MIN_SPREAD} pts for trend)")


# ── simulation ─────────────────────────────────────────────────────────────────
def run_simulation(tp_mult: float):
    """
    Simulate 5-min NQ trend-pullback strategy.

    Per-day flow:
      1. At first bar at/after TREND_START, confirm trend from EMA cross.
         Skip day if |ema_fast - ema_slow| < EMA_MIN_SPREAD.
      2. For each subsequent bar: if no active trade, check for pullback entry.
         Entry: bar's extreme touches ema_fast AND close is back through ema_fast.
         Stop: bar's extreme (structure), capped at $150.
      3. Manage trade: TP / STOP / TIME / CLOSE.
      4. Re-arm after each exit if trend still intact. One position at a time.
    """
    trades      = []
    active      = None
    hold_count  = 0
    trend_dir   = 0           # 1=long bias, -1=short bias, 0=no trade
    trend_confirmed = False
    prev_date   = None

    hi_arr    = rth_5min["high"].values
    lo_arr    = rth_5min["low"].values
    cl_arr    = rth_5min["close"].values
    ema_f_arr = rth_5min["ema_fast"].values
    ema_s_arr = rth_5min["ema_slow"].values
    ts_arr    = rth_5min.index
    date_arr  = rth_5min["date"].values

    for k in range(len(rth_5min)):
        ts   = ts_arr[k]
        d    = date_arr[k]
        ema_f = ema_f_arr[k]
        ema_s = ema_s_arr[k]
        time_str = ts.strftime("%H:%M")

        if np.isnan(ema_f) or np.isnan(ema_s):
            continue

        # ── day boundary ─────────────────────────────────────────────────────
        if d != prev_date:
            prev_date = d
            trend_dir = 0
            trend_confirmed = False
            if active is not None:
                ep  = cl_arr[k - 1]
                pnl = (ep - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": ep, "pnl": pnl,
                               "win": pnl > 0, "reason": "DAYEND"})
                active = None; hold_count = 0

        # ── confirm trend at TREND_START ─────────────────────────────────────
        if not trend_confirmed and time_str >= TREND_START:
            spread = ema_f - ema_s
            if spread > EMA_MIN_SPREAD:
                trend_dir = 1
            elif spread < -EMA_MIN_SPREAD:
                trend_dir = -1
            else:
                trend_dir = 0    # flat/choppy — no trade today
            trend_confirmed = True

        if not trend_confirmed or trend_dir == 0:
            continue

        cl_k = cl_arr[k]; hi_k = hi_arr[k]; lo_k = lo_arr[k]

        # ── manage active trade ──────────────────────────────────────────────
        if active is not None:
            hold_count += 1
            at_close = time_str >= SESSION_CLOSE

            hit_tp   = ((active["dir"] ==  1 and hi_k >= active["tp"]) or
                        (active["dir"] == -1 and lo_k <= active["tp"]))
            hit_stop = ((active["dir"] ==  1 and lo_k <= active["stop"]) or
                        (active["dir"] == -1 and hi_k >= active["stop"]))

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": active["tp"],
                               "pnl": pnl, "win": True, "reason": "TP"})
                active = None; hold_count = 0
                continue   # don't re-enter same bar
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                trades.append({**active, "exit_p": active["stop"],
                               "pnl": pnl, "win": False, "reason": "STOP"})
                active = None; hold_count = 0
                continue
            elif at_close or hold_count >= HOLD_MAX:
                pnl = (cl_k - active["entry"]) * active["dir"] * MNQ_PV - COMMISSION
                reason = "CLOSE" if at_close else "TIME"
                trades.append({**active, "exit_p": cl_k,
                               "pnl": pnl, "win": pnl > 0, "reason": reason})
                active = None; hold_count = 0
            else:
                continue   # trade still live, skip entry check

        # ── entry check: only after TREND_START, before SESSION_CLOSE ────────
        if time_str < TREND_START or time_str >= SESSION_CLOSE:
            continue

        # Re-arm guard: skip if trend has flipped since confirmation
        spread_now = ema_f - ema_s
        if trend_dir == 1 and spread_now < 0:
            continue     # trend flipped bearish — no new longs
        if trend_dir == -1 and spread_now > 0:
            continue     # trend flipped bullish — no new shorts

        # EMA touch-and-recover pullback
        direction  = None
        stop_price = None

        if trend_dir == 1:
            # Long: bar touches ema_fast (lo ≤ ema_f) AND recovers (close > ema_f)
            if lo_k <= ema_f and cl_k > ema_f:
                direction  = 1
                stop_price = lo_k    # structure stop at pullback bar's low
        elif trend_dir == -1:
            # Short: bar touches ema_fast (hi ≥ ema_f) AND recovers (close < ema_f)
            if hi_k >= ema_f and cl_k < ema_f:
                direction  = -1
                stop_price = hi_k    # structure stop at pullback bar's high

        if direction is None:
            continue

        entry = cl_k
        stop_dist_raw = abs(entry - stop_price)
        if stop_dist_raw <= 0:
            continue    # degenerate: entry == pullback extreme

        stop_usd_raw = stop_dist_raw * MNQ_PV
        stop_usd     = min(stop_usd_raw, STOP_CAP_USD)
        stop_dist    = stop_usd / MNQ_PV

        stop_p = entry - direction * stop_dist
        tp_p   = entry + direction * stop_dist * tp_mult

        active = {
            "dir":       direction,
            "entry":     entry,
            "tp":        tp_p,
            "stop":      stop_p,
            "stop_dist": stop_dist,
            "stop_usd":  stop_usd,
            "stop_raw":  stop_usd_raw,
            "ema_fast":  ema_f,
            "ema_slow":  ema_s,
            "date":      d,
            "month":     ts.to_period("M"),
            "bar_of_day": time_str,
        }
        hold_count = 0

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
                    pnls=np.array([]), mo={}, mo_pnl={},
                    exit_tp=0, exit_stop=0, exit_time=0,
                    n_long=0, n_short=0, n_capped=0)
    n      = len(trades)
    wins   = sum(t["win"] for t in trades)
    pnls   = np.array([t["pnl"] for t in trades])
    stops  = np.array([t["stop_usd"] for t in trades])
    stops_raw = np.array([t["stop_raw"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf  = gross_w / max(1e-9, gross_l)
    mo: dict = {}; mo_pnl: dict = {}
    for t in trades:
        m = t["month"]
        mo.setdefault(m, [0, 0])
        mo[m][0 if t["win"] else 1] += 1
        mo_pnl.setdefault(m, [])
        mo_pnl[m].append(t["pnl"])
    worst_mo_wr  = min((w/(w+l) if w+l else 0) for w, l in mo.values()) if mo else 0.0
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


# ── main grid (TP_MULT) ────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"5-MIN TREND-PULLBACK GRID  "
      f"(EMA {EMA_FAST}/{EMA_SLOW}, structure stop capped ${STOP_CAP_USD:.0f}, 1 MNQ)")
print(f"{'='*100}")
print(f"  {'TP×':>4}  {'BEven':>6}  {'N':>5}  {'Freq/d':>8}  {'WR':>7}  "
      f"{'PF':>5}  {'AvgP&L':>8}  {'StopMed$':>9}  {'WorstMoP&L':>11}")
print(f"  {'----':>4}  {'------':>6}  {'---':>5}  {'-------':>8}  {'---':>7}  "
      f"{'----':>5}  {'------':>8}  {'--------':>9}  {'-----------':>11}")

grid_res: dict = {}
for tp in TP_MULTS:
    t = run_simulation(tp)
    s = summarise(t)
    grid_res[tp] = (t, s)
    is_primary = (tp == PRIMARY_TP)
    prim  = " ◀ PRIMARY" if is_primary else ""
    be_wr = ((s["stop_med"] + COMMISSION) / ((tp + 1) * s["stop_med"])
             if s["stop_med"] > 0 else 0.0)
    ev_f  = "✅" if s["avg_pnl"] > GATE0_EV_MIN   else "❌"
    pf_f  = "✅" if s["pf"]      >= GATE0_PF_MIN   else "❌"
    fr_f  = "✅" if s["freq"]    >= GATE0_FREQ_MIN  else "❌"
    print(f"  {tp:>3.1f}×  {be_wr:>6.1%}  {s['n']:>5}  "
          f"{s['freq']:>6.2f}/d{fr_f}  "
          f"{s['wr']:>7.1%}  {s['pf']:>4.2f}{pf_f}  "
          f"${s['avg_pnl']:>6.2f}{ev_f}  "
          f"${s['stop_med']:>7.0f}  "
          f"${s['worst_mo_pnl']:>9.2f}{prim}")

# ── primary spec deep dive ─────────────────────────────────────────────────────
pt, ps = grid_res[PRIMARY_TP]

print(f"\n{'='*100}")
print(f"PRIMARY SPEC DEEP DIVE  "
      f"(EMA {EMA_FAST}/{EMA_SLOW}, TP={PRIMARY_TP}×, HOLD_MAX={HOLD_MAX} bars × 5 min)")
print(f"{'='*100}")

print(f"\n  Funnel:")
print(f"    5-min RTH bars:     {len(rth_5min):,}")
print(f"    Trading days:       {n_days}")

# Count days with and without trend
trend_days = 0; no_trend_days = 0
dates_5min = sorted(set(rth_5min["date"]))
for d in dates_5min:
    day = rth_5min[rth_5min["date"] == d]
    post = day[day.index.strftime("%H:%M") >= TREND_START]
    if len(post) == 0:
        no_trend_days += 1; continue
    spread = float(post.iloc[0]["ema_fast"] - post.iloc[0]["ema_slow"])
    if abs(spread) >= EMA_MIN_SPREAD:
        trend_days += 1
    else:
        no_trend_days += 1

print(f"    Days with trend (|spread|≥{EMA_MIN_SPREAD}): {trend_days}  "
      f"({trend_days/max(1,n_days):.0%}) → eligible for entries")
print(f"    Days without trend (flat/choppy): {no_trend_days}  "
      f"({no_trend_days/max(1,n_days):.0%}) → no trades, no bleed")
print(f"    Trades taken:       {ps['n']}  ({ps['freq']:.2f}/day)")

if ps['n'] > 0:
    print(f"    Long / Short:       {ps['n_long']} / {ps['n_short']}  "
          f"({ps['n_long']/ps['n']:.0%} long, {ps['n_short']/ps['n']:.0%} short)")
    print(f"    Stop capped (>{STOP_CAP_USD:.0f}):   {ps['n_capped']}  "
          f"({ps['n_capped']/ps['n']:.0%} of trades)")
    print(f"    Exit breakdown:     "
          f"TP={ps['exit_tp']}  STOP={ps['exit_stop']}  "
          f"TIME/CLOSE={ps['exit_time']}")

print(f"\n  Performance:")
be_wr_primary = ((ps["stop_med"] + COMMISSION) / ((PRIMARY_TP + 1) * ps["stop_med"])
                 if ps["stop_med"] > 0 else 0.0)
gate_wr_threshold = be_wr_primary + 0.05
print(f"    Win rate:           {ps['wr']:.1%}  "
      f"(breakeven={be_wr_primary:.1%}; gate≥{gate_wr_threshold:.1%})")
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
    for label, subset in [("Long (uptrend pullback)", long_trades),
                          ("Short (downtrend pullback)", short_trades)]:
        if subset:
            ns = len(subset)
            ws = sum(t["win"] for t in subset)
            ps_sub = np.array([t["pnl"] for t in subset])
            gw = sum(p for p in ps_sub if p > 0); gl = abs(sum(p for p in ps_sub if p < 0))
            pf_s = gw / max(1e-9, gl)
            print(f"    {label:<28}  N={ns:>4}  WR={ws/ns:>6.1%}  "
                  f"PF={pf_s:>5.2f}  Avg=${ps_sub.mean():>7.2f}")
        else:
            print(f"    {label:<28}  N=0")

# ── by-month table ─────────────────────────────────────────────────────────────
print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'freq/d':>8}  {'Status'}")
for m in sorted(ps["mo"]):
    w, l   = ps["mo"][m]
    n_mo   = w + l
    mwr    = w / n_mo if n_mo else 0
    avg    = float(np.mean(ps["mo_pnl"].get(m, [0])))
    mo_bars = rth_5min[rth_5min.index.to_period("M") == m]
    mo_days = len(set(mo_bars["date"]))
    if n_mo < 3:
        flag = "⚠️ N<3"
    elif avg < GATE0_WOMO_PNL:
        flag = "❌ avg P&L<-$50"
    else:
        flag = "✅"
    print(f"  {str(m):<10}  {n_mo:>5}  {mwr:>7.1%}  ${avg:>7.2f}  "
          f"{n_mo/max(1, mo_days):>7.2f}/d  {flag}")

# ── time-of-day distribution ──────────────────────────────────────────────────
if pt:
    print(f"\n  Time-of-day distribution (when entry fires):")
    from collections import Counter
    tod = Counter(t["bar_of_day"] for t in pt)
    buckets: dict = {}
    for time_str, cnt in tod.items():
        h, mi = int(time_str[:2]), int(time_str[3:])
        bucket = f"{h:02d}:{(mi//60)*60:02d}" if False else time_str  # hourly buckets
        # bucket by hour
        bucket = f"{h:02d}:00"
        buckets[bucket] = buckets.get(bucket, 0) + cnt
    total_t = sum(buckets.values())
    for b in sorted(buckets):
        cnt = buckets[b]
        bar = "█" * int(cnt / max(buckets.values()) * 20)
        print(f"    {b}  {bar:<20}  {cnt:>4} ({cnt/total_t:.0%})")

# ── path-shape: qualifying-day analysis ───────────────────────────────────────
print(f"\n  Path-shape: qualifying-day analysis (combine needs ≥$150/session):")
if pt:
    day_pnl: dict = {}
    for t in pt:
        d = t["date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + t["pnl"]

    daily_series = [day_pnl.get(d, 0.0) for d in all_trading_days]

    days_qualifying = sum(1 for p in day_pnl.values() if p >= 150.0)
    days_flat       = sum(1 for d in all_trading_days if d not in day_pnl)
    days_losing     = sum(1 for p in day_pnl.values() if p < 0)
    days_small_pos  = sum(1 for p in day_pnl.values() if 0 <= p < 150.0)

    print(f"    Days qualifying (≥$150):  {days_qualifying}/{n_days} = {days_qualifying/n_days:.0%}")
    print(f"    Days no-trade (flat):     {days_flat}/{n_days} = {days_flat/n_days:.0%}")
    print(f"    Days losing (<$0):        {days_losing}/{n_days} = {days_losing/n_days:.0%}")
    print(f"    Days small positive:      {days_small_pos}/{n_days} = {days_small_pos/n_days:.0%}")
    print(f"    Avg daily P&L (trade days): ${float(np.mean([p for p in day_pnl.values()])):>7.2f}")
    print(f"    Median daily P&L (trade days): ${float(np.median([p for p in day_pnl.values()])):>7.2f}")

# ── Victor's rolling-5-day variance diagnostic ────────────────────────────────
print(f"\n  Victor's rolling-5-day variance check (combine DD guard):")
if pt:
    if len(daily_series) >= 5:
        rolling5  = [sum(daily_series[i:i+5]) for i in range(len(daily_series) - 4)]
        worst5    = min(rolling5)
        best5     = max(rolling5)
        median5   = float(np.median(rolling5))
        pct_neg   = sum(1 for x in rolling5 if x < 0) / len(rolling5)
        worst_day = min(daily_series)
        pct_profit = sum(1 for x in daily_series if x > 0) / len(daily_series)

        print(f"    Worst  rolling 5-day P&L:  ${worst5:,.0f}  "
              f"({'✅ positive' if worst5 >= 0 else '❌ negative'})")
        print(f"    Best   rolling 5-day P&L:  ${best5:,.0f}")
        print(f"    Median rolling 5-day P&L:  ${median5:,.0f}")
        print(f"    % of 5-day windows < $0:   {pct_neg:.0%}")
        print(f"    Worst single-day P&L:       ${worst_day:,.0f}")
        print(f"    Days with P&L > $0:         {pct_profit:.0%}  of {n_days} days")
else:
    print(f"    No trades generated")

# ── equity / max-DD sketch ─────────────────────────────────────────────────────
print(f"\n  Equity / Max-DD sketch (trade-by-trade, 1 MNQ):")
if len(ps["pnls"]) > 0:
    pnl_cumsum = np.cumsum(ps["pnls"])
    max_dd     = float((pnl_cumsum - np.maximum.accumulate(pnl_cumsum)).min())
    final_pnl  = float(pnl_cumsum[-1])
    hwm        = float(np.maximum.accumulate(pnl_cumsum).max())

    print(f"    Final cumulative P&L:      ${final_pnl:>8,.0f}")
    print(f"    Peak equity (HWM):         ${hwm:>8,.0f}")
    print(f"    Max drawdown (HWM→trough): ${max_dd:>8,.0f}  "
          f"({'✅ inside $2k combine limit' if max_dd >= -2000 else '❌ EXCEEDS $2k combine limit'})")
    if len(ps["pnls"]) >= 20:
        roll_dds = []
        for i in range(len(ps["pnls"]) - 19):
            chunk = np.cumsum(ps["pnls"][i:i+20])
            dd = float((chunk - np.maximum.accumulate(chunk)).min())
            roll_dds.append(dd)
        print(f"    Worst rolling-20-trade DD: ${min(roll_dds):>8,.0f}")
else:
    print(f"    No trades — cannot sketch equity curve")

# ── gate 0 verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"GATE 0 VERDICT — PRIMARY SPEC  (TP={PRIMARY_TP}×)  [R/R-AWARE GATES]")
print(f"{'='*100}")
print(f"  Note: flat ≥50% WR is WRONG for {PRIMARY_TP}:1 R/R momentum.  "
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


print(v(g_ev,   "EV > $0 (avg net P&L > $0)",
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

all_gates  = [g_ev, g_pf, g_freq, g_stop, g_wr, g_womo]
edge_gates = [g_ev, g_pf]
var_gates  = [g_womo]

gate_pass  = all(all_gates)
edge_real  = all(edge_gates) and g_freq and g_stop
variance_ok = g_womo

print()
if gate_pass:
    print("  ✅ GATE 0 PASS — 5-min trend-pullback edge confirmed.")
    print("     Next: pre-register → Gate 1 combine backtest → Gate 2 OOS holdout.")
elif edge_real and not variance_ok:
    print("  ⚠️  EDGE REAL BUT VARIANCE FAILS — consider Track 3 stacking or EMA-spread filter.")
    print(f"     PF={ps['pf']:.3f} ≥ 1.20, EV>$0, but worst-month avg=${ps['worst_mo_pnl']:.2f} < -$50.")
elif not all(edge_gates):
    print("  ❌ GATE 0 FAIL — edge absent (EV≤$0 or PF<1.20).")
    print("     5-min trend-pullback does not have combine-viable edge at primary spec.")
    print("     Record in memory; evaluate GC post-catalyst or instrument pivot.")
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

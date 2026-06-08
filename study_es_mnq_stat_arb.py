"""
ES/MNQ Stat Arb — Proper Gate 0 Simulation
Entry:  5-bar cumulative MNQ divergence from beta-predicted ES exceeds THRESH pts
Trade:  fade the divergence — expect MNQ to give back the overshoot
Target: MNQ recovers exactly the divergence from entry price (1× TP)
Stop:   MNQ moves STOP_MULT× the divergence further against us from entry
State:  one trade at a time; session close (15:55 ET) forces exit
Grid:   THRESH ∈ {5, 10, 15, 25} × STOP_MULT ∈ {1.0, 2.0}

PRIMARY SPEC (pre-frozen before reading results):
    THRESH=10 pts ($20/contract), STOP_MULT=1.0 (1:1 R), need WR>55%
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

MNQ_PATH  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_PATH   = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

BETA_WIN   = 60    # bars for rolling beta
SPREAD_WIN = 5     # look-back bars for cumulative divergence
HOLD_MAX   = 30    # max bars held before forced exit
SESSION_CLOSE = "15:55"
RTH_START     = "09:30"
MNQ_PV        = 2.0   # $/point
COMMISSION    = 4.80  # round-trip $/contract

THRESHOLDS  = [5, 10, 15, 25]
STOP_MULTS  = [1.0, 2.0]

# Primary spec (frozen before reading grid)
PRIMARY_THRESH     = 10
PRIMARY_STOP_MULT  = 1.0
GATE0_WR_MIN       = 0.55
GATE0_FREQ_MIN     = 1.0
GATE0_STOP_USD_MAX = 150.0
GATE0_WORST_MO_WR  = 0.35

# ── data loading ──────────────────────────────────────────────────────────────
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

# ── rolling beta (price-change space) ────────────────────────────────────────
rth["mnq_chg"] = rth["mnq"].diff()
rth["es_chg"]  = rth["es"].diff()
roll_cov = rth["mnq_chg"].rolling(BETA_WIN).cov(rth["es_chg"])
roll_var = rth["es_chg"].rolling(BETA_WIN).var()
rth["beta"] = (roll_cov / roll_var.replace(0, np.nan)).ffill().clip(0, 10)

# ── 5-bar cumulative divergence ───────────────────────────────────────────────
rth["div"] = (rth["mnq_chg"].rolling(SPREAD_WIN).sum()
              - rth["beta"] * rth["es_chg"].rolling(SPREAD_WIN).sum())
rth = rth.dropna(subset=["div", "beta"])

# ── trade simulation (one trade at a time per day) ────────────────────────────
@dataclass
class Trade:
    date: object
    ts_entry: object
    direction: int       # +1 long MNQ, -1 short MNQ
    entry_price: float
    tp_price: float
    stop_price: float
    div_pts: float       # abs divergence at entry
    thresh: float
    stop_mult: float
    exit_price: float = 0.0
    win: bool = False
    exit_reason: str = ""

    def net_pnl(self):
        pts = (self.exit_price - self.entry_price) * self.direction
        return pts * MNQ_PV - COMMISSION


def run_simulation(thresh: float, stop_mult: float):
    """Simulate stat arb on RTH bars, one trade at a time."""
    trades: list[Trade] = []
    active: Trade | None = None
    hold_count = 0

    mnq_arr  = rth["mnq"].values
    div_arr  = rth["div"].values
    ts_arr   = rth.index
    date_arr = rth.index.normalize()

    for k in range(len(rth)):
        ts   = ts_arr[k]
        mnq_k = mnq_arr[k]
        div_k = div_arr[k]
        date_k = date_arr[k]

        # ── manage active trade ──
        if active is not None:
            hit_tp   = (active.direction == 1  and mnq_k >= active.tp_price) or \
                       (active.direction == -1 and mnq_k <= active.tp_price)
            hit_stop = (active.direction == 1  and mnq_k <= active.stop_price) or \
                       (active.direction == -1 and mnq_k >= active.stop_price)
            at_close = ts.strftime("%H:%M") >= SESSION_CLOSE
            hold_count += 1

            if hit_tp:
                active.exit_price  = active.tp_price
                active.win         = True
                active.exit_reason = "TP"
                trades.append(active)
                active = None; hold_count = 0
            elif hit_stop:
                active.exit_price  = active.stop_price
                active.win         = False
                active.exit_reason = "STOP"
                trades.append(active)
                active = None; hold_count = 0
            elif at_close or hold_count >= HOLD_MAX:
                active.exit_price  = mnq_k
                active.win         = (active.direction == 1  and mnq_k > active.entry_price) or \
                                     (active.direction == -1 and mnq_k < active.entry_price)
                active.exit_reason = "CLOSE" if at_close else "TIME"
                trades.append(active)
                active = None; hold_count = 0
            # stay in trade otherwise
            continue

        # ── look for new entry ──
        if abs(div_k) < thresh:
            continue
        # only first signal of each direction per session (no re-entry same direction same day)
        direction = -1 if div_k > 0 else +1   # fade: short if overbought, long if underperformed
        entry_price = mnq_k
        div_pts_abs = abs(div_k)
        tp_price   = entry_price + direction * div_pts_abs
        stop_price = entry_price - direction * div_pts_abs * stop_mult
        stop_usd   = div_pts_abs * stop_mult * MNQ_PV

        if stop_usd > GATE0_STOP_USD_MAX:
            continue  # skip if stop exceeds combine cap

        active = Trade(
            date=date_k, ts_entry=ts, direction=direction,
            entry_price=entry_price, tp_price=tp_price, stop_price=stop_price,
            div_pts=div_pts_abs, thresh=thresh, stop_mult=stop_mult,
        )
        hold_count = 0

    # flush any open trade at end
    if active is not None:
        active.exit_price  = mnq_arr[-1]
        active.win         = False
        active.exit_reason = "END"
        trades.append(active)

    return trades


# ── run grid ──────────────────────────────────────────────────────────────────
def summarise(trades):
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, worst_mo_wr=0.0,
                    stop_med_usd=0.0, stop_p75_usd=0.0)
    n     = len(trades)
    wins  = sum(t.win for t in trades)
    wr    = wins / n
    n_days = rth.index.normalize().nunique()
    freq  = n / n_days
    pnls  = np.array([t.net_pnl() for t in trades])
    stops = np.array([t.div_pts * t.stop_mult * MNQ_PV for t in trades])

    # by-month WR
    mo = {}
    for t in trades:
        m = pd.Timestamp(str(t.date)).to_period("M")
        mo.setdefault(m, [0, 0])
        mo[m][0 if t.win else 1] += 1
    worst_mo = min((w/(w+l) if w+l else 0) for w, l in mo.values()) if mo else 0.0

    return dict(n=n, wr=wr, freq=freq, avg_pnl=pnls.mean(),
                worst_mo_wr=worst_mo,
                stop_med_usd=float(np.median(stops)),
                stop_p75_usd=float(np.percentile(stops, 75)),
                exit_tp=sum(1 for t in trades if t.exit_reason=="TP"),
                exit_stop=sum(1 for t in trades if t.exit_reason=="STOP"),
                exit_time=sum(1 for t in trades if t.exit_reason in ("TIME","CLOSE","END")),
                by_month=mo, pnls=pnls)

n_days = rth.index.normalize().nunique()

print(f"\n{'='*80}")
print(f"STAT ARB GRID  (1-trade-at-a-time, stop cap ${GATE0_STOP_USD_MAX}/contract)")
print(f"{'='*80}")
print(f"  {'Thresh':>6}  {'Stop×':>6}  {'N':>5}  {'Freq/d':>7}  {'WR':>7}  "
      f"{'Avg P&L':>9}  {'Stop med $':>11}  {'Worst-mo':>9}")
print(f"  {'------':>6}  {'-----':>6}  {'---':>5}  {'------':>7}  {'---':>7}  "
      f"{'-------':>9}  {'----------':>11}  {'--------':>9}")

all_results = {}
for th in THRESHOLDS:
    for sm in STOP_MULTS:
        trades = run_simulation(th, sm)
        s = summarise(trades)
        all_results[(th, sm)] = (trades, s)
        prim = "◀ PRIMARY" if th == PRIMARY_THRESH and sm == PRIMARY_STOP_MULT else ""
        wr_flag = "✅" if s["wr"] >= 0.50 else "❌"
        print(f"  {th:>5}pt  {sm:>5.1f}×  {s['n']:>5}  {s['freq']:>7.2f}/d  "
              f"{s['wr']:>7.1%}{wr_flag}  ${s['avg_pnl']:>7.2f}  "
              f"${s['stop_med_usd']:>9.0f}  {s['worst_mo_wr']:>8.1%}  {prim}")

# ── primary spec deep dive ────────────────────────────────────────────────────
primary_trades, ps = all_results[(PRIMARY_THRESH, PRIMARY_STOP_MULT)]

print(f"\n{'='*80}")
print(f"PRIMARY SPEC — THRESH={PRIMARY_THRESH}pts (${PRIMARY_THRESH*MNQ_PV:.0f}/contract), "
      f"STOP={PRIMARY_STOP_MULT}×, 1:1 R/R")
print(f"{'='*80}")
print(f"\n  Funnel:")
print(f"    RTH bars total:          {len(rth):,}")
print(f"    Divergence events seen:  "
      f"{(rth['div'].abs() > PRIMARY_THRESH).sum():,}")
print(f"    Trades taken (1-at-a-time, stop≤${GATE0_STOP_USD_MAX}): {ps['n']}")
print(f"    Exit breakdown:  TP={ps.get('exit_tp',0)}  STOP={ps.get('exit_stop',0)}  "
      f"TIME/CLOSE={ps.get('exit_time',0)}")

print(f"\n  Performance:")
pnls = ps["pnls"]
print(f"    N trades:    {ps['n']}  ({ps['freq']:.2f}/day over {n_days} days)")
print(f"    Win rate:    {ps['wr']:.1%}")
pf = (sum(p for p in pnls if p>0)) / max(1e-9, abs(sum(p for p in pnls if p<0)))
print(f"    Profit factor: {pf:.2f}")
print(f"    Avg net P&L: ${ps['avg_pnl']:.2f}/contract")
print(f"    Total P&L:   ${pnls.sum():.0f}  (1 MNQ contract, {n_days} days)")
print(f"    Med stop:    ${ps['stop_med_usd']:.0f}/contract")
print(f"    75th-pct stop: ${ps['stop_p75_usd']:.0f}/contract")
print(f"    Worst-month WR: {ps['worst_mo_wr']:.1%}")

print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'Avg P&L':>10}  {'freq/d':>8}")
mo_d = ps["by_month"]
mo_pnl: dict = {}
for t in primary_trades:
    m = pd.Timestamp(str(t.date)).to_period("M")
    mo_pnl.setdefault(m, [])
    mo_pnl[m].append(t.net_pnl())

for m in sorted(mo_d):
    w, l = mo_d[m]
    n = w + l
    mwr = w / n if n else 0
    avg = np.mean(mo_pnl.get(m, [0]))
    # estimate days in month from bar count
    mo_bars = rth[rth.index.to_period("M") == m]
    mo_days = mo_bars.index.normalize().nunique()
    freq = n / max(1, mo_days)
    flag = "❌" if mwr < GATE0_WORST_MO_WR else "✅"
    print(f"  {str(m):<10}  {n:>5}  {mwr:>7.1%}  ${avg:>8.2f}  {freq:>7.2f}/d  {flag}")

# ── gate 0 verdict ────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"GATE 0 VERDICT — PRIMARY SPEC "
      f"(THRESH={PRIMARY_THRESH}pts, STOP={PRIMARY_STOP_MULT}×)")
print(f"{'='*80}")
g_wr   = ps["wr"]    >= GATE0_WR_MIN
g_freq = ps["freq"]  >= GATE0_FREQ_MIN
g_stop = ps["stop_med_usd"] <= GATE0_STOP_USD_MAX
g_womo = ps["worst_mo_wr"]  >= GATE0_WORST_MO_WR

def verdict(flag, label, measured, unit=""):
    sym = "✅ PASS" if flag else "❌ FAIL"
    return f"  {sym}  {label:<38} [measured: {measured}{unit}]"

print(verdict(g_wr,   f"Win rate ≥ {GATE0_WR_MIN:.0%}", f"{ps['wr']:.1%}"))
print(verdict(g_freq, f"Frequency ≥ {GATE0_FREQ_MIN} trades/day",
              f"{ps['freq']:.2f}", "/day"))
print(verdict(g_stop, f"Median stop ≤ ${GATE0_STOP_USD_MAX}/contract",
              f"${ps['stop_med_usd']:.0f}"))
print(verdict(g_womo, f"Worst-month WR ≥ {GATE0_WORST_MO_WR:.0%}",
              f"{ps['worst_mo_wr']:.1%}"))

gate_pass = all([g_wr, g_freq, g_stop, g_womo])
print()
if gate_pass:
    print("  ✅ GATE 0 PASS — proceed to pre-registration and full backtest.")
else:
    fails = [l for l, f in [
        (f"WR {ps['wr']:.1%} < {GATE0_WR_MIN:.0%}", not g_wr),
        (f"Frequency {ps['freq']:.2f}/day < {GATE0_FREQ_MIN}", not g_freq),
        (f"Median stop ${ps['stop_med_usd']:.0f} > ${GATE0_STOP_USD_MAX}", not g_stop),
        (f"Worst-month WR {ps['worst_mo_wr']:.1%} < {GATE0_WORST_MO_WR:.0%}", not g_womo),
    ] if f]
    print(f"  ❌ GATE 0 FAIL. Failing criteria:")
    for f in fails:
        print(f"     • {f}")
print(f"{'='*80}")

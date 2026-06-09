#!/usr/bin/env python3
"""
backtest_stat_arb_short.py — Gate 1 Full Combine Backtest: ES/MNQ Stat Arb Short-Only

Pre-registration : _bmad-output/preregistration_stat_arb_short_combine.md
Config           : stat_arb_short_config.yaml   (SHA-256 verified)
Gate 0 study     : study_stat_arb_short_only.py  (SHA-256 verified)

Adds over Gate 0 diagnostic:
  • Hash verification — aborts if config or study modified since pre-reg
  • Trailing drawdown tracking (HWM-based, checked against combine $2k limit)
  • Daily halt: session P&L ≤ daily_loss_halt → no new entries rest of day
  • Qualifying day tracking (≥ qualifying_day_min session P&L)
  • Daily consistency cap check (50% rule — WARNING only, not a stop gate)
  • Victor's rolling-5-day P&L variance diagnostic
  • Trailing-DD path by month

Gate 1 criteria (pre-registration, immutable after commit 2e9fb90):
  Required (STOP if below):
    WR ≥ 56%   |  Avg net P&L > $0   |  PF ≥ 1.20
    Max trailing DD in-sample ≤ $1,500
    Freq ≥ 1.0/day   |  N ≥ 80   |  Worst-month WR ≥ 35%
  Warning only:
    Qualifying sessions / last 20 ≥ 6
    Largest single day ≤ 50% of total gross profit
"""

import csv
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT   = Path(__file__).parent
PREREG = ROOT / "_bmad-output" / "preregistration_stat_arb_short_combine.md"
CONFIG = ROOT / "stat_arb_short_config.yaml"
STUDY  = ROOT / "study_stat_arb_short_only.py"

# Gate 1 thresholds (from pre-registration — DO NOT CHANGE)
G1_WR_MIN       = 0.56
G1_AVGPNL_MIN   = 0.0
G1_PF_MIN       = 1.20
G1_DD_MAX_ABS   = 1500.0   # max |trailing DD| allowed in-sample
G1_FREQ_MIN     = 1.0
G1_N_MIN        = 80
G1_WORSTMO_MIN  = 0.35
# Warnings only:
G1_QUAL_WARN    = 6        # qualifying sessions in last 20 days
G1_CONSIST_WARN = 0.50     # largest single day / total gross profit

SEP  = "=" * 88
SEP2 = "-" * 88


# ── 1. Hash verification ──────────────────────────────────────────────────────
def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_hashes() -> bool:
    print(SEP)
    print("HASH VERIFICATION")
    print(SEP)
    try:
        text = PREREG.read_text()
    except FileNotFoundError:
        print(f"  ERROR: pre-registration not found at {PREREG}")
        return False

    m_a = re.search(r"\(a\)[^|]+\|[^|]+\|\s*`([0-9a-f]{64})`", text)
    m_b = re.search(r"\(b\)[^|]+\|[^|]+\|\s*`([0-9a-f]{64})`", text)
    m_c = re.search(r"\(c\)[^|]+\|[^|]+\|\s*`([0-9a-f]{40})`", text)

    if not m_a or not m_b:
        print("  ERROR: could not parse hashes from pre-registration document")
        return False

    reg_a = m_a.group(1)
    reg_b = m_b.group(1)
    reg_c = m_c.group(1) if m_c else "(not found)"

    cur_a = _sha256(CONFIG)
    cur_b = _sha256(STUDY)

    ok_a = reg_a == cur_a
    ok_b = reg_b == cur_b

    print(f"  Pre-registration sealed at commit: {reg_c}")
    print(f"  Backtest commit: (run `git rev-parse HEAD` to confirm post-prereg)")
    print()
    print(f"  Config hash  (a): {'✅ MATCH' if ok_a else '❌ MISMATCH'}")
    if not ok_a:
        print(f"    pre-reg : {reg_a}")
        print(f"    current : {cur_a}")
    else:
        print(f"    {cur_a}")
    print(f"  Study  hash  (b): {'✅ MATCH' if ok_b else '❌ MISMATCH'}")
    if not ok_b:
        print(f"    pre-reg : {reg_b}")
        print(f"    current : {cur_b}")
    else:
        print(f"    {cur_b}")

    if not (ok_a and ok_b):
        print()
        print("  ❌ HASH MISMATCH — backtest aborted.")
        print("     Config or study script modified since pre-registration.")
        print("     A new pre-registration cycle is required before backtest.")
        return False

    print()
    print("  ✅ All hashes verified — running against pre-registered artifacts.")
    return True


if not verify_hashes():
    sys.exit(1)

print()


# ── 2. Load config ─────────────────────────────────────────────────────────────
cfg = yaml.safe_load(CONFIG.read_text())

THRESH        = float(cfg["threshold_pts"])
STOP_MULT     = float(cfg["stop_mult"])
BETA_WIN      = int(cfg["beta_window"])
SPREAD_WIN    = int(cfg["spread_window"])
HOLD_MAX      = int(cfg["hold_max_bars"])
RTH_START     = cfg["rth_start"]
SESSION_CLOSE = cfg["session_close"]
MNQ_PV        = float(cfg["mnq_point_value"])
COMMISSION    = float(cfg["commission_rt"])
STOP_CAP      = float(cfg["stop_cap_usd"])
DAILY_HALT    = float(cfg["daily_loss_halt"])
QUAL_DAY_MIN  = float(cfg["qualifying_day_min"])
COMB_DD_LIM   = float(cfg["combine_trailing_dd"])
COMB_PT_TARG  = float(cfg["combine_profit_target"])
IS_START      = cfg["in_sample_start"]
IS_END        = cfg["in_sample_end"]
CONTRACTS     = 1  # pre-registration strategy logic: one trade at a time, 1 contract

print(SEP)
print("CONFIG")
print(SEP)
print(f"  strategy   : {cfg['strategy']}  {cfg['version']}")
print(f"  direction  : SHORT ONLY (fade MNQ outperformance of ES)")
print(f"  THRESH={THRESH}pt  STOP_MULT={STOP_MULT}×  BETA_WIN={BETA_WIN}  SPREAD_WIN={SPREAD_WIN}")
print(f"  HOLD_MAX={HOLD_MAX}bars  RTH_START={RTH_START}  SESSION_CLOSE={SESSION_CLOSE}")
print(f"  MNQ_PV=${MNQ_PV}  COMMISSION=${COMMISSION}  STOP_CAP=${STOP_CAP}")
print(f"  DAILY_HALT=${DAILY_HALT}  QUAL_DAY_MIN=${QUAL_DAY_MIN}")
print(f"  COMBINE: DD_LIMIT=${COMB_DD_LIM}  PROFIT_TARGET=${COMB_PT_TARG}")
print(f"  POSITION SIZE: {CONTRACTS} MNQ contract  |  IN-SAMPLE: {IS_START} → {IS_END}")
print()


# ── 3. Load and process data ──────────────────────────────────────────────────
def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


print("Loading bars…")
mnq = pd.concat([load_et(ROOT / p) for p in cfg["mnq_paths"]])
mnq = mnq[~mnq.index.duplicated(keep="first")]
es  = load_et(ROOT / cfg["es_path"])

both = (mnq[["close"]].rename(columns={"close": "mnq"})
        .join(es[["close"]].rename(columns={"close": "es"}), how="inner"))
both = both[IS_START:IS_END]
rth  = both.between_time(RTH_START, SESSION_CLOSE).copy()

print(f"  RTH bars: {len(rth):,}  |  {rth.index.normalize().nunique()} days  "
      f"({rth.index[0].date()} → {rth.index[-1].date()})")

rth["mnq_chg"] = rth["mnq"].diff()
rth["es_chg"]  = rth["es"].diff()
roll_cov  = rth["mnq_chg"].rolling(BETA_WIN).cov(rth["es_chg"])
roll_var  = rth["es_chg"].rolling(BETA_WIN).var()
rth["beta"] = (roll_cov / roll_var.replace(0, np.nan)).ffill().clip(0, 10)
rth["div"]  = (rth["mnq_chg"].rolling(SPREAD_WIN).sum()
               - rth["beta"] * rth["es_chg"].rolling(SPREAD_WIN).sum())
rth = rth.dropna(subset=["div", "beta"])

n_days   = rth.index.normalize().nunique()
all_days = sorted(set(rth.index.normalize().date))
mnq_arr  = rth["mnq"].values
div_arr  = rth["div"].values
ts_arr   = rth.index

print()


# ── 4. Gate 1 simulation ──────────────────────────────────────────────────────
print(SEP)
print(f"GATE 1 SIMULATION  (THRESH={THRESH}pt, STOP={STOP_MULT}×, SHORT ONLY, {CONTRACTS}C)")
print(SEP)

trades: list     = []
active           = None
hold_count       = 0

equity           = 0.0
hwm              = 0.0
min_trailing_dd  = 0.0   # worst (most negative) trailing DD seen

session_date     = None
session_pnl      = 0.0
session_halted   = False
day_pnls: dict   = {}
halted_dates: set = set()

for k in range(len(rth)):
    ts    = ts_arr[k]
    mnq_k = mnq_arr[k]
    div_k = div_arr[k]
    today = ts.date()

    # Day boundary: flush previous session stats
    if today != session_date:
        if session_date is not None:
            day_pnls[session_date] = session_pnl
            if session_halted:
                halted_dates.add(session_date)
        session_date   = today
        session_pnl    = 0.0
        session_halted = False

    # Advance active trade
    if active is not None:
        hit_tp   = mnq_k <= active["tp"]
        hit_stop = mnq_k >= active["stop"]
        at_close  = ts.strftime("%H:%M") >= SESSION_CLOSE
        hold_count += 1

        if hit_tp or hit_stop or at_close or hold_count >= HOLD_MAX:
            if hit_tp:
                exit_px = active["tp"]
                reason  = "TP"
            elif hit_stop:
                exit_px = active["stop"]
                reason  = "STOP"
            else:
                exit_px = mnq_k
                reason  = "CLOSE" if at_close else "TIME"

            pnl_1c  = (exit_px - active["entry"]) * (-1) * MNQ_PV - COMMISSION
            net_pnl = pnl_1c * CONTRACTS
            equity     += net_pnl
            session_pnl += net_pnl
            hwm             = max(hwm, equity)
            trailing_dd     = equity - hwm
            min_trailing_dd = min(min_trailing_dd, trailing_dd)

            trades.append({
                **active,
                "exit"        : exit_px,
                "pnl"         : net_pnl,
                "pnl_1c"      : pnl_1c,
                "win"         : pnl_1c > 0,
                "reason"      : reason,
                "equity"      : equity,
                "trailing_dd" : trailing_dd,
            })
            active     = None
            hold_count = 0

            if not session_halted and session_pnl <= DAILY_HALT:
                session_halted = True

        continue   # no entry on same bar as active (or mid-trade)

    # Entry: short only when divergence exceeds threshold
    if session_halted:
        continue
    if div_k <= THRESH:
        continue

    div_abs     = div_k           # always positive at entry
    stop_usd_1c = div_abs * STOP_MULT * MNQ_PV
    if stop_usd_1c > STOP_CAP:
        continue

    active = {
        "entry"    : mnq_k,
        "div"      : div_abs,
        "tp"       : mnq_k - div_abs,
        "stop"     : mnq_k + div_abs * STOP_MULT,
        "stop_usd" : stop_usd_1c,
        "date"     : today,
        "month"    : ts.to_period("M"),
        "hour"     : ts.hour,
    }
    hold_count = 0

# Flush final day
if session_date is not None:
    day_pnls[session_date] = session_pnl
    if session_halted:
        halted_dates.add(session_date)

# Flush remaining open trade at end of in-sample period
if active:
    exit_px = mnq_arr[-1]
    pnl_1c  = (exit_px - active["entry"]) * (-1) * MNQ_PV - COMMISSION
    net_pnl = pnl_1c * CONTRACTS
    equity     += net_pnl
    session_pnl += net_pnl
    hwm             = max(hwm, equity)
    trailing_dd     = equity - hwm
    min_trailing_dd = min(min_trailing_dd, trailing_dd)
    trades.append({**active, "exit": exit_px, "pnl": net_pnl, "pnl_1c": pnl_1c,
                   "win": pnl_1c > 0, "reason": "END",
                   "equity": equity, "trailing_dd": trailing_dd})


# ── 5. Compute Gate 1 metrics ─────────────────────────────────────────────────
N    = len(trades)
if N == 0:
    print("  ERROR: no trades generated — check data paths and config")
    sys.exit(1)

pnls_1c  = np.array([t["pnl_1c"] for t in trades])
wins_arr = pnls_1c[pnls_1c > 0]
loss_arr = pnls_1c[pnls_1c < 0]
gross_w  = wins_arr.sum() if len(wins_arr) else 0.0
gross_l  = abs(loss_arr.sum()) if len(loss_arr) else 0.0
pf       = gross_w / max(1e-9, gross_l)
wr       = sum(1 for t in trades if t["win"]) / N
freq     = N / n_days
avg_pnl  = pnls_1c.mean()

# By-month stats
month_stats: dict = {}
for t in trades:
    m = t["month"]
    month_stats.setdefault(m, {"w": 0, "l": 0, "pnls": []})
    month_stats[m]["w" if t["win"] else "l"] += 1
    month_stats[m]["pnls"].append(t["pnl_1c"])

worst_mo_wr = min(
    (s["w"] / (s["w"] + s["l"]) if s["w"] + s["l"] else 0.0)
    for s in month_stats.values()
) if month_stats else 0.0

# Qualifying days
qual_days    = [d for d, p in day_pnls.items() if p >= QUAL_DAY_MIN]
last_20_days = set(all_days[-20:])
qual_last_20 = [d for d in qual_days if d in last_20_days]

# Daily consistency (50% rule)
pos_day_pnls = [p for p in day_pnls.values() if p > 0]
total_profit = sum(pos_day_pnls) if pos_day_pnls else 0.0
max_day_pnl  = max(day_pnls.values()) if day_pnls else 0.0
consistency  = max_day_pnl / total_profit if total_profit > 0 else 0.0

# Victor's rolling-5-day check
day_pnl_series = [day_pnls.get(d, 0.0) for d in all_days]
rolling5 = [sum(day_pnl_series[i:i+5]) for i in range(len(day_pnl_series) - 4)]
worst5   = min(rolling5) if rolling5 else 0.0
best5    = max(rolling5) if rolling5 else 0.0
pct_neg5 = sum(1 for x in rolling5 if x < 0) / len(rolling5) if rolling5 else 0.0


# ── 6. Report ─────────────────────────────────────────────────────────────────
print(f"\n  N={N}  WR={wr:.1%}  PF={pf:.3f}  AvgP&L=${avg_pnl:.2f}  "
      f"Freq={freq:.2f}/d  Days={n_days}")
print()

# Exit breakdown
n_tp   = sum(1 for t in trades if t["reason"] == "TP")
n_stop = sum(1 for t in trades if t["reason"] == "STOP")
n_time = sum(1 for t in trades if t["reason"] in ("TIME", "CLOSE", "END"))
print(f"  Exit: TP={n_tp} ({n_tp/N:.0%})  STOP={n_stop} ({n_stop/N:.0%})  "
      f"TIME/CLOSE={n_time} ({n_time/N:.0%})")
print(f"  Daily halt triggered: {len(halted_dates)} days")

# Equity curve
eq_arr   = np.array([t["equity"] for t in trades])
eq_peak  = eq_arr.max()
eq_final = eq_arr[-1]
print(f"\n  Equity curve (1 MNQ contract):")
print(f"    Start: $0  →  Peak: ${eq_peak:.0f}  →  Final: ${eq_final:.0f}")
print(f"    Total gross profit: ${total_profit:.0f}  |  Total gross loss: ${gross_l:.0f}")
print(f"    Max trailing DD from HWM: ${min_trailing_dd:.0f}  "
      f"({'✅ ≤ $' + str(int(G1_DD_MAX_ABS)) if abs(min_trailing_dd) <= G1_DD_MAX_ABS else '❌ > $' + str(int(G1_DD_MAX_ABS))})")
print(f"    (Combine limit: ${COMB_DD_LIM:.0f}; Gate 1 criterion: ≤ ${G1_DD_MAX_ABS:.0f})")
combine_blown_dd = abs(min_trailing_dd) > COMB_DD_LIM
if combine_blown_dd:
    print(f"    ⚠️  Would have blown combine trailing DD at some point!")
else:
    print(f"    ✅  Never exceeds combine trailing DD limit of ${COMB_DD_LIM:.0f}")

# Qualifying days
print(f"\n  Qualifying days (session P&L ≥ ${QUAL_DAY_MIN:.0f}):")
print(f"    Total: {len(qual_days)} / {n_days} = {len(qual_days)/n_days:.0%} of trading days")
print(f"    Last 20 trading days: {len(qual_last_20)} / 20  "
      f"({'✅' if len(qual_last_20) >= G1_QUAL_WARN else '⚠️  WARNING < 6'})")

# Daily consistency
print(f"\n  Daily consistency (50% rule):")
print(f"    Max single-day P&L: ${max_day_pnl:.0f}")
print(f"    Total gross profit:  ${total_profit:.0f}")
print(f"    Concentration: {consistency:.1%}  "
      f"({'✅ ≤ 50%' if consistency <= G1_CONSIST_WARN else '⚠️  WARNING — may violate combine rule'})")

# Victor's rolling-5-day
print(f"\n  Victor's rolling-5-day variance check:")
print(f"    Worst 5-day stretch:  ${worst5:.0f}  "
      f"({'✅ ≥ $0' if worst5 >= 0 else '❌ negative'})")
print(f"    Best  5-day stretch:  ${best5:.0f}")
print(f"    % windows with loss:  {pct_neg5:.0%}")
print(f"    Worst single-day P&L: ${min(day_pnl_series):.0f}")

# Divergence distribution at entry
divs = np.array([t["div"] for t in trades])
print(f"\n  Divergence at entry (pts, all positive = MNQ outperformed ES):")
print(f"    p25={np.percentile(divs,25):.1f}  median={np.median(divs):.1f}  "
      f"p75={np.percentile(divs,75):.1f}  p90={np.percentile(divs,90):.1f}  "
      f"max={divs.max():.1f}")

# By-month table
print(f"\n  By-month breakdown:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'Total':>9}  "
      f"{'freq/d':>7}  {'Halts':>6}  {'WorstDD':>9}")
for m in sorted(month_stats):
    s       = month_stats[m]
    n_mo    = s["w"] + s["l"]
    mo_wr   = s["w"] / n_mo if n_mo else 0
    avg_m   = float(np.mean(s["pnls"]))
    tot_m   = float(np.sum(s["pnls"]))
    mo_bars = rth[rth.index.to_period("M") == m]
    mo_days = mo_bars.index.normalize().nunique()
    n_halt  = sum(1 for d in halted_dates if pd.Period(d, freq="M") == m)
    mo_trades = [t for t in trades if t["month"] == m]
    worst_dd_m = min(t["trailing_dd"] for t in mo_trades) if mo_trades else 0.0
    wr_flag = "✅" if mo_wr >= G1_WORSTMO_MIN else "❌"
    print(f"  {str(m):<10}  {n_mo:>5}  {mo_wr:>7.1%}{wr_flag}  ${avg_m:>7.2f}  "
          f"${tot_m:>7.0f}  {n_mo/max(1,mo_days):>5.2f}/d  {n_halt:>5}  ${worst_dd_m:>7.0f}")

# Trailing DD path
print(f"\n  Trailing DD path (by month — worst intra-month and end-of-month):")
for m in sorted(month_stats):
    mo_trades = [t for t in trades if t["month"] == m]
    if mo_trades:
        dd_worst = min(t["trailing_dd"] for t in mo_trades)
        dd_end   = mo_trades[-1]["trailing_dd"]
        eq_end   = mo_trades[-1]["equity"]
        print(f"    {str(m)}:  worst DD=${dd_worst:>7.0f}  "
              f"end DD=${dd_end:>7.0f}  equity=${eq_end:>7.0f}")

# Time-of-day WR
print(f"\n  Time-of-day WR:")
hour_data: dict = {}
hour_stops: dict = {}
for t in trades:
    h = t["hour"]
    hour_data.setdefault(h, [0, 0])
    hour_data[h][0 if t["win"] else 1] += 1
    hour_stops.setdefault(h, [])
    hour_stops[h].append(t["stop_usd"])
print(f"  {'Hour':>6}  {'N':>5}  {'WR':>7}  {'AvgStop':>9}  {'%trades':>8}")
for h in sorted(hour_data):
    w, l = hour_data[h]
    n_h  = w + l
    avg_s = float(np.mean(hour_stops.get(h, [0])))
    print(f"  {h:>5}ET  {n_h:>5}  {w/n_h:>7.1%}  ${avg_s:>7.0f}  {n_h/N:>7.1%}")


# ── 7. Gate 1 verdict ─────────────────────────────────────────────────────────
print()
print(SEP)
print("GATE 1 VERDICT")
print(SEP)

stop_est = THRESH * STOP_MULT * MNQ_PV

g_wr   = wr                    >= G1_WR_MIN
g_ev   = avg_pnl               >  G1_AVGPNL_MIN
g_pf   = pf                    >= G1_PF_MIN
g_dd   = abs(min_trailing_dd)  <= G1_DD_MAX_ABS
g_freq = freq                  >= G1_FREQ_MIN
g_n    = N                     >= G1_N_MIN
g_womo = worst_mo_wr           >= G1_WORSTMO_MIN


def vline(flag: bool, label: str, measured: str) -> str:
    status = "✅ PASS" if flag else "❌ FAIL"
    return f"  {status}  {label:<58} [{measured}]"


print(vline(g_wr,   f"Win rate ≥ {G1_WR_MIN:.0%}  (breakeven at 1:1, ${stop_est:.0f} stop)",
            f"{wr:.1%}"))
print(vline(g_ev,   "Avg net P&L/trade > $0  (positive EV required)",
            f"${avg_pnl:.2f}"))
print(vline(g_pf,   f"Profit factor ≥ {G1_PF_MIN:.2f}",
            f"{pf:.3f}"))
print(vline(g_dd,   f"Max trailing DD in-sample ≤ ${G1_DD_MAX_ABS:.0f}  "
            f"(combine limit ${COMB_DD_LIM:.0f})",
            f"${min_trailing_dd:.0f}"))
print(vline(g_freq, f"Frequency ≥ {G1_FREQ_MIN:.1f}/day",
            f"{freq:.2f}/day"))
print(vline(g_n,    f"N trades ≥ {G1_N_MIN}",
            f"{N}"))
print(vline(g_womo, f"Worst-month WR ≥ {G1_WORSTMO_MIN:.0%}",
            f"{worst_mo_wr:.1%}"))

# Warnings
w_qual  = len(qual_last_20) >= G1_QUAL_WARN
w_conc  = consistency       <= G1_CONSIST_WARN
print()
print(f"  {'✅ OK  ' if w_qual else '⚠️ WARN'}  "
      f"{'Qualifying sessions in last 20 ≥ 6  (WARNING only)':<58}"
      f"[{len(qual_last_20)}/20]")
print(f"  {'✅ OK  ' if w_conc else '⚠️ WARN'}  "
      f"{'Largest single day ≤ 50% of total P&L  (WARNING only)':<58}"
      f"[{consistency:.1%}]")

gate_pass = all([g_wr, g_ev, g_pf, g_dd, g_freq, g_n, g_womo])
print()

# Concentration warning — informational, not a gate
jan_feb = sum(len(month_stats[m]["pnls"])
              for m in month_stats
              if str(m) in ("2026-01", "2026-02"))
conc_pct = jan_feb / N if N else 0.0
if conc_pct > 0.40:
    print(f"  ⚠️  CONCENTRATION NOTE: {jan_feb}/{N} trades ({conc_pct:.0%}) fall in Jan-Feb 2026.")
    print(f"     Jan/Feb 2026 freq={sum(month_stats[m]['w']+month_stats[m]['l'] for m in month_stats if str(m) in ('2026-01','2026-02'))}/{n_days} days — higher vol / divergence regime.")
    print(f"     OOS result will clarify whether edge persists across vol regimes.")
    print()

if gate_pass:
    print("  ✅  GATE 1 PASS — all required criteria met.")
    print()
    print("  Next steps:")
    print("    1. Fetch ES OOS data (≥ 2026-03-01) from TradeStation.")
    print("    2. Run OOS backtest (Gate 2) against sealed holdout.")
    print(f"       OOS MNQ: {cfg['oos_path']}")
    print( "       OOS ES:  data/sealed_holdout/es_1min_holdout_20260301_plus.csv  "
           "(must fetch)")
    print("    3. Gate 2 criteria: OOS WR ≥ 53%, avg P&L > $0, PF ≥ 1.10, "
          "PF-retention ≥ 75%, N ≥ 20.")
else:
    print("  ❌  GATE 1 FAIL — OOS holdout access denied.")
    print()
    fails = [
        (f"WR {wr:.1%} < {G1_WR_MIN:.0%}",                    not g_wr),
        (f"Avg P&L ${avg_pnl:.2f} ≤ $0",                      not g_ev),
        (f"PF {pf:.3f} < {G1_PF_MIN:.2f}",                    not g_pf),
        (f"Max DD ${min_trailing_dd:.0f} worse than -${G1_DD_MAX_ABS:.0f}", not g_dd),
        (f"Freq {freq:.2f}/day < {G1_FREQ_MIN:.1f}",           not g_freq),
        (f"N={N} < {G1_N_MIN}",                                not g_n),
        (f"Worst-mo WR {worst_mo_wr:.1%} < {G1_WORSTMO_MIN:.0%}", not g_womo),
    ]
    for label, failed in fails:
        if failed:
            print(f"     • {label}")

print(SEP)


# ── 8. Save trades CSV and summary report ─────────────────────────────────────
ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
rep_dir  = ROOT / "data" / "reports"
csv_path = rep_dir / f"stat_arb_short_gate1_{ts_str}.csv"
txt_path = rep_dir / f"stat_arb_short_gate1_{ts_str}.txt"

rep_dir.mkdir(parents=True, exist_ok=True)

# Trades CSV
fields = ["date", "month", "hour", "entry", "exit", "div", "stop_usd",
          "tp", "stop", "pnl", "pnl_1c", "win", "reason", "equity", "trailing_dd"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for t in trades:
        w.writerow({k: t.get(k, "") for k in fields})

# Summary text
summary_lines = [
    f"Gate 1 Backtest: ES/MNQ Stat Arb Short-Only",
    f"Run: {datetime.now().isoformat()}",
    f"Pre-registration commit (seal): 44cc6192c7f18b284ec9e4d5ade81430cf9bc900",
    f"Backtest commit: (see git log)",
    f"",
    f"N={N}  WR={wr:.1%}  PF={pf:.3f}  AvgP&L=${avg_pnl:.2f}  "
    f"Freq={freq:.2f}/d  Days={n_days}",
    f"Max trailing DD: ${min_trailing_dd:.0f}  "
    f"(Gate 1 limit: ${G1_DD_MAX_ABS:.0f}, combine limit: ${COMB_DD_LIM:.0f})",
    f"Qual days: {len(qual_days)}/{n_days}  "
    f"Qual last-20: {len(qual_last_20)}/20",
    f"Consistency: {consistency:.1%}  Worst-month WR: {worst_mo_wr:.1%}",
    f"Victor worst-5d: ${worst5:.0f}  pct-neg: {pct_neg5:.0%}",
    f"Daily halts: {len(halted_dates)} days",
    f"",
    f"Gate 1: {'PASS' if gate_pass else 'FAIL'}",
]
txt_path.write_text("\n".join(summary_lines))

print(f"\n  Trades saved  → {csv_path.name}")
print(f"  Summary saved → {txt_path.name}")

#!/usr/bin/env python3
"""
backtest_stat_arb_short_oos.py — Gate 2 OOS Backtest: ES/MNQ Stat Arb Short-Only

Pre-registration : _bmad-output/preregistration_stat_arb_short_combine.md
Config           : stat_arb_short_config.yaml   (SHA-256 verified)
Gate 0 study     : study_stat_arb_short_only.py  (SHA-256 verified)

Data files (sealed holdout — access logged in data/sealed_holdout/ACCESS_LOG.md):
  MNQ OOS : data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv
  ES  OOS : data/sealed_holdout/es_1min_holdout_20260301_plus.csv

Gate 2 criteria (from pre-registration, one-shot, requires Gate 1 pass):
  OOS WR ≥ 53%
  OOS avg net P&L/trade > $0
  OOS PF ≥ 1.10
  OOS PF retention vs in-sample ≥ 75%  (Gate 1 PF=1.268 × 0.75 = 0.951 minimum)
  N OOS trades ≥ 20

Early stopping diagnostic (not a gate — informational):
  If PF < 1.05 after first 25 trades, would halt (live combine stopping rule)

In-sample reference (Gate 1):
  N=631  WR=57.8%  PF=1.268  AvgP&L=$6.45  freq=2.95/d  MaxDD=-$816
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

ROOT    = Path(__file__).parent
PREREG  = ROOT / "_bmad-output" / "preregistration_stat_arb_short_combine.md"
CONFIG  = ROOT / "stat_arb_short_config.yaml"
STUDY   = ROOT / "study_stat_arb_short_only.py"
MNQ_OOS = ROOT / "data" / "sealed_holdout" / "mnq_1min_holdout_20260301_plus.csv"
ES_OOS  = ROOT / "data" / "sealed_holdout" / "es_1min_holdout_20260301_plus.csv"

# Gate 2 thresholds (pre-registration, immutable after commit 2e9fb90)
G2_WR_MIN       = 0.53
G2_AVGPNL_MIN   = 0.0
G2_PF_MIN       = 1.10
G2_PF_RETENTION = 0.75
G2_N_MIN        = 20
# In-sample reference (Gate 1 results, commit 0ed5dff)
IS_PF           = 1.268
IS_WR           = 0.578
IS_AVGPNL       = 6.45
IS_N            = 631
IS_MAXDD        = -816

EARLY_STOP_PF   = 1.05   # live stopping rule diagnostic (after N=25)

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

    print(f"  Pre-reg sealed at commit: {reg_c}")
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
        print("\n  ❌ HASH MISMATCH — OOS backtest aborted.")
        print("     Parameters modified since pre-registration. New pre-reg required.")
        return False

    print("  ✅ Verified — running against pre-registered artifacts.")
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
CONTRACTS     = 1

print(SEP)
print("OOS CONFIG (FROZEN — pre-registration 2e9fb90)")
print(SEP)
print(f"  THRESH={THRESH}pt  STOP_MULT={STOP_MULT}×  BETA_WIN={BETA_WIN}  SPREAD_WIN={SPREAD_WIN}")
print(f"  HOLD_MAX={HOLD_MAX}bars  RTH_START={RTH_START}  SESSION_CLOSE={SESSION_CLOSE}")
print(f"  MNQ_PV=${MNQ_PV}  COMMISSION=${COMMISSION}  STOP_CAP=${STOP_CAP}")
print(f"  DAILY_HALT=${DAILY_HALT}  QUAL_DAY_MIN=${QUAL_DAY_MIN}")
print(f"  CONTRACTS={CONTRACTS}")
print(f"  OOS PERIOD: {cfg['oos_start']} → end of holdout file")
print()


# ── 3. Load and process OOS data ──────────────────────────────────────────────
def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


print("Loading OOS bars…")
mnq_raw = load_et(MNQ_OOS)
es_raw  = load_et(ES_OOS)

both = (mnq_raw[["close"]].rename(columns={"close": "mnq"})
        .join(es_raw[["close"]].rename(columns={"close": "es"}), how="inner"))
rth  = both.between_time(RTH_START, SESSION_CLOSE).copy()

print(f"  MNQ OOS bars: {len(mnq_raw):,}  ({mnq_raw.index[0].date()} → "
      f"{mnq_raw.index[-1].date()})")
print(f"  ES  OOS bars: {len(es_raw):,}   ({es_raw.index[0].date()} → "
      f"{es_raw.index[-1].date()})")
print(f"  Inner-join RTH bars: {len(rth):,}  "
      f"|  {rth.index.normalize().nunique()} days  "
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


# ── 4. OOS simulation (identical logic to Gate 1) ────────────────────────────
print(SEP)
print(f"GATE 2 OOS SIMULATION  (THRESH={THRESH}pt, STOP={STOP_MULT}×, SHORT ONLY)")
print(SEP)

trades: list     = []
active           = None
hold_count       = 0

equity           = 0.0
hwm              = 0.0
min_trailing_dd  = 0.0

session_date    = None
session_pnl     = 0.0
session_halted  = False
day_pnls: dict  = {}
halted_dates: set = set()

# Early-stop check state
early_stop_triggered = False
early_stop_at_n      = None

for k in range(len(rth)):
    ts    = ts_arr[k]
    mnq_k = mnq_arr[k]
    div_k = div_arr[k]
    today = ts.date()

    if today != session_date:
        if session_date is not None:
            day_pnls[session_date] = session_pnl
            if session_halted:
                halted_dates.add(session_date)
        session_date   = today
        session_pnl    = 0.0
        session_halted = False

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

            # Check early-stop diagnostic after N=25
            n_so_far = len(trades)
            if not early_stop_triggered and n_so_far == 25:
                pnls_so_far = np.array([t["pnl_1c"] for t in trades])
                gw = pnls_so_far[pnls_so_far > 0].sum()
                gl = abs(pnls_so_far[pnls_so_far < 0].sum())
                pf_25 = gw / max(1e-9, gl)
                if pf_25 < EARLY_STOP_PF:
                    early_stop_triggered = True
                    early_stop_at_n = n_so_far

        continue

    if session_halted:
        continue
    if div_k <= THRESH:
        continue

    div_abs     = div_k
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

if session_date is not None:
    day_pnls[session_date] = session_pnl
    if session_halted:
        halted_dates.add(session_date)

if active:
    exit_px = mnq_arr[-1]
    pnl_1c  = (exit_px - active["entry"]) * (-1) * MNQ_PV - COMMISSION
    net_pnl = pnl_1c * CONTRACTS
    equity     += net_pnl
    hwm             = max(hwm, equity)
    trailing_dd     = equity - hwm
    min_trailing_dd = min(min_trailing_dd, trailing_dd)
    trades.append({**active, "exit": exit_px, "pnl": net_pnl, "pnl_1c": pnl_1c,
                   "win": pnl_1c > 0, "reason": "END",
                   "equity": equity, "trailing_dd": trailing_dd})


# ── 5. Compute Gate 2 metrics ─────────────────────────────────────────────────
N    = len(trades)
if N == 0:
    print("  No trades generated — check OOS data alignment and config.")
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
pf_retention = pf / IS_PF

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

qual_days = [d for d, p in day_pnls.items() if p >= QUAL_DAY_MIN]
day_pnl_series = [day_pnls.get(d, 0.0) for d in all_days]
rolling5 = [sum(day_pnl_series[i:i+5]) for i in range(len(day_pnl_series) - 4)]
worst5 = min(rolling5) if rolling5 else 0.0


# ── 6. Print report ───────────────────────────────────────────────────────────
print(f"\n  N={N}  WR={wr:.1%}  PF={pf:.3f}  AvgP&L=${avg_pnl:.2f}  "
      f"Freq={freq:.2f}/d  Days={n_days}")
print(f"  PF retention vs in-sample ({IS_PF:.3f}): {pf_retention:.1%}")
print()

n_tp   = sum(1 for t in trades if t["reason"] == "TP")
n_stop = sum(1 for t in trades if t["reason"] == "STOP")
n_time = sum(1 for t in trades if t["reason"] in ("TIME", "CLOSE", "END"))
print(f"  Exit: TP={n_tp} ({n_tp/N:.0%})  STOP={n_stop} ({n_stop/N:.0%})  "
      f"TIME/CLOSE={n_time} ({n_time/N:.0%})")
print(f"  Daily halt triggered: {len(halted_dates)} days")

eq_arr  = np.array([t["equity"] for t in trades])
print(f"\n  Equity curve (1 MNQ contract):")
print(f"    Start: $0  →  Peak: ${eq_arr.max():.0f}  →  Final: ${eq_arr[-1]:.0f}")
print(f"    Max trailing DD from HWM: ${min_trailing_dd:.0f}")
comb_ok = abs(min_trailing_dd) <= COMB_DD_LIM
print(f"    (Combine limit ${COMB_DD_LIM:.0f}: {'✅ never blown' if comb_ok else '❌ would have blown'})")

print(f"\n  Qualifying days (≥${QUAL_DAY_MIN:.0f}): "
      f"{len(qual_days)} / {n_days} = {len(qual_days)/n_days:.0%}")
print(f"  Victor's worst 5-day stretch: ${worst5:.0f}")

# Early-stop diagnostic
print(f"\n  Early-stop diagnostic (live combine stopping rule: halt if PF < {EARLY_STOP_PF} after N=25):")
if N >= 25:
    p25 = pnls_1c[:25]
    pf25 = p25[p25 > 0].sum() / max(1e-9, abs(p25[p25 < 0].sum()))
    print(f"    PF at N=25: {pf25:.3f}  "
          f"({'✅ would NOT halt' if pf25 >= EARLY_STOP_PF else '⚠️  would HALT combine'})")
else:
    print(f"    N={N} < 25 — diagnostic not applicable")

# By-month
print(f"\n  By-month breakdown:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'AvgP&L':>9}  {'Total':>9}  "
      f"{'freq/d':>7}")
for m in sorted(month_stats):
    s     = month_stats[m]
    n_mo  = s["w"] + s["l"]
    mo_wr = s["w"] / n_mo if n_mo else 0
    avg_m = float(np.mean(s["pnls"]))
    tot_m = float(np.sum(s["pnls"]))
    mo_bars = rth[rth.index.to_period("M") == m]
    mo_days = mo_bars.index.normalize().nunique()
    flag = "✅" if mo_wr >= G2_WR_MIN else "❌"
    print(f"  {str(m):<10}  {n_mo:>5}  {mo_wr:>7.1%}{flag}  ${avg_m:>7.2f}  "
          f"${tot_m:>7.0f}  {n_mo/max(1,mo_days):>5.2f}/d")

# Divergence distribution
divs = np.array([t["div"] for t in trades])
print(f"\n  Divergence at entry (pts):")
print(f"    p25={np.percentile(divs,25):.1f}  median={np.median(divs):.1f}  "
      f"p75={np.percentile(divs,75):.1f}  p90={np.percentile(divs,90):.1f}  "
      f"max={divs.max():.1f}")

# Comparison table
print(f"\n  In-sample vs OOS comparison:")
print(f"  {'Metric':<22}  {'In-sample':>12}  {'OOS':>12}  {'Ratio':>8}")
print(f"  {'N':<22}  {IS_N:>12}  {N:>12}")
print(f"  {'WR':<22}  {IS_WR:>12.1%}  {wr:>12.1%}  {wr/IS_WR:>8.1%}")
print(f"  {'PF':<22}  {IS_PF:>12.3f}  {pf:>12.3f}  {pf_retention:>8.1%}")
print(f"  {'Avg P&L/trade':<22}  ${IS_AVGPNL:>11.2f}  ${avg_pnl:>11.2f}  "
      f"{avg_pnl/IS_AVGPNL if IS_AVGPNL else 0:>8.1%}")
print(f"  {'Max trailing DD':<22}  ${IS_MAXDD:>11.0f}  ${min_trailing_dd:>11.0f}")
print(f"  {'Freq (/d)':<22}  {'2.95':>12}  {freq:>12.2f}")


# ── 7. Gate 2 verdict ─────────────────────────────────────────────────────────
print()
print(SEP)
print("GATE 2 VERDICT (one-shot OOS)")
print(SEP)

g_wr        = wr             >= G2_WR_MIN
g_ev        = avg_pnl        >  G2_AVGPNL_MIN
g_pf        = pf             >= G2_PF_MIN
g_retention = pf_retention   >= G2_PF_RETENTION
g_n         = N              >= G2_N_MIN


def vline(flag: bool, label: str, measured: str) -> str:
    return f"  {'✅ PASS' if flag else '❌ FAIL'}  {label:<58} [{measured}]"


print(vline(g_wr,        f"OOS WR ≥ {G2_WR_MIN:.0%}",
            f"{wr:.1%}"))
print(vline(g_ev,        "OOS avg net P&L/trade > $0",
            f"${avg_pnl:.2f}"))
print(vline(g_pf,        f"OOS PF ≥ {G2_PF_MIN:.2f}",
            f"{pf:.3f}"))
print(vline(g_retention, f"OOS PF retention ≥ {G2_PF_RETENTION:.0%}  "
            f"(in-sample PF={IS_PF:.3f} × 0.75 = {IS_PF*G2_PF_RETENTION:.3f})",
            f"{pf_retention:.1%}"))
print(vline(g_n,         f"N OOS trades ≥ {G2_N_MIN}",
            f"{N}"))

gate_pass = all([g_wr, g_ev, g_pf, g_retention, g_n])
print()
if gate_pass:
    print("  ✅  GATE 2 PASS — OOS edge confirmed.")
    print()
    print("  Next steps:")
    print("    • Pre-register a ProjectX/Topstep combine account setup.")
    print("    • Confirm position sizing (1 MNQ contract) viable for combine economics.")
    print("    • Apply live OOS stopping rule: halt if PF < 1.05 after first 25 trades.")
    print("    • S25 (tier2_streaming_working.py on account 23884932) is unaffected.")
else:
    print("  ❌  GATE 2 FAIL — OOS does not confirm in-sample edge.")
    fails = [
        (f"OOS WR {wr:.1%} < {G2_WR_MIN:.0%}",                      not g_wr),
        (f"OOS avg P&L ${avg_pnl:.2f} ≤ $0",                        not g_ev),
        (f"OOS PF {pf:.3f} < {G2_PF_MIN:.2f}",                      not g_pf),
        (f"OOS PF retention {pf_retention:.1%} < {G2_PF_RETENTION:.0%}", not g_retention),
        (f"N={N} < {G2_N_MIN}",                                      not g_n),
    ]
    for label, failed in fails:
        if failed:
            print(f"     • {label}")
    print()
    print("  Strategy does not qualify for combine deployment.")
    print("  Record result, update memory, and return to strategy search.")

print(SEP)


# ── 8. Save trades CSV and summary ────────────────────────────────────────────
ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
rep_dir  = ROOT / "data" / "reports"
csv_path = rep_dir / f"stat_arb_short_gate2_oos_{ts_str}.csv"
txt_path = rep_dir / f"stat_arb_short_gate2_oos_{ts_str}.txt"

rep_dir.mkdir(parents=True, exist_ok=True)

fields = ["date", "month", "hour", "entry", "exit", "div", "stop_usd",
          "tp", "stop", "pnl", "pnl_1c", "win", "reason", "equity", "trailing_dd"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for t in trades:
        w.writerow({k: t.get(k, "") for k in fields})

summary_lines = [
    "Gate 2 OOS Backtest: ES/MNQ Stat Arb Short-Only",
    f"Run: {datetime.now().isoformat()}",
    f"Pre-registration commit: 2e9fb90  (seal: 44cc6192)",
    f"OOS period: {rth.index[0].date()} → {rth.index[-1].date()}  ({n_days} days)",
    "",
    f"N={N}  WR={wr:.1%}  PF={pf:.3f}  AvgP&L=${avg_pnl:.2f}  Freq={freq:.2f}/d",
    f"PF retention: {pf_retention:.1%} (in-sample PF={IS_PF:.3f})",
    f"Max trailing DD: ${min_trailing_dd:.0f}",
    f"Qual days: {len(qual_days)}/{n_days}  Worst-mo WR: {worst_mo_wr:.1%}",
    f"Victor worst-5d: ${worst5:.0f}",
    f"Daily halts: {len(halted_dates)} days",
    "",
    f"Gate 2: {'PASS' if gate_pass else 'FAIL'}",
]
txt_path.write_text("\n".join(summary_lines))

print(f"\n  Trades saved  → {csv_path.name}")
print(f"  Summary saved → {txt_path.name}")

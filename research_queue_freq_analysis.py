"""
Research Queue Frequency Analysis — S27/S28/S29/S30
Applies the same frequency lens used on S26/S26v2 to each queued hypothesis.
Key question: at current S25 signal rate, does the hypothesis hit a sample-starvation wall?

S26v2 lesson: any filter reducing Window B to <20 trades/year is untestable in 180 days.
General rule: if a hypothesis reduces N below ~25 trades/year, it'll need 10+ months to evaluate.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Load S25 baseline trades ───────────────────────────────────────────────────
df = pd.read_csv("/root/Silver-Bullet-ML-BMAD/data/reports/backtest_1year_20260526_004254.csv")
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
df['entry_et'] = df['entry_time'].dt.tz_convert('US/Eastern')
df['entry_date'] = df['entry_et'].dt.date
df['entry_hour'] = df['entry_et'].dt.hour
df['entry_minute'] = df['entry_et'].dt.minute
df['pnl'] = pd.to_numeric(df['pnl'])

N_TOTAL = len(df)
DAYS = 365
RATE = N_TOTAL / DAYS

def pf(pnls):
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    return gp / gl if gl > 0 else float('inf')

print("=" * 72)
print("RESEARCH QUEUE FREQUENCY ANALYSIS")
print(f"  S25 baseline: N={N_TOTAL} trades over {DAYS} days ({RATE:.3f}/day)")
print(f"  Decision gate reference: N≥20 in ≤365 days")
print(f"  S26v2 lesson: structural <9 trades/year → untestable")
print("=" * 72)

# ── S27-revised: Hold-Period Exit Management ───────────────────────────────────
print("\n─── S27-revised: Hold-Period Exit Management ─────────────────────────")
time_stops = df[df['exit_type'] == 'time']
non_time   = df[df['exit_type'] != 'time']
time_wins  = time_stops[time_stops['pnl'] > 0]
time_loss  = time_stops[time_stops['pnl'] <= 0]

print(f"  Total trades:          N={N_TOTAL}")
print(f"  Time-stop exits:       N={len(time_stops)} ({len(time_stops)/N_TOTAL*100:.0f}%)")
print(f"    └─ profitable:       N={len(time_wins)} (mean ${time_wins['pnl'].mean():+.0f})")
print(f"    └─ losing:           N={len(time_loss)} (mean ${time_loss['pnl'].mean():+.0f})")
print(f"  TP exits:              N={(df['exit_type']=='tp').sum()}")
print(f"  SL exits:              N={(df['exit_type']=='sl').sum()}")
print()
print(f"  FREQUENCY VERDICT: ✅ NOT A FILTER")
print(f"  All N={N_TOTAL} trades eligible — changes exit logic, not entry criteria.")
print(f"  No sample-scarcity risk. Full baseline N applies.")
print(f"  Pre-reg needed for: breakeven stop levels, trailing ATR multiplier, partial TP %")

# ── S28: News Calendar Filter ──────────────────────────────────────────────────
print("\n─── S28: News Calendar Filter ────────────────────────────────────────")

# Major US high-impact events 2025-2026 (8:30am ET or 2:00pm ET)
# FOMC rate decisions (2pm ET, day-of)
fomc_dates = [
    date(2025,1,29), date(2025,3,19), date(2025,5,7),  date(2025,6,18),
    date(2025,7,30), date(2025,9,17), date(2025,10,29), date(2025,12,10),
    date(2026,1,28), date(2026,3,18), date(2026,4,29),
]
# NFP — first Friday of each month, 8:30am ET
nfp_dates = [
    date(2025,5,2),  date(2025,6,6),  date(2025,7,3 ), date(2025,8,1),
    date(2025,9,5),  date(2025,10,3), date(2025,11,7), date(2025,12,5),
    date(2026,1,9),  date(2026,2,6),  date(2026,3,6),  date(2026,4,3),
    date(2026,5,1),
]
# CPI — typically 2nd or 3rd Wednesday, 8:30am ET (approximate)
cpi_dates = [
    date(2025,5,13), date(2025,6,11), date(2025,7,15), date(2025,8,12),
    date(2025,9,10), date(2025,10,15),date(2025,11,12),date(2025,12,10),
    date(2026,1,15), date(2026,2,12), date(2026,3,12), date(2026,4,10),
    date(2026,5,13),
]

all_events = {}
for d in fomc_dates:
    all_events.setdefault(d, []).append('FOMC')
for d in nfp_dates:
    all_events.setdefault(d, []).append('NFP')
for d in cpi_dates:
    all_events.setdefault(d, []).append('CPI')

# Count trades that enter within ±30 min of a news event
WINDOW_MIN = 30
blocked_narrow = []  # ±30 min window
blocked_day    = []  # same calendar day (broader blackout option)

for _, row in df.iterrows():
    d = row['entry_date']
    h = row['entry_hour']
    m = row['entry_minute']
    entry_min = h * 60 + m  # minutes since midnight ET

    if d not in all_events:
        continue

    blocked_day.append(row.name)
    for evt in all_events[d]:
        if evt in ('NFP', 'CPI'):
            event_min = 8 * 60 + 30   # 8:30am ET
        else:  # FOMC
            event_min = 14 * 60        # 2:00pm ET
        if abs(entry_min - event_min) <= WINDOW_MIN:
            blocked_narrow.append(row.name)
            break

n_blocked_narrow = len(set(blocked_narrow))
n_blocked_day    = len(set(blocked_day))
n_survive_narrow = N_TOTAL - n_blocked_narrow
n_survive_day    = N_TOTAL - n_blocked_day
rate_survive_narrow = n_survive_narrow / DAYS
rate_survive_day    = n_survive_day / DAYS

print(f"  News events catalogued: {len(fomc_dates)} FOMC, {len(nfp_dates)} NFP, {len(cpi_dates)} CPI")
print()
print(f"  Option A — ±30min blackout around event time:")
print(f"    Trades blocked:   N={n_blocked_narrow} ({n_blocked_narrow/N_TOTAL*100:.0f}%)")
print(f"    Trades surviving: N={n_survive_narrow} ({n_survive_narrow/N_TOTAL*100:.0f}%) → {rate_survive_narrow:.3f}/day")
print(f"    Days to N=20:     ~{round(20/rate_survive_narrow)} days")
surviving = df[~df.index.isin(blocked_narrow)]
print(f"    Surviving PF:     {pf(surviving['pnl'].tolist()):.4f} (baseline: {pf(df['pnl'].tolist()):.4f})")

print()
print(f"  Option B — full-day blackout:")
print(f"    Trades blocked:   N={n_blocked_day} ({n_blocked_day/N_TOTAL*100:.0f}%)")
print(f"    Trades surviving: N={n_survive_day} ({n_survive_day/N_TOTAL*100:.0f}%) → {rate_survive_day:.3f}/day")
print(f"    Days to N=20:     ~{round(20/rate_survive_day)} days")
surviving_day = df[~df.index.isin(blocked_day)]
print(f"    Surviving PF:     {pf(surviving_day['pnl'].tolist()):.4f}")

# Show which events hit
if blocked_narrow:
    print(f"\n  Blocked trades (±30min):")
    for idx in set(blocked_narrow):
        row = df.loc[idx]
        d = row['entry_date']
        evts = all_events.get(d, [])
        print(f"    {row['entry_et'].strftime('%Y-%m-%d %H:%M ET')} | {row['exit_type']:4s} | ${row['pnl']:+.0f} | {'+'.join(evts)}")

freq_ok = rate_survive_narrow >= (20/365)
print(f"\n  FREQUENCY VERDICT: {'✅ VIABLE' if freq_ok else '❌ RISKY'}")
print(f"  ±30min filter: {n_survive_narrow} trades/year → N=20 in ~{round(20/rate_survive_narrow)} days")
print(f"  Note: calendar accuracy ±1 week for CPI dates — real analysis needs ForexFactory CSV")

# ── S29: ES/MNQ Divergence ─────────────────────────────────────────────────────
print("\n─── S29: ES/MNQ Divergence Confirmation ──────────────────────────────")
print(f"  Baseline: N={N_TOTAL} trades/year")
print()
print(f"  ES data not available — estimating from divergence rate assumptions:")
for diverge_pct in [10, 25, 40, 60]:
    n_blocked = round(N_TOTAL * diverge_pct / 100)
    n_survive = N_TOTAL - n_blocked
    rate = n_survive / DAYS
    days_n20 = round(20 / rate) if rate > 0 else 9999
    ok = days_n20 <= 365
    flag = '✅' if ok else '⚠️ '
    print(f"  {flag} If {diverge_pct:2d}% of MNQ sweeps lack ES confirmation: "
          f"N={n_survive}/yr → N=20 in ~{days_n20} days")
print()
print(f"  FREQUENCY VERDICT: ⚠️  UNKNOWN — depends on ES divergence rate")
print(f"  Viable as long as >50% of MNQ setups have ES confirmation (conservative).")
print(f"  Requires sourcing ES 1-min CSV before frequency can be measured.")
print(f"  Recommend: acquire ES data, run divergence rate check BEFORE pre-registering.")

# ── S30: Bullish CHoCH ─────────────────────────────────────────────────────────
print("\n─── S30: Bullish CHoCH (Symmetric Direction) ─────────────────────────")
print(f"  Baseline (bearish): N={N_TOTAL} trades/year ({RATE:.3f}/day)")
print(f"  Bidir run (bearish_only=False): N=8 LONG trades/year")
print(f"    └─ Due to missing symmetric bullish CHoCH — not a valid signal count")
print()
print(f"  If bullish CHoCH is symmetric with bearish:")
print(f"    Optimistic: ~62 new LONG trades/year → total ~124/yr ({124/365:.3f}/day)")
print(f"    Conservative: ~30 new LONG trades/year → total ~92/yr ({92/365:.3f}/day)")
print(f"    N=20 LONG-only in ~{round(20/(30/365))} days (conservative) → ✅ VIABLE")
print()
print(f"  FREQUENCY VERDICT: ✅ ADDITIVE — increases total N, no scarcity risk")
print(f"  Key dependency: architecture work required (implement bullish CHoCH + H1 sweep)")
print(f"  Must pre-register BEFORE backtest. First backtest reveals true bullish rate.")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY")
print(f"  S25 baseline: {N_TOTAL} trades/year, {RATE:.3f}/day, N=20 in ~{round(20/RATE)} days")
print()
print(f"  S27-revised  ✅ No filter — all {N_TOTAL} trades eligible, no scarcity risk")
print(f"  S28          ✅ Viable — ±30min news blackout blocks ~{n_blocked_narrow} trades, "
      f"N=20 in ~{round(20/rate_survive_narrow)} days")
print(f"  S29          ⚠️  Unknown — need ES data; viable if <50% divergence rate")
print(f"  S30          ✅ Additive — bullish signals add to N, no scarcity risk")
print()
print(f"  None of these repeat the S26v2 failure pattern.")
print(f"  S29 is the only one that could hit a wall — acquire ES data first.")
print("=" * 72)

"""
study_mim_noise_bands_gate2_mc.py
---------------------------------
GATE 2 — Topstep 50K combine Monte Carlo for prereg mim-noise-bands-mnq (50f111b).
Both variants passed Gate 0 (dcccbf0) and Gate 1 OOS.

Corrected combine rules (per seal §5 / research doc a0427e7):
- Start $50,000; target: balance >= $53,000 AND best day < 50% of total profit
- MLL: floor starts $48,000; ratchets at END OF DAY to min($50,000, max(floor, EOD-2000));
  breach checked per-trade against current equity -> BLOW
- DLL: if day P&L <= -$1,000 after a trade, the day deactivates (no more trades that day)
- Costs: trades are NET of $2.24/ct (1.12 pts) per contract; P&L scales linearly with size
- 5,000 sims, ET-day block bootstrap (sample traded days with replacement, intra-day order kept)
- 90-day cap -> STALL if neither pass nor blow
Gate: pass% >= 50% at some integer size 1..10 AND pass% > blow% at that size.
"""
import pandas as pd
import numpy as np

COST_PTS = 1.12
PT_VAL = 2.0
N_SIM, MAX_DAYS = 5000, 90
rng = np.random.default_rng(42)

def run_mc(trades, contracts):
    tdf = trades.copy()
    tdf['net_usd_1ct'] = (tdf['pnl_pts'] - COST_PTS) * PT_VAL
    by_day = [g['net_usd_1ct'].values for _, g in tdf.groupby('day', sort=True)]
    nd = len(by_day)
    pass_n = blow_n = 0
    days_to_pass = []
    for _ in range(N_SIM):
        bal = 50_000.0
        floor = 48_000.0
        best_day = 0.0
        outcome = None
        sampled = rng.integers(0, nd, size=MAX_DAYS)
        for dn, di in enumerate(sampled):
            day_pnl = 0.0
            for pnl1 in by_day[di]:
                pnl = pnl1 * contracts
                bal += pnl
                day_pnl += pnl
                if bal <= floor:
                    outcome = 'blow'
                    break
                if day_pnl <= -1000.0:
                    break  # DLL: day deactivated, account lives
            if outcome:
                break
            best_day = max(best_day, day_pnl)
            profit = bal - 50_000.0
            if profit >= 3000.0 and best_day < 0.5 * profit:
                outcome = 'pass'
                days_to_pass.append(dn + 1)
                break
            floor = min(50_000.0, max(floor, bal - 2000.0))  # EOD ratchet
        if outcome == 'pass':
            pass_n += 1
        elif outcome == 'blow':
            blow_n += 1
    med = int(np.median(days_to_pass)) if days_to_pass else -1
    return pass_n / N_SIM, blow_n / N_SIM, med

print("=" * 72)
print("GATE 2 — TOPSTEP 50K COMBINE MONTE CARLO (corrected rules)")
print("pooled dev 2025 + OOS 2026 trades, net of $2.24/ct, 5000 sims, 90-day cap")
print("=" * 72)

for variant in ('v1', 'v2'):
    dev = pd.read_csv(f"data/reports/mim_nb_gate0_{variant}_2025.csv")
    oos = pd.read_csv(f"data/reports/mim_nb_gate1_{variant}_2026oos.csv")
    pooled = pd.concat([dev, oos], ignore_index=True)
    print(f"\n--- {variant.upper()}  (pooled N={len(pooled)}, "
          f"{pooled['day'].nunique()} traded days, "
          f"worst trade {pooled['pnl_pts'].min():+.0f} pts) ---")
    print(f"{'Size':>5} | {'Pass%':>6} | {'Blow%':>6} | {'Stall%':>6} | MedDaysToPass")
    best = (0, 0.0, 0.0, -1)
    for k in range(1, 11):
        p, b, med = run_mc(pooled, k)
        flag = " <== gate met" if (p >= 0.50 and p > b) else ""
        print(f"{k:>4}ct | {p*100:>5.1f}% | {b*100:>5.1f}% | {(1-p-b)*100:>5.1f}% | {med:>6}{flag}")
        if p > best[1]:
            best = (k, p, b, med)
    k, p, b, med = best
    verdict = "GATE 2 PASS" if (p >= 0.50 and p > b) else "GATE 2 FAIL"
    print(f"  Best size: {k}ct  pass={p*100:.1f}%  blow={b*100:.1f}%  median {med} days")
    print(f"  ==> {verdict}")

print("\nDone.", flush=True)

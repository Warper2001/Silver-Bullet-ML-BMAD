"""
study_mim_noise_bands_gate1_oos.py
----------------------------------
GATE 1 — ONE-SHOT OOS (2026 YTD) for prereg mim-noise-bands-mnq (50f111b).
Both variants passed Gate 0 (commit dcccbf0), so both are evaluated, once.

Spec identical to study_mim_noise_bands_gate0.py (imports its run/load).
Gates: N >= 40, net PF >= 1.05, net expectancy > $0.
"""
import importlib.util
spec = importlib.util.spec_from_file_location(
    "g0", "/root/Silver-Bullet-ML-BMAD/study_mim_noise_bands_gate0.py")

# Import only the functions (the module body runs the dev test on import;
# avoid that by reading and exec'ing just the definitions).
import pandas as pd
import numpy as np

src = open("/root/Silver-Bullet-ML-BMAD/study_mim_noise_bands_gate0.py").read()
defs_only = src.split('print("Loading dev 2025...', 1)[0]
ns = {}
exec(defs_only, ns)
load, run, COST_PTS = ns['load'], ns['run'], ns['COST_PTS']

BASE = "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute"

def gate1(name, tdf, min_n=40):
    print(f"\n{'='*68}\n{name}\n{'='*68}")
    n = len(tdf)
    if n == 0:
        print("  ZERO trades — INCONCLUSIVE")
        return
    gross = tdf['pnl_pts'].values
    net = gross - COST_PTS
    gpf = gross[gross > 0].sum()/abs(gross[gross < 0].sum()) if (gross < 0).any() else float('inf')
    npf = net[net > 0].sum()/abs(net[net < 0].sum()) if (net < 0).any() else float('inf')
    exp_usd = net.mean()*2.0
    print(f"  N={n} over {tdf['day'].nunique()} traded days")
    print(f"  WR gross={(gross>0).mean()*100:.1f}%  gross PF={gpf:.3f}  NET PF={npf:.3f}")
    print(f"  NET expectancy = ${exp_usd:+.2f}/contract/trade")
    print(f"  worst trade = {gross.min():+.2f} pts  best = {gross.max():+.2f} pts")
    for r, g in tdf.groupby('reason'):
        print(f"    {r:<9} N={len(g):>4}  avg={g['pnl_pts'].mean():>+8.2f} pts")
    checks = {f"N >= {min_n}": n >= min_n,
              "net PF >= 1.05": npf >= 1.05,
              "net expectancy > 0": exp_usd > 0}
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if n < min_n:
        print("  ==> INCONCLUSIVE")
    else:
        print(f"  ==> {'GATE 1 PASS' if all(checks.values()) else 'GATE 1 FAIL'}")

print("Loading OOS 2026 YTD (ONE SHOT)...", flush=True)
oos = load(f"{BASE}/mnq_1min_2026_ytd.csv")
for variant in ('V1', 'V2'):
    t = run(oos, variant)
    t.to_csv(f"data/reports/mim_nb_gate1_{variant.lower()}_2026oos.csv", index=False)
    gate1(f"{variant} — OOS 2026-01-01 → 2026-05-19", t)
print("\nDone.", flush=True)

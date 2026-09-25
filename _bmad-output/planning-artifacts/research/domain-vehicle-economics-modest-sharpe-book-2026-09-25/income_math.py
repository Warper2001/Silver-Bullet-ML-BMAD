"""Local income math for a modest-Sharpe book (not a research claim; pure arithmetic + simulation).
Excess return = SR x vol x capital (before tax, after costs embedded in SR). Losing-year prob = Phi(-SR).
Median worst drawdown over 10 years: simulated monthly iid normal paths (seed fixed)."""
import json, math
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(20260925)
rows = []
for sr in (0.3, 0.5, 0.7, 1.0):
    for vol in (0.08, 0.10, 0.12):
        mu_m, sd_m = sr * vol / 12, vol / math.sqrt(12)
        paths = rng.normal(mu_m, sd_m, size=(20000, 120))
        eq = np.cumprod(1 + paths, axis=1)
        dd = (1 - eq / np.maximum.accumulate(eq, axis=1)).max(axis=1)
        rows.append({"SR": sr, "vol": vol, "excess_ret": round(sr * vol, 4),
                     "usd_yr_25k": round(sr * vol * 25_000), "usd_yr_50k": round(sr * vol * 50_000),
                     "p_losing_year": round(float(norm.cdf(-sr)), 3),
                     "median_maxdd_10y": round(float(np.median(dd)), 3), "p90_maxdd_10y": round(float(np.percentile(dd, 90)), 3),
                     "capital_for_240k_yr": round(240_000 / (sr * vol))})
json.dump(rows, open(__file__.replace(".py", ".json"), "w"), indent=1)
for r in rows:
    if r["vol"] == 0.10: print(r)

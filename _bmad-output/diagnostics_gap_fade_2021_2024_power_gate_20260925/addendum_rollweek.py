"""ADDENDUM to power_gate.py (outcome-blind; run after it). Found after the gate ran:
data/mim_x/mnq_1min_2021_2024_frontmonth.csv switches contract at expiry (every switch is on expiry Friday),
so the ~5 sessions before each expiry sit on the expiring contract after volume has migrated. The by_contract
file has no next-contract bars before the switch, so front-month bars can't be rebuilt. Proposal: the primary
excludes setups within 8 calendar days before each quarterly expiry. This recomputes power at that N with the
same rule, inputs and seed. RTH sessions were verified single-contract; the switch happens outside RTH."""
import importlib.util, json, sys
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pg", HERE / "power_gate.py"); pg = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(pg)
EXPIRIES = pd.to_datetime(["2021-03-19", "2021-06-18", "2021-09-17", "2021-12-17", "2022-03-18", "2022-06-17",
                           "2022-09-16", "2022-12-16", "2023-03-17", "2023-06-16", "2023-09-15", "2023-12-15",
                           "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20"])  # = the file's contract-switch dates
u = pd.read_csv(HERE / "unseen_setups_outcome_blind.csv", parse_dates=["date"])
u["roll_week"] = u["date"].apply(lambda d: any(e - pd.Timedelta(days=8) <= d < e for e in EXPIRIES))
k = u[~u["roll_week"]]
rng = np.random.default_rng(pg.SEED)
dev = pd.read_csv(pg.DEV)["pnl_usd"].to_numpy(float)
res0 = json.loads((HERE / "results.json").read_text())
cost = res0["inputs"]["cost_per_trade_usd"]
net = dev - cost; mu, sd = net.mean(), net.std(ddof=1); z = (net - mu) / sd
n, ni, no = len(k), int(k["inside_prior_range"].sum()), int((~k["inside_prior_range"]).sum())
live_d = res0["primary"]["cost_measured"]["anchors"]["live_ledger"]["d"]
anch = {"ceiling_dev": mu / sd, "0.75x": .75 * mu / sd, "0.5x": .5 * mu / sd, "0.25x": .25 * mu / sd, "live_ledger": live_d}
out = {"n_excl_roll_week": n, "n_roll_week_excluded": int(u["roll_week"].sum()), "n_inside": ni, "n_outside": no,
       "primary": {a: {"d": round(d, 4), "power_deff1": pg.one_sample_power(z, d, n, rng),
                       "power_deff1.5": pg.one_sample_power(z, d, int(n / 1.5), rng)} for a, d in anch.items()},
       "secondary": {f"{m}x_ceiling": pg.welch_power(z, m * mu / sd, ni, no, rng) for m in (1.0, 1.5, 2.0)},
       "roll_week_dates": [str(d.date()) for d in u.loc[u["roll_week"], "date"]]}
out["verdict"] = {"primary": "POWERED" if out["primary"]["ceiling_dev"]["power_deff1"] >= pg.POWER_BAR else "UNDERPOWERED",
                  "secondary_V_IN": "POWERED" if out["secondary"]["1.0x_ceiling"] >= pg.POWER_BAR else "UNDERPOWERED"}
(HERE / "addendum_rollweek.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k2: v for k2, v in out.items() if k2 != "roll_week_dates"}, indent=1))

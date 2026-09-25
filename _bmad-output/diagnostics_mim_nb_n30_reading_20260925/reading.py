"""MIM-NB N=30 reading, per precommitment_mim_nb_n30_decision_20260920.md. Read-only.

Touches no config, bot, unit, or sealed holdout. Reads data/mim_nb/trades.csv (D1 counter),
data/mim_nb/projectx_fills.json (observed broker fees), data/trades.db (D3 monitor window),
data/combine_joint/floor_state.json (equity).

Run: .venv/bin/python _bmad-output/diagnostics_mim_nb_n30_reading_20260925/reading.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
OUT = Path(__file__).resolve().parent
PF_HALT = 0.70
COMBINE_START = "2026-08-13T16:54:13.915478+00:00"


def pf_of(x: np.ndarray) -> float:
    gl = -x[x < 0].sum()
    return float(x[x > 0].sum() / gl) if gl else float("inf")


def split(x: np.ndarray) -> dict:
    return {"n": int(len(x)), "net": round(float(x.sum()), 2), "pf": round(pf_of(x), 3)}


csv = pd.read_csv(ROOT / "data/mim_nb/trades.csv")
csv["pnl"] = csv["pnl_usd"].astype(str).str.replace("+", "", regex=False).astype(float)

# Observed broker cost per contract round trip (ProjectX fills: fees + commissions per side,
# normalised by fill size -- the file holds 1- and 2-lot fills at exactly proportional cost).
fills = json.load(open(ROOT / "data/mim_nb/projectx_fills.json"))
per_side = sorted({round((float(f["fees"]) + float(f["commissions"])) / int(f["size"]), 4)
                   for f in fills if not f.get("voided")})
assert len(per_side) == 1, per_side
rt_cost = 2 * per_side[0]
csv["pnl_fee"] = csv["pnl"] - rt_cost

gross, fee = csv["pnl"].to_numpy(), csv["pnl_fee"].to_numpy()
path = [
    {"n": n, "day": csv["day"].iloc[n - 1], "pf_gross": round(pf_of(gross[:n]), 3),
     "pf_fee": round(pf_of(fee[:n]), 3), "net_gross": round(float(gross[:n].sum()), 2)}
    for n in range(28, len(csv) + 1)
]
fired = [p for p in path if p["n"] >= 30 and (p["pf_gross"] < PF_HALT or p["pf_fee"] < PF_HALT)]

run = (csv["day"] >= "2026-07-29") & (csv["day"] <= "2026-08-04")
autoroll = csv["day"] == "2026-09-15"
n30 = csv.iloc[:30]
splits = {
    "D1_at_30_gross": split(gross[:30]),
    "D1_at_30_fee_adjusted": split(fee[:30]),
    "D1_all_gross": split(gross),
    "D1_all_fee_adjusted": split(fee),
    "excluding_0729_0804_run_at30": split(n30.loc[~run[:30], "pnl"].to_numpy()),
    "the_0729_0804_run_alone": split(csv.loc[run, "pnl"].to_numpy()),
    "excluding_0915_autoroll_at30": split(n30.loc[~autoroll[:30], "pnl"].to_numpy()),
    # Same boundary as the 09-20 diagnostic: combine start 2026-08-13 16:54Z = 12:54 ET. The 08-13
    # 10:00 EXTERNAL_CLOSE (+$7.50) belongs to the retired account 23884932.
    "current_acct_26556101_mim_only_all": split(csv.loc[
        (csv["day"] > "2026-08-13") | ((csv["day"] == "2026-08-13") & (csv["entry_t"] >= "12:54")),
        "pnl"].to_numpy()),
    "live_250pt_config_since_0625_all": split(csv.loc[csv["day"] >= "2026-06-25", "pnl"].to_numpy()),
    "trades_29_to_last": split(gross[28:]),
}

con = sqlite3.connect(ROOT / "data/trades.db")
d3 = np.array([r[0] for r in con.execute(
    "SELECT pnl FROM trades WHERE trader_id IN ('trader-mim-nb','trader-yank') "
    "AND timestamp >= ? AND pnl IS NOT NULL", (COMBINE_START,)).fetchall()], dtype=float)
floor = json.load(open(ROOT / "data/combine_joint/floor_state.json"))

res = {
    "n_rows": len(csv), "trade30_day": csv["day"].iloc[29],
    "rt_cost_observed_usd": rt_cost, "fee_source_fills": len(fills),
    "pf_path": path, "d1_fired_any_n_ge_30": fired,
    "splits": splits,
    "D3_monitor_window": split(d3),
    "equity": {k: floor[k] for k in ("equity", "floor", "hwm", "ts_utc")},
    "equity_minus_48400": round(floor["equity"] - 48_400, 2),
    "buffer_to_floor": round(floor["equity"] - floor["floor"], 2),
    "last_rows": csv.tail(4)[["day", "dir", "reason", "pnl"]].to_dict("records"),
}
(OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))

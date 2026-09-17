"""Is MIM-NB's live drawdown (-$1,540.50 / 26 trades) inside its own modeled variance,
or does it approach the sealed halt trigger?

Reference: `_bmad-output/preregistration_mim_nb_honest_expectations.md` (sealed 2026-06-14,
no code/param change) — the current live bot is unchanged since that seal.
  - Sec 2a: OOS-2026 forward expectancy net PF 1.299, +$31.99/ct/trade.
  - Sec 2b: edge is 1-3 fat-tail days out of ~163; ~160 days near breakeven.
  - Sec 5 (decision rule, pre-registered): "If live MIM-NB reaches N >= 20-30 completed
    trades with net PF tracking the OOS 1.30, the central-case expectation upgrades from
    hypothesis to evidence. If net PF < 0.70 over 30 trades, the deployment halt trigger
    fires and this whole expectation is void."

This script does not touch config or the live bot. It reads:
  - data/trades.db, trader_id='trader-mim-nb', write_mode='realtime' (live ledger)
  - data/reports/mim_nb_gate1_v2_2026oos.csv (the OOS reference trade list cited in the
    seal's own gate1 v2 study; gross pnl_pts, converted to USD at $2/pt MNQ)

Known caveat (memory: project_mim_nb_roll_contamination_20260915): the 2026-09-15 trade
(-$355) is a documented LIVE BUG — AUTOROLL took `open_d` from the wrong contract during
the Z26 roll, forcing a spurious entry. It is not organic strategy performance. Reported
both with and without it.

Run: .venv/bin/python check_variance.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

TRADES_DB = Path("/root/Silver-Bullet-ML-BMAD/data/trades.db")
OOS_REF = Path("/root/Silver-Bullet-ML-BMAD/data/reports/mim_nb_gate1_v2_2026oos.csv")
CONTAMINATED_DATE = "2026-09-15"
B = 200_000
SEED = 20260917


def load_live() -> pd.DataFrame:
    con = sqlite3.connect(TRADES_DB)
    df = pd.read_sql(
        "SELECT * FROM trades WHERE trader_id='trader-mim-nb' AND write_mode='realtime'", con
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_reference() -> np.ndarray:
    df = pd.read_csv(OOS_REF)
    return (df["pnl_pts"] * 2.0).to_numpy()  # MNQ = $2/pt, gross (pre-commission)


def stats(pnl: pd.Series) -> dict:
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return {
        "n": int(len(pnl)),
        "net": float(pnl.sum()),
        "mean": float(pnl.mean()),
        "sd": float(pnl.std()),
        "pf": float(gw / gl) if gl else float("inf"),
        "gross_win": float(gw),
        "gross_loss": float(gl),
    }


def bootstrap_tail(ref: np.ndarray, n: int, target_net: float, target_pf: float) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.choice(ref, size=(B, n), replace=True)
    net = draws.sum(axis=1)
    gw = np.where(draws > 0, draws, 0).sum(axis=1)
    gl = np.where(draws < 0, -draws, 0).sum(axis=1)
    pf = np.divide(gw, gl, out=np.full(B, np.inf), where=gl > 0)
    return {
        "n": n,
        "p_net_le_observed": float((net <= target_net).mean()),
        "p_pf_le_observed": float((pf <= target_pf).mean()),
        "sim_net_mean": float(net.mean()),
        "sim_net_sd": float(net.std()),
        "sim_net_pct_5_25_50": [float(x) for x in np.percentile(net, [5, 25, 50])],
    }


def main() -> int:
    live = load_live()
    ref = load_reference()
    ref_stats = stats(pd.Series(ref))

    with_contam = stats(live["pnl"])
    clean = live[live["timestamp"].dt.date.astype(str) != CONTAMINATED_DATE]
    without_contam = stats(clean["pnl"])

    boot_with = bootstrap_tail(ref, with_contam["n"], with_contam["net"], with_contam["pf"])
    boot_without = bootstrap_tail(ref, without_contam["n"], without_contam["net"], without_contam["pf"])

    out = {
        "oos_reference_gross": ref_stats,
        "live_with_contamination": with_contam,
        "live_excl_20260915_contamination": without_contam,
        "bootstrap_vs_with_contamination": boot_with,
        "bootstrap_vs_excl_contamination": boot_without,
        "sealed_halt_trigger": "PF < 0.70 at N>=30 -> halt (preregistration_mim_nb_honest_expectations.md sec 5)",
        "sealed_upgrade_trigger": "N 20-30 with PF tracking ~1.30 -> hypothesis to evidence (same sec 5)",
    }
    print(json.dumps(out, indent=2))
    Path("results.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

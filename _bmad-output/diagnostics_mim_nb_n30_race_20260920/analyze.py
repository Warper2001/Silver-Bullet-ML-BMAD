"""MIM-NB: can the sealed N=30 halt rule fire, and does the capital outlast it?

Forks _bmad-output/diagnostics_mim_nb_live_variance_20260917/check_variance.py and
corrects its ledger window. Read-only: touches no config, no bot, no sealed holdout.

WHAT THIS CORRECTS (see REPORT.md):
  The 09-17 script filtered trades.db to write_mode='realtime', which drops the first two
  live trades (06-11 is tagged 'backfilled'; 06-12 is absent). The authoritative,
  hash-chained ledger is data/mim_nb/trades.csv: N=28, net -$753.00, PF 0.848 --
  not N=26 / -$1,540.50 / PF 0.689.

THREE DIFFERENT "N=30" DEFINITIONS EXIST (all reported):
  D1  sealed deployment prereg §4: "30 completed trades with net PF < 0.70" -- the strategy's own
      trades since deployment (trades.csv, N=28 now).
  D2  09-17 diagnostic's window (trades.db realtime, N=26) -- a mis-windowed D1; kept only to
      reproduce the 09-17 published figures as a sanity check.
  D3  combine_floor_monitor.py PF trigger: MIM-NB + YANK rows in trades.db since the CURRENT
      account's start (2026-08-13T16:54), REPORT-ONLY. N=12 now.

REFERENCE DISTRIBUTIONS (none is a benchmark for the live config -- the 250-pt prereg says a
fresh benchmark was never built):
  R1  data/reports/mim_nb_gate1_v2_2026oos.csv x $2/pt. 500-pt cat-stop variant, gross. Sealed model.
  R2  R1 with every loss truncated at -$500. Approximates the 250-pt stop on the LOSS side only;
      it cannot convert winners that would have been stopped out, so it is OPTIMISTIC.
  R3  live trades since the 250-pt config went live (2026-06-25, commit b7ef63a): N=25 of trades.csv.

Run: .venv/bin/python analyze.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
OUT = Path(__file__).resolve().parent
TRADES_CSV = ROOT / "data/mim_nb/trades.csv"
TRADES_DB = ROOT / "data/trades.db"
OOS_REF = ROOT / "data/reports/mim_nb_gate1_v2_2026oos.csv"

B = 200_000
SEED = 20260920
PF_HALT = 0.70
EQUITY_NOW = 48_917.42          # data/combine_joint/floor_state.json, 2026-09-20T21:09Z
FLOOR = 48_298.96               # Topstep trailing MLL floor (hwm 50,298.96 - 2,000)
HALT_REVIEW_EQUITY = 48_400.0   # sealed deployment prereg §4, first halt-and-review trigger
CAT_STOP_LIVE_START = "2026-06-25"
COMBINE_START = "2026-08-13T16:54:13.915478+00:00"
ASOF = pd.Timestamp("2026-09-20", tz="UTC")


def pf_of(x: np.ndarray) -> float:
    gl = -x[x < 0].sum()
    return float(x[x > 0].sum() / gl) if gl else float("inf")


def load_csv_ledger() -> pd.DataFrame:
    df = pd.read_csv(TRADES_CSV)
    df["pnl"] = df["pnl_usd"].astype(str).str.replace("+", "", regex=False).astype(float)
    return df


def load_db_realtime() -> pd.DataFrame:
    con = sqlite3.connect(TRADES_DB)
    df = pd.read_sql(
        "SELECT * FROM trades WHERE trader_id='trader-mim-nb' AND write_mode='realtime'", con
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_combined_monitor_window() -> np.ndarray:
    """D3 exactly as combine_floor_monitor.combined_pf_and_count() queries it."""
    con = sqlite3.connect(TRADES_DB)
    rows = con.execute(
        "SELECT pnl FROM trades WHERE trader_id IN ('trader-mim-nb','trader-yank') "
        "AND timestamp >= ? AND pnl IS NOT NULL",
        (COMBINE_START,),
    ).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def scenario_dists(csv: pd.DataFrame) -> dict[str, np.ndarray]:
    r1 = (pd.read_csv(OOS_REF)["pnl_pts"] * 2.0).to_numpy()
    r2 = np.where(r1 < 0, np.maximum(r1, -500.0), r1)
    r3 = csv.loc[csv["day"] >= CAT_STOP_LIVE_START, "pnl"].to_numpy()
    return {"R1_sealed_500pt_gross": r1, "R2_truncated_-500_optimistic": r2, "R3_live_250pt_config": r3}


def reproduce_0917(ref: np.ndarray) -> dict:
    """Sanity check: reproduce the 09-17 published bootstrap on its own (mis-windowed) N=26."""
    db = load_db_realtime()
    net, pf = float(db["pnl"].sum()), pf_of(db["pnl"].to_numpy())
    rng = np.random.default_rng(20260917)  # the 09-17 seed
    d = rng.choice(ref, size=(B, len(db)), replace=True)
    n = d.sum(axis=1)
    gw = np.where(d > 0, d, 0).sum(axis=1)
    gl = np.where(d < 0, -d, 0).sum(axis=1)
    p = np.divide(gw, gl, out=np.full(B, np.inf), where=gl > 0)
    return {"n": len(db), "net": net, "pf": pf,
            "p_net_le": float((n <= net).mean()), "p_pf_le": float((p <= pf).mean()),
            "published_0917": {"p_net_le": 0.035, "p_pf_le": 0.055}}


def tail_vs_reference(ref: np.ndarray, n: int, net: float, pf: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    d = rng.choice(ref, size=(B, n), replace=True)
    s = d.sum(axis=1)
    gw = np.where(d > 0, d, 0).sum(axis=1)
    gl = np.where(d < 0, -d, 0).sum(axis=1)
    p = np.divide(gw, gl, out=np.full(B, np.inf), where=gl > 0)
    return {"p_net_le_observed": float((s <= net).mean()), "p_pf_le_observed": float((p <= pf).mean())}


def forward_halt(gw0: float, gl0: float, n0: int, dist: np.ndarray, n_at: int, n_max: int, seed: int) -> dict:
    """P(PF<0.70) exactly at N=n_at, and at ANY N in [n_at, n_max] (monitor semantics), from a live prefix."""
    k = n_max - n0
    rng = np.random.default_rng(seed)
    d = rng.choice(dist, size=(B, k), replace=True)
    cw = gw0 + np.cumsum(np.where(d > 0, d, 0), axis=1)
    cl = gl0 + np.cumsum(np.where(d < 0, -d, 0), axis=1)
    pf = np.divide(cw, cl, out=np.full_like(cw, np.inf), where=cl > 0)
    at = pf[:, n_at - n0 - 1] < PF_HALT
    anyt = (pf[:, n_at - n0 - 1:] < PF_HALT).any(axis=1)
    return {"p_halt_at_n": float(at.mean()), "p_halt_any_n_to_max": float(anyt.mean())}


def rule_power_fresh(ref: np.ndarray, mean_target: float, n: int, seed: int) -> float:
    """P(PF<0.70 over a FRESH n-trade sample) when the true per-trade mean is mean_target (net)."""
    rng = np.random.default_rng(seed)
    d = rng.choice(ref, size=(B, n), replace=True) - (ref.mean() - mean_target)
    gw = np.where(d > 0, d, 0).sum(axis=1)
    gl = np.where(d < 0, -d, 0).sum(axis=1)
    p = np.divide(gw, gl, out=np.full(B, np.inf), where=gl > 0)
    return float((p < PF_HALT).mean())


def capital_race(dist: np.ndarray, ks=(2, 5, 10), seed=0) -> dict:
    rng = np.random.default_rng(seed)
    d = rng.choice(dist, size=(B, max(ks)), replace=True)
    eq = EQUITY_NOW + np.cumsum(d, axis=1)
    out = {}
    for k in ks:
        m = eq[:, :k].min(axis=1)
        out[f"k={k}"] = {
            "p_touch_48400_halt_review": float((m <= HALT_REVIEW_EQUITY).mean()),
            "p_touch_mll_floor": float((m <= FLOOR).mean()),
        }
    return out


def n30_arrival(csv: pd.DataFrame, need: int) -> dict:
    """Trailing-56d trade rate -> Gamma waiting time for `need` more trades."""
    days = pd.to_datetime(csv["day"], utc=True)
    r = float((days > ASOF - pd.Timedelta(days=56)).sum() / 56.0)
    rng = np.random.default_rng(1)
    w = rng.gamma(need, 1.0 / r, size=B)
    q = np.percentile(w, [5, 50, 95])
    f = lambda x: str((ASOF + pd.Timedelta(days=float(x))).date())
    return {"rate_per_day_trailing56": r, "rate_per_week": r * 7, "need": need,
            "date_p5_p50_p95": [f(v) for v in q]}


def main() -> int:
    csv = load_csv_ledger()
    dists = scenario_dists(csv)
    live = csv["pnl"].to_numpy()
    gw0, gl0 = float(live[live > 0].sum()), float(-live[live < 0].sum())
    d3 = load_combined_monitor_window()
    gw3, gl3 = float(d3[d3 > 0].sum()), float(-d3[d3 < 0].sum())

    out: dict = {
        "asof": "2026-09-20",
        "ledgers": {
            "D1_trades_csv_authoritative": {"n": len(live), "net": float(live.sum()), "pf": pf_of(live),
                                            "gross_win": gw0, "gross_loss": gl0},
            "D1_on_250pt_config_only": {"n": len(dists["R3_live_250pt_config"]),
                                        "net": float(dists["R3_live_250pt_config"].sum()),
                                        "pf": pf_of(dists["R3_live_250pt_config"])},
            "D3_monitor_combined_since_20260813": {"n": len(d3), "net": float(d3.sum()), "pf": pf_of(d3),
                                                   "gross_win": gw3, "gross_loss": gl3},
        },
        "reproduction_of_0917": reproduce_0917(dists["R1_sealed_500pt_gross"]),
        "halt_rule_arithmetic_D1": {
            "loss_needed_for_pf_below_0.70_at_n30": gw0 / PF_HALT - gl0,
            "two_max_cat_stops_pf": gw0 / (gl0 + 1000.0),
        },
        "scenarios": {},
    }
    for name, ref in dists.items():
        s = {
            "n_ref": len(ref), "mean": float(ref.mean()), "min": float(ref.min()), "pf": pf_of(ref),
            "tail_of_live_D1_vs_ref": tail_vs_reference(ref, len(live), float(live.sum()), pf_of(live), 11),
            "D1_forward_halt": forward_halt(gw0, gl0, len(live), ref, 30, 60, 21),
            "D3_forward_halt": forward_halt(gw3, gl3, len(d3), ref, 30, 60, 31),
            "capital_race": capital_race(ref, seed=41),
        }
        out["scenarios"][name] = s

    ref1 = dists["R1_sealed_500pt_gross"]
    out["rule_power_fresh_n30_R1_shifted"] = {
        "true_mean_+31.99_sealed_net_edge": {"p_halt": rule_power_fresh(ref1, 31.99, 30, 51)},
        "true_mean_0_zero_edge": {"p_halt": rule_power_fresh(ref1, 0.0, 30, 52)},
        "true_mean_-31.99": {"p_halt": rule_power_fresh(ref1, -31.99, 30, 53)},
        "true_mean_-51.0": {"p_halt": rule_power_fresh(ref1, -51.0, 30, 54)},
    }
    # Disclosed fragility/splits of D1 (descriptive only -- nothing here is a threshold).
    def _split(mask) -> dict:
        x = csv.loc[mask, "pnl"].to_numpy()
        return {"n": int(len(x)), "net": float(x.sum()), "pf": pf_of(x)}

    run = (csv["day"] >= "2026-07-29") & (csv["day"] <= "2026-08-04")
    db = load_db_realtime()
    cur = db[db["timestamp"] >= pd.Timestamp(COMBINE_START)]
    out["D1_splits_descriptive"] = {
        "all_28": _split(csv["day"] >= "0"),
        "excluding_0729_0804_run": _split(~run),
        "the_0729_0804_run_alone": _split(run),
        "pre_reset_acct_23884932_rows_day_lt_0813": _split(csv["day"] < "2026-08-13"),
        "current_acct_26556101_mim_only_since_0813_1654": {
            "n": int(len(cur)), "net": float(cur["pnl"].sum()), "pf": pf_of(cur["pnl"].to_numpy())},
    }
    out["n30_arrival_D1"] = n30_arrival(csv, 30 - len(live))
    out["n30_arrival_D3"] = n30_arrival(csv, 30 - len(d3))
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

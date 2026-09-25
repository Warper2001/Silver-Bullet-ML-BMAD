"""POWER GATE — one-shot confirmatory test of the sealed GAP-1 rules on MNQ front-month 2021-2024.

OUTCOME-BLIND. No GAP-1 trade is simulated on the target window: no target, stop, time-stop or P&L is
evaluated. The target is read only for prices that DEFINE a setup (the first RTH open, the prior session's
same-contract RTH close, high and low). Effect and dispersion come from the corrected Gate-0 dev trades
and the live ledger. Both are inputs, not results on the target.
Target: data/mim_x/mnq_1min_2021_2024_frontmonth.csv plus mnq_1min_by_contract.csv for same-contract prior
sessions. NOT data/sealed_holdout/.

ENGINE CONVENTIONS (mirrored from the sealed engine, not chosen here). Bars are close-stamped in both the
target and the dev files. RTH = gap_fade_live._is_rth, i.e. stamps 09:30-15:59 ET, so:
  - "RTH open" = the open of the bar stamped 09:30 (09:29-09:30)
  - "prior close" = the close of the bar stamped 15:59
This is the sealed and live definition, disclosed rather than changed. Other rules: prior session must have
>= MIN_RTH_BARS RTH bars; Fridays excluded; |gap| / prior close >= GAP_MIN_PCT; prior close taken from the
session's own contract (as in the 2026-09-16 Gate-0 rescore).

WINDOW HYGIENE. Unseen = 2021-2024 sessions, minus 2023 Sep-Nov and 2024 Sep-Nov, which GAP-V
(preregistration_gap_velocity_conditioned.md) already ran GAP-1 trades on.

INPUTS
  dev   : _bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/corrected_gate0_trades.csv (N=115, gross 1ct $)
  live  : data/trades.db trader-gap-fade write_mode='realtime' (per-contract $; all rows 1ct as of this gate)
  cost  : $1.22 per round trip in fees (ProjectX per-contract rate) plus the mean realized-minus-modeled
          slippage in data/gap_fade/fills.csv (TS SIM, measured). Sensitivity: $10.
  shape : the dev net trade distribution, standardised, so the tails are kept rather than assuming normality.

PRE-COMMITTED DECISION RULE (written before any number below was produced; mirrors the 2026-09-21 MIM-NB gate):
  PRIMARY  (baseline): one-sided one-sample t-test, mean net $/trade > 0 on the unseen window, alpha 0.05.
    Power is simulated with 20,000 draws, trades iid from the dev shape shifted to the anchor d, DEFF 1.0.
    DEFF 1.5 (regime clustering) is reported as a sensitivity.
    N = the exact count of unseen qualifying sessions (one trade per qualifying session by construction).
    POWERED       if power >= 0.80 at the CEILING d (dev, in-sample, upward-biased)
    UNDERPOWERED  otherwise
    Power at 0.75x, 0.5x and 0.25x the ceiling, and at the live anchor, is reported for reading, not for the verdict.
  SECONDARY (one variant, fixed-sequence: tested only if PRIMARY passes, so family-wise alpha stays 0.05):
    V_IN = take the trade only when the RTH open lies inside the prior session's same-contract RTH [low, high].
    Hypothesis: outside-range ("breakaway") gaps revert less, so excluding them raises mean $/trade.
    Test: one-sided Welch t-test, mean(inside) - mean(outside) > 0, alpha 0.05.
    n_in and n_out are counted outcome-blind on the target.
    Power is simulated at best-case contrasts Delta = {1.0, 1.5, 2.0} x the ceiling mean, from the shared dev shape.
    Delta = 1.0 means outside gaps have zero edge; 2.0 means they lose as much as inside gaps earn.
    POWERED       if power >= 0.80 at Delta = 1.0 x ceiling
    UNDERPOWERED  otherwise (a best case is used, so an UNDERPOWERED verdict is conclusive)
  (0.80 power and alpha 0.05 are the repo's convention in earlier gates.)

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_2021_2024_power_gate_20260925/power_gate.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/Silver-Bullet-ML-BMAD")
OUT = Path(__file__).resolve().parent
TARGET = REPO / "data/mim_x/mnq_1min_2021_2024_frontmonth.csv"
BY_CONTRACT = REPO / "data/mim_x/mnq_1min_by_contract.csv"
DEV = REPO / "_bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/corrected_gate0_trades.csv"
FILLS = REPO / "data/gap_fade/fills.csv"
TRADES_DB = REPO / "data/trades.db"
assert "sealed_holdout" not in str(TARGET) + str(BY_CONTRACT)

spec = importlib.util.spec_from_file_location("gfl", REPO / "src/research/gap_fade_live.py")
gfl = importlib.util.module_from_spec(spec)
sys.modules["gfl"] = gfl
spec.loader.exec_module(gfl)

B = 20_000
SEED = 20260925
ALPHA = 0.05
POWER_BAR = 0.80
FEES_RT = 1.22
COST_SENS = 10.0
SEEN = {(2023, m) for m in (9, 10, 11)} | {(2024, m) for m in (9, 10, 11)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rth_sessions(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(gfl.ET)
    df = df.assign(ts=ts)
    h, mi = ts.dt.hour, ts.dt.minute
    rth = (((h == 9) & (mi >= 30)) | (h > 9)) & (h < 16)       # vectorised gap_fade_live._is_rth
    sample = df["ts"].sample(min(5000, len(df)), random_state=0)
    assert all(gfl._is_rth(t) == bool(rth[i]) for i, t in sample.items()), "RTH mask != _is_rth"
    df = df[rth]
    df = df.assign(date=df["ts"].dt.date)
    g = df.sort_values("ts").groupby(keys + ["date"])
    return pd.DataFrame({"open": g["open"].first(), "close": g["close"].last(),
                         "high": g["high"].max(), "low": g["low"].min(), "n": g["close"].count(),
                         "first_ts": g["ts"].first()})


def count_target() -> dict:
    front = rth_sessions(pd.read_csv(TARGET), []).reset_index()
    byc = pd.read_csv(BY_CONTRACT, usecols=["contract", "timestamp", "open", "high", "low", "close"])
    bc = rth_sessions(byc, ["contract"]).reset_index()
    # each front session's contract = the contract whose first RTH bar matches it (timestamp and open)
    m = front.merge(bc[["contract", "date", "first_ts", "open"]], on=["date", "first_ts", "open"], how="left")
    amb = m.groupby("date")["contract"].nunique()
    m = m.drop_duplicates("date")
    contract_of = dict(zip(m["date"], m["contract"]))
    bc_idx = bc.set_index(["contract", "date"])
    dates = sorted(front["date"])
    fr = front.set_index("date")
    rows, skipped = [], []
    for prev, day in zip(dates, dates[1:]):
        if fr.loc[prev, "n"] < gfl.MIN_RTH_BARS or day.weekday() in gfl.EXCLUDE_DOW:
            continue
        c = contract_of.get(day)
        if pd.isna(c) or (c, prev) not in bc_idx.index:
            skipped.append({"date": str(day), "contract": None if pd.isna(c) else c})
            continue
        p = bc_idx.loc[(c, prev)]
        ro = fr.loc[day, "open"]
        gap = ro - p["close"]
        if abs(gap) / p["close"] < gfl.GAP_MIN_PCT:
            continue
        rows.append({"date": str(day), "year": day.year, "month": day.month, "contract": c,
                     "side": "short" if gap > 0 else "long",
                     "gap_pct": round(100 * abs(gap) / p["close"], 3),
                     "inside_prior_range": bool(p["low"] <= ro <= p["high"]),
                     "roll_boundary": contract_of.get(prev) != c})
    q = pd.DataFrame(rows)
    q["seen_by_gapv"] = [(y, mo) in SEEN for y, mo in zip(q["year"], q["month"])]
    u = q[~q["seen_by_gapv"]]
    return {"sessions_front": len(dates), "ambiguous_contract_days": int((amb > 1).sum()),
            "skipped_no_same_contract_prior": skipped, "qualifying_all": len(q),
            "qualifying_seen_by_gapv": int(q["seen_by_gapv"].sum()), "N_unseen": len(u),
            "unseen_inside": int(u["inside_prior_range"].sum()),
            "unseen_outside": int((~u["inside_prior_range"]).sum()),
            "unseen_roll_boundary_sessions": int(u["roll_boundary"].sum()),
            "by_year_side": u.groupby(["year", "side"]).size().unstack(fill_value=0).to_dict(),
            "inside_share_by_year": u.groupby("year")["inside_prior_range"].mean().round(3).to_dict(),
            "_frame": u}


def one_sample_power(z: np.ndarray, d: float, n: int, rng: np.random.Generator) -> float:
    x = rng.choice(z, size=(B, n), replace=True) + d          # standardised shape, mean shifted to d
    t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n))
    return float((t > stats.t.ppf(1 - ALPHA, n - 1)).mean())


def welch_power(z: np.ndarray, delta: float, n1: int, n2: int, rng: np.random.Generator) -> float:
    a = rng.choice(z, size=(B, n1), replace=True) + delta
    b = rng.choice(z, size=(B, n2), replace=True)
    va, vb = a.var(axis=1, ddof=1) / n1, b.var(axis=1, ddof=1) / n2
    t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (n1 - 1) + vb ** 2 / (n2 - 1))
    return float((t > stats.t.ppf(1 - ALPHA, df)).mean())


def main() -> int:
    rng = np.random.default_rng(SEED)
    fills = pd.read_csv(FILLS)
    delta = pd.to_numeric(fills["delta_usd"], errors="coerce").dropna()
    cost = round(FEES_RT - float(delta.mean()), 2)             # delta < 0 = realized worse than modeled

    dev = pd.read_csv(DEV)["pnl_usd"].to_numpy(float)
    con = sqlite3.connect(TRADES_DB)
    live = pd.read_sql("SELECT pnl, metadata FROM trades WHERE trader_id='trader-gap-fade' "
                       "AND write_mode='realtime'", con)
    ct = [json.loads(m).get("contracts", 1) if m else 1 for m in live["metadata"]]
    live_pc = live["pnl"].to_numpy(float) / np.array(ct, float)

    tgt = count_target()
    frame = tgt.pop("_frame")
    frame.to_csv(OUT / "unseen_setups_outcome_blind.csv", index=False)
    n = tgt["N_unseen"]

    res = {"inputs": {"target_sha256": sha(TARGET), "by_contract_sha256": sha(BY_CONTRACT),
                      "dev_sha256": sha(DEV), "script_sha256": sha(Path(__file__)),
                      "cost_per_trade_usd": cost, "cost_basis": {"fees_rt": FEES_RT,
                      "mean_slippage_delta_usd": round(float(delta.mean()), 2), "n_fills": len(delta)},
                      "dev_n": len(dev), "live_n": len(live_pc)},
           "target_counts": tgt, "primary": {}, "secondary": {}}

    for label, c in (("cost_measured", cost), ("cost_10", COST_SENS)):
        net = dev - c
        mu, sd = float(net.mean()), float(net.std(ddof=1))
        z = (net - mu) / sd
        live_d = float((live_pc - c).mean() / sd)
        anchors = {"ceiling_dev": mu / sd, "0.75x": 0.75 * mu / sd, "0.5x": 0.5 * mu / sd,
                   "0.25x": 0.25 * mu / sd, "live_ledger": live_d}
        pw = {k: {"d": round(d, 4), "power_deff1": one_sample_power(z, d, n, rng),
                  "power_deff1.5": one_sample_power(z, d, int(n / 1.5), rng)} for k, d in anchors.items()}
        res["primary"][label] = {"dev_net_mean": round(mu, 2), "dev_net_sd": round(sd, 2), "anchors": pw}
        if label == "cost_measured":
            ni, no = tgt["unseen_inside"], tgt["unseen_outside"]
            res["secondary"] = {"n_inside": ni, "n_outside": no, "contrasts": {
                f"{k}x_ceiling": {"delta_d": round(k * mu / sd, 4),
                                  "power": welch_power(z, k * mu / sd, ni, no, rng)} for k in (1.0, 1.5, 2.0)}}

    p_ceiling = res["primary"]["cost_measured"]["anchors"]["ceiling_dev"]["power_deff1"]
    p_var = res["secondary"]["contrasts"]["1.0x_ceiling"]["power"]
    res["verdict"] = {"primary": "POWERED" if p_ceiling >= POWER_BAR else "UNDERPOWERED",
                      "secondary_V_IN": "POWERED" if p_var >= POWER_BAR else "UNDERPOWERED"}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""backtest_dix_gex_swing.py — DIX x GEX swing-horizon interaction test (SPX).

Runs EXACTLY the pre-registered spec sealed in
_bmad-output/preregistration_dix_gex_swing_interaction.md (commit a55166c).

Hypothesis (one-tailed): the forward-10d SPX return payoff to DIX accumulation
rises as dealer gamma becomes more negative (fragile) -> interaction beta3 > 0 in

    R_10d = b0 + b1*DIX_z + b2*FRAG + b3*(DIX_z * FRAG) + e,   FRAG = -GEX_z

Frozen parameters (NOT tuned on data): horizon=10, standardization=trailing-252
(min 60), entry=close of D+1, SPX-native returns, cell=DIX>80th & GEX<20th
(trailing-252, descriptive only), derive 2011-2019 / sealed holdout 2020-2026,
block bootstrap block=10. Decision rule per prereg section 3.

Run from repo root:  .venv/bin/python backtest_dix_gex_swing.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---- frozen constants (from the sealed prereg) ----
DATA = Path(__file__).parent / "_bmad-output" / "dix_gex_seal_data_20260619.csv"
SEAL_SHA = "46994107673eb8f47b8ea09f2e073d07fbbfcee801491682403b1c9ab454ba2a"
HORIZON = 10            # trading days
STD_WIN, STD_MIN = 252, 60
CELL_DIX_PCT, CELL_GEX_PCT = 0.80, 0.20
SPLIT = "2019-12-31"   # in-sample <= this signal date; holdout after
B = 10_000             # bootstrap resamples
BLOCK = HORIZON        # moving-block length
SEED = 42              # reproducibility only (affects MC noise, not estimates)

DIX_CORNER = norm.ppf(CELL_DIX_PCT)        # ~+0.8416
FRAG_CORNER = -norm.ppf(CELL_GEX_PCT)      # GEX@20th -> z=-0.8416 -> FRAG=+0.8416

# decision-rule thresholds (frozen)
P_BAR = 0.05           # in-sample one-tailed
ECON_BAR = 0.008       # +0.8% corner excess over 10d


def trailing_z(s: pd.Series) -> pd.Series:
    """Causal trailing-252 z-score (min 60 obs), using only data <= t."""
    m = s.rolling(STD_WIN, min_periods=STD_MIN).mean()
    sd = s.rolling(STD_WIN, min_periods=STD_MIN).std(ddof=0)
    return (s - m) / sd


def trailing_pct_flag_high(s: pd.Series, q: float) -> pd.Series:
    """True where s[t] is at/above its trailing-252 q-quantile (causal)."""
    thr = s.rolling(STD_WIN, min_periods=STD_MIN).quantile(q)
    return s >= thr


def trailing_pct_flag_low(s: pd.Series, q: float) -> pd.Series:
    """True where s[t] is at/below its trailing-252 q-quantile (causal)."""
    thr = s.rolling(STD_WIN, min_periods=STD_MIN).quantile(q)
    return s <= thr


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def block_bootstrap_beta3(X: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Moving-block bootstrap distribution of the interaction coef (index 3)."""
    n = len(y)
    n_blocks = int(np.ceil(n / BLOCK))
    max_start = n - BLOCK
    out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        out[b] = ols_beta(X[idx], y[idx])[3]
    return out


def design(df: pd.DataFrame):
    dix_z = df["DIX_z"].to_numpy()
    frag = df["FRAG"].to_numpy()
    X = np.column_stack([np.ones(len(df)), dix_z, frag, dix_z * frag])
    return X, df["R10"].to_numpy()


def fit_report(df: pd.DataFrame, label: str, rng):
    X, y = design(df)
    beta = ols_beta(X, y)
    boot = block_bootstrap_beta3(X, y, rng)
    p_one_tailed = float(np.mean(boot <= 0.0))      # H0: beta3 <= 0
    lo95_1s = float(np.percentile(boot, 5))         # one-sided 95% lower bound
    lo025 = float(np.percentile(boot, 2.5))
    hi975 = float(np.percentile(boot, 97.5))
    corner_excess = (beta[1] * DIX_CORNER + beta[2] * FRAG_CORNER
                     + beta[3] * DIX_CORNER * FRAG_CORNER)
    n_blocks = len(df) // HORIZON
    print(f"\n=== {label} ===")
    print(f"  n daily obs = {len(df)}   ~independent 10d blocks = {n_blocks}")
    print(f"  b0={beta[0]:+.5f}  b1(DIX)={beta[1]:+.5f}  b2(FRAG)={beta[2]:+.5f}  "
          f"b3(DIX*FRAG)={beta[3]:+.5f}")
    print(f"  beta3 one-tailed bootstrap p (P[beta3<=0]) = {p_one_tailed:.4f}")
    print(f"  beta3 one-sided 95% lower bound = {lo95_1s:+.5f}   "
          f"(2.5/97.5 CI: {lo025:+.5f} .. {hi975:+.5f})")
    print(f"  corner (DIX{int(CELL_DIX_PCT*100)}th & GEX{int(CELL_GEX_PCT*100)}th) "
          f"excess 10d return = {corner_excess:+.4%}")
    return dict(beta3=beta[3], p=p_one_tailed, lo95_1s=lo95_1s,
                corner_excess=corner_excess, n=len(df))


def cell_descriptive(df: pd.DataFrame, uncond_mean: float, rng):
    """Secondary, DESCRIPTIVE ONLY: non-overlapping coiled-spring episodes."""
    cell = df[df["CELL"]].copy()
    picks, last = [], -10**9
    for t in cell["row"].to_numpy():
        if t >= last + HORIZON + 1:          # non-overlapping forward windows
            picks.append(t)
            last = t
    ep = df[df["row"].isin(picks)]
    rets = ep["R10"].to_numpy()
    print(f"\n--- SECONDARY (descriptive only): coiled-spring cell ---")
    if len(rets) < 3:
        print(f"  independent episodes = {len(rets)} (too few to summarize)")
        return
    boot = np.array([rng.choice(rets, len(rets), replace=True).mean()
                     for _ in range(B)])
    print(f"  independent episodes = {len(rets)}")
    print(f"  mean 10d fwd return = {rets.mean():+.4%}  vs unconditional "
          f"{uncond_mean:+.4%}  (excess {rets.mean()-uncond_mean:+.4%})")
    print(f"  episode-bootstrap 95% CI: {np.percentile(boot,2.5):+.4%} .. "
          f"{np.percentile(boot,97.5):+.4%}")


def main():
    # integrity: confirm we are running against the sealed snapshot
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    print(f"data: {DATA.name}\nSHA-256: {sha}")
    if sha != SEAL_SHA:
        print(f"FATAL: data hash != sealed hash {SEAL_SHA}. Aborting."); sys.exit(1)
    print("integrity OK — matches sealed snapshot.\n")

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["DIX_z"] = trailing_z(df["dix"])
    df["GEX_z"] = trailing_z(df["gex"])
    df["FRAG"] = -df["GEX_z"]
    df["CELL"] = (trailing_pct_flag_high(df["dix"], CELL_DIX_PCT)
                  & trailing_pct_flag_low(df["gex"], CELL_GEX_PCT))
    # entry close of D+1, exit close of D+11  -> R10 indexed at signal day D (row t)
    df["R10"] = df["price"].shift(-(HORIZON + 1)) / df["price"].shift(-1) - 1.0
    df["row"] = np.arange(len(df))

    valid = df.dropna(subset=["DIX_z", "GEX_z", "R10"]).copy()
    rng = np.random.default_rng(SEED)

    insample = valid[valid["date"] <= SPLIT]
    holdout = valid[valid["date"] > SPLIT]
    uncond = valid["R10"].mean()
    print(f"in-sample signal dates: {insample['date'].min().date()} .. "
          f"{insample['date'].max().date()}  (n={len(insample)})")
    print(f"holdout  signal dates: {holdout['date'].min().date()} .. "
          f"{holdout['date'].max().date()}  (n={len(holdout)})")
    print(f"unconditional mean 10d fwd return = {uncond:+.4%}")

    ins = fit_report(insample, "PRIMARY in-sample (2011-2019)", rng)
    cell_descriptive(insample, insample["R10"].mean(), rng)
    hod = fit_report(holdout, "PRIMARY sealed holdout (2020-2026) — SINGLE LOOK", rng)

    # ---- frozen decision rule (prereg section 3) ----
    in_pass = (ins["beta3"] > 0 and ins["p"] < P_BAR
               and ins["corner_excess"] >= ECON_BAR)
    holdout_consistent = (hod["beta3"] > 0 and hod["lo95_1s"] > 0)
    print("\n================ VERDICT ================")
    print(f"in-sample: beta3>0={ins['beta3']>0}  p<{P_BAR}={ins['p']<P_BAR} "
          f"(p={ins['p']:.4f})  corner>=+0.8%={ins['corner_excess']>=ECON_BAR} "
          f"({ins['corner_excess']:+.4%})  -> in-sample {'PASS' if in_pass else 'FAIL'}")
    print(f"holdout: same sign={hod['beta3']>0}  one-sided95 lower>0="
          f"{hod['lo95_1s']>0} ({hod['lo95_1s']:+.5f})")
    if not in_pass:
        verdict = "FAIL — DIX x GEX interaction dead. No resurrection by sub-slicing."
    elif holdout_consistent:
        verdict = "PASS — real orthogonal swing signal. Proceed to SPX/ES overlay spec."
    else:
        verdict = "AMBIGUOUS — suggestive, not validated. Do not deploy; prospective data only."
    print(f"\n>>> {verdict}")


if __name__ == "__main__":
    main()

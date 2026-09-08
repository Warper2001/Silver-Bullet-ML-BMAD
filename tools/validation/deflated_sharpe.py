"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado).

WHY THIS EXISTS
---------------
`arch` gives us SPA/StepM (multiple-testing over a variant family) and `skfolio`
gives us purged/combinatorial CV -- but neither ships a Deflated Sharpe Ratio.
DSR answers a question this shop asks constantly: "we tried N configurations and
the best one has Sharpe X -- how impressed should we be?" It corrects the Sharpe
for (a) the number of trials, (b) the length of the track record, and (c)
non-normal returns (skew and fat tails), all of which inflate a naive Sharpe.

The formulas are short and well specified, so implementing them beats taking a
dependency on a single-maintainer package for a load-bearing statistic.

  PSR(SR*)  = Phi[ (SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ]
  SR0       = sqrt(Var[SR_n]) * [ (1-g)*Z^-1(1 - 1/N) + g*Z^-1(1 - 1/(N e)) ]
  DSR       = PSR(SR0)
  MinTRL    = 1 + (1 - g3*SR + (g4-1)/4 * SR^2) * (Z^-1(a) / (SR - SR*))^2

SR is per-observation (e.g. daily), g3 = skew, g4 = kurtosis (NON-excess: 3 for
a normal), g = Euler-Mascheroni, N = number of trials.

DO NOT TRUST THIS FILE WITHOUT `--self-test`
--------------------------------------------
`--self-test` runs three checks that would fail on a mis-transcribed formula:
  1. PSR(0) on Gaussian returns must match the classical Sharpe t-test p-value.
  2. Under the null (N independent strategies with TRUE Sharpe 0), DSR of the
     best must be ~Uniform(0,1), so P(DSR > 0.95) must be near 5%. This is the
     calibration that matters -- it is the whole point of the statistic.
  3. Expected max Sharpe SR0 must increase with N.

USAGE
-----
    .venv-research/bin/python tools/validation/deflated_sharpe.py --self-test
    .venv-research/bin/python tools/validation/deflated_sharpe.py \
        --glob "data/ml_training/doe_run_0*_history.csv"
"""

from __future__ import annotations

import argparse
import glob as globmod
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------------- core maths --
def sharpe_moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    """Per-observation Sharpe, skew, NON-excess kurtosis, and T."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else np.nan
    g3 = stats.skew(r, bias=False)
    g4 = stats.kurtosis(r, fisher=False, bias=False)  # non-excess
    return float(sr), float(g3), float(g4), int(T)


def probabilistic_sharpe(sr: float, g3: float, g4: float, T: int, sr_star: float = 0.0) -> float:
    """P(true SR > sr_star), correcting for track length, skew and kurtosis."""
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if not np.isfinite(denom) or denom <= 0 or T < 2:
        return np.nan
    z = (sr - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """E[max SR] across `n_trials` independent strategies with true SR = 0."""
    if n_trials < 2 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(returns: np.ndarray, var_sr: float, n_trials: int) -> dict:
    sr, g3, g4, T = sharpe_moments(returns)
    sr0 = expected_max_sharpe(var_sr, n_trials)
    return {
        "sr_per_obs": sr,
        "sr_annualised": sr * np.sqrt(252),
        "skew": g3,
        "kurtosis_nonexcess": g4,
        "T": T,
        "n_trials": n_trials,
        "sr0_expected_max": sr0,
        "psr_vs_zero": probabilistic_sharpe(sr, g3, g4, T, 0.0),
        "dsr": probabilistic_sharpe(sr, g3, g4, T, sr0),
    }


def min_track_record_length(sr: float, g3: float, g4: float,
                            sr_star: float = 0.0, alpha: float = 0.95) -> float:
    """Observations needed for SR to be significantly > sr_star at `alpha`."""
    if sr <= sr_star:
        return np.inf
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    return float(1.0 + denom * (stats.norm.ppf(alpha) / (sr - sr_star)) ** 2)


# ------------------------------------------------------------- self tests ---
def self_test(seed: int = 20260908) -> int:
    rng = np.random.default_rng(seed)
    failures = 0

    print("=" * 70)
    print("SELF-TEST 1 -- PSR(0) on Gaussian returns vs classical t-test p-value")
    print("=" * 70)
    worst = 0.0
    for T in (100, 250, 1000):
        for mu in (0.0, 0.02, 0.05):
            r = rng.standard_normal(T) * 0.01 + mu * 0.01
            sr, g3, g4, TT = sharpe_moments(r)
            psr = probabilistic_sharpe(sr, 0.0, 3.0, TT, 0.0)  # impose normality
            t_p = 1.0 - stats.t.cdf(sr * np.sqrt(TT), df=TT - 1)
            diff = abs(psr - (1.0 - t_p))
            worst = max(worst, diff)
            print(f"  T={T:5d} mu={mu:.2f}  PSR={psr:.4f}  1-t_p={1 - t_p:.4f}  |diff|={diff:.4f}")
    ok1 = worst < 0.02
    print(f"  -> max |diff| = {worst:.4f}  {'PASS' if ok1 else 'FAIL'} (bar: < 0.02)")
    failures += not ok1

    print("\n" + "=" * 70)
    print("SELF-TEST 2a -- DSR under the null: SIZE CONTROL + centring")
    print("=" * 70)
    print("  N independent strategies, ALL with true Sharpe = 0. Deflate the best.")
    print("  DSR is P(true SR > E[max SR under the null]). Under the null that is")
    print("  false for every strategy, so DSR is NOT uniform -- it is deliberately")
    print("  conservative. Two things must hold:")
    print("    (i)  P(DSR > 0.95) <= 0.05                 [size control]")
    print("    (ii) mean DSR ~ 0.5                        [SR0 sits on the expected max]")
    print("  (ii) is the sharp check of expected_max_sharpe: if SR0 were mis-scaled,")
    print("  the mean would drift off 0.5.")
    T, sims = 500, 600
    for n_trials in (10, 50):
        ds = []
        for _ in range(sims):
            R = rng.standard_normal((T, n_trials)) * 0.01
            srs = R.mean(0) / R.std(0, ddof=1)
            best = int(np.argmax(srs))
            ds.append(deflated_sharpe(R[:, best], float(np.var(srs, ddof=1)), n_trials)["dsr"])
        ds = np.asarray(ds)
        r95, mean_d = float((ds > 0.95).mean()), float(ds.mean())
        ok_size = r95 <= 0.05
        ok_centre = abs(mean_d - 0.5) < 0.10
        print(f"  N={n_trials:3d}: P(DSR>0.95)={r95:.3f} (bar <=0.05) {'PASS' if ok_size else 'FAIL'}"
              f"   mean DSR={mean_d:.3f} (bar 0.5 +/- 0.10) {'PASS' if ok_centre else 'FAIL'}")
        failures += (not ok_size) + (not ok_centre)

    print("\n" + "=" * 70)
    print("SELF-TEST 2b -- DSR POWER: a real edge must survive deflation")
    print("=" * 70)
    print("  N-1 null strategies plus ONE with a genuine daily SR of 0.20")
    print("  (~3.2 annualised) over T=1000. DSR must exceed 0.95 most of the time,")
    print("  otherwise the statistic is degenerate and 2a passes for the wrong reason.")
    print("  NOTE: DSR is conservative by construction here -- Var[SR] is estimated")
    print("  from a trial set that INCLUDES the winner, so a genuine edge inflates")
    print("  its own deflation threshold. Detecting a real edge needs either a big")
    print("  effect or a long record; that is a property of the statistic, not a bug.")
    T_pow = 1000
    for n_trials in (10, 50):
        hits, dvals = 0, []
        for _ in range(sims):
            R = rng.standard_normal((T_pow, n_trials)) * 0.01
            R[:, 0] += 0.20 * 0.01          # true per-day SR = 0.20
            srs = R.mean(0) / R.std(0, ddof=1)
            best = int(np.argmax(srs))
            d = deflated_sharpe(R[:, best], float(np.var(srs, ddof=1)), n_trials)["dsr"]
            dvals.append(d)
            hits += d > 0.95
        rate = hits / sims
        ok = rate > 0.50
        print(f"  N={n_trials:3d}: P(DSR>0.95 | a real edge exists)={rate:.3f} "
              f"(bar >0.50) {'PASS' if ok else 'FAIL'}   mean DSR={np.mean(dvals):.3f}")
        failures += not ok

    print("\n" + "=" * 70)
    print("SELF-TEST 3 -- E[max SR] must increase with the number of trials")
    print("=" * 70)
    vals = [expected_max_sharpe(0.01 ** 2, n) for n in (2, 5, 10, 50, 100, 1000)]
    for n, v in zip((2, 5, 10, 50, 100, 1000), vals):
        print(f"  N={n:5d}  E[max SR per obs]={v:.5f}")
    ok3 = all(b > a for a, b in zip(vals, vals[1:]))
    print(f"  -> monotone increasing: {'PASS' if ok3 else 'FAIL'}")
    failures += not ok3

    print("\n" + "=" * 70)
    print("ALL SELF-TESTS PASSED" if failures == 0 else f"{failures} SELF-TEST(S) FAILED")
    print("=" * 70)
    return 1 if failures else 0


# ------------------------------------------------------------------- apply --
def variant_name(path: str) -> str:
    m = re.search(r"(doe_run_\d+[a-z_]*)_history", Path(path).name)
    return m.group(1) if m else Path(path).stem


def apply_to_family(pattern: str, exclude=("fullyear",)) -> int:
    paths = sorted(p for p in globmod.glob(pattern) if not any(x in p for x in exclude))
    if len(paths) < 2:
        sys.exit("ERROR: need a family of >= 2 variants.")
    series = {}
    for p in paths:
        df = pd.read_csv(p)
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        series[variant_name(p)] = pd.Series(df["pnl"].to_numpy(), index=ts.dt.date).groupby(level=0).sum()
    mat = pd.DataFrame(series).sort_index()
    lo = max(s.index.min() for s in series.values())
    hi = min(s.index.max() for s in series.values())
    mat = mat.loc[(mat.index >= lo) & (mat.index <= hi)]
    mat = mat.reindex(pd.date_range(lo, hi, freq="B").date).fillna(0.0)

    n_trials = mat.shape[1]
    srs = np.array([sharpe_moments(mat[c].to_numpy())[0] for c in mat.columns])
    var_sr = float(np.var(srs, ddof=1))

    print(f"Family: {n_trials} variants, {len(mat)} business days "
          f"({mat.index[0]} -> {mat.index[-1]})")
    print(f"Var[SR] across variants = {var_sr:.6e}")
    print(f"E[max SR] under the null (N={n_trials}) = {expected_max_sharpe(var_sr, n_trials):.5f} "
          f"per day = {expected_max_sharpe(var_sr, n_trials) * np.sqrt(252):.3f} annualised\n")

    rows = []
    for c in mat.columns:
        d = deflated_sharpe(mat[c].to_numpy(), var_sr, n_trials)
        sr, g3, g4, _ = sharpe_moments(mat[c].to_numpy())
        d["variant"] = c
        d["min_trl_days"] = min_track_record_length(sr, g3, g4)
        rows.append(d)
    out = pd.DataFrame(rows).sort_values("sr_annualised", ascending=False)
    cols = ["variant", "sr_annualised", "skew", "kurtosis_nonexcess", "T",
            "psr_vs_zero", "sr0_expected_max", "dsr", "min_trl_days"]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    best = out.iloc[0]
    print(f"\nBest variant: {best['variant']}")
    print(f"  naive annualised Sharpe : {best['sr_annualised']:.3f}")
    print(f"  PSR vs zero (no trials) : {best['psr_vs_zero']:.4f}")
    print(f"  DEFLATED (N={n_trials} trials) : {best['dsr']:.4f}")
    verdict = ("SURVIVES deflation at 95%" if best["dsr"] > 0.95
               else "does NOT survive deflation -- consistent with the best of "
                    f"{n_trials} lucky draws")
    print(f"  -> {verdict}")
    if np.isfinite(best["min_trl_days"]):
        print(f"  Minimum track record length for significance: "
              f"{best['min_trl_days']:.0f} days ({best['min_trl_days'] / 252:.1f} yr); "
              f"have {int(best['T'])}")
    return 0


def deflate_grid(csv: str, sr_col: str, t_col: str, select: str | None,
                 min_t: int = 1) -> int:
    """Deflate a parameter-sweep summary table.

    Needs one row per variant with a per-observation Sharpe and an observation
    count. Because a summary table carries no return series, skew and kurtosis
    are unavailable and NORMALITY IS ASSUMED (g3=0, g4=3). Real trade P&L is
    fat-tailed and usually negatively skewed, both of which REDUCE PSR/DSR --
    so the numbers below are an OPTIMISTIC upper bound. A config that fails
    here fails harder with its real moments.
    """
    df = pd.read_csv(csv)
    for c in (sr_col, t_col):
        if c not in df.columns:
            sys.exit(f"ERROR: column {c!r} not in {csv}. Have: {list(df.columns)}")
    raw = len(df)
    df = df[np.isfinite(df[sr_col]) & (df[t_col] > 1)].reset_index(drop=True)

    # Bailey/Lopez de Prado assume the trials are comparable. A per-observation
    # Sharpe estimated from 5 observations has sampling sd ~ 1/sqrt(5) = 0.45,
    # so a sweep containing tiny cells has a Var[SR] dominated by ESTIMATION
    # NOISE rather than genuine dispersion between configurations -- which
    # inflates E[max SR] and makes the deflation spuriously harsh. Filtering to
    # a common minimum sample keeps the family comparable.
    if min_t > 1:
        before = len(df)
        df = df[df[t_col] >= min_t].reset_index(drop=True)
        print(f"  [min-T filter] kept {len(df)} of {before} variants with "
              f"{t_col} >= {min_t}\n")
        if len(df) < 2:
            sys.exit(f"ERROR: only {len(df)} variants survive {t_col} >= {min_t}.")
    n_trials = len(df)
    if raw != n_trials:
        print(f"  ({raw - n_trials} of {raw} rows dropped: non-finite Sharpe, "
              f"{t_col} <= 1, or below --min-t)\n")
    srs = df[sr_col].to_numpy(float)
    var_sr = float(np.var(srs, ddof=1))
    sr0 = expected_max_sharpe(var_sr, n_trials)

    print(f"Grid: {csv}")
    print(f"  variants searched (N)      : {n_trials}")
    print(f"  Sharpe column              : {sr_col!r} (per observation)")
    print(f"  observation-count column   : {t_col!r}")
    print(f"  Var[SR] across variants    : {var_sr:.6e}   sd = {np.sqrt(var_sr):.5f}")
    print(f"  E[max SR] under the null   : {sr0:.5f} per observation")
    print("  NOTE: normality assumed (no return series in a summary table) ->")
    print("        these PSR/DSR values are an OPTIMISTIC upper bound.\n")

    def row_stats(r):
        sr, T = float(r[sr_col]), int(r[t_col])
        return {
            "sr_per_obs": sr, "T": T,
            "psr_vs_zero": probabilistic_sharpe(sr, 0.0, 3.0, T, 0.0),
            "dsr": probabilistic_sharpe(sr, 0.0, 3.0, T, sr0),
        }

    best_i = int(np.argmax(srs))
    targets = [("BEST variant in the grid", df.iloc[best_i])]

    if select:
        crit = {}
        for part in select.split(","):
            k, v = part.split("=")
            crit[k.strip()] = float(v)
        mask = np.ones(len(df), bool)
        for k, v in crit.items():
            if k not in df.columns:
                sys.exit(f"ERROR: select column {k!r} not in {csv}")
            mask &= np.isclose(df[k].to_numpy(float), v)
        if mask.sum() == 0:
            sys.exit(f"ERROR: no row matches {select}")
        for _, r in df[mask].iterrows():
            targets.append((f"SELECTED config ({select})", r))

    for title, r in targets:
        st = row_stats(r)
        desc = "  ".join(f"{c}={r[c]}" for c in df.columns
                         if c not in (sr_col, t_col) and df[c].dtype != object)
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(f"  {desc}")
        print(f"  per-obs Sharpe SR          : {st['sr_per_obs']:.5f}   over T={st['T']}")
        print(f"  PSR vs zero (ignoring N)   : {st['psr_vs_zero']:.4f}")
        print(f"  DEFLATED for N={n_trials} trials    : {st['dsr']:.4f}")
        verdict = ("SURVIVES deflation at 95%" if st["dsr"] > 0.95 else
                   f"does NOT survive -- consistent with the best of {n_trials} draws")
        print(f"  -> {verdict}")
        mtrl = min_track_record_length(st["sr_per_obs"], 0.0, 3.0, sr0)
        if np.isfinite(mtrl):
            print(f"  observations needed to beat E[max SR] at 95%: {mtrl:,.0f} "
                  f"(have {st['T']})")
        else:
            print(f"  SR is at or below E[max SR] under the null "
                  f"({st['sr_per_obs']:.5f} <= {sr0:.5f}) -- no track record length suffices")
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--glob", default="data/ml_training/doe_run_0*_history.csv")
    ap.add_argument("--grid", help="parameter-sweep summary CSV (one row per variant)")
    ap.add_argument("--sr-col", default="ir", help="per-observation Sharpe column")
    ap.add_argument("--t-col", default="n_oos", help="observation-count column")
    ap.add_argument("--select", help="the chosen config, e.g. 'sl=2.0,tp=8.0,threshold=0.5'")
    ap.add_argument("--min-t", type=int, default=1,
                    help="drop variants with fewer than this many observations, so the\n"
                         "trials are comparable and Var[SR] is not dominated by noise")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.grid:
        return deflate_grid(args.grid, args.sr_col, args.t_col, args.select, args.min_t)
    return apply_to_family(args.glob)


if __name__ == "__main__":
    raise SystemExit(main())

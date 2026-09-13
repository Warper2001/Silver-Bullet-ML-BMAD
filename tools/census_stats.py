#!/usr/bin/env python3
"""Shared statistics for the census harnesses (tier2_entry_arms.py, gap_fade_census.py).

Every function takes an explicit ``rng`` (numpy Generator) so a run is reproducible
from its seed, and so tests can pin behavior. The implementations are the ones used
in the 2026-09-13 pre-registered diagnostics:
  _bmad-output/diagnostics_yank_entry_mechanics_20260913/  (plan sha 1b4eabab)
  _bmad-output/diagnostics_gap_fade_census_20260913/       (plan sha 3df09ccd)
Changing the order of rng calls changes results; the regression check in each tool's
docstring compares against those runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def boot_mean_ci(x, rng, groups=None, reps=10_000):
    """Percentile CI of the mean; iid resampling or whole-group (cluster) resampling.

    Returns (ci, bootstrap_means)."""
    x = np.asarray(x, float)
    n = len(x)
    if groups is None:
        m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(reps)])
    else:
        ug = np.unique(groups)
        gi = {g: np.flatnonzero(groups == g) for g in ug}
        m = np.array([x[np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(reps)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], m


def p_greater_zero(x, rng, reps=10_000) -> float:
    """One-sided bootstrap p for mean > 0 (shift method)."""
    x = np.asarray(x, float)
    _, m = boot_mean_ci(x, rng, reps=reps)
    return float(np.mean(m - x.mean() >= x.mean()))


def boot_p_gt0_with_ci(x, rng, reps=10_000):
    """p for mean > 0 (shift method) and percentile CI from the same draws."""
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)])
    return float(np.mean(m - x.mean() >= x.mean())), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cluster_se(d, groups) -> float:
    """Standard error of the mean with errors clustered by ``groups``."""
    d = np.asarray(d, float)
    n, mu = len(d), d.mean()
    s = pd.Series(d - mu).groupby(groups).sum().to_numpy()
    return float(np.sqrt((s ** 2).sum()) / n)


def paired_cluster_test(d, groups, rng, reps=10_000) -> dict:
    """Studentized cluster bootstrap for mean(d); one-sided p for >0 and <0."""
    d = np.asarray(d, float)
    mu, se = d.mean(), cluster_se(d, groups)
    if se == 0:
        return {"mean": float(mu), "se": 0.0, "p_gt": 1.0, "p_lt": 1.0}
    t_obs = mu / se
    ug = np.unique(groups)
    gi = {g: np.flatnonzero(groups == g) for g in ug}
    ts = []
    for _ in range(reps):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        db, gb = d[idx], groups[idx]
        seb = cluster_se(db, gb)
        if seb > 0:
            ts.append((db.mean() - mu) / seb)
    ts = np.array(ts)
    return {"mean": float(mu), "se": se, "t": float(t_obs),
            "p_gt": float(np.mean(ts >= t_obs)), "p_lt": float(np.mean(ts <= t_obs))}


def paired_iid_test(d, rng, reps=10_000) -> dict:
    """Studentized iid bootstrap for mean(d); one-sided p for >0 and <0."""
    d = np.asarray(d, float)
    n, mu, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    ts = []
    for _ in range(reps):
        b = d[rng.integers(0, n, n)]
        sb = b.std(ddof=1) / np.sqrt(n)
        if sb > 0:
            ts.append((b.mean() - mu) / sb)
    ts, t0 = np.array(ts), mu / se
    return {"mean": float(mu), "se": float(se), "p_gt": float(np.mean(ts >= t0)), "p_lt": float(np.mean(ts <= t0))}


def sr_diff_ci(a, b, groups, rng, reps=4000) -> list[float]:
    """Cluster-bootstrap percentile CI of sharpe(a) - sharpe(b) on paired arrays."""
    ug = np.unique(groups)
    gi = {g: np.flatnonzero(groups == g) for g in ug}
    v = []
    for _ in range(reps):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        v.append(sharpe(a[idx]) - sharpe(b[idx]))
    v = np.array([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mde_dsr(rho: float, n: int, z: float = 2.487) -> float:
    """Minimum detectable paired Sharpe difference (one-sided 5%, 80% power by default)."""
    return float(z * np.sqrt(2 * (1 - rho) / n))

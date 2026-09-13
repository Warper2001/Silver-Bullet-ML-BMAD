#!/usr/bin/env python3
"""Tests for tools/census_stats.py — the statistics behind the census harnesses.

Run:   .venv/bin/python -m pytest tools/test_census_stats.py -v
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "tools"))
cs = importlib.import_module("census_stats")


def test_holm_known_values_and_input_order():
    assert cs.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert cs.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_sharpe_constant_series_is_nan():
    assert math.isnan(cs.sharpe([1.0, 1.0, 1.0]))
    assert cs.sharpe([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_cluster_se_equals_iid_se_when_every_obs_is_its_own_cluster():
    d = np.array([0.3, -1.2, 0.8, 2.0, -0.4])
    iid = d.std(ddof=0) / np.sqrt(len(d))
    assert cs.cluster_se(d, np.arange(len(d))) == pytest.approx(iid)


def test_cluster_se_grows_when_correlated_obs_share_a_cluster():
    d = np.array([1.0, 1.0, -1.0, -1.0])
    assert cs.cluster_se(d, np.array([0, 0, 1, 1])) > cs.cluster_se(d, np.arange(4))


def test_boot_mean_ci_reproducible_and_covers_mean():
    x = np.random.default_rng(1).normal(0.5, 1.0, 200)
    ci1, _ = cs.boot_mean_ci(x, np.random.default_rng(7), reps=2000)
    ci2, _ = cs.boot_mean_ci(x, np.random.default_rng(7), reps=2000)
    assert ci1 == ci2
    assert ci1[0] < x.mean() < ci1[1]


def test_p_greater_zero_small_for_clear_positive_and_large_for_noise():
    rng = np.random.default_rng(3)
    assert cs.p_greater_zero(rng.normal(1.0, 1.0, 100), np.random.default_rng(0), reps=2000) < 0.01
    assert cs.p_greater_zero(rng.normal(0.0, 1.0, 100) - 0.2, np.random.default_rng(0), reps=2000) > 0.05


@pytest.mark.parametrize("test", ["cluster", "iid"])
def test_paired_tests_detect_a_real_shift_and_not_noise(test):
    rng = np.random.default_rng(11)
    shifted = rng.normal(1.0, 1.0, 80)
    noise = rng.normal(0.0, 1.0, 80)
    groups = np.repeat(np.arange(40), 2)
    run = (lambda d: cs.paired_cluster_test(d, groups, np.random.default_rng(5), reps=2000)) if test == "cluster" \
        else (lambda d: cs.paired_iid_test(d, np.random.default_rng(5), reps=2000))
    r = run(shifted)
    assert r["p_gt"] < 0.01 and r["p_lt"] > 0.9
    assert run(noise)["p_gt"] > 0.05


def test_paired_cluster_test_zero_variance_returns_no_evidence():
    r = cs.paired_cluster_test(np.zeros(6), np.arange(6), np.random.default_rng(0))
    assert r["p_gt"] == 1.0 and r["p_lt"] == 1.0


def test_mde_matches_the_power_table_in_the_plans():
    assert cs.mde_dsr(0.3, 80) == pytest.approx(0.329, abs=1e-3)
    assert cs.mde_dsr(0.9, 120) == pytest.approx(0.102, abs=1e-3)


def test_sr_diff_ci_brackets_zero_for_identical_arms():
    a = np.random.default_rng(2).normal(0.1, 1.0, 60)
    ci = cs.sr_diff_ci(a, a.copy(), np.arange(60), np.random.default_rng(0), reps=500)
    assert ci == pytest.approx([0.0, 0.0])

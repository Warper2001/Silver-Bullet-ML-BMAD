"""Offline analytical examples, not a power gate or a market-data interface.

No input files, network, forecasts, or repository imports. All numbers are
explicit synthetic assumptions, published schedule illustrations, or copies
of the prior conditional capacity scenarios. Prints JSON; writes no files.
"""

import json
import math
from statistics import NormalDist

from scipy.stats import chi2


def paired_variance(vk: float, vm: float, covariance: float) -> float:
    if not all(math.isfinite(x) for x in (vk, vm, covariance)):
        raise ValueError("finite moments required")
    if vk < 0 or vm < 0 or abs(covariance) > math.sqrt(vk * vm):
        raise ValueError("invalid covariance matrix")
    return vk + vm - 2 * covariance


def ar1_mean_variance(n: int, rho: float, variance: float = 1.0) -> float:
    """Exact AR(1) mean variance with known stationary synthetic parameters."""
    if type(n) is not int or n < 1:
        raise ValueError("positive integer n required")
    if not math.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("stationary rho required")
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("positive finite variance required")
    return (
        variance * (n + 2 * sum((n - h) * rho**h for h in range(1, n))) / n**2
    )


def variance_upper(n: int, sample_variance: float, gamma: float) -> float:
    """Normal-iid variance limit; no dependent effective-df shortcut."""
    if type(n) is not int or n < 2 or not 0 < gamma < 1:
        raise ValueError("insufficient calibration or invalid confidence")
    if not math.isfinite(sample_variance) or sample_variance <= 0:
        raise ValueError(
            "degenerate/invalid calibration is not evidence of zero risk"
        )
    return (n - 1) * sample_variance / float(chi2.ppf(gamma, n - 1))


def required_n(
    effect_distance: float, variance: float, inflation: float
) -> int:
    if not all(
        math.isfinite(x) for x in (effect_distance, variance, inflation)
    ):
        raise ValueError("finite assumptions required")
    if effect_distance <= 0 or variance <= 0 or inflation < 1:
        raise ValueError(
            "positive distance/variance and inflation >= 1 required"
        )
    zsum = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.90)
    return math.ceil(zsum**2 * variance * inflation**2 / effect_distance**2)


def calculate() -> dict:
    normal = NormalDist()
    zsum = normal.inv_cdf(0.975) + normal.inv_cdf(0.90)
    # Synthetic dollar moments, not estimates or planning alternatives.
    vk, vm, cov = 10000.0, 6400.0, 4000.0
    vd = paired_variance(vk, vm, cov)
    u = variance_upper(20, vd, 0.05)
    capacities = {
        "proxy-no-roll-capacity-ceiling": [59, 119, 246],
        "assumed-quarterly-resets": [54, 109, 226],
        "assumed-resets-losses-and-20-calibration": [15, 46, 119],
    }
    horizon_rows = []
    for scenario, counts in capacities.items():
        for months, n in zip((3, 6, 12), counts):
            horizon_rows.append(
                {
                    "scenario": scenario,
                    "months": months,
                    "conditional_evaluation_sessions": n,
                    "standardized_mde": {
                        str(f): zsum * f / math.sqrt(n)
                        for f in (1.0, 1.5, 2.0)
                    },
                }
            )
    precision = []
    for n in (2, 5, 20, 60):
        nu = n - 1
        precision.append(
            {
                "synthetic_iid_normal_calibration_n": n,
                "degrees_of_freedom": nu,
                "variance_ucl_factor_95pct": variance_upper(n, 1, 0.05),
                "variance_ci_upper_lower_ratio_95pct": float(
                    chi2.ppf(0.975, nu) / chi2.ppf(0.025, nu)
                ),
            }
        )
    # Finite-variance rare tail, absent in most small synthetic pilots.
    p, tail, ncal = 0.001, 1000.0, 20
    return {
        "scope": "SYNTHETIC_AND_CONDITIONAL_PLANNING_ONLY",
        "strategy_test_permitted": False,
        "trading_authorized": False,
        "zsum": zsum,
        "schedule_illustration": {
            "fee_per_side_current": 0.50 + 0.10 + 0.35 + 0.01,
            "round_trip_current": 2 * (0.50 + 0.10 + 0.35 + 0.01),
            "round_trip_one_tick_each_side": 2
            * (0.50 + 0.10 + 0.35 + 0.01 + 0.50),
            "round_trip_from_2027_07_other_rates_held_fixed": 2
            * (0.50 + 0.10 + 0.35 + 0.02),
        },
        "synthetic_pair": {
            "variance_k": vk,
            "variance_m": vm,
            "covariance": cov,
            "variance_difference": vd,
            "sd_difference": math.sqrt(vd),
            "cauchy_upper_variance": (math.sqrt(vk) + math.sqrt(vm)) ** 2,
            "ncal": 20,
            "one_sided_gamma": 0.05,
            "variance_ucl": u,
            "sd_ucl": math.sqrt(u),
            "illustrative_distance_not_economic_target": 20,
            "iid_plugin_required_n": required_n(20, vd, 1),
            "iid_ucl_required_n": required_n(20, u, 1),
            "ucl_and_assumed_1_5_se_inflation_required_n": required_n(
                20, u, 1.5
            ),
        },
        "dependence_examples": [
            {
                "n": 20,
                "rho": rho,
                "variance_mean": ar1_mean_variance(20, rho),
                "exact_se_inflation_vs_iid": math.sqrt(
                    20 * ar1_mean_variance(20, rho)
                ),
                "asymptotic_variance_inflation": (1 + rho) / (1 - rho),
            }
            for rho in (0.0, 0.5, 0.9)
        ],
        "permutation_counterexamples": {
            "K_equals_M_actual_variance": paired_variance(1, 1, 1),
            "K_equals_negative_M_actual_variance": paired_variance(1, 1, -1),
            "independent_mismatch_variance": paired_variance(1, 1, 0),
            "nonidentity_pairing_with_fixed_points": [0, 2, 1, 3],
        },
        "rare_tail_counterexample": {
            "probability_tail": p,
            "tail_value": tail,
            "ncal": ncal,
            "true_variance": p * (1 - p) * tail**2,
            "probability_zero_observed_variance_from_all_zero_pilot": (1 - p)
            ** ncal,
        },
        "uncertainty_accounting_example": {
            "assumed_joint_nuisance_region_coverage": 0.95,
            "conditional_joint_power_lower_bound": 0.8,
            "unconditional_assurance_lower_bound": 0.95 * 0.8,
            "two_separate_95pct_bounds_joint_coverage_lower_bound": 0.9,
            "two_separate_bounds_assurance_lower_bound": 0.9 * 0.8,
        },
        "precision_examples": precision,
        "horizons": horizon_rows,
    }


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2, sort_keys=True, allow_nan=False))

"""Conditional known-variance planning, never actual strategy admission."""

from __future__ import annotations
import math
from statistics import NormalDist
from typing import Any
from . import FLAGS

NORMAL = NormalDist()


def standardized(n: int, inflation: float = 1.0) -> dict[str, Any]:
    if type(n) is not int or n <= 0 or not math.isfinite(inflation) or inflation < 1:
        raise ValueError(
            "positive integer sessions and finite SE inflation >=1 required"
        )
    return {
        **FLAGS,
        "sessions": n,
        "se_inflation": inflation,
        "alpha_per_test": 0.025,
        "marginal_target_power": 0.90,
        "joint_target_lower_bound": 0.80,
        "detectable_standardized_effect": (NORMAL.inv_cdf(0.975) + NORMAL.inv_cdf(0.90))
        * inflation
        / math.sqrt(n),
    }


def dollar_scenario(
    n: int,
    inflation: float,
    *,
    k_effect: float,
    incremental_effect: float,
    k_variance: float,
    m_variance: float,
    covariance: float,
    independent_evidence: dict[str, str],
) -> dict[str, Any]:
    """Caller must supply independently reviewed evidence for all economic inputs.

    References document conditional assumptions; they cannot certify them or admit a test.
    """
    standardized(n, inflation)
    required = {
        "k_effect",
        "incremental_effect",
        "k_variance",
        "m_variance",
        "covariance",
        "dependence",
    }
    if set(independent_evidence) != required or not all(
        isinstance(v, str) and v.strip() for v in independent_evidence.values()
    ):
        raise ValueError(
            "independent evidence references required for effects, variances, covariance and dependence"
        )
    values = (k_effect, incremental_effect, k_variance, m_variance, covariance)
    if not all(math.isfinite(x) for x in values) or min(values[:4]) <= 0:
        raise ValueError("positive finite effects/variances required")
    if abs(covariance) > math.sqrt(k_variance * m_variance):
        raise ValueError("invalid covariance")
    paired_variance = k_variance + m_variance - 2 * covariance
    if not math.isfinite(paired_variance) or paired_variance <= 0:
        raise ValueError("positive paired variance required")
    powers = [
        NORMAL.cdf(effect * math.sqrt(n / variance) / inflation - NORMAL.inv_cdf(0.975))
        for effect, variance in (
            (k_effect, k_variance),
            (incremental_effect, paired_variance),
        )
    ]
    return {
        **FLAGS,
        "status": "CONDITIONAL_PLANNING_ONLY",
        "actual_power": "UNASSESSABLE",
        "independent_evidence_references": independent_evidence,
        "assumptions": {
            "sessions": n,
            "se_inflation": inflation,
            "k_effect": k_effect,
            "incremental_effect": incremental_effect,
            "k_variance": k_variance,
            "m_variance": m_variance,
            "covariance": covariance,
        },
        "paired_variance": paired_variance,
        "marginal_powers": powers,
        "joint_power_lower_bound": max(0.0, sum(powers) - 1),
    }


def planning() -> dict[str, Any]:
    return {
        **FLAGS,
        "actual_power": "UNASSESSABLE",
        "actual_sessions": None,
        "dollar_effects_adopted": False,
        "sampling_unit": "one eligible session",
        "scenarios": [
            standardized(n, i) for n in (20, 60, 120, 252, 504) for i in (1.0, 1.5, 2.0)
        ],
        "limitations": [
            "N is assumed, not an admitted population.",
            "Seeds, arms and overlapping forecasts do not increase N.",
            "Normal known-variance approximation; dependence inflation is hypothetical.",
            "Useful dollar effects and independent variances/covariance remain absent.",
        ],
    }

"""Kolmogorov-Smirnov (KS) test calculator for prediction drift detection.

The KS test is a non-parametric test that compares two cumulative distribution functions.
It detects whether two samples come from the same distribution.

For drift detection:
- We compare recent predictions (last 24 hours) to baseline predictions (training data)
- A low p-value (< 0.05) indicates significant drift (distributions have changed)
- The KS statistic measures the maximum distance between the two CDFs
"""

import logging

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def calculate_ks_statistic(
    baseline_predictions: np.ndarray,
    recent_predictions: np.ndarray,
    min_samples: int = 100,
) -> tuple[float, float]:
    """Calculate Kolmogorov-Smirnov test statistic and p-value.

    The KS test compares two distributions to detect significant differences.
    For drift detection, we compare prediction distributions.

    Args:
        baseline_predictions: Baseline predictions (e.g., training data)
        recent_predictions: Recent predictions (e.g., last 24 hours)
        min_samples: Minimum samples required for valid KS test

    Returns:
        Tuple of (ks_statistic, p_value)
        - ks_statistic: Maximum distance between CDFs (0 to 1)
        - p_value: P-value for the test (0 to 1, lower = more significant drift)

    Raises:
        ValueError: If insufficient data or invalid parameters

    Examples:
        >>> baseline = np.array([0.5, 0.6, 0.7] * 100)
        >>> recent = np.array([0.5, 0.6, 0.7] * 100)
        >>> stat, p_value = calculate_ks_statistic(baseline, recent)
        >>> print(f"KS statistic: {stat:.3f}, p-value: {p_value:.3f}")
        KS statistic: 0.000, p-value: 1.000  # No drift
    """
    # Validate inputs
    if len(baseline_predictions) < min_samples:
        raise ValueError(
            f"Baseline predictions insufficient: "
            f"{len(baseline_predictions)} < {min_samples}"
        )
    if len(recent_predictions) < min_samples:
        raise ValueError(
            f"Recent predictions insufficient: "
            f"{len(recent_predictions)} < {min_samples}"
        )

    # Remove NaN/Inf values
    baseline_clean = baseline_predictions[~np.isnan(baseline_predictions)]
    recent_clean = recent_predictions[~np.isnan(recent_predictions)]

    if len(baseline_clean) < min_samples or len(recent_clean) < min_samples:
        raise ValueError(
            f"After cleaning, insufficient samples: "
            f"baseline={len(baseline_clean)}, recent={len(recent_clean)}, "
            f"required={min_samples}"
        )

    # Perform KS test
    ks_statistic, p_value = ks_2samp(baseline_clean, recent_clean)

    return float(ks_statistic), float(p_value)


def classify_prediction_drift(
    ks_statistic: float, p_value: float, p_value_threshold: float = 0.05
) -> bool:
    """Classify whether prediction drift has occurred.

    Args:
        ks_statistic: KS statistic value
        p_value: P-value from KS test
        p_value_threshold: Threshold for drift detection (default: 0.05)

    Returns:
        True if drift detected (p-value < threshold), False otherwise

    Examples:
        >>> classify_prediction_drift(ks_statistic=0.1, p_value=0.15)
        False  # No drift
        >>> classify_prediction_drift(ks_statistic=0.3, p_value=0.01)
        True  # Drift detected
    """
    return p_value < p_value_threshold


def calculate_drift_magnitude(
    baseline_predictions: np.ndarray, recent_predictions: np.ndarray
) -> dict[str, float]:
    """Calculate various drift magnitude metrics.

    Provides multiple measures of how much the prediction distribution has shifted.

    Args:
        baseline_predictions: Baseline predictions
        recent_predictions: Recent predictions

    Returns:
        Dictionary with drift metrics:
        - mean_shift: Difference in means
        - std_shift: Difference in standard deviations
        - median_shift: Difference in medians
        - distribution_shift: KS statistic

    Examples:
        >>> baseline = np.array([0.5, 0.6, 0.7] * 100)
        >>> recent = np.array([0.6, 0.7, 0.8] * 100)
        >>> magnitude = calculate_drift_magnitude(baseline, recent)
        >>> print(f"Mean shift: {magnitude['mean_shift']:.3f}")
    """
    # Clean data
    baseline_clean = baseline_predictions[~np.isnan(baseline_predictions)]
    recent_clean = recent_predictions[~np.isnan(recent_predictions)]

    # Calculate KS statistic
    ks_statistic, _ = calculate_ks_statistic(baseline_clean, recent_clean)

    # Calculate various shift metrics
    mean_shift = float(np.mean(recent_clean) - np.mean(baseline_clean))
    std_shift = float(np.std(recent_clean) - np.std(baseline_clean))
    median_shift = float(np.median(recent_clean) - np.median(baseline_clean))

    return {
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "median_shift": median_shift,
        "distribution_shift": ks_statistic,
    }

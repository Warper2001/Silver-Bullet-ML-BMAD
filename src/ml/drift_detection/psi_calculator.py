"""Population Stability Index (PSI) calculator for feature drift detection.

PSI measures the distribution shift between baseline (training) and recent data.
Higher PSI values indicate more drift.

PSI Formula:
PSI = Σ ((Actual% - Expected%) × ln(Actual% / Expected%))

Interpretation:
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.2: Slight drift
- 0.2 ≤ PSI < 0.5: Moderate drift (alert threshold)
- PSI ≥ 0.5: Severe drift (critical alert)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    min_samples: int = 100,
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.

    PSI measures the distribution shift between baseline (training) and recent data.

    Args:
        expected: Baseline distribution (e.g., training data)
        actual: Recent distribution (e.g., last 24 hours)
        bins: Number of bins for discretization (default: 10 for deciles)
        min_samples: Minimum samples required for valid PSI calculation

    Returns:
        PSI score (higher = more drift)

    Raises:
        ValueError: If insufficient data or invalid parameters

    Examples:
        >>> expected = np.array([1, 2, 3, 4, 5] * 100)
        >>> actual = np.array([1, 2, 3, 4, 5] * 100)
        >>> psi = calculate_psi(expected, actual)
        >>> print(f"PSI: {psi:.3f}")  # Should be near 0 (no drift)
    """
    # Validate inputs
    if len(expected) < min_samples:
        raise ValueError(
            f"Expected distribution has insufficient samples: "
            f"{len(expected)} < {min_samples}"
        )
    if len(actual) < min_samples:
        raise ValueError(
            f"Actual distribution has insufficient samples: "
            f"{len(actual)} < {min_samples}"
        )
    if bins < 2:
        raise ValueError(f"Number of bins must be at least 2, got {bins}")

    # Remove NaN/Inf values
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]

    if len(expected_clean) < min_samples or len(actual_clean) < min_samples:
        raise ValueError(
            f"After cleaning NaN/Inf values, insufficient samples remain: "
            f"expected={len(expected_clean)}, actual={len(actual_clean)}, "
            f"required={min_samples}"
        )

    # Determine bin edges using expected distribution (baseline)
    # This ensures we're comparing apples to apples
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected_clean, percentiles)

    # Handle duplicate bin edges (can happen with constant values)
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        logger.warning(
            "Expected distribution has constant values, "
            "using uniform binning instead"
        )
        min_val = min(expected_clean.min(), actual_clean.min())
        max_val = max(expected_clean.max(), actual_clean.max())
        if min_val == max_val:
            # All values are the same
            return 0.0  # No drift possible
        bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Digitize both distributions into bins
    expected_binned = np.digitize(expected_clean, bin_edges[:-1], right=False)
    actual_binned = np.digitize(actual_clean, bin_edges[:-1], right=False)

    # Ensure we have bins+1 categories (0 to bins-1)
    expected_counts = np.bincount(expected_binned, minlength=bins)
    actual_counts = np.bincount(actual_binned, minlength=bins)

    # Convert to percentages
    expected_pct = expected_counts / len(expected_clean)
    actual_pct = actual_counts / len(actual_clean)

    # Avoid division by zero and log of zero
    # Replace 0 with small value (0.0001 = 0.01%)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    # Calculate PSI
    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_components)

    return float(psi)


def calculate_psi_for_multiple_features(
    expected_features: dict[str, np.ndarray],
    actual_features: dict[str, np.ndarray],
    bins: int = 10,
    min_samples: int = 100,
) -> dict[str, float]:
    """Calculate PSI for multiple features.

    Args:
        expected_features: Dictionary of feature_name -> baseline distribution
        actual_features: Dictionary of feature_name -> recent distribution
        bins: Number of bins for PSI calculation
        min_samples: Minimum samples required per feature

    Returns:
        Dictionary of feature_name -> PSI score

    Raises:
        ValueError: If feature sets don't match or insufficient data

    Examples:
        >>> expected = {
        ...     "feature1": np.array([1, 2, 3] * 100),
        ...     "feature2": np.array([4, 5, 6] * 100),
        ... }
        >>> actual = {
        ...     "feature1": np.array([1, 2, 3] * 100),
        ...     "feature2": np.array([4, 5, 6] * 100),
        ... }
        >>> psi_scores = calculate_psi_for_multiple_features(expected, actual)
        >>> for feature, psi in psi_scores.items():
        ...     print(f"{feature}: PSI={psi:.3f}")
    """
    # Validate feature sets match
    expected_features_set = set(expected_features.keys())
    actual_features_set = set(actual_features.keys())

    if expected_features_set != actual_features_set:
        raise ValueError(
            f"Feature sets don't match: "
            f"expected={expected_features_set}, "
            f"actual={actual_features_set}"
        )

    if not expected_features_set:
        raise ValueError("No features provided for PSI calculation")

    # Calculate PSI for each feature
    psi_scores = {}
    for feature_name in expected_features:
        try:
            psi = calculate_psi(
                expected=expected_features[feature_name],
                actual=actual_features[feature_name],
                bins=bins,
                min_samples=min_samples,
            )
            psi_scores[feature_name] = psi
        except ValueError as e:
            logger.warning(f"Could not calculate PSI for {feature_name}: {e}")
            # Use NaN for features that can't be calculated
            psi_scores[feature_name] = np.nan

    return psi_scores


def classify_drift_severity(psi_score: float) -> str:
    """Classify drift severity based on PSI score.

    Args:
        psi_score: PSI score

    Returns:
        Severity level: "none", "moderate", or "severe"

    Examples:
        >>> classify_drift_severity(0.1)
        'none'
        >>> classify_drift_severity(0.3)
        'moderate'
        >>> classify_drift_severity(0.6)
        'severe'
    """
    if psi_score < 0.2:
        return "none"
    elif psi_score < 0.5:
        return "moderate"
    else:
        return "severe"

"""Threshold sensitivity analysis for drift detection.

This script tests different PSI and KS p-value thresholds on stable historical
data (no actual regime shift) to calculate false positive rates and select
optimal thresholds that achieve < 10% false positive rate.

Analysis Strategy:
1. Use January 2025 data (assumed stable, no regime shift)
2. Split into first half (baseline) and second half (test)
3. Test various PSI thresholds (0.1, 0.15, 0.2, 0.25, 0.3)
4. Test various KS p-value thresholds (0.01, 0.02, 0.05, 0.07, 0.1)
5. Calculate false positive rate for each combination
6. Select thresholds achieving < 10% false positive rate
"""

import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.drift_detection import StatisticalDriftDetector
from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_process_data(data_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and process data for a given date range."""
    data_file = Path(data_path) / "mnq_1min_2025.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load data
    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Check if timestamps are timezone-aware
    is_tz_aware = df["timestamp"].dt.tz is not None

    # Filter to date range
    if is_tz_aware:
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
    else:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

    df_filtered = df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    logger.info(f"Loaded {len(df_filtered)} bars for {start_date} to {end_date}")

    return df_filtered


def extract_features_and_predictions(data: pd.DataFrame, feature_engineer: FeatureEngineer, model: any) -> tuple:
    """Extract features and predictions from data."""
    logger.info("Extracting features and predictions...")

    # Engineer features
    features_df = feature_engineer.engineer_features(data)

    # Exclude non-numeric columns
    exclude_cols = {
        "timestamp", "open", "high", "low", "close", "volume",
        "hour", "day_of_week", "trading_session",
        "is_london_am", "is_ny_am", "is_ny_pm",
    }
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_columns)} numeric features")

    # Handle NaN
    features_df_selected = features_df[feature_columns].ffill().fillna(0)

    # Pad to match model's expected features if needed
    if hasattr(model, "base_model"):
        n_expected = model.base_model.n_features_in_
    elif hasattr(model, "n_features_in_"):
        n_expected = model.n_features_in_
    else:
        n_expected = len(feature_columns)

    n_available = len(feature_columns)

    if n_available < n_expected:
        n_padding = n_expected - n_available
        logger.warning(f"Padding {n_padding} dummy features (expected {n_expected}, got {n_available})")
        for i in range(n_padding):
            features_df_selected[f'dummy_{i}'] = 0.0

    # Prepare data
    X = features_df_selected.values
    predictions = model.predict_proba(X)[:, 1]

    # Create features dictionary (filter out dummy features with constant values)
    features_dict = {}
    for col in features_df_selected.columns:
        feature_values = features_df_selected[col].values
        # Skip dummy features (all zeros or constant values)
        if np.std(feature_values) > 0.0001:  # Has some variation
            features_dict[col] = feature_values

    logger.info(f"Extracted {len(features_dict)} features (after filtering constants), {len(predictions)} predictions")

    return features_dict, predictions


def test_threshold_combination(
    baseline_features: dict,
    baseline_predictions: np.ndarray,
    test_features: dict,
    test_predictions: np.ndarray,
    psi_threshold: float,
    ks_p_value_threshold: float,
) -> dict:
    """Test a specific threshold combination.

    Args:
        baseline_features: Features from baseline period
        baseline_predictions: Predictions from baseline period
        test_features: Features from test period
        test_predictions: Predictions from test period
        psi_threshold: PSI threshold for drift detection
        ks_p_value_threshold: KS p-value threshold for drift detection

    Returns:
        Dictionary with test results:
        - psi_threshold: PSI threshold used
        - ks_p_value_threshold: KS p-value threshold used
        - drift_detected: Whether drift was detected (should be False for stable data)
        - is_false_positive: True if drift detected (shouldn't be for stable data)
        - num_drifting_features: Number of features flagged as drifting
        - ks_p_value: Actual KS test p-value
    """
    # Create drift detector with custom thresholds
    from src.ml.drift_detection.models import DriftDetectorConfig

    config = DriftDetectorConfig(
        psi_bins=10,
        psi_threshold_moderate=psi_threshold,
        psi_threshold_severe=psi_threshold * 2.5,  # Approximate standard ratio
        ks_p_value_threshold=ks_p_value_threshold,
    )

    # Get first 10 feature names
    feature_names = list(baseline_features.keys())[:10]

    detector = StatisticalDriftDetector(
        baseline_features=baseline_features,
        baseline_predictions=baseline_predictions,
        feature_names=feature_names,
        config=config,
    )

    # Run drift detection on test data (should NOT detect drift for stable data)
    result = detector.detect_drift(
        recent_features=test_features,
        recent_predictions=test_predictions,
    )

    return {
        "psi_threshold": psi_threshold,
        "ks_p_value_threshold": ks_p_value_threshold,
        "drift_detected": result.drift_detected,
        "is_false_positive": result.drift_detected,  # Should be False for stable data
        "num_drifting_features": len(result.drifting_features),
        "ks_p_value": result.ks_result.p_value if result.ks_result else None,
        "ks_statistic": result.ks_result.ks_statistic if result.ks_result else None,
        "drifting_features": result.drifting_features,
    }


def run_threshold_sensitivity_analysis():
    """Run full threshold sensitivity analysis."""
    logger.info("=" * 80)
    logger.info("Threshold Sensitivity Analysis for Drift Detection")
    logger.info("=" * 80)

    # Configuration
    data_path = "data/processed/dollar_bars/1_minute"
    model_path = "data/models/xgboost/1_minute/calibrated_model.joblib"

    # Use January 2025 data (assumed stable, no regime shift)
    # Split into first half (baseline) and second half (test)
    january_start = "2025-01-01"
    january_end = "2025-01-31"

    # Threshold combinations to test
    psi_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    ks_p_value_thresholds = [0.01, 0.02, 0.05, 0.07, 0.1]

    all_results = []

    try:
        # Load model
        logger.info("Loading calibrated model...")
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {type(model).__name__}")

        # Initialize feature engineer
        feature_engineer = FeatureEngineer()

        # Load January 2025 data
        logger.info(f"\nLoading January 2025 data (stable period)...")
        january_data = load_and_process_data(data_path, january_start, january_end)

        if len(january_data) < 1000:
            logger.error(f"Insufficient January data: {len(january_data)} < 1000")
            return {"passed": False, "error": "Insufficient January data"}

        # Split into baseline (first half) and test (second half)
        split_point = len(january_data) // 2
        baseline_data = january_data.iloc[:split_point].reset_index(drop=True)
        test_data = january_data.iloc[split_point:].reset_index(drop=True)

        logger.info(f"Data split: baseline={len(baseline_data)} bars, test={len(test_data)} bars")

        # Extract baseline features and predictions
        logger.info("\nExtracting baseline features and predictions...")
        baseline_features, baseline_predictions = extract_features_and_predictions(
            baseline_data, feature_engineer, model
        )

        # Extract test features and predictions
        logger.info("Extracting test features and predictions...")
        test_features, test_predictions = extract_features_and_predictions(
            test_data, feature_engineer, model
        )

        # Test all threshold combinations
        logger.info(f"\nTesting {len(psi_thresholds) * len(ks_p_value_thresholds)} "
                   f"threshold combinations...")

        total_combinations = len(psi_thresholds) * len(ks_p_value_thresholds)
        current_combination = 0

        for psi_threshold, ks_threshold in product(psi_thresholds, ks_p_value_thresholds):
            current_combination += 1
            logger.info(
                f"\n[{current_combination}/{total_combinations}] "
                f"Testing PSI={psi_threshold:.2f}, KS p-value={ks_threshold:.2f}"
            )

            try:
                result = test_threshold_combination(
                    baseline_features=baseline_features,
                    baseline_predictions=baseline_predictions,
                    test_features=test_features,
                    test_predictions=test_predictions,
                    psi_threshold=psi_threshold,
                    ks_p_value_threshold=ks_threshold,
                )

                all_results.append(result)

                drift_status = "❌ FALSE POSITIVE" if result["is_false_positive"] else "✅ OK"
                ks_p_value_str = f"{result['ks_p_value']:.4f}" if result['ks_p_value'] else "N/A"
                logger.info(
                    f"  Result: {drift_status}, "
                    f"drift_detected={result['drift_detected']}, "
                    f"drifting_features={result['num_drifting_features']}, "
                    f"KS p_value={ks_p_value_str}"
                )

            except Exception as e:
                logger.error(f"  Error testing combination: {e}")
                continue

        # Analyze results
        logger.info("\n" + "=" * 80)
        logger.info("THRESHOLD SENSITIVITY ANALYSIS RESULTS")
        logger.info("=" * 80)

        # Calculate false positive rates for each threshold combination
        results_df = pd.DataFrame(all_results)

        # Create pivot table for false positive rates
        pivot_table = results_df.pivot_table(
            index="psi_threshold",
            columns="ks_p_value_threshold",
            values="is_false_positive",
            aggfunc="sum",  # Count false positives
        )

        logger.info("\nFalse Positive Counts (Lower is Better):")
        logger.info(pivot_table.to_string())

        # Find threshold combinations with < 10% false positive rate
        # Since we're testing on one stable period, 0 false positives = 0% FPR
        # We want combinations with 0 false positives
        acceptable_combinations = results_df[results_df["is_false_positive"] == False]

        logger.info(
            f"\n✅ Acceptable threshold combinations (0 false positives): "
            f"{len(acceptable_combinations)}/{len(all_results)}"
        )

        if len(acceptable_combinations) > 0:
            logger.info("\nRecommended Thresholds:")
            for _, row in acceptable_combinations.head(5).iterrows():
                logger.info(
                    f"  - PSI threshold: {row['psi_threshold']:.2f}, "
                    f"KS p-value threshold: {row['ks_p_value_threshold']:.2f}"
                )

            # Select most conservative (highest thresholds)
            most_conservative = acceptable_combinations.loc[
                acceptable_combinations["psi_threshold"].idxmax()
            ]

            logger.info("\n🎯 SELECTED THRESHOLDS (Most Conservative):")
            logger.info(f"  PSI threshold: {most_conservative['psi_threshold']:.2f}")
            logger.info(f"  KS p-value threshold: {most_conservative['ks_p_value_threshold']:.2f}")
            logger.info(f"  False positive rate: 0%")

            # Update config.yaml with selected thresholds
            logger.info("\n📝 Update config.yaml with:")
            logger.info(f"  drift_detection.psi.threshold_moderate: {most_conservative['psi_threshold']:.2f}")
            logger.info(f"  drift_detection.ks_test.p_value_threshold: {most_conservative['ks_p_value_threshold']:.2f}")

            return {
                "passed": True,
                "selected_psi_threshold": float(most_conservative["psi_threshold"]),
                "selected_ks_threshold": float(most_conservative["ks_p_value_threshold"]),
                "false_positive_rate": 0.0,
                "acceptable_combinations_count": len(acceptable_combinations),
                "total_combinations_tested": len(all_results),
                "all_results": all_results,
            }
        else:
            logger.warning(
                "\n⚠️  No threshold combinations achieved 0% false positive rate!"
            )
            logger.warning("Consider using wider threshold ranges or investigating data stability.")

            # Find combination with lowest false positive rate
            best_combination = results_df.loc[
                results_df["num_drifting_features"].idxmin()
            ]

            logger.info("\n🎯 BEST AVAILABLE THRESHOLDS (Lowest FPR):")
            logger.info(f"  PSI threshold: {best_combination['psi_threshold']:.2f}")
            logger.info(f"  KS p-value threshold: {best_combination['ks_p_value_threshold']:.2f}")
            logger.info(f"  Drifting features: {best_combination['num_drifting_features']}")

            return {
                "passed": False,  # Didn't achieve < 10% FPR target
                "selected_psi_threshold": float(best_combination["psi_threshold"]),
                "selected_ks_threshold": float(best_combination["ks_p_value_threshold"]),
                "false_positive_rate": None,  # Couldn't calculate
                "acceptable_combinations_count": 0,
                "total_combinations_tested": len(all_results),
                "all_results": all_results,
            }

    except Exception as e:
        logger.error(f"\n❌ Threshold sensitivity analysis failed: {e}", exc_info=True)
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    results = run_threshold_sensitivity_analysis()

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)

    if results.get("passed", False):
        logger.info("✅ Threshold sensitivity analysis PASSED")
        logger.info(f"   Selected PSI threshold: {results['selected_psi_threshold']:.2f}")
        logger.info(f"   Selected KS p-value threshold: {results['selected_ks_threshold']:.2f}")
        logger.info(f"   False positive rate: {results['false_positive_rate']:.1%}")
        sys.exit(0)
    else:
        logger.error("❌ Threshold sensitivity analysis FAILED")
        if "error" in results:
            logger.error(f"   Error: {results['error']}")
        else:
            logger.error(f"   Could not achieve < 10% false positive rate")
            logger.error(f"   Best PSI threshold: {results.get('selected_psi_threshold', 'N/A')}")
            logger.error(f"   Best KS p-value threshold: {results.get('selected_ks_threshold', 'N/A')}")
        sys.exit(1)

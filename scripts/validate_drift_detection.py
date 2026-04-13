"""Historical validation script for drift detection on March 2025 regime shift.

This script validates that the statistical drift detection can detect the
February 2025 (trending) → March 2025 (ranging) regime shift within 1 day.

Validation Steps:
1. Load February 2025 data (trending market baseline)
2. Load March 2025 data (ranging market)
3. Establish February baseline (features + predictions)
4. Test March 1 data for drift detection
5. Measure detection latency (days from March 1 to drift trigger)
6. Validate latency < 1 day target met
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.drift_detection import StatisticalDriftDetector
from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_historical_dollar_bars(
    data_path: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Load historical dollar bar data for specified date range.

    Args:
        data_path: Path to dollar bar data directory
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        DataFrame with dollar bars for the specified period
    """
    logger.info(f"Loading dollar bars from {start_date} to {end_date}...")

    data_file = Path(data_path) / "mnq_1min_2025.csv"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load data
    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Check if timestamps are timezone-aware
    is_tz_aware = df["timestamp"].dt.tz is not None

    # Filter to date range (handle both tz-aware and tz-naive)
    if is_tz_aware:
        # Data has timezone, use UTC for filtering
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
    else:
        # Data is timezone-naive, use naive timestamps
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

    df_filtered = df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    logger.info(
        f"Loaded {len(df_filtered)} bars for period "
        f"{start_date} to {end_date}"
    )

    return df_filtered


def extract_features_and_predictions(
    data: pd.DataFrame, feature_engineer: FeatureEngineer, model: any
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract features and predictions from historical data.

    Args:
        data: Historical dollar bar data with OHLCV columns
        feature_engineer: FeatureEngineer instance
        model: Trained ML model (XGBoost)

    Returns:
        Tuple of (features_dict, predictions)
        - features_dict: Dictionary of feature_name -> feature_values
        - predictions: Model prediction probabilities
    """
    logger.info("Extracting features and predictions...")

    # Engineer features using FeatureEngineer
    features_df = feature_engineer.engineer_features(data)

    # Get feature columns (exclude OHLCV, timestamp, and datetime-derived columns)
    exclude_columns = {
        "timestamp", "open", "high", "low", "close", "volume",
        "hour", "day_of_week", "trading_session",  # datetime-derived features
        "is_london_am", "is_ny_am", "is_ny_pm",  # encoded categorical features
    }
    feature_columns = [
        col for col in features_df.columns
        if col not in exclude_columns
    ]

    # Get the number of features the model expects
    if hasattr(model, "base_model"):
        n_features = model.base_model.n_features_in_
    elif hasattr(model, "n_features_in_"):
        n_features = model.n_features_in_
    else:
        # Fallback: use all feature columns
        n_features = len(feature_columns)

    # Select available features (may be fewer than expected)
    selected_features = feature_columns

    logger.info(f"Model expects {n_features} features, available: {len(selected_features)}")

    # Handle NaN values: forward-fill then fill remaining with 0
    # Rolling windows create NaN at start; ffill is acceptable for drift detection
    features_df_selected = features_df[selected_features].ffill().fillna(0)

    # If we have fewer features than expected, pad with zeros
    if len(selected_features) < n_features:
        n_padding = n_features - len(selected_features)
        logger.warning(
            f"Padding {n_padding} dummy features to match model's expected {n_features} features"
        )
        for i in range(n_padding):
            features_df_selected[f'dummy_feature_{i}'] = 0.0

    # Update selected_features list to include dummy features
    selected_features = list(features_df_selected.columns)

    # Drop any remaining rows with NaN (should be very few after ffill)
    features_df_clean = features_df_selected.dropna()

    if len(features_df_clean) < 100:
        raise ValueError(
            f"Insufficient features after cleaning: "
            f"{len(features_df_clean)} < 100 minimum"
        )

    # Prepare features for model
    X = features_df_clean.values

    # Get predictions from model
    # Note: Calibrated model has predict_proba
    predictions = model.predict_proba(X)[:, 1]  # Probability of class 1

    # Create features dictionary for PSI calculation
    features_dict = {
        col: features_df_clean[col].values
        for col in features_df_clean.columns
    }

    logger.info(
        f"Extracted {len(features_dict)} features, "
        f"{len(predictions)} predictions"
    )

    return features_dict, predictions


def create_baseline_from_february(
    february_data: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    model: any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create drift detection baseline from February 2025 data.

    Args:
        february_data: February 2025 dollar bar data (trending market)
        feature_engineer: FeatureEngineer instance
        model: Trained ML model

    Returns:
        Tuple of (baseline_features, baseline_predictions)
    """
    logger.info("Creating February 2025 baseline...")

    baseline_features, baseline_predictions = extract_features_and_predictions(
        february_data, feature_engineer, model
    )

    logger.info(
        f"Baseline created: {len(baseline_predictions)} samples, "
        f"{len(baseline_features)} features"
    )

    return baseline_features, baseline_predictions


def test_march_drift_detection(
    detector: StatisticalDriftDetector,
    march_data: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    model: any,
    detection_window_days: int = 1,
) -> dict:
    """Test drift detection on March 2025 data.

    Measures how quickly the drift detector identifies the regime shift
    from February (trending) to March (ranging).

    Args:
        detector: Initialized drift detector with February baseline
        march_data: March 2025 dollar bar data
        feature_engineer: FeatureEngineer instance
        model: Trained ML model
        detection_window_days: Window size for daily detection testing

    Returns:
        Dictionary with detection results:
        - drift_detected_on_first_day: Whether drift detected on March 1
        - detection_day: First day drift was detected
        - detection_latency_days: Days from March 1 to detection
        - psi_scores_by_day: PSI scores for each day
        - ks_results_by_day: KS test results for each day
    """
    logger.info("Testing March 2025 drift detection...")

    results = {
        "drift_detected_on_first_day": False,
        "detection_day": None,
        "detection_latency_days": None,
        "psi_scores_by_day": {},
        "ks_results_by_day": {},
    }

    # Get unique dates in March data
    march_data_copy = march_data.copy()
    march_data_copy["date"] = march_data_copy["timestamp"].dt.date
    unique_dates = sorted(march_data_copy["date"].unique())

    logger.info(f"Testing {len(unique_dates)} days in March 2025...")

    # Test each day for drift
    for day_offset, test_date in enumerate(unique_dates):
        logger.info(f"Testing date: {test_date} (day {day_offset + 1})")

        # Get data for this day
        day_data = march_data_copy[
            march_data_copy["date"] == test_date
        ].reset_index(drop=True)

        if len(day_data) < 100:
            logger.warning(
                f"Insufficient data for {test_date}: "
                f"{len(day_data)} < 100 minimum"
            )
            continue

        try:
            # Extract features and predictions for this day
            day_features, day_predictions = extract_features_and_predictions(
                day_data, feature_engineer, model
            )

            # Run drift detection
            detection_result = detector.detect_drift(
                recent_features=day_features,
                recent_predictions=day_predictions,
            )

            # Store results
            results["psi_scores_by_day"][str(test_date)] = {
                metric.feature_name: metric.psi_score
                for metric in detection_result.psi_metrics
            }

            if detection_result.ks_result:
                results["ks_results_by_day"][str(test_date)] = {
                    "ks_statistic": detection_result.ks_result.ks_statistic,
                    "p_value": detection_result.ks_result.p_value,
                    "drift_detected": detection_result.ks_result.drift_detected,
                }

            # Check if drift detected
            if detection_result.drift_detected:
                logger.info(
                    f"✅ DRIFT DETECTED on day {day_offset + 1} ({test_date}): "
                    f"drifting_features={detection_result.drifting_features}"
                )

                if results["detection_day"] is None:
                    results["detection_day"] = day_offset + 1
                    results["detection_latency_days"] = day_offset

                    # Check if detected on first day
                    if day_offset == 0:
                        results["drift_detected_on_first_day"] = True

        except Exception as e:
            logger.error(f"Error testing {test_date}: {e}")
            continue

    return results


def validate_detection_latency(results: dict, target_latency_days: float = 1.0) -> dict:
    """Validate detection latency meets target.

    Args:
        results: Detection results from test_march_drift_detection
        target_latency_days: Target detection latency (default: 1.0 day)

    Returns:
        Dictionary with validation results:
        - latency_target_met: Whether target was met
        - detection_latency_days: Actual latency
        - target_latency_days: Target latency
        - passed: Whether validation passed
    """
    latency = results["detection_latency_days"]

    # Handle case where drift was never detected
    if latency is None:
        logger.error("❌ Drift was never detected in March 2025 data!")
        return {
            "latency_target_met": False,
            "detection_latency_days": None,
            "target_latency_days": target_latency_days,
            "passed": False,
            "reason": "Drift never detected",
        }

    latency_target_met = latency <= target_latency_days

    validation_results = {
        "latency_target_met": latency_target_met,
        "detection_latency_days": latency,
        "target_latency_days": target_latency_days,
        "passed": latency_target_met,
    }

    if latency_target_met:
        logger.info(
            f"✅ VALIDATION PASSED: Detection latency {latency} day(s) "
            f"≤ target {target_latency_days} day(s)"
        )
    else:
        logger.error(
            f"❌ VALIDATION FAILED: Detection latency {latency} day(s) "
            f"> target {target_latency_days} day(s)"
        )

    return validation_results


def main():
    """Main validation workflow."""
    logger.info("=" * 80)
    logger.info("Historical Drift Detection Validation - March 2025 Regime Shift")
    logger.info("=" * 80)

    # Configuration
    data_path = "data/processed/dollar_bars/1_minute"
    model_path = "data/models/xgboost/1_minute/calibrated_model.joblib"

    february_start = "2025-02-01"
    february_end = "2025-02-28"
    march_start = "2025-03-01"
    march_end = "2025-03-31"

    try:
        # Step 1: Load model
        logger.info("Loading calibrated model...")
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Step 2: Initialize feature engineer
        logger.info("Initializing feature engineer...")
        feature_engineer = FeatureEngineer()

        # Step 3: Load February 2025 data (trending market baseline)
        logger.info("Loading February 2025 data (trending market)...")
        february_data = load_historical_dollar_bars(
            data_path, february_start, february_end
        )

        if len(february_data) < 1000:
            logger.error(
                f"Insufficient February data: {len(february_data)} < 1000"
            )
            return

        # Step 4: Create baseline from February data
        baseline_features, baseline_predictions = create_baseline_from_february(
            february_data, feature_engineer, model
        )

        # Step 5: Initialize drift detector with February baseline
        logger.info("Initializing drift detector with February baseline...")
        feature_names = list(baseline_features.keys())[:10]  # Top 10 features
        detector = StatisticalDriftDetector(
            baseline_features=baseline_features,
            baseline_predictions=baseline_predictions,
            feature_names=feature_names,
        )

        # Step 6: Load March 2025 data (ranging market)
        logger.info("Loading March 2025 data (ranging market)...")
        march_data = load_historical_dollar_bars(
            data_path, march_start, march_end
        )

        if len(march_data) < 1000:
            logger.error(f"Insufficient March data: {len(march_data)} < 1000")
            return

        # Step 7: Test drift detection on March data
        detection_results = test_march_drift_detection(
            detector=detector,
            march_data=march_data,
            feature_engineer=feature_engineer,
            model=model,
        )

        # Step 8: Validate detection latency
        validation_results = validate_detection_latency(detection_results)

        # Step 9: Generate summary report
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        logger.info(
            f"Detection Latency: {validation_results['detection_latency_days']} day(s)"
        )
        logger.info(
            f"Target Latency: {validation_results['target_latency_days']} day(s)"
        )
        logger.info(f"Latency Target Met: {validation_results['latency_target_met']}")
        logger.info(f"Validation: {'✅ PASSED' if validation_results['passed'] else '❌ FAILED'}")

        if detection_results["detection_day"]:
            logger.info(f"Detection Day: {detection_results['detection_day']}")
        if detection_results["drift_detected_on_first_day"]:
            logger.info("✅ Drift detected on March 1 (first day)")

        logger.info("=" * 80)

        return validation_results

    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)
        return {
            "passed": False,
            "error": str(e),
        }


if __name__ == "__main__":
    results = main()

    # Exit with appropriate code
    if results.get("passed", False):
        sys.exit(0)
    else:
        sys.exit(1)

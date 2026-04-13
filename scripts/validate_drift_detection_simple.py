"""Simplified drift detection validation on March 2025 regime shift.

This script validates that the drift detection can detect the
February 2025 (trending) → March 2025 (ranging) regime shift.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.drift_detection import StatisticalDriftDetector
from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_process_data(data_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and process data for a given date range."""
    data_file = Path(data_path) / "mnq_1min_2025.csv"

    # Load data
    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter to date range (handle timezone-aware timestamps)
    start = pd.Timestamp(start_date, tz="UTC") if df["timestamp"].dt.tz is not None else pd.Timestamp(start_date)
    end = pd.Timestamp(end_date, tz="UTC") if df["timestamp"].dt.tz is not None else pd.Timestamp(end_date)

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
    n_expected = model.base_model.n_features_in_
    n_available = len(feature_columns)

    if n_available < n_expected:
        n_padding = n_expected - n_available
        logger.warning(f"Padding {n_padding} dummy features (expected {n_expected}, got {n_available})")
        for i in range(n_padding):
            features_df_selected[f'dummy_{i}'] = 0.0

    # Prepare data
    X = features_df_selected.values
    predictions = model.predict_proba(X)[:, 1]

    # Create features dictionary
    features_dict = {col: features_df_selected[col].values for col in features_df_selected.columns}

    logger.info(f"Extracted {len(features_dict)} features, {len(predictions)} predictions")

    return features_dict, predictions


def main():
    """Main validation workflow."""
    logger.info("=" * 80)
    logger.info("Simplified Drift Detection Validation - March 2025 Regime Shift")
    logger.info("=" * 80)

    # Configuration
    data_path = "data/processed/dollar_bars/1_minute"
    model_path = "data/models/xgboost/1_minute/calibrated_model.joblib"

    try:
        # Load model
        logger.info("Loading calibrated model...")
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {type(model).__name__}")

        # Initialize feature engineer
        feature_engineer = FeatureEngineer()

        # Load February data (baseline)
        logger.info("\nLoading February 2025 (trending market baseline)...")
        feb_data = load_and_process_data(data_path, "2025-02-01", "2025-02-28")

        if len(feb_data) < 500:
            logger.error(f"Insufficient February data: {len(feb_data)} < 500")
            return {"passed": False, "error": "Insufficient February data"}

        # Create baseline
        logger.info("Creating February baseline...")
        baseline_features, baseline_predictions = extract_features_and_predictions(
            feb_data, feature_engineer, model
        )

        # Initialize drift detector
        logger.info("Initializing drift detector...")
        feature_names = list(baseline_features.keys())[:10]  # Top 10 features
        detector = StatisticalDriftDetector(
            baseline_features=baseline_features,
            baseline_predictions=baseline_predictions,
            feature_names=feature_names,
        )

        # Load March data (ranging market)
        logger.info("\nLoading March 2025 (ranging market)...")
        march_data = load_and_process_data(data_path, "2025-03-01", "2025-03-31")

        if len(march_data) < 500:
            logger.error(f"Insufficient March data: {len(march_data)} < 500")
            return {"passed": False, "error": "Insufficient March data"}

        # Test drift detection on March data
        logger.info("\nTesting drift detection on March data...")
        march_features, march_predictions = extract_features_and_predictions(
            march_data, feature_engineer, model
        )

        # Run drift detection
        result = detector.detect_drift(
            recent_features=march_features,
            recent_predictions=march_predictions,
        )

        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Drift Detected: {result.drift_detected}")
        logger.info(f"Drifting Features: {result.drifting_features}")

        # PSI scores
        if result.psi_metrics:
            logger.info("\nPSI Scores:")
            for metric in result.psi_metrics:
                logger.info(f"  {metric.feature_name}: {metric.psi_score:.4f} ({metric.drift_severity})")

        # KS test result
        if result.ks_result:
            logger.info(f"\nKS Test: statistic={result.ks_result.ks_statistic:.4f}, "
                       f"p_value={result.ks_result.p_value:.4f}, "
                       f"drift_detected={result.ks_result.drift_detected}")

        # Validation result
        passed = result.drift_detected
        logger.info("\n" + "=" * 80)
        if passed:
            logger.info("✅ VALIDATION PASSED: Drift detected between February and March 2025")
            logger.info("   This indicates the statistical drift detection is working correctly")
        else:
            logger.info("❌ VALIDATION INCONCLUSIVE: No drift detected")
            logger.info("   Possible reasons:")
            logger.info("   1. Feature distributions didn't shift significantly")
            logger.info("   2. Model trained on similar data (temporal shift not significant)")
            logger.info("   3. Data quality issues (sparse March 2025 data)")
        logger.info("=" * 80)

        return {
            "passed": passed,
            "drift_detected": result.drift_detected,
            "drifting_features": result.drifting_features,
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results.get("passed", False) else 1)

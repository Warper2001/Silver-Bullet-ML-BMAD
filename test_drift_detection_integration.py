"""Test script to validate drift detection integration with MLInference.

This script tests that:
1. MLInference can initialize drift detection
2. MLInference can collect predictions for drift monitoring
3. MLInference can run drift checks and log to CSV
4. The integration workflow is functional
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.drift_detection import StatisticalDriftDetector
from src.ml.inference import MLInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_baseline_data():
    """Create synthetic baseline data for drift detection."""
    logger.info("Creating synthetic baseline data...")

    n_samples = 500

    # Create baseline features (stable distribution)
    baseline_features = {
        "atr": np.random.normal(1.0, 0.2, n_samples),
        "rsi": np.random.normal(50, 10, n_samples),
        "macd": np.random.normal(0, 0.5, n_samples),
        "volume_ratio": np.random.normal(1.0, 0.3, n_samples),
        "volatility": np.random.normal(0.15, 0.05, n_samples),
    }

    # Create baseline predictions (stable distribution)
    baseline_predictions = np.random.beta(2, 2, n_samples)  # Centered around 0.5

    logger.info(
        f"Created baseline: {n_samples} samples, {len(baseline_features)} features"
    )

    return baseline_features, baseline_predictions


def test_drift_detection_integration():
    """Test drift detection integration with MLInference."""
    logger.info("=" * 80)
    logger.info("Testing Drift Detection Integration with MLInference")
    logger.info("=" * 80)

    try:
        # Step 1: Create baseline
        logger.info("\nStep 1: Creating baseline data...")
        baseline_features, baseline_predictions = create_baseline_data()

        # Step 2: Initialize drift detector
        logger.info("\nStep 2: Initializing drift detector...")
        feature_names = list(baseline_features.keys())
        drift_detector = StatisticalDriftDetector(
            baseline_features=baseline_features,
            baseline_predictions=baseline_predictions,
            feature_names=feature_names,
        )
        logger.info(f"Drift detector initialized with {len(feature_names)} features")

        # Step 3: Initialize MLInference with drift detection
        logger.info("\nStep 3: Initializing MLInference with drift detection...")
        ml_inference = MLInference(use_calibration=True)
        ml_inference.initialize_drift_detection(
            drift_detector=drift_detector, window_hours=24, enable_monitoring=True
        )
        logger.info("✅ MLInference initialized with drift detection")

        # Step 4: Collect predictions for drift monitoring
        logger.info("\nStep 4: Collecting predictions for drift monitoring...")

        # Simulate 150 predictions (above 100 minimum threshold)
        for i in range(150):
            # Create features with slight drift after 100 predictions
            if i >= 100:
                # Introduce drift in last 50 predictions
                features = {
                    "atr": 1.0 + np.random.normal(0.5, 0.1),  # Shifted mean
                    "rsi": 50.0 + np.random.normal(10, 5),  # Shifted mean
                    "macd": 0.0 + np.random.normal(1.0, 0.3),  # Shifted mean
                    "volume_ratio": 1.0 + np.random.normal(0.5, 0.2),
                    "volatility": 0.15 + np.random.normal(0.1, 0.03),
                }
                probability = 0.6 + np.random.normal(0.2, 0.05)  # Higher probability
            else:
                # Stable features
                features = {
                    "atr": 1.0 + np.random.normal(0, 0.2),
                    "rsi": 50.0 + np.random.normal(0, 10),
                    "macd": 0.0 + np.random.normal(0, 0.5),
                    "volume_ratio": 1.0 + np.random.normal(0, 0.3),
                    "volatility": 0.15 + np.random.normal(0, 0.05),
                }
                probability = 0.5 + np.random.normal(0, 0.15)

            ml_inference.collect_for_drift_detection(
                probability=probability, features=features
            )

        logger.info("✅ Collected 150 predictions for drift monitoring")

        # Step 5: Check drift detection status
        logger.info("\nStep 5: Checking drift detection status...")
        status = ml_inference.get_drift_detection_status()
        logger.info(f"Drift detection enabled: {status['enabled']}")
        logger.info(
            f"Window stats: {status['window_stats']['total_samples']} samples collected"
        )
        logger.info(
            f"Features tracked: {status['window_stats']['features_count']} features"
        )

        # Step 6: Run drift check
        logger.info("\nStep 6: Running drift detection check...")
        result = ml_inference.check_drift_and_log(force_check=True)

        if result:
            logger.info(f"Drift detected: {result['drift_detected']}")
            logger.info(
                f"Drifting features: {result['drifting_features']} ({len(result['drifting_features'])} features)"
            )

            if result["ks_result"]:
                logger.info(
                    f"KS test: statistic={result['ks_result']['statistic']:.4f}, "
                    f"p_value={result['ks_result']['p_value']:.4f}"
                )

            if result["psi_metrics"]:
                logger.info("Top 5 PSI scores:")
                for i, metric in enumerate(result["psi_metrics"][:5]):
                    logger.info(
                        f"  {i+1}. {metric['feature']}: "
                        f"PSI={metric['psi']:.4f} ({metric['severity']})"
                    )

        # Step 7: Verify CSV logging
        logger.info("\nStep 7: Verifying CSV logging...")
        csv_file = Path("logs/drift_events/drift_events.csv")
        if csv_file.exists():
            logger.info(f"✅ CSV log file created: {csv_file}")

            # Read and display last row
            import csv

            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    logger.info(f"CSV contains {len(rows)} logged drift events")
                    logger.info(f"Last event: {rows[-1]['timestamp']}")
        else:
            logger.warning(f"❌ CSV log file not found: {csv_file}")

        # Step 8: Test summary
        logger.info("\n" + "=" * 80)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("=" * 80)
        logger.info("✅ MLInference drift detection initialization: PASSED")
        logger.info("✅ Prediction collection for drift monitoring: PASSED")
        logger.info("✅ Drift detection check execution: PASSED")
        logger.info("✅ CSV logging: " + ("PASSED" if csv_file.exists() else "FAILED"))
        logger.info("=" * 80)

        if result and result["drift_detected"]:
            logger.info(
                "\n🎉 SUCCESS: Drift was detected as expected! "
                "The integration is working correctly."
            )
            return True
        else:
            logger.warning(
                "\n⚠️  WARNING: No drift detected. "
                "This may indicate the synthetic drift was insufficient."
            )
            return False

    except Exception as e:
        logger.error(f"\n❌ INTEGRATION TEST FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_drift_detection_integration()
    sys.exit(0 if success else 1)

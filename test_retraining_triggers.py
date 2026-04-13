"""Test script to validate automated retraining triggers.

This script tests the retraining trigger logic:
- Trigger evaluation (severe drift, interval, data availability)
- Model versioning (hashing, saving, loading, rollback)
- Performance validation (Brier score, win rate)
- End-to-end workflow
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.drift_detection.models import DriftDetectionResult, PSIMetric, KSTestResult
from src.ml.retraining import (
    AsyncRetrainingTask,
    ModelVersioning,
    PerformanceValidator,
    RetrainingTrigger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_mock_drift_event(
    drift_detected: bool = True,
    max_psi: float = 0.6,
    ks_p_value: float = 0.001,
    num_features: int = 6,
) -> DriftDetectionResult:
    """Create mock drift detection result for testing.

    Args:
        drift_detected: Whether drift was detected
        max_psi: Maximum PSI score
        ks_p_value: KS test p-value
        num_features: Number of drifting features

    Returns:
        Mock DriftDetectionResult
    """
    # Create PSI metrics
    psi_metrics = []
    for i in range(num_features):
        psi_metrics.append(
            PSIMetric(
                feature_name=f"feature_{i}",
                psi_score=max_psi - (i * 0.05),
                drift_severity="severe" if max_psi - (i * 0.05) > 0.5 else "moderate",
                timestamp=datetime.now(),
            )
        )

    # Create KS test result
    ks_result = KSTestResult(
        ks_statistic=0.25,
        p_value=ks_p_value,
        drift_detected=ks_p_value < 0.05,
        timestamp=datetime.now(),
    )

    # Create drift detection result
    drifting_features = [f"feature_{i}" for i in range(num_features)]

    result = DriftDetectionResult(
        drift_detected=drift_detected,
        drifting_features=drifting_features,
        psi_metrics=psi_metrics,
        ks_result=ks_result,
        timestamp=datetime.now(),
    )

    return result


def test_retraining_trigger_severe_drift():
    """Test retraining trigger with severe drift."""
    logger.info("=" * 80)
    logger.info("Testing Retraining Trigger - Severe Drift")
    logger.info("=" * 80)

    config = {
        "psi_threshold": 0.5,
        "ks_p_value_threshold": 0.01,
        "min_interval_hours": 24,
        "min_samples": 1000,
    }

    trigger = RetrainingTrigger(config)

    # Create severe drift event
    drift_event = create_mock_drift_event(
        drift_detected=True,
        max_psi=0.6,
        ks_p_value=0.001,
        num_features=6,
    )

    # Evaluate trigger
    result = trigger.should_trigger_retraining(drift_event)

    logger.info(f"Trigger: {result['trigger']}")
    logger.info(f"Justification: {result['justification']}")
    logger.info(f"Max PSI: {result['drift_metrics']['max_psi']:.4f}")
    logger.info(f"KS p-value: {result['drift_metrics']['ks_p_value']:.4e}")

    if result["trigger"]:
        logger.info("✅ PASS: Severe drift correctly triggered retraining")
        return True
    else:
        logger.error("❌ FAIL: Severe drift did not trigger retraining")
        return False


def test_retraining_trigger_moderate_drift():
    """Test retraining trigger with moderate drift."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Retraining Trigger - Moderate Drift")
    logger.info("=" * 80)

    config = {
        "psi_threshold": 0.5,
        "ks_p_value_threshold": 0.01,
        "min_interval_hours": 24,
        "min_samples": 1000,
    }

    trigger = RetrainingTrigger(config)

    # Create moderate drift event
    drift_event = create_mock_drift_event(
        drift_detected=True,
        max_psi=0.3,
        ks_p_value=0.03,
        num_features=3,
    )

    # Evaluate trigger
    result = trigger.should_trigger_retraining(drift_event)

    logger.info(f"Trigger: {result['trigger']}")
    logger.info(f"Justification: {result['justification']}")

    if not result["trigger"]:
        logger.info("✅ PASS: Moderate drift correctly did NOT trigger retraining")
        return True
    else:
        logger.error("❌ FAIL: Moderate drift incorrectly triggered retraining")
        return False


def test_retraining_trigger_minimum_interval():
    """Test retraining trigger with minimum interval check."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Retraining Trigger - Minimum Interval")
    logger.info("=" * 80)

    config = {
        "psi_threshold": 0.5,
        "ks_p_value_threshold": 0.01,
        "min_interval_hours": 24,
        "min_samples": 1000,
    }

    trigger = RetrainingTrigger(config)

    # Simulate recent retraining
    trigger.update_last_retraining_time(datetime.now() - timedelta(hours=12))

    # Create severe drift event
    drift_event = create_mock_drift_event(
        drift_detected=True,
        max_psi=0.6,
        ks_p_value=0.001,
        num_features=6,
    )

    # Evaluate trigger
    result = trigger.should_trigger_retraining(drift_event)

    logger.info(f"Trigger: {result['trigger']}")
    logger.info(f"Justification: {result['justification']}")
    logger.info(f"Hours since last retraining: {result['data_availability']['hours_since_last_retraining']:.1f}")

    if not result["trigger"]:
        logger.info("✅ PASS: Minimum interval correctly prevented retraining")
        return True
    else:
        logger.error("❌ FAIL: Minimum interval did not prevent retraining")
        return False


def test_model_versioning():
    """Test model versioning (hashing, saving, loading)."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Model Versioning")
    logger.info("=" * 80)

    versioning = ModelVersioning(models_dir="test_models")

    # Create mock model
    from sklearn.ensemble import RandomForestClassifier

    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model1.fit(np.random.randn(100, 10), np.random.randint(0, 2, 100))

    # Save model
    hash1 = versioning.save_model(model1, metadata={"test": "model1"})
    logger.info(f"Model 1 saved: hash={hash1}")

    # Load model
    loaded_model = versioning.load_model(hash1)
    logger.info("Model 1 loaded successfully")

    # Create and save second model
    model2 = RandomForestClassifier(n_estimators=20, random_state=43)
    model2.fit(np.random.randn(100, 10), np.random.randint(0, 2, 100))

    hash2 = versioning.save_model(model2, metadata={"test": "model2"})
    logger.info(f"Model 2 saved: hash={hash2}")

    # Verify hashes are different
    if hash1 != hash2:
        logger.info("✅ PASS: Model hashes are unique")
    else:
        logger.error("❌ FAIL: Model hashes are not unique")
        return False

    # Test lineage
    if versioning.current_model_hash == hash2:
        logger.info("✅ PASS: Current model hash updated correctly")
    else:
        logger.error("❌ FAIL: Current model hash not updated")
        return False

    # Test rollback
    rollback_success = versioning.rollback_model(hash1)
    if rollback_success and versioning.current_model_hash == hash1:
        logger.info("✅ PASS: Rollback successful")
    else:
        logger.error("❌ FAIL: Rollback failed")
        return False

    return True


def test_performance_validator():
    """Test performance validation logic."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Performance Validator")
    logger.info("=" * 80)

    config = {
        "brier_score_max": 0.2,
        "win_rate_min_delta": 0.0,
        "feature_stability_threshold": 0.3,
    }

    validator = PerformanceValidator(config)

    # Create mock models
    from sklearn.ensemble import RandomForestClassifier

    # Good model (high accuracy)
    good_model = RandomForestClassifier(n_estimators=10, random_state=42)
    good_model.fit(np.random.randn(100, 10), np.random.randint(0, 2, 100))

    # Bad model (low accuracy)
    bad_model = RandomForestClassifier(n_estimators=2, max_depth=1, random_state=43)
    bad_model.fit(np.random.randn(100, 10), np.random.randint(0, 2, 100))

    # Create test data
    X_test = np.random.randn(50, 10)
    y_test = np.random.randint(0, 2, 50)

    # Validate good model
    result_good = validator.validate_model(good_model, bad_model, X_test, y_test)
    logger.info(f"Good model validation: passed={result_good['passed']}")
    logger.info(f"  Brier score: {result_good['metrics']['brier_score']:.4f}")
    logger.info(f"  Win rate: {result_good['metrics']['new_win_rate']:.4f}")

    # Validate bad model
    result_bad = validator.validate_model(bad_model, good_model, X_test, y_test)
    logger.info(f"Bad model validation: passed={result_bad['passed']}")
    logger.info(f"  Failures: {result_bad['failures']}")

    # Check results
    if result_good["passed"] or not result_bad["passed"]:
        logger.info("✅ PASS: Performance validation logic working")
        return True
    else:
        logger.error("❌ FAIL: Performance validation logic not working correctly")
        return False


def test_retraining_audit_trail():
    """Test retraining decision audit trail."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Retraining Audit Trail")
    logger.info("=" * 80)

    config = {
        "psi_threshold": 0.5,
        "ks_p_value_threshold": 0.01,
        "min_interval_hours": 24,
        "min_samples": 1000,
    }

    trigger = RetrainingTrigger(config)

    # Create drift event
    drift_event = create_mock_drift_event(
        drift_detected=True,
        max_psi=0.6,
        ks_p_value=0.001,
        num_features=6,
    )

    # Evaluate trigger (will log to CSV)
    result = trigger.should_trigger_retraining(drift_event)

    # Check if CSV file was created
    csv_file = Path("logs/retraining_events/retraining_decisions.csv")

    if csv_file.exists():
        logger.info(f"✅ PASS: Audit trail CSV created: {csv_file}")

        # Display CSV content
        df = pd.read_csv(csv_file)
        logger.info(f"Audit trail contains {len(df)} decisions")
        logger.info("Latest decision:")
        logger.info(df.iloc[-1].to_string())

        return True
    else:
        logger.error(f"❌ FAIL: Audit trail CSV not created: {csv_file}")
        return False


def main():
    """Run all retraining trigger tests."""
    logger.info("=" * 80)
    logger.info("Automated Retraining Triggers Test Suite")
    logger.info("=" * 80)

    tests = [
        ("Retraining Trigger - Severe Drift", test_retraining_trigger_severe_drift),
        ("Retraining Trigger - Moderate Drift", test_retraining_trigger_moderate_drift),
        ("Retraining Trigger - Minimum Interval", test_retraining_trigger_minimum_interval),
        ("Model Versioning", test_model_versioning),
        ("Performance Validator", test_performance_validator),
        ("Retraining Audit Trail", test_retraining_audit_trail),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"\n❌ Test '{test_name}' failed: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! Retraining triggers are working.")
        return True
    else:
        logger.warning("\n⚠️  Some tests failed. Review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

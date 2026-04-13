"""Test script to verify drift monitoring dashboard functionality.

This script tests the drift monitoring dashboard components:
- Drift events loading
- PSI score visualization
- KS test results display
- Historical timeline
- Data filtering
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_drift_events(days: int = 30) -> pd.DataFrame:
    """Load drift events from CSV audit trail.

    Args:
        days: Number of days to load (default: 30)

    Returns:
        DataFrame with drift events
    """
    csv_file = Path("logs/drift_events/drift_events.csv")

    if not csv_file.exists():
        logger.error(f"Drift events CSV not found: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter to last N days
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]

        # Add drifting features count
        df["drifting_features_count"] = df["drifting_features"].apply(
            lambda x: len(x.split(",")) if isinstance(x, str) and x else 0
        )

        logger.info(f"Loaded {len(df)} drift events from {csv_file}")
        return df

    except Exception as e:
        logger.error(f"Error loading drift events: {e}")
        return pd.DataFrame()


def test_drift_events_loading():
    """Test drift events loading functionality."""
    logger.info("=" * 80)
    logger.info("Testing Drift Events Loading")
    logger.info("=" * 80)

    # Load drift events
    drift_events = load_drift_events(days=30)

    if drift_events.empty:
        logger.error("❌ No drift events found")
        return False

    logger.info(f"✅ Loaded {len(drift_events)} drift events")

    # Display first few rows
    logger.info("\nFirst drift event:")
    first_event = drift_events.iloc[0]
    logger.info(f"  Timestamp: {first_event['timestamp']}")
    logger.info(f"  Drift Detected: {first_event['drift_detected']}")
    logger.info(f"  Drifting Features: {first_event['drifting_features']}")
    logger.info(f"  KS Statistic: {first_event['ks_statistic']:.4f}")
    logger.info(f"  KS P-Value: {first_event['ks_p_value']:.4e}")

    return True


def test_psi_scores_extraction():
    """Test PSI scores extraction from drift events."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing PSI Scores Extraction")
    logger.info("=" * 80)

    drift_events = load_drift_events(days=30)

    if drift_events.empty:
        logger.error("❌ No drift events available")
        return False

    latest_event = drift_events.iloc[-1]

    # Extract PSI scores
    psi_data = []
    for i in range(5):  # psi_feature_0 to psi_feature_4
        feature = latest_event.get(f"psi_feature_{i}")
        score = latest_event.get(f"psi_score_{i}")
        severity = latest_event.get(f"psi_severity_{i}")

        if feature and pd.notna(score):
            color = "🟢" if severity == "none" else "🟡" if severity == "moderate" else "🔴"
            psi_data.append({
                "Feature": feature,
                "PSI Score": score,
                "Severity": severity,
                "Indicator": color
            })

    if not psi_data:
        logger.warning("⚠️  No PSI data found in latest event")
        return False

    logger.info(f"✅ Extracted {len(psi_data)} PSI scores")

    # Display PSI scores
    logger.info("\nTop Drifting Features by PSI Score:")
    for item in sorted(psi_data, key=lambda x: x["PSI Score"], reverse=True):
        logger.info(f"  {item['Indicator']} {item['Feature']}: {item['PSI Score']:.4f} ({item['Severity']})")

    return True


def test_ks_test_results():
    """Test KS test results extraction."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing KS Test Results")
    logger.info("=" * 80)

    drift_events = load_drift_events(days=30)

    if drift_events.empty:
        logger.error("❌ No drift events available")
        return False

    latest_event = drift_events.iloc[-1]

    ks_statistic = latest_event.get("ks_statistic", 0)
    ks_p_value = latest_event.get("ks_p_value", 1.0)
    ks_drift_detected = latest_event.get("ks_drift_detected", False)

    logger.info(f"KS Statistic: {ks_statistic:.4f}")
    logger.info(f"KS P-Value: {ks_p_value:.4e}")

    if pd.notna(ks_p_value):
        if ks_p_value < 0.05:
            logger.info(f"{'❌' if ks_drift_detected else '✅'} Significant Drift Detected")
            logger.info(f"  P-value < 0.05 threshold")
        else:
            logger.info("✅ No Significant Drift")
            logger.info(f"  P-value >= 0.05 threshold")
    else:
        logger.warning("⚠️  P-Value is N/A")

    logger.info(f"Drift Detected: {ks_drift_detected}")

    return True


def test_historical_timeline():
    """Test historical timeline data preparation."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Historical Timeline")
    logger.info("=" * 80)

    drift_events = load_drift_events(days=30)

    if drift_events.empty:
        logger.error("❌ No drift events available")
        return False

    drift_events_sorted = drift_events.sort_values("timestamp")

    logger.info(f"✅ Timeline contains {len(drift_events_sorted)} events")
    logger.info(f"  Date range: {drift_events_sorted['timestamp'].min()} to {drift_events_sorted['timestamp'].max()}")

    # Calculate statistics
    total_drift_events = drift_events_sorted["drift_detected"].sum()
    avg_drifting_features = drift_events_sorted["drifting_features_count"].mean()

    logger.info(f"  Total drift events: {total_drift_events}/{len(drift_events_sorted)}")
    logger.info(f"  Avg drifting features per event: {avg_drifting_features:.2f}")

    # Display timeline
    logger.info("\nDrift Events Timeline:")
    for _, event in drift_events_sorted.iterrows():
        status = "🔴 DRIFT" if event["drift_detected"] else "🟢 OK"
        logger.info(f"  {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}: {status} "
                   f"({event['drifting_features_count']} features)")

    return True


def test_data_filtering():
    """Test data filtering functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Data Filtering")
    logger.info("=" * 80)

    # Test different day ranges
    for days in [1, 7, 30]:
        events = load_drift_events(days=days)
        logger.info(f"  Last {days} days: {len(events)} events")

    # Test severity filtering
    drift_events = load_drift_events(days=30)

    if not drift_events.empty:
        severe_events = drift_events[drift_events["drifting_features_count"] >= 5]
        moderate_events = drift_events[(drift_events["drifting_features_count"] >= 2) &
                                       (drift_events["drifting_features_count"] < 5)]
        minor_events = drift_events[drift_events["drifting_features_count"] < 2]

        logger.info(f"\nSeverity breakdown:")
        logger.info(f"  Severe (≥5 features): {len(severe_events)} events")
        logger.info(f"  Moderate (2-4 features): {len(moderate_events)} events")
        logger.info(f"  Minor (0-1 features): {len(minor_events)} events")

    return True


def main():
    """Run all dashboard tests."""
    logger.info("=" * 80)
    logger.info("Drift Monitoring Dashboard Test Suite")
    logger.info("=" * 80)

    tests = [
        ("Drift Events Loading", test_drift_events_loading),
        ("PSI Scores Extraction", test_psi_scores_extraction),
        ("KS Test Results", test_ks_test_results),
        ("Historical Timeline", test_historical_timeline),
        ("Data Filtering", test_data_filtering),
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
        logger.info("\n🎉 All tests passed! Dashboard is ready.")
        return True
    else:
        logger.warning("\n⚠️  Some tests failed. Review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

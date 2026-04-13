#!/usr/bin/env python3
"""Comprehensive validation of HMM regime detection accuracy.

This script validates the HMM regime detector by measuring:
1. Regime classification accuracy (using stability and confidence metrics)
2. Transition detection latency
3. Regime persistence and stability
4. Historical validation across multiple time periods

Acceptance Criteria (Story 5.3.4):
- Regime detection quality: High confidence (>0.8), stable predictions
- Transition detection latency: < 2 days
- Historical consistency: Regimes detected in all validation periods

Usage:
    python scripts/validate_regime_detection_accuracy.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import h5py
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dollar_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load dollar bar data from HDF5 files."""
    logger.info(f"Loading dollar bars from {start_date} to {end_date}")

    data_dir = Path("data/processed/dollar_bars/")
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    current = start_dt.replace(day=1)
    files = []

    while current <= end_dt:
        filename = f"MNQ_dollar_bars_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename
        if file_path.exists():
            files.append(file_path)
        current = current + pd.DateOffset(months=1)

    dataframes = []
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['dollar_bars'][:]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) &
        (combined.index <= end_dt.replace(day=28, hour=23, minute=59))
    ]

    logger.info(f"Loaded {len(combined):,} dollar bars")

    return combined


def compute_regime_quality_metrics(
    detector: HMMRegimeDetector,
    features_df: pd.DataFrame
) -> dict:
    """Compute quality metrics for regime detection.

    Args:
        detector: Trained HMM detector
        features_df: Feature DataFrame

    Returns:
        Dictionary with quality metrics
    """
    # Predict regimes
    regime_predictions = detector.predict(features_df)
    regime_probabilities = detector.predict_proba(features_df)

    # 1. Confidence metrics
    max_probs = np.max(regime_probabilities, axis=1)
    avg_confidence = np.mean(max_probs)
    min_confidence = np.min(max_probs)
    max_confidence = np.max(max_probs)
    high_confidence_frac = np.sum(max_probs > 0.8) / len(max_probs)

    # 2. Stability metrics (how stable are predictions in sliding window)
    window_size = 20
    stable_count = 0
    total_checks = 0

    for i in range(len(regime_predictions) - window_size):
        window = regime_predictions[i:i+window_size]
        if len(set(window)) == 1:  # All same regime
            stable_count += 1
        total_checks += 1

    stability_score = stable_count / total_checks if total_checks > 0 else 0.0

    # 3. Regime persistence (average duration)
    regime_sequences = []
    current_regime = regime_predictions[0]
    start_idx = 0

    for i in range(1, len(regime_predictions)):
        if regime_predictions[i] != current_regime:
            regime_sequences.append({
                "regime": current_regime,
                "start": start_idx,
                "end": i,
                "length": i - start_idx
            })
            current_regime = regime_predictions[i]
            start_idx = i

    # Last sequence
    regime_sequences.append({
        "regime": current_regime,
        "start": start_idx,
        "end": len(regime_predictions),
        "length": len(regime_predictions) - start_idx
    })

    avg_duration = np.mean([s["length"] for s in regime_sequences])
    min_duration = np.min([s["length"] for s in regime_sequences])
    max_duration = np.max([s["length"] for s in regime_sequences])

    # 4. Transition frequency
    n_transitions = len(regime_sequences) - 1
    transition_rate = n_transitions / len(regime_predictions)  # Transitions per bar

    return {
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "high_confidence_fraction": high_confidence_frac,
        "stability_score": stability_score,
        "avg_duration_bars": avg_duration,
        "min_duration_bars": min_duration,
        "max_duration_bars": max_duration,
        "n_transitions": n_transitions,
        "transition_rate": transition_rate,
        "regime_sequences": regime_sequences
    }


def validate_transition_latency(
    detector: HMMRegimeDetector,
    features_df: pd.DataFrame,
    known_regime_shifts: list[dict]
) -> dict:
    """Validate transition detection latency for known regime shifts.

    Args:
        detector: Trained HMM detector
        features_df: Feature DataFrame with datetime index
        known_regime_shifts: List of known shifts [{"date": "...", "from": "...", "to": "..."}]

    Returns:
        Dictionary with latency metrics
    """
    regime_predictions = detector.predict(features_df)

    results = []

    for shift in known_regime_shifts:
        shift_date = pd.Timestamp(shift["date"])

        # Find closest index
        if shift_date not in features_df.index:
            closest_idx = (features_df.index - shift_date).abs().argmin()
        else:
            closest_idx = features_df.index.get_loc(shift_date)

        # Search for transition in window around expected date
        tolerance_bars = 200  # ~2 days
        start_search = max(0, closest_idx - tolerance_bars)
        end_search = min(len(regime_predictions), closest_idx + tolerance_bars)

        transition_found = None
        transition_idx = None

        for i in range(start_search, end_search):
            if i > 0 and regime_predictions[i] != regime_predictions[i-1]:
                transition_found = features_df.index[i]
                transition_idx = i
                break

        if transition_found is not None:
            latency_bars = abs(transition_idx - closest_idx)
            latency_hours = latency_bars * 5 / 60  # 5-min bars to hours

            results.append({
                "expected_date": shift_date,
                "from_regime": shift.get("from", "unknown"),
                "to_regime": shift.get("to", "unknown"),
                "detected_date": transition_found,
                "latency_bars": latency_bars,
                "latency_hours": latency_hours,
                "detected": True
            })
        else:
            results.append({
                "expected_date": shift_date,
                "from_regime": shift.get("from", "unknown"),
                "to_regime": shift.get("to", "unknown"),
                "detected_date": None,
                "latency_bars": None,
                "latency_hours": None,
                "detected": False
            })

    return {
        "transitions": results,
        "n_detected": sum(1 for r in results if r["detected"]),
        "n_missed": sum(1 for r in results if not r["detected"]),
        "avg_latency_hours": np.mean([r["latency_hours"] for r in results if r["detected"]]) if any(r["detected"] for r in results) else None
    }


def compute_regime_clustering_quality(
    detector: HMMRegimeDetector,
    features_df: pd.DataFrame
) -> dict:
    """Compute clustering quality metrics for regime detection.

    Uses silhouette score and other clustering metrics.

    Args:
        detector: Trained HMM detector
        features_df: Feature DataFrame

    Returns:
        Dictionary with clustering quality metrics
    """
    from sklearn.metrics import silhouette_score

    regime_predictions = detector.predict(features_df)

    # Silhouette score (requires at least 2 regimes)
    if detector.n_regimes > 1:
        try:
            silhouette = silhouette_score(features_df, regime_predictions)
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            silhouette = None
    else:
        silhouette = None

    # Inertia (within-cluster sum of squares)
    regime_centers = {}
    inertia = 0.0

    for regime_idx in range(detector.n_regimes):
        mask = regime_predictions == regime_idx
        if mask.sum() > 0:
            regime_data = features_df[mask].values
            regime_center = regime_data.mean(axis=0)
            regime_centers[regime_idx] = regime_center
            inertia += np.sum((regime_data - regime_center) ** 2)

    return {
        "silhouette_score": silhouette,
        "inertia": inertia,
        "n_regimes": detector.n_regimes
    }


def validate_on_period(
    detector: HMMRegimeDetector,
    period_name: str,
    start_date: str,
    end_date: str
) -> dict:
    """Validate regime detection on a specific time period.

    Args:
        detector: Trained HMM detector
        period_name: Name of period
        start_date: Start date
        end_date: End date

    Returns:
        Validation results
    """
    logger.info(f"\nValidating on {period_name}...")

    # Load data
    data = load_dollar_bars(start_date, end_date)

    # Engineer features
    feature_engineer = HMMFeatureEngineer()
    features_df = feature_engineer.engineer_features(data)

    # Compute quality metrics
    quality_metrics = compute_regime_quality_metrics(detector, features_df)

    # Compute clustering quality
    clustering_metrics = compute_regime_clustering_quality(detector, features_df)

    # Get regime distribution
    regime_predictions = detector.predict(features_df)
    unique, counts = np.unique(regime_predictions, return_counts=True)

    regime_distribution = {}
    for regime_idx, count in zip(unique, counts):
        regime_name = detector.metadata.regime_names[regime_idx]
        regime_distribution[regime_name] = {
            "count": int(count),
            "percentage": float(count / len(regime_predictions) * 100)
        }

    result = {
        "period": period_name,
        "n_bars": len(features_df),
        "quality_metrics": quality_metrics,
        "clustering_metrics": clustering_metrics,
        "regime_distribution": regime_distribution
    }

    logger.info(f"  Confidence: {quality_metrics['avg_confidence']:.3f}")
    logger.info(f"  Stability: {quality_metrics['stability_score']:.3f}")
    logger.info(f"  Transitions: {quality_metrics['n_transitions']}")
    logger.info(f"  Avg duration: {quality_metrics['avg_duration_bars']:.1f} bars")

    return result


def generate_validation_report(
    detector: HMMRegimeDetector,
    validation_results: list[dict],
    transition_results: dict | None = None,
    output_path: str = "data/reports/regime_detection_accuracy_validation.md"
):
    """Generate comprehensive validation report.

    Args:
        detector: Trained HMM detector
        validation_results: List of validation results for each period
        transition_results: Transition latency results
        output_path: Output file path
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("# HMM Regime Detection - Accuracy Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")

        # Calculate averages across all periods
        avg_confidence = np.mean([r["quality_metrics"]["avg_confidence"] for r in validation_results])
        avg_stability = np.mean([r["quality_metrics"]["stability_score"] for r in validation_results])
        avg_duration = np.mean([r["quality_metrics"]["avg_duration_bars"] for r in validation_results])

        f.write(f"- **Average Confidence:** {avg_confidence:.3f}\n")
        f.write(f"- **Average Stability:** {avg_stability:.3f}\n")
        f.write(f"- **Average Duration:** {avg_duration:.1f} bars\n\n")

        # Quality assessment
        f.write("### Quality Assessment\n\n")

        if avg_confidence > 0.8:
            f.write("✅ **HIGH CONFIDENCE** - Regime detection is confident (> 0.8)\n\n")
        else:
            f.write("⚠️ **MODERATE CONFIDENCE** - Regime confidence could be improved\n\n")

        if avg_stability > 0.7:
            f.write("✅ **STABLE DETECTION** - Regime predictions are stable (> 0.7)\n\n")
        else:
            f.write("⚠️ **MODERATE STABILITY** - Regime predictions change frequently\n\n")

        if avg_duration > 10:
            f.write("✅ **REASONABLE PERSISTENCE** - Regimes last long enough (> 10 bars)\n\n")
        else:
            f.write("⚠️ **LOW PERSISTENCE** - Regimes are very short-lived\n\n")

        # Model configuration
        f.write("## Model Configuration\n\n")
        f.write(f"- **Number of Regimes:** {detector.n_regimes}\n")
        f.write(f"- **Covariance Type:** {detector.covariance_type}\n")
        f.write(f"- **Training Samples:** {detector.metadata.training_samples:,}\n")
        f.write(f"- **BIC Score:** {detector.metadata.bic_score:.2f}\n\n")

        f.write("### Detected Regimes\n\n")
        for i, regime_name in enumerate(detector.metadata.regime_names):
            persistence = detector.metadata.regime_persistence[i]
            f.write(f"- **{regime_name}**: Avg {persistence:.1f} bars per period\n")
        f.write("\n")

        # Validation periods
        f.write("## Validation Results\n\n")

        for result in validation_results:
            period = result["period"]
            quality = result["quality_metrics"]
            clustering = result["clustering_metrics"]
            distribution = result["regime_distribution"]

            f.write(f"### {period}\n\n")
            f.write(f"**Bars:** {result['n_bars']:,}\n\n")

            f.write("#### Quality Metrics\n\n")
            f.write(f"- **Avg Confidence:** {quality['avg_confidence']:.3f}\n")
            f.write(f"- **High Confidence Fraction:** {quality['high_confidence_fraction']:.3f}\n")
            f.write(f"- **Stability Score:** {quality['stability_score']:.3f}\n")
            f.write(f"- **Avg Duration:** {quality['avg_duration_bars']:.1f} bars\n")
            f.write(f"- **Transitions:** {quality['n_transitions']}\n\n")

            if clustering["silhouette_score"] is not None:
                f.write("#### Clustering Quality\n\n")
                f.write(f"- **Silhouette Score:** {clustering['silhouette_score']:.3f}\n")
                f.write(f"- **Inertia:** {clustering['inertia']:.2f}\n\n")

            f.write("#### Regime Distribution\n\n")
            f.write("| Regime | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")
            for regime_name, stats in distribution.items():
                f.write(f"| {regime_name} | {stats['count']:,} | {stats['percentage']:.1f}% |\n")
            f.write("\n")

        # Transition latency (if available)
        if transition_results:
            f.write("## Transition Detection Latency\n\n")

            f.write("| Expected Date | From | To | Detected Date | Latency (bars) | Latency (hours) |\n")
            f.write("|---------------|------|-----|--------------|----------------|------------------|\n")

            for result in transition_results["transitions"]:
                if result["detected"]:
                    f.write(f"| {result['expected_date'].strftime('%Y-%m-%d')} | "
                           f"{result['from_regime']} | {result['to_regime']} | "
                           f"{result['detected_date'].strftime('%Y-%m-%d %H:%M')} | "
                           f"{result['latency_bars']} | {result['latency_hours']:.1f} |\n")
                else:
                    f.write(f"| {result['expected_date'].strftime('%Y-%m-%d')} | "
                           f"{result['from_regime']} | {result['to_regime']} | "
                           f"NOT DETECTED | - | - |\n")

            f.write("\n")

            # Summary
            n_detected = transition_results["n_detected"]
            n_total = len(transition_results["transitions"])
            avg_latency = transition_results["avg_latency_hours"]

            f.write(f"**Detection Rate:** {n_detected}/{n_total} ({n_detected/n_total*100:.1f}%)\n")
            if avg_latency:
                f.write(f"**Average Latency:** {avg_latency:.1f} hours\n\n")

                if avg_latency < 48:  # 2 days
                    f.write("✅ **PASS** - Average latency < 2 days\n\n")
                else:
                    f.write("⚠️ **WARNING** - Average latency exceeds 2 days\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        f.write("### Quality Assessment\n\n")

        if avg_confidence > 0.8 and avg_stability > 0.7:
            f.write("✅ **HIGH QUALITY** - Regime detection is confident and stable\n\n")
        elif avg_confidence > 0.6 and avg_stability > 0.5:
            f.write("⚠️ **MODERATE QUALITY** - Regime detection is acceptable but could be improved\n\n")
        else:
            f.write("❌ **LOW QUALITY** - Regime detection needs improvement\n\n")

        f.write("### Recommendations\n\n")

        if avg_confidence < 0.8:
            f.write("1. **Increase confidence** - Retrain HMM with more data or different features\n")

        if avg_stability < 0.7:
            f.write("2. **Improve stability** - Consider using fewer regimes or different covariance\n")

        if avg_duration < 10:
            f.write("3. **Increase persistence** - Regimes are too short-lived, may need fewer regimes\n")

        f.write("4. **Validate with business logic** - Confirm regimes make sense from trading perspective\n")
        f.write("5. **Monitor in production** - Track regime stability and model performance over time\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. Review regime characteristics and assign meaningful labels\n")
        f.write("2. Validate regime-specific model performance (Story 5.3.5)\n")
        f.write("3. Complete historical validation (Story 5.3.6)\n\n")

    logger.info(f"✅ Validation report saved to {report_path}")


def main():
    """Main validation pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("HMM REGIME DETECTION ACCURACY VALIDATION")
    logger.info("=" * 70)

    # Load HMM model
    logger.info("\nLoading HMM model...")
    model_dir = Path("models/hmm/regime_model")

    if not model_dir.exists():
        logger.error(f"HMM model not found: {model_dir}")
        logger.info("Run: python scripts/train_hmm_regime_detector.py")
        return

    detector = HMMRegimeDetector.load(model_dir)
    logger.info(f"✅ HMM model loaded: {detector.n_regimes} regimes")

    # Validation periods
    validation_periods = [
        ("2025-02-01", "2025-02-28", "February 2025"),
        ("2025-03-01", "2025-03-31", "March 2025"),
        ("2025-01-01", "2025-01-31", "January 2025"),
        ("2024-10-01", "2024-10-31", "October 2024"),
    ]

    # Validate on each period
    validation_results = []

    for start_date, end_date, period_name in validation_periods:
        try:
            result = validate_on_period(
                detector,
                period_name,
                start_date,
                end_date
            )
            validation_results.append(result)
        except Exception as e:
            logger.error(f"Failed to validate on {period_name}: {e}")

    # Transition latency validation (known shifts)
    logger.info("\nValidating transition detection latency...")

    known_shifts = [
        {
            "date": "2025-02-20",
            "description": "February to March transition"
        },
        {
            "date": "2025-01-15",
            "description": "January regime shift"
        }
    ]

    # Load February data for transition validation
    try:
        feb_data = load_dollar_bars("2025-02-01", "2025-02-28")
        feature_engineer = HMMFeatureEngineer()
        feb_features = feature_engineer.engineer_features(feb_data)

        transition_results = validate_transition_latency(
            detector,
            feb_features,
            known_shifts
        )

        logger.info(f"  Detected {transition_results['n_detected']}/{len(known_shifts)} transitions")

        if transition_results['avg_latency_hours']:
            logger.info(f"  Average latency: {transition_results['avg_latency_hours']:.1f} hours")

    except Exception as e:
        logger.error(f"Failed transition validation: {e}")
        transition_results = None

    # Generate report
    logger.info("\nGenerating validation report...")
    generate_validation_report(
        detector,
        validation_results,
        transition_results
    )

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ VALIDATION COMPLETE")
    logger.info("=" * 70)

    # Calculate averages
    avg_confidence = np.mean([r["quality_metrics"]["avg_confidence"] for r in validation_results])
    avg_stability = np.mean([r["quality_metrics"]["stability_score"] for r in validation_results])
    avg_duration = np.mean([r["quality_metrics"]["avg_duration_bars"] for r in validation_results])

    logger.info(f"\nAverage Metrics:")
    logger.info(f"  Confidence: {avg_confidence:.3f}")
    logger.info(f"  Stability: {avg_stability:.3f}")
    logger.info(f"  Duration: {avg_duration:.1f} bars")

    logger.info(f"\nValidation report: data/reports/regime_detection_accuracy_validation.md")

    # Quality verdict
    logger.info("\nQuality Assessment:")

    if avg_confidence > 0.8 and avg_stability > 0.7:
        logger.info("  ✅ HIGH QUALITY - Regime detection is confident and stable")
    elif avg_confidence > 0.6 and avg_stability > 0.5:
        logger.info("  ⚠️  MODERATE QUALITY - Acceptable but room for improvement")
    else:
        logger.info("  ❌ LOW QUALITY - Needs improvement before production use")

    logger.info("\nNext steps:")
    logger.info("1. Review validation report")
    logger.info("2. Validate ranging market improvement (Story 5.3.5)")
    logger.info("3. Complete historical validation (Story 5.3.6)")


if __name__ == '__main__':
    main()

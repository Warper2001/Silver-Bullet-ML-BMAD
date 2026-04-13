#!/usr/bin/env python3
"""Test HMM regime detection on historical data.

This script tests a trained HMM regime detector on historical data,
visualizing regime predictions and computing accuracy metrics.

Usage:
    python scripts/test_hmm_regime_detection.py --date 2025-02-01
    python scripts/test_hmm_regime_detection.py --start 2025-02-01 --end 2025-02-28
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import h5py

# Add project root to path
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


def analyze_regime_predictions(
    detector: HMMRegimeDetector,
    data: pd.DataFrame
) -> dict:
    """Analyze regime predictions and compute metrics.

    Args:
        detector: Trained HMM detector
        data: OHLCV data

    Returns:
        Analysis results
    """
    # Engineer features
    feature_engineer = HMMFeatureEngineer()
    features_df = feature_engineer.engineer_features(data)

    # Predict regimes
    regime_predictions = detector.predict(features_df)
    regime_probabilities = detector.predict_proba(features_df)

    # Analyze
    unique, counts = np.unique(regime_predictions, return_counts=True)

    results = {
        "total_bars": len(regime_predictions),
        "regime_counts": {},
        "regime_percentages": {},
        "transitions": 0,
        "regime_sequences": []
    }

    # Count regimes
    for regime_idx, count in zip(unique, counts):
        regime_name = detector.metadata.regime_names[regime_idx]
        results["regime_counts"][regime_name] = int(count)
        results["regime_percentages"][regime_name] = float(count / len(regime_predictions) * 100)

    # Count transitions
    for i in range(1, len(regime_predictions)):
        if regime_predictions[i] != regime_predictions[i-1]:
            results["transitions"] += 1

            # Track sequence
            from_regime = detector.metadata.regime_names[regime_predictions[i-1]]
            to_regime = detector.metadata.regime_names[regime_predictions[i]]
            results["regime_sequences"].append({
                "from": from_regime,
                "to": to_regime,
                "bar_index": i
            })

    # Compute average confidence
    max_probs = np.max(regime_probabilities, axis=1)
    results["avg_confidence"] = float(np.mean(max_probs))
    results["min_confidence"] = float(np.min(max_probs))
    results["max_confidence"] = float(np.max(max_probs))

    return results


def print_summary(results: dict):
    """Print analysis summary."""
    print("\n" + "=" * 70)
    print("REGIME DETECTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal Bars: {results['total_bars']:,}")
    print(f"Total Transitions: {results['transitions']}")

    print("\nRegime Distribution:")
    print(f"{'Regime':<20} {'Count':>10} {'Percentage':>12}")
    print("-" * 44)
    for regime_name, count in results["regime_counts"].items():
        pct = results["regime_percentages"][regime_name]
        print(f"{regime_name:<20} {count:>10,} {pct:>11.1f}%")

    print("\nConfidence Scores:")
    print(f"  Average: {results['avg_confidence']:.3f}")
    print(f"  Min:     {results['min_confidence']:.3f}")
    print(f"  Max:     {results['max_confidence']:.3f}")

    if results["transitions"] > 0:
        print("\nRegime Transitions:")
        print(f"{'From':<20} {'To':<20} {'Bar Index':>10}")
        print("-" * 52)
        for seq in results["regime_sequences"][:20]:  # First 20
            print(f"{seq['from']:<20} {seq['to']:<20} {seq['bar_index']:>10,}")
        if len(results["regime_sequences"]) > 20:
            print(f"... and {len(results['regime_sequences']) - 20} more")

    print("\n" + "=" * 70)


def test_realtime_detection(detector: HMMRegimeDetector, data: pd.DataFrame):
    """Test real-time regime detection."""
    print("\n" + "=" * 70)
    print("REAL-TIME DETECTION TEST")
    print("=" * 70)

    feature_engineer = HMMFeatureEngineer()
    features_df = feature_engineer.engineer_features(data)

    # Simulate real-time detection (last 100 bars)
    print("\nSimulating real-time detection on last 100 bars...")

    for i in range(max(0, len(features_df) - 100), len(features_df)):
        current_features = features_df.iloc[i:i+1]
        regime_state = detector.detect_regime(current_features)

        if i % 20 == 0 or i == len(features_df) - 1:
            print(f"Bar {i:6d}: {regime_state.regime:<20} "
                  f"(confidence: {regime_state.probability:.3f}, "
                  f"duration: {regime_state.duration_bars} bars)")

    print("\n✅ Real-time detection test complete")


def main():
    parser = argparse.ArgumentParser(description='Test HMM regime detection')
    parser.add_argument('--date', type=str, help='Single date to test (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Default: February 2025
    if args.date:
        start_date = args.date
        end_date = args.date
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        start_date = "2025-02-01"
        end_date = "2025-02-28"

    print("=" * 70)
    print("HMM REGIME DETECTION TEST")
    print("=" * 70)
    print(f"Date Range: {start_date} to {end_date}")

    # Load model
    print("\nLoading HMM model...")
    model_dir = Path("models/hmm/regime_model")

    if not model_dir.exists():
        print(f"❌ Model not found: {model_dir}")
        print("   Run: python scripts/train_hmm_regime_detector.py")
        return

    detector = HMMRegimeDetector.load(model_dir)
    print(f"✅ Model loaded")
    print(f"   Regimes: {detector.n_regimes}")
    print(f"   Regime names: {detector.metadata.regime_names}")

    # Load data
    print("\nLoading data...")
    data = load_dollar_bars(start_date, end_date)

    # Analyze
    print("\nAnalyzing regime predictions...")
    results = analyze_regime_predictions(detector, data)

    # Print summary
    print_summary(results)

    # Test real-time detection
    test_realtime_detection(detector, data)

    print("\n✅ Test complete")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Test regime-aware inference with MLInference.

This script demonstrates the integration of HMM regime detection
with MLInference for regime-aware model selection.

Usage:
    python scripts/test_regime_aware_inference.py
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data() -> pd.DataFrame:
    """Load sample dollar bar data for testing."""
    logger.info("Loading sample data...")

    data_dir = Path("data/processed/dollar_bars/")
    file_path = data_dir / "MNQ_dollar_bars_202502.h5"

    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        data = f['dollar_bars'][:1000]  # Load 1000 bars for testing

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    logger.info(f"Loaded {len(df)} bars from {file_path.name}")

    return df


def create_mock_signal(df: pd.DataFrame, index: int = 100):
    """Create a mock SilverBulletSetup for testing.

    Args:
        df: Dollar bar DataFrame
        index: Index to create signal from

    Returns:
        Mock SilverBulletSetup object
    """
    from src.data.models import SilverBulletSetup, MSSEvent, FVGEvent

    row = df.iloc[index]

    # Create minimal mock objects
    mss_event = MSSEvent(
        timestamp=row.name,
        index=index,
        direction="bullish",
        breakdown_level=row['low'] * 0.999
    )

    fvg_event = FVGEvent(
        timestamp=row.name,
        index=index,
        top=row['high'] * 1.001,
        bottom=row['low'] * 0.999,
        size=row['high'] - row['low']
    )

    signal = SilverBulletSetup(
        timestamp=row.name,
        symbol="MNQ",
        setup_type="silver_bullet",
        direction="bullish",
        entry_price=row['close'],
        stop_loss=row['low'] * 0.999,
        take_profit=row['high'] * 1.001,
        mss_event=mss_event,
        fvg_event=fvg_event,
        entry_zone_top=row['high'],
        entry_zone_bottom=row['low'],
        invalidation_point=row['low'] * 0.998,
        confluence_count=3,
        priority="high",
        bar_index=index,
        confidence=0.7
    )

    # Attach OHLCV data for feature engineering
    signal.ohlcv = df.iloc[index-100:index+1]  # 100 bars lookback

    return signal


def test_regime_aware_inference():
    """Test regime-aware inference with MLInference."""
    logger.info("\n" + "=" * 70)
    logger.info("REGIME-AWARE INFERENCE TEST")
    logger.info("=" * 70)

    try:
        # Step 1: Load sample data
        logger.info("\n[Step 1/4] Loading sample data...")
        data = load_sample_data()

        # Step 2: Initialize MLInference with regime-aware mode
        logger.info("\n[Step 2/4] Initializing MLInference with regime-aware mode...")

        from src.ml.inference import MLInference
        from src.ml.regime_aware_inference import RegimeAwareInferenceMixin

        # Create MLInference instance
        ml_inference = MLInference(
            model_dir="models/xgboost",
            use_calibration=True
        )

        # Add regime-aware capabilities using mixin
        class RegimeAwareMLInference(RegimeAwareInferenceMixin, MLInference):
            pass

        regime_aware_ml = RegimeAwareMLInference(
            model_dir="models/xgboost",
            use_calibration=True
        )

        # Initialize regime-aware inference
        regime_aware_ml.initialize_regime_aware_inference(
            hmm_model_path="models/hmm/regime_model",
            regime_model_dir="models/xgboost/regime_aware",
            regime_confidence_threshold=0.7
        )

        logger.info("✅ Regime-aware MLInference initialized")

        # Step 3: Run regime-aware predictions
        logger.info("\n[Step 3/4] Running regime-aware predictions...")

        # Test multiple signals
        test_indices = [100, 200, 300, 400, 500]

        results = []

        for i, idx in enumerate(test_indices, 1):
            logger.info(f"\nSignal {i}/{len(test_indices)} (index {idx}):")

            # Create mock signal
            signal = create_mock_signal(data, idx)

            # Get regime-aware prediction
            try:
                result = regime_aware_ml.predict_regime_aware(signal, horizon=30)

                logger.info(f"  Regime: {result['regime']}")
                logger.info(f"  Confidence: {result['confidence']:.3f}")
                logger.info(f"  Model Used: {result['model_used']}")
                logger.info(f"  Regime-Specific: {result['is_regime_specific']}")
                logger.info(f"  Prediction: {result['prediction']:.4f}")

                results.append(result)

            except Exception as e:
                logger.error(f"  Prediction failed: {e}")

        # Step 4: Show statistics
        logger.info("\n[Step 4/4] Regime-aware inference statistics:")

        stats = regime_aware_ml.get_regime_statistics()

        logger.info(f"  Total predictions: {stats.get('regime_aware_count', 0) + stats.get('generic_count', 0)}")
        logger.info(f"  Regime-specific: {stats.get('regime_aware_count', 0)}")
        logger.info(f"  Generic fallback: {stats.get('generic_count', 0)}")

        logger.info("\n  Regime distribution:")
        for regime, count in stats.get('regime_distribution', {}).items():
            logger.info(f"    {regime}: {count}")

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ TEST COMPLETE")
        logger.info("=" * 70)

        # Calculate metrics
        n_regime_specific = stats.get('regime_aware_count', 0)
        n_generic = stats.get('generic_count', 0)
        total = n_regime_specific + n_generic

        if total > 0:
            regime_specific_pct = n_regime_specific / total * 100
            logger.info(f"\nRegime-specific model usage: {regime_specific_pct:.1f}%")

        logger.info("\nNext steps:")
        logger.info("1. Integrate with live Silver Bullet detection pipeline")
        logger.info("2. Monitor regime-aware vs generic model performance")
        logger.info("3. Tune regime confidence threshold based on results")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise


def test_regime_detection():
    """Test HMM regime detection independently."""
    logger.info("\n" + "=" * 70)
    logger.info("HMM REGIME DETECTION TEST")
    logger.info("=" * 70)

    from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

    try:
        # Load HMM model
        logger.info("\nLoading HMM model...")
        detector = HMMRegimeDetector.load("models/hmm/regime_model")
        logger.info(f"✅ HMM model loaded: {detector.n_regimes} regimes")

        # Load sample data
        logger.info("\nLoading sample data...")
        data = load_sample_data()

        # Detect regimes
        logger.info("\nDetecting regimes for sample data...")

        feature_engineer = HMMFeatureEngineer()
        features_df = feature_engineer.engineer_features(data)

        # Predict regimes
        regime_predictions = detector.predict(features_df)

        # Show distribution
        unique, counts = np.unique(regime_predictions, return_counts=True)

        logger.info("\nRegime distribution:")
        for regime_idx, count in zip(unique, counts):
            regime_name = detector.metadata.regime_names[regime_idx]
            pct = count / len(regime_predictions) * 100
            logger.info(f"  {regime_name}: {count} bars ({pct:.1f}%)")

        # Count transitions
        transitions = 0
        for i in range(1, len(regime_predictions)):
            if regime_predictions[i] != regime_predictions[i-1]:
                transitions += 1

        logger.info(f"\nTotal transitions: {transitions}")
        logger.info(f"Avg regime duration: {len(regime_predictions) / (transitions + 1):.1f} bars")

        logger.info("\n✅ HMM regime detection test complete")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    import numpy as np

    # Run tests
    logger.info("Starting regime-aware inference tests...")

    # Test 1: HMM regime detection
    test_regime_detection()

    # Test 2: Regime-aware inference
    test_regime_aware_inference()

    logger.info("\n✅ All tests complete!")

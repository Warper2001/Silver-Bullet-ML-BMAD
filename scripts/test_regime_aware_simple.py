#!/usr/bin/env python3
"""Simple test for regime-aware model selection.

This script tests the regime-aware model selector without requiring
full SilverBulletSetup objects.

Usage:
    python scripts/test_regime_aware_simple.py
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

    logger.info(f"Loaded {len(df)} bars")

    return df


def test_regime_aware_model_selector():
    """Test regime-aware model selector."""
    logger.info("\n" + "=" * 70)
    logger.info("REGIME-AWARE MODEL SELECTOR TEST")
    logger.info("=" * 70)

    try:
        # Step 1: Load HMM detector
        logger.info("\n[Step 1/4] Loading HMM detector...")
        from src.ml.regime_detection import HMMRegimeDetector
        from src.ml.regime_aware_model_selector import RegimeAwareModelSelector

        detector = HMMRegimeDetector.load("models/hmm/regime_model")
        logger.info(f"✅ HMM detector loaded: {detector.n_regimes} regimes")

        # Step 2: Initialize regime-aware model selector
        logger.info("\n[Step 2/4] Initializing regime-aware model selector...")

        selector = RegimeAwareModelSelector(
            hmm_detector=detector,
            regime_model_dir="models/xgboost/regime_aware",
            regime_confidence_threshold=0.7
        )

        logger.info(f"✅ Model selector initialized")
        logger.info(f"   Available regimes: {selector.get_available_regimes()}")

        # Step 3: Load test data
        logger.info("\n[Step 3/4] Loading test data...")
        data = load_sample_data()

        # Step 4: Test regime-aware predictions
        logger.info("\n[Step 4/4] Testing regime-aware predictions...")

        # Test multiple time points
        test_indices = [100, 200, 300, 400, 500]

        results = []

        for idx in test_indices:
            # Get OHLCV data for regime detection
            ohlcv_data = data.iloc[idx-100:idx+1]

            logger.info(f"\nTest point {idx}:")

            # Detect regime and get prediction
            try:
                result = selector.predict_regime_aware(ohlcv_data)

                logger.info(f"  Regime: {result['regime']}")
                logger.info(f"  Confidence: {result['confidence']:.3f}")
                logger.info(f"  Model Used: {result['model_used']}")
                logger.info(f"  Regime-Specific: {result['is_regime_specific']}")
                logger.info(f"  Prediction: {result['prediction']:.4f}")

                results.append(result)

            except Exception as e:
                logger.error(f"  Prediction failed: {e}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)

        n_regime_specific = sum(1 for r in results if r['is_regime_specific'])
        n_generic = len(results) - n_regime_specific

        logger.info(f"\nTotal predictions: {len(results)}")
        logger.info(f"Regime-specific: {n_regime_specific} ({n_regime_specific/len(results)*100:.1f}%)")
        logger.info(f"Generic fallback: {n_generic} ({n_generic/len(results)*100:.1f}%)")

        # Regime distribution
        regime_counts = {}
        for r in results:
            regime = r['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        logger.info("\nRegime distribution:")
        for regime, count in regime_counts.items():
            logger.info(f"  {regime}: {count}")

        logger.info("\n✅ Test complete!")

        logger.info("\nNext steps:")
        logger.info("1. Integrate with MLInference for live predictions")
        logger.info("2. Monitor regime-aware vs generic model performance")
        logger.info("3. Test with full backtesting pipeline")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise


def test_model_selection_logic():
    """Test model selection logic independently."""
    logger.info("\n" + "=" * 70)
    logger.info("MODEL SELECTION LOGIC TEST")
    logger.info("=" * 70)

    from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

    try:
        # Load HMM
        logger.info("\nLoading HMM detector...")
        detector = HMMRegimeDetector.load("models/hmm/regime_model")
        logger.info(f"✅ Loaded: {detector.n_regimes} regimes")

        # Load data
        logger.info("\nLoading test data...")
        data = load_sample_data()

        # Engineer features
        logger.info("\nEngineering features...")
        feature_engineer = HMMFeatureEngineer()
        features_df = feature_engineer.engineer_features(data)

        # Test model selection at different confidence thresholds
        logger.info("\nTesting model selection at different confidence thresholds...")

        # Test last 10 bars
        for i in range(len(features_df) - 10, len(features_df)):
            current_features = features_df.iloc[i-100:i+1]

            # Detect regime
            regime_state = detector.detect_regime(current_features)

            # Determine which model would be used
            regime_name = regime_state.regime
            confidence = regime_state.probability

            # Check against thresholds
            use_regime_specific_07 = confidence >= 0.7
            use_regime_specific_05 = confidence >= 0.5
            use_regime_specific_03 = confidence >= 0.3

            logger.info(
                f"Bar {i}: regime={regime_name}, "
                f"conf={confidence:.3f}, "
                f"use_specific@0.7={use_regime_specific_07}, "
                f"use_specific@0.5={use_regime_specific_05}, "
                f"use_specific@0.3={use_regime_specific_03}"
            )

        logger.info("\n✅ Model selection logic test complete!")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    logger.info("Starting regime-aware model selector tests...")

    # Test 1: Model selection logic
    test_model_selection_logic()

    # Test 2: Regime-aware model selector
    test_regime_aware_model_selector()

    logger.info("\n✅ All tests complete!")

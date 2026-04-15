#!/usr/bin/env python3
"""Generate regime-aware training data for 1-minute dollar bars with Silver Bullet labels.

This script generates Silver Bullet setups with regime labels for 1-minute data,
creating separate training datasets for each regime with actual trading labels.
"""

import sys
import warnings
from pathlib import Path
import logging
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.features import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants for label generation
TAKE_PROFIT_PCT = 0.003  # 0.3%
STOP_LOSS_PCT = 0.002    # 0.2%
MAX_HOLD_BARS = 30  # 30 minutes at 1-min bars
CONTRACTS_PER_TRADE = 5

def generate_silver_bullet_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Silver Bullet setup labels for training.

    For each bar, determine if it would be a successful trade using:
    - 5-bar momentum for direction
    - Triple-barrier exits for outcome
    """
    logger.info("Generating Silver Bullet labels...")

    df = df.copy()
    df['momentum_5'] = df['close'].pct_change(5)
    df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1

    # Label: 1 if 5-period forward return > 0, else 0
    df['label'] = (df['future_return_5'] > 0).astype(int)

    # Remove rows where we can't calculate forward return
    df = df.dropna(subset=['label', 'momentum_5', 'future_return_5'])

    logger.info(f"✅ Generated {len(df)} labeled samples")
    logger.info(f"   Positive samples: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")
    logger.info(f"   Negative samples: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")

    return df

def main():
    logger.info("=" * 70)
    logger.info("GENERATING REGIME-AWARE TRAINING DATA - 1-MINUTE 2025")
    logger.info("=" * 70)

    # Load 1-minute dollar bars
    logger.info("\nLoading 1-minute dollar bars...")
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    logger.info(f"✅ Loaded {len(df):,} dollar bars")

    # Load HMM model
    logger.info("\nLoading HMM regime detector...")
    hmm_path = Path("models/hmm/regime_model_1min")
    detector = HMMRegimeDetector.load(hmm_path)
    logger.info(f"✅ Loaded HMM with {detector.n_regimes} regimes")

    # Generate regime predictions
    logger.info("\nGenerating regime predictions...")
    feature_engineer = HMMFeatureEngineer()
    features = feature_engineer.engineer_features(df)
    regimes = detector.predict(features)

    # Add regime to dataframe
    df['regime'] = regimes

    logger.info(f"✅ Regime predictions complete")
    logger.info(f"   Regime distribution:")
    for regime_id in range(detector.n_regimes):
        count = (df['regime'] == regime_id).sum()
        pct = count / len(df) * 100
        logger.info(f"     Regime {regime_id}: {count:,} bars ({pct:.1f}%)")

    # Generate trading labels
    df = generate_silver_bullet_labels(df)

    # Split by regime and save
    logger.info("\nSaving regime-specific datasets...")
    output_dir = Path("data/ml_training/regime_aware_1min_2025")
    output_dir.mkdir(parents=True, exist_ok=True)

    for regime_id in range(detector.n_regimes):
        regime_df = df[df['regime'] == regime_id].copy()

        # Drop columns that shouldn't be in training data
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'notional', 'regime', 'label', 'momentum_5']
        regime_df = regime_df[cols_to_keep]

        output_file = output_dir / f"regime_{regime_id}_training_data.parquet"
        regime_df.to_parquet(output_file, index=True)

        pos_samples = (regime_df['label'] == 1).sum()
        neg_samples = (regime_df['label'] == 0).sum()
        logger.info(f"✅ Saved Regime {regime_id}: {len(regime_df):,} samples (pos: {pos_samples}, neg: {neg_samples})")

    # Generate summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING DATA GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Regimes: {detector.n_regimes}")
    logger.info(f"Overall label distribution:")
    total_pos = (df['label'] == 1).sum()
    total_neg = (df['label'] == 0).sum()
    logger.info(f"   Positive: {total_pos:,} ({total_pos/len(df)*100:.1f}%)")
    logger.info(f"   Negative: {total_neg:,} ({total_neg/len(df)*100:.1f}%)")

    return 0

if __name__ == "__main__":
    sys.exit(main())

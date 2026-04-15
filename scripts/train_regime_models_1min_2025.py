#!/usr/bin/env python3
"""Train regime-specific XGBoost models for 1-minute dollar bars.

This script trains separate XGBoost models for each regime using temporal
train/test splits to prevent look-ahead bias.
"""

import sys
import warnings
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def train_regime_model(regime_id: int, data_path: Path, output_dir: Path):
    """Train XGBoost model for specific regime.

    Args:
        regime_id: Regime number (0, 1, or 2)
        data_path: Path to training data parquet file
        output_dir: Output directory for models
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TRAINING REGIME {regime_id} MODEL")
    logger.info(f"{'=' * 70}")

    # Load data
    logger.info(f"Loading training data...")
    df = pd.read_parquet(data_path)

    # Filter to regime-specific data
    df = df[df['regime'] == regime_id].copy()

    logger.info(f"✅ Loaded {len(df):,} samples for Regime {regime_id}")

    # Prepare features
    label_col = 'label' if 'label' in df.columns else 'success'
    exclude_cols = ['regime', label_col, 'timestamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[label_col]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    # Temporal train/test split (80/20 by time)
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train):,} samples")
    logger.info(f"Test: {len(X_test):,} samples")

    # Train model
    logger.info(f"\nTraining XGBoost classifier...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    brier_loss = brier_score_loss(y_test, y_prob)

    logger.info(f"\n✅ Training complete")
    logger.info(f"   Accuracy: {accuracy:.2%}")
    logger.info(f"   Brier Score: {brier_loss:.3f}")

    # Detailed report
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Save model
    model_name = f"xgboost_regime_{regime_id}_real_labels.joblib"
    if regime_id == 1:
        model_name = "xgboost_generic_real_labels.joblib"

    model_path = output_dir / model_name
    joblib.dump(model, model_path)

    logger.info(f"✅ Model saved to: {model_path}")

    return accuracy, brier_loss

def main():
    logger.info("=" * 70)
    logger.info("REGIME-SPECIFIC MODEL TRAINING - 1-MINUTE 2025")
    logger.info("=" * 70)

    # Setup
    training_data_dir = Path("data/ml_training/regime_aware_1min_2025")
    output_dir = Path("models/xgboost/regime_aware_1min_2025")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Train models for each regime
    for regime_id in [0, 1, 2]:
        data_file = training_data_dir / f"regime_{regime_id}_training_data.parquet"

        if not data_file.exists():
            logger.warning(f"⚠️  Training data for Regime {regime_id} not found: {data_file}")
            continue

        accuracy, brier_loss = train_regime_model(regime_id, data_file, output_dir)
        results[regime_id] = {'accuracy': accuracy, 'brier_loss': brier_loss}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MODEL TRAINING SUMMARY")
    logger.info("=" * 70)

    for regime_id, metrics in results.items():
        logger.info(f"Regime {regime_id}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"  Brier Score: {metrics['brier_loss']:.3f}")

    logger.info("\n✅ All models trained successfully")

    return 0

if __name__ == "__main__":
    sys.exit(main())

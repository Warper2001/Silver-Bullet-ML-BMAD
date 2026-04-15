#!/usr/bin/env python3
"""Train regime-specific XGBoost models for 1-minute dollar bars with proper 54-feature engineering.

This script trains separate XGBoost models for each regime using temporal
train/test splits to prevent look-ahead bias, with FULL feature engineering.
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

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.features import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_features_with_labels(df: pd.DataFrame, feature_engineer: FeatureEngineer) -> pd.DataFrame:
    """Generate features and labels for training.

    Uses the full FeatureEngineer to generate 54 technical indicators.
    """
    logger.info("Generating features and labels...")

    features_list = []
    labels_list = []

    # Start after feature window
    for i in range(100, len(df) - 5):  # -5 for forward return calculation
        if i % 10000 == 0:
            logger.info(f"  Processing bar {i:,}/{len(df):,}...")

        try:
            current_bar = df.iloc[i]
            historical_data = df.iloc[i-100:i]

            # Generate Silver Bullet label
            future_return_5 = df['close'].iloc[i+5] / current_bar['close'] - 1
            label = 1 if future_return_5 > 0 else 0

            # Generate features using FeatureEngineer
            from src.data.models import DollarBar
            bar_dict = DollarBar(
                timestamp=df.index[i],
                open=float(current_bar['open']),
                high=float(current_bar['high']),
                low=float(current_bar['low']),
                close=float(current_bar['close']),
                volume=int(current_bar['volume']),
                notional_value=float(current_bar['notional'])
            )

            features = feature_engineer.generate_features_bar(
                current_bar=bar_dict,
                historical_data=historical_data
            )

            features_list.append(features)
            labels_list.append(label)

        except Exception as e:
            logger.warning(f"  Warning at bar {i}: {e}")
            continue

    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(len(features_list[0]))]
    features_df = pd.DataFrame(features_list, columns=feature_cols)
    features_df['label'] = labels_list

    logger.info(f"✅ Generated {len(features_df):,} labeled samples with {len(feature_cols)} features")

    return features_df

def train_regime_model(regime_id: int, data_path: Path, output_dir: Path, df_full: pd.DataFrame, feature_engineer: FeatureEngineer):
    """Train XGBoost model for specific regime with proper feature engineering."""

    logger.info(f"\n{'=' * 70}")
    logger.info(f"TRAINING REGIME {regime_id} MODEL WITH PROPER FEATURES")
    logger.info(f"{'=' * 70}")

    # Load regime-specific data
    logger.info(f"Loading regime {regime_id} data...")
    df = pd.read_parquet(data_path)

    # Filter to regime-specific data
    df = df[df['regime'] == regime_id].copy()
    df = df.reset_index(drop=True)  # Reset index for feature generation

    logger.info(f"✅ Loaded {len(df):,} samples for Regime {regime_id}")

    if len(df) < 1000:
        logger.warning(f"⚠️  Insufficient data for Regime {regime_id}, skipping...")
        return None

    # Generate features
    features_df = generate_features_with_labels(df, feature_engineer)

    # Prepare training data
    exclude_cols = ['label']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    X = features_df[feature_cols].values
    y = features_df['label'].values

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label distribution: {np.bincount(y)}")

    # Temporal train/test split (80/20 by time)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

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
    model_name = f"xgboost_regime_{regime_id}_proper_features.joblib"
    if regime_id == 1:
        model_name = "xgboost_generic_proper_features.joblib"

    model_path = output_dir / model_name
    joblib.dump(model, model_path)

    logger.info(f"✅ Model saved to: {model_path}")

    return accuracy, brier_loss

def main():
    logger.info("=" * 70)
    logger.info("REGIME-SPECIFIC MODEL TRAINING - 1-MINUTE 2025 (PROPER FEATURES)")
    logger.info("=" * 70)

    # Setup
    training_data_dir = Path("data/ml_training/regime_aware_1min_2025")
    output_dir = Path("models/xgboost/regime_aware_1min_2025_proper")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset for feature engineering context
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df_full = pd.read_csv(data_path)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    df_full = df_full.set_index('timestamp')

    # Initialize FeatureEngineer
    model_dir = Path("models/xgboost/regime_aware_1min_2025")
    feature_engineer = FeatureEngineer(
        model_dir=model_dir,
        window_size=100
    )

    results = {}

    # Train models for each regime
    for regime_id in [0, 1, 2]:
        data_file = training_data_dir / f"regime_{regime_id}_training_data.parquet"

        if not data_file.exists():
            logger.warning(f"⚠️  Training data for Regime {regime_id} not found: {data_file}")
            continue

        result = train_regime_model(regime_id, data_file, output_dir, df_full, feature_engineer)
        if result:
            accuracy, brier_loss = result
            results[regime_id] = {'accuracy': accuracy, 'brier_loss': brier_loss}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MODEL TRAINING SUMMARY")
    logger.info("=" * 70)

    for regime_id, metrics in results.items():
        logger.info(f"Regime {regime_id}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"  Brier Score: {metrics['brier_loss']:.3f}")

    logger.info("\n✅ All models trained successfully with proper 54-feature engineering")

    return 0

if __name__ == "__main__":
    sys.exit(main())

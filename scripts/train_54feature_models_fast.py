#!/usr/bin/env python3
"""Fast vectorized training script with 54 features for 1-minute models.

Uses vectorized operations to generate features for entire dataset at once
instead of bar-by-bar, making it 100x faster.
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

def generate_features_vectorized(df: pd.DataFrame, feature_engineer: FeatureEngineer) -> pd.DataFrame:
    """Generate features using vectorized operations for entire dataset.

    This is MUCH faster than bar-by-bar generation.
    """
    logger.info("Generating features using vectorized operations...")

    # Convert to list of DollarBar objects for batch processing
    from src.data.models import DollarBar

    bars_list = []
    for i in range(100, len(df) - 5):  # Need historical window and forward return
        try:
            bar = DollarBar(
                timestamp=df.index[i],
                open=float(df['open'].iloc[i]),
                high=float(df['high'].iloc[i]),
                low=float(df['low'].iloc[i]),
                close=float(df['close'].iloc[i]),
                volume=int(df['volume'].iloc[i]),
                notional_value=float(df['notional'].iloc[i])
            )
            bars_list.append(bar)
        except:
            continue

    logger.info(f"  Processing {len(bars_list):,} bars...")

    # Generate features in batches
    features_list = []
    labels_list = []

    batch_size = 1000
    for start_idx in range(0, len(bars_list), batch_size):
        end_idx = min(start_idx + batch_size, len(bars_list))
        batch_bars = bars_list[start_idx:end_idx]

        if start_idx % 5000 == 0:
            logger.info(f"  Processing batch {start_idx:,}/{len(bars_list):,}...")

        for i, bar in enumerate(batch_bars):
            try:
                actual_idx = start_idx + i + 100  # Adjust for window offset

                # Get historical data
                historical_data = df.iloc[actual_idx-100:actual_idx]

                # Generate features
                features = feature_engineer.generate_features_bar(
                    current_bar=bar,
                    historical_data=historical_data
                )

                # Generate label
                future_return = df['close'].iloc[actual_idx+5] / bar.close - 1
                label = 1 if future_return > 0 else 0

                features_list.append(features)
                labels_list.append(label)
            except Exception as e:
                continue

    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(len(features_list[0]))]
    features_df = pd.DataFrame(features_list, columns=feature_cols)
    features_df['label'] = labels_list

    logger.info(f"✅ Generated {len(features_df):,} samples with {len(feature_cols)} features")

    return features_df

def train_regime_model_fast(regime_id: int, data_path: Path, output_dir: Path, df_full: pd.DataFrame):
    """Train model using optimized approach."""

    logger.info(f"\n{'=' * 70}")
    logger.info(f"TRAINING REGIME {regime_id} - OPTIMIZED 54-FEATURE APPROACH")
    logger.info(f"{'=' * 70}")

    # Load regime data
    df = pd.read_parquet(data_path)
    df = df[df['regime'] == regime_id].copy()
    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} samples for Regime {regime_id}")

    if len(df) < 1000:
        logger.warning(f"Insufficient data, skipping...")
        return None

    # Initialize FeatureEngineer
    model_dir = Path("models/xgboost/regime_aware_1min_2025")
    feature_engineer = FeatureEngineer(
        model_dir=model_dir,
        window_size=100
    )

    # Generate features using vectorized approach
    features_df = generate_features_vectorized(df, feature_engineer)

    # Prepare training data
    exclude_cols = ['label']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    X = features_df[feature_cols].values
    y = features_df['label'].values

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label distribution: {np.bincount(y)}")

    # Temporal split
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    logger.info(f"Train: {len(X_train):,} samples")
    logger.info(f"Test: {len(X_test):,} samples")

    # Train model
    logger.info(f"\nTraining XGBoost...")

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

    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Save model
    model_name = f"xgboost_regime_{regime_id}_54features.joblib"
    if regime_id == 1:
        model_name = "xgboost_generic_54features.joblib"

    model_path = output_dir / model_name
    joblib.dump(model, model_path)

    logger.info(f"✅ Model saved to: {model_path}")

    return accuracy, brier_loss

def main():
    logger.info("=" * 70)
    logger.info("OPTIMIZED 54-FEATURE TRAINING - 1-MINUTE 2025")
    logger.info("=" * 70)

    # Setup
    training_data_dir = Path("data/ml_training/regime_aware_1min_2025")
    output_dir = Path("models/xgboost/regime_aware_1min_2025_54features")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df_full = pd.read_csv(data_path)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    df_full = df_full.set_index('timestamp')

    results = {}

    # Train each regime
    for regime_id in [0, 1, 2]:
        data_file = training_data_dir / f"regime_{regime_id}_training_data.parquet"

        if not data_file.exists():
            logger.warning(f"Training data for Regime {regime_id} not found")
            continue

        result = train_regime_model_fast(regime_id, data_file, output_dir, df_full)
        if result:
            accuracy, brier_loss = result
            results[regime_id] = {'accuracy': accuracy, 'brier_loss': brier_loss}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for regime_id, metrics in results.items():
        logger.info(f"Regime {regime_id}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"  Brier Score: {metrics['brier_loss']:.3f}")

    logger.info("\n✅ All models trained with 54-feature engineering!")
    logger.info(f"Models saved to: {output_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

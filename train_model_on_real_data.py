#!/usr/bin/env python
"""
Train XGBoost model on real 1-minute MNQ data.

This script:
1. Loads real 1-minute dollar bars from CSV
2. Generates features using FeatureEngineer
3. Creates labels based on forward price movement (Silver Bullet-style setups)
4. Trains XGBoost model with walk-forward validation
5. Saves model for optimization researcher
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_silver_bullet_labels(
    df: pd.DataFrame,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 0.25,
    max_bars: int = 50,
) -> pd.Series:
    """
    Create binary labels based on Silver Bullet-style exit conditions.

    A label of 1 means the setup would have been profitable (hit take profit before stop loss).
    A label of 0 means the setup would have hit stop loss first or not hit take profit within max_bars.

    Args:
        df: DataFrame with OHLCV data
        take_profit_pct: Take profit as percentage of entry price
        stop_loss_pct: Stop loss as percentage of entry price
        max_bars: Maximum number of bars to hold position

    Returns:
        Series of binary labels (0 or 1)
    """
    labels = []

    for i in range(len(df)):
        if i + max_bars >= len(df):
            # Not enough future data, assign 0
            labels.append(0)
            continue

        entry_price = df.iloc[i]['close']
        take_profit = entry_price * (1 + take_profit_pct / 100)
        stop_loss = entry_price * (1 - stop_loss_pct / 100)

        # Check forward bars
        future_bars = df.iloc[i+1:i+max_bars+1]

        # Find first bar that hits take profit or stop loss
        hit_tp = False
        hit_sl = False

        for _, bar in future_bars.iterrows():
            if bar['high'] >= take_profit:
                hit_tp = True
                break
            if bar['low'] <= stop_loss:
                hit_sl = True
                break

        # Label is 1 if take profit was hit first or within max_bars
        # Label is 0 if stop loss was hit or neither was hit
        if hit_tp:
            labels.append(1)
        else:
            labels.append(0)

    return pd.Series(labels, index=df.index)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load 1-minute CSV data and prepare for feature engineering.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with timestamp index and OHLCV columns
    """
    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)

    # Reset index to keep timestamp as a column for FeatureEngineer
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    return df


def main():
    """Main training pipeline."""
    # Paths
    csv_path = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    model_dir = Path("models/xgboost/1_minute")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data(csv_path)

    # Create labels (Silver Bullet-style setups)
    logger.info("Creating Silver Bullet labels...")
    labels = create_silver_bullet_labels(
        df,
        take_profit_pct=0.5,
        stop_loss_pct=0.25,
        max_bars=50,
    )

    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    logger.info(f"Positive class ratio: {labels.mean():.2%}")

    # Generate features
    logger.info("Generating features with FeatureEngineer...")
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)

    logger.info(f"Generated {len(features_df.columns)} features")

    # Remove non-numeric columns (timestamp, trading_session, etc.)
    logger.info("Removing non-numeric columns...")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]
    logger.info(f"Selected {len(numeric_cols)} numeric features")

    # Remove NaN values from features
    logger.info("Removing NaN values...")
    valid_idx = features_df.dropna().index.intersection(labels.index)
    features_df = features_df.loc[valid_idx]
    labels = labels.loc[valid_idx]

    logger.info(f"After dropping NaNs: {len(features_df)} samples")

    # Chronological split: 75% train, 25% validation
    split_idx = int(len(features_df) * 0.75)
    X_train = features_df.iloc[:split_idx]
    X_val = features_df.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_val = labels.iloc[split_idx:]

    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Train XGBoost model
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    logger.info("Evaluating model...")
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    logger.info(f"Validation Accuracy: {accuracy:.2%}")
    logger.info(f"Validation Precision: {precision:.2%}")
    logger.info(f"Validation Recall: {recall:.2%}")
    logger.info(f"Validation F1: {f1:.2%}")

    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature names
    feature_names_path = model_dir / "feature_names.json"
    import json
    with open(feature_names_path, 'w') as f:
        json.dump({
            "feature_names": X_train.columns.tolist(),
            "n_features": len(X_train.columns),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }, f, indent=2)
    logger.info(f"Feature names saved to {feature_names_path}")

    # Save metadata
    metadata = {
        "model_type": "XGBClassifier",
        "training_date": datetime.now().isoformat(),
        "data_period": f"{df.index.min()} to {df.index.max()}",
        "n_features": len(X_train.columns),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "random_state": model.random_state,
        },
        "label_params": {
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.25,
            "max_bars": 50,
        },
    }

    metadata_path = model_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    logger.info("Training complete!")

    return model, metadata


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train premium regime-aware XGBoost models.

This script trains separate XGBoost models for each regime using
premium-labeled training data.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_premium_model(
    training_data_path: Path,
    model_output_path: Path,
    regime_name: str
) -> Dict:
    """Train premium XGBoost model for a specific regime.

    Args:
        training_data_path: Path to training data parquet file
        model_output_path: Path to save trained model
        regime_name: Name of the regime (for logging)

    Returns:
        Dictionary with training results
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training Premium Model: {regime_name}")
    logger.info(f"{'=' * 70}")

    # Load training data
    df = pd.read_parquet(training_data_path)
    logger.info(f"Loaded {len(df)} trades")

    # Prepare features
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 'notional_value',
        'atr', 'atr_ratio', 'returns', 'high_low_range', 'close_position',
        'volume_ratio', 'vwap', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'stoch_k', 'stoch_d', 'roc', 'historical_volatility',
        'parkinson_volatility', 'garman_klass_volatility',
        'price_momentum_5', 'price_momentum_10',
        'volume_ma_20', 'volume_std_20', 'range_ma_20', 'range_std_20',
        'volatility_ma_20', 'volatility_std_20',
        'rsi_ma_14', 'rsi_std_14', 'macd_ma_9', 'macd_std_9',
        'stoch_k_ma_14', 'stoch_d_ma_14', 'atr_ma_14', 'atr_std_14',
        'return_ma_10', 'return_std_10',
        'close_position_ma_20', 'close_position_std_20',
        'is_london_am', 'is_ny_am', 'is_ny_pm',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]

    # Filter to available columns
    available_features = [f for f in feature_columns if f in df.columns]
    logger.info(f"Using {len(available_features)} features")

    X = df[available_features].fillna(0)
    y = df['label']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Train XGBoost model
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 200,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist'
    }

    model = xgb.XGBClassifier(**params)

    logger.info("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, zero_division=0)
    val_recall = recall_score(y_val, y_pred_val, zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val, zero_division=0)

    results = {
        'regime': regime_name,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'feature_importance': dict(zip(
            available_features,
            model.feature_importances_
        ))
    }

    logger.info(f"\nResults:")
    logger.info(f"  Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"  Val Accuracy: {val_accuracy*100:.2f}%")
    logger.info(f"  Val Precision: {val_precision*100:.2f}%")
    logger.info(f"  Val Recall: {val_recall*100:.2f}%")
    logger.info(f"  Val F1: {val_f1:.2f}")

    # Save model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    logger.info(f"✅ Model saved to {model_output_path}")

    # Save metadata
    metadata = {
        'model_type': 'premium_regime_aware',
        'regime': regime_name,
        'training_date': datetime.now().isoformat(),
        'n_features': len(available_features),
        'feature_names': available_features,
        'params': params,
        'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in results.items()
                    if k != 'feature_importance'}
    }

    metadata_path = model_output_path.parent / f"{model_output_path.stem}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return results


def main():
    """Main execution."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PREMIUM REGIME-AWARE MODELS")
    logger.info("=" * 70)

    try:
        output_dir = Path("models/xgboost/premium_regime_aware")
        input_dir = Path("data/ml_training/premium_regime_aware")

        results_list = []

        # Train generic premium model
        generic_input = input_dir / "generic_premium.parquet"
        if generic_input.exists():
            results = train_premium_model(
                training_data_path=generic_input,
                model_output_path=output_dir / "xgboost_generic_premium.joblib",
                regime_name="Generic"
            )
            results_list.append(results)

        # Train regime-specific premium models
        for regime_id in [0, 1, 2]:
            input_file = input_dir / f"regime_{regime_id}_premium.parquet"

            if not input_file.exists():
                logger.warning(f"Regime {regime_id} data not found: {input_file}")
                continue

            results = train_premium_model(
                training_data_path=input_file,
                model_output_path=output_dir / f"xgboost_regime_{regime_id}_premium.joblib",
                regime_name=f"Regime_{regime_id}"
            )
            results_list.append(results)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)

        for results in results_list:
            logger.info(f"\n{results['regime']}:")
            logger.info(f"  Val Accuracy: {results['val_accuracy']*100:.2f}%")
            logger.info(f"  Val F1: {results['val_f1']:.2f}")

        logger.info(f"\n✅ All premium models saved to {output_dir}")

        logger.info("\nNext steps:")
        logger.info("  1. Backtest premium + hybrid system:")
        logger.info("     .venv/bin/python scripts/backtest_premium_hybrid.py")
        logger.info("  2. If results good, deploy to paper trading")

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

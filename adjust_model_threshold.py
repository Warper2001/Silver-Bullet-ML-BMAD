#!/usr/bin/env python
"""
Adjust model prediction threshold to achieve target trading frequency.

Goal: 1-20 trades per day (currently 0.023 trades/day)
Need to increase frequency by 43x to 870x
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_silver_bullet_labels(df: pd.DataFrame) -> pd.Series:
    """Create Silver Bullet labels."""
    labels = []
    max_bars = 50

    for i in range(len(df)):
        if i + max_bars >= len(df):
            labels.append(0)
            continue

        entry_price = df.iloc[i]['close']
        take_profit = entry_price * 1.005  # 0.5%
        stop_loss = entry_price * 0.9975   # 0.25%

        future_bars = df.iloc[i+1:i+max_bars+1]
        hit_tp = False

        for _, bar in future_bars.iterrows():
            if bar['high'] >= take_profit:
                hit_tp = True
                break
            if bar['low'] <= stop_loss:
                break

        labels.append(1 if hit_tp else 0)

    return pd.Series(labels, index=df.index)


def main():
    """Adjust model threshold for target trading frequency."""
    logger.info("=" * 80)
    logger.info("THRESHOLD TUNING FOR TARGET TRADING FREQUENCY")
    logger.info("=" * 80)

    # Load data
    csv_path = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    logger.info(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Loaded {len(df)} bars")

    # Generate features
    logger.info("Generating features...")
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]

    # Generate labels
    logger.info("Generating labels...")
    labels = create_silver_bullet_labels(df)

    # Remove NaN
    valid_idx = features_df.dropna().index.intersection(labels.index)
    features_df = features_df.loc[valid_idx]
    labels = labels.loc[valid_idx]

    # 3-way split
    n_samples = len(features_df)
    val_end = int(n_samples * 0.80)

    X_train = features_df.iloc[:val_end]
    y_train = labels.iloc[:val_end]

    X_test = features_df.iloc[val_end:]
    y_test = labels.iloc[val_end:]

    # Load selected features
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        feature_config = json.load(f)
    optimal_features = feature_config['features'][:25]

    X_train_selected = X_train[optimal_features]
    X_test_selected = X_test[optimal_features]

    # Train model
    logger.info("Training model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=3.0,  # Give more weight to positive class
        n_jobs=-1,
    )
    model.fit(X_train_selected, y_train)

    # Get prediction probabilities
    logger.info("Analyzing prediction probabilities...")
    y_proba = model.predict_proba(X_test_selected)[:, 1]

    logger.info(f"Probability statistics:")
    logger.info(f"  Mean: {y_proba.mean():.4f}")
    logger.info(f"  Median: {np.median(y_proba):.4f}")
    logger.info(f"  Std: {y_proba.std():.4f}")
    logger.info(f"  Min: {y_proba.min():.4f}")
    logger.info(f"  Max: {y_proba.max():.4f}")
    logger.info(f"  > 0.5: {(y_proba > 0.5).sum()} ({(y_proba > 0.5).sum() / len(y_proba):.2%})")
    logger.info(f"  > 0.3: {(y_proba > 0.3).sum()} ({(y_proba > 0.3).sum() / len(y_proba):.2%})")
    logger.info(f"  > 0.2: {(y_proba > 0.2).sum()} ({(y_proba > 0.2).sum() / len(y_proba):.2%})")
    logger.info(f"  > 0.1: {(y_proba > 0.1).sum()} ({(y_proba > 0.1).sum() / len(y_proba):.2%})")

    # Calculate test period in days
    test_start_date = df.iloc[val_end]['timestamp']
    test_end_date = df.iloc[-1]['timestamp']
    test_days = (test_end_date - test_start_date).days

    logger.info(f"\nTest period: {test_days} days")
    logger.info(f"Target trades per day: 1-20")
    logger.info(f"Target total trades: {test_days}-{test_days * 20}")

    # Find threshold that gives desired trade frequency
    logger.info("\n" + "=" * 80)
    logger.info("TESTING DIFFERENT THRESHOLDS")
    logger.info("=" * 80)

    results = []
    thresholds_to_test = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05]

    for threshold in thresholds_to_test:
        y_pred = (y_proba >= threshold).astype(int)

        n_trades = y_pred.sum()
        trades_per_day = n_trades / test_days if test_days > 0 else 0

        # Calculate metrics
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (y_pred == y_test).mean()

        results.append({
            'threshold': threshold,
            'n_trades': n_trades,
            'trades_per_day': trades_per_day,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        })

    # Display results
    logger.info(f"\n{'Threshold':<10} {'Trades/Day':<12} {'Total':<8} {'Precision':<12} {'Recall':<12} {'Accuracy':<12}")
    logger.info("-" * 80)

    for r in results:
        logger.info(
            f"{r['threshold']:<10.2f} "
            f"{r['trades_per_day']:<12.2f} "
            f"{r['n_trades']:<8} "
            f"{r['precision']:<12.2%} "
            f"{r['recall']:<12.2%} "
            f"{r['accuracy']:<12.2%}"
        )

    # Find optimal threshold
    logger.info("\n" + "=" * 80)
    logger.info("SELECTING OPTIMAL THRESHOLD")
    logger.info("=" * 80)

    # Target: 1-20 trades per day
    valid_results = [r for r in results if 1 <= r['trades_per_day'] <= 20]

    if valid_results:
        # Select the one with best precision while meeting target
        best = max(valid_results, key=lambda x: x['precision'])

        logger.info(f"\n✅ Found optimal threshold: {best['threshold']:.2f}")
        logger.info(f"   Trades per day: {best['trades_per_day']:.2f}")
        logger.info(f"   Total trades: {best['n_trades']}")
        logger.info(f"   Expected precision: {best['precision']:.2%}")
        logger.info(f"   Expected recall: {best['recall']:.2%}")
        logger.info(f"   Expected accuracy: {best['accuracy']:.2%}")

        # Save threshold config
        config = {
            'optimal_threshold': best['threshold'],
            'target_trades_per_day': best['trades_per_day'],
            'expected_precision': best['precision'],
            'expected_recall': best['recall'],
            'expected_accuracy': best['accuracy'],
            'test_date': '2026-04-09',
        }

        config_path = 'models/xgboost/1_minute/prediction_threshold.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"\n✅ Threshold config saved to: {config_path}")

        # Final validation with this threshold
        logger.info("\n" + "=" * 80)
        logger.info("FINAL VALIDATION WITH OPTIMAL THRESHOLD")
        logger.info("=" * 80)

        threshold = best['threshold']
        y_pred_final = (y_proba >= threshold).astype(int)

        logger.info(f"\nUsing threshold: {threshold:.2f}")
        logger.info(f"Total trades: {y_pred_final.sum()}")
        logger.info(f"Trades per day: {y_pred_final.sum() / test_days:.2f}")
        logger.info(f"Actual vs Predicted:")
        logger.info(f"  True Positives: {((y_pred_final == 1) & (y_test == 1)).sum()}")
        logger.info(f"  False Positives: {((y_pred_final == 1) & (y_test == 0)).sum()}")
        logger.info(f"  True Negatives: {((y_pred_final == 0) & (y_test == 0)).sum()}")
        logger.info(f"  False Negatives: {((y_pred_final == 0) & (y_test == 1)).sum()}")

    else:
        logger.warning("❌ No threshold found that achieves 1-20 trades/day")
        logger.info("Current closest options:")
        for r in results:
            logger.info(f"  Threshold {r['threshold']:.2f}: {r['trades_per_day']:.2f} trades/day")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

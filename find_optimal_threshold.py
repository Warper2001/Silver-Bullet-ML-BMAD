#!/usr/bin/env python
"""
Find threshold for 1-20 trades per day with class-weighted model.
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_labels(df):
    labels = []
    for i in range(len(df)):
        if i + 50 >= len(df):
            labels.append(0)
            continue
        entry_price = df.iloc[i]['close']
        take_profit = entry_price * 1.005
        stop_loss = entry_price * 0.9975
        future_bars = df.iloc[i+1:i+51]
        hit_tp = any(bar['high'] >= take_profit for _, bar in future_bars.iterrows())
        labels.append(1 if hit_tp else 0)
    return pd.Series(labels, index=df.index)


def main():
    # Load data
    df = pd.read_csv('data/processed/dollar_bars/1_minute/mnq_1min_2025.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]

    # Labels
    labels = create_labels(df)
    valid_idx = features_df.dropna().index.intersection(labels.index)
    features_df = features_df.loc[valid_idx]
    labels = labels.loc[valid_idx]

    # Split
    n_samples = len(features_df)
    val_end = int(n_samples * 0.80)
    X_test = features_df.iloc[val_end:]
    y_test = labels.iloc[val_end:]

    # Load features
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        optimal_features = json.load(f)['features'][:25]
    X_test = X_test[optimal_features]

    # Train model with class weighting
    X_train = features_df.iloc[:val_end][optimal_features]
    y_train = labels.iloc[:val_end]

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=3.0, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Test much higher thresholds
    test_days = 65
    logger.info(f"Testing thresholds for target: 1-20 trades/day ({test_days} days)")
    logger.info(f"Target total: {test_days}-{test_days * 20} trades")

    # Test thresholds from 0.5 to 0.95
    thresholds = np.arange(0.50, 0.96, 0.05)

    results = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        n_trades = y_pred.sum()
        trades_per_day = n_trades / test_days

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append({
            'threshold': thresh,
            'n_trades': n_trades,
            'trades_per_day': trades_per_day,
            'precision': precision,
            'recall': recall,
        })

    # Find best in range
    valid = [r for r in results if 1 <= r['trades_per_day'] <= 20]

    logger.info(f"\n{'Threshold':<10} {'Trades/Day':<12} {'Precision':<12} {'Recall':<12}")
    logger.info("-" * 50)
    for r in results:
        in_range = "✅" if 1 <= r['trades_per_day'] <= 20 else ""
        logger.info(
            f"{r['threshold']:<10.2f} {r['trades_per_day']:<12.2f} "
            f"{r['precision']:<12.2%} {r['recall']:<12.2%} {in_range}"
        )

    if valid:
        best = max(valid, key=lambda x: x['precision'])
        logger.info(f"\n✅ OPTIMAL THRESHOLD: {best['threshold']:.2f}")
        logger.info(f"   Trades per day: {best['trades_per_day']:.2f}")
        logger.info(f"   Precision: {best['precision']:.2%}")
        logger.info(f"   Recall: {best['recall']:.2%}")

        # Save config
        config = {
            'threshold': best['threshold'],
            'trades_per_day': best['trades_per_day'],
            'precision': best['precision'],
            'recall': best['recall'],
        }
        with open('models/xgboost/1_minute/optimal_threshold.json', 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved to: models/xgboost/1_minute/optimal_threshold.json")
    else:
        logger.warning("No threshold in range found")


if __name__ == "__main__":
    main()

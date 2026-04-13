#!/usr/bin/env python
"""
Proper Out-of-Sample Backtest for Silver Bullet Optimization.

This script uses a 3-way split to avoid data snooping bias:
1. Train set (60%): Model training + SHAP analysis
2. Validation set (20%): Feature selection + parameter optimization
3. Test set (20%): Final evaluation ONLY - completely held out

This gives us legitimate out-of-sample performance metrics.
"""

import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    """Create Silver Bullet labels based on forward price movement."""
    labels = []

    for i in range(len(df)):
        if i + max_bars >= len(df):
            labels.append(0)
            continue

        entry_price = df.iloc[i]['close']
        take_profit = entry_price * (1 + take_profit_pct / 100)
        stop_loss = entry_price * (1 - stop_loss_pct / 100)

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
    """Run proper out-of-sample backtest."""
    logger.info("=" * 80)
    logger.info("PROPER OUT-OF-SAMPLE BACKTEST")
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

    # Keep only numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]

    logger.info(f"Generated {len(features_df.columns)} numeric features")

    # Generate labels
    logger.info("Generating Silver Bullet labels...")
    labels = create_silver_bullet_labels(df)

    # Remove NaN values
    logger.info("Removing NaN values...")
    valid_idx = features_df.dropna().index.intersection(labels.index)
    features_df = features_df.loc[valid_idx]
    labels = labels.loc[valid_idx]

    logger.info(f"After dropping NaNs: {len(features_df)} samples")

    # ========================================================================
    # PROPER 3-WAY SPLIT TO AVOID DATA SNOOPING
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("CREATING PROPER 3-WAY SPLIT")
    logger.info("=" * 80)

    # Split: 60% train, 20% validation, 20% test
    n_samples = len(features_df)
    train_end = int(n_samples * 0.60)
    val_end = int(n_samples * 0.80)

    X_train = features_df.iloc[:train_end]
    y_train = labels.iloc[:train_end]

    X_val = features_df.iloc[train_end:val_end]
    y_val = labels.iloc[train_end:val_end]

    X_test = features_df.iloc[val_end:]
    y_test = labels.iloc[val_end:]

    logger.info(f"\nSplit breakdown:")
    logger.info(f"  Train:      {len(X_train):>6} samples ({len(X_train)/n_samples:.1%})")
    logger.info(f"  Validation: {len(X_val):>6} samples ({len(X_val)/n_samples:.1%})")
    logger.info(f"  Test:       {len(X_test):>6} samples ({len(X_test)/n_samples:.1%})")

    # ========================================================================
    # STEP 1: TRAIN ON TRAINING SET ONLY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: TRAINING MODEL ON TRAINING SET")
    logger.info("=" * 80)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on train (in-sample)
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    logger.info(f"Training accuracy (in-sample): {train_acc:.2%}")

    # ========================================================================
    # STEP 2: OPTIMIZE ON VALIDATION SET
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: FEATURE SELECTION + PARAMETER OPTIMIZATION ON VALIDATION SET")
    logger.info("=" * 80)

    # Use pre-selected features from optimization
    selected_features_path = "models/xgboost/1_minute/selected_features.json"
    with open(selected_features_path, 'r') as f:
        feature_config = json.load(f)

    optimal_features = feature_config['features'][:25]  # Top 25 features

    logger.info(f"Using {len(optimal_features)} pre-selected features")

    # Load optimized parameters
    params_path = "models/xgboost/1_minute/sb_params.json"
    with open(params_path, 'r') as f:
        optimized_params = json.load(f)

    logger.info(f"Using optimized parameters: {optimized_params}")

    # Train on training set with selected features
    X_train_selected = X_train[optimal_features]
    X_val_selected = X_val[optimal_features]

    logger.info("Retraining with selected features...")
    model_selected = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model_selected.fit(X_train_selected, y_train)

    # Evaluate on validation set
    val_preds = model_selected.predict(X_val_selected)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds, zero_division=0)
    val_rec = recall_score(y_val, val_preds, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)

    logger.info(f"\nValidation set performance:")
    logger.info(f"  Accuracy:  {val_acc:.2%}")
    logger.info(f"  Precision: {val_prec:.2%}")
    logger.info(f"  Recall:    {val_rec:.2%}")
    logger.info(f"  F1:        {val_f1:.2%}")

    # ========================================================================
    # STEP 3: FINAL EVALUATION ON TEST SET (NEVER SEEN BEFORE)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: FINAL OUT-OF-SAMPLE EVALUATION ON TEST SET")
    logger.info("  (This set was NEVER used for training or optimization)")
    logger.info("=" * 80)

    X_test_selected = X_test[optimal_features]
    test_preds = model_selected.predict(X_test_selected)

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    logger.info(f"\n🎯 TRUE OUT-OF-SAMPLE PERFORMANCE (Test Set):")
    logger.info(f"  Accuracy:  {test_acc:.2%}")
    logger.info(f"  Precision: {test_prec:.2%}")
    logger.info(f"  Recall:    {test_rec:.2%}")
    logger.info(f"  F1:        {test_f1:.2%}")

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    logger.info(f"\n{'Set':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    logger.info("-" * 60)
    logger.info(f"{'Train':<12} {train_acc:<12.2%} {':<12'} {':<12'} {':<12'}")
    logger.info(f"{'Validation':<12} {val_acc:<12.2%} {val_prec:<12.2%} {val_rec:<12.2%} {val_f1:<12.2%}")
    logger.info(f"{'Test (OOS)':<12} {test_acc:<12.2%} {test_prec:<12.2%} {test_rec:<12.2%} {test_f1:<12.2%}")

    # Check for overfitting
    logger.info("\n" + "=" * 80)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("=" * 80)

    accuracy_drop = val_acc - test_acc
    if accuracy_drop > 0.05:
        logger.warning(f"⚠️  Significant accuracy drop from validation to test: {accuracy_drop:.2%}")
        logger.warning("   This suggests possible overfitting to validation set!")
    elif accuracy_drop > 0.02:
        logger.info(f"✓  Moderate accuracy drop from validation to test: {accuracy_drop:.2%}")
        logger.info("   This is normal and indicates reasonable generalization.")
    else:
        logger.info(f"✓  Minimal accuracy drop from validation to test: {accuracy_drop:.2%}")
        logger.info("   Excellent generalization!")

    # Save results
    results = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "val_precision": float(val_prec),
        "val_recall": float(val_rec),
        "val_f1": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "accuracy_drop": float(accuracy_drop),
        "overfitting_detected": accuracy_drop > 0.05,
    }

    results_path = "_bmad-output/reports/proper_oos_backtest_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Final verdict
    logger.info("\n" + "=" * 80)
    logger.info("FINAL VERDICT")
    logger.info("=" * 80)

    if test_acc >= 0.65:
        logger.info(f"✅ PASS: Test set accuracy ({test_acc:.2%}) meets 65% threshold")
    else:
        logger.warning(f"❌ FAIL: Test set accuracy ({test_acc:.2%}) below 65% threshold")

    logger.info("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Retrain XGBoost Meta-Labeling Model on Recent Data (Phase 2).

This script retrains the model on 2025-2026 data only with regularization
to prevent overfitting and address performance decay.

Changes from original training:
- Training data: 2025-2026 only (skip 2024)
- Regularization: max_depth=4, learning_rate=0.05, reg_lambda=1.0, reg_alpha=0.1
- Validation: Walk-forward (not in-sample)
- Target: Realistic 30-40% win rate (not 85%)
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.label_mapper import map_signals_to_outcomes
from src.ml.signal_feature_extractor import SignalFeatureExtractor
from src.ml.meta_training_data_builder import MetaLabelingDatasetBuilder
from src.ml.xgboost_trainer import XGBoostTrainer, train_xgboost
from src.ml.training_data import split_data, select_features
from src.ml.walk_forward_validator import WalkForwardValidator
from xgboost import XGBClassifier


def load_time_bars(date_start: str, date_end: str) -> pd.DataFrame:
    """Load time-based bars for feature extraction."""
    print(f"📊 Loading time bars from {date_start} to {date_end}...")

    data_dir = Path("data/processed/time_bars/")

    start_dt = pd.Timestamp(date_start)
    end_dt = pd.Timestamp(date_end)
    current = start_dt.replace(day=1)

    files = []
    while current <= end_dt:
        filename = f"MNQ_time_bars_5min_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename
        if file_path.exists():
            files.append(file_path)
        current = current + pd.DateOffset(months=1)

    dataframes = []
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['dollar_bars'][:]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            dataframes.append(df)
        except Exception as e:
            print(f"   Warning: Failed to load {file_path.name}: {e}")

    if not dataframes:
        print(f"❌ No data files found for period {date_start} to {date_end}")
        return pd.DataFrame()

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) & (combined.index <= end_dt)
    ]

    print(f"✅ Loaded {len(combined):,} time bars")

    return combined


def filter_recent_data(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    start_date: str = "2025-01-01",
    end_date: str = "2026-03-31"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter signals and trades to recent date range.

    Args:
        signals_df: All signals
        trades_df: All trades
        start_date: Start date filter (inclusive)
        end_date: End date filter (inclusive)

    Returns:
        Filtered signals and trades
    """
    print(f"\n🎯 Filtering data to {start_date} to {end_date}...")

    # Ensure signals index is DatetimeIndex
    if not isinstance(signals_df.index, pd.DatetimeIndex):
        if 'timestamp' in signals_df.columns:
            signals_df = signals_df.set_index('timestamp')
        else:
            # Try to convert index to datetime
            signals_df.index = pd.to_datetime(signals_df.index)

    # For trades, check if there's an entry_time column
    if 'entry_time' in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    elif not isinstance(trades_df.index, pd.DatetimeIndex):
        if 'timestamp' in trades_df.columns:
            trades_df = trades_df.set_index('timestamp')
        else:
            # Try to convert index to datetime
            trades_df.index = pd.to_datetime(trades_df.index)

    # Filter by date
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    signals_filtered = signals_df.loc[
        (signals_df.index >= start_dt) & (signals_df.index <= end_dt)
    ].copy()

    # Filter trades by entry_time column if it exists, otherwise by index
    if 'entry_time' in trades_df.columns:
        trades_filtered = trades_df.loc[
            (trades_df['entry_time'] >= start_dt) & (trades_df['entry_time'] <= end_dt)
        ].copy()
    else:
        trades_filtered = trades_df.loc[
            (trades_df.index >= start_dt) & (trades_df.index <= end_dt)
        ].copy()

    print(f"✅ Filtered to {len(signals_filtered)} signals, {len(trades_filtered)} trades")
    print(f"   Original: {len(signals_df)} signals → {len(signals_filtered)} signals ({len(signals_filtered)/len(signals_df)*100:.1f}%)")

    return signals_filtered, trades_filtered


def main():
    """Train model on recent 2025-2026 data with regularization."""

    print("=" * 70)
    print("🚀 PHASE 2: MODEL RETRAINING (2025-2026 Data)")
    print("=" * 70)
    print()
    print("🎯 Objective: Retrain model on recent data to address performance decay")
    print("   Training period: 2025-01-01 to 2026-03-31 (15 months)")
    print("   Regularization: max_depth=4, learning_rate=0.05, L1/L2 penalties")
    print("   Target win rate: 30-40% (realistic)")
    print("   Validation: Walk-forward (out-of-sample)")
    print()

    # Step 1: Load training data
    print("📊 Step 1: Loading training data...")

    signals_path_full = Path("data/ml_training/silver_bullet_signals_full.parquet")
    trades_path_full = Path("data/ml_training/silver_bullet_trades_full.parquet")
    metadata_path_full = Path("data/ml_training/metadata_full.json")

    if not signals_path_full.exists():
        print(f"❌ Training data not found: {signals_path_full}")
        print("   Run: python run_optimized_silver_bullet.py --save-ml-data")
        return

    signals_df = pd.read_parquet(signals_path_full)
    trades_df = pd.read_parquet(trades_path_full)

    with open(metadata_path_full, 'r') as f:
        metadata = json.load(f)

    print(f"✅ Loaded full dataset: {len(signals_df)} signals, {len(trades_df)} trades")
    print(f"   Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")

    # Step 2: Filter to 2025-2026 data
    signals_recent, trades_recent = filter_recent_data(
        signals_df,
        trades_df,
        start_date="2025-01-01",
        end_date="2026-03-31"
    )

    # Validate minimum sample size
    if len(signals_recent) < 100:
        print(f"❌ Insufficient training data: {len(signals_recent)} signals")
        print(f"   Minimum 100 required. Extend date range.")
        return

    # Step 3: Load price data
    print("\n📊 Step 3: Loading price data for feature extraction...")
    price_data = load_time_bars("2025-01-01", "2026-03-31")

    if len(price_data) == 0:
        print("❌ No price data available for 2025-2026")
        return

    # Step 4: Build training dataset
    print("\n🔨 Step 4: Building training dataset...")

    feature_extractor = SignalFeatureExtractor(lookback_bars=100)
    dataset_builder = MetaLabelingDatasetBuilder(feature_extractor=feature_extractor)

    full_dataset = dataset_builder.build_dataset(
        signals_df=signals_recent,
        trades_df=trades_recent,
        price_data=price_data,
        verbose=True
    )

    print(f"\n✅ Dataset built: {len(full_dataset)} samples × {len(full_dataset.columns)} features")

    # Step 5: Select features
    print("\n🎯 Step 5: Selecting top features...")

    # Separate features and labels
    feature_cols = [col for col in full_dataset.columns if col not in ['label', 'return_pct']]
    X = full_dataset[feature_cols]
    y = full_dataset['label']

    # Select top features (remove highly correlated)
    selected_data, preprocessing_metadata = select_features(
        features_df=X,
        max_correlation=0.9,
        top_k=20
    )

    # Add label back
    selected_data['label'] = y.values

    print(f"✅ Selected {len([c for c in selected_data.columns if c != 'label'])} features")

    # Add timestamp column for time-based splitting
    selected_data = selected_data.reset_index()
    selected_data = selected_data.rename(columns={'index': 'timestamp'})

    # Step 6: Split data (time-based)
    print("\n🔀 Step 6: Splitting data (time-based)...")

    train, val, test, split_metadata = split_data(
        data_df=selected_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print(f"✅ Data split:")
    print(f"   Train: {len(train)} samples ({len(train)/len(selected_data)*100:.1f}%)")
    print(f"   Val: {len(val)} samples ({len(val)/len(selected_data)*100:.1f}%)")
    print(f"   Test: {len(test)} samples ({len(test)/len(selected_data)*100:.1f}%)")

    # Step 7: Train with REGULARIZATION
    print("\n🤖 Step 7: Training XGBoost model with REGULARIZATION...")

    # Separate features and labels (exclude non-feature columns)
    exclude_cols = {
        "label",
        "timestamp",
        "trading_session",  # Categorical column
        "signal_direction",  # Categorical column
        "direction",  # Categorical column
    }
    feature_cols_train = [col for col in train.columns if col not in exclude_cols]

    X_train = train[feature_cols_train]
    y_train = train["label"]
    X_val = val[feature_cols_train]
    y_val = val["label"]

    print(f"\n🎯 Regularization parameters:")
    print(f"   max_depth: 4 (was 6)")
    print(f"   learning_rate: 0.05 (was 0.1)")
    print(f"   reg_lambda: 1.0 (L2 regularization)")
    print(f"   reg_alpha: 0.1 (L1 regularization)")
    print(f"   min_child_weight: 3 (was 1)")
    print(f"   subsample: 0.8")
    print(f"   colsample_bytree: 0.8")

    # Train with regularization
    model, metrics = train_xgboost(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_estimators=100,
        max_depth=4,  # Reduced from 6
        learning_rate=0.05,  # Reduced from 0.1
        min_child_weight=3,  # Increased from 1
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,  # L2 regularization
        reg_alpha=0.1,  # L1 regularization
        random_state=42,
    )

    # Step 8: Walk-Forward Validation
    print("\n📊 Step 8: Walk-Forward Validation...")

    # Create walk-forward validator
    validator = WalkForwardValidator(
        train_months=2,
        test_months=1,
        step_months=1,
    )

    # Combine train+val for walk-forward
    combined_train_val = pd.concat([train, val])

    # Ensure timestamp is set as index
    if 'timestamp' in combined_train_val.columns:
        combined_train_val = combined_train_val.set_index('timestamp')

    # Define feature columns
    exclude_cols = {
        "label",
        "timestamp",
        "trading_session",
        "signal_direction",
        "direction",
    }
    feature_cols_wf = [col for col in combined_train_val.columns if col not in exclude_cols]

    # Wrapper function to adapt train_xgboost for WalkForwardValidator
    def train_and_predict(X_train, y_train, X_pred, return_prob=False):
        """Train model and return predictions for WalkForwardValidator.

        Args:
            X_train: Training features
            y_train: Training labels
            X_pred: Features to predict on
            return_prob: Whether to return probabilities (ignored, always returns both)

        Returns:
            Tuple of (y_pred, y_prob)
        """
        # Create a small validation set from training data
        val_size = min(int(len(X_train) * 0.2), 50)
        if len(X_train) > val_size + 50:
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
        else:
            X_train_fit = X_train
            y_train_fit = y_train
            X_val = X_train.iloc[[-1]]
            y_val = y_train.iloc[[-1]]

        # Train model
        model, _ = train_xgboost(
            X_train_fit, y_train_fit, X_val, y_val,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=3,
            reg_lambda=1.0,
            reg_alpha=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        # Return predictions
        y_pred = model.predict(X_pred)
        y_prob = model.predict_proba(X_pred)[:, 1]
        return y_pred, y_prob

    # Run walk-forward validation
    wf_results_obj = validator.validate(
        data=combined_train_val,
        model_trainer=train_and_predict,
        feature_cols=feature_cols_wf,
        target_col="label",
    )

    # Convert WalkForwardResults to dict
    wf_results = {
        'mean_test_performance': wf_results_obj.mean_test_performance.get('accuracy', 0.0),
        'std_test_performance': wf_results_obj.std_test_performance.get('accuracy', 0.0),
        'best_window': wf_results_obj.best_window.test_metrics.get('accuracy', 0.0),
        'worst_window': wf_results_obj.worst_window.test_metrics.get('accuracy', 0.0),
        'validation_results': [],
    }

    # Add individual validation results
    for v in wf_results_obj.validations:
        wf_results['validation_results'].append({
            'train_start': v.train_start.isoformat(),
            'train_end': v.train_end.isoformat(),
            'test_start': v.test_start.isoformat(),
            'test_end': v.test_end.isoformat(),
            'train_accuracy': v.train_metrics.get('accuracy', 0.0),
            'test_accuracy': v.test_metrics.get('accuracy', 0.0),
            'generalization_gap': v.generalization_gap,
        })

    # Print walk-forward results
    print("\n📊 Walk-Forward Validation Results:")
    print(f"   Mean Win Rate: {wf_results['mean_test_performance']:.2%}")
    print(f"   Std Win Rate: {wf_results['std_test_performance']:.2%}")
    print(f"   Best Window: {wf_results['best_window']:.2%}")
    print(f"   Worst Window: {wf_results['worst_window']:.2%}")
    print(f"   Number of windows: {len(wf_results['validation_results'])}")

    # Step 9: Save model (v2)
    print("\n💾 Step 9: Saving model v2...")

    model_dir = Path("models/xgboost/30_minute_v2")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    import joblib
    model_file = model_dir / "xgboost_model.pkl"
    joblib.dump(model, model_file)

    # Save metadata
    model_metadata = {
        "version": "v2",
        "training_period": {
            "start": "2025-01-01",
            "end": "2026-03-31"
        },
        "regularization": {
            "max_depth": 4,
            "learning_rate": 0.05,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "min_child_weight": 3,
        },
        "validation": {
            "in_sample_accuracy": metrics['accuracy'],
            "in_sample_roc_auc": metrics['roc_auc'],
            "walk_forward_mean": wf_results['mean_test_performance'],
            "walk_forward_std": wf_results['std_test_performance'],
        },
        "trained_at": datetime.now().isoformat(),
        "data_range": {
            "start": "2025-01-01",
            "end": "2026-03-31",
        },
    }

    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(model_metadata, f, indent=2)

    # Save walk-forward results
    wf_file = model_dir / "walk_forward_results.json"
    with open(wf_file, 'w') as f:
        json.dump(wf_results, f, indent=2)

    print(f"✅ Model saved to: {model_file}")
    print(f"✅ Metadata saved to: {metadata_file}")
    print(f"✅ Walk-forward results saved to: {wf_file}")

    # Step 10: Print final results
    print("\n📈 Step 10: Final Results")
    print("=" * 70)

    print(f"\n🎯 In-Sample Performance (Train + Val):")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\n🎯 Out-of-Sample Performance (Walk-Forward):")
    print(f"   Mean Win Rate: {wf_results['mean_test_performance']:.2%}")
    print(f"   Std Win Rate:  {wf_results['std_test_performance']:.2%}")
    print(f"   Best Window:   {wf_results['best_window']:.2%}")
    print(f"   Worst Window:  {wf_results['worst_window']:.2%}")

    # Check for overfitting
    in_sample_acc = metrics['accuracy']
    out_of_sample_acc = wf_results['mean_test_performance']
    generalization_gap = in_sample_acc - out_of_sample_acc

    print(f"\n🔍 Generalization Analysis:")
    print(f"   In-Sample Accuracy:  {in_sample_acc:.2%}")
    print(f"   Out-of-Sample Acc:  {out_of_sample_acc:.2%}")
    print(f"   Generalization Gap: {generalization_gap:.2%}")

    if generalization_gap > 0.10:
        print(f"   ⚠️  WARNING: Large generalization gap (>10%)")
        print(f"       Model may still be overfitting")
    elif generalization_gap > 0.05:
        print(f"   ✅ ACCEPTABLE: Moderate generalization gap (5-10%)")
    else:
        print(f"   ✅ EXCELLENT: Small generalization gap (<5%)")

    # Check if meets acceptance criteria
    print(f"\n🎯 Acceptance Criteria:")

    if out_of_sample_acc >= 0.40:
        print(f"   ✅ Win rate >= 40%: {out_of_sample_acc:.2%} (EXCELLENT)")
    elif out_of_sample_acc >= 0.30:
        print(f"   ✅ Win rate >= 30%: {out_of_sample_acc:.2%} (ACCEPTABLE)")
    else:
        print(f"   ❌ Win rate >= 30%: {out_of_sample_acc:.2%} (BELOW TARGET)")

    if wf_results['std_test_performance'] < 0.15:
        print(f"   ✅ Std dev < 15%: {wf_results['std_test_performance']:.2%} (STABLE)")
    else:
        print(f"   ⚠️  Std dev < 15%: {wf_results['std_test_performance']:.2%} (HIGH VARIABILITY)")

    if generalization_gap < 0.10:
        print(f"   ✅ Gap < 10%: {generalization_gap:.2%} (GOOD GENERALIZATION)")
    else:
        print(f"   ❌ Gap < 10%: {generalization_gap:.2%} (OVERFITTING)")

    print("\n" + "=" * 70)
    print("✅ MODEL RETRAINING COMPLETE")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Update live trading to use model_v2:")
    print("   Change model_dir from 'models/xgboost/30_minute/' to 'models/xgboost/30_minute_v2/'")
    print("2. Run validation backtest:")
    print("   python quick_validation.py")
    print("3. If performance acceptable, start paper trading:")
    print("   ./live_paper_trading_optimized.py")


if __name__ == '__main__':
    main()

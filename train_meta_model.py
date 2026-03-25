#!/usr/bin/env python3
"""Train XGBoost meta-labeling model for Silver Bullet signals.

This script trains a binary classifier to predict which Silver Bullet signals
will be profitable, enabling intelligent signal filtering.
"""

import sys
from pathlib import Path
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.label_mapper import map_signals_to_outcomes
from src.ml.signal_feature_extractor import SignalFeatureExtractor
from src.ml.meta_training_data_builder import MetaLabelingDatasetBuilder
from src.ml.xgboost_trainer import XGBoostTrainer
from src.ml.training_data import split_data, select_features


def load_time_bars(date_start: str, date_end: str) -> pd.DataFrame:
    """Load time-based bars for feature extraction."""
    print(f"📊 Loading time bars from {date_start} to {date_end}...")

    import h5py
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

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) & (combined.index <= end_dt)
    ]

    print(f"✅ Loaded {len(combined):,} time bars")

    return combined


def main():
    """Train meta-labeling model."""

    print("🚀 TRAINING META-LABELING MODEL")
    print("=" * 70)

    # Step 1: Load training data
    print("\n📊 Step 1: Loading training data...")

    signals_path = Path("data/ml_training/silver_bullet_signals.parquet")
    trades_path = Path("data/ml_training/silver_bullet_trades.parquet")
    metadata_path = Path("data/ml_training/metadata.json")

    if not signals_path.exists():
        print(f"❌ Signals file not found: {signals_path}")
        print("   Run: python run_optimized_silver_bullet.py --save-ml-data")
        return

    signals_df = pd.read_parquet(signals_path)
    trades_df = pd.read_parquet(trades_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"✅ Loaded {len(signals_df)} signals, {len(trades_df)} trades")
    print(f"   Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")

    # Validate minimum sample size
    if len(signals_df) < 100:
        print(f"❌ Insufficient training data: {len(signals_df)} signals")
        print(f"   Minimum 100 required. Extend backtest period.")
        return

    # Step 2: Load price data
    print("\n📊 Step 2: Loading price data for feature extraction...")
    price_data = load_time_bars(
        metadata['date_range']['start'],
        metadata['date_range']['end']
    )

    # Step 3: Build training dataset
    print("\n🔨 Step 3: Building training dataset...")

    feature_extractor = SignalFeatureExtractor(lookback_bars=100)
    dataset_builder = MetaLabelingDatasetBuilder(feature_extractor=feature_extractor)

    full_dataset = dataset_builder.build_dataset(
        signals_df=signals_df,
        trades_df=trades_df,
        price_data=price_data,
        verbose=True
    )

    print(f"\n✅ Dataset built: {len(full_dataset)} samples × {len(full_dataset.columns)} features")

    # Step 4: Select features
    print("\n🎯 Step 4: Selecting top features...")

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

    # Step 5: Split data (time-based)
    print("\n🔀 Step 5: Splitting data (time-based)...")

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

    # Step 6: Train model
    print("\n🤖 Step 6: Training XGBoost model...")

    trainer = XGBoostTrainer(model_dir='data/models/xgboost')

    # Prepare datasets for XGBoostTrainer (30-minute horizon)
    time_horizons = [30]

    datasets = {
        30: {
            'train': train,
            'val': val,
            'preprocessing_metadata': preprocessing_metadata
        }
    }

    models = trainer.train_models(
        datasets=datasets,
        time_horizons=time_horizons,
        perform_tuning=False,  # Use default hyperparameters
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    # Step 7: Print results
    print("\n📈 Step 7: Model Performance")
    print("=" * 70)

    model_data = models[30]
    metrics = model_data['metrics']
    importance = model_data['feature_importance']

    print(f"\nModel Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\n🎯 Top 10 Most Important Features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
        print(f"   {i:2d}. {feat:40s} {imp:.4f}")

    # Validate performance
    print(f"\n✅ Model saved to data/models/xgboost/30_minute/")

    # Check if performance meets criteria
    if metrics['roc_auc'] < 0.60:
        print(f"\n⚠️ WARNING: Low model performance (ROC-AUC: {metrics['roc_auc']:.4f})")
        print("   Consider:")
        print("   1. Hyperparameter tuning (perform_tuning=True)")
        print("   2. Feature engineering improvements")
        print("   3. Increasing training data size")
    else:
        print(f"\n✅ Model performance acceptable (ROC-AUC >= 0.60)")

    print("\n" + "=" * 70)
    print("✅ META-LABELING MODEL TRAINING COMPLETE")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Run: python run_meta_labeling_backtest.py")
    print("   This will test the model with A/B comparison")


if __name__ == '__main__':
    main()

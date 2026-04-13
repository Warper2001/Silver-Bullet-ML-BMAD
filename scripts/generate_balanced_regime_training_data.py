#!/usr/bin/env python3
"""Generate balanced regime-aware ML training data with real Silver Bullet labels.

This script creates training data for regime-specific models by:
1. Loading Silver Bullet signals with real trade outcomes
2. Adding HMM regime labels
3. Expanding date range to ensure adequate samples per regime
4. Applying data augmentation for undersampled regimes

Usage:
    python scripts/generate_balanced_regime_training_data.py [--start-date] [--end-date]
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dollar_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load dollar bar data.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading dollar bars from {start_date} to {end_date}")

    data_dir = Path("data/processed/dollar_bars/")
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    current = start_dt.replace(day=1)
    files = []

    while current <= end_dt:
        filename = f"MNQ_dollar_bars_{current.strftime('%Y%m')}.h5"
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
            logger.error(f"Failed to load {file_path.name}: {e}")

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) &
        (combined.index <= end_dt.replace(day=28, hour=23, minute=59))
    ]

    logger.info(f"Loaded {len(combined):,} dollar bars")

    return combined


def load_silver_bullet_signals_with_date_range() -> pd.DataFrame:
    """Load existing Silver Bullet signals with trade outcomes.

    Returns:
        DataFrame with signals and trade results
    """
    logger.info("Loading Silver Bullet signals with trade outcomes...")

    # Try to load from ML training data
    ml_dir = Path("data/ml_training")

    # Try full dataset first
    signals_path = ml_dir / "silver_bullet_signals_full.parquet"
    trades_path = ml_dir / "silver_bullet_trades_full.parquet"

    if not signals_path.exists():
        logger.warning(f"Full dataset not found: {signals_path}")
        logger.info("Please run: python generate_ml_training_data.py")
        return pd.DataFrame()

    signals_df = pd.read_parquet(signals_path)
    trades_df = pd.read_parquet(trades_path)

    logger.info(f"Loaded {len(signals_df):,} signals and {len(trades_df):,} trades")

    # Merge signals with trades
    # Trades has 'entry_time' column, signals has timestamp index
    # Set entry_time as index for trades to align
    trades_indexed = trades_df.set_index('entry_time')
    combined = signals_df.join(trades_indexed, how='inner', rsuffix='_trade')

    # Drop duplicate direction column from trades
    if 'direction_trade' in combined.columns:
        combined = combined.drop(columns=['direction_trade'])

    # Create binary label: 1 if profitable (return_pct > 0), 0 if not
    combined['label'] = (combined['return_pct'] > 0).astype(int)

    logger.info(f"Win rate: {combined['label'].mean():.2%}")
    logger.info(f"Total trades: {len(combined):,}")
    logger.info(f"Date range: {combined.index.min()} to {combined.index.max()}")

    return combined


def add_regime_labels(
    data: pd.DataFrame,
    signals_df: pd.DataFrame,
    detector: HMMRegimeDetector
) -> pd.DataFrame:
    """Add HMM regime labels to signals.

    Args:
        data: OHLCV dollar bar data
        signals_df: Signals DataFrame with timestamp index
        detector: Trained HMM regime detector

    Returns:
        DataFrame with regime labels added
    """
    logger.info("Adding HMM regime labels to signals...")

    # Engineer HMM features
    feature_engineer = HMMFeatureEngineer()
    hmm_features = feature_engineer.engineer_features(data)

    # Predict regimes for entire dataset
    logger.info("Predicting regimes for entire dataset...")
    regime_predictions = detector.predict(hmm_features)

    # Create regime series
    regime_series = pd.Series(regime_predictions, index=hmm_features.index)

    # Map each signal to its regime
    # Use the regime at the signal timestamp
    signals_with_regime = signals_df.copy()

    # For each signal, find the closest regime prediction
    regimes = []
    confidences = []

    for signal_time in signals_with_regime.index:
        # Find the closest time in regime predictions
        if signal_time in regime_series.index:
            regime = regime_series.loc[signal_time]
        else:
            # Find closest time
            idx = regime_series.index.get_indexer([signal_time], method='nearest')[0]
            regime = regime_series.iloc[idx]

        # Get confidence for this regime
        # For now, use a default confidence
        # In production, would extract from detector.predict_proba()
        confidence = 0.85  # Default high confidence

        regimes.append(regime)
        confidences.append(confidence)

    signals_with_regime['regime'] = regimes
    signals_with_regime['regime_confidence'] = confidences

    # Count signals per regime
    regime_counts = signals_with_regime['regime'].value_counts().sort_index()
    logger.info(f"Regime distribution:")
    for regime, count in regime_counts.items():
        regime_name = detector.metadata.regime_names[regime]
        logger.info(f"  Regime {regime} ({regime_name}): {count:,} signals ({count/len(signals_with_regime)*100:.1f}%)")

    return signals_with_regime


def add_ml_features(
    data: pd.DataFrame,
    signals_df: pd.DataFrame,
    feature_engineer: FeatureEngineer
) -> pd.DataFrame:
    """Add ML features to signals.

    Args:
        data: OHLCV dollar bar data
        signals_df: Signals DataFrame
        feature_engineer: ML feature engineer

    Returns:
        DataFrame with ML features added
    """
    logger.info("Adding ML features to signals...")

    # FeatureEngineer expects timestamp as a column, not index
    # Reset index to make timestamp a column
    data_with_timestamp = data.reset_index()

    # Generate features for entire dataset
    logger.info("Engineering features for entire dataset...")
    features_df = feature_engineer.engineer_features(data_with_timestamp)

    # The features_df will have a sequential integer index
    # We need to align it with the original timestamps
    # Create a mapping from row index to timestamp
    features_df['timestamp'] = data_with_timestamp['timestamp'].values
    features_df = features_df.set_index('timestamp')

    # For each signal, extract features at signal time
    feature_columns = []
    signal_features = []

    for signal_time in signals_df.index:
        if signal_time in features_df.index:
            features = features_df.loc[signal_time].to_dict()
        else:
            # Find closest time
            idx = features_df.index.get_indexer([signal_time], method='nearest')[0]
            features = features_df.iloc[idx].to_dict()

        signal_features.append(features)

    # Create DataFrame with features
    features_df_result = pd.DataFrame(signal_features, index=signals_df.index)

    # Combine with signals
    combined = pd.concat([signals_df, features_df_result], axis=1)

    logger.info(f"Added {len(features_df_result.columns)} features")

    return combined


def augment_minority_regimes(
    dataset: pd.DataFrame,
    target_samples_per_regime: int = 200,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply data augmentation to minority regimes.

    Uses SMOTE-like oversampling for minority classes:
    - Add small noise to features
    - Keep labels the same
    - Track which samples are augmented

    Args:
        dataset: Dataset with 'regime' column
        target_samples_per_regime: Target number of samples per regime
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (augmented_dataset, augmented_samples_only)
    """
    logger.info("Applying data augmentation to minority regimes...")

    np.random.seed(random_state)
    augmented_datasets = []
    augmented_samples_list = []

    # Get numeric feature columns (exclude metadata)
    exclude_columns = {'direction', 'confidence', 'mss_detected', 'fvg_detected',
                      'sweep_detected', 'time_window', 'direction_trade',
                      'exit_reason', 'return_pct', 'regime_confidence', 'label', 'regime'}

    feature_columns = [col for col in dataset.columns
                      if col not in exclude_columns and
                      dataset[col].dtype in ['int64', 'float64', 'int32', 'float32', 'float']]

    for regime in dataset['regime'].unique():
        regime_data = dataset[dataset['regime'] == regime].copy()

        if len(regime_data) >= target_samples_per_regime:
            # No augmentation needed
            # Mark all as not augmented
            regime_data['is_augmented'] = False
            augmented_datasets.append(regime_data)
            logger.info(f"  Regime {regime}: {len(regime_data)} samples (no augmentation)")
        else:
            # Need to augment
            current_samples = len(regime_data)
            needed_samples = target_samples_per_regime - current_samples
            n_augment = int(needed_samples * 1.2)  # Generate 20% extra, will filter later

            logger.info(f"  Regime {regime}: {current_samples} samples (augmenting by {n_augment} samples)")

            # Sample with replacement
            sample_indices = np.random.choice(current_samples, size=n_augment, replace=True)
            augmented_samples = regime_data.iloc[sample_indices].copy()

            # Add small noise to features (0.5% standard deviation)
            for col in feature_columns:
                noise = np.random.normal(0, regime_data[col].std() * 0.005, size=n_augment)
                augmented_samples[col] = augmented_samples[col].values + noise

            # Mark as augmented
            augmented_samples['is_augmented'] = True

            # Reset index to avoid duplicate indices
            augmented_samples = augmented_samples.reset_index(drop=True)

            # Track augmented samples for validation
            augmented_samples_list.append(augmented_samples)

            # Mark original samples as not augmented
            regime_data['is_augmented'] = False

            # Append original and augmented
            combined = pd.concat([regime_data, augmented_samples], ignore_index=True)
            augmented_datasets.append(combined)

            logger.info(f"  Regime {regime}: {len(combined)} samples after augmentation")

    # Combine all regimes
    result = pd.concat(augmented_datasets, ignore_index=True)
    augmented_only = pd.concat(augmented_samples_list, ignore_index=True) if augmented_samples_list else pd.DataFrame()

    logger.info(f"Total samples after augmentation: {len(result):,}")
    logger.info(f"Total augmented samples: {len(augmented_only):,}")

    return result, augmented_only


def create_regime_specific_datasets(
    signals_with_features: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    """Create separate datasets for each regime.

    Args:
        signals_with_features: DataFrame with signals, labels, regimes, and features

    Returns:
        Dict mapping regime number to dataset
    """
    logger.info("Creating regime-specific datasets...")

    # Log all columns for debugging
    logger.info(f"Total columns in signals_with_features: {len(signals_with_features.columns)}")

    regime_datasets = {}

    for regime in signals_with_features['regime'].unique():
        regime_data = signals_with_features[signals_with_features['regime'] == regime].copy()

        # Exclude non-numeric metadata columns
        exclude_columns = {'direction', 'confidence', 'mss_detected', 'fvg_detected',
                          'sweep_detected', 'time_window', 'direction_trade',
                          'exit_reason', 'return_pct', 'regime_confidence', 'label', 'regime'}

        # Get all numeric columns except the excluded ones
        feature_columns = [col for col in regime_data.columns
                          if col not in exclude_columns and
                          regime_data[col].dtype in ['int64', 'float64', 'int32', 'float32', 'float']]

        logger.info(f"Regime {regime}: Found {len(feature_columns)} numeric features")

        # Create clean dataset
        # Keep is_augmented column for validation
        dataset = regime_data[['label', 'regime', 'is_augmented'] + feature_columns].copy()

        # Drop rows with NaN
        dataset_before = len(dataset)
        dataset = dataset.dropna()
        dataset_after = len(dataset)

        logger.info(f"Regime {regime}: {dataset_before} samples before NaN drop, {dataset_after} samples after")

        regime_datasets[regime] = dataset

        if len(dataset) > 0:
            logger.info(f"Regime {regime}: {len(dataset):,} samples, {dataset['label'].mean():.2%} win rate")

    return regime_datasets


def save_regime_datasets(
    regime_datasets: dict[int, pd.DataFrame],
    output_dir: Path
) -> dict[str, str]:
    """Save regime-specific datasets to disk.

    Args:
        regime_datasets: Dict mapping regime to dataset
        output_dir: Output directory

    Returns:
        Dict mapping regime to file path
    """
    logger.info("Saving regime-specific datasets...")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for regime, dataset in regime_datasets.items():
        filename = output_dir / f"regime_{regime}_training_data.parquet"
        dataset.to_parquet(filename, index=True)
        saved_paths[f"regime_{regime}"] = str(filename)
        logger.info(f"Saved regime {regime} dataset to {filename}")

    return saved_paths


def generate_metadata(
    regime_datasets: dict[int, pd.DataFrame],
    saved_paths: dict[str, str],
    output_dir: Path,
    augmented: bool = False
) -> dict:
    """Generate metadata for regime datasets.

    Args:
        regime_datasets: Dict mapping regime to dataset
        saved_paths: Dict mapping regime to file path
        output_dir: Output directory
        augmented: Whether data augmentation was applied

    Returns:
        Metadata dict
    """
    logger.info("Generating metadata...")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_regimes": len(regime_datasets),
        "augmented": augmented,
        "regimes": {}
    }

    for regime, dataset in regime_datasets.items():
        regime_metadata = {
            "regime_number": int(regime),
            "file_path": saved_paths[f"regime_{regime}"],
            "n_samples": int(len(dataset)),
            "n_features": int(len(dataset.columns) - 2),  # Exclude label and regime
            "win_rate": float(dataset['label'].mean()),
            "n_winners": int(dataset['label'].sum()),
            "n_losers": int(len(dataset) - dataset['label'].sum())
        }
        metadata["regimes"][f"regime_{regime}"] = regime_metadata

    # Save metadata
    metadata_path = output_dir / "regime_aware_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")

    return metadata


def main():
    """Main generation pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("BALANCED REGIME-AWARE TRAINING DATA GENERATION")
    logger.info("=" * 70)

    try:
        # Step 1: Load HMM model
        logger.info("\nStep 1: Loading HMM regime detector...")
        model_dir = Path("models/hmm/regime_model")

        if not model_dir.exists():
            logger.error(f"HMM model not found: {model_dir}")
            logger.info("Run: python scripts/train_hmm_regime_detector.py")
            return

        detector = HMMRegimeDetector.load(model_dir)
        logger.info(f"✅ Loaded HMM model: {detector.n_regimes} regimes")

        # Step 2: Load Silver Bullet signals
        logger.info("\nStep 2: Loading Silver Bullet signals...")
        signals_df = load_silver_bullet_signals_with_date_range()

        if signals_df.empty:
            logger.error("No signals found. Please run: python generate_ml_training_data.py")
            return

        # Step 3: Load dollar bars for feature engineering
        # Use expanded date range to get more data
        logger.info("\nStep 3: Loading dollar bar data...")
        data = load_dollar_bars("2023-12-01", "2025-03-31")

        # Step 4: Add regime labels
        logger.info("\nStep 4: Adding regime labels...")
        signals_with_regime = add_regime_labels(data, signals_df, detector)

        # Step 5: Add ML features
        logger.info("\nStep 5: Adding ML features...")
        feature_engineer = FeatureEngineer()
        signals_with_features = add_ml_features(data, signals_with_regime, feature_engineer)

        # Step 6: Check balance and apply augmentation if needed
        logger.info("\nStep 6: Checking class balance...")
        regime_counts = signals_with_features['regime'].value_counts()
        min_samples = 200  # Target minimum samples per regime

        # Convert values to list for iteration
        needs_augmentation = any(count < min_samples for count in list(regime_counts.values))

        augmented_samples_only = None
        if needs_augmentation:
            logger.info(f"Applying data augmentation (target: {min_samples} samples per regime)...")
            signals_with_features, augmented_samples_only = augment_minority_regimes(
                signals_with_features,
                target_samples_per_regime=min_samples
            )
        else:
            logger.info("Dataset is balanced, no augmentation needed")
            # Mark all as not augmented
            signals_with_features['is_augmented'] = False

        # Step 7: Create regime-specific datasets
        logger.info("\nStep 7: Creating regime-specific datasets...")
        regime_datasets = create_regime_specific_datasets(signals_with_features)

        # Step 8: Save datasets
        logger.info("\nStep 8: Saving datasets...")
        output_dir = Path("data/ml_training/regime_aware_balanced")
        saved_paths = save_regime_datasets(regime_datasets, output_dir)

        # Save augmented samples separately for validation
        if augmented_samples_only is not None and len(augmented_samples_only) > 0:
            augmented_path = output_dir / "augmented_samples_only.parquet"
            augmented_samples_only.to_parquet(augmented_path, index=True)
            logger.info(f"Saved augmented samples to {augmented_path}")

        # Step 9: Generate metadata
        logger.info("\nStep 9: Generating metadata...")
        metadata = generate_metadata(
            regime_datasets,
            saved_paths,
            output_dir,
            augmented=needs_augmentation
        )

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ BALANCED REGIME-AWARE TRAINING DATA GENERATION COMPLETE")
        logger.info("=" * 70)

        logger.info("\n📊 Dataset Summary:")
        total_samples = sum(len(ds) for ds in regime_datasets.values())
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Regimes: {len(regime_datasets)}")
        logger.info(f"   Augmented: {needs_augmentation}")

        for regime_key, regime_meta in metadata["regimes"].items():
            logger.info(f"\n   {regime_key}:")
            logger.info(f"     Samples: {regime_meta['n_samples']:,}")
            logger.info(f"     Win rate: {regime_meta['win_rate']:.2%}")
            logger.info(f"     Features: {regime_meta['n_features']}")

        logger.info(f"\n📁 Output directory: {output_dir}")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Review dataset metadata: {output_dir}/regime_aware_metadata.json")
        logger.info(f"2. Train regime-specific models with real labels")

    except Exception as e:
        logger.error(f"\n❌ Generation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

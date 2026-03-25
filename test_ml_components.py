#!/usr/bin/env python3
"""Quick test of ML components."""

import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.label_mapper import map_signals_to_outcomes
from src.ml.signal_feature_extractor import SignalFeatureExtractor
from src.ml.meta_training_data_builder import MetaLabelingDatasetBuilder

def load_time_bars(date_start: str, date_end: str) -> pd.DataFrame:
    """Load time-based bars for testing."""
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

    return combined

def main():
    print("🧪 Testing ML Components")
    print("=" * 70)

    # Load data
    print("\n📊 Loading data...")
    signals_df = pd.read_parquet('data/ml_training/silver_bullet_signals.parquet')
    trades_df = pd.read_parquet('data/ml_training/silver_bullet_trades.parquet')
    price_data = load_time_bars('2024-10-01', '2025-03-05')

    print(f"✅ Loaded {len(signals_df)} signals, {len(trades_df)} trades, {len(price_data)} price bars")

    # Test 1: Label Mapper
    print("\n🧪 Test 1: Label Mapper")
    print("-" * 70)
    labeled_signals = map_signals_to_outcomes(signals_df, trades_df)
    print(f"✅ Labeled {len(labeled_signals)} signals")
    print(f"   Label distribution: {labeled_signals['label'].value_counts().to_dict()}")
    assert len(labeled_signals) == len(signals_df), "Label count mismatch!"
    assert 'label' in labeled_signals.columns, "Missing label column!"
    print("✅ Label mapper test PASSED")

    # Test 2: Feature Extractor
    print("\n🧪 Test 2: Signal Feature Extractor")
    print("-" * 70)
    extractor = SignalFeatureExtractor(lookback_bars=100)

    # Test single signal
    sample_signal_time = signals_df.index[0]
    single_features = extractor.extract_features_at_signal_time(sample_signal_time, price_data)
    print(f"✅ Extracted {len(single_features)} features for single signal")
    assert len(single_features) >= 40, "Insufficient features extracted!"

    # Test all signals (subset for speed)
    subset_signals = signals_df.head(20)
    all_features = extractor.extract_for_all_signals(subset_signals, price_data, verbose=True)
    print(f"✅ Extracted features for {len(all_features)} signals")
    print(f"   Feature columns: {len(all_features.columns)}")
    assert len(all_features) == len(subset_signals), "Feature count mismatch!"
    assert len(all_features.columns) >= 40, "Insufficient feature columns!"
    print("✅ Feature extractor test PASSED")

    # Test 3: Dataset Builder
    print("\n🧪 Test 3: Meta-Labeling Dataset Builder")
    print("-" * 70)
    builder = MetaLabelingDatasetBuilder(feature_extractor=extractor)

    # Build dataset with subset
    subset_signals = signals_df.head(20)
    dataset = builder.build_dataset(
        signals_df=subset_signals,
        trades_df=trades_df,
        price_data=price_data,
        verbose=True
    )

    print(f"✅ Built dataset: {len(dataset)} samples × {len(dataset.columns)} features")
    assert 'label' in dataset.columns, "Missing label column!"
    assert len(dataset) > 0, "Empty dataset!"
    print("✅ Dataset builder test PASSED")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)

if __name__ == '__main__':
    main()

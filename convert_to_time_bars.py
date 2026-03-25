#!/usr/bin/env python3
"""Convert MNQ tick data to time-based bars with killzone filtering.

This script loads real MNQ data from TradeStation, filters to killzones only,
and saves time-based bars for proper Silver Bullet pattern detection.
"""

import json
import logging
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MNQ contract specifications
MNQ_MULTIPLIER = 0.5  # $0.5 per point per contract

# Killzone times (EST)
LONDON_AM_START = 2  # 2 AM EST
LONDON_AM_END = 8    # 8 AM EST
NY_AM_START = 9       # 9 AM EST
NY_AM_END = 11        # 11 AM EST
NY_PM_START = 13      # 1 PM EST
NY_PM_END = 16        # 4 PM EST


def is_in_killzone(timestamp: pd.Timestamp) -> bool:
    """Check if timestamp falls within a killzone.

    Args:
        timestamp: Pandas timestamp (assumed UTC)

    Returns:
        True if in killzone, False otherwise
    """
    # Convert UTC to EST (UTC-5)
    est_time = timestamp - timedelta(hours=5)
    hour = est_time.hour

    # Check killzones
    if LONDON_AM_START <= hour < LONDON_AM_END:
        return True
    if NY_AM_START <= hour < NY_AM_END:
        return True
    if NY_PM_START <= hour < NY_PM_END:
        return True

    return False


def load_tradestation_data(file_path: str) -> pd.DataFrame:
    """Load MNQ data from TradeStation JSON file.

    Args:
        file_path: Path to mnq_historical.json

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading data from {file_path}...")

    with open(file_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data):,} records")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['TimeStamp'])

    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'TotalVolume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename columns to lowercase for consistency
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'TotalVolume': 'volume'
    })

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Sort by timestamp
    df = df.sort_index()

    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def filter_to_killzones(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to only include killzone bars.

    Args:
        df: DataFrame with timestamp index

    Returns:
        DataFrame filtered to killzones only
    """
    logger.info("Filtering to killzones only...")

    # Apply killzone filter
    killzone_mask = df.index.map(is_in_killzone)
    killzone_df = df[killzone_mask].copy()

    reduction_pct = (1 - len(killzone_df) / len(df)) * 100
    logger.info(f"Original bars: {len(df):,}")
    logger.info(f"Killzone bars: {len(killzone_df):,}")
    logger.info(f"Reduction: {reduction_pct:.1f}%")

    return killzone_df


def resample_to_time_bars(df: pd.DataFrame, frequency: str = '5min') -> pd.DataFrame:
    """Resample data to time-based bars.

    Args:
        df: DataFrame with timestamp index
        frequency: Pandas resample frequency ('1min', '5min', etc.)

    Returns:
        DataFrame with resampled OHLCV data
    """
    logger.info(f"Resampling to {frequency} bars...")

    # Resample
    resampled = df.resample(frequency).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Remove rows with NaN (gaps in data)
    resampled = resampled.dropna()

    logger.info(f"Generated {len(resampled):,} {frequency} bars")
    logger.info(f"Date range: {resampled.index.min()} to {resampled.index.max()}")

    return resampled


def save_to_hdf5(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to HDF5 format compatible with existing loaders.

    Args:
        df: DataFrame with time-based bars
        output_path: Path to output HDF5 file
    """
    logger.info(f"Saving to {output_path}...")

    # Prepare data array
    data_array = []
    for idx, row in df.iterrows():
        # Convert timestamp to milliseconds since epoch
        ts_ms = int(idx.timestamp() * 1000)

        # Calculate notional value
        notional_value = row['close'] * row['volume'] * MNQ_MULTIPLIER

        data_array.append([
            ts_ms,
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            int(row['volume']),
            notional_value
        ])

    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            "dollar_bars",
            data=np.array(data_array),
            compression="gzip"
        )

    logger.info(f"✅ Saved {len(df)} bars to {output_path}")


def save_monthly_files(df: pd.DataFrame, output_dir: str) -> None:
    """Save data to monthly HDF5 files.

    Args:
        df: DataFrame with time-based bars
        output_dir: Output directory for HDF5 files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by month
    df_grouped = df.groupby(pd.Grouper(freq='ME'))

    for month, month_df in df_grouped:
        month_str = month.strftime("%Y%m")
        filename = f"MNQ_time_bars_5min_{month_str}.h5"
        filepath = output_path / filename

        save_to_hdf5(month_df, str(filepath))


def main():
    """Convert MNQ data to time-based killzone-filtered bars."""

    print("🎯 CONVERTING MNQ DATA TO TIME-BASED KILLZONE BARS")
    print("=" * 70)

    # Step 1: Load data
    df = load_tradestation_data('mnq_historical.json')

    # Step 2: Filter to killzones
    killzone_df = filter_to_killzones(df)

    # Step 3: Resample to 5-minute bars
    time_bars = resample_to_time_bars(killzone_df, frequency='5min')

    # Step 4: Save to monthly HDF5 files
    print("\n💾 Saving to monthly HDF5 files...")
    save_monthly_files(time_bars, 'data/processed/time_bars/')

    print("\n✅ CONVERSION COMPLETE!")
    print(f"\n📊 Summary:")
    print(f"   Input: {len(df):,} 1-minute bars")
    print(f"   Killzone filtered: {len(killzone_df):,} bars")
    print(f"   Time bars: {len(time_bars):,} 5-minute bars")
    print(f"   Date range: {time_bars.index.min()} to {time_bars.index.max()}")
    print(f"   Price range: ${time_bars['close'].min():.2f} - ${time_bars['close'].max():.2f}")

    # Sample of data
    print(f"\n📈 Sample Data (last 5 bars):")
    print(time_bars[['open', 'high', 'low', 'close', 'volume']].tail().to_string())
    print(f"\n✅ Time bars saved to data/processed/time_bars/")
    print(f"   Ready for Silver Bullet pattern detection!")


if __name__ == '__main__':
    main()

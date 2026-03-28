#!/usr/bin/env python3
"""Process real MNQ historical data from JSON to dollar bars in HDF5 format."""

import json
import h5py
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# MNQ contract specifications
MNQ_MULTIPLIER = 0.5  # $0.5 per point per contract
DOLLAR_THRESHOLD = 50_000_000  # $50M notional value per bar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_tradestation_json(file_path: str) -> pd.DataFrame:
    """Load TradeStation JSON historical data.

    Args:
        file_path: Path to JSON file

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading data from {file_path}...")

    with open(file_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} records from JSON")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['TimeStamp'])

    # Convert price columns to float
    df['open'] = df['Open'].astype(float)
    df['high'] = df['High'].astype(float)
    df['low'] = df['Low'].astype(float)
    df['close'] = df['Close'].astype(float)
    df['volume'] = df['TotalVolume'].astype(int)

    # Calculate notional value
    df['notional'] = df['close'] * df['volume'] * MNQ_MULTIPLIER

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional']]


def create_dollar_bars(df: pd.DataFrame, threshold: float = DOLLAR_THRESHOLD) -> pd.DataFrame:
    """Aggregate tick data into dollar bars.

    Args:
        df: DataFrame with tick data
        threshold: Dollar threshold per bar

    Returns:
        DataFrame with dollar bars
    """
    logger.info(f"Creating dollar bars with ${threshold:,.0f} threshold...")

    bars = []
    current_bar = {
        'timestamp': None,
        'open': None,
        'high': None,
        'low': None,
        'close': None,
        'volume': 0,
        'notional': 0.0
    }

    for idx, row in df.iterrows():
        # Initialize new bar if needed
        if current_bar['timestamp'] is None:
            current_bar['timestamp'] = row['timestamp']
            current_bar['open'] = row['open']
            current_bar['high'] = row['high']
            current_bar['low'] = row['low']
            current_bar['close'] = row['close']
            current_bar['volume'] = row['volume']
            current_bar['notional'] = row['notional']
        else:
            # Update current bar
            current_bar['high'] = max(current_bar['high'], row['high'])
            current_bar['low'] = min(current_bar['low'], row['low'])
            current_bar['close'] = row['close']
            current_bar['volume'] += row['volume']
            current_bar['notional'] += row['notional']

        # Check if threshold reached
        if current_bar['notional'] >= threshold:
            bars.append(current_bar.copy())
            current_bar = {
                'timestamp': None,
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': 0,
                'notional': 0.0
            }

            if len(bars) % 1000 == 0:
                logger.info(f"Created {len(bars)} bars...")

    # Don't forget the last bar
    if current_bar['timestamp'] is not None:
        bars.append(current_bar)

    bars_df = pd.DataFrame(bars)

    logger.info(f"Created {len(bars_df)} dollar bars")
    logger.info(f"Date range: {bars_df['timestamp'].min()} to {bars_df['timestamp'].max()}")

    return bars_df


def save_to_hdf5(df: pd.DataFrame, output_dir: str = "data/processed/dollar_bars") -> None:
    """Save DataFrame to monthly HDF5 files.

    Args:
        df: DataFrame with dollar bar data
        output_dir: Output directory for HDF5 files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by month
    df['month'] = df['timestamp'].dt.to_period('M')

    for month, month_df in df.groupby('month'):
        month_str = month.strftime("%Y%m")
        filename = f"MNQ_dollar_bars_{month_str}.h5"
        filepath = output_path / filename

        # Prepare data for HDF5
        data_array = []
        for _, row in month_df.iterrows():
            ts_ms = int(row["timestamp"].timestamp() * 1000)
            data_array.append([
                ts_ms,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["notional"]
            ])

        # Save to HDF5
        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "dollar_bars",
                data=np.array(data_array),
                compression="gzip"
            )

        logger.info(f"✅ Saved {len(month_df)} bars to {filepath}")


def main():
    """Process real MNQ data from JSON to HDF5 dollar bars."""

    logger.info("🎯 Processing real MNQ historical data...")

    # Step 1: Load JSON data
    df = load_tradestation_json("mnq_historical.json")

    if df.empty:
        logger.error("❌ No data loaded!")
        return

    # Step 2: Create dollar bars
    bars_df = create_dollar_bars(df)

    if bars_df.empty:
        logger.error("❌ No bars created!")
        return

    # Step 3: Save to HDF5
    save_to_hdf5(bars_df)

    logger.info("✅ Real MNQ data processing completed!")
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Total ticks processed: {len(df):,}")
    logger.info(f"   Total dollar bars created: {len(bars_df):,}")
    logger.info(f"   Date range: {bars_df['timestamp'].min()} to {bars_df['timestamp'].max()}")
    logger.info(f"   Price range: ${bars_df['close'].min():.2f} - ${bars_df['close'].max():.2f}")

    # Show sample
    print(f"\n📈 Sample Dollar Bars (last 5):")
    print(bars_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate realistic MNQ test data for backtesting.

This script generates synthetic but realistic MNQ futures data for testing
the backtesting framework while waiting for real market data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import h5py
import logging

logger = logging.getLogger(__name__)


def generate_realistic_mnq_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    bar_size: str = "5min"
) -> pd.DataFrame:
    """Generate realistic MNQ futures price data.

    Uses geometric Brownian motion with realistic MNQ characteristics:
    - Price around $21,000
    - Volatility of 0.2% per 5-minute bar
    - Trading hours 6pm-5pm CT (Sunday-Friday)
    - Volume patterns (higher at open/close)

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bar_size: Bar size (5min, 1min, 1hour)

    Returns:
        DataFrame with realistic OHLCV data
    """
    np.random.seed(42)

    # Setup date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Generate timestamps for trading hours only
    # MNQ trades Sunday-Friday, 6:00pm - 5:00pm CT
    timestamps = pd.date_range(start, end, freq=bar_size)

    # Filter for trading hours
    trading_hours = []
    for ts in timestamps:
        # Skip Saturday (day 5)
        if ts.dayofweek == 5:
            continue
        # Trading: 6pm-12am and 12am-5pm CT
        hour = ts.hour
        if 0 <= hour < 17 or 18 <= hour < 24:
            trading_hours.append(ts)

    if not trading_hours:
        return pd.DataFrame()

    logger.info(f"Generating {len(trading_hours)} bars for {start_date} to {end_date}")

    # Generate price paths using GBM
    base_price = 21000  # MNQ typical level
    dt = 5 / (252 * 117)  # 5-min bars in a trading year
    mu = 0.05  # 5% annual drift
    sigma = 0.3  # 30% annual volatility

    # Generate returns
    returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        len(trading_hours)
    )

    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(trading_hours, prices)):
        # Generate realistic intrabar noise
        noise = np.random.normal(0, 0.0002, 4)  # Small noise for OHLC

        open_price = close * (1 + noise[0])
        high = close * (1 + abs(noise[1]) + 0.0003)  # High > close
        low = close * (1 - abs(noise[2]) - 0.0003)   # Low < close

        # Fix OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate volume with time-of-day pattern
        hour = timestamp.hour
        if 8 <= hour <= 10:  # Morning rush
            base_volume = np.random.randint(500, 800)
        elif 14 <= hour <= 16:  # Close rush
            base_volume = np.random.randint(600, 900)
        elif 0 <= hour <= 1:  # Sunday evening open
            base_volume = np.random.randint(400, 700)
        else:  # Regular trading
            base_volume = np.random.randint(200, 500)

        volume = base_volume + np.random.randint(-50, 50)
        volume = max(volume, 100)  # Minimum volume

        # Calculate dollar volume (MNQ: $2 per point)
        dollar_volume = volume * close * 2

        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'dollar_volume': round(dollar_volume, 2)
        })

    df = pd.DataFrame(data)

    # Add some price gaps (weekend gaps, overnight moves)
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']

    # Create gaps at day boundaries (simulating overnight moves)
    day_change = df['timestamp'].dt.day != df['timestamp'].shift(1).dt.day
    gap_indices = df[day_change].index

    for idx in gap_indices:
        if idx > 0:
            gap_size = np.random.uniform(-0.003, 0.003)  # ±0.3% gap
            df.loc[idx, 'open'] *= (1 + gap_size)
            df.loc[idx, 'high'] *= (1 + gap_size)
            df.loc[idx, 'low'] *= (1 + gap_size)
            df.loc[idx, 'close'] *= (1 + gap_size)

    # Recalculate after gaps
    for idx in gap_indices:
        if idx > 0:
            df.loc[idx, 'high'] = max(df.loc[idx, 'open'], df.loc[idx, 'high'], df.loc[idx, 'close'])
            df.loc[idx, 'low'] = min(df.loc[idx, 'open'], df.loc[idx, 'low'], df.loc[idx, 'close'])

    df = df.drop(columns=['prev_close', 'gap'])

    logger.info(f"Generated realistic data with {len(df)} bars")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"Average daily volume: {df.groupby(df['timestamp'].dt.date)['volume'].sum().mean():.0f}")

    return df


def save_to_hdf5(df: pd.DataFrame, output_dir: str = "data/processed/dollar_bars") -> None:
    """Save DataFrame to monthly HDF5 files.

    Args:
        df: DataFrame with OHLCV data
        output_dir: Output directory for HDF5 files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by month and save
    df_grouped = df.set_index("timestamp").resample("ME")

    for month_start, month_df in df_grouped:
        month_str = month_start.strftime("%Y%m")
        filename = f"MNQ_dollar_bars_{month_str}.h5"
        filepath = output_path / filename

        # Reset index to have timestamp as column
        month_df = month_df.reset_index()

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
                row["dollar_volume"]
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
    """Generate test data for backtesting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("🎯 Generating realistic MNQ test data...")

    # Generate 6 months of test data
    df = generate_realistic_mnq_data(
        start_date="2024-09-01",
        end_date="2025-03-19",
        bar_size="5min"
    )

    if df.empty:
        logger.error("❌ No data generated!")
        return

    # Save to HDF5
    save_to_hdf5(df)

    logger.info("✅ Test data generation completed!")
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Total bars: {len(df):,}")
    logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"   Average volume: {df['volume'].mean():.0f} contracts/bar")

    # Show sample of data
    print(f"\n📈 Sample Data (last 5 bars):")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
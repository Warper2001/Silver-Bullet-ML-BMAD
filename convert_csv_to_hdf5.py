#!/usr/bin/env python3
"""Convert MNQ CSV data to HDF5 format for backtesting.

This script converts downloaded CSV data (from CME, Interactive Brokers, etc.)
into the HDF5 format required by the Silver Bullet ML system.

Usage:
    python convert_csv_to_hdf5.py --input mnq_data.csv --output mnq_converted.h5
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


# MNQ contract multiplier
MNQ_MULTIPLIER = 0.5


def parse_csv_file(csv_path: Path) -> pd.DataFrame:
    """Parse CSV file with flexible column mapping.

    Supports common CSV formats from:
    - CME Group
    - Interactive Brokers
    - TradingView exports
    - Generic OHLCV formats

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with standardized columns
    """
    # Try different CSV parsing strategies
    try:
        # Try reading first to detect format
        with open(csv_path, 'r') as f:
            first_line = f.readline()

        # Detect delimiter
        if ',' in first_line:
            delimiter = ','
        elif ';' in first_line:
            delimiter = ';'
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','

        # Read CSV
        df = pd.read_csv(csv_path, delimiter=delimiter)

        # Column name mapping (case-insensitive)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()

            if 'date' in col_lower or 'time' in col_lower:
                column_map[col] = 'timestamp'
            elif 'open' in col_lower:
                column_map[col] = 'open'
            elif 'high' in col_lower:
                column_map[col] = 'high'
            elif 'low' in col_lower:
                column_map[col] = 'low'
            elif 'close' in col_lower or 'last' in col_lower:
                column_map[col] = 'close'
            elif 'volume' in col_lower:
                column_map[col] = 'volume'

        # Rename columns
        if column_map:
            df = df.rename(columns=column_map)

        # Verify required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"⚠️  Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            print("\nExpected columns: timestamp, open, high, low, close, volume")
            print("Mapping will attempt to find closest matches...")

            # Try to auto-detect columns by position
            if len(df.columns) >= 6:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + list(df.columns[6:])
                print("✓ Auto-mapped columns by position")

        return df

    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")


def standardize_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Standardize dataframe format.

    Args:
        df: Input dataframe
        symbol: MNQ contract symbol (e.g., MNQM26)

    Returns:
        Standardized dataframe
    """
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Handle timezone
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing critical data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate notional value
    df['notional_value'] = df['close'] * df['volume'] * MNQ_MULTIPLIER

    # Basic validation for MNQ price ranges
    # MNQ typically trades around 20,000-25,000 (as of 2026)
    price_mean = df['close'].mean()
    if price_mean < 1000 or price_mean > 100000:
        print(f"\n⚠️  Warning: Unusual price range detected!")
        print(f"   Mean price: ${price_mean:,.2f}")
        print(f"   Typical MNQ range: $20,000 - $25,000")
        print(f"   Please verify your data is for MNQ (Micro E-mini Nasdaq-100)")

    return df


def save_to_hdf5(df: pd.DataFrame, output_path: Path, symbol: str) -> None:
    """Save dataframe to HDF5 format.

    Args:
        df: Standardized dataframe
        output_path: Output HDF5 file path
        symbol: MNQ contract symbol
    """
    # Prepare structured numpy array
    dt = np.dtype([
        ("timestamp", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "i8"),
        ("notional_value", "f8"),
    ])

    # Convert to numpy array
    data = np.zeros(len(df), dtype=dt)

    for i, row in df.iterrows():
        data[i] = (
            int(row['timestamp'].timestamp() * 1e9),  # nanoseconds
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            int(row['volume']),
            row['notional_value'],
        )

    # Write to HDF5
    with h5py.File(output_path, "w") as h5file:
        dataset = h5file.create_dataset(
            "historical_bars",
            data=data,
            compression="gzip",
            compression_opts=1,
        )

        # Add metadata
        dataset.attrs["symbol"] = symbol
        dataset.attrs["count"] = len(df)
        dataset.attrs["created_at"] = datetime.utcnow().isoformat()
        dataset.attrs["multiplier"] = MNQ_MULTIPLIER
        dataset.attrs["date_range"] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Successfully saved to {output_path}")
    print(f"   Bars: {len(df):,}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MNQ CSV data to HDF5 format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="MNQM26",
        help="MNQ contract symbol (default: MNQM26)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MNQ CSV to HDF5 Converter")
    print("=" * 70)

    # Parse CSV
    print(f"\n📂 Reading CSV: {args.input}")
    df = parse_csv_file(args.input)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Display sample data
    print("\n📊 Sample data (first 3 rows):")
    print(df.head(3).to_string())

    # Standardize
    print("\n⚙️  Standardizing format...")
    df = standardize_dataframe(df, args.symbol)
    print(f"✓ Standardized {len(df)} bars")

    # Save to HDF5
    print(f"\n💾 Saving to HDF5: {args.output}")
    save_to_hdf5(df, args.output, args.symbol)

    print("\n" + "=" * 70)
    print("✅ Conversion complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Place file in: data/historical/mnq/")
    print(f"2. Verify with: python -m pytest tests/")
    print(f"3. Run backtest with your strategy")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Triple-Barrier Labels for 1-Minute Dollar Bar Trading

Implements Lopez de Prado's triple-barrier method for 1-minute timeframe:
- Vertical barrier: 30 minutes (time exit)
- Horizontal barriers: TP 0.3%, SL 0.2% (from current system)
- Transaction costs incorporated into barrier calculations
- Labels: 1 (TP hit), -1 (SL hit), 0 (time exit)

Reference: Advances in Financial Machine Learning, Chapter 3
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    tp_barrier: float = 0.003,  # 0.3% take profit
    sl_barrier: float = 0.002,  # 0.2% stop loss
    vertical_barrier: int = 30,  # 30 minutes
    transaction_cost_bps: int = 2,  # 2 bps round-trip
    min_return: float = 0.0005  # Minimum 0.05% return to label
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for dollar bar data.

    Parameters
    ----------
    df : pd.DataFrame
        Dollar bar data with columns: timestamp, open, high, low, close, volume
    tp_barrier : float
        Take profit barrier (decimal, e.g., 0.003 = 0.3%)
    sl_barrier : float
        Stop loss barrier (decimal, e.g., 0.002 = 0.2%)
    vertical_barrier : int
        Number of bars for vertical barrier (time exit)
    transaction_cost_bps : int
        Transaction cost in basis points (round-trip)
    min_return : float
        Minimum return threshold to assign label (filters noise)

    Returns
    -------
    pd.DataFrame
        Original data with additional columns:
        - label: Triple-barrier label (1, -1, 0)
        - barrier_hit: Which barrier was hit ('tp', 'sl', 'vertical', 'none')
        - bars_to_barrier: Number of bars until barrier hit
        - max_profit: Maximum profit achieved during barrier period
        - max_loss: Maximum loss achieved during barrier period
    """

    df = df.copy()
    df = df.reset_index(drop=True)

    # Adjust barriers for transaction costs
    # TP needs to be higher to cover costs, SL needs to be tighter to limit loss
    cost_adjusted_tp = tp_barrier - (transaction_cost_bps / 10000)
    cost_adjusted_sl = sl_barrier + (transaction_cost_bps / 10000)

    # Initialize label arrays
    labels = np.zeros(len(df))
    barrier_hit = ['none'] * len(df)
    bars_to_barrier = np.zeros(len(df))
    max_profit = np.zeros(len(df))
    max_loss = np.zeros(len(df))

    # Compute triple-barrier labels
    for i in range(len(df) - vertical_barrier):
        if i % 10000 == 0:
            print(f"Processing bar {i}/{len(df)}")

        entry_price = df.loc[i, 'close']
        entry_time = df.loc[i, 'timestamp'] if 'timestamp' in df.columns else i

        # Look ahead for barrier hits
        hit_barrier = None
        hit_bar = None
        max_prof = 0.0
        max_loss_price = 0.0

        for j in range(i + 1, min(i + vertical_barrier + 1, len(df))):
            high = df.loc[j, 'high']
            low = df.loc[j, 'low']

            # Calculate potential profit/loss
            potential_profit = (high - entry_price) / entry_price
            potential_loss = (entry_price - low) / entry_price

            # Track max profit/loss during period
            max_prof = max(max_prof, potential_profit)
            max_loss_price = max(max_loss_price, potential_loss)

            # Check if barriers hit
            if potential_profit >= cost_adjusted_tp:
                hit_barrier = 'tp'
                hit_bar = j
                break
            elif potential_loss >= cost_adjusted_sl:
                hit_barrier = 'sl'
                hit_bar = j
                break

        # Assign label based on which barrier hit first
        if hit_barrier == 'tp':
            labels[i] = 1
            barrier_hit[i] = 'tp'
            bars_to_barrier[i] = hit_bar - i if hit_bar else vertical_barrier
        elif hit_barrier == 'sl':
            labels[i] = -1
            barrier_hit[i] = 'sl'
            bars_to_barrier[i] = hit_bar - i if hit_bar else vertical_barrier
        else:
            # Vertical barrier hit (time exit)
            labels[i] = 0
            barrier_hit[i] = 'vertical'
            bars_to_barrier[i] = vertical_barrier

        max_profit[i] = max_prof
        max_loss[i] = max_loss_price

    # Add columns to dataframe
    df['label'] = labels
    df['barrier_hit'] = barrier_hit
    df['bars_to_barrier'] = bars_to_barrier
    df['max_profit'] = max_profit
    df['max_loss'] = max_loss

    # Filter: Remove labels where max movement < min_return (noise filter)
    # This filters out bars that didn't move enough to be meaningful
    noise_filter = (df['max_profit'] > min_return) | (df['max_loss'] > min_return)

    print(f"\n=== Label Statistics ===")
    print(f"Total bars: {len(df)}")
    print(f"TP hits (label=1): {(df['label'] == 1).sum()}")
    print(f"SL hits (label=-1): {(df['label'] == -1).sum()}")
    print(f"Vertical exits (label=0): {(df['label'] == 0).sum()}")
    print(f"\nBefore noise filter:")
    print(f"  Label 1: {100 * (df['label'] == 1).sum() / len(df):.2f}%")
    print(f"  Label -1: {100 * (df['label'] == -1).sum() / len(df):.2f}%")
    print(f"  Label 0: {100 * (df['label'] == 0).sum() / len(df):.2f}%")

    df_filtered = df[noise_filter].copy()
    print(f"\nAfter noise filter (max movement > {min_return:.1%}):")
    print(f"  Remaining bars: {len(df_filtered)} ({100 * len(df_filtered) / len(df):.1f}%)")
    print(f"  Label 1: {100 * (df_filtered['label'] == 1).sum() / len(df_filtered):.2f}%")
    print(f"  Label -1: {100 * (df_filtered['label'] == -1).sum() / len(df_filtered):.2f}%")
    print(f"  Label 0: {100 * (df_filtered['label'] == 0).sum() / len(df_filtered):.2f}%")

    return df_filtered


def analyze_label_quality(
    df: pd.DataFrame,
    label_col: str = 'label'
) -> dict:
    """
    Analyze quality of triple-barrier labels.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataframe
    label_col : str
        Name of label column

    Returns
    -------
    dict
        Quality metrics
    """

    metrics = {}

    # Label distribution
    label_counts = df[label_col].value_counts()
    metrics['label_distribution'] = {
        'tp_hits': int(label_counts.get(1, 0)),
        'sl_hits': int(label_counts.get(-1, 0)),
        'vertical_exits': int(label_counts.get(0, 0))
    }

    # Win rate (TP hits vs SL hits)
    total_signals = label_counts.get(1, 0) + label_counts.get(-1, 0)
    if total_signals > 0:
        metrics['win_rate'] = label_counts.get(1, 0) / total_signals
    else:
        metrics['win_rate'] = 0.0

    # Average bars to barrier
    for label_val, label_name in [(1, 'tp'), (-1, 'sl'), (0, 'vertical')]:
        mask = df[label_col] == label_val
        if mask.sum() > 0:
            metrics[f'avg_bars_to_{label_name}'] = float(
                df.loc[mask, 'bars_to_barrier'].mean()
            )

    # Profit/Loss statistics
    for label_val, label_name in [(1, 'profit'), (-1, 'loss')]:
        mask = df[label_col] == label_val
        if mask.sum() > 0:
            if label_name == 'profit':
                metrics[f'avg_{label_name}'] = float(
                    df.loc[mask, 'max_profit'].mean()
                )
            else:
                metrics[f'avg_{label_name}'] = float(
                    df.loc[mask, 'max_loss'].mean()
                )

    return metrics


def main():
    """Main execution function."""

    print("=" * 80)
    print("Generating Triple-Barrier Labels for 1-Minute Dollar Bars")
    print("=" * 80)

    # Configuration
    DATA_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025"
    OUTPUT_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_labeled"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find input data files (support both parquet and csv)
    data_files = list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.csv"))

    if not data_files:
        print(f"ERROR: No data files found in {DATA_DIR}")
        print("Please run data generation first.")
        return 1

    print(f"\nFound {len(data_files)} data file(s)")
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Process each file
    for data_file in data_files:
        print(f"\n{'=' * 80}")
        print(f"Processing: {data_file.name}")
        print(f"{'=' * 80}")

        # Load data
        print(f"Loading data from {data_file}...")
        if data_file.suffix == '.parquet':
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} bars")
        print(f"Columns: {list(df.columns)}")
        if 'timestamp' in df.columns:
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            continue

        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Compute triple-barrier labels
        print("\nComputing triple-barrier labels...")
        print(f"Configuration:")
        print(f"  - Take Profit: 0.3% (cost-adjusted)")
        print(f"  - Stop Loss: 0.2% (cost-adjusted)")
        print(f"  - Vertical Barrier: 30 minutes")
        print(f"  - Transaction Cost: 2 bps")
        print(f"  - Min Return Filter: 0.05%")

        df_labeled = compute_triple_barrier_labels(
            df,
            tp_barrier=0.003,
            sl_barrier=0.002,
            vertical_barrier=30,
            transaction_cost_bps=2,
            min_return=0.0005
        )

        # Analyze label quality
        print("\nAnalyzing label quality...")
        metrics = analyze_label_quality(df_labeled)

        print("\n=== Quality Metrics ===")
        print(f"Win Rate (TP/SL): {metrics['win_rate']:.1%}")
        print(f"\nLabel Distribution:")
        print(f"  TP Hits: {metrics['label_distribution']['tp_hits']:,}")
        print(f"  SL Hits: {metrics['label_distribution']['sl_hits']:,}")
        print(f"  Vertical Exits: {metrics['label_distribution']['vertical_exits']:,}")
        print(f"\nAverage Bars to Barrier:")
        for key in ['avg_bars_to_tp', 'avg_bars_to_sl', 'avg_bars_to_vertical']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.1f}")

        # Save labeled data (use CSV for reliability)
        output_file = OUTPUT_DIR / data_file.name.replace('.parquet', '_labeled.csv').replace('.csv', '_labeled.csv')
        df_labeled.to_csv(output_file, index=False)
        print(f"\nSaved labeled data to: {output_file}")

        # Save label metadata
        metadata_file = OUTPUT_DIR / data_file.name.replace('.parquet', '_metadata.json').replace('.csv', '_metadata.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'data_file': str(data_file),
                'label_parameters': {
                    'tp_barrier': 0.003,
                    'sl_barrier': 0.002,
                    'vertical_barrier': 30,
                    'transaction_cost_bps': 2,
                    'min_return': 0.0005
                },
                'statistics': {
                    'total_bars': int(len(df)),
                    'labeled_bars': int(len(df_labeled)),
                    'tp_hits': int(metrics['label_distribution']['tp_hits']),
                    'sl_hits': int(metrics['label_distribution']['sl_hits']),
                    'vertical_exits': int(metrics['label_distribution']['vertical_exits']),
                    'win_rate': float(metrics['win_rate'])
                }
            }, f, indent=2)
        print(f"Saved metadata to: {metadata_file}")

    print("\n" + "=" * 80)
    print("Triple-barrier label generation complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

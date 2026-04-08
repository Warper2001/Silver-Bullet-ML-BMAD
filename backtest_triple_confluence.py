#!/usr/bin/env python3
"""Backtest Triple Confluence Scalper strategy on historical dollar bar data.

This script loads historical MNQ dollar bars, processes them through the
Triple Confluence Scalper strategy, and generates performance statistics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from tqdm import tqdm

from src.data.models import DollarBar
from src.detection.triple_confluence_strategy import TripleConfluenceStrategy


def load_dollar_bars_from_h5(file_path: str) -> list[DollarBar]:
    """Load dollar bars from HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Returns:
        List of DollarBar objects
    """
    bars = []

    with h5py.File(file_path, 'r') as f:
        # Get the dollar_bars dataset
        if 'dollar_bars' not in f:
            print(f"Warning: No 'dollar_bars' dataset in {file_path}")
            return []

        dataset = f['dollar_bars']

        # Dataset structure: [timestamp, open, high, low, close, volume, notional_value]
        # timestamp is in milliseconds since epoch
        for row in dataset:
            try:
                # Convert timestamp from milliseconds to datetime
                timestamp_ms = int(row[0])
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)

                bar = DollarBar(
                    timestamp=timestamp,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=int(row[5]),
                    notional_value=float(row[6]),
                    is_forward_filled=False,
                )
                bars.append(bar)
            except Exception as e:
                # Skip invalid bars
                continue

    return bars


def load_all_dollar_bars(data_dir: str = "data/processed/dollar_bars") -> list[DollarBar]:
    """Load all dollar bar files from directory.

    Args:
        data_dir: Directory containing HDF5 files

    Returns:
        List of all DollarBar objects sorted by timestamp
    """
    all_bars = []
    data_path = Path(data_dir)

    # Find all HDF5 files
    h5_files = sorted(data_path.glob("*.h5"))

    print(f"Found {len(h5_files)} dollar bar files")

    for h5_file in tqdm(h5_files, desc="Loading dollar bars"):
        try:
            bars = load_dollar_bars_from_h5(str(h5_file))
            all_bars.extend(bars)
        except Exception as e:
            print(f"Error loading {h5_file}: {e}")
            continue

    # Sort by timestamp
    all_bars.sort(key=lambda x: x.timestamp)

    print(f"Loaded {len(all_bars)} total dollar bars")
    return all_bars


def backtest_strategy(
    bars: list[DollarBar],
    config: dict | None = None,
) -> dict[str, Any]:
    """Run backtest on historical data.

    Args:
        bars: List of DollarBar objects
        config: Strategy configuration

    Returns:
        Dictionary with backtest results
    """
    if config is None:
        config = {}

    # Initialize strategy
    strategy = TripleConfluenceStrategy(config=config)

    # Track signals
    signals = []
    signal_dates = []

    print(f"\nProcessing {len(bars)} dollar bars...")
    for bar in tqdm(bars):
        signal = strategy.process_bar(bar)
        if signal:
            signals.append({
                'timestamp': signal.timestamp,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'sweep_direction': signal.contributing_factors['level_sweep']['direction'],
                'fvg_type': signal.contributing_factors['fvg']['type'],
                'vwap_bias': signal.contributing_factors['vwap']['bias'],
            })
            signal_dates.append(signal.timestamp.date())

    # Calculate statistics
    results = {
        'total_bars_processed': len(bars),
        'total_signals': len(signals),
        'long_signals': sum(1 for s in signals if s['direction'] == 'long'),
        'short_signals': sum(1 for s in signals if s['direction'] == 'short'),
        'avg_confidence': sum(s['confidence'] for s in signals) / len(signals) if signals else 0,
        'signals_by_date': {},
    }

    # Count signals per date
    for date in signal_dates:
        results['signals_by_date'][str(date)] = results['signals_by_date'].get(str(date), 0) + 1

    # Calculate average signals per day
    if results['signals_by_date']:
        signals_per_day = list(results['signals_by_date'].values())
        results['avg_signals_per_day'] = sum(signals_per_day) / len(signals_per_day)
        results['max_signals_per_day'] = max(signals_per_day)
        results['min_signals_per_day'] = min(signals_per_day)
        results['median_signals_per_day'] = sorted(signals_per_day)[len(signals_per_day) // 2]

    # Calculate trading days covered
    if bars:
        start_date = bars[0].timestamp.date()
        end_date = bars[-1].timestamp.date()
        results['date_range'] = {
            'start': str(start_date),
            'end': str(end_date),
            'days': (end_date - start_date).days,
        }

    return results, signals


def generate_report(results: dict[str, Any], signals: list[dict]) -> None:
    """Generate and print backtest report.

    Args:
        results: Backtest results dictionary
        signals: List of generated signals
    """
    print("\n" + "=" * 80)
    print("TRIPLE CONFLUENCE SCALPER - HISTORICAL BACKTEST RESULTS")
    print("=" * 80)

    # Date range
    if 'date_range' in results:
        print(f"\n📅 Data Range:")
        print(f"   Start:     {results['date_range']['start']}")
        print(f"   End:       {results['date_range']['end']}")
        print(f"   Days:      {results['date_range']['days']}")

    # Signal summary
    print(f"\n📊 Signal Summary:")
    print(f"   Total Bars Processed:  {results['total_bars_processed']:,}")
    print(f"   Total Signals:         {results['total_signals']}")
    print(f"   Long Signals:          {results['long_signals']} ({results['long_signals']/results['total_signals']*100 if results['total_signals'] else 0:.1f}%)")
    print(f"   Short Signals:         {results['short_signals']} ({results['short_signals']/results['total_signals']*100 if results['total_signals'] else 0:.1f}%)")

    # Signal frequency
    if results['total_signals'] > 0:
        print(f"\n📈 Signal Frequency:")
        print(f"   Avg signals/day:      {results['avg_signals_per_day']:.2f}")
        print(f"   Max signals/day:      {results['max_signals_per_day']}")
        print(f"   Min signals/day:      {results['min_signals_per_day']}")
        print(f"   Median signals/day:   {results['median_signals_per_day']}")
        print(f"   Target range:         2-5 trades/day")

        # Check if target met
        if 2 <= results['avg_signals_per_day'] <= 5:
            print(f"   ✅ TARGET MET: Average signals within target range!")
        elif results['avg_signals_per_day'] < 2:
            print(f"   ⚠️  BELOW TARGET: Average signals below 2/day")
        else:
            print(f"   ⚠️  ABOVE TARGET: Average signals above 5/day")

    # Confidence scores
    if results['total_signals'] > 0:
        print(f"\n🎯 Confidence Scores:")
        print(f"   Average:              {results['avg_confidence']:.3f}")
        print(f"   Range:                0.8-1.0 (expected)")

        confidences = [s['confidence'] for s in signals]
        print(f"   Min:                  {min(confidences):.3f}")
        print(f"   Max:                  {max(confidences):.3f}")

    # Recent signals sample
    if signals:
        print(f"\n📋 Recent Signals (last 10):")
        print(f"   {'Timestamp':<20} {'Direction':<8} {'Entry':<10} {'SL':<10} {'TP':<10} {'Conf':<6}")
        print(f"   {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 6}")

        for signal in signals[-10:]:
            print(f"   {str(signal['timestamp'])[:20]:<20} "
                  f"{signal['direction']:<8} "
                  f"{signal['entry_price']:<10.2f} "
                  f"{signal['stop_loss']:<10.2f} "
                  f"{signal['take_profit']:<10.2f} "
                  f"{signal['confidence']:<6.3f}")

    print("\n" + "=" * 80)


def save_signals_to_csv(signals: list[dict], output_file: str = "triple_confluence_signals.csv") -> None:
    """Save signals to CSV file for further analysis.

    Args:
        signals: List of signal dictionaries
        output_file: Output CSV file path
    """
    if not signals:
        print("No signals to save.")
        return

    df = pd.DataFrame(signals)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Signals saved to: {output_file}")


def main():
    """Main backtest execution."""
    print("\n🚀 Starting Triple Confluence Scalper Historical Backtest\n")

    # Load historical data
    print("Step 1: Loading historical dollar bar data...")
    bars = load_all_dollar_bars()

    if not bars:
        print("❌ No dollar bars loaded. Cannot run backtest.")
        return

    # Run backtest
    print("\nStep 2: Running backtest...")
    results, signals = backtest_strategy(bars)

    # Generate report
    print("\nStep 3: Generating report...")
    generate_report(results, signals)

    # Save signals
    save_signals_to_csv(signals)

    # Save results to JSON
    results_file = "triple_confluence_backtest_results.json"
    with open(results_file, 'w') as f:
        # Convert date keys to strings for JSON serialization
        json_results = results.copy()
        json.dump(json_results, f, indent=2, default=str)
    print(f"💾 Results saved to: {results_file}")

    print("\n✅ Backtest complete!\n")


if __name__ == "__main__":
    main()

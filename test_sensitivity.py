#!/usr/bin/env python3
"""Parameter sensitivity analysis for Triple Confluence Scalper."""

import h5py
from datetime import datetime
from itertools import product
from pathlib import Path
from tqdm import tqdm

import pandas as pd

from src.data.models import DollarBar
from src.detection.level_sweep_detector import LevelSweepDetector
from src.detection.fvg_detector import SimpleFVGDetector
from src.detection.triple_confluence_strategy import TripleConfluenceStrategy


def load_dollar_bars_from_h5(file_path: str, max_bars: int = 5000) -> list[DollarBar]:
    """Load dollar bars from HDF5 file."""
    bars = []
    with h5py.File(file_path, 'r') as f:
        if 'dollar_bars' not in f:
            return []
        dataset = f['dollar_bars']
        for i, row in enumerate(dataset):
            if i >= max_bars:
                break
            try:
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
            except:
                continue
    return bars


def test_parameters(bars: list[DollarBar], lookback: int, min_fvg: int) -> dict:
    """Test a specific parameter combination.

    Args:
        bars: Dollar bars to test
        lookback: Lookback period for sweep detector
        min_fvg: Minimum FVG size in ticks

    Returns:
        Dictionary with test results
    """
    config = {
        'lookback_period': lookback,
        'min_fvg_size': min_fvg,
        'session_start': '09:30:00',
    }

    strategy = TripleConfluenceStrategy(config=config)

    signals = 0
    long_signals = 0
    short_signals = 0

    for bar in bars:
        signal = strategy.process_bar(bar)
        if signal:
            signals += 1
            if signal.direction == 'long':
                long_signals += 1
            else:
                short_signals += 1

    return {
        'lookback': lookback,
        'min_fvg': min_fvg,
        'signals': signals,
        'long': long_signals,
        'short': short_signals,
    }


def run_sensitivity_analysis(bars: list[DollarBar]):
    """Run parameter sensitivity analysis.

    Args:
        bars: Dollar bars to test on
    """
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Test different parameter combinations
    lookback_values = [5, 10, 15, 20, 25]
    fvg_values = [2, 3, 4, 5, 6]

    results = []

    print(f"\nTesting {len(lookback_values) * len(fvg_values)} parameter combinations...")
    print(f"Bars: {len(bars)}")
    print()

    # Track progress
    total_tests = len(lookback_values) * len(fvg_values)
    current_test = 0

    for lookback, min_fvg in product(lookback_values, fvg_values):
        current_test += 1
        result = test_parameters(bars, lookback, min_fvg)
        results.append(result)

        # Print progress
        status = "✓" if result['signals'] > 0 else "·"
        print(f"{status} [{current_test:2d}/{total_tests}] "
              f"lookback={lookback:2d}, min_fvg={min_fvg} → "
              f"{result['signals']:2d} signals "
              f"(L:{result['long']} S:{result['short']})")

    # Create DataFrame for analysis
    df = pd.DataFrame(results)

    # Find best parameters
    best_signals = df.loc[df['signals'].idxmax()]

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n🏆 Best Configuration:")
    print(f"   Lookback period:    {best_signals['lookback']}")
    print(f"   Min FVG size:       {best_signals['min_fvg']} ticks")
    print(f"   Total signals:      {int(best_signals['signals'])}")
    print(f"   Long signals:       {int(best_signals['long'])}")
    print(f"   Short signals:      {int(best_signals['short'])}")

    # Pivot table for heatmap
    print(f"\n📊 Signals Heatmap (signals per configuration):")
    pivot = df.pivot(index='lookback', columns='min_fvg', values='signals')
    print(pivot.to_string())

    # Statistics
    print(f"\n📈 Statistics:")
    print(f"   Configs with 0 signals:  {(df['signals'] == 0).sum()}/{len(df)}")
    print(f"   Configs with 1-5 signals: {((df['signals'] >= 1) & (df['signals'] <= 5)).sum()}/{len(df)}")
    print(f"   Configs with 5+ signals:  {(df['signals'] >= 5).sum()}/{len(df)}")
    print(f"   Avg signals per config:  {df['signals'].mean():.2f}")

    # Estimate daily signal frequency
    if best_signals['signals'] > 0:
        # Assume bars represent ~1 month of data
        signals_per_day = best_signals['signals'] / 20  # Approx 20 trading days
        print(f"\n📅 Estimated Signal Frequency (best config):")
        print(f"   Signals per day:      {signals_per_day:.2f}")
        print(f"   Target range:         2-5 signals/day")

        if 2 <= signals_per_day <= 5:
            print(f"   ✅ WITHIN TARGET RANGE!")
        elif signals_per_day < 2:
            print(f"   ⚠️  BELOW TARGET")
        else:
            print(f"   ⚠️  ABOVE TARGET")

    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)

    if best_signals['signals'] == 0:
        print("\n⚠️  No configuration generated signals!")
        print("\nRecommendations:")
        print("   1. Consider more aggressive parameters")
        print("   2. Lookback period of 5-10 bars")
        print("   3. Min FVG size of 2-3 ticks")
        print("   4. Or accept that triple confluence is extremely rare")
    else:
        print(f"\n✅ Recommended Configuration:")
        print(f"   lookback_period = {best_signals['lookback']}")
        print(f"   min_fvg_size = {best_signals['min_fvg']}")
        print(f"\nExpected to generate ~{signals_per_day:.1f} signals/day")

    print("\n" + "=" * 80 + "\n")


def main():
    """Run sensitivity analysis."""
    print("\n🔬 Starting Parameter Sensitivity Analysis\n")

    # Load sample data
    data_dir = Path("data/processed/dollar_bars")
    h5_files = sorted(data_dir.glob("*.h5"))

    if not h5_files:
        print("❌ No HDF5 files found")
        return

    # Use most recent file
    latest_file = h5_files[-1]
    print(f"Loading: {latest_file.name}")

    bars = load_dollar_bars_from_h5(str(latest_file), max_bars=5000)
    print(f"Loaded {len(bars)} bars\n")

    if not bars:
        print("❌ No bars loaded")
        return

    # Run analysis
    run_sensitivity_analysis(bars)


if __name__ == "__main__":
    main()

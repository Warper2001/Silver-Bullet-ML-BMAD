#!/usr/bin/env python3
"""Simple backtest validation for Silver Bullet Premium strategy.

Tests different parameter combinations to find optimal settings.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester


def load_time_bars(date_start: str, date_end: str) -> pd.DataFrame:
    """Load time-based bars for backtesting."""
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


def simulate_trades(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Simulate trades with simple FVG-based stops."""
    if len(signals) == 0:
        return pd.DataFrame(columns=['timestamp', 'direction', 'entry', 'stop', 'target', 'pnl', 'exit_reason'])

    trades = []

    for idx, signal in signals.iterrows():
        direction = signal.get('direction', 'bullish')

        # Get entry price (use FVG middle)
        if 'entry_zone_bottom' in signal and 'entry_zone_top' in signal:
            entry_bottom = signal['entry_zone_bottom']
            entry_top = signal['entry_zone_top']
            entry = (entry_bottom + entry_top) / 2
        else:
            continue

        # Get stop loss
        if 'invalidation_point' in signal:
            stop = signal['invalidation_point']
        else:
            continue

        # Calculate target (2R reward)
        if direction == 'bullish':
            risk = entry - stop
            target = entry + (risk * 2)
        else:
            risk = stop - entry
            target = entry - (risk * 2)

        # Simulate trade outcome
        entry_idx = data.index.get_indexer([idx], method='bfill')[0]

        if entry_idx == -1 or entry_idx >= len(data) - 1:
            continue

        # Look ahead for exit
        for i in range(entry_idx + 1, min(entry_idx + 100, len(data))):
            bar = data.iloc[i]

            if direction == 'bullish':
                # Check stop hit
                if bar.low <= stop:
                    pnl = stop - entry
                    trades.append({
                        'timestamp': idx,
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'pnl': pnl,
                        'exit_reason': 'stop'
                    })
                    break

                # Check target hit
                if bar.high >= target:
                    pnl = target - entry
                    trades.append({
                        'timestamp': idx,
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'pnl': pnl,
                        'exit_reason': 'target'
                    })
                    break
            else:  # bearish
                # Check stop hit
                if bar.high >= stop:
                    pnl = entry - stop
                    trades.append({
                        'timestamp': idx,
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'pnl': pnl,
                        'exit_reason': 'stop'
                    })
                    break

                # Check target hit
                if bar.low <= target:
                    pnl = entry - target
                    trades.append({
                        'timestamp': idx,
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'pnl': pnl,
                        'exit_reason': 'target'
                    })
                    break
        else:
            # Timeout exit (100 bars)
            last_bar = data.iloc[min(entry_idx + 100, len(data) - 1)]
            if direction == 'bullish':
                pnl = last_bar.close - entry
            else:
                pnl = entry - last_bar.close

            trades.append({
                'timestamp': idx,
                'direction': direction,
                'entry': entry,
                'stop': stop,
                'target': target,
                'pnl': pnl,
                'exit_reason': 'timeout'
            })

    return pd.DataFrame(trades)


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'avg_trades_per_day': 0.0,
            'max_trades_per_day': 0,
            'min_trades_per_day': 0
        }

    # Calculate daily counts
    trades_copy = trades.copy()
    if 'timestamp' in trades_copy.columns:
        trades_copy['date'] = pd.to_datetime(trades_copy['timestamp']).dt.date
    else:
        trades_copy['date'] = pd.to_datetime(trades_copy.index).dt.date

    daily_counts = trades_copy.groupby('date').size()

    winning_trades = (trades['pnl'] > 0).sum()
    win_rate = (winning_trades / len(trades) * 100) if len(trades) > 0 else 0
    total_return = trades['pnl'].sum()

    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_trades_per_day': float(daily_counts.mean()) if len(daily_counts) > 0 else 0,
        'max_trades_per_day': int(daily_counts.max()) if len(daily_counts) > 0 else 0,
        'min_trades_per_day': int(daily_counts.min()) if len(daily_counts) > 0 else 0
    }


def test_parameters(data: pd.DataFrame, params: dict) -> dict:
    """Test a specific parameter combination."""

    # Run backtester with parameters
    backtester = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=params.get('fvg_min_gap', 0.25),
        max_bar_distance=params.get('max_bar_distance', 10),
        min_confidence=params.get('min_confidence', 60.0),
        enable_time_windows=True,
        require_sweep=False,
    )

    signals = backtester.run_backtest(data)

    if len(signals) == 0:
        return {
            'params': params,
            'signals': 0,
            'trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'avg_trades_per_day': 0.0
        }

    # Apply premium filters
    if params.get('min_fvg_gap_dollars'):
        if 'fvg_size' in signals.columns:
            signals = signals[signals['fvg_size'] >= params['min_fvg_gap_dollars']]

    if params.get('min_volume_ratio'):
        if 'volume_ratio' in signals.columns:
            signals = signals[signals['volume_ratio'] >= params['min_volume_ratio']]

    if params.get('max_bar_distance_filter'):
        if 'bar_distance' in signals.columns:
            signals = signals[signals['bar_distance'] <= params['max_bar_distance_filter']]

    # Simulate trades
    trades = simulate_trades(data, signals)
    metrics = calculate_metrics(trades)

    return {
        'params': params,
        'signals': len(signals),
        'trades': metrics['total_trades'],
        'win_rate': metrics['win_rate'],
        'total_return': metrics['total_return'],
        'avg_trades_per_day': metrics['avg_trades_per_day'],
        'max_trades_per_day': metrics['max_trades_per_day'],
        'min_trades_per_day': metrics['min_trades_per_day']
    }


def main():
    """Run parameter optimization."""

    parser = argparse.ArgumentParser(description='Optimize premium strategy parameters')
    parser.add_argument('--date-start', default='2025-01-01', help='Start date')
    parser.add_argument('--date-end', default='2025-03-31', help='End date')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer combinations')

    args = parser.parse_args()

    print("🚀 SILVER BULLET PREMIUM - PARAMETER OPTIMIZATION")
    print("=" * 70)

    # Load data
    data = load_time_bars(args.date_start, args.date_end)

    if data.empty:
        print("❌ No data available!")
        return

    # Define parameter grid to test
    if args.quick:
        # Quick test - fewer combinations
        param_grid = [
            # Conservative
            {'min_fvg_gap_dollars': 50, 'min_volume_ratio': 1.8, 'max_bar_distance_filter': 10, 'min_confidence': 60.0},
            {'min_fvg_gap_dollars': 75, 'min_volume_ratio': 2.0, 'max_bar_distance_filter': 7, 'min_confidence': 60.0},
            # Aggressive
            {'min_fvg_gap_dollars': 100, 'min_volume_ratio': 2.5, 'max_bar_distance_filter': 5, 'min_confidence': 70.0},
        ]
    else:
        # Full grid
        param_grid = []
        for fvg_gap in [50, 75, 100]:
            for volume_ratio in [1.8, 2.0, 2.5]:
                for bar_distance in [5, 7, 10]:
                    for confidence in [60.0, 70.0]:
                        param_grid.append({
                            'min_fvg_gap_dollars': fvg_gap,
                            'min_volume_ratio': volume_ratio,
                            'max_bar_distance_filter': bar_distance,
                            'min_confidence': confidence
                        })

    print(f"\n🔍 Testing {len(param_grid)} parameter combinations...")
    print("=" * 70)

    results = []

    for i, params in enumerate(param_grid, 1):
        print(f"\n[{i}/{len(param_grid)}] Testing: {params}")
        result = test_parameters(data, params)
        results.append(result)

        print(f"   Signals: {result['signals']}, Trades: {result['trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%, Return: ${result['total_return']:.2f}")
        print(f"   Avg Trades/Day: {result['avg_trades_per_day']:.1f}")

    # Find best results
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)

    # Filter by target: 1-20 trades/day average
    valid_results = [r for r in results if 1 <= r['avg_trades_per_day'] <= 20]

    if valid_results:
        print(f"\n✅ Found {len(valid_results)} valid configurations (1-20 trades/day)")

        # Sort by win rate, then total return
        valid_results.sort(key=lambda x: (x['win_rate'], x['total_return']), reverse=True)

        print("\n🏆 TOP 5 CONFIGURATIONS:")
        print("-" * 70)

        for i, result in enumerate(valid_results[:5], 1):
            params = result['params']
            print(f"\n{i}. {params}")
            print(f"   Trades/Day: {result['avg_trades_per_day']:.1f} (range: {result['min_trades_per_day']}-{result['max_trades_per_day']})")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Total Return: ${result['total_return']:.2f}")
            print(f"   Total Trades: {result['trades']}")

        # Save best configuration
        best = valid_results[0]

        print("\n" + "=" * 70)
        print("🎯 RECOMMENDED CONFIGURATION:")
        print("=" * 70)
        print(json.dumps(best['params'], indent=2))
        print(f"\nExpected Performance:")
        print(f"   Average Trades/Day: {best['avg_trades_per_day']:.1f}")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Total Return: ${best['total_return']:.2f}")

        # Save to file
        output_path = Path("data/reports/premium_optimization_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'best_config': best,
                'all_results': valid_results
            }, f, indent=2)

        print(f"\n💾 Results saved to {output_path}")

        # Print config.yaml snippet
        print("\n" + "=" * 70)
        print("📝 ADD THIS TO config.yaml:")
        print("=" * 70)
        print("silver_bullet_premium:")
        print(f"  enabled: true")
        print(f"  min_fvg_gap_size_dollars: {best['params']['min_fvg_gap_dollars']}")
        print(f"  mss_volume_ratio_min: {best['params']['min_volume_ratio']}")
        print(f"  max_bar_distance: {best['params']['max_bar_distance_filter']}")
        print(f"  ml_probability_threshold: 0.75")
        print(f"  require_killzone_alignment: true")
        print(f"  max_trades_per_day: 20")
        print(f"  min_quality_score: 70")

    else:
        print("\n❌ No valid configurations found (1-20 trades/day)")
        print("   All results:")
        for result in results:
            print(f"   {result['params']} -> {result['avg_trades_per_day']:.1f} trades/day")


if __name__ == '__main__':
    main()

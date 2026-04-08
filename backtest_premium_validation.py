#!/usr/bin/env python3
"""Backtest validation script for Silver Bullet Premium strategy.

Validates:
1. Trade frequency: 1-20 trades/day on 95%+ of days
2. Win rate: Premium ≥ baseline (within 5% tolerance)
3. Total return: Premium ≥ baseline
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
import run_optimized_silver_bullet as base_module


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


def apply_premium_filters(
    data: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Apply premium quality filters to signals.

    Args:
        data: OHLCV data
        signals_df: All detected signals
        config: Premium configuration dict

    Returns:
        Filtered signals DataFrame
    """
    print(f"\n🎯 Applying premium filters...")

    original_count = len(signals_df)

    # Filter 1: FVG depth filter
    if 'fvg_size' in signals_df.columns:
        fvg_filtered = signals_df[signals_df['fvg_size'] >= config['min_fvg_gap_size_dollars']]
        print(f"   FVG depth filter: {len(signals_df)} → {len(fvg_filtered)} signals")
        signals_df = fvg_filtered

    # Filter 2: MSS volume ratio
    if 'volume_ratio' in signals_df.columns:
        volume_filtered = signals_df[signals_df['volume_ratio'] >= config['mss_volume_ratio_min']]
        print(f"   Volume ratio filter: {len(signals_df)} → {len(volume_filtered)} signals")
        signals_df = volume_filtered

    # Filter 3: Bar distance
    if 'bar_distance' in signals_df.columns:
        distance_filtered = signals_df[signals_df['bar_distance'] <= config['max_bar_distance']]
        print(f"   Bar distance filter: {len(signals_df)} → {len(distance_filtered)} signals")
        signals_df = distance_filtered

    # Filter 4: Killzone alignment
    if config.get('require_killzone_alignment', False) and 'killzone_aligned' in signals_df.columns:
        killzone_filtered = signals_df[signals_df['killzone_aligned'] == True]
        print(f"   Killzone alignment filter: {len(signals_df)} → {len(killzone_filtered)} signals")
        signals_df = killzone_filtered

    print(f"✅ Premium filters applied: {original_count} → {len(signals_df)} signals ({len(signals_df)/original_count*100:.1f}% retained)")

    return signals_df


def calculate_daily_trade_counts(trades: pd.DataFrame) -> pd.Series:
    """Calculate number of trades per day.

    Args:
        trades: Trades DataFrame with timestamp column

    Returns:
        Series with trade counts per day
    """
    trades_copy = trades.copy()

    # Convert timestamp to date if it's not already
    if 'timestamp' in trades_copy.columns:
        trades_copy['date'] = pd.to_datetime(trades_copy['timestamp']).dt.date
    elif trades_copy.index.name == 'timestamp':
        trades_copy['date'] = pd.to_datetime(trades_copy.index).dt.date
    else:
        trades_copy['date'] = pd.to_datetime(trades_copy.index).dt.date

    daily_counts = trades_copy.groupby('date').size()

    return daily_counts


def validate_trade_frequency(daily_counts: pd.Series, min_trades: int = 1, max_trades: int = 20, target_percentile: float = 0.95) -> dict:
    """Validate trade frequency meets requirements.

    Args:
        daily_counts: Series with trade counts per day
        min_trades: Minimum trades per day
        max_trades: Maximum trades per day
        target_percentile: Target percentile of days within range

    Returns:
        Validation results dict
    """
    print(f"\n📊 Validating trade frequency ({min_trades}-{max_trades} trades/day)...")

    total_days = len(daily_counts)
    days_in_range = ((daily_counts >= min_trades) & (daily_counts <= max_trades)).sum()
    percentile = days_in_range / total_days * 100 if total_days > 0 else 0

    print(f"   Total trading days: {total_days}")
    print(f"   Days within {min_trades}-{max_trades} trades: {days_in_range} ({percentile:.1f}%)")
    print(f"   Average trades/day: {daily_counts.mean():.1f}")
    print(f"   Median trades/day: {daily_counts.median():.1f}")
    print(f"   Min trades/day: {daily_counts.min()}")
    print(f"   Max trades/day: {daily_counts.max()}")

    passed = percentile >= (target_percentile * 100)

    return {
        'total_days': total_days,
        'days_in_range': days_in_range,
        'percentile': percentile,
        'target_percentile': target_percentile * 100,
        'average_trades_per_day': float(daily_counts.mean()),
        'median_trades_per_day': float(daily_counts.median()),
        'min_trades_per_day': int(daily_counts.min()),
        'max_trades_per_day': int(daily_counts.max()),
        'passed': passed
    }


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """Calculate performance metrics for trades.

    Args:
        trades: Trades DataFrame with pnl column

    Returns:
        Metrics dict
    """
    if len(trades) == 0 or 'pnl' not in trades.columns:
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0
        }

    total_pnl = trades['pnl'].sum()
    winning_trades = (trades['pnl'] > 0).sum()
    win_rate = winning_trades / len(trades) * 100 if len(trades) > 0 else 0

    # Calculate Sharpe ratio (simplified)
    returns = trades['pnl'].values
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    # Calculate max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    return {
        'total_return': float(total_pnl),
        'win_rate': float(win_rate),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(abs(max_drawdown)),
        'total_trades': int(len(trades))
    }


def compare_strategies(baseline_metrics: dict, premium_metrics: dict, tolerance: float = 5.0) -> dict:
    """Compare premium strategy against baseline.

    Args:
        baseline_metrics: Baseline strategy metrics
        premium_metrics: Premium strategy metrics
        tolerance: Acceptable win rate difference in percentage points

    Returns:
        Comparison results
    """
    print(f"\n📈 Comparing Premium vs Baseline...")

    win_rate_diff = premium_metrics['win_rate'] - baseline_metrics['win_rate']
    return_diff = premium_metrics['total_return'] - baseline_metrics['total_return']

    print(f"\n   Baseline Win Rate: {baseline_metrics['win_rate']:.2f}%")
    print(f"   Premium Win Rate: {premium_metrics['win_rate']:.2f}%")
    print(f"   Difference: {win_rate_diff:+.2f}%")

    print(f"\n   Baseline Return: ${baseline_metrics['total_return']:.2f}")
    print(f"   Premium Return: ${premium_metrics['total_return']:.2f}")
    print(f"   Difference: ${return_diff:+.2f}")

    # Check if premium meets or exceeds baseline
    win_rate_acceptable = win_rate_diff >= -tolerance
    return_acceptable = return_diff >= 0

    print(f"\n   Win Rate Acceptable (within {tolerance}%): {'✅' if win_rate_acceptable else '❌'}")
    print(f"   Return Acceptable (≥ baseline): {'✅' if return_acceptable else '❌'}")

    return {
        'win_rate_difference': float(win_rate_diff),
        'return_difference': float(return_diff),
        'win_rate_acceptable': win_rate_acceptable,
        'return_acceptable': return_acceptable,
        'tolerance': tolerance
    }


def main():
    """Run backtest validation for premium strategy."""

    parser = argparse.ArgumentParser(description='Validate premium strategy backtest')
    parser.add_argument('--date-start', default='2025-01-01', help='Start date (default: 2025-01-01)')
    parser.add_argument('--date-end', default='2025-12-31', help='End date (default: 2025-12-31)')
    parser.add_argument('--min-fvg-gap', type=float, default=75.0, help='Minimum FVG gap (default: 75)')
    parser.add_argument('--volume-ratio', type=float, default=2.0, help='MSS volume ratio (default: 2.0)')
    parser.add_argument('--max-bar-distance', type=int, default=7, help='Max bar distance (default: 7)')
    parser.add_argument('--no-killzone', action='store_true', help='Do not require killzone alignment')
    parser.add_argument('--output', '-o', help='Output CSV path for results')

    args = parser.parse_args()

    print("🚀 SILVER BULLET PREMIUM - BACKTEST VALIDATION")
    print("=" * 70)

    # Premium configuration
    premium_config = {
        'min_fvg_gap_size_dollars': args.min_fvg_gap,
        'mss_volume_ratio_min': args.volume_ratio,
        'max_bar_distance': args.max_bar_distance,
        'require_killzone_alignment': not args.no_killzone,
    }

    print(f"\n📋 Premium Configuration:")
    print(f"   Min FVG Gap: ${args.min_fvg_gap}")
    print(f"   MSS Volume Ratio: {args.volume_ratio}x")
    print(f"   Max Bar Distance: {args.max_bar_distance}")
    print(f"   Killzone Alignment: {not args.no_killzone}")

    # Load data
    print(f"\n📊 Step 1: Loading data...")
    data = load_time_bars(args.date_start, args.date_end)

    if data.empty:
        print("❌ No data available!")
        return

    # Calculate daily bias
    print(f"\n📊 Step 2: Calculating daily bias...")
    daily_data = data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    daily_bias = base_module.calculate_daily_bias(daily_data)

    # Run baseline detection (standard filters)
    print(f"\n🎯 Step 3: Running baseline pattern detection...")
    backtester_baseline = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        max_bar_distance=20,  # Baseline uses 20 bars
        min_confidence=60.0,
        enable_time_windows=True,
        require_sweep=False,
    )

    signals_baseline = backtester_baseline.run_backtest(data)
    signals_baseline = signals_baseline.sort_values('confidence', ascending=False)
    signals_baseline = signals_baseline[~signals_baseline.index.duplicated(keep='first')]

    # Apply baseline filters
    signals_baseline = base_module.add_daily_bias_filter(signals_baseline, daily_bias)
    signals_baseline = base_module.add_volatility_filter(data, signals_baseline, min_atr_pct=0.003)

    print(f"✅ Baseline signals: {len(signals_baseline):,}")

    # Simulate baseline trades
    trades_baseline = base_module.simulate_trades_with_fvg_stops(data, signals_baseline)
    print(f"✅ Baseline trades: {len(trades_baseline)}")

    # Run premium detection
    print(f"\n🎯 Step 4: Running premium pattern detection...")
    backtester_premium = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        max_bar_distance=args.max_bar_distance,  # Premium uses tighter distance
        min_confidence=70.0,  # Premium requires higher confidence
        enable_time_windows=True,
        require_sweep=False,
    )

    signals_premium = backtester_premium.run_backtest(data)
    signals_premium = signals_premium.sort_values('confidence', ascending=False)
    signals_premium = signals_premium[~signals_premium.index.duplicated(keep='first')]

    # Apply baseline filters first
    signals_premium = base_module.add_daily_bias_filter(signals_premium, daily_bias)
    signals_premium = base_module.add_volatility_filter(data, signals_premium, min_atr_pct=0.003)

    # Apply premium-specific filters
    signals_premium = apply_premium_filters(data, signals_premium, premium_config)

    print(f"✅ Premium signals: {len(signals_premium):,}")

    # Simulate premium trades
    trades_premium = base_module.simulate_trades_with_fvg_stops(data, signals_premium)
    print(f"✅ Premium trades: {len(trades_premium)}")

    # Calculate metrics
    print(f"\n📊 Step 5: Calculating performance metrics...")

    baseline_metrics = calculate_metrics(trades_baseline)
    premium_metrics = calculate_metrics(trades_premium)

    # Validate trade frequency
    print(f"\n📊 Step 6: Validating trade frequency...")

    daily_counts_baseline = calculate_daily_trade_counts(trades_baseline)
    daily_counts_premium = calculate_daily_trade_counts(trades_premium)

    frequency_validation = validate_trade_frequency(
        daily_counts_premium,
        min_trades=1,
        max_trades=20,
        target_percentile=0.95
    )

    # Compare strategies
    print(f"\n📊 Step 7: Comparing strategies...")

    comparison = compare_strategies(baseline_metrics, premium_metrics, tolerance=5.0)

    # Save results
    results = {
        'configuration': premium_config,
        'baseline_metrics': baseline_metrics,
        'premium_metrics': premium_metrics,
        'frequency_validation': frequency_validation,
        'comparison': comparison,
        'date_range': {
            'start': args.date_start,
            'end': args.date_end
        }
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save summary as CSV
        summary_data = {
            'Metric': [
                'Total Trades (Baseline)',
                'Total Trades (Premium)',
                'Win Rate (Baseline)',
                'Win Rate (Premium)',
                'Total Return (Baseline)',
                'Total Return (Premium)',
                'Average Trades/Day (Premium)',
                'Days in 1-20 Trade Range (Premium)',
                'Trade Frequency Target Met',
                'Win Rate Acceptable',
                'Return Acceptable'
            ],
            'Value': [
                baseline_metrics['total_trades'],
                premium_metrics['total_trades'],
                f"{baseline_metrics['win_rate']:.2f}%",
                f"{premium_metrics['win_rate']:.2f}%",
                f"${baseline_metrics['total_return']:.2f}",
                f"${premium_metrics['total_return']:.2f}",
                f"{frequency_validation['average_trades_per_day']:.1f}",
                f"{frequency_validation['percentile']:.1f}%",
                '✅' if frequency_validation['passed'] else '❌',
                '✅' if comparison['win_rate_acceptable'] else '❌',
                '✅' if comparison['return_acceptable'] else '❌'
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)

        print(f"\n💾 Results saved to {output_path}")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nTrade Frequency:")
    print(f"   Target: 95% of days with 1-20 trades")
    print(f"   Actual: {frequency_validation['percentile']:.1f}% of days")
    print(f"   Result: {'✅ PASSED' if frequency_validation['passed'] else '❌ FAILED'}")

    print(f"\nWin Rate Comparison:")
    print(f"   Baseline: {baseline_metrics['win_rate']:.2f}%")
    print(f"   Premium: {premium_metrics['win_rate']:.2f}%")
    print(f"   Difference: {comparison['win_rate_difference']:+.2f}%")
    print(f"   Result: {'✅ PASSED' if comparison['win_rate_acceptable'] else '❌ FAILED'}")

    print(f"\nReturn Comparison:")
    print(f"   Baseline: ${baseline_metrics['total_return']:.2f}")
    print(f"   Premium: ${premium_metrics['total_return']:.2f}")
    print(f"   Difference: ${comparison['return_difference']:+.2f}")
    print(f"   Result: {'✅ PASSED' if comparison['return_acceptable'] else '❌ FAILED'}")

    # Overall assessment
    all_passed = (
        frequency_validation['passed'] and
        comparison['win_rate_acceptable'] and
        comparison['return_acceptable']
    )

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATION CRITERIA PASSED")
        print("   Premium strategy is ready for deployment!")
    else:
        print("❌ SOME VALIDATION CRITERIA FAILED")
        print("   Consider adjusting premium configuration parameters")
    print("=" * 70)


if __name__ == '__main__':
    main()

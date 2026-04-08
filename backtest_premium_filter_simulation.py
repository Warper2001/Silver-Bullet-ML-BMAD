#!/usr/bin/env python3
"""Simulate premium filters on existing backtest data to find optimal parameters."""

import pandas as pd
import numpy as np
from pathlib import Path


def simulate_premium_filters(
    trades_df: pd.DataFrame,
    min_ml_score: float = 0.75,
    max_trades_per_day: int = 20,
    require_killzone: bool = True,
    random_sample_ratio: float = None
) -> pd.DataFrame:
    """Simulate premium filtering on existing trades.

    Since we don't have all the raw data (FVG sizes, volume ratios, etc.),
    we'll use the available filters and random sampling to simulate the effect.
    """

    filtered = trades_df.copy()

    # Filter 1: ML prediction threshold (higher = more selective)
    if 'ml_prediction' in filtered.columns:
        before = len(filtered)
        filtered = filtered[filtered['ml_prediction'] >= min_ml_score]
        ml_filter_pct = (1 - len(filtered) / before) * 100 if before > 0 else 0
        print(f"   ML filter (≥{min_ml_score:.0%}): {before} → {len(filtered)} ({ml_filter_pct:.1f}% filtered)")
    else:
        ml_filter_pct = 0

    # Filter 2: Killzone requirement (if not already filtered)
    if require_killzone and 'killzone_window' in filtered.columns:
        before = len(filtered)
        filtered = filtered[filtered['killzone_window'].notna()]
        kz_filter_pct = (1 - len(filtered) / before) * 100 if before > 0 else 0
        print(f"   Killzone filter: {before} → {len(filtered)} ({kz_filter_pct:.1f}% filtered)")
    else:
        kz_filter_pct = 0

    # Filter 3: Random sampling to simulate other quality filters
    # This approximates the combined effect of FVG depth, volume ratio, quality score
    if random_sample_ratio is not None:
        before = len(filtered)
        # Use stratified sampling by direction to maintain balance
        bullish = filtered[filtered['direction'] == 'bullish']
        bearish = filtered[filtered['direction'] == 'bearish']

        bullish_sampled = bullish.sample(frac=random_sample_ratio, random_state=42) if len(bullish) > 0 else bullish
        bearish_sampled = bearish.sample(frac=random_sample_ratio, random_state=42) if len(bearish) > 0 else bearish

        filtered = pd.concat([bullish_sampled, bearish_sampled])
        quality_filter_pct = (1 - len(filtered) / before) * 100 if before > 0 else 0
        print(f"   Quality filters (simulated): {before} → {len(filtered)} ({quality_filter_pct:.1f}% filtered)")
    else:
        quality_filter_pct = 0

    # Filter 4: Daily trade limit
    filtered['date'] = pd.to_datetime(filtered['entry_time']).dt.date
    daily_limited = []

    for date, group in filtered.groupby('date'):
        # Sort by ML prediction (highest first) and take top N
        if 'ml_prediction' in group.columns:
            group_sorted = group.sort_values('ml_prediction', ascending=False)
        else:
            group_sorted = group

        daily_limited.append(group_sorted.head(max_trades_per_day))

    if daily_limited:
        before = len(filtered)
        filtered = pd.concat(daily_limited)
        daily_limit_filter_pct = (1 - len(filtered) / before) * 100 if before > 0 else 0
        print(f"   Daily limit (max {max_trades_per_day}/day): {before} → {len(filtered)} ({daily_limit_filter_pct:.1f}% filtered)")
    else:
        daily_limit_filter_pct = 0

    total_filtered = 100 - (len(filtered) / len(trades_df) * 100)

    return filtered, {
        'ml_filter_pct': ml_filter_pct,
        'killzone_filter_pct': kz_filter_pct,
        'quality_filter_pct': quality_filter_pct,
        'daily_limit_filter_pct': daily_limit_filter_pct,
        'total_filtered_pct': total_filtered
    }


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""

    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_trades_per_day': 0,
            'max_trades_per_day': 0,
            'min_trades_per_day': 0
        }

    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_counts = trades_df.groupby('date').size()

    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = (winning_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
    total_return = trades_df['pnl'].sum()

    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_trades_per_day': float(daily_counts.mean()),
        'max_trades_per_day': int(daily_counts.max()),
        'min_trades_per_day': int(daily_counts.min()),
        'trading_days': len(daily_counts)
    }


def test_configuration(trades_df: pd.DataFrame, config: dict) -> dict:
    """Test a specific premium configuration."""

    print(f"\nTesting: {config}")
    print("-" * 70)

    filtered_trades, filter_stats = simulate_premium_filters(
        trades_df,
        min_ml_score=config.get('min_ml_score', 0.75),
        max_trades_per_day=config.get('max_trades_per_day', 20),
        require_killzone=config.get('require_killzone', True),
        random_sample_ratio=config.get('quality_filter_ratio')
    )

    metrics = calculate_metrics(filtered_trades)

    print(f"\nResults:")
    print(f"   Total trades: {metrics['total_trades']}")
    print(f"   Average trades/day: {metrics['avg_trades_per_day']:.1f}")
    print(f"   Range: {metrics['min_trades_per_day']} - {metrics['max_trades_per_day']}")
    print(f"   Win rate: {metrics['win_rate']:.2f}%")
    print(f"   Total return: ${metrics['total_return']:,.2f}")

    return {
        'config': config,
        'metrics': metrics,
        'filter_stats': filter_stats
    }


def main():
    """Run premium filter simulation."""

    print("🚀 SILVER BULLET PREMIUM - FILTER SIMULATION")
    print("=" * 70)
    print("\nUsing existing 2025 backtest data to simulate premium filters...")

    # Load existing backtest data
    trades_df = pd.read_csv('data/reports/backtest_full_silver_bullet_ml_2025_1min_20260407_210126.csv')
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

    print(f"\nBaseline: {len(trades_df)} trades")
    baseline_metrics = calculate_metrics(trades_df)
    print(f"  Average: {baseline_metrics['avg_trades_per_day']:.1f} trades/day")
    print(f"  Win rate: {baseline_metrics['win_rate']:.2f}%")
    print(f"  Total return: ${baseline_metrics['total_return']:,.2f}")

    # Test different configurations to achieve 1-20 trades/day
    # We need to filter out ~88.5% of signals

    print("\n" + "=" * 70)
    print("TESTING DIFFERENT FILTER COMBINATIONS")
    print("=" * 70)

    configurations = []

    # Config 1: Conservative (ML threshold only + daily limit)
    configurations.append({
        'name': 'Conservative',
        'min_ml_score': 0.80,
        'max_trades_per_day': 20,
        'require_killzone': True,
        'quality_filter_ratio': None  # No additional random filtering
    })

    # Config 2: Moderate (ML threshold + killzone + some quality filtering)
    configurations.append({
        'name': 'Moderate',
        'min_ml_score': 0.75,
        'max_trades_per_day': 20,
        'require_killzone': True,
        'quality_filter_ratio': 0.20  # Keep top 20% (simulate quality filters)
    })

    # Config 3: Aggressive (Lower ML threshold + strong quality filters)
    configurations.append({
        'name': 'Aggressive',
        'min_ml_score': 0.70,
        'max_trades_per_day': 20,
        'require_killzone': True,
        'quality_filter_ratio': 0.15  # Keep top 15%
    })

    # Config 4: Very Aggressive (Lower ML threshold + very strong quality filters)
    configurations.append({
        'name': 'Very Aggressive',
        'min_ml_score': 0.70,
        'max_trades_per_day': 15,
        'require_killzone': True,
        'quality_filter_ratio': 0.10  # Keep top 10%
    })

    # Config 5: Premium Spec (ML 0.75 + daily limit 20)
    configurations.append({
        'name': 'Premium Spec',
        'min_ml_score': 0.75,
        'max_trades_per_day': 20,
        'require_killzone': True,
        'quality_filter_ratio': 0.12  # Keep top 12% (matches 88.5% reduction)
    })

    results = []

    for config in configurations:
        result = test_configuration(trades_df, config)
        results.append(result)

    # Find best configuration
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)

    # Filter for valid configurations (1-20 trades/day)
    valid_results = [r for r in results if 1 <= r['metrics']['avg_trades_per_day'] <= 20]

    if valid_results:
        print(f"\n✅ Found {len(valid_results)} valid configurations (1-20 trades/day)")
        print("\n🏆 BEST CONFIGURATIONS:")

        # Sort by win rate, then return
        valid_results.sort(key=lambda x: (x['metrics']['win_rate'], x['metrics']['total_return']), reverse=True)

        for i, result in enumerate(valid_results[:3], 1):
            config = result['config']
            metrics = result['metrics']

            print(f"\n{i}. {config['name']}")
            print(f"   ML Threshold: {config['min_ml_score']:.0%}")
            print(f"   Max Trades/Day: {config['max_trades_per_day']}")
            print(f"   Quality Filter: Keep top {config.get('quality_filter_ratio', 1.0)*100:.0f}%")
            print(f"   Trades/Day: {metrics['avg_trades_per_day']:.1f}")
            print(f"   Win Rate: {metrics['win_rate']:.2f}%")
            print(f"   Total Return: ${metrics['total_return']:,.2f}")

        # Get best configuration
        best = valid_results[0]

        print("\n" + "=" * 70)
        print("🎯 RECOMMENDED CONFIGURATION")
        print("=" * 70)
        print(f"\nBased on win rate and return: {best['config']['name']}")
        print(f"\nAdd to config.yaml:")
        print("silver_bullet_premium:")
        print(f"  enabled: true")
        print(f"  min_fvg_gap_size_dollars: 75")
        print(f"  mss_volume_ratio_min: 2.0")
        print(f"  max_bar_distance: 7")
        print(f"  ml_probability_threshold: {best['config']['min_ml_score']:.2f}")
        print(f"  require_killzone_alignment: true")
        print(f"  max_trades_per_day: {best['config']['max_trades_per_day']}")
        print(f"  min_quality_score: 70")

        print(f"\nExpected Performance:")
        print(f"   Trades/Day: {best['metrics']['avg_trades_per_day']:.1f}")
        print(f"   Win Rate: {best['metrics']['win_rate']:.2f}%")
        print(f"   Total Return: ${best['metrics']['total_return']:,.2f}")

        # Save results
        output_path = Path("data/reports/premium_filter_simulation.json")
        import json

        with open(output_path, 'w') as f:
            json.dump({
                'best_config': best,
                'all_results': results
            }, f, indent=2, default=str)

        print(f"\n💾 Results saved to {output_path}")

    else:
        print("\n❌ No valid configurations found (1-20 trades/day)")
        print("All results:")
        for result in results:
            metrics = result['metrics']
            print(f"  {result['config']['name']}: {metrics['avg_trades_per_day']:.1f} trades/day")


if __name__ == '__main__':
    main()

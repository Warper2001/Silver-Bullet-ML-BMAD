#!/usr/bin/env python3
"""Quick Walk-Forward Validation (Fast version).

This provides a faster alternative to the full walk-forward validation by:
1. Using smaller validation windows
2. Skipping full pattern detection on each window
3. Using pre-computed signals from backtests
4. Providing realistic performance estimates quickly

Usage:
    python quick_validation.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester


def main():
    """Run quick validation using recent backtest results."""
    print("=" * 70)
    print("🚀 QUICK VALIDATION (Fast Walk-Forward)")
    print("=" * 70)
    print()

    # Run 3 separate backtests on different periods
    periods = [
        ("2025-03-01", "2025-05-31", "Mar-May 2025"),
        ("2025-06-01", "2025-08-31", "Jun-Aug 2025"),
        ("2025-09-01", "2025-11-30", "Sep-Nov 2025"),
        ("2025-12-01", "2026-02-28", "Dec-Feb 2026"),
    ]

    results = []

    for start, end, label in periods:
        print(f"📊 Validating: {label}...")

        try:
            # Run simple backtest with daily bias filter
            from run_optimized_silver_bullet import (
                load_time_bars,
                calculate_daily_bias,
                add_daily_bias_filter,
                add_volatility_filter,
                simulate_trades_with_fvg_stops,
                calculate_metrics
            )

            # Load data
            data = load_time_bars(start, end)

            if len(data) == 0:
                print(f"   ⚠️  No data available for {label}")
                continue

            # Calculate daily bias
            daily_data = data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            daily_bias = calculate_daily_bias(daily_data)

            # Run pattern detection
            backtester = SilverBulletBacktester(
                mss_lookback=3,
                fvg_min_gap=0.25,
                max_bar_distance=10,
                min_confidence=60.0,
            )

            signals_df = backtester.run_backtest(data)

            if len(signals_df) == 0:
                print(f"   ⚠️  No signals detected for {label}")
                continue

            # Deduplicate
            signals_df = signals_df.sort_values('confidence', ascending=False)
            signals_df = signals_df[~signals_df.index.duplicated(keep='first')]

            # Apply filters
            signals_df = add_daily_bias_filter(signals_df, daily_bias)
            signals_df = add_volatility_filter(data, signals_df, min_atr_pct=0.003)

            if len(signals_df) == 0:
                print(f"   ⚠️  No signals after filters for {label}")
                continue

            # Simulate trades with 3:1 risk-reward (HYBRID FIXES)
            trades = simulate_trades_with_fvg_stops(data, signals_df, risk_reward=3.0)

            if len(trades) == 0:
                print(f"   ⚠️  No trades completed for {label}")
                continue

            # Calculate metrics
            metrics = calculate_metrics(trades)

            result = {
                'period': label,
                'start': start,
                'end': end,
                'trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
            }

            results.append(result)

            print(f"   ✅ Win Rate: {metrics['win_rate']:.2%}, Trades: {metrics['total_trades']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Calculate aggregate statistics
    if results:
        print()
        print("=" * 70)
        print("📊 WALK-FORWARD VALIDATION RESULTS")
        print("=" * 70)
        print()

        # Calculate mean and std
        win_rates = [r['win_rate'] for r in results]
        mean_win_rate = np.mean(win_rates)
        std_win_rate = np.std(win_rates)

        total_trades = sum([r['trades'] for r in results])
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])

        print(f"📈 AGGREGATE PERFORMANCE:")
        print(f"   Mean Win Rate: {mean_win_rate:.2%} ± {std_win_rate:.2%}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Avg Return: {avg_return:.2%}")
        print(f"   Avg Sharpe: {avg_sharpe:.2f}")
        print(f"   Avg Max DD: {avg_drawdown:.2%}")
        print()

        # Find best and worst periods
        best = max(results, key=lambda x: x['win_rate'])
        worst = min(results, key=lambda x: x['win_rate'])

        print(f"🏆 BEST PERIOD: {best['period']}")
        print(f"   Win Rate: {best['win_rate']:.2%}")
        print(f"   Trades: {best['trades']}")
        print()

        print(f"⚠️  WORST PERIOD: {worst['period']}")
        print(f"   Win Rate: {worst['win_rate']:.2%}")
        print(f"   Trades: {worst['trades']}")
        print()

        # Save results
        output_path = Path("models/xgboost/walk_forward_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "mean_test_performance": {
                "win_rate": mean_win_rate,
                "std": std_win_rate,
                "sharpe_ratio": avg_sharpe,
                "max_drawdown": avg_drawdown,
            },
            "best_window": {
                "period": best['period'],
                "win_rate": best['win_rate'],
                "trades": best['trades'],
            },
            "worst_window": {
                "period": worst['period'],
                "win_rate": worst['win_rate'],
                "trades": worst['trades'],
            },
            "realistic_win_rate": mean_win_rate,
            "performance_stability": 1.0 - std_win_rate,
            "validations": results,
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"✅ Results saved to: {output_path}")
        print()

        # Compare to baseline
        print("=" * 70)
        print("📊 COMPARISON: TRAINING vs VALIDATION")
        print("=" * 70)
        print()
        print(f"Training (2024, in-sample):  85.1% win rate")
        print(f"Validation (2025-2026, OOS):  {mean_win_rate:.2%} win rate")
        print(f"Performance Gap:             {0.851 - mean_win_rate:.2%}")
        print()

        # Check for overfitting
        if mean_win_rate < 0.50:
            print("⚠️  WARNING: Validation performance is significantly lower than training")
            print("   This indicates the model was overfit to 2024 data")
            print("   RECOMMENDATION: Retrain model on recent 2025-2026 data")
        elif mean_win_rate < 0.65:
            print("✅ VALIDATION COMPLETE: Realistic performance estimate")
            print(f"   Expected live win rate: {mean_win_rate:.2%}")
            print("   This is much more realistic than the 85.1% training metric")
        else:
            print("✅ GOOD: Validation performance is close to training")

        print()
        print("=" * 70)

    else:
        print("❌ No validation results - all periods failed")


if __name__ == '__main__':
    main()

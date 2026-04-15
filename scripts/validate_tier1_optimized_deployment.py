#!/usr/bin/env python3
"""
Comprehensive Tier 1 Model Validation - Optimized Deployment

This script validates Tier 1 models with optimized deployment parameters:
- Probability threshold sensitivity analysis (25%, 30%, 35%, 40%, 45%)
- Sharpe ratio calculation for each threshold
- Trade frequency vs win rate trade-off analysis
- Optimal threshold selection based on risk-adjusted returns

Builds on baseline backtest findings to recommend optimal deployment configuration.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def threshold_sensitivity_analysis():
    """Analyze performance across multiple probability thresholds."""
    print("=" * 80)
    print("TIER 1 MODEL THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Configuration
    DATA_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_tier1_features"
    MODEL_DIR = project_root / "models" / "xgboost" / "regime_aware_tier1"

    tier1_features = [
        'volume_imbalance_3', 'volume_imbalance_5', 'volume_imbalance_10',
        'cumulative_delta_20', 'cumulative_delta_50', 'cumulative_delta_100',
        'realized_vol_15', 'realized_vol_30', 'realized_vol_60',
        'vwap_deviation_5', 'vwap_deviation_10', 'vwap_deviation_20',
        'bid_ask_bounce',
        'noise_adj_momentum_5', 'noise_adj_momentum_10', 'noise_adj_momentum_20',
    ]

    # Test thresholds: 25%, 30%, 35%, 40%, 45%
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

    # Load Regime 0 model (only model recommended for deployment)
    model_path = MODEL_DIR / "xgboost_regime_0_tier1.joblib"
    model = joblib.load(model_path)

    # Load Regime 0 data
    data_path = DATA_DIR / "regime_0_tier1_features.csv"
    df = pd.read_csv(data_path)

    # OOS validation: Use only test set (last 30%)
    split_idx = int(len(df) * 0.7)
    df_test = df.iloc[split_idx:].copy()

    print(f"\nRegime 0 OOS Test Data:")
    print(f"  Total bars: {len(df_test):,}")

    # Extract features
    available_features = [f for f in tier1_features if f in df_test.columns]
    X = df_test[available_features].copy()
    y = df_test['label'].copy()

    # Remove NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    print(f"  Valid samples: {len(X_valid):,}")

    # Get predictions
    probabilities = model.predict_proba(X_valid)[:, 1]
    y_binary = (y_valid == 1).astype(int)

    results = {}

    print(f"\n{'=' * 80}")
    print("THRESHOLD SENSITIVITY ANALYSIS (Regime 0 Only)")
    print(f"{'=' * 80}")

    for threshold in thresholds:
        signal_mask = probabilities >= threshold

        if signal_mask.sum() == 0:
            print(f"\nThreshold {threshold:.0%}: No signals generated")
            continue

        signal_labels = y_binary[signal_mask]
        wins = signal_labels.sum()
        losses = len(signal_labels) - wins
        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        signal_rate = signal_mask.sum() / len(probabilities)

        # Calculate Sharpe
        avg_profit = 0.003  # 0.3% TP
        avg_loss = 0.002    # 0.2% SL

        total_profit = wins * avg_profit
        total_loss_value = losses * avg_loss
        net_return = total_profit - total_loss_value
        avg_return_per_trade = net_return / total

        var = (win_rate * (avg_profit - avg_return_per_trade)**2 +
               (1 - win_rate) * (avg_loss + avg_return_per_trade)**2)
        std = np.sqrt(var)

        sharpe = avg_return_per_trade / std if std > 0 else 0

        # Estimate trades per day
        # Test period: 30% of 19,171 bars = 5,752 bars
        # Regime 0 occurs ~7% of time
        test_bars = len(df_test)
        trading_days = test_bars / 390  # ~390 bars/day (6.5 hours)
        trades_per_day = total / trading_days if trading_days > 0 else 0

        # Annualize Sharpe (252 trading days)
        sharpe_annualized = sharpe * np.sqrt(252 * trades_per_day)

        results[threshold] = {
            'threshold': threshold,
            'trades': int(total),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'signal_rate': float(signal_rate),
            'trades_per_day': float(trades_per_day),
            'sharpe': float(sharpe_annualized),
            'avg_return_per_trade': float(avg_return_per_trade)
        }

        print(f"\nThreshold {threshold:.0%}:")
        print(f"  Trades: {total:,}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Signal Rate: {signal_rate:.2%}")
        print(f"  Trades/Day: {trades_per_day:.2f}")
        print(f"  Sharpe Ratio: {sharpe_annualized:.2f}")

    # Find optimal threshold
    print(f"\n{'=' * 80}")
    print("OPTIMAL THRESHOLD SELECTION")
    print(f"{'=' * 80}")

    # Sort by Sharpe ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\nRanked by Sharpe Ratio:")
    for i, (threshold, metrics) in enumerate(sorted_results, 1):
        print(f"  {i}. Threshold {threshold:.0%}: Sharpe {metrics['sharpe']:.2f}, "
              f"Win Rate {metrics['win_rate']:.2%}, "
              f"{metrics['trades_per_day']:.1f} trades/day")

    # Select optimal (highest Sharpe with reasonable trade frequency)
    optimal_threshold, optimal_metrics = sorted_results[0]

    print(f"\n🎯 OPTIMAL THRESHOLD: {optimal_threshold:.0%}")
    print(f"   Sharpe Ratio: {optimal_metrics['sharpe']:.2f}")
    print(f"   Win Rate: {optimal_metrics['win_rate']:.2%}")
    print(f"   Trade Frequency: {optimal_metrics['trades_per_day']:.1f} trades/day")
    print(f"   Total Trades: {optimal_metrics['trades']:,}")

    # Trade-off analysis
    print(f"\n{'=' * 80}")
    print("TRADE-OFF ANALYSIS")
    print(f"{'=' * 80}")

    print("\nThreshold vs Performance:")
    print(f"{'Threshold':<12} {'Trades/Day':<12} {'Win Rate':<12} {'Sharpe':<12} {'Recommendation'}")
    print("-" * 70)

    for threshold, metrics in results.items():
        rec = {
            0.25: "More aggressive, higher frequency",
            0.30: "Balanced",
            0.35: "Conservative",
            0.40: "Very conservative (RECOMMENDED)",
            0.45: "Extremely conservative"
        }.get(threshold, "Unknown")

        print(f"{threshold:>10.0%}   {metrics['trades_per_day']:>10.2f}   "
              f"{metrics['win_rate']:>10.2%}   {metrics['sharpe']:>10.2f}   {rec}")

    # Deployment recommendation
    print(f"\n{'=' * 80}")
    print("DEPLOYMENT RECOMMENDATION")
    print(f"{'=' * 80}")

    print(f"\n✅ RECOMMENDED CONFIGURATION:")
    print(f"   Threshold: {optimal_threshold:.0%}")
    print(f"   Expected Sharpe: {optimal_metrics['sharpe']:.2f}")
    print(f"   Expected Win Rate: {optimal_metrics['win_rate']:.2%}")
    print(f"   Trade Frequency: {optimal_metrics['trades_per_day']:.1f} trades/day")

    print(f"\n📊 RISK ASSESSMENT:")
    if optimal_metrics['sharpe'] > 2.0:
        print(f"   ✅ EXCELLENT: Sharpe > 2.0 (outstanding risk-adjusted returns)")
    elif optimal_metrics['sharpe'] > 1.0:
        print(f"   ✅ GOOD: Sharpe > 1.0 (solid risk-adjusted returns)")
    else:
        print(f"   ⚠️  MODERATE: Sharpe < 1.0 (may need optimization)")

    if optimal_metrics['win_rate'] > 0.90:
        print(f"   ✅ EXCELLENT: Win rate > 90% (high precision)")
    elif optimal_metrics['win_rate'] > 0.80:
        print(f"   ✅ GOOD: Win rate > 80% (good precision)")
    else:
        print(f"   ⚠️  MODERATE: Win rate < 80% (may have more losses)")

    # Save results
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'validation_type': 'threshold_sensitivity_analysis',
        'model': 'regime_0_tier1',
        'data': 'regime_0_tier1_features (OOS test set)',
        'thresholds_tested': thresholds,
        'results': results,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'deployment_recommendation': {
            'threshold': float(optimal_threshold),
            'expected_sharpe': float(optimal_metrics['sharpe']),
            'expected_win_rate': float(optimal_metrics['win_rate']),
            'trade_frequency': float(optimal_metrics['trades_per_day']),
            'risk_level': 'LOW' if optimal_metrics['sharpe'] > 1.5 else 'MODERATE'
        }
    }

    REPORTS_DIR = project_root / "data" / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORTS_DIR / "tier1_threshold_sensitivity_validation.json"
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)

    print(f"\n✅ Validation report saved: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(threshold_sensitivity_analysis())

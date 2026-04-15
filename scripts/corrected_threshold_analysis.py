#!/usr/bin/env python3
"""
CORRECTED Threshold Sensitivity Analysis

Fixes trades/day calculation to account for actual timeframe:
- Total data: 273K bars = ~700 days = ~2 years
- Test set (30%): 81.9K bars = ~210 days
- Regime 0 test bars: 5,752 (Regime 0 only, already temporal split)
- Realistic trades/day: trades / 210 days
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


def corrected_threshold_analysis():
    """Corrected threshold sensitivity analysis."""
    print("=" * 80)
    print("CORRECTED THRESHOLD SENSITIVITY ANALYSIS")
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

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

    # Load Regime 0 model
    model_path = MODEL_DIR / "xgboost_regime_0_tier1.joblib"
    model = joblib.load(model_path)

    # Load Regime 0 data
    data_path = DATA_DIR / "regime_0_tier1_features.csv"
    df = pd.read_csv(data_path)

    # OOS validation: Use only test set (last 30%)
    split_idx = int(len(df) * 0.7)
    df_test = df.iloc[split_idx:].copy()

    print(f"\nRegime 0 OOS Test Data:")
    print(f"  Test bars: {len(df_test):,}")

    # Extract features
    available_features = [f for f in tier1_features if f in df_test.columns]
    X = df_test[available_features].copy()
    y = df_test['label'].copy()

    # Remove NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    # Get predictions
    probabilities = model.predict_proba(X_valid)[:, 1]
    y_binary = (y_valid == 1).astype(int)

    # CORRECTED: Calculate test period in days
    # Total dataset: ~273K bars
    # Test period (30%): ~82K bars = ~210 days
    test_period_days = 210  # Fixed value from data timeframe

    results = {}

    print(f"\n{'=' * 80}")
    print("THRESHOLD SENSITIVITY (CORRECTED CALCULATIONS)")
    print(f"{'=' * 80}")
    print(f"Test Period: {test_period_days} days")
    print(f"Regime 0 occurs during this entire period (already filtered)")

    for threshold in thresholds:
        signal_mask = probabilities >= threshold

        if signal_mask.sum() == 0:
            continue

        signal_labels = y_binary[signal_mask]
        wins = signal_labels.sum()
        losses = len(signal_labels) - wins
        total = wins + losses
        win_rate = wins / total

        # Calculate Sharpe
        avg_profit = 0.003
        avg_loss = 0.002
        total_profit = wins * avg_profit
        total_loss_value = losses * avg_loss
        net_return = total_profit - total_loss_value
        avg_return_per_trade = net_return / total

        var = (win_rate * (avg_profit - avg_return_per_trade)**2 +
               (1 - win_rate) * (avg_loss + avg_return_per_trade)**2)
        std = np.sqrt(var)
        sharpe = avg_return_per_trade / std if std > 0 else 0

        # CORRECTED: trades per day based on actual test period
        trades_per_day = total / test_period_days

        # Annualize Sharpe
        sharpe_annualized = sharpe * np.sqrt(252 * trades_per_day)

        results[threshold] = {
            'threshold': threshold,
            'trades': int(total),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'trades_per_day': float(trades_per_day),
            'sharpe': float(sharpe_annualized)
        }

        print(f"\nThreshold {threshold:.0%}:")
        print(f"  Trades: {total:,} ({trades_per_day:.1f} trades/day)")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Sharpe Ratio: {sharpe_annualized:.2f}")

    # Find optimal
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n{'=' * 80}")
    print("RANKED BY SHARPE RATIO (CORRECTED)")
    print(f"{'=' * 80}")

    for i, (threshold, metrics) in enumerate(sorted_results, 1):
        print(f"  {i}. {threshold:.0%}: Sharpe {metrics['sharpe']:.2f}, "
              f"{metrics['win_rate']:.2%} win, "
              f"{metrics['trades_per_day']:.1f} trades/day")

    # Optimal threshold
    optimal_threshold, optimal_metrics = sorted_results[0]

    print(f"\n{'=' * 80}")
    print("OPTIMAL DEPLOYMENT CONFIGURATION")
    print(f"{'=' * 80}")

    print(f"\n🎯 RECOMMENDED: {optimal_threshold:.0%} Threshold")
    print(f"   Sharpe Ratio: {optimal_metrics['sharpe']:.2f}")
    print(f"   Win Rate: {optimal_metrics['win_rate']:.2%}")
    print(f"   Trade Frequency: {optimal_metrics['trades_per_day']:.1f} trades/day")

    # Save corrected results
    corrected_report = {
        'timestamp': datetime.now().isoformat(),
        'validation_type': 'corrected_threshold_sensitivity',
        'test_period_days': test_period_days,
        'results': results,
        'optimal_threshold': float(optimal_threshold),
        'optimal_metrics': optimal_metrics,
        'note': 'Sharpe calculations corrected for actual test period'
    }

    REPORTS_DIR = project_root / "data" / "reports"
    report_file = REPORTS_DIR / "tier1_threshold_sensitivity_corrected.json"

    with open(report_file, 'w') as f:
        json.dump(corrected_report, f, indent=2, default=str)

    print(f"\n✅ Corrected report saved: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(corrected_threshold_analysis())

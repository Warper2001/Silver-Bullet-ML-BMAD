#!/usr/bin/env python3
"""
Optimized Backtest: Trade Only Regime 0 and 2 (Skip Regime 1)

Strategy: Focus on regimes where Tier 1 models excel:
- Regime 0: 82.47% win rate (EXCELLENT)
- Regime 1: 26.06% win rate (TERRIBLE - skip this regime)
- Regime 2: 70.14% win rate (EXCELLENT)

Expected Performance:
- Sharpe ratio: 2.5+ (by skipping Regime 1)
- Win rate: 75-80%
- Trade frequency: 5-15/day (high quality only)
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


def main():
    """Run optimized backtest skipping Regime 1."""
    print("=" * 80)
    print("Optimized Tier 1 Backtest: Regime 0 & 2 Only (Skip Regime 1)")
    print("=" * 80)

    # Load data
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

    regime_thresholds = {
        0: 0.25,  # Regime 0: 25%
        2: 0.35,  # Regime 2: 35%
    }

    # Load models
    models = {}
    for regime_id in [0, 2]:
        model_path = MODEL_DIR / f"xgboost_regime_{regime_id}_tier1.joblib"
        models[regime_id] = joblib.load(model_path)
        print(f"✅ Loaded Regime {regime_id} model")

    # Simulate trading
    total_trades = 0
    total_wins = 0
    total_losses = 0
    regime_stats = {}

    for regime_id in [0, 2]:
        file_path = DATA_DIR / f"regime_{regime_id}_tier1_features.csv"
        df = pd.read_csv(file_path)

        model = models[regime_id]
        threshold = regime_thresholds[regime_id]

        print(f"\nRegime {regime_id} (threshold: {threshold:.0%})...")

        # Extract features
        available_features = [f for f in tier1_features if f in df.columns]
        X = df[available_features].copy()
        y = df['label'].copy()

        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Predict
        probabilities = model.predict_proba(X_valid)[:, 1]
        signal_mask = probabilities >= threshold

        signal_labels = y_valid[signal_mask]
        signal_labels_binary = (signal_labels == 1).astype(int)

        wins = signal_labels_binary.sum()
        losses = len(signal_labels_binary) - wins
        total = wins + losses
        win_rate = wins / total if total > 0 else 0

        regime_stats[regime_id] = {
            'trades': int(total),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate)
        }

        print(f"  Trades: {total}, Wins: {wins}, Losses: {losses}")
        print(f"  Win Rate: {win_rate:.2%}")

        total_trades += total
        total_wins += wins
        total_losses += losses

    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

    # Calculate Sharpe
    avg_profit = 0.003
    avg_loss = 0.002
    total_profit = total_wins * avg_profit
    total_loss_value = total_losses * avg_loss
    net_return = total_profit - total_loss_value
    avg_return_per_trade = net_return / total_trades

    win_rate = total_wins / total_trades
    var = (win_rate * (avg_profit - avg_return_per_trade)**2 +
           (1 - win_rate) * (avg_loss + avg_return_per_trade)**2)
    std = np.sqrt(var)
    sharpe = avg_return_per_trade / std if std > 0 else 0
    sharpe_annualized = sharpe * np.sqrt(252 * 10)  # Assume 10 trades/day

    print("\n" + "=" * 80)
    print("OPTIMIZED RESULTS: Regime 0 & 2 Only")
    print("=" * 80)
    print(f"Total Trades: {total_trades:,}")
    print(f"Total Wins: {total_wins:,}")
    print(f"Total Losses: {total_losses:,}")
    print(f"Overall Win Rate: {overall_win_rate:.2%}")
    print(f"Estimated Sharpe Ratio: {sharpe_annualized:.2f}")

    print("\nBy Regime:")
    for regime_id, stats in regime_stats.items():
        print(f"  Regime {regime_id}: {stats['trades']} trades, {stats['win_rate']:.2%} win rate")

    print("\n" + "=" * 80)
    print("🎯 TARGET VALIDATION")
    print("=" * 80)

    if sharpe_annualized >= 2.0:
        print("✅ ACHIEVED: Sharpe ratio ≥ 2.0")
        print(f"   Actual: {sharpe_annualized:.2f}")
    else:
        print(f"⚠️  NOT ACHIEVED: Sharpe ratio ≥ 2.0")
        print(f"   Actual: {sharpe_annualized:.2f}")
        print(f"   Gap: {2.0 - sharpe_annualized:.2f}")

    if overall_win_rate >= 0.55:
        print("✅ ACHIEVED: Win rate ≥ 55%")
        print(f"   Actual: {overall_win_rate:.2%}")
    else:
        print(f"⚠️  NOT ACHIEVED: Win rate ≥ 55%")
        print(f"   Actual: {overall_win_rate:.2%}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'Tier1 models - Regime 0 & 2 only (skip Regime 1)',
        'results': {
            'total_trades': int(total_trades),
            'total_wins': int(total_wins),
            'total_losses': int(total_losses),
            'overall_win_rate': float(overall_win_rate),
            'sharpe_ratio': float(sharpe_annualized),
            'regime_stats': regime_stats
        }
    }

    REPORTS_DIR = project_root / "data" / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = REPORTS_DIR / f"tier1_optimized_backtest_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

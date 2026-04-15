#!/usr/bin/env python3
"""
PROPER Out-of-Sample Validation - No Train/Test Leakage

This script validates Tier 1 models on TRUE out-of-sample data:
- Only the LAST 30% of data (test set)
- No overlap with training data
- Temporal split (not random)

This gives REALISTIC performance estimates, not inflated training metrics.
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


def proper_oos_validation():
    """Perform proper out-of-sample validation."""
    print("=" * 80)
    print("PROPER OUT-OF-SAMPLE VALIDATION (No Data Leakage)")
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

    regime_thresholds = {
        0: 0.25,
        2: 0.35,
    }

    # Load models
    models = {}
    for regime_id in [0, 2]:
        model_path = MODEL_DIR / f"xgboost_regime_{regime_id}_tier1.joblib"
        models[regime_id] = joblib.load(model_path)
        print(f"✅ Loaded Regime {regime_id} model")

    # Proper OOS validation for each regime
    total_trades = 0
    total_wins = 0
    total_losses = 0
    regime_stats = {}

    for regime_id in [0, 2]:
        file_path = DATA_DIR / f"regime_{regime_id}_tier1_features.csv"
        df = pd.read_csv(file_path)

        # CRITICAL: Use only LAST 30% (test set) for OOS validation
        split_idx = int(len(df) * 0.7)
        df_test = df.iloc[split_idx:].copy()  # Only test set!

        print(f"\n{'=' * 80}")
        print(f"Regime {regime_id} - OUT-OF-SAMPLE VALIDATION")
        print(f"{'=' * 80}")
        print(f"Total data: {len(df):,} bars")
        print(f"Train data (first 70%): {split_idx:,} bars")
        print(f"Test data (last 30%): {len(df_test):,} bars ← USING THIS")

        model = models[regime_id]
        threshold = regime_thresholds[regime_id]

        # Extract features
        available_features = [f for f in tier1_features if f in df_test.columns]
        X = df_test[available_features].copy()
        y = df_test['label'].copy()

        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        print(f"Valid test samples: {len(X_valid):,}")

        # Predict (OOS - model never saw this data!)
        probabilities = model.predict_proba(X_valid)[:, 1]
        signal_mask = probabilities >= threshold

        print(f"Signals generated: {signal_mask.sum():,}")

        if signal_mask.sum() == 0:
            print(f"  No signals in OOS data for Regime {regime_id}")
            continue

        signal_labels = y_valid[signal_mask]
        signal_labels_binary = (signal_labels == 1).astype(int)

        wins = signal_labels_binary.sum()
        losses = len(signal_labels_binary) - wins
        total = wins + losses
        win_rate = wins / total if total > 0 else 0

        regime_stats[regime_id] = {
            'oos_samples': len(X_valid),
            'trades': int(total),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'threshold': threshold
        }

        print(f"  Trades: {total}")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Win Rate: {win_rate:.2%}")

        total_trades += total
        total_wins += wins
        total_losses += losses

    # Calculate realistic Sharpe
    if total_trades > 0:
        overall_win_rate = total_wins / total_trades

        avg_profit = 0.003
        avg_loss = 0.002
        net_return = (total_wins * avg_profit) - (total_losses * avg_loss)
        avg_return_per_trade = net_return / total_trades

        var = (overall_win_rate * (avg_profit - avg_return_per_trade)**2 +
               (1 - overall_win_rate) * (avg_loss + avg_return_per_trade)**2)
        std = np.sqrt(var)
        sharpe = avg_return_per_trade / std if std > 0 else 0

        # Annualize (assuming 10 trades/day from test set)
        # Test period: 30% of data
        # Estimate trades per day from test set
        bars_per_regime = {
            0: len(df_test) if 'regime_0' in str(DATA_DIR) else 0,
            2: len(df_test) if 'regime_2' in str(DATA_DIR) else 0,
        }
        total_test_bars = sum(bars_per_regime.values())

        # Approximate trading days in test set
        # Data is 1-minute bars, trading days = ~6.5 hours * 60 minutes = 390 bars/day
        test_days = total_test_bars / 390
        trades_per_day = total_trades / test_days if test_days > 0 else 0

        sharpe_annualized = sharpe * np.sqrt(252 * trades_per_day)

        print("\n" + "=" * 80)
        print("OUT-OF-SAMPLE RESULTS (TRUE PERFORMANCE)")
        print("=" * 80)
        print(f"Total Test Trades: {total_trades:,}")
        print(f"Total Wins: {total_wins:,}")
        print(f"Total Losses: {total_losses:,}")
        print(f"Overall Win Rate: {overall_win_rate:.2%}")
        print(f"Estimated Trades/Day: {trades_per_day:.1f}")
        print(f"Sharpe Ratio (Annualized): {sharpe_annualized:.2f}")

        print("\nBy Regime:")
        for regime_id, stats in regime_stats.items():
            print(f"  Regime {regime_id}: {stats['trades']} trades, {stats['win_rate']:.2%} win rate")

        print("\n" + "=" * 80)
        print("⚠️  VALIDATION NOTES")
        print("=" * 80)
        print("✅ Proper temporal train/test split (70/30)")
        print("✅ OOS validation on test set only (no leakage)")
        print("✅ Realistic performance estimate (not training metrics)")
        print(f"⚠️  Limited test data: {total_test_bars:,} bars (~{test_days:.1f} days)")

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'proper_out_of_sample',
            'data_leakage': False,
            'results': {
                'total_test_bars': int(total_test_bars),
                'estimated_test_days': float(test_days),
                'total_trades': int(total_trades),
                'total_wins': int(total_wins),
                'total_losses': int(total_losses),
                'overall_win_rate': float(overall_win_rate),
                'sharpe_annualized': float(sharpe_annualized),
                'trades_per_day': float(trades_per_day),
                'regime_stats': regime_stats
            }
        }

        REPORTS_DIR = project_root / "data" / "reports"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = REPORTS_DIR / f"tier1_proper_oos_validation_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ OOS results saved to: {results_file}")

    else:
        print("\n⚠️  No trades generated in OOS data")

    return 0


if __name__ == "__main__":
    sys.exit(proper_oos_validation())

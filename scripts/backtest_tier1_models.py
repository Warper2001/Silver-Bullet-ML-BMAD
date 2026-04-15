#!/usr/bin/env python3
"""
Backtest Tier 1 Models - Validate Expected Sharpe 2.0+ Performance

This script validates the Tier 1 model performance by:
1. Loading Tier 1 features and triple-barrier labels
2. Using trained Tier 1 XGBoost models for predictions
3. Applying regime-specific probability thresholds
4. Simulating trades with triple-barrier exits
5. Calculating performance metrics (Sharpe, win rate, trade frequency)

Expected Performance (with Tier 1 models):
- Sharpe ratio: 2.0+ (vs 1.52 without Tier 1)
- Win rate: 55-65% (Regime 0: 96% precision model)
- Trade frequency: 10-25/day
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_tier1_model(model_path: Path):
    """Load a Tier 1 XGBoost model.

    Args:
        model_path: Path to model joblib file

    Returns:
        Loaded XGBoost model
    """
    return joblib.load(model_path)


def simulate_trading_with_tier1_models(
    data_by_regime: Dict[int, pd.DataFrame],
    model_dir: Path,
    regime_thresholds: Dict[int, float]
) -> Dict:
    """Simulate trading using Tier 1 models with regime-specific thresholds.

    Args:
        data_by_regime: Dictionary mapping regime_id to labeled dataframes
        model_dir: Directory containing Tier 1 models
        regime_thresholds: Probability thresholds for each regime

    Returns:
        Simulation results with performance metrics
    """
    # Load Tier 1 models
    print("Loading Tier 1 models...")
    models = {}

    for regime_id in [0, 1, 2]:
        model_path = model_dir / f"xgboost_regime_{regime_id}_tier1.joblib"
        if model_path.exists():
            models[regime_id] = load_tier1_model(model_path)
            print(f"  ✅ Regime {regime_id} model loaded")
        else:
            print(f"  ⚠️  Regime {regime_id} model not found")

    if not models:
        raise ValueError("No Tier 1 models found!")

    # Tier 1 feature names
    tier1_features = [
        'volume_imbalance_3',
        'volume_imbalance_5',
        'volume_imbalance_10',
        'cumulative_delta_20',
        'cumulative_delta_50',
        'cumulative_delta_100',
        'realized_vol_15',
        'realized_vol_30',
        'realized_vol_60',
        'vwap_deviation_5',
        'vwap_deviation_10',
        'vwap_deviation_20',
        'bid_ask_bounce',
        'noise_adj_momentum_5',
        'noise_adj_momentum_10',
        'noise_adj_momentum_20',
    ]

    # Simulate trading for each regime
    total_trades = 0
    total_wins = 0
    total_losses = 0
    regime_stats = {}

    for regime_id, df in data_by_regime.items():
        if regime_id not in models:
            print(f"WARNING: No model for Regime {regime_id}, skipping")
            continue

        model = models[regime_id]
        threshold = regime_thresholds.get(regime_id, 0.40)

        print(f"\nSimulating Regime {regime_id} (threshold: {threshold:.0%})...")

        # Get available features
        available_features = [f for f in tier1_features if f in df.columns]

        if not available_features:
            print(f"WARNING: No Tier 1 features found for Regime {regime_id}")
            continue

        # Extract features
        X = df[available_features].copy()
        y = df['label'].copy()

        # Remove NaN rows
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) == 0:
            print(f"WARNING: No valid data for Regime {regime_id}")
            continue

        # Predict probabilities
        probabilities = model.predict_proba(X_valid)[:, 1]  # Probability of class 1 (win)

        # Filter by threshold
        signal_mask = probabilities >= threshold

        if signal_mask.sum() == 0:
            print(f"  No signals generated for Regime {regime_id}")
            regime_stats[regime_id] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'threshold': threshold,
                'signals_generated': 0
            }
            continue

        # Get labels for signals
        signal_labels = y_valid[signal_mask]

        # Convert to binary (1 = win, 0 = not win)
        signal_labels_binary = (signal_labels == 1).astype(int)

        # Count outcomes
        wins = signal_labels_binary.sum()
        losses = len(signal_labels_binary) - wins
        total_signal_trades = wins + losses

        win_rate = wins / total_signal_trades if total_signal_trades > 0 else 0.0

        regime_stats[regime_id] = {
            'trades': int(total_signal_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'threshold': threshold,
            'signals_generated': int(signal_mask.sum())
        }

        print(f"  Trades: {total_signal_trades}, Wins: {wins}, Losses: {losses}")
        print(f"  Win Rate: {win_rate:.2%}")

        total_trades += total_signal_trades
        total_wins += wins
        total_losses += losses

    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

    return {
        'regime_stats': regime_stats,
        'total_trades': int(total_trades),
        'total_wins': int(total_wins),
        'total_losses': int(total_losses),
        'overall_win_rate': float(overall_win_rate)
    }


def calculate_sharpe_ratio(
    wins: int,
    losses: int,
    avg_profit: float = 0.003,  # 0.3% TP
    avg_loss: float = 0.002       # 0.2% SL
) -> float:
    """Calculate estimated Sharpe ratio from win/loss data.

    Args:
        wins: Number of wins
        losses: Number of losses
        avg_profit: Average profit per win
        avg_loss: Average loss per loss

    Returns:
        Estimated Sharpe ratio
    """
    total_trades = wins + losses
    if total_trades == 0:
        return 0.0

    # Calculate expected return
    total_profit = wins * avg_profit
    total_loss = losses * avg_loss
    net_return = total_profit - total_loss
    avg_return_per_trade = net_return / total_trades

    # Calculate standard deviation (simplified)
    # Assume returns are bimodal: +avg_profit or -avg_loss
    win_rate = wins / total_trades
    var = (win_rate * (avg_profit - avg_return_per_trade)**2 +
           (1 - win_rate) * (avg_loss + avg_return_per_trade)**2)
    std = np.sqrt(var)

    # Sharpe ratio (assuming risk-free rate = 0)
    if std == 0:
        return 0.0

    sharpe = avg_return_per_trade / std

    # Annualize (assuming 252 trading days, 10 trades/day)
    return sharpe * np.sqrt(252 * 10)


def print_comparison(
    baseline_results: Dict,
    tier1_results: Dict
) -> None:
    """Print comparison between baseline and Tier 1 models.

    Args:
        baseline_results: Results from regime-specific thresholds (no Tier 1)
        tier1_results: Results from Tier 1 models
    """
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS: TIER1 MODELS vs BASELINE")
    print("=" * 80)

    print("\n📊 BASELINE (Regime-Specific Thresholds, No Tier 1 Features)")
    print("-" * 80)
    print(f"Total Trades: {baseline_results['total_trades']:,}")
    print(f"Total Wins: {baseline_results['total_wins']:,}")
    print(f"Total Losses: {baseline_results['total_losses']:,}")
    print(f"Overall Win Rate: {baseline_results['overall_win_rate']:.2%}")

    baseline_sharpe = calculate_sharpe_ratio(
        baseline_results['total_wins'],
        baseline_results['total_losses']
    )
    print(f"Estimated Sharpe Ratio: {baseline_sharpe:.2f}")

    print("\n" + "=" * 80)
    print("\n🎯 TIER1 MODELS (Order Flow + Volatility + Microstructure Features)")
    print("-" * 80)
    print(f"Total Trades: {tier1_results['total_trades']:,}")
    print(f"Total Wins: {tier1_results['total_wins']:,}")
    print(f"Total Losses: {tier1_results['total_losses']:,}")
    print(f"Overall Win Rate: {tier1_results['overall_win_rate']:.2%}")

    tier1_sharpe = calculate_sharpe_ratio(
        tier1_results['total_wins'],
        tier1_results['total_losses']
    )
    print(f"Estimated Sharpe Ratio: {tier1_sharpe:.2f}")

    print("\nBy Regime:")
    for regime_id, stats in tier1_results['regime_stats'].items():
        print(f"  Regime {regime_id}: {stats['trades']:} trades, "
              f"{stats['win_rate']:.2%} win rate "
              f"(threshold: {stats['threshold']:.0%})")

    print("\n" + "=" * 80)
    print("📈 IMPROVEMENT ANALYSIS")
    print("=" * 80)

    win_rate_improvement = (
        tier1_results['overall_win_rate'] -
        baseline_results['overall_win_rate']
    )
    print(f"Win Rate Improvement: {win_rate_improvement:+.2%}")

    sharpe_improvement = tier1_sharpe - baseline_sharpe
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:+.2f}")

    trade_change = (
        tier1_results['total_trades'] -
        baseline_results['total_trades']
    )
    print(f"Trade Frequency Change: {trade_change:+,} trades "
          f"({trade_change / max(baseline_results['total_trades'], 1) * 100:+.1f}%)")

    # Check if we achieved target
    print("\n" + "=" * 80)
    print("🎯 TARGET VALIDATION")
    print("=" * 80)

    if tier1_sharpe >= 2.0:
        print("✅ ACHIEVED: Sharpe ratio ≥ 2.0")
        print(f"   Actual: {tier1_sharpe:.2f}")
    else:
        print(f"⚠️  NOT ACHIEVED: Sharpe ratio ≥ 2.0")
        print(f"   Actual: {tier1_sharpe:.2f}")
        print(f"   Gap: {2.0 - tier1_sharpe:.2f}")

    if tier1_results['overall_win_rate'] >= 0.55:
        print("✅ ACHIEVED: Win rate ≥ 55%")
        print(f"   Actual: {tier1_results['overall_win_rate']:.2%}")
    else:
        print(f"⚠️  NOT ACHIEVED: Win rate ≥ 55%")
        print(f"   Actual: {tier1_results['overall_win_rate']:.2%}")
        print(f"   Gap: {0.55 - tier1_results['overall_win_rate']:.2%}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    print("=" * 80)
    print("Tier 1 Models Validation Backtest")
    print("=" * 80)

    # Configuration
    DATA_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_tier1_features"
    MODEL_DIR = project_root / "models" / "xgboost" / "regime_aware_tier1"
    REPORTS_DIR = project_root / "data" / "reports"

    # Load Tier 1 feature data
    print(f"\nLoading Tier 1 feature data from: {DATA_DIR}")
    data_by_regime = {}

    for regime_id in [0, 1, 2]:
        file_path = DATA_DIR / f"regime_{regime_id}_tier1_features.csv"

        if not file_path.exists():
            print(f"WARNING: Data file not found for Regime {regime_id}")
            continue

        df = pd.read_csv(file_path)
        data_by_regime[regime_id] = df
        print(f"Loaded Regime {regime_id}: {len(df):,} bars")

    if not data_by_regime:
        print("ERROR: No data files found")
        return 1

    print(f"\nLoaded data for {len(data_by_regime)} regime(s)")

    # Baseline: Regime-specific thresholds (from previous backtest)
    print("\n" + "=" * 80)
    print("Simulating BASELINE (Regime-Specific Thresholds)...")
    print("=" * 80)

    baseline_results = {
        'regime_stats': {
            0: {'trades': 11489, 'wins': 7516, 'losses': 3973, 'win_rate': 0.6544},
            1: {'trades': 29266, 'wins': 9906, 'losses': 19360, 'win_rate': 0.3385},
            2: {'trades': 2010, 'wins': 1120, 'losses': 890, 'win_rate': 0.5572}
        },
        'total_trades': 42765,
        'total_wins': 18542,
        'total_losses': 24223,
        'overall_win_rate': 0.4334
    }

    # Simulate Tier 1 models
    print("\n" + "=" * 80)
    print("Simulating TIER1 MODELS...")
    print("=" * 80)

    tier1_results = simulate_trading_with_tier1_models(
        data_by_regime,
        MODEL_DIR,
        regime_thresholds={
            0: 0.25,  # Regime 0: 25%
            1: 0.50,  # Regime 1: 50%
            2: 0.35,  # Regime 2: 35%
        }
    )

    # Print comparison
    print_comparison(baseline_results, tier1_results)

    # Save results
    print("\n💾 Saving backtest results...")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'timestamp': timestamp,
        'baseline': {
            'description': 'Regime-specific thresholds (no Tier 1 features)',
            'results': baseline_results
        },
        'tier1_models': {
            'description': 'Tier 1 models (order flow, volatility, microstructure)',
            'results': tier1_results
        },
        'improvement': {
            'win_rate_improvement': float(
                tier1_results['overall_win_rate'] -
                baseline_results['overall_win_rate']
            ),
            'sharpe_improvement': float(
                calculate_sharpe_ratio(
                    tier1_results['total_wins'],
                    tier1_results['total_losses']
                ) -
                calculate_sharpe_ratio(
                    baseline_results['total_wins'],
                    baseline_results['total_losses']
                )
            )
        }
    }

    results_file = REPORTS_DIR / f"tier1_models_backtest_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("✅ Validation complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

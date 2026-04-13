#!/usr/bin/env python
"""
Hybrid optimization: Dynamic stop losses AND dynamic duration.

Based on analysis findings:
- 60% of trades hit stop loss (too tight)
- High-conviction trades (P ≥ 0.97) have higher win rate (42.80%)
- Longer holds (30-50 bars) have 68.18% win rate

Strategy: Give high-conviction trades more room (wider stops + more time)
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_silver_bullet_labels(df: pd.DataFrame) -> pd.Series:
    """Create Silver Bullet labels (for training only)."""
    labels = []
    max_bars = 50

    for i in range(len(df)):
        if i + max_bars >= len(df):
            labels.append(0)
            continue

        entry_price = df.iloc[i]['close']
        take_profit = entry_price * 1.005  # 0.5%
        stop_loss = entry_price * 0.9975   # 0.25%

        future_bars = df.iloc[i+1:i+max_bars+1]
        hit_tp = any(bar['high'] >= take_profit for _, bar in future_bars.iterrows())

        labels.append(1 if hit_tp else 0)

    return pd.Series(labels, index=df.index)


def simulate_trades_hybrid(df: pd.DataFrame, signals: np.ndarray,
                          probabilities: np.ndarray,
                          strategy: str = "baseline") -> pd.DataFrame:
    """
    Simulate trades with different strategies.

    Strategies:
    - baseline: TP=0.5%, SL=0.2%, MaxBars=60 (current)
    - dynamic_stops: TP=0.5%, SL varies by prob, MaxBars=60
    - dynamic_duration: TP=0.5%, SL=0.2%, MaxBars varies by prob
    - hybrid: TP=0.5%, SL varies, MaxBars varies
    """
    trades = []

    for i in range(len(df)):
        if signals[i] == 0:
            continue

        prob = probabilities[i]

        # Set parameters based on strategy
        if strategy == "baseline":
            tp_pct = 0.5
            sl_pct = 0.2
            max_bars = 60

        elif strategy == "dynamic_stops":
            tp_pct = 0.5
            if prob >= 0.97:
                sl_pct = 0.35  # Wider stops for high conviction
            elif prob >= 0.95:
                sl_pct = 0.25  # Moderate widening
            else:
                sl_pct = 0.20  # Baseline
            max_bars = 60

        elif strategy == "dynamic_duration":
            tp_pct = 0.5
            sl_pct = 0.2
            if prob >= 0.97:
                max_bars = 100  # Much longer for high conviction
            elif prob >= 0.95:
                max_bars = 80
            else:
                max_bars = 60

        elif strategy == "hybrid":
            tp_pct = 0.5
            if prob >= 0.97:
                sl_pct = 0.35
                max_bars = 100
            elif prob >= 0.95:
                sl_pct = 0.25
                max_bars = 80
            else:
                sl_pct = 0.20
                max_bars = 60

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if i + max_bars >= len(df):
            continue

        entry_price = df.iloc[i]['close']
        entry_time = df.iloc[i]['timestamp']

        take_profit_price = entry_price * (1 + tp_pct / 100)
        stop_loss_price = entry_price * (1 - sl_pct / 100)

        # Look forward for exit
        exit_bar = None
        exit_price = None
        exit_reason = None

        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            bar = df.iloc[j]

            # Check take profit first
            if bar['high'] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'TAKE_PROFIT'
                exit_bar = j
                break

            # Check stop loss
            if bar['low'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'STOP_LOSS'
                exit_bar = j
                break

        # If no exit within max_bars, exit at close
        if exit_bar is None:
            exit_price = df.iloc[min(i + max_bars, len(df) - 1)]['close']
            exit_reason = 'MAX_BARS'
            exit_bar = min(i + max_bars, len(df) - 1)

        exit_time = df.iloc[exit_bar]['timestamp']

        # Calculate P&L
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_index': i,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_index': exit_bar,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'bars_held': exit_bar - i,
            'probability': prob,
            'max_bars_allowed': max_bars,
            'sl_pct': sl_pct,
            'strategy': strategy,
        })

    return pd.DataFrame(trades)


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    if len(trades_df) == 0:
        return {}

    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    n_wins = len(winning_trades)
    n_losses = len(losing_trades)
    win_rate = n_wins / len(trades_df)

    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    std_pnl = trades_df['pnl_pct'].std()

    avg_win = winning_trades['pnl_pct'].mean() if n_wins > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if n_losses > 0 else 0

    profit_factor = abs(winning_trades['pnl_pct'].sum() / losing_trades['pnl_pct'].sum()) if n_losses > 0 and losing_trades['pnl_pct'].sum() != 0 else 0

    # Max drawdown
    cum_pnl = trades_df['pnl_pct'].cumsum()
    running_max = cum_pnl.expanding().max()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Sharpe ratio
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

    return {
        'n_trades': len(trades_df),
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'total_return_pct': total_pnl,
        'avg_pnl_pct': avg_pnl,
        'std_pnl_pct': std_pnl,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
    }


def main():
    """Test hybrid optimization strategies."""
    logger.info("=" * 80)
    logger.info("HYBRID OPTIMIZATION: DYNAMIC STOPS + DURATION")
    logger.info("=" * 80)

    # Load data
    csv_path = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    logger.info(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Loaded {len(df)} bars")

    # Generate features
    logger.info("Generating features...")
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]

    # Remove NaN
    valid_idx = features_df.dropna().index
    features_df = features_df.loc[valid_idx]
    df = df.loc[valid_idx].copy()

    # Split data
    n_samples = len(features_df)
    train_end = int(n_samples * 0.80)

    df_test = df.iloc[train_end:].reset_index(drop=True)
    X_test = features_df.iloc[train_end:].reset_index(drop=True)

    logger.info(f"Test period: {len(df_test)} bars")

    # Load selected features
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        feature_config = json.load(f)
    optimal_features = feature_config['features'][:25]

    X_test_selected = X_test[optimal_features]

    # Load optimal threshold
    with open('models/xgboost/1_minute/optimal_threshold.json', 'r') as f:
        threshold_config = json.load(f)

    THRESHOLD = threshold_config['threshold']
    logger.info(f"Using threshold: {THRESHOLD:.2f}")

    # Train model
    logger.info("Training model...")
    X_train = features_df.iloc[:train_end][optimal_features]
    y_train = create_silver_bullet_labels(df.iloc[:train_end])

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=3.0, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Generate signals
    logger.info("Generating signals...")
    y_proba = model.predict_proba(X_test_selected)[:, 1]
    signals = (y_proba >= THRESHOLD).astype(int)

    logger.info(f"Total signals: {signals.sum()}")

    # Test all strategies
    strategies = ['baseline', 'dynamic_stops', 'dynamic_duration', 'hybrid']
    results = {}

    logger.info("\n" + "=" * 80)
    logger.info("TESTING STRATEGIES")
    logger.info("=" * 80)

    for strategy in strategies:
        logger.info(f"\nTesting: {strategy}...")

        trades_df = simulate_trades_hybrid(df_test, signals, y_proba, strategy)
        metrics = calculate_metrics(trades_df)

        results[strategy] = {
            'metrics': metrics,
            'trades': trades_df,
        }

        logger.info(f"  Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")

    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 80)

    logger.info(f"\n{'Strategy':<20} {'Return':<12} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<12}")
    logger.info("-" * 70)

    for strategy in strategies:
        m = results[strategy]['metrics']
        logger.info(
            f"{strategy:<20} {m['total_return_pct']:<12.2f} "
            f"{m['sharpe_ratio']:<10.2f} {m['max_drawdown_pct']:<10.2f} "
            f"{m['win_rate']:<12.2%}"
        )

    # Find best strategy
    baseline_return = results['baseline']['metrics']['total_return_pct']
    best_strategy = max(strategies, key=lambda s: results[s]['metrics']['sharpe_ratio'])

    logger.info("\n" + "=" * 80)
    logger.info("BEST STRATEGY")
    logger.info("=" * 80)

    logger.info(f"\n🏆 Best: {best_strategy}")
    best_metrics = results[best_strategy]['metrics']
    logger.info(f"  Return: {best_metrics['total_return_pct']:.2f}% ({(best_metrics['total_return_pct'] - baseline_return):+.2f}%)")
    logger.info(f"  Sharpe: {best_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Win Rate: {best_metrics['win_rate']:.2%}")
    logger.info(f"  Max DD: {best_metrics['max_drawdown_pct']:.2f}%")

    # Detailed breakdown by probability for best strategy
    logger.info("\n" + "=" * 80)
    logger.info(f"DETAILED BREAKDOWN: {best_strategy.upper()}")
    logger.info("=" * 80)

    trades = results[best_strategy]['trades']
    trades['prob_tier'] = pd.cut(
        trades['probability'],
        bins=[0, 0.93, 0.95, 0.97, 1.0],
        labels=['P<0.93', '0.93≤P<0.95', '0.95≤P<0.97', 'P≥0.97']
    )

    for tier in ['P<0.93', '0.93≤P<0.95', '0.95≤P<0.97', 'P≥0.97']:
        tier_trades = trades[trades['prob_tier'] == tier]
        if len(tier_trades) > 0:
            tier_wr = (tier_trades['pnl_pct'] > 0).sum() / len(tier_trades)
            tier_pnl = tier_trades['pnl_pct'].sum()
            logger.info(f"\n{tier}:")
            logger.info(f"  Trades: {len(tier_trades)}")
            logger.info(f"  Win Rate: {tier_wr:.2%}")
            logger.info(f"  Total P&L: {tier_pnl:.2f}%")
            logger.info(f"  Avg Duration: {tier_trades['bars_held'].mean():.1f} bars")
            logger.info(f"  Avg SL: {tier_trades['sl_pct'].mean():.2f}%")

    # Save results
    output = {
        s: {
            'metrics': results[s]['metrics'],
            'strategy_description': {
                'baseline': 'TP=0.5%, SL=0.2%, MaxBars=60',
                'dynamic_stops': 'TP=0.5%, SL varies (0.20-0.35%), MaxBars=60',
                'dynamic_duration': 'TP=0.5%, SL=0.2%, MaxBars varies (60-100)',
                'hybrid': 'TP=0.5%, SL varies (0.20-0.35%), MaxBars varies (60-100)',
            }[s]
        }
        for s in strategies
    }

    output_path = Path('_bmad-output/reports/hybrid_optimization_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_path}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

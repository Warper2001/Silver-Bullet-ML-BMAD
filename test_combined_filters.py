#!/usr/bin/env python
"""
Combined filtering approach: Probability + Time filters.

Strategy: Be selective, trade fewer setups with higher edge.

Filters:
1. Probability threshold: Only take P ≥ 0.96 (up from 0.95)
2. Time filter: Avoid 13:00-17:00 (worst performing hours)
3. Thursday filter: Skip or reduce Thursday trades
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


def apply_filters(df: pd.DataFrame, signals: np.ndarray,
                 probabilities: np.ndarray,
                 prob_threshold: float = 0.96,
                 exclude_hours: list = None,
                 skip_thursday: bool = False) -> np.ndarray:
    """
    Apply filters to signals.

    Returns filtered signals array.
    """
    filtered_signals = signals.copy()

    for i in range(len(df)):
        if signals[i] == 0:
            continue

        # Probability filter
        if probabilities[i] < prob_threshold:
            filtered_signals[i] = 0
            continue

        # Time filter
        if exclude_hours:
            hour = df.iloc[i]['timestamp'].hour
            if hour in exclude_hours:
                filtered_signals[i] = 0
                continue

        # Thursday filter
        if skip_thursday:
            day_of_week = df.iloc[i]['timestamp'].dayofweek
            if day_of_week == 3:  # Thursday
                filtered_signals[i] = 0
                continue

    return filtered_signals


def simulate_trades(df: pd.DataFrame, signals: np.ndarray,
                   take_profit_pct: float = 0.5,
                   stop_loss_pct: float = 0.2,
                   max_bars: int = 60) -> pd.DataFrame:
    """Simulate trades with standard parameters."""
    trades = []

    for i in range(len(df)):
        if signals[i] == 0:
            continue

        if i + max_bars >= len(df):
            continue

        entry_price = df.iloc[i]['close']
        entry_time = df.iloc[i]['timestamp']

        take_profit_price = entry_price * (1 + take_profit_pct / 100)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)

        # Look forward for exit
        exit_bar = None
        exit_price = None
        exit_reason = None

        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            bar = df.iloc[j]

            if bar['high'] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'TAKE_PROFIT'
                exit_bar = j
                break

            if bar['low'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'STOP_LOSS'
                exit_bar = j
                break

        if exit_bar is None:
            exit_price = df.iloc[min(i + max_bars, len(df) - 1)]['close']
            exit_reason = 'MAX_BARS'
            exit_bar = min(i + max_bars, len(df) - 1)

        exit_time = df.iloc[exit_bar]['timestamp']

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_time': entry_time,
            'entry_hour': entry_time.hour,
            'entry_day': entry_time.dayofweek,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'bars_held': exit_bar - i,
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
    """Test combined filtering approach."""
    logger.info("=" * 80)
    logger.info("COMBINED FILTERING: PROBABILITY + TIME FILTERS")
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
    logger.info(f"Test dates: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

    # Calculate test period in days
    test_days = (df_test['timestamp'].max() - df_test['timestamp'].min()).days + 1
    logger.info(f"Test period: {test_days} days")

    # Load selected features
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        feature_config = json.load(f)
    optimal_features = feature_config['features'][:25]

    X_test_selected = X_test[optimal_features]

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

    baseline_threshold = 0.95
    baseline_signals = (y_proba >= baseline_threshold).astype(int)

    logger.info(f"Baseline signals: {baseline_signals.sum()} ({baseline_signals.sum()/test_days:.2f} trades/day)")

    # Test different filter combinations
    filter_configs = [
        {
            'name': 'Baseline',
            'prob_threshold': 0.95,
            'exclude_hours': None,
            'skip_thursday': False,
        },
        {
            'name': 'Prob ≥ 0.96',
            'prob_threshold': 0.96,
            'exclude_hours': None,
            'skip_thursday': False,
        },
        {
            'name': 'Prob ≥ 0.97',
            'prob_threshold': 0.97,
            'exclude_hours': None,
            'skip_thursday': False,
        },
        {
            'name': 'Prob 0.96 + No 1-5PM',
            'prob_threshold': 0.96,
            'exclude_hours': [13, 14, 15, 16, 17],
            'skip_thursday': False,
        },
        {
            'name': 'Prob 0.96 + Skip Thursday',
            'prob_threshold': 0.96,
            'exclude_hours': None,
            'skip_thursday': True,
        },
        {
            'name': 'Prob 0.96 + No 1-5PM + Skip Thursday',
            'prob_threshold': 0.96,
            'exclude_hours': [13, 14, 15, 16, 17],
            'skip_thursday': True,
        },
        {
            'name': 'Prob 0.97 + No 1-5PM + Skip Thursday',
            'prob_threshold': 0.97,
            'exclude_hours': [13, 14, 15, 16, 17],
            'skip_thursday': True,
        },
    ]

    results = {}

    logger.info("\n" + "=" * 80)
    logger.info("TESTING FILTER COMBINATIONS")
    logger.info("=" * 80)

    for config in filter_configs:
        logger.info(f"\nTesting: {config['name']}...")

        # Apply filters
        filtered_signals = apply_filters(
            df_test,
            baseline_signals,
            y_proba,
            prob_threshold=config['prob_threshold'],
            exclude_hours=config['exclude_hours'],
            skip_thursday=config['skip_thursday']
        )

        n_signals = filtered_signals.sum()
        trades_per_day = n_signals / test_days

        logger.info(f"  Signals: {n_signals} ({trades_per_day:.2f} trades/day)")

        # Simulate trades
        trades_df = simulate_trades(df_test, filtered_signals)
        metrics = calculate_metrics(trades_df)

        if metrics:
            logger.info(f"  Return: {metrics['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")

        results[config['name']] = {
            'config': config,
            'metrics': metrics,
            'n_signals': n_signals,
            'trades_per_day': trades_per_day,
        }

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("FILTER COMPARISON")
    logger.info("=" * 80)

    logger.info(f"\n{'Strategy':<40} {'Trades/Day':<12} {'Return':<10} {'Sharpe':<8} {'DD':<8} {'WR':<8}")
    logger.info("-" * 90)

    for name, result in results.items():
        m = result['metrics']
        if m:
            logger.info(
                f"{name:<40} {result['trades_per_day']:<12.2f} "
                f"{m['total_return_pct']:<10.2f} {m['sharpe_ratio']:<8.2f} "
                f"{m['max_drawdown_pct']:<8.2f} {m['win_rate']:<8.2%}"
            )
        else:
            logger.info(f"{name:<40} {result['trades_per_day']:<12.2f} NO TRADES")

    # Find best by Sharpe ratio
    valid_results = {k: v for k, v in results.items() if v['metrics']}
    if valid_results:
        best_by_sharpe = max(valid_results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        best_by_return = max(valid_results.items(), key=lambda x: x[1]['metrics']['total_return_pct'])

        logger.info("\n" + "=" * 80)
        logger.info("BEST STRATEGIES")
        logger.info("=" * 80)

        logger.info(f"\n🏆 Best Sharpe: {best_by_sharpe[0]}")
        logger.info(f"  Trades/Day: {best_by_sharpe[1]['trades_per_day']:.2f}")
        logger.info(f"  Sharpe: {best_by_sharpe[1]['metrics']['sharpe_ratio']:.2f}")
        logger.info(f"  Return: {best_by_sharpe[1]['metrics']['total_return_pct']:.2f}%")
        logger.info(f"  Win Rate: {best_by_sharpe[1]['metrics']['win_rate']:.2%}")
        logger.info(f"  Max DD: {best_by_sharpe[1]['metrics']['max_drawdown_pct']:.2f}%")

        logger.info(f"\n💰 Best Return: {best_by_return[0]}")
        logger.info(f"  Trades/Day: {best_by_return[1]['trades_per_day']:.2f}")
        logger.info(f"  Return: {best_by_return[1]['metrics']['total_return_pct']:.2f}%")
        logger.info(f"  Sharpe: {best_by_return[1]['metrics']['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate: {best_by_return[1]['metrics']['win_rate']:.2%}")

        # Compare to baseline
        baseline_metrics = results['Baseline']['metrics']
        best_metrics = best_by_sharpe[1]['metrics']

        logger.info("\n" + "=" * 80)
        logger.info("IMPROVEMENT VS BASELINE")
        logger.info("=" * 80)

        logger.info(f"\nBaseline: 46.51% return, 3.60 Sharpe, 9.23 trades/day")
        logger.info(f"{best_by_sharpe[0]}: {best_metrics['total_return_pct']:.2f}% return, "
                   f"{best_metrics['sharpe_ratio']:.2f} Sharpe, "
                   f"{best_by_sharpe[1]['trades_per_day']:.2f} trades/day")

        logger.info(f"\nImprovement:")
        logger.info(f"  Return: {(best_metrics['total_return_pct'] - 46.51):+.2f}% "
                   f"({(best_metrics['total_return_pct']/46.51 - 1)*100:+.1f}%)")
        logger.info(f"  Sharpe: {(best_metrics['sharpe_ratio'] - 3.60):+.2f} "
                   f"({(best_metrics['sharpe_ratio']/3.60 - 1)*100:+.1f}%)")
        logger.info(f"  Trades/Day: {(best_by_sharpe[1]['trades_per_day'] - 9.23):+.2f} "
                   f"({(best_by_sharpe[1]['trades_per_day']/9.23 - 1)*100:+.1f}%)")

    # Save results
    output = {
        name: {
            'config': result['config'],
            'metrics': result['metrics'],
            'trades_per_day': result['trades_per_day'],
        }
        for name, result in results.items()
    }

    output_path = Path('_bmad-output/reports/combined_filtering_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to: {output_path}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

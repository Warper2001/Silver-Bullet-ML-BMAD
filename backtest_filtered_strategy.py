#!/usr/bin/env python
"""
Comprehensive backtest of filtered strategy:
- Probability threshold: 0.96 (up from 0.95)
- Exclude hours: 13-17 (1 PM - 5 PM)
- Skip Thursday trading

This backtest validates the strategy on the full test period.
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime

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


def apply_filters(df: pd.DataFrame, signals: np.ndarray, probabilities: np.ndarray,
                 prob_threshold: float = 0.96,
                 exclude_hours: list = None,
                 skip_thursday: bool = True) -> np.ndarray:
    """
    Apply trading filters.

    Filters:
    1. Probability threshold (only take highest conviction)
    2. Time filter (exclude worst performing hours)
    3. Day filter (skip Thursday)
    """
    filtered_signals = signals.copy()
    exclude_hours = exclude_hours or [13, 14, 15, 16, 17]

    for i in range(len(df)):
        if signals[i] == 0:
            continue

        # Probability filter
        if probabilities[i] < prob_threshold:
            filtered_signals[i] = 0
            continue

        # Time filter
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
        entry_hour = entry_time.hour
        entry_day = entry_time.dayofweek

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
            'entry_index': i,
            'entry_time': entry_time,
            'entry_hour': entry_hour,
            'entry_day': entry_day,
            'entry_price': entry_price,
            'exit_index': exit_bar,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'bars_held': exit_bar - i,
        })

    return pd.DataFrame(trades)


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Calculate comprehensive performance metrics."""
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

    # Sortino ratio
    downside_std = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].std()
    sortino = (avg_pnl / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Calmar ratio
    calmar = (total_pnl / abs(max_drawdown)) if max_drawdown != 0 else 0

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
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
    }


def main():
    """Run comprehensive filtered strategy backtest."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE BACKTEST: FILTERED STRATEGY")
    logger.info("=" * 80)

    # Load data
    csv_path = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    logger.info(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

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

    logger.info(f"After dropping NaN: {len(df)} samples")

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

    # BASELINE: No filters
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE: NO FILTERS (Threshold 0.95)")
    logger.info("=" * 80)

    baseline_threshold = 0.95
    baseline_signals = (y_proba >= baseline_threshold).astype(int)

    logger.info(f"Baseline signals: {baseline_signals.sum()} ({baseline_signals.sum()/test_days:.2f} trades/day)")

    baseline_trades = simulate_trades(df_test, baseline_signals)
    baseline_metrics = calculate_metrics(baseline_trades)

    logger.info(f"\nBaseline Results:")
    logger.info(f"  Trades: {baseline_metrics['n_trades']}")
    logger.info(f"  Win Rate: {baseline_metrics['win_rate']:.2%}")
    logger.info(f"  Total Return: {baseline_metrics['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max DD: {baseline_metrics['max_drawdown_pct']:.2f}%")

    # FILTERED STRATEGY: Apply all filters
    logger.info("\n" + "=" * 80)
    logger.info("FILTERED STRATEGY: Prob ≥ 0.96 + No 1-5PM + Skip Thursday")
    logger.info("=" * 80)

    filtered_signals = apply_filters(
        df_test,
        baseline_signals,
        y_proba,
        prob_threshold=0.96,
        exclude_hours=[13, 14, 15, 16, 17],
        skip_thursday=True
    )

    logger.info(f"Filtered signals: {filtered_signals.sum()} ({filtered_signals.sum()/test_days:.2f} trades/day)")
    logger.info(f"Filter reduction: {(baseline_signals.sum() - filtered_signals.sum())/baseline_signals.sum()*100:.1f}%")

    filtered_trades = simulate_trades(df_test, filtered_signals)
    filtered_metrics = calculate_metrics(filtered_trades)

    logger.info(f"\nFiltered Strategy Results:")
    logger.info(f"  Trades: {filtered_metrics['n_trades']}")
    logger.info(f"  Win Rate: {filtered_metrics['win_rate']:.2%}")
    logger.info(f"  Total Return: {filtered_metrics['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe: {filtered_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Sortino: {filtered_metrics['sortino_ratio']:.2f}")
    logger.info(f"  Calmar: {filtered_metrics['calmar_ratio']:.2f}")
    logger.info(f"  Max DD: {filtered_metrics['max_drawdown_pct']:.2f}%")

    # COMPARISON
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: FILTERED VS BASELINE")
    logger.info("=" * 80)

    logger.info(f"\n{'Metric':<25} {'Baseline':<15} {'Filtered':<15} {'Change':<15}")
    logger.info("-" * 70)

    logger.info(f"{'Trades':<25} {baseline_metrics['n_trades']:<15} {filtered_metrics['n_trades']:<15} "
               f"{((filtered_metrics['n_trades']/baseline_metrics['n_trades']-1)*100):.1f}%")
    logger.info(f"{'Trades/Day':<25} {baseline_metrics['n_trades']/test_days:<15.2f} {filtered_metrics['n_trades']/test_days:<15.2f} "
               f"{((filtered_metrics['n_trades']/test_days)/(baseline_metrics['n_trades']/test_days)-1)*100:.1f}%")
    logger.info(f"{'Win Rate':<25} {baseline_metrics['win_rate']:<15.2%} {filtered_metrics['win_rate']:<15.2%} "
               f"{((filtered_metrics['win_rate']/baseline_metrics['win_rate']-1)*100):.1f}%")
    logger.info(f"{'Total Return':<25} {baseline_metrics['total_return_pct']:<15.2f}% {filtered_metrics['total_return_pct']:<15.2f}% "
               f"{filtered_metrics['total_return_pct']-baseline_metrics['total_return_pct']:+.2f}%")
    logger.info(f"{'Sharpe Ratio':<25} {baseline_metrics['sharpe_ratio']:<15.2f} {filtered_metrics['sharpe_ratio']:<15.2f} "
               f"{filtered_metrics['sharpe_ratio']-baseline_metrics['sharpe_ratio']:+.2f}")
    logger.info(f"{'Sortino Ratio':<25} {baseline_metrics['sortino_ratio']:<15.2f} {filtered_metrics['sortino_ratio']:<15.2f} "
               f"{filtered_metrics['sortino_ratio']-baseline_metrics['sortino_ratio']:+.2f}")
    logger.info(f"{'Max Drawdown':<25} {baseline_metrics['max_drawdown_pct']:<15.2f}% {filtered_metrics['max_drawdown_pct']:<15.2f}% "
               f"{filtered_metrics['max_drawdown_pct']-baseline_metrics['max_drawdown_pct']:+.2f}%")

    # Exit reason breakdown
    logger.info("\n" + "=" * 80)
    logger.info("EXIT REASON BREAKDOWN")
    logger.info("=" * 80)

    baseline_exit_counts = baseline_trades['exit_reason'].value_counts()
    filtered_exit_counts = filtered_trades['exit_reason'].value_counts()

    logger.info(f"\n{'Exit Reason':<15} {'Baseline':<15} {'Filtered':<15}")
    logger.info("-" * 45)

    for reason in ['TAKE_PROFIT', 'STOP_LOSS', 'MAX_BARS']:
        baseline_count = baseline_exit_counts.get(reason, 0)
        filtered_count = filtered_exit_counts.get(reason, 0)
        logger.info(f"{reason:<15} {baseline_count:<15} {filtered_count:<15}")

    # Time of day analysis for filtered trades
    logger.info("\n" + "=" * 80)
    logger.info("FILTERED TRADES: TIME OF DAY BREAKDOWN")
    logger.info("=" * 80)

    hour_stats = filtered_trades.groupby('entry_hour').agg({
        'pnl_pct': ['sum', 'count', 'mean'],
    })
    hour_stats.columns = ['total_pnl', 'n_trades', 'avg_pnl']

    logger.info(f"\n{'Hour':<6} {'Trades':<10} {'Total P&L':<12} {'Avg P&L':<10}")
    logger.info("-" * 45)

    for hour, row in hour_stats.iterrows():
        logger.info(f"{int(hour):<6} {int(row['n_trades']):<10} {row['total_pnl']:<12.2f} {row['avg_pnl']:<10.4f}")

    # Day of week analysis for filtered trades
    logger.info("\n" + "=" * 80)
    logger.info("FILTERED TRADES: DAY OF WEEK BREAKDOWN")
    logger.info("=" * 80)

    dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    dow_stats = filtered_trades.groupby('entry_day').agg({
        'pnl_pct': ['sum', 'count', 'mean'],
    })
    dow_stats.columns = ['total_pnl', 'n_trades', 'avg_pnl']

    logger.info(f"\n{'Day':<12} {'Trades':<10} {'Total P&L':<12} {'Avg P&L':<10}")
    logger.info("-" * 50)

    for day, row in dow_stats.iterrows():
        logger.info(f"{dow_names[day]:<12} {int(row['n_trades']):<10} {row['total_pnl']:<12.2f} {row['avg_pnl']:<10.4f}")

    # Save detailed results
    results = {
        'backtest_date': str(datetime.now()),
        'strategy': {
            'name': 'Filtered Strategy',
            'probability_threshold': 0.96,
            'exclude_hours': [13, 14, 15, 16, 17],
            'skip_thursday': True,
        },
        'test_period': {
            'start': str(df_test['timestamp'].min()),
            'end': str(df_test['timestamp'].max()),
            'days': test_days,
        },
        'baseline': {
            'trades': int(baseline_metrics['n_trades']),
            'trades_per_day': baseline_metrics['n_trades'] / test_days,
            'win_rate': baseline_metrics['win_rate'],
            'total_return_pct': baseline_metrics['total_return_pct'],
            'sharpe_ratio': baseline_metrics['sharpe_ratio'],
            'max_drawdown_pct': baseline_metrics['max_drawdown_pct'],
        },
        'filtered': {
            'trades': int(filtered_metrics['n_trades']),
            'trades_per_day': filtered_metrics['n_trades'] / test_days,
            'win_rate': filtered_metrics['win_rate'],
            'total_return_pct': filtered_metrics['total_return_pct'],
            'sharpe_ratio': filtered_metrics['sharpe_ratio'],
            'sortino_ratio': filtered_metrics['sortino_ratio'],
            'calmar_ratio': filtered_metrics['calmar_ratio'],
            'max_drawdown_pct': filtered_metrics['max_drawdown_pct'],
        },
        'improvement': {
            'sharpe_improvement_pct': (filtered_metrics['sharpe_ratio'] / baseline_metrics['sharpe_ratio'] - 1) * 100,
            'drawdown_reduction_pct': (baseline_metrics['max_drawdown_pct'] - filtered_metrics['max_drawdown_pct']) / abs(baseline_metrics['max_drawdown_pct']) * 100,
            'winrate_improvement_pct': (filtered_metrics['win_rate'] / baseline_metrics['win_rate'] - 1) * 100,
        },
    }

    output_path = Path('_bmad-output/reports/filtered_strategy_backtest.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Full results saved to: {output_path}")

    # Save trade details
    trades_path = Path('_bmad-output/reports/filtered_strategy_trades.csv')
    filtered_trades.to_csv(trades_path, index=False)
    logger.info(f"✅ Trade details saved to: {trades_path}")

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

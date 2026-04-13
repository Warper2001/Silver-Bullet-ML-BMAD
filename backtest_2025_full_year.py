#!/usr/bin/env python
"""
Full year 2025 walk-forward backtest of filtered strategy.

Uses walk-forward validation:
- Train on expanding window
- Test on subsequent period
- Roll forward through 2025

This validates the strategy works across different market conditions.
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
    """Apply trading filters."""
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
        entry_month = entry_time.month

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
            'entry_month': entry_month,
            'entry_hour': entry_hour,
            'entry_day': entry_day,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
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

    # Sortino ratio
    downside_std = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].std()
    sortino = (avg_pnl / downside_std) * np.sqrt(252) if downside_std > 0 else 0

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
    }


def main():
    """Run full year 2025 walk-forward backtest."""
    logger.info("=" * 80)
    logger.info("FULL YEAR 2025 WALK-FORWARD BACKTEST")
    logger.info("=" * 80)

    # Load data
    csv_path = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    logger.info(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filter to 2025 only
    df = df[df['timestamp'].dt.year == 2025].copy()
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

    # Load selected features
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        feature_config = json.load(f)
    optimal_features = feature_config['features'][:25]

    features_df = features_df[optimal_features]

    # Walk-forward parameters
    # Train on 3 months, test on 1 month, roll forward
    walk_forward_months = [
        # Train period, Test period
        ('2025-01', '2025-02'),  # Train Jan, test Feb
        ('2025-02', '2025-03'),  # Train Feb, test Mar
        ('2025-03', '2025-04'),  # Train Mar, test Apr
        ('2025-04', '2025-05'),  # Train Apr, test May
        ('2025-05', '2025-06'),  # Train May, test Jun
        ('2025-06', '2025-07'),  # Train Jun, test Jul
        ('2025-07', '2025-08'),  # Train Jul, test Aug
        ('2025-08', '2025-09'),  # Train Aug, test Sep
        ('2025-09', '2025-10'),  # Train Sep, test Oct
        ('2025-10', '2025-11'),  # Train Oct, test Nov
        ('2025-11', '2025-12'),  # Train Nov, test Dec
    ]

    all_baseline_trades = []
    all_filtered_trades = []
    monthly_results = []

    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 80)

    for train_month, test_month in walk_forward_months:
        logger.info(f"\n--- Training: {train_month}, Testing: {test_month} ---")

        # Split data
        train_mask = df['timestamp'].dt.strftime('%Y-%m') == train_month
        test_mask = df['timestamp'].dt.strftime('%Y-%m') == test_month

        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)
        X_train = features_df[train_mask].reset_index(drop=True)
        X_test = features_df[test_mask].reset_index(drop=True)

        if len(df_train) == 0 or len(df_test) == 0:
            logger.warning(f"Skipping {train_month}/{test_month} - insufficient data")
            continue

        logger.info(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

        # Create labels
        y_train = create_silver_bullet_labels(df_train)

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=3.0, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Generate signals
        y_proba = model.predict_proba(X_test)[:, 1]

        # Baseline: Threshold 0.95, no filters
        baseline_signals = (y_proba >= 0.95).astype(int)
        baseline_trades = simulate_trades(df_test, baseline_signals)
        baseline_metrics = calculate_metrics(baseline_trades)

        # Filtered: Threshold 0.96 + time filters
        filtered_signals = apply_filters(
            df_test, baseline_signals, y_proba,
            prob_threshold=0.96,
            exclude_hours=[13, 14, 15, 16, 17],
            skip_thursday=True
        )
        filtered_trades = simulate_trades(df_test, filtered_signals)
        filtered_metrics = calculate_metrics(filtered_trades)

        # Store results
        all_baseline_trades.append(baseline_trades)
        all_filtered_trades.append(filtered_trades)

        monthly_results.append({
            'train_month': train_month,
            'test_month': test_month,
            'baseline': {
                'trades': len(baseline_trades),
                'return_pct': baseline_metrics.get('total_return_pct', 0),
                'sharpe': baseline_metrics.get('sharpe_ratio', 0),
                'win_rate': baseline_metrics.get('win_rate', 0),
            },
            'filtered': {
                'trades': len(filtered_trades),
                'return_pct': filtered_metrics.get('total_return_pct', 0),
                'sharpe': filtered_metrics.get('sharpe_ratio', 0),
                'win_rate': filtered_metrics.get('win_rate', 0),
            },
        })

        logger.info(f"Baseline: {len(baseline_trades)} trades, {baseline_metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Filtered: {len(filtered_trades)} trades, {filtered_metrics.get('total_return_pct', 0):.2f}%")

    # Combine all trades
    logger.info("\n" + "=" * 80)
    logger.info("FULL YEAR 2025 RESULTS")
    logger.info("=" * 80)

    all_baseline_df = pd.concat(all_baseline_trades, ignore_index=True)
    all_filtered_df = pd.concat(all_filtered_trades, ignore_index=True)

    baseline_metrics = calculate_metrics(all_baseline_df)
    filtered_metrics = calculate_metrics(all_filtered_df)

    # Calculate test period in days
    test_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1

    logger.info(f"\nTest Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()} ({test_days} days)")

    logger.info(f"\n{'Metric':<25} {'Baseline':<15} {'Filtered':<15} {'Change':<15}")
    logger.info("-" * 70)

    logger.info(f"{'Total Trades':<25} {baseline_metrics['n_trades']:<15} {filtered_metrics['n_trades']:<15} "
               f"{((filtered_metrics['n_trades']/baseline_metrics['n_trades']-1)*100):.1f}%")
    logger.info(f"{'Trades/Day':<25} {baseline_metrics['n_trades']/test_days:<15.2f} {filtered_metrics['n_trades']/test_days:<15.2f} "
               f"{((filtered_metrics['n_trades']/test_days)/(baseline_metrics['n_trades']/test_days)-1)*100:.1f}%")
    logger.info(f"{'Win Rate':<25} {baseline_metrics['win_rate']:<15.2%} {filtered_metrics['win_rate']:<15.2%} "
               f"{((filtered_metrics['win_rate']/baseline_metrics['win_rate']-1)*100):.1f}%")
    logger.info(f"{'Total Return':<25} {baseline_metrics['total_return_pct']:<15.2f}% {filtered_metrics['total_return_pct']:<15.2f}% "
               f"{filtered_metrics['total_return_pct']-baseline_metrics['total_return_pct']:+.2f}%")
    logger.info(f"{'Sharpe Ratio':<25} {baseline_metrics['sharpe_ratio']:<15.2f} {filtered_metrics['sharpe_ratio']:<15.2f} "
               f"{filtered_metrics['sharpe_ratio']-baseline_metrics['sharpe_ratio']:+.2f}")
    logger.info(f"{'Max Drawdown':<25} {baseline_metrics['max_drawdown_pct']:<15.2f}% {filtered_metrics['max_drawdown_pct']:<15.2f}% "
               f"{filtered_metrics['max_drawdown_pct']-baseline_metrics['max_drawdown_pct']:+.2f}%")

    # Monthly breakdown
    logger.info("\n" + "=" * 80)
    logger.info("MONTHLY BREAKDOWN")
    logger.info("=" * 80)

    logger.info(f"\n{'Month':<10} {'Baseline Trades':<15} {'Baseline %':<12} {'Filtered Trades':<15} {'Filtered %':<12}")
    logger.info("-" * 70)

    for result in monthly_results:
        logger.info(f"{result['test_month']:<10} "
                   f"{result['baseline']['trades']:<15} "
                   f"{result['baseline']['return_pct']:<12.2f} "
                   f"{result['filtered']['trades']:<15} "
                   f"{result['filtered']['return_pct']:<12.2f}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n✅ Full Year 2025 Backtest Complete")
    logger.info(f"   Test Period: {test_days} days")
    logger.info(f"   Walk-Forward: 11 monthly test periods")
    logger.info(f"\n📊 Filtered Strategy Performance:")
    logger.info(f"   Total Return: {filtered_metrics['total_return_pct']:.2f}%")
    logger.info(f"   Sharpe Ratio: {filtered_metrics['sharpe_ratio']:.2f}")
    logger.info(f"   Win Rate: {filtered_metrics['win_rate']:.2%}")
    logger.info(f"   Max Drawdown: {filtered_metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"   Trades: {filtered_metrics['n_trades']} ({filtered_metrics['n_trades']/test_days:.2f}/day)")

    # Save results
    results = {
        'backtest_date': str(datetime.now()),
        'strategy': 'filtered_strategy',
        'test_period': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max()),
            'days': test_days,
        },
        'methodology': 'walk_forward_validation',
        'walk_forward_config': 'train_1_month_test_1_month',
        'baseline': baseline_metrics,
        'filtered': filtered_metrics,
        'monthly_results': monthly_results,
    }

    output_path = Path('_bmad-output/reports/2025_full_year_backtest.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Full results saved to: {output_path}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

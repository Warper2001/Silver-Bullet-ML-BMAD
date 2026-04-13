#!/usr/bin/env python
"""
Full backtest with optimized threshold (0.95).

Simulates actual trading with:
- Entry signals from ML model
- Stop loss and take profit exits
- P&L calculation for each trade
- Comprehensive performance statistics
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


def simulate_trades(df: pd.DataFrame, signals: np.ndarray,
                    take_profit_pct: float = 0.5,
                    stop_loss_pct: float = 0.2,
                    max_bars: int = 60) -> pd.DataFrame:
    """
    Simulate actual trades with entry, SL, TP exits.

    Returns DataFrame with trade details.
    """
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
        })

    return pd.DataFrame(trades)


def main():
    """Run full backtest."""
    logger.info("=" * 80)
    logger.info("FULL BACKTEST WITH OPTIMIZED THRESHOLD")
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

    # Test set only
    df_test = df.iloc[train_end:].reset_index(drop=True)
    X_test = features_df.iloc[train_end:].reset_index(drop=True)

    logger.info(f"Test period: {len(df_test)} bars")
    logger.info(f"Test dates: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

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
    logger.info("Generating trading signals...")
    y_proba = model.predict_proba(X_test_selected)[:, 1]
    signals = (y_proba >= THRESHOLD).astype(int)

    n_signals = signals.sum()
    logger.info(f"Total signals generated: {n_signals}")

    # Calculate test period in days
    test_days = (df_test['timestamp'].max() - df_test['timestamp'].min()).days
    logger.info(f"Test period: {test_days} days")
    logger.info(f"Signals per day: {n_signals / test_days:.2f}")

    # Simulate trades
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATING TRADES")
    logger.info("=" * 80)

    trades_df = simulate_trades(df_test, signals)

    logger.info(f"Total trades simulated: {len(trades_df)}")

    # Calculate statistics
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    n_wins = len(winning_trades)
    n_losses = len(losing_trades)
    win_rate = n_wins / len(trades_df) if len(trades_df) > 0 else 0

    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    std_pnl = trades_df['pnl_pct'].std()

    avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

    profit_factor = abs(winning_trades['pnl_pct'].sum() / losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_pct'].sum() != 0 else 0

    # Max drawdown
    cum_pnl = trades_df['pnl_pct'].cumsum()
    running_max = cum_pnl.expanding().max()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Exit reason breakdown
    exit_counts = trades_df['exit_reason'].value_counts()

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)

    logger.info("\n📊 Trade Statistics:")
    logger.info(f"  Total Trades: {len(trades_df)}")
    logger.info(f"  Winning Trades: {n_wins} ({win_rate:.2%})")
    logger.info(f"  Losing Trades: {n_losses}")
    logger.info(f"  Win Rate: {win_rate:.2%}")

    logger.info("\n💰 P&L Statistics:")
    logger.info(f"  Total Return: {total_pnl:.2f}%")
    logger.info(f"  Average P&L: {avg_pnl:.4f}%")
    logger.info(f"  Std Dev P&L: {std_pnl:.4f}%")
    logger.info(f"  Average Win: {avg_win:.4f}%")
    logger.info(f"  Average Loss: {avg_loss:.4f}%")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")

    logger.info("\n📉 Risk Metrics:")
    logger.info(f"  Max Drawdown: {max_drawdown:.2f}%")

    logger.info("\n🎯 Exit Reasons:")
    for reason, count in exit_counts.items():
        pct = count / len(trades_df) * 100
        logger.info(f"  {reason}: {count} ({pct:.1f}%)")

    # Sharpe ratio (simplified, assuming 252 trading days per year)
    if len(trades_df) > 1:
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
        logger.info(f"\n📈 Sharpe Ratio (annualized): {sharpe:.2f}")

    # Trade duration
    avg_bars_held = trades_df['bars_held'].mean()
    logger.info(f"\n⏱️  Average Trade Duration: {avg_bars_held:.1f} bars")

    # Save detailed results
    results = {
        'test_period': {
            'start': str(df_test['timestamp'].min()),
            'end': str(df_test['timestamp'].max()),
            'days': test_days,
        },
        'trading_stats': {
            'total_trades': len(trades_df),
            'winning_trades': n_wins,
            'losing_trades': n_losses,
            'win_rate': win_rate,
            'trades_per_day': len(trades_df) / test_days,
        },
        'pnl_stats': {
            'total_return_pct': total_pnl,
            'avg_pnl_pct': avg_pnl,
            'std_pnl_pct': std_pnl,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
        },
        'risk_metrics': {
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe if len(trades_df) > 1 else 0,
        },
        'exit_reasons': exit_counts.to_dict(),
    }

    output_path = Path('_bmad-output/reports/full_backtest_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_path}")

    # Save trade details
    trades_path = Path('_bmad-output/reports/trade_details.csv')
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"✅ Trade details saved to: {trades_path}")

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

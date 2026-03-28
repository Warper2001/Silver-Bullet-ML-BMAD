#!/usr/bin/env python3
"""Run ML-enhanced backtest with trained XGBoost model."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.historical_data_loader import HistoricalDataLoader
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator

def calculate_technical_indicators(df):
    """Calculate technical indicators."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['momentum'] = df['close'] - df['close'].shift(10)
    df['momentum_pct'] = df['momentum'] / df['close']

    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    df['roc'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100

    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility_pct'] = df['volatility'] / df['close']

    return df

def generate_signals_with_ml(df, model, feature_cols, threshold=0.55):
    """Generate signals filtered by ML model."""
    signals = []

    for i in range(100, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]

        # Determine signal type
        if current_bar['rsi'] < 30:
            signal_type = 'RSI_OVERSOLD'
            direction = 'bullish'
        elif current_bar['rsi'] > 70:
            signal_type = 'RSI_OVERBOUGHT'
            direction = 'bearish'
        elif current_bar['momentum_pct'] > 0 and prev_bar['momentum_pct'] <= 0:
            signal_type = 'MOMENTUM_UP'
            direction = 'bullish'
        elif current_bar['momentum_pct'] < 0 and prev_bar['momentum_pct'] >= 0:
            signal_type = 'MOMENTUM_DOWN'
            direction = 'bearish'
        elif current_bar['close'] > current_bar['bb_upper']:
            signal_type = 'BB_BREAKOUT_UP'
            direction = 'bullish'
        elif current_bar['close'] < current_bar['bb_lower']:
            signal_type = 'BB_BREAKOUT_DOWN'
            direction = 'bearish'
        else:
            continue

        # Prepare features for ML
        features = {
            'rsi': current_bar['rsi'],
            'rsi_change': current_bar['rsi'] - prev_bar['rsi'],
            'sma_20_above_sma_50': 1 if current_bar['sma_20'] > current_bar['sma_50'] else 0,
            'price_vs_sma20': (current_bar['close'] - current_bar['sma_20']) / current_bar['sma_20'],
            'bb_position': current_bar['bb_position'],
            'bb_width': current_bar['bb_width'],
            'atr_pct': current_bar['atr_pct'],
            'momentum_pct': current_bar['momentum_pct'],
            'volume_ratio': current_bar['volume_ratio'],
            'roc': current_bar['roc'],
            'volatility_pct': current_bar['volatility_pct'],
        }

        # One-hot encode signal_type
        for sig_type in ['RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'MOMENTUM_UP', 'MOMENTUM_DOWN',
                        'BB_BREAKOUT_UP', 'BB_BREAKOUT_DOWN']:
            features[f'signal_{sig_type}'] = 1 if sig_type == signal_type else 0

        # Create feature DataFrame
        X = pd.DataFrame([features])[feature_cols].fillna(0)

        # Get ML prediction
        proba = model.predict_proba(X)[0, 1]  # Probability of positive return

        # Filter by threshold
        if proba >= threshold:
            signals.append({
                'timestamp': current_bar.name,
                'signal_type': signal_type,
                'direction': direction,
                'entry_price': current_bar['close'],
                'ml_probability': proba,
                'stop_loss': current_bar['close'] - (current_bar['atr'] * 1.5) if direction == 'bullish' else current_bar['close'] + (current_bar['atr'] * 1.5),
                'take_profit': current_bar['close'] + (current_bar['atr'] * 2.5) if direction == 'bullish' else current_bar['close'] - (current_bar['atr'] * 2.5),
            })

    return pd.DataFrame(signals)

def simulate_trades(df, signals):
    """Simulate trades with stop loss and take profit."""
    trades = []

    for idx, signal in signals.iterrows():
        entry_time = signal['timestamp']
        entry_price = signal['entry_price']
        direction = signal['direction']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']

        exit_price = None
        exit_time = None
        exit_reason = None

        future_data = df.loc[entry_time:]

        for future_idx, future_bar in future_data.iterrows():
            if future_idx == entry_time:
                continue

            if direction == 'bullish':
                if future_bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = future_idx
                    exit_reason = 'stop_loss'
                    break
                if future_bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_time = future_idx
                    exit_reason = 'take_profit'
                    break
            else:
                if future_bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_time = future_idx
                    exit_reason = 'stop_loss'
                    break
                if future_bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_time = future_idx
                    exit_reason = 'take_profit'
                    break

        if exit_price is None and len(future_data) > 1:
            exit_price = future_data.iloc[-1]['close']
            exit_time = future_data.index[-1]
            exit_reason = 'end_of_data'

        if exit_price is not None:
            if direction == 'bullish':
                pnl = (exit_price - entry_price)
            else:
                pnl = (entry_price - exit_price)

            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'signal_type': signal['signal_type'],
                'ml_probability': signal['ml_probability'],
                'exit_reason': exit_reason,
                'pnl': pnl,
                'return_pct': (pnl / entry_price) * 100
            })

    return pd.DataFrame(trades)

def main():
    """Run ML-enhanced backtest."""

    print("🤖 ML-ENHANCED SILVER BULLET BACKTEST")
    print("=" * 60)

    # Load trained model
    print("\n📂 Loading trained ML model...")
    try:
        model = joblib.load('data/models/xgboost_mnq_classifier.pkl')
        feature_cols = joblib.load('data/models/feature_columns.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Please run train_ml_model.py first")
        return

    # Load data (use out-of-sample period)
    print("\n📊 Loading backtest data...")
    loader = HistoricalDataLoader(
        data_directory="data/processed/dollar_bars/",
        min_completeness=0.1
    )

    # Use data AFTER training period for true out-of-sample test
    data = loader.load_data('2025-02-20', '2025-03-05')
    print(f"✅ Loaded {len(data)} bars (out-of-sample)")

    # Calculate indicators
    print("\n🔬 Calculating technical indicators...")
    data = calculate_technical_indicators(data)

    # Test different ML thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    results = []

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"🎯 Testing ML threshold: {threshold}")
        print(f"{'='*60}")

        # Generate signals with ML filtering
        signals = generate_signals_with_ml(data, model, feature_cols, threshold)
        print(f"✅ Generated {len(signals)} signals (ML filtered)")

        if len(signals) == 0:
            print("   No signals passed ML filter")
            continue

        # Simulate trades
        trades = simulate_trades(data, signals)
        print(f"✅ Simulated {len(trades)} trades")

        if len(trades) == 0:
            print("   No trades completed")
            continue

        # Calculate metrics
        metrics_calc = PerformanceMetricsCalculator()
        metrics = metrics_calc.calculate_all_metrics(trades)

        total_return = metrics.get('total_return', {})
        if isinstance(total_return, dict):
            total_return_val = total_return.get('total_return_pct', 0)
        else:
            total_return_val = total_return

        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)

        max_dd = metrics.get('max_drawdown', {})
        if isinstance(max_dd, dict):
            max_dd_val = max_dd.get('max_drawdown_pct', 0)
        else:
            max_dd_val = max_dd

        print(f"\n   Performance at threshold {threshold}:")
        print(f"   Total Return: {total_return_val:.2f}%")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Max Drawdown: {max_dd_val:.2f}%")
        print(f"   Avg ML Probability: {trades['ml_probability'].mean():.3f}")

        results.append({
            'threshold': threshold,
            'signals': len(signals),
            'trades': len(trades),
            'return': total_return_val,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_dd': max_dd_val
        })

    # Summary comparison
    if results:
        print(f"\n{'='*60}")
        print("📊 THRESHOLD COMPARISON")
        print(f"{'='*60}")

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        # Best threshold
        best_idx = results_df['return'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_return = results_df.loc[best_idx, 'return']

        print(f"\n🏆 Best threshold: {best_threshold} (Return: {best_return:.2f}%)")

        # Run detailed analysis with best threshold
        print(f"\n{'='*60}")
        print(f"📈 DETAILED ANALYSIS (threshold={best_threshold})")
        print(f"{'='*60}")

        signals = generate_signals_with_ml(data, model, feature_cols, best_threshold)
        trades = simulate_trades(data, signals)

        if len(trades) > 0:
            print(f"\n   Trade Analysis:")
            print(f"   Avg Return: {trades['return_pct'].mean():.2f}%")
            print(f"   Best Trade: +{trades['return_pct'].max():.2f}%")
            print(f"   Worst Trade: {trades['return_pct'].min():.2f}%")

            print(f"\n   Exit Reason Breakdown:")
            for reason, count in trades['exit_reason'].value_counts().items():
                pct = (count / len(trades)) * 100
                avg_return = trades[trades['exit_reason'] == reason]['return_pct'].mean()
                print(f"   - {reason}: {count} ({pct:.1f}%) | Avg: {avg_return:.2f}%")

            print(f"\n   Strategy Performance:")
            for strategy in trades['signal_type'].unique():
                strategy_trades = trades[trades['signal_type'] == strategy]
                if len(strategy_trades) > 0:
                    win_rate = (strategy_trades['return_pct'] > 0).sum() / len(strategy_trades) * 100
                    avg_return = strategy_trades['return_pct'].mean()
                    total_return = strategy_trades['return_pct'].sum()
                    print(f"   - {strategy}:")
                    print(f"     Trades: {len(strategy_trades)} | Win Rate: {win_rate:.1f}% | Avg: {avg_return:.2f}% | Total: {total_return:.2f}%")

    print("\n✅ ML Backtest complete!")

if __name__ == '__main__':
    main()

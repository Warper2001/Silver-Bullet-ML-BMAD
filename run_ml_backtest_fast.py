#!/usr/bin/env python3
"""Quick ML backtest - single threshold."""

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

def calculate_indicators(df):
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
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']
    df['momentum_pct'] = (df['close'] - df['close'].shift(10)) / df['close']
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['roc'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility_pct'] = df['volatility'] / df['close']
    return df

def main():
    print("🤖 ML BACKTEST (Single Threshold)")
    print("=" * 50)

    # Load model
    print("\n📂 Loading ML model...")
    model = joblib.load('data/models/xgboost_mnq_classifier.pkl')
    feature_cols = joblib.load('data/models/feature_columns.pkl')
    print("✅ Model loaded")

    # Load data
    print("\n📊 Loading data...")
    loader = HistoricalDataLoader(data_directory="data/processed/dollar_bars/", min_completeness=0.1)
    data = loader.load_data('2025-02-20', '2025-03-05')
    data = calculate_indicators(data)
    print(f"✅ Ready with {len(data)} bars")

    # Generate signals
    print("\n🎯 Generating ML-filtered signals...")
    threshold = 0.55
    signals = []

    for i in range(100, len(data)):
        curr = data.iloc[i]
        prev = data.iloc[i-1]

        # Signal type (all types that model was trained on)
        if curr['rsi'] < 30:
            sig_type, direction = 'RSI_OVERSOLD', 'bullish'
        elif curr['rsi'] > 70:
            sig_type, direction = 'RSI_OVERBOUGHT', 'bearish'
        elif curr['momentum_pct'] > 0 and prev['momentum_pct'] <= 0:
            sig_type, direction = 'MOMENTUM_UP', 'bullish'
        elif curr['momentum_pct'] < 0 and prev['momentum_pct'] >= 0:
            sig_type, direction = 'MOMENTUM_DOWN', 'bearish'
        elif curr['close'] > curr['bb_upper']:
            sig_type, direction = 'BB_BREAKOUT_UP', 'bullish'
        elif curr['close'] < curr['bb_lower']:
            sig_type, direction = 'BB_BREAKOUT_DOWN', 'bearish'
        else:
            continue

        # Features
        features = {
            'rsi': curr['rsi'],
            'rsi_change': curr['rsi'] - prev['rsi'],
            'sma_20_above_sma_50': 1 if curr['sma_20'] > curr['sma_50'] else 0,
            'price_vs_sma20': (curr['close'] - curr['sma_20']) / curr['sma_20'],
            'bb_position': curr['bb_position'],
            'bb_width': curr['bb_width'],
            'atr_pct': curr['atr_pct'],
            'momentum_pct': curr['momentum_pct'],
            'volume_ratio': curr['volume_ratio'],
            'roc': curr['roc'],
            'volatility_pct': curr['volatility_pct'],
        }

        # One-hot encode all signal types
        for st in ['RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'MOMENTUM_UP', 'MOMENTUM_DOWN',
                   'BB_BREAKOUT_UP', 'BB_BREAKOUT_DOWN']:
            features[f'signal_{st}'] = 1 if st == sig_type else 0

        X = pd.DataFrame([features])[feature_cols].fillna(0)
        proba = model.predict_proba(X)[0, 1]

        if proba >= threshold:
            signals.append({
                'timestamp': curr.name,
                'signal_type': sig_type,
                'direction': direction,
                'entry_price': curr['close'],
                'ml_prob': proba,
                'sl': curr['close'] - (curr['atr'] * 1.5) if direction == 'bullish' else curr['close'] + (curr['atr'] * 1.5),
                'tp': curr['close'] + (curr['atr'] * 2.5) if direction == 'bullish' else curr['close'] - (curr['atr'] * 2.5),
            })

    signals_df = pd.DataFrame(signals)
    print(f"✅ {len(signals_df)} signals generated (ML threshold={threshold})")

    # Simulate trades
    print("\n⚡ Simulating trades...")
    trades = []

    for _, sig in signals_df.iterrows():
        future = data.loc[sig['timestamp']:]

        for idx, bar in future.iterrows():
            if idx == sig['timestamp']:
                continue

            exit_price = None
            exit_reason = None

            if sig['direction'] == 'bullish':
                if bar['low'] <= sig['sl']:
                    exit_price = sig['sl']
                    exit_reason = 'stop_loss'
                    break
                if bar['high'] >= sig['tp']:
                    exit_price = sig['tp']
                    exit_reason = 'take_profit'
                    break
            else:
                if bar['high'] >= sig['sl']:
                    exit_price = sig['sl']
                    exit_reason = 'stop_loss'
                    break
                if bar['low'] <= sig['tp']:
                    exit_price = sig['tp']
                    exit_reason = 'take_profit'
                    break

        if exit_price is None and len(future) > 1:
            exit_price = future.iloc[-1]['close']
            exit_reason = 'end_of_data'

        if exit_price is not None:
            pnl = (exit_price - sig['entry_price']) if sig['direction'] == 'bullish' else (sig['entry_price'] - exit_price)
            trades.append({
                'entry_time': sig['timestamp'],
                'direction': sig['direction'],
                'signal_type': sig['signal_type'],
                'ml_prob': sig['ml_prob'],
                'exit_reason': exit_reason,
                'return_pct': (pnl / sig['entry_price']) * 100
            })

    trades_df = pd.DataFrame(trades)
    print(f"✅ {len(trades_df)} trades completed")

    if len(trades_df) > 0:
        # Metrics
        total_return = trades_df['return_pct'].sum()
        win_rate = (trades_df['return_pct'] > 0).sum() / len(trades_df) * 100
        avg_return = trades_df['return_pct'].mean()

        print(f"\n📈 RESULTS (threshold={threshold}):")
        print(f"   Signals: {len(signals_df)}")
        print(f"   Trades: {len(trades_df)}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Avg Return: {avg_return:.2f}%")
        print(f"   Best Trade: +{trades_df['return_pct'].max():.2f}%")
        print(f"   Worst Trade: {trades_df['return_pct'].min():.2f}%")
        print(f"   Avg ML Prob: {trades_df['ml_prob'].mean():.3f}")

        print(f"\n   By Exit Reason:")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            avg_ret = trades_df[trades_df['exit_reason'] == reason]['return_pct'].mean()
            print(f"   - {reason}: {count} | Avg: {avg_ret:.2f}%")

    print("\n✅ Backtest complete!")

if __name__ == '__main__':
    main()

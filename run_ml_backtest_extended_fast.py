#!/usr/bin/env python3
"""Fast extended ML backtest."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.historical_data_loader import HistoricalDataLoader

def main():
    print("📊 EXTENDED ML BACKTEST")
    print("=" * 60)

    # Load model
    print("\n📂 Loading ML model...")
    model = joblib.load('data/models/xgboost_mnq_classifier.pkl')
    feature_cols = joblib.load('data/models/feature_columns.pkl')
    print("✅ Model loaded")

    # Extended data
    print("\n📊 Loading 6-month dataset...")
    loader = HistoricalDataLoader(data_directory="data/processed/dollar_bars/", min_completeness=0.1)
    data = loader.load_data('2024-10-01', '2025-03-05')
    print(f"✅ Loaded {len(data):,} bars")
    print(f"   Period: {data.index.min().date()} to {data.index.max().date()}")

    # Quick indicators
    print("\n🔬 Calculating indicators...")
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain/loss))
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['bb_middle'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + data['bb_std'] * 2
    data['bb_lower'] = data['bb_middle'] - data['bb_std'] * 2
    data['atr'] = pd.concat([
        data['high'] - data['low'],
        (data['high'] - data['close'].shift()).abs(),
        (data['low'] - data['close'].shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    data['momentum_pct'] = (data['close'] - data['close'].shift(10)) / data['close']
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volatility'] = data['close'].rolling(20).std()
    data['roc'] = ((data['close'] - data['close'].shift(5)) / data['close'].shift(5)) * 100
    data['rsi_prev'] = data['rsi'].shift(1)
    print("✅ Indicators ready")

    # Generate signals with ML filtering (threshold 0.55)
    print(f"\n🎯 Generating ML-filtered signals...")
    signals = []

    total_bars = len(data)
    for i in range(100, total_bars):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{total_bars} ({i/total_bars*100:.1f}%)")
        curr = data.iloc[i]
        prev = data.iloc[i-1]

        # Signal type
        if curr['rsi'] < 30:
            sig_type, direction = 'RSI_OVERSOLD', 'bullish'
        elif curr['rsi'] > 70:
            sig_type, direction = 'RSI_OVERBOUGHT', 'bearish'
        elif curr['momentum_pct'] > 0 and prev['momentum_pct'] <= 0:
            sig_type, direction = 'MOMENTUM_UP', 'bullish'
        elif curr['momentum_pct'] < 0 and prev['momentum_pct'] >= 0:
            sig_type, direction = 'MOMENTUM_DOWN', 'bearish'
        else:
            continue

        # Features
        features = {
            'rsi': curr['rsi'],
            'rsi_change': curr['rsi'] - curr['rsi_prev'],
            'sma_20_above_sma_50': 1 if curr['sma_20'] > curr['sma_50'] else 0,
            'price_vs_sma20': (curr['close'] - curr['sma_20']) / curr['sma_20'],
            'bb_position': (curr['close'] - curr['bb_lower']) / (curr['bb_upper'] - curr['bb_lower']),
            'bb_width': (curr['bb_upper'] - curr['bb_lower']) / curr['bb_middle'],
            'atr_pct': curr['atr'] / curr['close'],
            'momentum_pct': curr['momentum_pct'],
            'volume_ratio': curr['volume'] / curr['volume_ma'],
            'roc': curr['roc'],
            'volatility_pct': curr['volatility'] / curr['close'],
        }

        # One-hot encode
        for st in ['RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'MOMENTUM_UP', 'MOMENTUM_DOWN',
                   'BB_BREAKOUT_UP', 'BB_BREAKOUT_DOWN']:
            features[f'signal_{st}'] = 1 if st == sig_type else 0

        X = pd.DataFrame([features])[feature_cols].fillna(0)
        proba = model.predict_proba(X)[0, 1]

        if proba >= 0.55:
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
    print(f"✅ Generated {len(signals_df):,} signals (ML filtered)")

    # Simulate trades
    print(f"\n⚡ Simulating trades...")
    trades = []

    total_signals = len(signals_df)
    for idx, sig in enumerate(signals_df.iterrows()):
        if idx % 100 == 0:
            print(f"   Progress: {idx}/{total_signals} ({idx/total_signals*100:.1f}%)")
        sig = sig[1]  # Get the actual signal data
        future = data.loc[sig['timestamp']:]

        exit_price = None
        exit_reason = None

        for idx, bar in future.iterrows():
            if idx == sig['timestamp']:
                continue

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
    print(f"✅ Completed {len(trades_df):,} trades")

    # Results
    if len(trades_df) > 0:
        total_return = trades_df['return_pct'].sum()
        win_rate = (trades_df['return_pct'] > 0).sum() / len(trades_df) * 100
        avg_return = trades_df['return_pct'].mean()
        std_return = trades_df['return_pct'].std()
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        cumulative = trades_df['return_pct'].cumsum()
        running_max = cumulative.expanding().max()
        max_dd = (cumulative - running_max).min()

        print(f"\n{'='*60}")
        print("6-MONTH BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Overall Performance:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Max Drawdown: {max_dd:.2f}%")
        print(f"   Total Trades: {len(trades_df):,}")

        print(f"\n📈 Trade Statistics:")
        print(f"   Avg Return: {avg_return:.3f}%")
        print(f"   Std Dev: {std_return:.3f}%")
        print(f"   Best Trade: +{trades_df['return_pct'].max():.2f}%")
        print(f"   Worst Trade: {trades_df['return_pct'].min():.2f}%")
        print(f"   Avg ML Prob: {trades_df['ml_prob'].mean():.3f}")

        # Monthly breakdown
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly = trades_df.groupby('month').agg({
            'return_pct': ['sum', 'count'],
        })

        print(f"\n📅 Monthly Performance:")
        for month, row in monthly.iterrows():
            month_return = row[('return_pct', 'sum')]
            month_trades = row[('return_pct', 'count')]
            print(f"   {month}: {month_return:+.2f}% ({month_trades:,} trades)")

        # Strategy breakdown
        print(f"\n🎯 Strategy Performance:")
        for strategy in trades_df['signal_type'].unique():
            strat = trades_df[trades_df['signal_type'] == strategy]
            wr = (strat['return_pct'] > 0).sum() / len(strat) * 100
            tr = strat['return_pct'].sum()
            ar = strat['return_pct'].mean()
            print(f"   {strategy}:")
            print(f"     Trades: {len(strat):,} | Win Rate: {wr:.1f}% | Total: {tr:+.2f}% | Avg: {ar:+.2f}%")

        # Compounded growth
        starting_capital = 100000
        ending_value = starting_capital * (1 + total_return/100)
        print(f"\n💰 Capital Growth:")
        print(f"   Starting: ${starting_capital:,.2f}")
        print(f"   Ending: ${ending_value:,.2f}")
        print(f"   Growth: {total_return:.2f}%")

    print(f"\n{'='*60}")
    print("✅ BACKTEST COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

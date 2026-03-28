#!/usr/bin/env python3
"""Extended ML backtest on longer time period."""

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

def generate_signals_ml(df, model, feature_cols, threshold):
    signals = []

    for i in range(100, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]

        # Signal types
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

    return pd.DataFrame(signals)

def simulate_trades(df, signals):
    trades = []

    for _, sig in signals.iterrows():
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

    return pd.DataFrame(trades)

def main():
    print("📊 EXTENDED ML BACKTEST (6 Months)")
    print("=" * 60)

    # Load model
    print("\n📂 Loading ML model...")
    model = joblib.load('data/models/xgboost_mnq_classifier.pkl')
    feature_cols = joblib.load('data/models/feature_columns.pkl')
    print("✅ Model loaded")

    # Extended data range
    print("\n📊 Loading extended historical data...")
    loader = HistoricalDataLoader(data_directory="data/processed/dollar_bars/", min_completeness=0.1)

    # Use 6 months: Oct 2024 to Mar 2025
    data = loader.load_data('2024-10-01', '2025-03-05')
    print(f"✅ Loaded {len(data)} bars (Oct 2024 - Mar 2025)")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")

    # Calculate indicators
    print("\n🔬 Calculating technical indicators...")
    data = calculate_indicators(data)

    # Test different thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"\n{'='*60}")
    print("TESTING MULTIPLE ML THRESHOLDS")
    print(f"{'='*60}\n")

    all_results = []

    for threshold in thresholds:
        print(f"Testing threshold {threshold}...")

        signals = generate_signals_ml(data, model, feature_cols, threshold)

        if len(signals) == 0:
            print(f"   No signals at threshold {threshold}")
            continue

        trades = simulate_trades(data, signals)

        if len(trades) == 0:
            print(f"   No trades at threshold {threshold}")
            continue

        # Metrics
        total_return = trades['return_pct'].sum()
        win_rate = (trades['return_pct'] > 0).sum() / len(trades) * 100
        avg_return = trades['return_pct'].mean()
        std_return = trades['return_pct'].std()

        # Sharpe ratio (simplified)
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Max drawdown
        cumulative = trades['return_pct'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max)
        max_dd = drawdown.min()

        print(f"   Signals: {len(signals):,} | Trades: {len(trades):,}")
        print(f"   Return: {total_return:.2f}% | Sharpe: {sharpe:.2f} | Win Rate: {win_rate:.1f}%")
        print(f"   Max DD: {max_dd:.2f}%\n")

        all_results.append({
            'threshold': threshold,
            'signals': len(signals),
            'trades': len(trades),
            'return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_dd': max_dd
        })

    # Summary table
    if all_results:
        print(f"{'='*60}")
        print("THRESHOLD COMPARISON (6-MONTH RESULTS)")
        print(f"{'='*60}\n")

        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))

        # Best threshold
        best_idx = results_df['return'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_return = results_df.loc[best_idx, 'return']
        best_sharpe = results_df.loc[best_idx, 'sharpe']

        print(f"\n🏆 Best threshold: {best_threshold}")
        print(f"   Return: {best_return:.2f}% | Sharpe: {best_sharpe:.2f}")

        # Detailed analysis with best threshold
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS (threshold={best_threshold})")
        print(f"{'='*60}\n")

        signals = generate_signals_ml(data, model, feature_cols, best_threshold)
        trades = simulate_trades(data, signals)

        # Monthly breakdown
        trades['month'] = pd.to_datetime(trades['entry_time']).dt.to_period('M')
        monthly_stats = trades.groupby('month').agg({
            'return_pct': 'sum',
            'entry_time': 'count'
        }).rename(columns={'entry_time': 'trades', 'return_pct': 'return'})

        print("Monthly Performance:")
        print(monthly_stats.to_string())

        print(f"\nStrategy Performance:")
        for strategy in trades['signal_type'].unique():
            strat_trades = trades[trades['signal_type'] == strategy]
            win_rate = (strat_trades['return_pct'] > 0).sum() / len(strat_trades) * 100
            total_return = strat_trades['return_pct'].sum()
            avg_return = strat_trades['return_pct'].mean()
            print(f"   - {strategy}:")
            print(f"     Trades: {len(strat_trades)} | Win Rate: {win_rate:.1f}% | Total: {total_return:.2f}% | Avg: {avg_return:.2f}%")

        print(f"\nExit Reason Analysis:")
        for reason, count in trades['exit_reason'].value_counts().items():
            avg_ret = trades[trades['exit_reason'] == reason]['return_pct'].mean()
            pct = (count / len(trades)) * 100
            print(f"   - {reason}: {count} ({pct:.1f}%) | Avg: {avg_ret:.2f}%")

    print(f"\n{'='*60}")
    print("✅ EXTENDED BACKTEST COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

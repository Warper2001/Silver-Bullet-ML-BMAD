#!/usr/bin/env python3
"""Run backtest with real MNQ dollar bars using simple trading strategies."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.research.historical_data_loader import HistoricalDataLoader
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    """Calculate technical indicators for trading signals."""
    # RSI (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    # ATR (Average True Range) for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(10)

    return df

def generate_trading_signals(df):
    """Generate trading signals based on technical indicators."""
    signals = []

    for i in range(50, len(df)):  # Start after indicators are calculated
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]

        # Strategy 1: RSI Mean Reversion
        if current_bar['rsi'] < 30 and prev_bar['rsi'] >= 30:
            signals.append({
                'timestamp': current_bar.name,
                'type': 'RSI_MEAN_REVERSION_LONG',
                'direction': 'bullish',
                'entry_price': current_bar['close'],
                'confidence': 65,
                'stop_loss': current_bar['close'] - (current_bar['atr'] * 2),
                'take_profit': current_bar['close'] + (current_bar['atr'] * 3)
            })

        elif current_bar['rsi'] > 70 and prev_bar['rsi'] <= 70:
            signals.append({
                'timestamp': current_bar.name,
                'type': 'RSI_MEAN_REVERSION_SHORT',
                'direction': 'bearish',
                'entry_price': current_bar['close'],
                'confidence': 65,
                'stop_loss': current_bar['close'] + (current_bar['atr'] * 2),
                'take_profit': current_bar['close'] - (current_bar['atr'] * 3)
            })

        # Strategy 2: Moving Average Crossover
        if (current_bar['sma_20'] > current_bar['sma_50'] and
            prev_bar['sma_20'] <= prev_bar['sma_50']):
            signals.append({
                'timestamp': current_bar.name,
                'type': 'MA_CROSSOVER_LONG',
                'direction': 'bullish',
                'entry_price': current_bar['close'],
                'confidence': 60,
                'stop_loss': current_bar['close'] - (current_bar['atr'] * 1.5),
                'take_profit': current_bar['close'] + (current_bar['atr'] * 2.5)
            })

        elif (current_bar['sma_20'] < current_bar['sma_50'] and
              prev_bar['sma_20'] >= prev_bar['sma_50']):
            signals.append({
                'timestamp': current_bar.name,
                'type': 'MA_CROSSOVER_SHORT',
                'direction': 'bearish',
                'entry_price': current_bar['close'],
                'confidence': 60,
                'stop_loss': current_bar['close'] + (current_bar['atr'] * 1.5),
                'take_profit': current_bar['close'] - (current_bar['atr'] * 2.5)
            })

        # Strategy 3: Bollinger Band Breakout
        if current_bar['close'] > current_bar['bb_upper']:
            signals.append({
                'timestamp': current_bar.name,
                'type': 'BB_BREAKOUT_LONG',
                'direction': 'bullish',
                'entry_price': current_bar['close'],
                'confidence': 55,
                'stop_loss': current_bar['bb_middle'],
                'take_profit': current_bar['close'] + (current_bar['atr'] * 2)
            })

        elif current_bar['close'] < current_bar['bb_lower']:
            signals.append({
                'timestamp': current_bar.name,
                'type': 'BB_BREAKOUT_SHORT',
                'direction': 'bearish',
                'entry_price': current_bar['close'],
                'confidence': 55,
                'stop_loss': current_bar['bb_middle'],
                'take_profit': current_bar['close'] - (current_bar['atr'] * 2)
            })

        # Strategy 4: Momentum Breakout
        if current_bar['momentum'] > 0 and prev_bar['momentum'] <= 0:
            signals.append({
                'timestamp': current_bar.name,
                'type': 'MOMENTUM_LONG',
                'direction': 'bullish',
                'entry_price': current_bar['close'],
                'confidence': 58,
                'stop_loss': current_bar['close'] - (current_bar['atr'] * 1.5),
                'take_profit': current_bar['close'] + (current_bar['atr'] * 2)
            })

        elif current_bar['momentum'] < 0 and prev_bar['momentum'] >= 0:
            signals.append({
                'timestamp': current_bar.name,
                'type': 'MOMENTUM_SHORT',
                'direction': 'bearish',
                'entry_price': current_bar['close'],
                'confidence': 58,
                'stop_loss': current_bar['close'] + (current_bar['atr'] * 1.5),
                'take_profit': current_bar['close'] - (current_bar['atr'] * 2)
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

        # Find exit
        exit_price = None
        exit_time = None
        exit_reason = None

        # Look ahead in data for exit
        future_data = df.loc[entry_time:]

        for future_idx, future_bar in future_data.iterrows():
            if future_idx == entry_time:
                continue

            if direction == 'bullish':
                # Check stop loss
                if future_bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = future_idx
                    exit_reason = 'stop_loss'
                    break

                # Check take profit
                if future_bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_time = future_idx
                    exit_reason = 'take_profit'
                    break

            else:  # bearish
                # Check stop loss
                if future_bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_time = future_idx
                    exit_reason = 'stop_loss'
                    break

                # Check take profit
                if future_bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_time = future_idx
                    exit_reason = 'take_profit'
                    break

        # If no exit found, use last available price
        if exit_price is None and len(future_data) > 1:
            exit_price = future_data.iloc[-1]['close']
            exit_time = future_data.index[-1]
            exit_reason = 'end_of_data'

        if exit_price is not None:
            # Calculate PnL
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
                'signal_type': signal['type'],
                'confidence': signal['confidence'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'return_pct': (pnl / entry_price) * 100
            })

    return pd.DataFrame(trades)

def main():
    """Run backtest with real MNQ dollar bars."""

    print("🚀 REAL MNQ DOLLAR BAR BACKTEST")
    print("=" * 70)

    # Load data
    print("\n📊 Loading historical data...")
    loader = HistoricalDataLoader(
        data_directory="data/processed/dollar_bars/",
        min_completeness=0.1
    )

    data = loader.load_data('2025-01-01', '2025-03-05')
    print(f"✅ Loaded {len(data)} bars")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Calculate indicators
    print("\n🔬 Calculating technical indicators...")
    data = calculate_technical_indicators(data)
    print(f"✅ Indicators calculated: RSI, SMA(20,50), BB, ATR, Momentum")

    # Generate signals
    print("\n🎯 Generating trading signals...")
    signals = generate_trading_signals(data)
    print(f"✅ Generated {len(signals)} signals")

    if len(signals) > 0:
        print(f"\n   Signal breakdown:")
        print(f"   - RSI Mean Reversion: {len(signals[signals['type'].str.contains('RSI')])}")
        print(f"   - MA Crossover: {len(signals[signals['type'].str.contains('MA')])}")
        print(f"   - BB Breakout: {len(signals[signals['type'].str.contains('BB')])}")
        print(f"   - Momentum: {len(signals[signals['type'].str.contains('MOMENTUM')])}")

        print(f"\n   Direction breakdown:")
        print(f"   - Bullish: {len(signals[signals['direction'] == 'bullish'])}")
        print(f"   - Bearish: {len(signals[signals['direction'] == 'bearish'])}")

    # Simulate trades
    print("\n⚡ Simulating trades...")
    trades = simulate_trades(data, signals)
    print(f"✅ Simulated {len(trades)} trades")

    if len(trades) == 0:
        print("\n❌ No trades completed!")
        return

    # Calculate metrics
    print("\n📈 Calculating performance metrics...")
    metrics_calc = PerformanceMetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(trades)

    # Extract values safely
    total_return = metrics.get('total_return', {})
    if isinstance(total_return, dict):
        total_return_val = total_return.get('total_return_pct', 0)
    else:
        total_return_val = total_return

    sharpe = metrics.get('sharpe_ratio', 0)
    win_rate = metrics.get('win_rate', 0)
    total_trades = len(trades)

    max_dd = metrics.get('max_drawdown', {})
    if isinstance(max_dd, dict):
        max_dd_val = max_dd.get('max_drawdown_pct', 0)
    else:
        max_dd_val = max_dd

    print(f"\n✅ Performance Metrics:")
    print(f"   Total Return: {total_return_val:.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Total Trades: {total_trades}")
    print(f"   Max Drawdown: {max_dd_val:.2f}%")

    # Trade analysis
    print(f"\n📊 Trade Analysis:")
    print(f"   Avg Return per Trade: {trades['return_pct'].mean():.2f}%")
    print(f"   Std Dev of Returns: {trades['return_pct'].std():.2f}%")
    print(f"   Best Trade: +{trades['return_pct'].max():.2f}%")
    print(f"   Worst Trade: {trades['return_pct'].min():.2f}%")

    print(f"\n   Exit Reason Breakdown:")
    for reason, count in trades['exit_reason'].value_counts().items():
        pct = (count / len(trades)) * 100
        avg_return = trades[trades['exit_reason'] == reason]['return_pct'].mean()
        print(f"   - {reason}: {count} ({pct:.1f}%) | Avg return: {avg_return:.2f}%")

    print(f"\n   Strategy Performance:")
    for strategy in trades['signal_type'].unique():
        strategy_trades = trades[trades['signal_type'] == strategy]
        win_rate = (strategy_trades['return_pct'] > 0).sum() / len(strategy_trades) * 100
        avg_return = strategy_trades['return_pct'].mean()
        total_return = strategy_trades['return_pct'].sum()
        print(f"   - {strategy}:")
        print(f"     Trades: {len(strategy_trades)} | Win Rate: {win_rate:.1f}% | Avg: {avg_return:.2f}% | Total: {total_return:.2f}%")

    # Show sample trades
    print(f"\n📋 Sample Trades (last 5):")
    sample_cols = ['entry_time', 'direction', 'signal_type', 'entry_price', 'exit_price', 'return_pct', 'exit_reason']
    print(trades[sample_cols].tail().to_string(index=False))

    print("\n✅ Backtest complete!")
    print(f"\n💰 Starting with $100,000, ending value would be: ${100000 * (1 + total_return_val/100):,.2f}")

if __name__ == '__main__':
    main()

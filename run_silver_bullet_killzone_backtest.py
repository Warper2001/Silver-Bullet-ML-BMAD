#!/usr/bin/env python3
"""Run Silver Bullet Killzone backtest with real MNQ data.

This script implements the actual ICT Silver Bullet strategy with:
- Time-based bars (5-minute)
- Killzone-only trading (London AM, NY AM, NY PM)
- Real pattern detection (MSS, FVG, Liquidity Sweeps)
- Silver Bullet confluence requirements
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
from src.data.models import DollarBar

# MNQ contract specifications
MNQ_MULTIPLIER = 0.5


def load_time_bars(date_start: str, date_end: str) -> pd.DataFrame:
    """Load time-based bars for backtesting.

    Args:
        date_start: Start date (YYYY-MM-DD)
        date_end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with time-based bars
    """
    print(f"📊 Loading time bars from {date_start} to {date_end}...")

    import h5py
    from pathlib import Path

    data_dir = Path("data/processed/time_bars/")
    start_dt = pd.to_datetime(date_start)
    end_dt = pd.to_datetime(date_end)

    # Collect files for date range
    files = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        filename = f"MNQ_time_bars_5min_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename
        if file_path.exists():
            files.append(file_path)
        current = current + pd.DateOffset(months=1)

    if not files:
        print(f"❌ No HDF5 files found for date range")
        return pd.DataFrame()

    # Load all files
    dataframes = []
    for file_path in files:
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

    if not dataframes:
        print(f"❌ No data loaded from files")
        return pd.DataFrame()

    # Combine data
    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')

    # Filter to date range
    combined = combined.loc[
        (combined.index >= start_dt) & (combined.index <= end_dt)
    ]

    print(f"✅ Loaded {len(combined):,} time bars")
    print(f"   Date range: {combined.index.min()} to {combined.index.max()}")
    print(f"   Price range: ${combined['close'].min():.2f} - ${combined['close'].max():.2f}")

    return combined


def dataframe_to_dollar_bars(df: pd.DataFrame) -> list[DollarBar]:
    """Convert time-bar DataFrame to DollarBar objects.

    Args:
        df: DataFrame with time-bar OHLCV data

    Returns:
        List of DollarBar objects
    """
    bars = []

    for idx, row in df.iterrows():
        # Calculate notional value
        notional_value = row['close'] * row['volume'] * MNQ_MULTIPLIER

        bar = DollarBar(
            timestamp=idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(idx),
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=int(row['volume']),
            notional_value=notional_value
        )
        bars.append(bar)

    return bars


def run_silver_bullet_backtest(data: pd.DataFrame, enable_killzones: bool = True):
    """Run Silver Bullet backtest with pattern detection.

    Args:
        data: DataFrame with time bars
        enable_killzones: Whether to use killzone time filtering

    Returns:
        DataFrame with backtest signals
    """
    print("\n🎯 Running Silver Bullet Pattern Detection...")

    # Create backtester with ICT parameters
    backtester = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        fvg_gap_atr_multiple=0.1,
        sweep_lookback=5,
        sweep_min_volume_ratio=1.3,
        sweep_min_depth=0.10,
        max_bar_distance=10,
        min_confidence=60.0,
        enable_time_windows=enable_killzones,
        time_windows=None  # Use default ICT killzones
    )

    # Run backtest (expects DataFrame)
    signals_df = backtester.run_backtest(data)

    print(f"\n✅ Pattern Detection Complete:")
    print(f"   Total signals detected: {len(signals_df)}")

    if len(signals_df) > 0:
        # Show signal breakdown
        print(f"\n   Signal Distribution:")
        print(f"   - Bullish: {len(signals_df[signals_df['direction'] == 'bullish'])}")
        print(f"   - Bearish: {len(signals_df[signals_df['direction'] == 'bearish'])}")

        # Show by killzone
        if 'time_window' in signals_df.columns:
            print(f"\n   By Killzone:")
            for zone, count in signals_df['time_window'].value_counts().items():
                if zone is not None:
                    print(f"   - {zone}: {count}")
                else:
                    print(f"   - Other: {count}")

        # Show confidence distribution
        print(f"\n   Confidence Levels:")
        print(f"   - Avg confidence: {signals_df['confidence'].mean():.1f}")
        print(f"   - Min confidence: {signals_df['confidence'].min():.0f}")
        print(f"   - Max confidence: {signals_df['confidence'].max():.0f}")

        # Pattern breakdown
        print(f"\n   Pattern Confluence:")
        if 'mss_detected' in signals_df.columns:
            mss_count = signals_df['mss_detected'].sum()
            print(f"   - MSS detected: {mss_count}/{len(signals_df)}")

        if 'fvg_detected' in signals_df.columns:
            fvg_count = signals_df['fvg_detected'].sum()
            print(f"   - FVG detected: {fvg_count}/{len(signals_df)}")

        if 'sweep_detected' in signals_df.columns:
            sweep_count = signals_df['sweep_detected'].sum()
            print(f"   - Sweep detected: {sweep_count}/{len(signals_df)}")

    return signals_df


def simulate_trades(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Simulate trades with basic risk management.

    Args:
        data: Price data
        signals: Trading signals

    Returns:
        DataFrame with trade results
    """
    print(f"\n⚡ Simulating trades on {len(signals)} signals...")

    trades = []
    atr = (data['high'] - data['low']).rolling(14).mean()

    for idx, signal in signals.iterrows():
        entry_time = signal.name
        entry_price = data.loc[entry_time, 'close']
        direction = signal['direction']

        # Calculate stop loss and take profit (2:1 risk-reward)
        current_atr = atr.loc[entry_time] if entry_time in atr.index else atr.iloc[-1]

        if direction == 'bullish':
            stop_loss = entry_price - (current_atr * 1.5)
            take_profit = entry_price + (current_atr * 3.0)
        else:  # bearish
            stop_loss = entry_price + (current_atr * 1.5)
            take_profit = entry_price - (current_atr * 3.0)

        # Find exit
        future_data = data.loc[entry_time:]

        exit_price = None
        exit_reason = None
        exit_time = None

        for exit_idx, bar in future_data.iterrows():
            if exit_idx == entry_time:
                continue

            if direction == 'bullish':
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    exit_time = exit_idx
                    break
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    exit_time = exit_idx
                    break
            else:  # bearish
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    exit_time = exit_idx
                    break
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    exit_time = exit_idx
                    break

        # If no exit found, use last available price
        if exit_price is None and len(future_data) > 1:
            exit_price = future_data.iloc[-1]['close']
            exit_reason = 'end_of_data'
            exit_time = future_data.index[-1]

        if exit_price is not None:
            if direction == 'bullish':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'confidence': signal['confidence'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'return_pct': (pnl / entry_price) * 100
            })

    trades_df = pd.DataFrame(trades)
    print(f"✅ Completed {len(trades_df)} trades")

    return trades_df


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Calculate performance metrics.

    Args:
        trades_df: DataFrame with trade results

    Returns:
        Dictionary with performance metrics
    """
    if len(trades_df) == 0:
        return {}

    metrics_calc = PerformanceMetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(trades_df)

    # Additional metrics
    total_return = trades_df['return_pct'].sum()
    win_rate = (trades_df['return_pct'] > 0).sum() / len(trades_df) * 100

    # Calculate max drawdown
    cumulative = trades_df['return_pct'].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    # Sharpe ratio (simplified)
    avg_return = trades_df['return_pct'].mean()
    std_return = trades_df['return_pct'].std()
    sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    print(f"\n📈 Performance Metrics:")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Max Drawdown: {max_dd:.2f}%")
    print(f"   Total Trades: {len(trades_df)}")

    print(f"\n📊 Trade Statistics:")
    print(f"   Avg Return: {avg_return:.3f}%")
    print(f"   Best Trade: +{trades_df['return_pct'].max():.2f}%")
    print(f"   Worst Trade: {trades_df['return_pct'].min():.2f}%")

    # Exit reason breakdown
    print(f"\n🎯 Exit Reason Breakdown:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        avg_return = trades_df[trades_df['exit_reason'] == reason]['return_pct'].mean()
        print(f"   - {reason}: {count} ({count/len(trades_df)*100:.1f}%) | Avg: {avg_return:.3f}%")

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'total_trades': len(trades_df)
    }


def main():
    """Run Silver Bullet Killzone backtest."""

    print("🚀 SILVER BULLET KILLZONE BACKTEST")
    print("=" * 70)
    print("Real ICT Silver Bullet Strategy with Time-Based Bars and Killzones")
    print("=" * 70)

    # Step 1: Load time-based killzone data
    print("\n📊 Step 1: Loading time-based killzone data...")
    # Use recent data for meaningful backtest
    data = load_time_bars('2024-12-01', '2025-03-05')

    if data.empty:
        print("❌ No data available!")
        return

    # Step 2: Run Silver Bullet pattern detection
    print("\n🎯 Step 2: Running Silver Bullet pattern detection...")
    signals = run_silver_bullet_backtest(data, enable_killzones=True)

    if len(signals) == 0:
        print("\n❌ No Silver Bullet patterns detected!")
        print("\n💡 Possible reasons:")
        print("   - Time bars may need different parameters")
        print("   - Killzone filtering too restrictive")
        print("   - Market conditions don't produce patterns")
        return

    # Step 3: Simulate trades
    print("\n⚡ Step 3: Simulating trades with risk management...")
    trades = simulate_trades(data, signals)

    if len(trades) == 0:
        print("❌ No trades completed!")
        return

    # Step 4: Calculate metrics
    print("\n📈 Step 4: Calculating performance metrics...")
    metrics = calculate_metrics(trades)

    # Final summary
    print(f"\n{'='*70}")
    print("✅ SILVER BULLET KILLZONE BACKTEST COMPLETE")
    print(f"{'='*70}")

    print(f"\n💰 Starting Capital: $100,000")
    final_value = 100000 * (1 + metrics['total_return'] / 100)
    print(f"   Final Capital: ${final_value:,.2f}")
    print(f"   Growth: {metrics['total_return']:.2f}%")

    print(f"\n🎯 Key Findings:")
    print(f"   - Strategy: {'PASS' if metrics['total_return'] > 0 else 'FAIL'}")
    print(f"   - Win Rate: {'GOOD' if metrics['win_rate'] >= 50 else 'NEEDS IMPROVEMENT'}")
    print(f"   - Max DD: {'ACCEPTABLE' if metrics['max_drawdown'] > -20 else 'HIGH RISK'}")
    print(f"   - Sharpe: {'EXCELLENT' if metrics['sharpe_ratio'] > 1 else 'NEEDS WORK'}")


if __name__ == '__main__':
    main()

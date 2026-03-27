#!/usr/bin/env python3
"""Enhanced Silver Bullet Baseline - No ML Required.

Improvements over baseline:
1. Adaptive Volatility Filter - Dynamic ATR% threshold by regime
2. Time-of-Day Filter - Trade only ICT killzones
3. Multi-Timeframe Confirmation - 15-min trend alignment required
4. Signal Confirmation Stack - Volume + RSI + VWAP checks
5. Dynamic Stop Loss - Scale by volatility regime

Target: 60%+ win rate without ML filtering.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester


def load_time_bars(date_start: str, date_end: str, timeframe: str = '5min') -> pd.DataFrame:
    """Load time-based bars for backtesting."""
    print(f"📊 Loading {timeframe} time bars from {date_start} to {date_end}...")

    import h5py
    data_dir = Path("data/processed/time_bars/")

    start_dt = pd.Timestamp(date_start)
    end_dt = pd.Timestamp(date_end)
    current = start_dt.replace(day=1)

    files = []
    while current <= end_dt:
        # Try to load requested timeframe, fall back to 5min
        filename = f"MNQ_time_bars_{timeframe}_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename

        # If not found, try 5min as source
        if not file_path.exists() and timeframe != '5min':
            filename_5min = f"MNQ_time_bars_5min_{current.strftime('%Y%m')}.h5"
            file_path_5min = data_dir / filename_5min
            if file_path_5min.exists():
                files.append((file_path_5min, True))  # True = needs resample
        else:
            if file_path.exists():
                files.append((file_path, False))  # False = direct load

        current = current + pd.DateOffset(months=1)

    dataframes = []
    for file_path, needs_resample in files:
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['dollar_bars'][:]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Resample if needed
            if needs_resample:
                df = df.set_index('timestamp')
                if timeframe == '15min':
                    df = df.resample('15min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'notional_value': 'sum'
                    }).dropna()
                    df = df.reset_index()
                else:
                    # For other timeframes, keep 5min
                    pass

            dataframes.append(df)
        except Exception as e:
            print(f"   Warning: Failed to load {file_path.name}: {e}")

    if not dataframes:
        print(f"   No data found for {timeframe}")
        return pd.DataFrame()

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) & (combined.index <= end_dt)
    ]

    print(f"✅ Loaded {len(combined):,} {timeframe} bars")

    return combined


def calculate_daily_bias(daily_data: pd.DataFrame) -> pd.Series:
    """Calculate daily trend bias using SMA(50)."""
    sma50 = daily_data['close'].rolling(50).mean()
    bias = daily_data['close'] > sma50
    return bias


def get_volatility_regime(atr_pct: float) -> str:
    """Classify market into volatility regimes."""
    if atr_pct >= 0.005:
        return 'high'
    elif atr_pct <= 0.002:
        return 'low'
    else:
        return 'normal'


def get_adaptive_volatility_threshold(regime: str) -> float:
    """Return dynamic ATR% threshold based on regime.

    FIXED: High volatility should require HIGHER threshold (be more selective).
    Low volatility can use LOWER threshold (still catch moves).
    """
    thresholds = {
        'high': 0.005,    # 0.5% - require MORE movement in high vol (be selective)
        'normal': 0.003,  # 0.3% - standard threshold
        'low': 0.002      # 0.2% - allow LESS in low vol (still catch moves)
    }
    return thresholds[regime]


def is_killzone(dt: pd.Timestamp) -> bool:
    """Check if time is within ICT killzone.

    Killzones:
    - London AM: 2am-5am ET
    - NY AM: 9:30am-11am ET
    - NY PM: 3pm-5pm ET
    """
    hour = dt.hour
    minute = dt.minute
    time_minutes = hour * 60 + minute

    # London AM: 2:00-5:00 ET (120-300 minutes)
    if 120 <= time_minutes <= 300:
        return True

    # NY AM: 9:30-11:00 ET (570-660 minutes)
    if 570 <= time_minutes <= 660:
        return True

    # NY PM: 15:00-17:00 ET (900-1020 minutes)
    if 900 <= time_minutes <= 1020:
        return True

    return False


def is_15min_uptrend(data_15min: pd.DataFrame, signal_time: pd.Timestamp) -> bool:
    """Check if 15-min timeframe is in uptrend at signal time."""
    if data_15min.empty:
        return True  # No data, assume OK

    # Get recent 15-min bars
    recent = data_15min.loc[:signal_time].tail(50)

    if len(recent) < 20:
        return True  # Not enough data, assume OK

    # Simple trend check: current close > SMA(20)
    sma20 = recent['close'].rolling(20).mean()
    current_close = recent['close'].iloc[-1]

    return current_close > sma20.iloc[-1]


def check_volume_surge(data: pd.DataFrame, signal_time: pd.Timestamp, threshold: float = 1.5) -> bool:
    """Check if volume is above average (surge)."""
    recent = data.loc[:signal_time].tail(50)
    if len(recent) < 20:
        return True

    avg_volume = recent['volume'].rolling(20).mean().iloc[-1]
    current_volume = data.loc[signal_time, 'volume']

    return current_volume >= (avg_volume * threshold)


def check_rsi_range(data: pd.DataFrame, signal_time: pd.Timestamp,
                    min_rsi: int = 30, max_rsi: int = 70) -> bool:
    """Check if RSI is in reasonable range (not extreme)."""
    recent = data.loc[:signal_time].tail(50)
    if len(recent) < 14:
        return True

    # Calculate RSI(14)
    deltas = recent['close'].diff()
    gain = (deltas.where(deltas > 0, 0)).rolling(14).mean()
    loss = (-deltas.where(deltas < 0, 0)).rolling(14).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]

    return min_rsi <= current_rsi <= max_rsi


def check_vwap_alignment(data: pd.DataFrame, signal_time: pd.Timestamp,
                        direction: str) -> bool:
    """Check if price is aligned with VWAP."""
    recent = data.loc[:signal_time].tail(50)
    if len(recent) < 20:
        return True

    # Calculate VWAP
    typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
    vwap = (typical_price * recent['volume']).cumsum() / recent['volume'].cumsum()
    current_vwap = vwap.iloc[-1]
    current_price = data.loc[signal_time, 'close']

    if direction == 'bullish':
        return current_price > current_vwap
    else:
        return current_price < current_vwap


def has_signal_confirmation(data: pd.DataFrame, signal_time: pd.Timestamp,
                           direction: str, min_confirmations: int = 2) -> bool:
    """Check signal confirmation stack (require N/3 checks)."""
    checks = []

    # Check 1: Volume surge
    checks.append(check_volume_surge(data, signal_time))

    # Check 2: RSI not extreme
    checks.append(check_rsi_range(data, signal_time))

    # Check 3: VWAP alignment
    checks.append(check_vwap_alignment(data, signal_time, direction))

    passed = sum(checks)
    return passed >= min_confirmations


def add_enhanced_filters(
    data: pd.DataFrame,
    data_15min: pd.DataFrame,
    signals_df: pd.DataFrame,
    daily_bias: pd.Series,
    use_time_filter: bool = True,
    use_adaptive_vol: bool = True,
    use_mtf: bool = True,
    use_confirmation: bool = True
) -> pd.DataFrame:
    """Apply all enhanced filters to signals.

    FIXED: Filter order is now critical - Daily Bias FIRST to remove wrong-direction signals early.
    """

    print(f"\n🎯 Applying Enhanced Filters...")
    print(f"   Starting signals: {len(signals_df)}")

    filtered_signals = []
    stats = {
        'daily_bias': 0,
        'adaptive_vol': 0,
        'killzone': 0,
        'mtf': 0,
        'confirmation': 0
    }

    for idx, signal in signals_df.iterrows():
        signal_time = pd.Timestamp(idx)
        keep = True

        # Filter 1: Daily Bias FIRST (remove wrong-direction signals early)
        daily_date = signal_time.date()
        bias_dates = [ts.date() for ts in daily_bias.index]

        if daily_date in bias_dates:
            bias_idx = [i for i, ts in enumerate(daily_bias.index) if ts.date() == daily_date][0]
            is_uptrend = daily_bias.iloc[bias_idx]

            if signal['direction'] == 'bullish' and not is_uptrend:
                stats['daily_bias'] += 1
                keep = False
            elif signal['direction'] == 'bearish' and is_uptrend:
                stats['daily_bias'] += 1
                keep = False

        if not keep:
            continue

        # Filter 2: Adaptive Volatility (second, after bias)
        if use_adaptive_vol:
            if signal_time in data.index:
                atr = (data['high'] - data['low']).rolling(14).mean()
                atr_pct = atr / data['close']
                current_atr_pct = atr_pct.loc[signal_time] if signal_time in atr_pct.index else 0.003

                regime = get_volatility_regime(current_atr_pct)
                threshold = get_adaptive_volatility_threshold(regime)

                if current_atr_pct < threshold:
                    stats['adaptive_vol'] += 1
                    keep = False

        if not keep:
            continue

        # Filter 3: Multi-Timeframe Confirmation
        if use_mtf and data_15min is not None:
            if signal['direction'] == 'bullish':
                if not is_15min_uptrend(data_15min, signal_time):
                    stats['mtf'] += 1
                    keep = False
            else:
                # Bearish: check 15-min downtrend
                if is_15min_uptrend(data_15min, signal_time):
                    stats['mtf'] += 1
                    keep = False

        if not keep:
            continue

        # Filter 4: Signal Confirmation Stack
        if use_confirmation:
            if not has_signal_confirmation(data, signal_time, signal['direction']):
                stats['confirmation'] += 1
                keep = False

        if not keep:
            continue

        # Filter 5: Time-of-Day (Killzone only) - LAST to fine-tune timing
        if use_time_filter:
            if not is_killzone(signal_time):
                stats['killzone'] += 1
                keep = False

        if keep:
            filtered_signals.append(signal)

    result_df = pd.DataFrame(filtered_signals) if filtered_signals else signals_df.iloc[:0]

    print(f"\n📊 Filter Breakdown:")
    print(f"   Daily bias (1st):      {stats['daily_bias']:,} removed")
    print(f"   Adaptive vol (2nd):    {stats['adaptive_vol']:,} removed")
    print(f"   Multi-timeframe (3rd): {stats['mtf']:,} removed")
    print(f"   Signal confirm (4th):  {stats['confirmation']:,} removed")
    print(f"   Killzone (5th):        {stats['killzone']:,} removed")

    if len(signals_df) > 0:
        print(f"\n   ✅ Remaining signals: {len(result_df)} ({len(result_df)/len(signals_df)*100:.1f}% retained)")
    else:
        print(f"\n   ⚠️  No signals detected in this period")

    return result_df


def simulate_trades_with_dynamic_stops(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    risk_reward: float = 2.0
) -> pd.DataFrame:
    """Simulate trades with dynamic volatility-based stops."""

    print(f"\n⚡ Simulating trades with dynamic stop losses ({risk_reward}:1 RR)...")

    trades = []

    for idx, signal in signals.iterrows():
        entry_time = signal.name
        if entry_time not in data.index:
            continue

        entry_price = data.loc[entry_time, 'close']
        direction = signal['direction']

        # Calculate ATR and volatility regime
        atr = (data['high'] - data['low']).rolling(14).mean()
        current_atr = atr.loc[entry_time] if entry_time in atr.index else atr.iloc[-1]

        atr_pct = (current_atr / entry_price)
        regime = get_volatility_regime(atr_pct)

        # Dynamic stop distance based on regime
        # FIXED: High volatility needs TIGHTER stops (not wider) to reduce risk
        if regime == 'high':
            stop_multiplier = 0.7  # Tighter stops in high vol (reduce risk)
        elif regime == 'low':
            stop_multiplier = 1.2  # Slightly wider in low vol (avoid noise)
        else:
            stop_multiplier = 1.0  # Normal

        # Calculate stop and target
        if direction == 'bullish':
            stop_loss = entry_price - (current_atr * stop_multiplier)
            take_profit = entry_price + (current_atr * stop_multiplier * risk_reward)
        else:
            stop_loss = entry_price + (current_atr * stop_multiplier)
            take_profit = entry_price - (current_atr * stop_multiplier * risk_reward)

        # Find exit
        future_data = data.loc[entry_time:]
        exit_price = None
        exit_reason = None

        for exit_idx, bar in future_data.iterrows():
            if exit_idx == entry_time:
                continue

            if direction == 'bullish':
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
            else:
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break

        if exit_price is None and len(future_data) > 1:
            exit_price = future_data.iloc[-1]['close']
            exit_reason = 'end_of_data'

        if exit_price is not None:
            if direction == 'bullish':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            trades.append({
                'entry_time': entry_time,
                'direction': direction,
                'exit_reason': exit_reason,
                'volatility_regime': regime,
                'stop_multiplier': stop_multiplier,
                'return_pct': (pnl / entry_price) * 100
            })

    trades_df = pd.DataFrame(trades)
    print(f"✅ Completed {len(trades_df)} trades")

    return trades_df


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    if len(trades_df) == 0:
        return {}

    total_return = trades_df['return_pct'].sum()
    win_rate = (trades_df['return_pct'] > 0).sum() / len(trades_df) * 100

    cumulative = trades_df['return_pct'].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    avg_return = trades_df['return_pct'].mean()
    std_return = trades_df['return_pct'].std()
    sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    print(f"\n📈 Performance Metrics:")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Max Drawdown: {max_dd:.2f}%")
    print(f"   Total Trades: {len(trades_df)}")

    # Breakdown by volatility regime
    if 'volatility_regime' in trades_df.columns:
        print(f"\n📊 Performance by Volatility Regime:")
        for regime in ['low', 'normal', 'high']:
            regime_trades = trades_df[trades_df['volatility_regime'] == regime]
            if len(regime_trades) > 0:
                regime_wr = (regime_trades['return_pct'] > 0).sum() / len(regime_trades) * 100
                print(f"   {regime.capitalize()}: {len(regime_trades)} trades, {regime_wr:.1f}% win rate")

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'total_trades': len(trades_df)
    }


def main():
    """Run enhanced baseline backtest."""

    print("🚀 ENHANCED SILVER BULLET BASELINE")
    print("=" * 70)
    print("Improvements: Adaptive Vol + Killzone + MTF + Confirmation + Dynamic Stops")
    print("=" * 70)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced Silver Bullet Baseline')
    parser.add_argument('--date-start', type=str, default='2026-02-01')
    parser.add_argument('--date-end', type=str, default='2026-02-28')
    parser.add_argument('--no-time-filter', action='store_true', help='Disable time-of-day filter')
    parser.add_argument('--no-adaptive-vol', action='store_true', help='Disable adaptive volatility')
    parser.add_argument('--no-mtf', action='store_true', help='Disable multi-timeframe confirmation')
    parser.add_argument('--no-confirmation', action='store_true', help='Disable signal confirmation')
    args = parser.parse_args()

    # Step 1: Load data
    print(f"\n📊 Step 1: Loading data...")
    data = load_time_bars(args.date_start, args.date_end, '5min')
    data_15min = load_time_bars(args.date_start, args.date_end, '15min')

    # Step 2: Calculate daily bias
    print(f"\n📊 Step 2: Calculating daily bias...")
    daily_data = data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    daily_bias = calculate_daily_bias(daily_data)
    print(f"✅ Daily bias calculated: {daily_bias.sum():.0f} uptrend days, {(~daily_bias).sum():.0f} downtrend days")

    # Step 3: Run pattern detection
    print(f"\n🎯 Step 3: Running Silver Bullet pattern detection...")
    backtester = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        min_confidence=60.0,  # Lower confidence for more signals
        enable_time_windows=False  # Disable time windows for more signals
    )
    signals_df = backtester.run_backtest(data)
    print(f"✅ Pattern detection complete: {len(signals_df):,} signals")

    # Step 4: Apply enhanced filters
    filtered_signals = add_enhanced_filters(
        data=data,
        data_15min=data_15min,
        signals_df=signals_df,
        daily_bias=daily_bias,
        use_time_filter=not args.no_time_filter,
        use_adaptive_vol=not args.no_adaptive_vol,
        use_mtf=not args.no_mtf,
        use_confirmation=not args.no_confirmation
    )

    if len(filtered_signals) == 0:
        print("\n❌ No signals remaining after filters!")
        return

    # Step 5: Simulate trades with dynamic stops
    trades_df = simulate_trades_with_dynamic_stops(data, filtered_signals)

    # Step 6: Calculate metrics
    metrics = calculate_metrics(trades_df)

    # Summary
    print("\n" + "=" * 70)
    print("✅ ENHANCED BASELINE BACKTEST COMPLETE")
    print("=" * 70)

    if metrics and metrics['win_rate'] >= 60:
        print(f"\n🎉 TARGET ACHIEVED: {metrics['win_rate']:.2f}% win rate (≥ 60%)")
    elif metrics:
        print(f"\n⚠️  Target not achieved: {metrics['win_rate']:.2f}% win rate (need ≥ 60%)")
        print(f"   Gap: {60 - metrics['win_rate']:.2f} percentage points")

    print(f"\n💰 Capital Growth:")
    print(f"   Starting: $100,000.00")
    if metrics:
        final_capital = 100000 * (1 + metrics['total_return'] / 100)
        print(f"   Ending:   ${final_capital:,.2f}")
        print(f"   Growth:   {metrics['total_return']:.2f}%")


if __name__ == '__main__':
    main()

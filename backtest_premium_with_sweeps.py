#!/usr/bin/env python3
"""
Premium Strategy Backtest - WITH LIQUIDITY SWEEPS (Correct Implementation)

This is the REAL ICT Silver Bullet implementation:
1. MSS (Market Structure Shift)
2. FVG (Fair Value Gap)
3. LIQUIDITY SWEEP of FVG (MISSING COMPONENT)

The liquidity sweep is the critical filter that prevents false breakouts.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

print("🚀 SILVER BULLET PREMIUM - WITH LIQUIDITY SWEEPS")
print("=" * 70)
print("✅ CORRECT IMPLEMENTATION - MSS + FVG + SWEEP")
print("📂 Data: /root/mnq_historical.json")
print("📅 Period: 2025")
print("=" * 70)

# Load 1-minute data
print(f"\n📊 Step 1: Loading 1-minute MNQ data...")

json_path = "/root/mnq_historical.json"

try:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    print(f"✅ Loaded {len(raw_data):,} total bars")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# Convert to list format for processing
print(f"\n📊 Step 2: Converting to bar format...")

bars_list = []

for bar_data in tqdm(raw_data, desc="Processing bars"):
    try:
        # Parse timestamp
        timestamp_str = bar_data.get('TimeStamp', '')
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Filter to 2025 data
        if timestamp.year != 2025:
            continue

        # Convert prices
        open_price = float(bar_data.get('Open', 0))
        high_price = float(bar_data.get('High', 0))
        low_price = float(bar_data.get('Low', 0))
        close_price = float(bar_data.get('Close', 0))
        volume = int(bar_data.get('TotalVolume', 0))

        if high_price == 0 or low_price == 0:
            continue

        bars_list.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
        })
    except Exception as e:
        continue

print(f"✅ Converted {len(bars_list):,} bars for 2025")

# Run strategy with SWEEP detection
print(f"\n🎯 Step 3: Running Detection with SWEEPS...")

config = {
    'min_fvg_gap': 75.0,
    'min_volume_ratio': 2.0,
    'max_bar_distance': 7,
    'min_quality_score': 0.0,
    'max_quality_score': 85.0,
    'stop_multiplier': 1.5,
    'killzone_weights': {
        'London AM': 0.90,
        'NY PM': 0.80,
        'NY AM': 0.60,
    },
    'dow_multipliers': {
        'Monday': 1.2,
        'Tuesday': 0.6,
        'Wednesday': 1.0,
        'Thursday': 1.1,
        'Friday': 1.0,
    },
    'max_trades_per_day': 20,
    # NEW: Sweep parameters
    'max_sweep_bars': 10,  # Max bars to wait for sweep
    'min_sweep_rejection': 0.5,  # Min rejection after sweep (points)
}

print(f"\n🔧 STRATEGY PARAMETERS:")
print(f"   FVG Size: ${config['min_fvg_gap']}+")
print(f"   Quality Score: 0-{config['max_quality_score']}")
print(f"   Stop Multiplier: {config['stop_multiplier']}x")
print(f"   Max Sweep Wait: {config['max_sweep_bars']} bars")

from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg
from src.detection.time_window_filter import is_within_trading_hours, DEFAULT_TRADING_WINDOWS
from silver_bullet_premium_enhanced import score_swing_point, calculate_setup_quality_score

# Track state
swing_highs = []
swing_lows = []
mss_events = []
detected_setups = []
pending_sweeps = []  # Track FVGs waiting for sweeps
trades = []

lookback = 3

print(f"   Processing {len(bars_list):,} bars...")

def detect_liquidity_sweep(fvg_top, fvg_bottom, direction, start_idx, bars_list, max_bars=10):
    """
    Detect if price sweeps the FVG and gets rejected.

    A liquidity sweep:
    1. Price enters the FVG zone
    2. Gets rejected (closes outside FVG)
    3. Shows rejection strength

    Returns: (swept, sweep_bar_index, rejection_strength)
    """
    end_idx = min(start_idx + max_bars, len(bars_list))

    for i in range(start_idx + 1, end_idx):
        bar = bars_list[i]

        if direction == 'bullish':
            # Bullish FVG: Looking for sweep downward into the gap
            # Price should drop into FVG, then reject upward
            if bar['low'] <= fvg_top and bar['low'] >= fvg_bottom:
                # Price swept into FVG
                # Check for rejection: close above FVG top
                if bar['close'] > fvg_top:
                    # Strong rejection - swept and rejected
                    rejection = bar['close'] - bar['low']
                    return True, i, rejection
                elif bar['high'] > fvg_top:
                    # Weak rejection - swept but weak close
                    rejection = bar['high'] - bar['close']
                    if rejection > 0:
                        return True, i, rejection
        else:  # bearish
            # Bearish FVG: Looking for sweep upward into the gap
            # Price should rise into FVG, then reject downward
            if bar['high'] >= fvg_bottom and bar['high'] <= fvg_top:
                # Price swept into FVG
                # Check for rejection: close below FVG bottom
                if bar['close'] < fvg_bottom:
                    # Strong rejection
                    rejection = bar['high'] - bar['close']
                    return True, i, rejection
                elif bar['low'] < fvg_bottom:
                    # Weak rejection
                    rejection = bar['close'] - bar['low']
                    if rejection > 0:
                        return True, i, rejection

    return False, None, 0.0

for i in tqdm(range(lookback, len(bars_list) - lookback)):
    current_bar = bars_list[i]

    # Detect swing points
    if i >= lookback and i < len(bars_list) - lookback:
        current_high = current_bar['high']
        is_swing_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and bars_list[j]['high'] >= current_high:
                is_swing_high = False
                break

        if is_swing_high:
            swing_strength = score_swing_point(
                [pd.Series(bar) for bar in bars_list[max(0, i-20):i+1]],
                len(bars_list[max(0, i-20):i+1]) - 1,
                'high'
            )
            swing_highs.append({
                'index': i,
                'timestamp': current_bar['timestamp'],
                'price': current_high,
                'strength': swing_strength
            })

        current_low = current_bar['low']
        is_swing_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and bars_list[j]['low'] <= current_low:
                is_swing_low = False
                break

        if is_swing_low:
            swing_strength = score_swing_point(
                [pd.Series(bar) for bar in bars_list[max(0, i-20):i+1]],
                len(bars_list[max(0, i-20):i+1]) - 1,
                'low'
            )
            swing_lows.append({
                'index': i,
                'timestamp': current_bar['timestamp'],
                'price': current_low,
                'strength': swing_strength
            })

        if len(swing_highs) > 50:
            swing_highs = swing_highs[-50:]
        if len(swing_lows) > 50:
            swing_lows = swing_lows[-50:]

    # Detect MSS events
    recent_bars = bars_list[max(0, i-20):i+1]
    avg_volume = sum(b['volume'] for b in recent_bars) / len(recent_bars)
    volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 0

    for swing_low in swing_lows[-5:]:
        if current_bar['low'] < swing_low['price']:
            if volume_ratio >= config['min_volume_ratio']:
                mss_events.append({
                    'index': i,
                    'timestamp': current_bar['timestamp'],
                    'direction': 'bullish',
                    'price': current_bar['low'],
                    'volume_ratio': volume_ratio,
                    'swing_point': swing_low,
                })
                break

    for swing_high in swing_highs[-5:]:
        if current_bar['high'] > swing_high['price']:
            if volume_ratio >= config['min_volume_ratio']:
                mss_events.append({
                    'index': i,
                    'timestamp': current_bar['timestamp'],
                    'direction': 'bearish',
                    'price': current_bar['high'],
                    'volume_ratio': volume_ratio,
                    'swing_point': swing_high,
                })
                break

    if len(mss_events) > 20:
        mss_events = mss_events[-20:]

    # Detect FVG and check for MSS confluence + SWEEP
    if i >= 2:
        candle_1 = bars_list[i - 2]
        candle_3 = bars_list[i]

        # Bullish FVG
        if candle_1['close'] > candle_3['open']:
            top = candle_1['high']
            bottom = candle_3['low']
            if top > bottom:
                gap_size_points = top - bottom
                gap_size_dollars = gap_size_points * 20.0

                if gap_size_dollars >= config['min_fvg_gap']:
                    for mss in mss_events:
                        if mss['direction'] == 'bearish':
                            bar_diff = abs(mss['index'] - i)

                            if bar_diff <= config['max_bar_distance']:
                                killzone_aligned, kz_window = is_within_trading_hours(
                                    current_bar['timestamp'], DEFAULT_TRADING_WINDOWS
                                )

                                quality_score = calculate_setup_quality_score({
                                    'fvg_size': gap_size_dollars,
                                    'volume_ratio': mss['volume_ratio'],
                                    'bar_diff': bar_diff,
                                    'killzone_aligned': killzone_aligned,
                                    'swing_strength': mss['swing_point']['strength']
                                })

                                if quality_score <= config['max_quality_score']:
                                    # CRITICAL: Check for liquidity sweep BEFORE entering
                                    swept, sweep_idx, rejection_strength = detect_liquidity_sweep(
                                        top, bottom, 'bullish', i, bars_list, config['max_sweep_bars']
                                    )

                                    if swept:
                                        # Entry AFTER sweep confirmed
                                        entry_bar = bars_list[sweep_idx]
                                        entry_price = (top + bottom) / 2
                                        swing_price = mss['swing_point']['price']

                                        if swing_price < entry_price:
                                            base_stop = swing_price
                                            stop_distance = abs(entry_price - base_stop) * config['stop_multiplier']
                                            optimized_stop = entry_price - stop_distance

                                            detected_setups.append({
                                                'index': sweep_idx,  # Entry at sweep bar
                                                'timestamp': entry_bar['timestamp'],
                                                'direction': 'bullish',
                                                'entry': entry_price,
                                                'stop': optimized_stop,
                                                'target': entry_price + (entry_price - optimized_stop) * 2,
                                                'quality_score': quality_score,
                                                'killzone_window': kz_window,
                                                'volume_ratio': mss['volume_ratio'],
                                                'fvg_size': gap_size_dollars,
                                                'bar_diff': bar_diff,
                                                'sweep_detected': True,
                                                'sweep_strength': rejection_strength,
                                            })
                                    break
                            break

        # Bearish FVG
        if candle_1['close'] < candle_3['open']:
            top = candle_3['high']
            bottom = candle_1['low']
            if top > bottom:
                gap_size_points = top - bottom
                gap_size_dollars = gap_size_points * 20.0

                if gap_size_dollars >= config['min_fvg_gap']:
                    for mss in mss_events:
                        if mss['direction'] == 'bullish':
                            bar_diff = abs(mss['index'] - i)

                            if bar_diff <= config['max_bar_distance']:
                                killzone_aligned, kz_window = is_within_trading_hours(
                                    current_bar['timestamp'], DEFAULT_TRADING_WINDOWS
                                )

                                quality_score = calculate_setup_quality_score({
                                    'fvg_size': gap_size_dollars,
                                    'volume_ratio': mss['volume_ratio'],
                                    'bar_diff': bar_diff,
                                    'killzone_aligned': killzone_aligned,
                                    'swing_strength': mss['swing_point']['strength']
                                })

                                if quality_score <= config['max_quality_score']:
                                    # CRITICAL: Check for liquidity sweep
                                    swept, sweep_idx, rejection_strength = detect_liquidity_sweep(
                                        top, bottom, 'bearish', i, bars_list, config['max_sweep_bars']
                                    )

                                    if swept:
                                        entry_bar = bars_list[sweep_idx]
                                        entry_price = (top + bottom) / 2
                                        swing_price = mss['swing_point']['price']

                                        if swing_price > entry_price:
                                            base_stop = swing_price
                                            stop_distance = abs(base_stop - entry_price) * config['stop_multiplier']
                                            optimized_stop = entry_price + stop_distance

                                            detected_setups.append({
                                                'index': sweep_idx,
                                                'timestamp': entry_bar['timestamp'],
                                                'direction': 'bearish',
                                                'entry': entry_price,
                                                'stop': optimized_stop,
                                                'target': entry_price - (optimized_stop - entry_price) * 2,
                                                'quality_score': quality_score,
                                                'killzone_window': kz_window,
                                                'volume_ratio': mss['volume_ratio'],
                                                'fvg_size': gap_size_dollars,
                                                'bar_diff': bar_diff,
                                                'sweep_detected': True,
                                                'sweep_strength': rejection_strength,
                                            })
                                    break
                            break

print(f"\n   Detected {len(detected_setups)} setups WITH SWEEPS")

# Simulate trades
print(f"\n🎯 Step 4: Simulating Trades...")

detected_setups_df = pd.DataFrame(detected_setups)
detected_setups_df['date'] = pd.to_datetime(detected_setups_df['timestamp']).dt.date
detected_setups_df['day_of_week'] = pd.to_datetime(detected_setups_df['timestamp']).dt.day_name

daily_trades = {}

for date, day_setups in tqdm(list(detected_setups_df.groupby('date'))):
    day_name = day_setups['day_of_week'].iloc[0]
    dow_multiplier = config['dow_multipliers'].get(day_name, 1.0)
    daily_limit = int(config['max_trades_per_day'] * dow_multiplier)

    day_setups_sorted = day_setups.sort_values('quality_score', ascending=False)
    selected = day_setups_sorted.head(daily_limit)

    for _, setup in selected.iterrows():
        entry_idx = setup['index']

        entry = setup['entry']
        stop = setup['stop']
        target = setup['target']
        direction = setup['direction']

        pnl = 0
        exit_reason = 'unknown'

        for j in range(entry_idx + 1, min(entry_idx + 100, len(bars_list))):
            future_bar = bars_list[j]

            if direction == 'bullish':
                if future_bar['low'] <= stop:
                    pnl = stop - entry
                    exit_reason = 'stop'
                    break
                if future_bar['high'] >= target:
                    pnl = target - entry
                    exit_reason = 'target'
                    break
            else:
                if future_bar['high'] >= stop:
                    pnl = entry - stop
                    exit_reason = 'stop'
                    break
                if future_bar['low'] <= target:
                    pnl = entry - target
                    exit_reason = 'target'
                    break
        else:
            timeout_idx = min(entry_idx + 100, len(bars_list) - 1)
            pnl = (bars_list[timeout_idx]['close'] - entry) if direction == 'bullish' else (entry - bars_list[timeout_idx]['close'])
            exit_reason = 'timeout'

        trade = {
            'entry_time': setup['timestamp'],
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'pnl': pnl * 20.0,
            'exit_reason': exit_reason,
            'quality_score': setup['quality_score'],
            'killzone_window': setup['killzone_window'],
            'fvg_size': setup['fvg_size'],
            'sweep_detected': setup.get('sweep_detected', False),
            'sweep_strength': setup.get('sweep_strength', 0),
        }
        trades.append(trade)

        if date not in daily_trades:
            daily_trades[date] = []
        daily_trades[date].append(trade)

        if len(daily_trades[date]) >= daily_limit:
            break

print(f"   Simulated {len(trades)} trades")

# Calculate metrics
trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("\n❌ No trades generated - sweep filter too strict")
    sys.exit(1)

print(f"\n📊 WITH SWEEPS - BACKTEST RESULTS:")
print("=" * 70)

trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
daily_counts = trades_df.groupby('date').size()

winning_trades = (trades_df['pnl'] > 0).sum()
win_rate = (winning_trades / len(trades_df) * 100)
total_return = trades_df['pnl'].sum()
avg_return_per_trade = trades_df['pnl'].mean()

winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
profit_factor = (winners / losers) if losers > 0 else 0

print(f"\nTrading Summary:")
print(f"   Setups Detected: {len(detected_setups)}")
print(f"   Trades Taken: {len(trades_df)}")
print(f"   Trading Days: {len(daily_counts)}")
print(f"   Trades/Day: {daily_counts.mean():.1f}")

print(f"\nPerformance Metrics:")
print(f"   Win Rate: {win_rate:.2f}%")
print(f"   Total Return: ${total_return:,.2f}")
print(f"   Avg Return/Trade: ${avg_return_per_trade:.2f}")
print(f"   Profit Factor: {profit_factor:.2f}")

print(f"\nSweep Analysis:")
print(f"   Trades with sweeps: {trades_df['sweep_detected'].sum()}")
print(f"   Avg sweep strength: {trades_df['sweep_strength'].mean():.2f} points")

print(f"\nExit Reason Breakdown:")
for reason in ['stop', 'target', 'timeout']:
    reason_trades = trades_df[trades_df['exit_reason'] == reason]
    if len(reason_trades) > 0:
        reason_win_rate = (reason_trades['pnl'] > 0).sum() / len(reason_trades) * 100
        print(f"   {reason.capitalize():10s}: {len(reason_trades):4d} trades ({len(reason_trades)/len(trades_df)*100:.1f}%) - {reason_win_rate:.1f}% win rate")

print(f"\n{'=' * 70}")
print("📊 COMPARISON: WITHOUT SWEEPS vs WITH SWEEPS")
print("=" * 70)
print(f"{'Metric':<25s} {'Without':>15s} {'With Sweeps':>15s} {'Change':>15s}")
print("-" * 70)
print(f"{'Total Trades':<25s} {'146':>15s} {f'{len(trades_df)}':>15s} {f'{len(trades_df) - 146:+d}':>15s}")
print(f"{'Win Rate':<25s} {'20.55%':>15s} {f'{win_rate:.2f}%':>15s} {f'{win_rate - 20.55:+.2f}%':>15s}")
print(f"{'Total Return':<25s} {'-$83,673':>15s} {f'${total_return:,.0f}':>15s} {f'${(total_return + 83673)/1000:+.1f}K':>15s}")
print(f"{'Profit Factor':<25s} {'0.30':>15s} {f'{profit_factor:.2f}':>15s} {f'{profit_factor - 0.30:+.2f}':>15s}")

# Save results
output_path = Path("data/reports/premium_backtest_with_sweeps.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
trades_df.to_csv(output_path, index=False)

print(f"\n💾 Results saved to: {output_path}")

print(f"\n" + "=" * 70)
if win_rate > 40 and profit_factor > 1.0:
    print("✅ SWEEPS SAVED THE STRATEGY!")
    print("   Liquidity sweeps are the critical missing component")
    print("   Win rate and profitability restored")
elif win_rate > 30:
    print("⚠️  IMPROVEMENT BUT NEEDS MORE WORK")
    print("   Sweeps help but may need parameter tuning")
else:
    print("❌ SWEEPS DIDN'T FIX IT")
    print("   May need different approach or strategy is flawed")
print("=" * 70)

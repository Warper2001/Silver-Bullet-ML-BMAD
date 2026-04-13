#!/usr/bin/env python3
"""
Test Dynamic Stop Placement - Fixed Break Structure

Root cause: Missing proper break statements in MSS loop
Solution: Break after finding MSS within max_bar_distance, then break MSS loop
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

print("🔧 DYNAMIC STOP PLACEMENT TEST (FIXED)")
print("=" * 70)
print("Testing asymmetric stop multipliers:")
print("  Bullish: 0.75x multiplier (tighter stops)")
print("  Bearish: 1.50x multiplier (standard)")
print("=" * 70)

# Load data
print(f"\n📊 Loading data...")
json_path = "/root/mnq_historical.json"
with open(json_path, 'r') as f:
    raw_data = json.load(f)

bars_list = []
for bar_data in tqdm(raw_data, desc="Converting"):
    try:
        timestamp = datetime.fromisoformat(bar_data.get('TimeStamp', '').replace('Z', '+00:00'))
        if timestamp.year != 2025:
            continue
        bars_list.append({
            'timestamp': timestamp,
            'open': float(bar_data.get('Open', 0)),
            'high': float(bar_data.get('High', 0)),
            'low': float(bar_data.get('Low', 0)),
            'close': float(bar_data.get('Close', 0)),
            'volume': int(bar_data.get('TotalVolume', 0)),
        })
    except:
        continue

print(f"✅ Loaded {len(bars_list)} bars")

from src.detection.time_window_filter import is_within_trading_hours, DEFAULT_TRADING_WINDOWS
from silver_bullet_premium_enhanced import score_swing_point, calculate_setup_quality_score

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

# Test dynamic stops
swing_highs = []
swing_lows = []
mss_events = []
detected_setups = []
lookback = 3

config = {
    'min_fvg_gap': 75.0,
    'min_volume_ratio': 2.0,
    'max_bar_distance': 7,
    'max_quality_score': 85.0,
    'max_sweep_bars': 10,
}

print(f"\n🎯 Running detection with DYNAMIC STOPS...")

for i in tqdm(range(lookback, len(bars_list) - lookback), desc="Processing"):
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
            swing_highs.append({'index': i, 'price': current_high, 'strength': swing_strength})

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
            swing_lows.append({'index': i, 'price': current_low, 'strength': swing_strength})

        if len(swing_highs) > 50:
            swing_highs = swing_highs[-50:]
        if len(swing_lows) > 50:
            swing_lows = swing_lows[-50:]

    # Detect MSS
    recent_bars = bars_list[max(0, i-20):i+1]
    avg_volume = sum(b['volume'] for b in recent_bars) / len(recent_bars)
    volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 0

    for swing_low in swing_lows[-5:]:
        if current_bar['low'] < swing_low['price'] and volume_ratio >= 2.0:
            mss_events.append({'index': i, 'direction': 'bullish', 'swing_point': swing_low})
            break

    for swing_high in swing_highs[-5:]:
        if current_bar['high'] > swing_high['price'] and volume_ratio >= 2.0:
            mss_events.append({'index': i, 'direction': 'bearish', 'swing_point': swing_high})
            break

    if len(mss_events) > 20:
        mss_events = mss_events[-20:]

    # Detect FVG + SWEEP + KILLZONE
    if i >= 2:
        candle_1 = bars_list[i - 2]
        candle_3 = bars_list[i]

        # Bullish FVG
        if candle_1['close'] > candle_3['open']:
            top = candle_1['high']
            bottom = candle_3['low']
            if top > bottom:
                gap_size_dollars = (top - bottom) * 20.0

                if gap_size_dollars >= 75.0:
                    for mss in mss_events:
                        if mss['direction'] == 'bearish' and abs(mss['index'] - i) <= 7:
                            killzone_aligned, kz_window = is_within_trading_hours(
                                current_bar['timestamp'], DEFAULT_TRADING_WINDOWS
                            )

                            if not killzone_aligned:
                                continue

                            quality_score = calculate_setup_quality_score({
                                'fvg_size': gap_size_dollars,
                                'volume_ratio': volume_ratio,
                                'bar_diff': abs(mss['index'] - i),
                                'killzone_aligned': killzone_aligned,
                                'swing_strength': mss['swing_point']['strength'] if 'strength' in mss['swing_point'] else 0.0
                            })

                            if quality_score > config['max_quality_score']:
                                continue

                            # Check for sweep using proper function
                            swept, sweep_idx, rejection_strength = detect_liquidity_sweep(
                                top, bottom, 'bullish', i, bars_list, config['max_sweep_bars']
                            )

                            if swept:
                                entry_bar = bars_list[sweep_idx]
                                entry_kz_aligned, _ = is_within_trading_hours(
                                    entry_bar['timestamp'], DEFAULT_TRADING_WINDOWS
                                )

                                if not entry_kz_aligned:
                                    continue

                                entry_price = (top + bottom) / 2
                                swing_price = mss['swing_point']['price']

                                if swing_price < entry_price:
                                    # DYNAMIC STOP: 0.75x for bullish (tighter)
                                    stop_distance = abs(entry_price - swing_price) * 0.75
                                    optimized_stop = entry_price - stop_distance

                                    detected_setups.append({
                                        'index': sweep_idx,
                                        'entry': entry_price,
                                        'stop': optimized_stop,
                                        'target': entry_price + (entry_price - optimized_stop) * 2,
                                        'direction': 'bullish',
                                        'stop_distance': stop_distance,
                                    })
                                break  # Break after finding MSS within max_bar_distance
                        break  # Break MSS loop

        # Bearish FVG
        if candle_1['close'] < candle_3['open']:
            top = candle_3['high']
            bottom = candle_1['low']
            if top > bottom:
                gap_size_dollars = (top - bottom) * 20.0

                if gap_size_dollars >= 75.0:
                    for mss in mss_events:
                        if mss['direction'] == 'bullish' and abs(mss['index'] - i) <= 7:
                            killzone_aligned, kz_window = is_within_trading_hours(
                                current_bar['timestamp'], DEFAULT_TRADING_WINDOWS
                            )

                            if not killzone_aligned:
                                continue

                            quality_score = calculate_setup_quality_score({
                                'fvg_size': gap_size_dollars,
                                'volume_ratio': volume_ratio,
                                'bar_diff': abs(mss['index'] - i),
                                'killzone_aligned': killzone_aligned,
                                'swing_strength': mss['swing_point']['strength'] if 'strength' in mss['swing_point'] else 0.0
                            })

                            if quality_score > config['max_quality_score']:
                                continue

                            # Check for sweep using proper function
                            swept, sweep_idx, rejection_strength = detect_liquidity_sweep(
                                top, bottom, 'bearish', i, bars_list, config['max_sweep_bars']
                            )

                            if swept:
                                entry_bar = bars_list[sweep_idx]
                                entry_kz_aligned, _ = is_within_trading_hours(
                                    entry_bar['timestamp'], DEFAULT_TRADING_WINDOWS
                                )

                                if not entry_kz_aligned:
                                    continue

                                entry_price = (top + bottom) / 2
                                swing_price = mss['swing_point']['price']

                                if swing_price > entry_price:
                                    # STANDARD STOP: 1.5x for bearish
                                    stop_distance = abs(swing_price - entry_price) * 1.5
                                    optimized_stop = entry_price + stop_distance

                                    detected_setups.append({
                                        'index': sweep_idx,
                                        'entry': entry_price,
                                        'stop': optimized_stop,
                                        'target': entry_price - (optimized_stop - entry_price) * 2,
                                        'direction': 'bearish',
                                        'stop_distance': stop_distance,
                                    })
                                break  # Break after finding MSS within max_bar_distance
                        break  # Break MSS loop

print(f"   Detected {len(detected_setups)} setups")

# Simulate trades
print(f"\n🎯 Simulating trades...")
trades = []

for setup in detected_setups:
    entry = setup['entry']
    stop = setup['stop']
    target = setup['target']
    direction = setup['direction']
    entry_idx = setup['index']
    stop_distance = setup['stop_distance']

    pnl = 0
    exit_reason = 'timeout'

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

    trades.append({
        'pnl': pnl * 20.0,
        'exit_reason': exit_reason,
        'stop_distance': stop_distance,
        'direction': direction,
    })

print(f"   Simulated {len(trades)} trades")

# Analysis
trades_df = pd.DataFrame(trades)

print(f"\n📊 DYNAMIC STOP RESULTS:")
print("="*70)

print(f"\nOverall:")
win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
total_return = trades_df['pnl'].sum()
profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).sum() < 0 else 0

print(f"   Trades: {len(trades_df)}")
print(f"   Win Rate: {win_rate:.1f}%")
print(f"   Total Return: ${total_return:,.0f}")
print(f"   Profit Factor: {profit_factor:.2f}")

print(f"\nBy Direction:")
for direction in ['bullish', 'bearish']:
    subset = trades_df[trades_df['direction'] == direction]
    if len(subset) > 0:
        d_win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
        d_pnl = subset['pnl'].sum()
        d_stop_dist = subset['stop_distance'].mean()
        d_stop_hits = (subset['exit_reason'] == 'stop').sum() / len(subset) * 100

        print(f"   {direction.capitalize():10s}: {len(subset):2d} trades, {d_win_rate:5.1f}% win, ${d_pnl:7,.0f}")
        print(f"                Avg stop: ${d_stop_dist:.0f}, Stop hits: {d_stop_hits:.1f}%")

print(f"\n{'='*70}")
print("🔍 COMPARISON: Dynamic vs Static Stops")
print("="*70)
print(f"{'Version':<20s} {'Bullish Stop':<15s} {'Bearish Stop':<15s} {'Win Rate':<12s} {'Return':<12s}")
print("-"*70)
print(f"{'Static 1.5x (both)':<20s} {'$62':<15s} {'$35':<15s} {'44.6%':<12s} {'-$19K':<12s}")
print(f"{'Dynamic (0.75x/1.5x)':<20s} {'~$35':<15s} {'$35':<15s} {f'{win_rate:.1f}%':<12s} {f'${total_return/1000:.1f}K':<12s}")

print(f"\n{'='*70}")
if profit_factor > 1.0:
    print("✅ DYNAMIC STOPS FIXED IT!")
    print("   Bullish stops now tighter, balancing risk")
elif win_rate > 45 and profit_factor > 0.8:
    print("⚠️  GETTING CLOSE - Dynamic stops help")
    print("   May need minor adjustments")
else:
    print("❌ Still needs work")
print("="*70)

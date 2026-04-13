#!/usr/bin/env python3
"""
Test wider stop losses on 1-minute data with sweeps + killzones
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

print("🔧 STOP LOSS OPTIMIZATION TEST")
print("=" * 70)
print("Testing different stop multipliers on 1-minute data")
print("Components: MSS + FVG + SWEEP + KILLZONE")
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

# Test different stop multipliers
stop_multipliers = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

results = []

for stop_mult in stop_multipliers:
    print(f"\n{'='*70}")
    print(f"Testing Stop Multiplier: {stop_mult}x")
    print(f"{'='*70}")

    # Detect setups (simplified for speed)
    swing_highs = []
    swing_lows = []
    mss_events = []
    detected_setups = []
    lookback = 3

    for i in tqdm(range(lookback, len(bars_list) - lookback), desc=f"Processing {stop_mult}x"):
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
                swing_highs.append({'index': i, 'price': current_high})

            current_low = current_bar['low']
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and bars_list[j]['low'] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append({'index': i, 'price': current_low})

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

                                # Check for sweep
                                swept = False
                                sweep_idx = None
                                for j in range(i + 1, min(i + 11, len(bars_list))):
                                    if bars_list[j]['low'] <= top and bars_list[j]['low'] >= bottom:
                                        if bars_list[j]['close'] > top:
                                            swept = True
                                            sweep_idx = j
                                            break

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
                                        stop_distance = abs(entry_price - swing_price) * stop_mult
                                        optimized_stop = entry_price - stop_distance

                                        detected_setups.append({
                                            'index': sweep_idx,
                                            'entry': entry_price,
                                            'stop': optimized_stop,
                                            'target': entry_price + (entry_price - optimized_stop) * 2,
                                            'direction': 'bullish',
                                        })
                                    break

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

                                # Check for sweep
                                swept = False
                                sweep_idx = None
                                for j in range(i + 1, min(i + 11, len(bars_list))):
                                    if bars_list[j]['high'] >= bottom and bars_list[j]['high'] <= top:
                                        if bars_list[j]['close'] < bottom:
                                            swept = True
                                            sweep_idx = j
                                            break

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
                                        stop_distance = abs(swing_price - entry_price) * stop_mult
                                        optimized_stop = entry_price + stop_distance

                                        detected_setups.append({
                                            'index': sweep_idx,
                                            'entry': entry_price,
                                            'stop': optimized_stop,
                                            'target': entry_price - (optimized_stop - entry_price) * 2,
                                            'direction': 'bearish',
                                        })
                                    break

    # Simulate trades
    trades = []
    for setup in detected_setups:
        entry = setup['entry']
        stop = setup['stop']
        target = setup['target']
        direction = setup['direction']
        entry_idx = setup['index']

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

        trades.append({
            'pnl': pnl * 20.0,
            'exit_reason': exit_reason,
            'stop_distance': abs(entry - stop),
        })

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
        total_return = trades_df['pnl'].sum()
        profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).sum() < 0 else 0
        stop_hit_rate = (trades_df['exit_reason'] == 'stop').sum() / len(trades_df) * 100
        avg_stop_distance = trades_df['stop_distance'].mean()

        results.append({
            'stop_multiplier': stop_mult,
            'trades': len(trades_df),
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'stop_hit_rate': stop_hit_rate,
            'avg_stop_distance': avg_stop_distance,
        })

        print(f"Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: ${total_return:,.0f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Stop Hit Rate: {stop_hit_rate:.1f}%")
        print(f"Avg Stop Distance: ${avg_stop_distance:.0f}")

# Summary
print(f"\n{'='*70}")
print("📊 STOP MULTIPLIER COMPARISON")
print(f"{'='*70}")
print(f"{'Multiplier':<12s} {'Trades':>10s} {'Win Rate':>12s} {'Return':>15s} {'Stop Hit':>12s} {'Avg Stop':>12s}")
print("-"*70)

for r in results:
    print(f"{r['stop_multiplier']:<12.1f}x {r['trades']:>10} {r['win_rate']:>11.1f}% {r['total_return']:>15,.0f} {r['stop_hit_rate']:>11.1f}% ${r['avg_stop_distance']:>10.0f}")

print(f"\n{'='*70}")
print("✅ FINDINGS:")
print("-"*70)

# Find best result
best = max(results, key=lambda x: x['profit_factor'])
print(f"Best Stop Multiplier: {best['stop_multiplier']:.1f}x")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Profit Factor: {best['profit_factor']:.2f}")
print(f"  Total Return: ${best['total_return']:,.0f}")

if best['profit_factor'] > 1.0:
    print(f"\n✅ PROFITABLE STOP LOSS FOUND!")
elif best['profit_factor'] > 0.7:
    print(f"\n⚠️  Getting close - may need minor adjustments")
else:
    print(f"\n❌ Stop widening alone doesn't fix it")

#!/usr/bin/env python3
"""
Balanced Optimization for Premium Strategy

Applies diagnostic insights with balanced filtering:
- Inverted quality score (60-85 range)
- Wider stops (1.5x multiplier)
- No max FVG size constraint (let reasonable sizes through)
- Focus on generating sufficient sample size for validation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

print("🚀 SILVER BULLET PREMIUM - BALANCED OPTIMIZATION")
print("=" * 70)
print("Method: Diagnostic-based balanced optimization")
print("Data: Real MNQ 2025 market data")
print("=" * 70)

# Load raw OHLCV data
print(f"\n📊 Step 1: Loading raw 2025 MNQ data...")

import h5py
data_dir = Path("data/processed/time_bars/")
h5_files = sorted(data_dir.glob("MNQ_time_bars_5min_2025*.h5"))

print(f"Found {len(h5_files)} monthly files")

all_bars = []
for h5_file in h5_files:
    try:
        with h5py.File(h5_file, 'r') as f:
            data = f['dollar_bars'][:]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        all_bars.append(df)
        print(f"  ✅ {h5_file.name}: {len(df)} bars")
    except Exception as e:
        print(f"  ❌ {h5_file.name}: {e}")

if not all_bars:
    print("❌ No data loaded!")
    sys.exit(1)

combined = pd.concat(all_bars, ignore_index=True)
combined = combined.sort_values('timestamp').reset_index(drop=True)

print(f"\n✅ Loaded {len(combined):,} total bars")
print(f"   Period: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

bars_list = []
for idx, row in combined.iterrows():
    bars_list.append({
        'timestamp': row['timestamp'],
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': int(row['volume']),
    })

print(f"   Converted to list for processing")

# Run balanced premium strategy detection
print(f"\n🎯 Step 2: Running Premium Pattern Detection (BALANCED)...")

# BALANCED Configuration
config = {
    'min_fvg_gap': 75.0,           # Original minimum (NO MAX constraint)
    'min_volume_ratio': 2.0,
    'max_bar_distance': 7,
    'min_quality_score': 60.0,     # INVERTED: Lower minimum
    'max_quality_score': 85.0,     # Wider range for sample size
    'stop_multiplier': 1.5,        # MODERATE: Wider stops
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
}

print(f"\n🔧 BALANCED PARAMETERS:")
print(f"   FVG Size: ${config['min_fvg_gap']}+ (no maximum constraint)")
print(f"   Quality Score: {config['min_quality_score']}-{config['max_quality_score']} (inverted)")
print(f"   Stop Multiplier: {config['stop_multiplier']}x (wider than original)")
print(f"   Goal: Balance optimization with sufficient sample size")

from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg
from src.detection.time_window_filter import is_within_trading_hours, DEFAULT_TRADING_WINDOWS
from silver_bullet_premium_enhanced import score_swing_point, calculate_setup_quality_score

# Track state
swing_highs = []
swing_lows = []
mss_events = []
detected_setups = []
trades = []

lookback = 3

print(f"   Processing {len(bars_list):,} bars...")

for i in range(lookback, len(bars_list) - lookback):
    if i % 1000 == 0:
        print(f"   Progress: {i}/{len(bars_list)} bars ({i/len(bars_list)*100:.1f}%)")

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

    # Detect FVG with BALANCED filter (no max constraint)
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

                # BALANCED: Only minimum FVG size (no maximum constraint)
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

                                # BALANCED: Wider quality range (60-85)
                                if config['min_quality_score'] <= quality_score <= config['max_quality_score']:
                                    entry_price = (top + bottom) / 2
                                    swing_price = mss['swing_point']['price']

                                    if swing_price < entry_price:
                                        base_stop = swing_price
                                        stop_distance = abs(entry_price - base_stop) * config['stop_multiplier']
                                        optimized_stop = entry_price - stop_distance

                                        detected_setups.append({
                                            'index': i,
                                            'timestamp': current_bar['timestamp'],
                                            'direction': 'bullish',
                                            'entry': entry_price,
                                            'stop': optimized_stop,
                                            'target': entry_price + (entry_price - optimized_stop) * 2,
                                            'quality_score': quality_score,
                                            'killzone_window': kz_window,
                                            'volume_ratio': mss['volume_ratio'],
                                            'fvg_size': gap_size_dollars,
                                            'bar_diff': bar_diff,
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

                                if config['min_quality_score'] <= quality_score <= config['max_quality_score']:
                                    entry_price = (top + bottom) / 2
                                    swing_price = mss['swing_point']['price']

                                    if swing_price > entry_price:
                                        base_stop = swing_price
                                        stop_distance = abs(base_stop - entry_price) * config['stop_multiplier']
                                        optimized_stop = entry_price + stop_distance

                                        detected_setups.append({
                                            'index': i,
                                            'timestamp': current_bar['timestamp'],
                                            'direction': 'bearish',
                                            'entry': entry_price,
                                            'stop': optimized_stop,
                                            'target': entry_price - (optimized_stop - entry_price) * 2,
                                            'quality_score': quality_score,
                                            'killzone_window': kz_window,
                                            'volume_ratio': mss['volume_ratio'],
                                            'fvg_size': gap_size_dollars,
                                            'bar_diff': bar_diff,
                                        })
                                    break
                            break

print(f"\n   Detected {len(detected_setups)} balanced premium setups")

# Simulate trades
print(f"\n🎯 Step 3: Simulating Trades...")

detected_setups_df = pd.DataFrame(detected_setups)
detected_setups_df['date'] = pd.to_datetime(detected_setups_df['timestamp']).dt.date
detected_setups_df['day_of_week'] = pd.to_datetime(detected_setups_df['timestamp']).dt.day_name

daily_trades = {}

for date, day_setups in detected_setups_df.groupby('date'):
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
    print("\n❌ No trades generated - filters may still be too strict")
    sys.exit(1)

print(f"\n📊 BALANCED OPTIMIZATION RESULTS:")
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
print(f"   Trades/Day: {daily_counts.mean():.1f} (range: {daily_counts.min()} - {daily_counts.max()})")

print(f"\nPerformance Metrics:")
print(f"   Win Rate: {win_rate:.2f}%")
print(f"   Total Return: ${total_return:,.2f}")
print(f"   Avg Return/Trade: ${avg_return_per_trade:.2f}")
print(f"   Profit Factor: {profit_factor:.2f}")

print(f"\nQuality Analysis:")
print(f"   Avg Quality Score: {trades_df['quality_score'].mean():.1f}/100")
print(f"   Avg FVG Size: ${trades_df['fvg_size'].mean():.0f}")

print(f"\nExit Reason Breakdown:")
for reason in ['stop', 'target', 'timeout']:
    reason_trades = trades_df[trades_df['exit_reason'] == reason]
    if len(reason_trades) > 0:
        reason_win_rate = (reason_trades['pnl'] > 0).sum() / len(reason_trades) * 100
        print(f"   {reason.capitalize():10s}: {len(reason_trades):4d} trades ({len(reason_trades)/len(trades_df)*100:.1f}%) - {reason_win_rate:.1f}% win rate")

print(f"\n{'=' * 70}")
print("🎯 OPTIMIZATION COMPARISON")
print("=" * 70)
print(f"{'Metric':<20s} {'Original':>15s} {'Balanced':>15s} {'Change':>15s}")
print("-" * 70)
print(f"{'Win Rate':<20s} {'19.89%':>15s} {f'{win_rate:.2f}%':>15s} {f'{win_rate - 19.89:+.2f}%':>15s}")
print(f"{'Total Return':<20s} {'-$1,249,000':>15s} {f'${total_return:,.0f}':>15s} {f'${(total_return + 1249000)/1000:+.0f}K':>15s}")
print(f"{'Profit Factor':<20s} {'0.37':>15s} {f'{profit_factor:.2f}':>15s} {f'{profit_factor - 0.37:+.2f}':>15s}")
print(f"{'Sample Size':<20s} {'2353':>15s} {f'{len(trades_df)}':>15s} {f'{len(trades_df) - 2353:+d}':>15s}")

# Success assessment
print(f"\n{'=' * 70}")
if win_rate > 35 and profit_factor > 1.0 and total_return > 0 and len(trades_df) > 100:
    print("✅ BALANCED OPTIMIZATION SUCCESSFUL!")
    print("   Strategy shows promise with sufficient sample size")
    print("   Ready for ML training data generation")
elif win_rate > 25 and profit_factor > 0.7 and len(trades_df) > 50:
    print("⚠️  OPTIMIZATION IMPROVED - Strategy shows potential")
    print("   Sample size adequate for further analysis")
    print("   Consider fine-tuning parameters")
else:
    print("❌ OPTIMIZATION INSUFFICIENT")
    print("   Strategy may require different approach")

print("=" * 70)

# Save results
output_path = Path("data/reports/premium_backtest_balanced.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
trades_df.to_csv(output_path, index=False)

summary_path = Path("data/reports/premium_backtest_balanced_summary.json")
with open(summary_path, 'w') as f:
    json.dump({
        'setups_detected': len(detected_setups),
        'trades_taken': len(trades_df),
        'win_rate': float(win_rate),
        'total_return': float(total_return),
        'avg_trades_per_day': float(daily_counts.mean()),
        'min_trades_per_day': int(daily_counts.min()),
        'max_trades_per_day': int(daily_counts.max()),
        'trading_days': int(len(daily_counts)),
        'profit_factor': float(profit_factor),
        'avg_quality_score': float(trades_df['quality_score'].mean()),
        'avg_fvg_size': float(trades_df['fvg_size'].mean()),
        'optimization_params': {
            'fvg_size': f"${config['min_fvg_gap']}+ (no max)",
            'quality_range': f"{config['min_quality_score']}-{config['max_quality_score']}",
            'stop_multiplier': config['stop_multiplier']
        }
    }, f, indent=2, default=str)

print(f"\n💾 Results saved:")
print(f"   Trades: {output_path}")
print(f"   Summary: {summary_path}")

print(f"\n" + "=" * 70)
print("✅ BALANCED OPTIMIZATION COMPLETE")
print("=" * 70)

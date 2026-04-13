#!/usr/bin/env python3
"""
Proper Backtest Validation for Silver Bullet Premium Enhanced

This runs the actual premium strategy on historical data (not simulation).
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester
from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg
from src.detection.swing_detection import (
    detect_swing_high, detect_swing_low,
    score_swing_point
)
from src.detection.time_window_filter import is_within_trading_hours, DEFAULT_TRADING_WINDOWS


def load_time_bars(date_start: str, date_end: str) -> pd.DataFrame:
    """Load time-based bars for backtesting."""
    print(f"📊 Loading time bars from {date_start} to {date_end}...")

    import h5py
    data_dir = Path("data/processed/time_bars/")

    start_dt = pd.Timestamp(date_start)
    end_dt = pd.Timestamp(date_end)
    current = start_dt.replace(day=1)

    files = []
    while current <= end_dt:
        filename = f"MNQ_time_bars_5min_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename
        if file_path.exists():
            files.append(file_path)
        current = current + pd.DateOffset(months=1)

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

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) & (combined.index <= end_dt)
    ]

    print(f"✅ Loaded {len(combined):,} time bars")
    print(f"   Period: {combined.index.min().date()} to {combined.index.max().date()}")

    return combined


def calculate_setup_quality_score(
    fvg_size: float,
    volume_ratio: float,
    bar_diff: int,
    killzone_aligned: bool,
    swing_strength: float
) -> float:
    """Calculate overall setup quality (0-100)."""

    # FVG score
    fvg_score = min(100, (fvg_size / 200.0) * 100)

    # MSS score
    mss_score = min(100, volume_ratio * 40)

    # Alignment score
    alignment_score = max(0, 100 - (bar_diff * 10))

    # Killzone score
    killzone_score = 100 if killzone_aligned else 50

    # Swing strength score
    swing_score = swing_strength

    # Combine scores
    total_score = (
        fvg_score * 0.25 +
        mss_score * 0.25 +
        alignment_score * 0.20 +
        killzone_score * 0.15 +
        swing_score * 0.15
    )

    return min(100.0, max(0.0, total_score))


def run_premium_backtest(
    data: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """Run premium strategy backtest on historical data."""

    print(f"\n🎯 Running Premium Strategy Backtest...")
    print(f"   Min FVG Gap: ${config['min_fvg_gap']}")
    print(f"   Min Volume Ratio: {config['min_volume_ratio']}x")
    print(f"   Max Bar Distance: {config['max_bar_distance']}")
    print(f"   Min Quality Score: {config['min_quality_score']}")

    trades = []
    signals_detected = 0

    # Convert data to list for faster indexing
    bars_list = []

    for idx, row in data.iterrows():
        bar = {
            'timestamp': idx,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': int(row['volume']),
        }
        bars_list.append(bar)

    print(f"   Processing {len(bars_list):,} bars...")

    # Track swing points, MSS, FVG
    swing_highs = []
    swing_lows = []
    mss_events = []

    lookback = 3

    for i in range(lookback, len(bars_list) - lookback):
        current_bar = bars_list[i]

        # Detect swing points
        is_swing_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and bars_list[j]['high'] >= current_bar['high']:
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
                'price': current_bar['high'],
                'strength': swing_strength
            })

        # Keep only recent swing highs
        if len(swing_highs) > 50:
            swing_highs = swing_highs[-50:]

        # Detect swing lows
        is_swing_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and bars_list[j]['low'] <= current_bar['low']:
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
                'price': current_bar['low'],
                'strength': swing_strength
            })

        # Keep only recent swing lows
        if len(swing_lows) > 50:
            swing_lows = swing_lows[-50:]

        # Detect MSS events
        recent_bars = bars_list[max(0, i-20):i+1]
        avg_volume = sum(b['volume'] for b in recent_bars) / len(recent_bars)
        volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 0

        # Check for bullish MSS
        for swing_high in swing_highs[-5:]:
            if current_bar['high'] > swing_high['price']:
                if volume_ratio >= config['min_volume_ratio']:
                    mss_events.append({
                        'index': i,
                        'direction': 'bullish',
                        'price': current_bar['high'],
                        'volume_ratio': volume_ratio,
                        'swing_point': swing_high,
                    })
                    break

        # Keep only recent MSS events
        if len(mss_events) > 20:
            mss_events = mss_events[-20:]

        # Check for bearish MSS
        for swing_low in swing_lows[-5:]:
            if current_bar['low'] < swing_low['price']:
                if volume_ratio >= config['min_volume_ratio']:
                    mss_events.append({
                        'index': i,
                        'direction': 'bearish',
                        'price': current_bar['low'],
                        'volume_ratio': volume_ratio,
                        'swing_point': swing_low,
                    })
                    break

        # Detect FVG
        if i >= 2:
            # Bullish FVG
            candle_1 = bars_list[i - 2]
            candle_3 = bars_list[i]

            if candle_1['close'] > candle_3['open']:
                top = candle_1['high']
                bottom = candle_3['low']

                if top > bottom:
                    gap_size_points = top - bottom
                    gap_size_dollars = gap_size_points * 20.0  # MNQ is $20/point

                    if gap_size_dollars >= config['min_fvg_gap']:
                        # Check for confluence with MSS
                        for mss in mss_events:
                            if mss['direction'] == 'bullish':
                                bar_diff = abs(mss['index'] - i)

                                if bar_diff <= config['max_bar_distance']:
                                    # Check killzone alignment
                                    timestamp = current_bar['timestamp']
                                    killzone_aligned, kz_window = is_within_trading_hours(
                                        timestamp, DEFAULT_TRADING_WINDOWS
                                    )

                                    # Calculate quality score
                                    quality_score = calculate_setup_quality_score(
                                        fvg_size=gap_size_dollars,
                                        volume_ratio=mss['volume_ratio'],
                                        bar_diff=bar_diff,
                                        killzone_aligned=killzone_aligned,
                                        swing_strength=mss['swing_point']['strength']
                                    )

                                    # Apply quality threshold
                                    if quality_score >= config['min_quality_score']:
                                        signals_detected += 1

                                        # Simulate trade
                                        entry = (top + bottom) / 2
                                        stop = mss['swing_point']['price']
                                        target = entry + (entry - stop) * 2

                                        # Simulate outcome
                                        entry_idx = i
                                        pnl = 0
                                        exit_reason = 'unknown'

                                        for j in range(entry_idx + 1, min(entry_idx + 100, len(bars_list))):
                                            future_bar = bars_list[j]

                                            # Check stop
                                            if future_bar['low'] <= stop:
                                                pnl = stop - entry
                                                exit_reason = 'stop'
                                                break

                                            # Check target
                                            if future_bar['high'] >= target:
                                                pnl = target - entry
                                                exit_reason = 'target'
                                                break

                                        else:
                                            # Timeout
                                            pnl = bars_list[min(entry_idx + 100, len(bars_list) - 1)]['close'] - entry
                                            exit_reason = 'timeout'

                                        trades.append({
                                            'entry_time': current_bar['timestamp'],
                                            'direction': 'bullish',
                                            'entry': entry,
                                            'stop': stop,
                                            'target': target,
                                            'pnl': pnl * 0.5,  # Convert points to dollars
                                            'exit_reason': exit_reason,
                                            'quality_score': quality_score,
                                            'fvg_size': gap_size_dollars,
                                            'volume_ratio': mss['volume_ratio'],
                                        })
                                        break

            # Bearish FVG
            if candle_1['close'] < candle_3['open']:
                top = candle_3['high']
                bottom = candle_1['low']

                if top > bottom:
                    gap_size_points = top - bottom
                    gap_size_dollars = gap_size_points * 20.0

                    if gap_size_dollars >= config['min_fvg_gap']:
                        # Check for confluence with MSS
                        for mss in mss_events:
                            if mss['direction'] == 'bearish':
                                bar_diff = abs(mss['index'] - i)

                                if bar_diff <= config['max_bar_distance']:
                                    # Check killzone alignment
                                    timestamp = current_bar['timestamp']
                                    killzone_aligned, kz_window = is_within_trading_hours(
                                        timestamp, DEFAULT_TRADING_WINDOWS
                                    )

                                    # Calculate quality score
                                    quality_score = calculate_setup_quality_score(
                                        fvg_size=gap_size_dollars,
                                        volume_ratio=mss['volume_ratio'],
                                        bar_diff=bar_diff,
                                        killzone_aligned=killzone_aligned,
                                        swing_strength=mss['swing_point']['strength']
                                    )

                                    # Apply quality threshold
                                    if quality_score >= config['min_quality_score']:
                                        signals_detected += 1

                                        # Simulate trade
                                        entry = (top + bottom) / 2
                                        stop = mss['swing_point']['price']
                                        target = entry - (stop - entry) * 2

                                        # Simulate outcome
                                        entry_idx = i
                                        pnl = 0
                                        exit_reason = 'unknown'

                                        for j in range(entry_idx + 1, min(entry_idx + 100, len(bars_list))):
                                            future_bar = bars_list[j]

                                            # Check stop
                                            if future_bar['high'] >= stop:
                                                pnl = entry - stop
                                                exit_reason = 'stop'
                                                break

                                            # Check target
                                            if future_bar['low'] <= target:
                                                pnl = entry - target
                                                exit_reason = 'target'
                                                break

                                        else:
                                            # Timeout
                                            pnl = entry - bars_list[min(entry_idx + 100, len(bars_list) - 1)]['close']
                                            exit_reason = 'timeout'

                                        trades.append({
                                            'entry_time': current_bar['timestamp'],
                                            'direction': 'bearish',
                                            'entry': entry,
                                            'stop': stop,
                                            'target': target,
                                            'pnl': pnl * 0.5,
                                            'exit_reason': exit_reason,
                                            'quality_score': quality_score,
                                            'fvg_size': gap_size_dollars,
                                            'volume_ratio': mss['volume_ratio'],
                                        })
                                        break

    print(f"   Signals detected: {signals_detected}")
    print(f"   Trades simulated: {len(trades)}")

    return pd.DataFrame(trades)


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""

    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_trades_per_day': 0,
            'max_trades_per_day': 0,
            'min_trades_per_day': 0,
            'trading_days': 0,
            'profit_factor': 0
        }

    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_counts = trades_df.groupby('date').size()

    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = (winning_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
    total_return = trades_df['pnl'].sum()

    # Profit factor
    winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = (winners / losers) if losers > 0 else 0

    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_trades_per_day': float(daily_counts.mean()) if len(daily_counts) > 0 else 0,
        'max_trades_per_day': int(daily_counts.max()) if len(daily_counts) > 0 else 0,
        'min_trades_per_day': int(daily_counts.min()) if len(daily_counts) > 0 else 0,
        'trading_days': len(daily_counts),
        'profit_factor': profit_factor
    }


def main():
    """Run year-long backtest validation."""

    parser = argparse.ArgumentParser(description='Validate premium strategy with proper backtest')
    parser.add_argument('--year', type=int, default=2025, help='Year to backtest (default: 2025)')
    parser.add_argument('--min-fvg-gap', type=float, default=75.0, help='Min FVG gap in dollars')
    parser.add_argument('--min-volume-ratio', type=float, default=2.0, help='Min volume ratio')
    parser.add_argument('--max-bar-distance', type=int, default=7, help='Max bar distance')
    parser.add_argument('--min-quality-score', type=float, default=70.0, help='Min quality score')
    parser.add_argument('--output', '-o', help='Output CSV path')

    args = parser.parse_args()

    print("🚀 SILVER BULLET PREMIUM - YEAR LONG BACKTEST")
    print("=" * 70)
    print(f"Year: {args.year}")
    print(f"Period: Jan 1 - Dec 31")

    # Configuration
    config = {
        'min_fvg_gap': args.min_fvg_gap,
        'min_volume_ratio': args.min_volume_ratio,
        'max_bar_distance': args.max_bar_distance,
        'min_quality_score': args.min_quality_score,
    }

    # Load data
    date_start = f"{args.year}-01-01"
    date_end = f"{args.year}-12-31"

    data = load_time_bars(date_start, date_end)

    if data.empty:
        print("❌ No data available!")
        return

    # Run backtest
    trades_df = run_premium_backtest(data, config)

    if len(trades_df) == 0:
        print("\n❌ No trades generated!")
        print("   Filters may be too strict or data insufficient")
        return

    # Calculate metrics
    metrics = calculate_metrics(trades_df)

    # Print results
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nTrading Summary:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Trading Days: {metrics['trading_days']}")
    print(f"   Trades/Day: {metrics['avg_trades_per_day']:.1f} (range: {metrics['min_trades_per_day']}-{metrics['max_trades_per_day']})")

    print(f"\nPerformance Metrics:")
    print(f"   Win Rate: {metrics['win_rate']:.2f}%")
    print(f"   Total Return: ${metrics['total_return']:,.2f}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

    if metrics['total_trades'] > 0:
        print(f"   Avg Return/Trade: ${metrics['total_return']/metrics['total_trades']:.2f}")

    # Validate against targets
    print(f"\n{'=' * 70}")
    print("✅ VALIDATION AGAINST TARGETS")
    print("=" * 70)

    target_min_trades = 1
    target_max_trades = 20
    target_win_rate = 84.82  # Baseline

    trades_in_range = (target_min_trades <= metrics['avg_trades_per_day'] <= target_max_trades)
    win_rate_passed = (metrics['win_rate'] >= target_win_rate)

    print(f"\nTrades/Day Target: {target_min_trades}-{target_max_trades}")
    print(f"   Actual: {metrics['avg_trades_per_day']:.1f}")
    print(f"   Status: {'✅ PASSED' if trades_in_range else '❌ FAILED'}")

    print(f"\nWin Rate Target: ≥{target_win_rate}%")
    print(f"   Actual: {metrics['win_rate']:.2f}%")
    print(f"   Status: {'✅ PASSED' if win_rate_passed else '❌ FAILED'}")

    # Quality score analysis
    print(f"\nQuality Score Analysis:")
    print(f"   Avg Quality Score: {trades_df['quality_score'].mean():.1f}/100")
    print(f"   Min Quality Score: {trades_df['quality_score'].min():.1f}/100")
    print(f"   Max Quality Score: {trades_df['quality_score'].max():.1f}/100")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trades_df.to_csv(output_path, index=False)

        # Save summary
        summary_path = output_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'config': config,
                'metrics': metrics,
                'validation': {
                    'trades_in_range': trades_in_range,
                    'win_rate_passed': win_rate_passed
                }
            }, f, indent=2, default=str)

        print(f"\n💾 Results saved to {output_path}")
        print(f"   Summary: {summary_path}")

    # Overall assessment
    print(f"\n{'=' * 70}")
    if trades_in_range and win_rate_passed:
        print("✅ ALL VALIDATION CRITERIA PASSED")
        print("   Strategy is ready for deployment!")
    else:
        print("❌ SOME VALIDATION CRITERIA FAILED")
        print("   Consider adjusting configuration parameters")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Single Period Validation for TIER 1 FVG System.

Tests the optimal configuration on a single time period.
Run this for each period separately to avoid memory issues.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import DollarBar, FVGEvent, GapRange

# Constants
MNQ_DATA_PATH = Path("/root/mnq_historical.json")
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE
DOLLAR_BAR_THRESHOLD = 50_000_000
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1

# OPTIMAL CONFIGURATION
SL_MULTIPLIER = 2.5
ATR_THRESHOLD = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS = 50.0
CONTRACTS_PER_TRADE = 1
MAX_HOLD_BARS = 10


def main():
    """Validate single period."""
    if len(sys.argv) < 3:
        print("Usage: python validate_single_period.py <start_idx> <end_idx> <period_name>")
        print("Example: python validate_single_period.py 775296 795296 'Dec 2025'")
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    period_name = sys.argv[3]

    print("=" * 80)
    print(f"SINGLE PERIOD VALIDATION - {period_name}")
    print("=" * 80)
    print(f"Bars {start_idx:,} to {end_idx:,}")
    print(f"Configuration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}")

    # Load data
    print(f"\nLoading data...")
    with open(MNQ_DATA_PATH, 'r') as f:
        data = json.loads(f.read())
        data = data[start_idx:end_idx]

    print(f"✓ Loaded {len(data):,} raw bars")

    # Transform to DataFrame
    df = pd.DataFrame(data)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    for col in ['High', 'Low', 'Open', 'Close', 'TotalVolume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    print(f"✓ Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

    # Transform to Dollar Bars
    print(f"\nTransforming to Dollar Bars...")
    df['notional'] = ((df['High'] + df['Low']) / 2) * df['TotalVolume'] * MNQ_POINT_VALUE
    df['cumulative_notional'] = df['notional'].cumsum()

    bar_boundaries = df[df['cumulative_notional'] % DOLLAR_BAR_THRESHOLD < df['notional']].index.tolist()
    if len(df) > 0 and (len(bar_boundaries) == 0 or bar_boundaries[-1] != len(df) - 1):
        bar_boundaries.append(len(df) - 1)

    dollar_bars = []
    prev_boundary = 0

    for boundary in bar_boundaries:
        if boundary == 0:
            continue
        segment = df.iloc[prev_boundary:boundary+1]
        if len(segment) == 0:
            continue

        capped_notional = min(float(segment['notional'].sum()), 1_500_000_000)
        dollar_bar = DollarBar(
            timestamp=segment.iloc[0]['TimeStamp'],
            open=float(segment.iloc[0]['Open']),
            high=float(segment['High'].max()),
            low=float(segment['Low'].min()),
            close=float(segment.iloc[-1]['Close']),
            volume=int(segment['TotalVolume'].sum()),
            notional_value=capped_notional,
            is_forward_filled=False,
        )
        dollar_bars.append(dollar_bar)
        prev_boundary = boundary + 1

    print(f"✓ Created {len(dollar_bars)} Dollar Bars")

    # Pre-calculate indicators
    print(f"\nCalculating indicators...")
    df_bars = pd.DataFrame([
        {'high': bar.high, 'low': bar.low, 'close': bar.close, 'volume': bar.volume}
        for bar in dollar_bars
    ])

    df_bars['prev_close'] = df_bars['close'].shift(1)
    df_bars['tr1'] = df_bars['high'] - df_bars['low']
    df_bars['tr2'] = abs(df_bars['high'] - df_bars['prev_close'])
    df_bars['tr3'] = abs(df_bars['low'] - df_bars['prev_close'])
    df_bars['true_range'] = df_bars[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr_values = df_bars['true_range'].ewm(span=14, adjust=False).mean().values
    mean_tr = df_bars['true_range'].mean()
    atr_values = np.nan_to_num(atr_values, nan=mean_tr)

    is_bullish = np.array([1 if bar.close > bar.open else 0 for bar in dollar_bars])
    is_bearish = np.array([1 if bar.close < bar.open else 0 for bar in dollar_bars])
    volumes = np.array([bar.volume for bar in dollar_bars])
    up_volumes = pd.Series(volumes * is_bullish).rolling(window=20, min_periods=1).sum().values
    down_volumes = pd.Series(volumes * is_bearish).rolling(window=20, min_periods=1).sum().values

    print(f"✓ ATR and volume ratios calculated")

    # Detect and simulate trades
    print(f"\nRunning backtest...")
    trades = []

    for i in range(2, len(dollar_bars)):
        if i % 1000 == 0:
            print(f"  Processing bar {i}/{len(dollar_bars)} ({100*i/len(dollar_bars):.1f}%)...")

        # Detect bullish FVG
        candle_1 = dollar_bars[i-2]
        candle_3 = dollar_bars[i]

        if candle_1.close > candle_3.open:  # Bullish FVG
            gap_bottom = candle_3.low
            gap_top = candle_1.high
            gap_size = gap_top - gap_bottom

            if gap_top > gap_bottom:
                atr = atr_values[i]
                if gap_size >= (atr * ATR_THRESHOLD):
                    gap_dollars = gap_size * MNQ_CONTRACT_VALUE
                    if gap_dollars <= MAX_GAP_DOLLARS:
                        up_volume = up_volumes[i]
                        down_volume = down_volumes[i]
                        volume_ratio = (up_volume / down_volume) if down_volume > 0 else float('inf')

                        if volume_ratio >= VOLUME_RATIO_THRESHOLD:
                            # Simulate trade
                            entry_price = gap_bottom
                            take_profit = gap_top
                            stop_loss = gap_bottom - (gap_size * SL_MULTIPLIER)

                            if stop_loss > 0 and take_profit > 0:
                                entry_bar_index = i + 1
                                if entry_bar_index < len(dollar_bars):
                                    exit_price = None
                                    exit_reason = None

                                    sl_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
                                    tp_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
                                    sl_trigger = stop_loss - sl_buffer
                                    tp_trigger = take_profit + tp_buffer

                                    max_index = min(entry_bar_index + MAX_HOLD_BARS + 1, len(dollar_bars))

                                    for j in range(entry_bar_index + 1, max_index):
                                        bar = dollar_bars[j]
                                        if bar.low <= sl_trigger:
                                            exit_price = min(stop_loss, bar.low + sl_buffer)
                                            exit_reason = "stop_loss"
                                            break
                                        if bar.high >= tp_trigger:
                                            exit_price = max(take_profit, bar.high - tp_buffer)
                                            exit_reason = "take_profit"
                                            break
                                        if j - entry_bar_index >= MAX_HOLD_BARS:
                                            exit_price = bar.close
                                            exit_reason = "max_time"
                                            break

                                    if exit_price is None:
                                        exit_price = dollar_bars[-1].close
                                        exit_reason = "end_of_data"

                                    commission = COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2
                                    slippage_cost = SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * CONTRACTS_PER_TRADE * 2
                                    price_diff = exit_price - entry_price
                                    pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE * CONTRACTS_PER_TRADE
                                    pnl_final = pnl_before_costs - commission - slippage_cost

                                    trades.append({
                                        "pnl": pnl_final,
                                        "exit_reason": exit_reason,
                                    })

        # Detect bearish FVG
        if candle_1.close < candle_3.open:  # Bearish FVG
            gap_bottom = candle_1.low
            gap_top = candle_3.high
            gap_size = gap_top - gap_bottom

            if gap_top > gap_bottom:
                atr = atr_values[i]
                if gap_size >= (atr * ATR_THRESHOLD):
                    gap_dollars = gap_size * MNQ_CONTRACT_VALUE
                    if gap_dollars <= MAX_GAP_DOLLARS:
                        down_volume = down_volumes[i]
                        up_volume = up_volumes[i]
                        volume_ratio = (down_volume / up_volume) if up_volume > 0 else float('inf')

                        if volume_ratio >= VOLUME_RATIO_THRESHOLD:
                            # Simulate trade
                            entry_price = gap_top
                            take_profit = gap_bottom
                            stop_loss = gap_top + (gap_size * SL_MULTIPLIER)

                            if stop_loss > 0 and take_profit > 0:
                                entry_bar_index = i + 1
                                if entry_bar_index < len(dollar_bars):
                                    exit_price = None
                                    exit_reason = None

                                    sl_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
                                    tp_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
                                    sl_trigger = stop_loss + sl_buffer
                                    tp_trigger = take_profit - tp_buffer

                                    max_index = min(entry_bar_index + MAX_HOLD_BARS + 1, len(dollar_bars))

                                    for j in range(entry_bar_index + 1, max_index):
                                        bar = dollar_bars[j]
                                        if bar.high >= sl_trigger:
                                            exit_price = max(stop_loss, bar.high - sl_buffer)
                                            exit_reason = "stop_loss"
                                            break
                                        if bar.low <= tp_trigger:
                                            exit_price = min(take_profit, bar.low + tp_buffer)
                                            exit_reason = "take_profit"
                                            break
                                        if j - entry_bar_index >= MAX_HOLD_BARS:
                                            exit_price = bar.close
                                            exit_reason = "max_time"
                                            break

                                    if exit_price is None:
                                        exit_price = dollar_bars[-1].close
                                        exit_reason = "end_of_data"

                                    commission = COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2
                                    slippage_cost = SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * CONTRACTS_PER_TRADE * 2
                                    price_diff = entry_price - exit_price
                                    pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE * CONTRACTS_PER_TRADE
                                    pnl_final = pnl_before_costs - commission - slippage_cost

                                    trades.append({
                                        "pnl": pnl_final,
                                        "exit_reason": exit_reason,
                                    })

    # Calculate metrics
    print(f"\n{'=' * 80}")
    print(f"RESULTS - {period_name}")
    print(f"{'=' * 80}")

    if not trades:
        print("❌ No trades generated!")
        sys.exit(1)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100
    total_pnl = sum(t["pnl"] for t in trades)
    total_won = sum(t["pnl"] for t in wins)
    total_lost = sum(t["pnl"] for t in losses)
    profit_factor = abs(total_won / total_lost) if total_lost != 0 else float('inf')
    expectancy = total_pnl / total_trades

    time_span_days = (dollar_bars[-1].timestamp - dollar_bars[0].timestamp).total_seconds() / 86400
    avg_trades_per_day = total_trades / time_span_days if time_span_days > 0 else 0.0

    print(f"Total Trades: {total_trades}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate:.2f}% {'✅' if win_rate >= 60.0 else '❌'}")
    print(f"Profit Factor: {profit_factor:.2f} {'✅' if profit_factor >= 1.7 else '❌'}")
    print(f"Trade Frequency: {avg_trades_per_day:.2f}/day {'✅' if 8.0 <= avg_trades_per_day <= 15.0 else '❌'}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Expectancy: ${expectancy:.2f}/trade")

    targets_met = sum([
        win_rate >= 60.0,
        profit_factor >= 1.7,
        8.0 <= avg_trades_per_day <= 15.0,
    ])
    print(f"\nTargets Met: {targets_met}/3 {'✅' if targets_met == 3 else '❌'}")
    print(f"{'=' * 80}")

    sys.exit(0 if targets_met == 3 else 1)


if __name__ == "__main__":
    main()

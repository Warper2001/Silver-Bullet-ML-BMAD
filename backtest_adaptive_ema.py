"""Backtest the fixed Adaptive EMA Momentum strategy."""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py
import pandas as pd

from src.data.models import DollarBar
from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
from src.research.exit_simulator import ExitSimulator

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_dollar_bars() -> list[DollarBar]:
    """Load all dollar bars from HDF5 files."""
    path = Path("data/processed/dollar_bars/")
    h5_files = sorted(path.glob("*.h5"))

    all_bars = []

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            bars = f['dollar_bars']
            for i in range(len(bars)):
                ts_ms = bars[i, 0]
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                bar = DollarBar(
                    timestamp=ts,
                    open=float(bars[i, 1]),
                    high=float(bars[i, 2]),
                    low=float(bars[i, 3]),
                    close=float(bars[i, 4]),
                    volume=int(bars[i, 5]),
                    notional_value=float(bars[i, 6]),
                    is_forward_filled=False,
                )
                all_bars.append(bar)

    return all_bars


def run_backtest():
    """Run backtest of Adaptive EMA strategy."""

    print("=" * 100)
    print("ADAPTIVE EMA MOMENTUM STRATEGY - BACKTEST")
    print("=" * 100)

    # Load data
    bars = load_dollar_bars()
    print(f"\nLoaded {len(bars)} dollar bars")
    print(f"Date range: {bars[0].timestamp} to {bars[-1].timestamp}")

    # Initialize strategy and exit simulator
    strategy = AdaptiveEMAStrategy()
    exit_sim = ExitSimulator()

    # Track trades
    trades = []

    print("\nProcessing bars...")

    for i, bar in enumerate(bars):
        # Generate signals
        signals = strategy.process_bars([bar])

        if not signals:
            continue

        signal = signals[0]

        # Simulate exit
        exit_bar, exit_price, exit_reason, bars_held = exit_sim.simulate_exit(
            entry_bar=bar,
            bars=bars,
            entry_index=i,
            direction=signal.direction.lower(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )

        # Calculate P&L
        if signal.direction == "LONG":
            pnl = exit_price - signal.entry_price
        else:  # SHORT
            pnl = signal.entry_price - exit_price

        # MNQ point value = $0.50 per point
        pnl_usd = pnl * 0.50

        trades.append({
            'entry_time': signal.timestamp,
            'direction': signal.direction,
            'entry_price': signal.entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl': pnl,
            'pnl_usd': pnl_usd,
            'confidence': signal.confidence,
        })

        if len(trades) <= 10:
            print(f"\nTrade {len(trades)}: {signal.direction} @ {signal.entry_price:.2f}")
            print(f"  Exit: {exit_price:.2f} ({exit_reason})")
            print(f"  P&L: ${pnl_usd:+.2f}")

    # Calculate statistics
    if not trades:
        print("\n❌ NO TRADES GENERATED")
        return 1

    df = pd.DataFrame(trades)

    total_trades = len(df)
    winning_trades = len(df[df['pnl_usd'] > 0])
    losing_trades = len(df[df['pnl_usd'] < 0])
    win_rate = 100 * winning_trades / total_trades if total_trades > 0 else 0

    total_pnl = df['pnl_usd'].sum()
    avg_pnl = df['pnl_usd'].mean()
    max_win = df['pnl_usd'].max()
    max_loss = df['pnl_usd'].min()

    avg_bars_held = df['bars_held'].mean()

    # Exit reason breakdown
    exit_breakdown = df['exit_reason'].value_counts()

    print("\n" + "=" * 100)
    print("BACKTEST RESULTS")
    print("=" * 100)

    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Losing Trades: {losing_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")

    print(f"\nP&L Statistics:")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  Avg P&L per Trade: ${avg_pnl:+,.2f}")
    print(f"  Largest Win: ${max_win:+,.2f}")
    print(f"  Largest Loss: ${max_loss:+,.2f}")

    print(f"\nTrade Duration:")
    print(f"  Avg Bars Held: {avg_bars_held:.1f}")

    print(f"\nExit Reason Breakdown:")
    for reason, count in exit_breakdown.items():
        pct = 100 * count / total_trades
        print(f"  {reason}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 100)

    if total_pnl > 0:
        print("✅ PROFITABLE STRATEGY")
        return 0
    else:
        print("❌ UNPROFITABLE STRATEGY")
        return 1


if __name__ == "__main__":
    sys.exit(run_backtest())

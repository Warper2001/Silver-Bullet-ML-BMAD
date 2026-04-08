"""Comprehensive Adaptive EMA diagnostic showing all conditions."""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py

from src.data.models import DollarBar
from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_bars(num_bars: int = 500) -> list[DollarBar]:
    """Load sample bars."""
    path = Path("data/processed/dollar_bars/")
    h5_files = sorted(path.glob("*.h5"))

    h5_file = h5_files[0]
    all_bars = []

    with h5py.File(h5_file, 'r') as f:
        bars = f['dollar_bars']
        for i in range(min(len(bars), num_bars)):
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


def main():
    """Run comprehensive diagnostic."""
    bars = load_sample_bars(num_bars=500)
    strategy = AdaptiveEMAStrategy()

    print("=" * 120)
    print("COMPREHENSIVE ADAPTIVE EMA DIAGNOSTIC - ALL CONDITIONS")
    print("=" * 120)

    # Track how many bars meet each condition
    ema_bullish = 0
    ema_bearish = 0
    macd_positive = 0
    macd_increasing = 0
    rsi_in_range = 0
    rsi_rising = 0
    rsi_falling = 0
    all_long_conditions = 0
    all_short_conditions = 0

    # Only analyze after all indicators are ready (bar 200+)
    for i, bar in enumerate(bars[200:], start=200):
        strategy.process_bars([bar])

        if i % 10 != 0:  # Only check every 10th bar
            continue

        # Get all indicator values
        ema_values = strategy.ema_calculator.get_current_emas()
        macd_values = strategy.macd_calculator.get_current_macd()
        rsi_value = strategy.rsi_calculator.get_current_rsi()

        if None in [ema_values['fast_ema'], ema_values['medium_ema'], ema_values['slow_ema'],
                  macd_values['macd_line'], macd_values['histogram'], rsi_value]:
            continue

        # Check individual conditions
        bullish_ema = ema_values['fast_ema'] > ema_values['medium_ema'] > ema_values['slow_ema']
        bearish_ema = ema_values['fast_ema'] < ema_values['medium_ema'] < ema_values['slow_ema']

        macd_pos = macd_values['macd_line'] > 0
        macd_inc = strategy.macd_calculator.is_momentum_increasing()

        rsi_mid = 40 <= rsi_value <= 60
        rsi_ris = strategy.rsi_calculator.is_rising()
        rsi_fal = strategy.rsi_calculator.is_falling()

        if bullish_ema:
            ema_bullish += 1
        if bearish_ema:
            ema_bearish += 1

        if macd_pos:
            macd_positive += 1
        if macd_inc:
            macd_increasing += 1

        if rsi_mid:
            rsi_in_range += 1
        if rsi_ris:
            rsi_rising += 1
        if rsi_fal:
            rsi_falling += 1

        # Check all LONG conditions
        long_ok = bullish_ema and macd_pos and macd_inc and rsi_mid and rsi_ris
        if long_ok:
            all_long_conditions += 1
            print(f"\n✅ BAR {i}: ALL LONG CONDITIONS MET!")
            print(f"   Price: {bar.close:.2f}")
            print(f"   EMAs: Fast={ema_values['fast_ema']:.2f} > Medium={ema_values['medium_ema']:.2f} > Slow={ema_values['slow_ema']:.2f}")
            print(f"   MACD: {macd_values['macd_line']:.2f} (positive, increasing)")
            print(f"   RSI: {rsi_value:.2f} (in 40-60 range, rising)")

        # Check all SHORT conditions
        short_ok = bearish_ema and macd_values['macd_line'] < 0 and strategy.macd_calculator.is_momentum_decreasing() and rsi_mid and rsi_fal
        if short_ok:
            all_short_conditions += 1
            print(f"\n✅ BAR {i}: ALL SHORT CONDITIONS MET!")
            print(f"   Price: {bar.close:.2f}")
            print(f"   EMAs: Fast={ema_values['fast_ema']:.2f} < Medium={ema_values['medium_ema']:.2f} < Slow={ema_values['slow_ema']:.2f}")
            print(f"   MACD: {macd_values['macd_line']:.2f} (negative, decreasing)")
            print(f"   RSI: {rsi_value:.2f} (in 40-60 range, falling)")

    # Print summary
    total_checked = (500 - 200) // 10  # Number of bars we actually checked

    print("\n" + "=" * 120)
    print("CONDITION FREQUENCY ANALYSIS")
    print("=" * 120)
    print(f"\nBars Analyzed (after bar 200, every 10th): {total_checked}")
    print("\nLONG Conditions:")
    print(f"  1. EMA Bullish (fast > medium > slow): {ema_bullish}/{total_checked} ({100*ema_bullish/total_checked:.1f}%)")
    print(f"  2. MACD Positive (> 0): {macd_positive}/{total_checked} ({100*macd_positive/total_checked:.1f}%)")
    print(f"  3. MACD Increasing (histogram rising): {macd_increasing}/{total_checked} ({100*macd_increasing/total_checked:.1f}%)")
    print(f"  4. RSI in Range (40-60): {rsi_in_range}/{total_checked} ({100*rsi_in_range/total_checked:.1f}%)")
    print(f"  5. RSI Rising: {rsi_rising}/{total_checked} ({100*rsi_rising/total_checked:.1f}%)")
    print(f"\n  ALL LONG CONDITIONS MET: {all_long_conditions}/{total_checked}")

    print("\nSHORT Conditions:")
    print(f"  1. EMA Bearish (fast < medium < slow): {ema_bearish}/{total_checked} ({100*ema_bearish/total_checked:.1f}%)")
    print(f"  2. RSI Falling: {rsi_falling}/{total_checked} ({100*rsi_falling/total_checked:.1f}%)")
    print(f"\n  ALL SHORT CONDITIONS MET: {all_short_conditions}/{total_checked}")

    print("\n" + "=" * 120)
    print("ROOT CAUSE")
    print("=" * 120)
    print("\nThe KILLER conditions are:")
    print("  1. MACD histogram must be INCREASING (not just positive)")
    print("  2. RSI must be RISING or FALLING (not just in range)")
    print("\nThese momentum requirements make it extremely rare for ALL conditions to align simultaneously.")
    print("\nSOLUTION: Relax these restrictive conditions")
    print("  1. Remove 'MACD increasing' requirement (just need MACD > 0)")
    print("  2. Remove 'RSI rising/falling' requirement (just need RSI in range)")
    print("  3. Or both")

    return 0


if __name__ == "__main__":
    sys.exit(main())

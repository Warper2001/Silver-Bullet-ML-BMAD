"""Deep diagnostic of Adaptive EMA EMA alignment.

This script shows actual EMA values and their relationships to understand
why the alignment condition is never met.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py

from src.data.models import DollarBar
from src.detection.ema_calculator import EMACalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_bars(num_bars: int = 300) -> list[DollarBar]:
    """Load sample bars."""
    path = Path("data/processed/dollar_bars/")
    h5_files = sorted(path.glob("*.h5"))

    if not h5_files:
        raise ValueError("No HDF5 files found")

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
    """Diagnose EMA alignment in detail."""
    bars = load_sample_bars(num_bars=300)

    ema_calc = EMACalculator()

    print("=" * 100)
    print("DETAILED EMA ALIGNMENT DIAGNOSTIC")
    print("=" * 100)

    # Process bars and show EMA values
    for i, bar in enumerate(bars):
        ema_calc.calculate_emas([bar])

        # Start showing after we have all EMAs (100 bars)
        if i >= 100:
            ema_values = ema_calc.get_current_emas()
            fast = ema_values['fast_ema']
            medium = ema_values['medium_ema']
            slow = ema_values['slow_ema']

            if all(v is not None for v in [fast, medium, slow]):
                bullish_aligned = fast > medium > slow
                bearish_aligned = fast < medium < slow

                spread_fast_medium = ((fast - medium) / medium) * 100
                spread_medium_slow = ((medium - slow) / slow) * 100

                if i % 10 == 0:  # Print every 10th bar
                    print(f"\nBar {i}: Close={bar.close:.2f}")
                    print(f"  EMAs: Fast={fast:.2f}, Medium={medium:.2f}, Slow={slow:.2f}")
                    print(f"  Spreads: Fast-Med={spread_fast_medium:+.3f}%, Med-Slow={spread_medium_slow:+.3f}%")
                    print(f"  Alignment: {'✅ BULLISH' if bullish_aligned else '✅ BEARISH' if bearish_aligned else '❌ NONE'}")

                if i >= 200:  # Only show 100 bars total
                    break

    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print("\nWhy EMAs rarely align:")
    print("1. Fast EMA (9) reacts very quickly to price changes")
    print("2. Medium EMA (55) lags behind")
    print("3. Slow EMA (100) lags even more")
    print("4. Perfect alignment (all three in order) is rare in volatile markets")
    print("\nSolutions:")
    print("Option 1: Relax EMA alignment requirement")
    print("  - Use: Fast > Medium AND Medium > Slow (not necessarily perfect ordering)")
    print("  - Or: Fast > Slow (ignore Medium)")
    print("\nOption 2: Use different EMA periods")
    print("  - Shorter periods: 5, 21, 50")
    print("  - Or: 13, 34, 89 (Fibonacci-based)")
    print("\nOption 3: Remove EMA alignment as hard requirement")
    print("  - Use MACD and RSI as primary signals")
    print("  - Use EMA for trend confirmation only")


if __name__ == "__main__":
    main()

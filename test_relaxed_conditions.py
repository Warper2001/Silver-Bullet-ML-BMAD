"""Test relaxed Adaptive EMA conditions."""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py

from src.data.models import DollarBar
from src.detection.ema_calculator import EMACalculator
from src.detection.macd_calculator import MACDCalculator
from src.detection.rsi_calculator import RSICalculator

logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    """Test relaxed conditions."""
    bars = load_sample_bars(num_bars=500)

    ema_calc = EMACalculator()
    macd_calc = MACDCalculator()
    rsi_calc = RSICalculator()

    print("=" * 100)
    print("RELAXED CONDITIONS TEST - EMA 9/55/100, RSI 30-70")
    print("=" * 100)

    # Test on bars 200-400 (after all indicators initialized)
    ema_bullish = 0
    macd_positive = 0
    rsi_in_range = 0
    all_conditions = 0

    for i, bar in enumerate(bars[200:400], start=200):
        ema_calc.calculate_emas([bar])
        macd_calc.calculate_macd([bar])
        rsi_calc.calculate_rsi([bar])

        ema_values = ema_calc.get_current_emas()
        macd_values = macd_calc.get_current_macd()
        rsi_value = rsi_calc.get_current_rsi()

        if None in [ema_values['fast_ema'], ema_values['medium_ema'], ema_values['slow_ema'],
                  macd_values['macd_line'], rsi_value]:
            continue

        # Check relaxed LONG conditions
        bullish_ema = ema_values['fast_ema'] > ema_values['medium_ema'] > ema_values['slow_ema']
        macd_pos = macd_values['macd_line'] > 0
        rsi_ok = 30 <= rsi_value <= 70

        if bullish_ema:
            ema_bullish += 1
        if macd_pos:
            macd_positive += 1
        if rsi_ok:
            rsi_in_range += 1

        if bullish_ema and macd_pos and rsi_ok:
            all_conditions += 1
            print(f"\n✅ Bar {i}: ALL RELAXED CONDITIONS MET!")
            print(f"   Price: {bar.close:.2f}")
            print(f"   EMAs: Fast={ema_values['fast_ema']:.2f} > Medium={ema_values['medium_ema']:.2f} > Slow={ema_values['slow_ema']:.2f}")
            print(f"   MACD: {macd_values['macd_line']:.2f} (> 0)")
            print(f"   RSI: {rsi_value:.2f} (in 30-70)")

    total_checked = 200

    print(f"\n{'=' * 100}")
    print(f"RELAXED CONDITIONS ANALYSIS (bars 200-400)")
    print(f"{'=' * 100}")
    print(f"\nTotal Bars Checked: {total_checked}")
    print(f"\nCondition Frequency:")
    print(f"  1. EMA Bullish (fast > medium > slow): {ema_bullish}/{total_checked} ({100*ema_bullish/total_checked:.1f}%)")
    print(f"  2. MACD Positive (> 0): {macd_positive}/{total_checked} ({100*macd_positive/total_checked:.1f}%)")
    print(f"  3. RSI in Range (30-70): {rsi_in_range}/{total_checked} ({100*rsi_in_range/total_checked:.1f}%)")
    print(f"\n  ALL RELAXED CONDITIONS MET: {all_conditions}/{total_checked}")

    if all_conditions > 0:
        print(f"\n✅ SUCCESS! {all_conditions} signals would be generated with relaxed conditions")
        return 0
    else:
        print(f"\n❌ Still no signals - even with relaxed conditions")
        print(f"\nThis suggests EMA alignment itself is the main bottleneck.")
        print(f"\nRecommendation: Consider removing EMA alignment requirement")
        return 1


if __name__ == "__main__":
    sys.exit(main())

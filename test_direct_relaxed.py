"""Direct test of relaxed conditions without caching."""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py

from src.data.models import DollarBar
from src.detection.ema_calculator import EMACalculator
from src.detection.macd_calculator import MACDCalculator
from src.detection.rsi_calculator import RSICalculator
from src.detection.models import MomentumSignal

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_sample_bars(num_bars: int = 400) -> list[DollarBar]:
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


def create_signal(direction: str, bar: DollarBar, ema_values, macd_values, rsi_value):
    """Create a MomentumSignal directly."""
    entry_price = bar.close
    atr_distance = entry_price * 0.0025  # 0.25% as fallback

    if direction == "LONG":
        stop_loss = entry_price - atr_distance
        take_profit = entry_price + (atr_distance * 2.0)
    else:  # SHORT
        stop_loss = entry_price + atr_distance
        take_profit = entry_price - (atr_distance * 2.0)

    confidence = 60.0  # Base confidence

    return MomentumSignal(
        timestamp=bar.timestamp,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        ema_fast=ema_values['fast_ema'],
        ema_medium=ema_values['medium_ema'],
        ema_slow=ema_values['slow_ema'],
        macd_line=macd_values['macd_line'],
        macd_signal=macd_values['signal_line'],
        macd_histogram=macd_values['histogram'],
        rsi_value=rsi_value,
        rsi_in_mid_band=30 <= rsi_value <= 70,
    )


def main():
    """Test with relaxed conditions directly."""
    bars = load_sample_bars(num_bars=400)

    ema_calc = EMACalculator()
    macd_calc = MACDCalculator()
    rsi_calc = RSICalculator()

    print("=" * 80)
    print("DIRECT RELAXED CONDITIONS TEST")
    print("=" * 80)
    print(f"Loaded {len(bars)} bars\n")

    signals = []

    for i, bar in enumerate(bars):
        ema_calc.calculate_emas([bar])
        macd_calc.calculate_macd([bar])
        rsi_calc.calculate_rsi([bar])

        ema_values = ema_calc.get_current_emas()
        macd_values = macd_calc.get_current_macd()
        rsi_value = rsi_calc.get_current_rsi()

        # Check if all indicators available
        if None in [ema_values['fast_ema'], ema_values['medium_ema'], ema_values['slow_ema'],
                  macd_values['macd_line'], rsi_value]:
            continue

        # Check relaxed LONG conditions
        if (ema_values['fast_ema'] > ema_values['medium_ema'] > ema_values['slow_ema'] and
            macd_values['macd_line'] > 0 and
            30 <= rsi_value <= 70):
            signal = create_signal('LONG', bar, ema_values, macd_values, rsi_value)
            signals.append(signal)
            print(f"\n✅ Signal {len(signals)}: LONG @ {bar.close:.2f} (bar {i})")

        # Check relaxed SHORT conditions
        elif (ema_values['fast_ema'] < ema_values['medium_ema'] < ema_values['slow_ema'] and
              macd_values['macd_line'] < 0 and
              30 <= rsi_value <= 70):
            signal = create_signal('SHORT', bar, ema_values, macd_values, rsi_value)
            signals.append(signal)
            print(f"\n✅ Signal {len(signals)}: SHORT @ {bar.close:.2f} (bar {i})")

        if len(signals) >= 10:  # Stop after 10 signals
            break

    print(f"\n{'=' * 80}")
    print(f"TOTAL SIGNALS: {len(signals)}")
    print('=' * 80)

    return 0 if len(signals) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

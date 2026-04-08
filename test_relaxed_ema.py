"""Test the relaxed Adaptive EMA strategy."""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py

from src.data.models import DollarBar

# Force reload to get latest changes
import importlib
import src.detection.adaptive_ema_strategy
importlib.reload(src.detection.adaptive_ema_strategy)
from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

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
    """Test relaxed Adaptive EMA."""
    bars = load_sample_bars(num_bars=500)

    print("=" * 80)
    print("TESTING RELAXED ADAPTIVE EMA STRATEGY")
    print("=" * 80)
    print(f"\nLoaded {len(bars)} bars")
    print(f"Date range: {bars[0].timestamp} to {bars[-1].timestamp}\n")

    strategy = AdaptiveEMAStrategy()

    signals_generated = 0

    for i, bar in enumerate(bars):
        signals = strategy.process_bars([bar])

        if signals:
            signals_generated += 1
            signal = signals[0]
            print(f"\n✅ Signal {signals_generated}: {signal.direction} @ {signal.entry_price:.2f}")
            print(f"   SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}")
            print(f"   Confidence: {signal.confidence:.1f}%")
            print(f"   Bar: {i}")

    print(f"\n{'=' * 80}")
    print(f"TOTAL SIGNALS: {signals_generated}")
    print('=' * 80)

    return 0 if signals_generated > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

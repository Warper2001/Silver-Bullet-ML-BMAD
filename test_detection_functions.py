#!/usr/bin/env python3
"""Quick test of pattern detection functions."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import h5py
from src.data.models import DollarBar
from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg
from src.detection.swing_detection import (
    detect_swing_high,
    detect_swing_low,
    detect_bullish_mss,
    detect_bearish_mss,
    RollingVolumeAverage,
)

# Load some data
with h5py.File('data/processed/time_bars/MNQ_time_bars_5min_202401.h5', 'r') as f:
    raw_data = f['dollar_bars'][:100]  # First 100 bars

# Convert to DollarBar objects
bars = []
for bar in raw_data:
    ts_ms = bar[0]
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    dollar_bar = DollarBar(
        timestamp=ts,
        open=float(bar[1]),
        high=float(bar[2]),
        low=float(bar[3]),
        close=float(bar[4]),
        volume=int(bar[5]),
        notional_value=float(bar[6]) if len(bar) > 6 else 0.0,
        is_forward_filled=False,
    )
    bars.append(dollar_bar)

print(f"Loaded {len(bars)} bars")
print(f"Date range: {bars[0].timestamp} to {bars[-1].timestamp}")
print()

# Test FVG detection
print("Testing FVG detection...")
for i in range(3, len(bars)):
    historical_bars = bars[max(0, i-10):i+1]

    if len(historical_bars) >= 3:
        # The current_index should be the last bar in the historical array
        current_index = len(historical_bars) - 1

        bullish_fvg = detect_bullish_fvg(historical_bars, current_index)
        bearish_fvg = detect_bearish_fvg(historical_bars, current_index)

        if bullish_fvg:
            print(f"  Bullish FVG found at bar {i}: {bullish_fvg}")
        if bearish_fvg:
            print(f"  Bearish FVG found at bar {i}: {bearish_fvg}")

print("\nTesting Swing detection...")
volume_ma = RollingVolumeAverage(window=20)
swing_highs = []
swing_lows = []

for i, bar in enumerate(bars):
    volume_ma.update(bar.volume)

    if len(bars[max(0, i-6):i+1]) >= 6:
        historical_bars = bars[max(0, i-6):i+1]
        current_index = len(historical_bars) - 1  # Last bar in the array
        swing_high = detect_swing_high(historical_bars, current_index, lookback=3)
        swing_low = detect_swing_low(historical_bars, current_index, lookback=3)

        if swing_high:
            swing_highs.append(swing_high)
            print(f"  Swing High found at bar {i}: {swing_high.price} at {swing_high.timestamp}")
        if swing_low:
            swing_lows.append(swing_low)
            print(f"  Swing Low found at bar {i}: {swing_low.price} at {swing_low.timestamp}")

print(f"\nFound {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")

print("\nTesting MSS detection...")
for i in range(20, len(bars)):
    historical_bars = bars[max(0, i-20):i+1]

    if swing_highs:
        bullish_mss = detect_bullish_mss(historical_bars, swing_highs, volume_ma, volume_ratio=1.5)
        if bullish_mss:
            print(f"  Bullish MSS found at bar {i}: {bullish_mss}")

    if swing_lows:
        bearish_mss = detect_bearish_mss(historical_bars, swing_lows, volume_ma, volume_ratio=1.5)
        if bearish_mss:
            print(f"  Bearish MSS found at bar {i}: {bearish_mss}")

print("\n✅ Detection test complete")
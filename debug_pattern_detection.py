#!/usr/bin/env python3
"""Debug pattern detection to see why 0 signals are found."""

import sys
from pathlib import Path
import pandas as pd
import logging
import h5py

sys.path.insert(0, str(Path(__file__).parent))

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Load time bars
print("Loading time bars...")
data_dir = Path("data/processed/time_bars/")
files = list(data_dir.glob("MNQ_time_bars_5min_2024*.h5"))[:3]  # Load 3 months

print(f"Loading {len(files)} files...")
dataframes = []
for file_path in files:
    with h5py.File(file_path, 'r') as f:
        data = f['dollar_bars'][:]
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    dataframes.append(df)

combined = pd.concat(dataframes, ignore_index=True)
combined = combined.sort_values('timestamp').set_index('timestamp')

print(f"Loaded {len(combined)} bars")
print(f"Date range: {combined.index.min()} to {combined.index.max()}")
print(f"\nSample data:\n{combined.head()}")

# Test pattern detection
from src.research.silver_bullet_backtester import SilverBulletBacktester

print("\n" + "="*70)
print("TESTING PATTERN DETECTION")
print("="*70)

backtester = SilverBulletBacktester(
    mss_lookback=3,
    fvg_min_gap=0.25,
    max_bar_distance=10,
    min_confidence=60.0,
    enable_time_windows=False,  # Disable time filtering to see all patterns
)

print("\nRunning backtest...")
signals = backtester.run_backtest(combined)

print(f"\n{'='*70}")
print(f"RESULTS: {len(signals)} signals detected")
print(f"{'='*70}")

if len(signals) > 0:
    print("\nSignal breakdown:")
    print(f"  Bullish: {len(signals[signals['direction'] == 'bullish'])}")
    print(f"  Bearish: {len(signals[signals['direction'] == 'bearish'])}")
    print(f"\nSample signals:")
    print(signals.head(10))
else:
    print("\nNo signals detected - checking individual patterns...")

    # Check individual pattern detection
    from src.data.models import DollarBar

    bars = []
    for idx, row in combined.iterrows():
        bar = DollarBar(
            timestamp=idx,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=int(row['volume']),
            notional_value=row['notional_value']
        )
        bars.append(bar)

    print(f"\nConverted to {len(bars)} DollarBar objects")

    # Test FVG detection
    from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg

    print("\nTesting FVG detection on first 1000 bars...")
    fvg_count = 0
    for i in range(2, min(1000, len(bars))):
        bullish = detect_bullish_fvg(bars, i)
        bearish = detect_bearish_fvg(bars, i)
        if bullish or bearish:
            fvg_count += 1
            if fvg_count <= 5:
                print(f"  Bar {i}: Bullish={bullish is not None}, Bearish={bearish is not None}")

    print(f"  Total FVGs found in first 1000 bars: {fvg_count}")

    # Test MSS detection
    from src.detection.swing_detection import detect_swing_high, detect_swing_low
    from src.data.models import SwingPoint
    from src.detection.mss_detection import detect_bullish_mss, detect_bearish_mss

    print("\nTesting MSS detection...")
    swing_highs = []
    swing_lows = []

    lookback = 3
    for i in range(lookback, min(1000, len(bars))):
        if detect_swing_high(bars, i, lookback=lookback):
            swing_highs.append(SwingPoint(
                timestamp=bars[i].timestamp,
                price=bars[i].high,
                bar_index=i
            ))

        if detect_swing_low(bars, i, lookback=lookback):
            swing_lows.append(SwingPoint(
                timestamp=bars[i].timestamp,
                price=bars[i].low,
                bar_index=i
            ))

    print(f"  Swing highs: {len(swing_highs)}")
    print(f"  Swing lows: {len(swing_lows)}")

    volumes = [bar.volume for bar in bars]
    volume_ma = sum(volumes[-20:]) / min(20, len(volumes))

    mss_count = 0
    for i in range(lookback, min(1000, len(bars))):
        bullish = detect_bullish_mss(bars[i], swing_highs, volume_ma, 1.5)
        bearish = detect_bearish_mss(bars[i], swing_lows, volume_ma, 1.5)
        if bullish or bearish:
            mss_count += 1
            if mss_count <= 5:
                print(f"  Bar {i}: Bullish MSS={bullish is not None}, Bearish MSS={bearish is not None}")

    print(f"  Total MSS found in first 1000 bars: {mss_count}")

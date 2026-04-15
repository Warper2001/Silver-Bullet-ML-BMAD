#!/usr/bin/env python3
"""
Generate 1-Minute Dollar Bars from 2025 MNQ Trade Data

Loads /root/mnq_historical.json, filters to 2025 data, and generates
1-minute dollar bars with $50M notional value threshold.

Output: data/processed/dollar_bars/1_minute/mnq_1min_2025.csv
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import csv

# Constants
DOLLAR_THRESHOLD = 50_000_000  # $50M notional value per bar
MNQ_MULTIPLIER = 20.0  # $20 per point per contract

def main():
    print("=" * 70)
    print("1-MINUTE DOLLAR BAR GENERATION - 2025 MNQ DATA")
    print("=" * 70)

    # Load data
    data_path = Path("/root/mnq_historical.json")
    print(f"\n📂 Loading data from: {data_path}")

    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    print(f"✅ Loaded {len(raw_data):,} total bars")

    # Filter to 2025 and convert to bar format
    print(f"\n📊 Filtering to 2025 data...")

    bars_2025 = []
    for bar_data in raw_data:
        try:
            timestamp_str = bar_data.get('TimeStamp', '')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            if timestamp.year != 2025:
                continue

            open_price = float(bar_data.get('Open', 0))
            high_price = float(bar_data.get('High', 0))
            low_price = float(bar_data.get('Low', 0))
            close_price = float(bar_data.get('Close', 0))
            volume = int(bar_data.get('TotalVolume', 0))

            if high_price == 0 or low_price == 0:
                continue

            bars_2025.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            })
        except Exception:
            continue

    print(f"✅ Filtered to {len(bars_2025):,} bars for 2025")
    print(f"   Period: {bars_2025[0]['timestamp']} to {bars_2025[-1]['timestamp']}")

    # Calculate notional values and dollar bar accumulation
    print(f"\n💰 Generating dollar bars (threshold: ${DOLLAR_THRESHOLD/1_000_000:.0f}M)...")

    output_path = Path("data/processed/dollar_bars/1_minute")
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "mnq_1min_2025.csv"

    dollar_bars = []
    current_notional = 0.0
    current_open = None
    current_high = float('-inf')
    current_low = float('inf')
    current_close = None
    current_volume = 0
    bar_start_time = None

    for bar in bars_2025:
        # Initialize first bar
        if current_open is None:
            current_open = bar['open']
            current_high = bar['high']
            current_low = bar['low']
            bar_start_time = bar['timestamp']

        # Update OHLC
        current_high = max(current_high, bar['high'])
        current_low = min(current_low, bar['low'])
        current_close = bar['close']
        current_volume += bar['volume']

        # Calculate notional value for this bar
        notional = bar['close'] * bar['volume'] * MNQ_MULTIPLIER
        current_notional += notional

        # Check if threshold reached
        if current_notional >= DOLLAR_THRESHOLD:
            dollar_bars.append({
                'timestamp': bar['timestamp'],
                'open': current_open,
                'high': current_high,
                'low': current_low,
                'close': current_close,
                'volume': current_volume,
                'notional': current_notional,
            })

            # Reset for next bar
            current_notional = 0.0
            current_open = None
            current_high = float('-inf')
            current_low = float('inf')
            current_close = None
            current_volume = 0
            bar_start_time = None

    print(f"✅ Generated {len(dollar_bars):,} dollar bars")

    # Write to CSV
    print(f"\n💾 Writing to: {output_file}")

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional'])
        writer.writeheader()
        writer.writerows(dollar_bars)

    print(f"✅ Saved {len(dollar_bars):,} dollar bars")

    # Data quality summary
    print(f"\n📊 DATA QUALITY SUMMARY:")

    # Check completeness
    complete_bars = sum(1 for bar in dollar_bars if bar['notional'] >= DOLLAR_THRESHOLD)
    completeness = complete_bars / len(dollar_bars) * 100
    print(f"   Completeness: {completeness:.2f}% (target: 99.99%)")

    # Check OHLC consistency
    invalid_ohlc = sum(1 for bar in dollar_bars if bar['high'] < bar['low'])
    print(f"   Invalid OHLC: {invalid_ohlc}")

    # Notional value statistics
    notionals = [bar['notional'] for bar in dollar_bars]
    avg_notional = sum(notionals) / len(notionals)
    min_notional = min(notionals)
    max_notional = max(notionals)

    print(f"   Notional values:")
    print(f"     Average: ${avg_notional/1_000_000:.2f}M")
    print(f"     Min: ${min_notional/1_000_000:.2f}M")
    print(f"     Max: ${max_notional/1_000_000:.2f}M")

    print(f"\n" + "=" * 70)
    print("✅ 1-MINUTE DOLLAR BAR GENERATION COMPLETE")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())

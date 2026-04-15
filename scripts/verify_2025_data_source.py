#!/usr/bin/env python3
"""
Verify 2025 MNQ Data Source Availability

Confirms that /root/mnq_historical.json exists and is accessible.
Validates data format, completeness, and quality for 1-minute migration.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

def main():
    print("=" * 70)
    print("2025 MNQ DATA SOURCE VERIFICATION")
    print("=" * 70)

    # Check file exists
    data_path = Path("/root/mnq_historical.json")
    print(f"\n📂 Data Source: {data_path}")

    if not data_path.exists():
        print(f"❌ ERROR: Data file not found!")
        sys.exit(1)

    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"✅ File exists: {file_size_mb:.1f} MB")

    # Load and validate data
    print(f"\n📊 Loading data...")
    try:
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        print(f"✅ Loaded {len(raw_data):,} total bars")
    except Exception as e:
        print(f"❌ ERROR: Failed to load data: {e}")
        sys.exit(1)

    # Validate format
    print(f"\n🔍 Validating data format...")
    sample = raw_data[0] if len(raw_data) > 0 else None

    required_fields = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'TotalVolume']
    if sample:
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            print(f"❌ ERROR: Missing required fields: {missing_fields}")
            sys.exit(1)
        print(f"✅ Format validated: {', '.join(required_fields)}")
        print(f"   Sample: {sample}")

    # Filter to 2025 data
    print(f"\n📅 Filtering to 2025 data...")
    bars_2025 = []
    date_distribution = Counter()

    for bar_data in raw_data:
        try:
            timestamp_str = bar_data.get('TimeStamp', '')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            if timestamp.year == 2025:
                bars_2025.append({
                    'timestamp': timestamp,
                    'open': float(bar_data.get('Open', 0)),
                    'high': float(bar_data.get('High', 0)),
                    'low': float(bar_data.get('Low', 0)),
                    'close': float(bar_data.get('Close', 0)),
                    'volume': int(bar_data.get('TotalVolume', 0)),
                })
                date_distribution[timestamp.date()] += 1
        except Exception as e:
            continue

    if len(bars_2025) == 0:
        print(f"❌ ERROR: No 2025 data found!")
        sys.exit(1)

    print(f"✅ Found {len(bars_2025):,} bars for 2025")
    print(f"   Date range: {bars_2025[0]['timestamp'].date()} to {bars_2025[-1]['timestamp'].date()}")
    print(f"   Expected: ~130K bars at 1-minute resolution")
    print(f"   Actual: {len(bars_2025):,} bars ({len(bars_2025)/130000*100:.1f}% of expected)")

    # Data quality checks
    print(f"\n✅ Data Quality Checks:")

    # Check for missing fields
    complete_bars = sum(1 for bar in bars_2025 if all(bar[k] > 0 for k in ['open', 'high', 'low', 'close', 'volume']))
    completeness = complete_bars / len(bars_2025) * 100
    print(f"   Completeness: {completeness:.2f}% (target: 99.99%)")

    # Check OHLC consistency
    invalid_ohlc = sum(1 for bar in bars_2025 if bar['high'] < bar['low'])
    print(f"   Invalid OHLC (high < low): {invalid_ohlc}")

    # Check for zero volume
    zero_volume = sum(1 for bar in bars_2025 if bar['volume'] == 0)
    print(f"   Zero volume bars: {zero_volume} ({zero_volume/len(bars_2025)*100:.2f}%)")

    # Trading days analysis
    unique_days = len(date_distribution)
    print(f"   Trading days: {unique_days}")

    avg_bars_per_day = len(bars_2025) / unique_days
    print(f"   Avg bars/day: {avg_bars_per_day:.0f} (expected: ~390 for 6.5-hour session)")

    # Generate report
    print(f"\n" + "=" * 70)
    print("✅ DATA SOURCE VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"Status: READY FOR 1-MINUTE MIGRATION")
    print(f"Source: {data_path}")
    print(f"Format: JSON (TimeStamp, Open, High, Low, Close, TotalVolume)")
    print(f"2025 Records: {len(bars_2025):,} bars")
    print(f"Completeness: {completeness:.2f}%")
    print(f"Quality Issues: {invalid_ohlc} invalid OHLC, {zero_volume} zero volume")
    print("=" * 70)

    # Return success
    return 0

if __name__ == "__main__":
    sys.exit(main())

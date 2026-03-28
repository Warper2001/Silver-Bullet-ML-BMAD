#!/usr/bin/env python3
"""Run backtest with Silver Bullet pattern detection (no ML)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.historical_data_loader import HistoricalDataLoader
from src.research.silver_bullet_backtester import SilverBulletBacktester
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import pandas as pd

def main():
    """Run pattern detection backtest."""

    print("🚀 SILVER BULLET PATTERN BACKTEST")
    print("=" * 60)

    # Load data
    print("\n📊 Loading historical data...")
    loader = HistoricalDataLoader(
        data_directory="data/processed/dollar_bars/",
        min_completeness=0.1
    )

    data = loader.load_data('2025-01-01', '2025-03-05')
    print(f"✅ Loaded {len(data)} bars")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Detect patterns
    print("\n🎯 Detecting Silver Bullet patterns...")
    sb_backtester = SilverBulletBacktester()
    signals = sb_backtester.run_backtest(data)

    print(f"\n📈 Pattern Detection Results:")
    print(f"   Total signals detected: {len(signals)}")

    if len(signals) > 0:
        print(f"\n   Breakdown by direction:")
        print(f"   - Bullish: {len(signals[signals['direction'] == 'bullish'])}")
        print(f"   - Bearish: {len(signals[signals['direction'] == 'bearish'])}")

        print(f"\n   Breakdown by pattern type:")
        for pattern_type in signals['pattern_type'].unique():
            count = len(signals[signals['pattern_type'] == pattern_type])
            print(f"   - {pattern_type}: {count}")

        print(f"\n   Sample signals (first 5):")
        print(signals.head().to_string())

        # Calculate basic metrics from signals
        print(f"\n📊 Signal Statistics:")
        print(f"   Avg confidence: {signals['confidence'].mean():.2f}%")
        print(f"   Price range at signals: ${signals['price'].min():.2f} - ${signals['price'].max():.2f}")
    else:
        print("\n⚠️  No patterns detected!")
        print("   This could mean:")
        print("   - Dollar bar threshold ($50M) is too high for this period")
        print("   - Market conditions didn't produce Silver Bullet setups")
        print("   - Pattern detection parameters need adjustment")

    print("\n✅ Backtest complete!")

if __name__ == '__main__':
    main()

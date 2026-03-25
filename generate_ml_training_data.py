#!/usr/bin/env python3
"""Generate ML training data from full dataset efficiently."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester
import run_optimized_silver_bullet as base_module


def main():
    print("🚀 GENERATING ML TRAINING DATA - FULL DATASET")
    print("=" * 70)

    # Load data (28 months)
    print("\n📊 Step 1: Loading time bars (Dec 2023 - Mar 2026)...")
    data = base_module.load_time_bars('2023-12-01', '2026-03-06')

    if data.empty:
        print("❌ No data available!")
        return

    # Calculate daily bias
    print("\n📊 Step 2: Calculating daily bias...")
    daily_data = data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    daily_bias = base_module.calculate_daily_bias(daily_data)
    print(f"✅ Daily bias calculated")

    # Run pattern detection
    print("\n🎯 Step 3: Running pattern detection...")
    backtester = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        max_bar_distance=10,
        min_confidence=60.0,
        enable_time_windows=True,
        require_sweep=False,
    )

    signals_df = backtester.run_backtest(data)
    print(f"✅ Pattern detection complete: {len(signals_df):,} signals")

    # Deduplicate
    signals_df = signals_df.sort_values('confidence', ascending=False)
    signals_df = signals_df[~signals_df.index.duplicated(keep='first')]
    print(f"   After deduplication: {len(signals_df):,} signals")

    # Apply filters
    signals_df = base_module.add_daily_bias_filter(signals_df, daily_bias)
    signals_df = base_module.add_volatility_filter(data, signals_df, min_atr_pct=0.003)
    print(f"   After filters: {len(signals_df):,} signals")

    # Simulate trades
    print("\n⚡ Step 4: Simulating trades...")
    trades = base_module.simulate_trades_with_fvg_stops(data, signals_df)
    print(f"✅ Completed {len(trades)} trades")

    # Save ML training data
    print("\n💾 Step 5: Saving ML training data...")
    ml_dir = Path("data/ml_training")
    ml_dir.mkdir(parents=True, exist_ok=True)

    signals_path = ml_dir / "silver_bullet_signals_full.parquet"
    trades_path = ml_dir / "silver_bullet_trades_full.parquet"
    metadata_path = ml_dir / "metadata_full.json"

    signals_df.to_parquet(signals_path, index=True)
    trades.to_parquet(trades_path, index=False)

    metadata = {
        "date_range": {"start": "2023-12-01", "end": "2026-03-06"},
        "total_signals": int(len(signals_df)),
        "total_trades": int(len(trades)),
        "filters_applied": ["daily_bias", "volatility"],
        "strategy": "silver_bullet_optimized_full"
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ Saved {len(signals_df)} signals to {signals_path}")
    print(f"   ✅ Saved {len(trades)} trades to {trades_path}")
    print(f"   ✅ Saved metadata to {metadata_path}")

    # Calculate metrics
    print("\n📈 Step 6: Calculating performance metrics...")
    metrics = base_module.calculate_metrics(trades)

    print("\n" + "=" * 70)
    print("✅ ML TRAINING DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Dataset Summary:")
    print(f"   Period: Dec 2023 - Mar 2026 (28 months)")
    print(f"   Signals: {len(signals_df):,}")
    print(f"   Trades: {len(trades):,}")
    print(f"   Total Return: {metrics['total_return']:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']:.2f}%")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics['max_drawdown']:.2f}%")
    print(f"\n💾 Next step: Run 'python train_meta_model.py' to train meta-model")


if __name__ == '__main__':
    main()

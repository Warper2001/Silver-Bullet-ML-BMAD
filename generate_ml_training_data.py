#!/usr/bin/env python3
"""Generate ML training data from full dataset efficiently.

Supports both standard and premium strategy labeling.
Use --premium flag to generate premium-labeled training data.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester
import run_optimized_silver_bullet as base_module


def label_premium_trade(trade: pd.Series, config: dict = None) -> bool:
    """Label trade as premium if high quality AND profitable.

    Premium quality requirements:
    - FVG size >= $75 (configurable)
    - Volume ratio >= 2.0x (configurable)
    - Bar distance <= 7 (configurable)
    - Killzone aligned (optional)
    - Profitable (pnl > 0)

    Args:
        trade: Trade row with columns: fvg_size, volume_ratio, bar_distance, killzone_aligned, pnl
        config: Premium configuration dict with thresholds

    Returns:
        True if trade meets premium quality standards
    """
    if config is None:
        config = {
            'min_fvg_gap_size_dollars': 75.0,
            'mss_volume_ratio_min': 2.0,
            'max_bar_distance': 7,
            'require_killzone_alignment': True,
        }

    # Quality filters
    fvg_size = trade.get('fvg_size', 0)
    volume_ratio = trade.get('volume_ratio', 0)
    bar_distance = trade.get('bar_distance', 0)
    killzone_aligned = trade.get('killzone_aligned', False)
    pnl = trade.get('pnl', 0)

    # Check all quality requirements
    checks_passed = (
        fvg_size >= config['min_fvg_gap_size_dollars'] and
        volume_ratio >= config['mss_volume_ratio_min'] and
        bar_distance <= config['max_bar_distance'] and
        (killzone_aligned or not config['require_killzone_alignment']) and
        pnl > 0  # Must be profitable
    )

    return checks_passed


def generate_premium_training_data(
    data: pd.DataFrame,
    signals_df: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate premium-labeled training data.

    Filters trades and signals to only include premium-quality setups.

    Args:
        data: OHLCV data
        signals_df: All detected signals
        trades: All simulated trades
        config: Premium configuration dict

    Returns:
        (premium_signals, premium_trades) filtered DataFrames
    """
    print("\n🎯 Step 5: Generating premium-labeled training data...")

    if config is None:
        config = {
            'min_fvg_gap_size_dollars': 75.0,
            'mss_volume_ratio_min': 2.0,
            'max_bar_distance': 7,
            'require_killzone_alignment': True,
        }

    # Label trades as premium
    trades['is_premium'] = trades.apply(label_premium_trade, axis=1, config=config)

    premium_trades = trades[trades['is_premium']].copy()
    standard_trades = trades[~trades['is_premium']].copy()

    print(f"   Total trades: {len(trades)}")
    print(f"   Premium trades: {len(premium_trades)} ({len(premium_trades)/len(trades)*100:.1f}%)")
    print(f"   Standard trades: {len(standard_trades)}")

    # Filter signals to only include premium-quality signals
    # Get indices of premium trades
    premium_indices = set(premium_trades.index)

    # Filter signals that led to premium trades
    premium_signals = signals_df[signals_df.index.isin(premium_indices)].copy()

    print(f"   Premium signals: {len(premium_signals)}")

    return premium_signals, premium_trades


def main():
    parser = argparse.ArgumentParser(description='Generate ML training data')
    parser.add_argument('--premium', '-p', action='store_true',
                        help='Generate premium-labeled training data')
    parser.add_argument('--min-fvg-gap', type=float, default=75.0,
                        help='Minimum FVG gap size in dollars (default: 75)')
    parser.add_argument('--volume-ratio', type=float, default=2.0,
                        help='Minimum MSS volume ratio (default: 2.0)')
    parser.add_argument('--max-bar-distance', type=int, default=7,
                        help='Maximum bar distance for confluence (default: 7)')
    parser.add_argument('--no-killzone', action='store_true',
                        help='Do not require killzone alignment for premium')

    args = parser.parse_args()

    config_type = "PREMIUM" if args.premium else "STANDARD"
    print(f"🚀 GENERATING {config_type} ML TRAINING DATA - FULL DATASET")
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

    # Generate premium data if requested
    if args.premium:
        premium_config = {
            'min_fvg_gap_size_dollars': args.min_fvg_gap,
            'mss_volume_ratio_min': args.volume_ratio,
            'max_bar_distance': args.max_bar_distance,
            'require_killzone_alignment': not args.no_killzone,
        }

        premium_signals, premium_trades = generate_premium_training_data(
            data, signals_df, trades, premium_config
        )

        # Use premium data for training
        signals_df = premium_signals
        trades = premium_trades

    # Save ML training data
    print("\n💾 Step 6: Saving ML training data...")
    ml_dir = Path("data/ml_training")
    ml_dir.mkdir(parents=True, exist_ok=True)

    suffix = "premium" if args.premium else "full"
    signals_path = ml_dir / f"silver_bullet_signals_{suffix}.parquet"
    trades_path = ml_dir / f"silver_bullet_trades_{suffix}.parquet"
    metadata_path = ml_dir / f"metadata_{suffix}.json"

    signals_df.to_parquet(signals_path, index=True)
    trades.to_parquet(trades_path, index=False)

    metadata = {
        "date_range": {"start": "2023-12-01", "end": "2026-03-06"},
        "total_signals": int(len(signals_df)),
        "total_trades": int(len(trades)),
        "filters_applied": ["daily_bias", "volatility"],
        "strategy": f"silver_bullet_{suffix}",
        "premium_config": {
            "min_fvg_gap_size_dollars": args.min_fvg_gap if args.premium else None,
            "mss_volume_ratio_min": args.volume_ratio if args.premium else None,
            "max_bar_distance": args.max_bar_distance if args.premium else None,
            "require_killzone_alignment": not args.no_killzone if args.premium else None,
        } if args.premium else None
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ Saved {len(signals_df)} signals to {signals_path}")
    print(f"   ✅ Saved {len(trades)} trades to {trades_path}")
    print(f"   ✅ Saved metadata to {metadata_path}")

    # Calculate metrics
    print("\n📈 Step 7: Calculating performance metrics...")
    metrics = base_module.calculate_metrics(trades)

    print("\n" + "=" * 70)
    print(f"✅ {config_type} ML TRAINING DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Dataset Summary:")
    print(f"   Period: Dec 2023 - Mar 2026 (28 months)")
    print(f"   Type: {config_type}")
    print(f"   Signals: {len(signals_df):,}")
    print(f"   Trades: {len(trades):,}")
    print(f"   Total Return: {metrics['total_return']:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']:.2f}%")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics['max_drawdown']:.2f}%")
    print(f"\n💾 Next step: Run 'python train_meta_model.py{' --premium' if args.premium else ''}' to train meta-model")


if __name__ == '__main__':
    main()

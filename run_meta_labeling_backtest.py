#!/usr/bin/env python3
"""Run meta-labeling backtest comparing baseline vs ML-filtered signals.

This script performs A/B testing to validate the meta-labeling model's
impact on strategy performance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.silver_bullet_backtester import SilverBulletBacktester
from src.ml.signal_feature_extractor import SignalFeatureExtractor
from src.ml.inference import MLInference


# Reuse functions from run_optimized_silver_bullet.py
import run_optimized_silver_bullet as base_module


def apply_meta_labeling_filter(
    signals_df: pd.DataFrame,
    price_data: pd.DataFrame,
    ml_inference: MLInference,
    probability_threshold: float = 0.60
) -> pd.DataFrame:
    """Apply ML probability filtering to signals.

    Args:
        signals_df: DataFrame with signals (index is timestamps)
        price_data: DataFrame with OHLCV data
        ml_inference: MLInference instance
        probability_threshold: Minimum probability to keep signal

    Returns:
        Filtered signals DataFrame with probability column added
    """
    print(f"\n🤖 Applying Meta-Labeling Filter (P >= {probability_threshold:.2f})...")

    feature_extractor = SignalFeatureExtractor(lookback_bars=100)

    filtered_signals = []
    probabilities = []
    feature_extraction_failures = 0
    inference_failures = 0

    for i, (idx, signal) in enumerate(signals_df.iterrows()):
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(signals_df)} signals...")

        try:
            # Extract features at signal time
            features = feature_extractor.extract_features_at_signal_time(
                signal_timestamp=idx,
                price_data=price_data
            )

            # Convert to DataFrame
            features_df = pd.DataFrame([features])

            # Generate probability score
            try:
                probability = ml_inference.predict_probability_from_features(
                    features_df=features_df,
                    horizon=30
                )
                probabilities.append(probability)

                # Filter by threshold
                if probability >= probability_threshold:
                    filtered_signals.append(signal)

            except Exception as e:
                logger = __import__('logging').getLogger(__name__)
                logger.warning(f"Inference failed for signal at {idx}: {e}")
                inference_failures += 1
                # Conservative: assume low probability
                probabilities.append(0.0)

        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.warning(f"Feature extraction failed for signal at {idx}: {e}")
            feature_extraction_failures += 1
            # Conservative: assume low probability
            probabilities.append(0.0)

    # Create result DataFrame
    result_df = pd.DataFrame(filtered_signals)

    if len(result_df) > 0 and len(probabilities) > 0:
        # Add probability column (only for filtered signals)
        filtered_probs = [p for i, p in enumerate(probabilities)
                         if p >= probability_threshold]
        result_df['probability'] = filtered_probs[:len(result_df)]

    print(f"   Feature extraction failures: {feature_extraction_failures}")
    print(f"   Inference failures: {inference_failures}")
    print(f"   Filtered: {len(signals_df)} → {len(result_df)} signals ({len(result_df)/len(signals_df)*100:.1f}% retained)")

    return result_df


def run_baseline_backtest(data: pd.DataFrame, signals_df: pd.DataFrame) -> dict:
    """Run baseline backtest (no ML filtering).

    Args:
        data: Price data DataFrame
        signals_df: Signals DataFrame

    Returns:
        Dictionary with trades, metrics, and signal count
    """
    print("\n⚡ Running BASELINE backtest (no ML filtering)...")

    trades = base_module.simulate_trades_with_fvg_stops(data, signals_df)
    metrics = base_module.calculate_metrics(trades)

    return {
        'trades': trades,
        'metrics': metrics,
        'signal_count': len(signals_df)
    }


def run_meta_filtered_backtest(
    data: pd.DataFrame,
    signals_df: pd.DataFrame,
    ml_inference: MLInference,
    probability_threshold: float = 0.60
) -> dict:
    """Run meta-filtered backtest (with ML filtering).

    Args:
        data: Price data DataFrame
        signals_df: Signals DataFrame
        ml_inference: MLInference instance
        probability_threshold: Minimum probability threshold

    Returns:
        Dictionary with trades, metrics, signal counts, and filter rate
    """
    print(f"\n⚡ Running META-FILTERED backtest (P >= {probability_threshold:.2f})...")

    # Apply ML filter
    filtered_signals = apply_meta_labeling_filter(
        signals_df=signals_df,
        price_data=data,
        ml_inference=ml_inference,
        probability_threshold=probability_threshold
    )

    # Simulate trades
    trades = base_module.simulate_trades_with_fvg_stops(data, filtered_signals)
    metrics = base_module.calculate_metrics(trades)

    return {
        'trades': trades,
        'metrics': metrics,
        'signal_count': len(signals_df),
        'filtered_signal_count': len(filtered_signals),
        'filter_rate': 1 - (len(filtered_signals) / len(signals_df)) if len(signals_df) > 0 else 0
    }


def print_comparison_table(baseline_results: dict, meta_results: dict, threshold: float):
    """Print performance comparison table.

    Args:
        baseline_results: Results from baseline backtest
        meta_results: Results from meta-filtered backtest
        threshold: Probability threshold used
    """
    baseline_metrics = baseline_results['metrics']
    meta_metrics = meta_results['metrics']

    print(f"\n{'='*70}")
    print(f"PERFORMANCE COMPARISON (Threshold: {threshold:.2f})")
    print(f"{'='*70}")

    print(f"\n{'Metric':<20} {'Baseline':<15} {'Meta-Filtered':<15} {'Improvement':<15}")
    print("-" * 70)

    # Win Rate
    wr_baseline = baseline_metrics['win_rate']
    wr_meta = meta_metrics['win_rate']
    wr_diff = wr_meta - wr_baseline
    print(f"{'Win Rate':<20} {wr_baseline:<15.2f}% {wr_meta:<15.2f}% {wr_diff:+.2f}%")

    # Max Drawdown
    dd_baseline = baseline_metrics['max_drawdown']
    dd_meta = meta_metrics['max_drawdown']
    dd_diff = dd_meta - dd_baseline
    print(f"{'Max Drawdown':<20} {dd_baseline:<15.2f}% {dd_meta:<15.2f}% {dd_diff:+.2f}%")

    # Sharpe Ratio
    sr_baseline = baseline_metrics['sharpe_ratio']
    sr_meta = meta_metrics['sharpe_ratio']
    sr_diff = sr_meta - sr_baseline
    print(f"{'Sharpe Ratio':<20} {sr_baseline:<15.2f} {sr_meta:<15.2f} {sr_diff:+.2f}")

    # Total Trades
    tt_baseline = baseline_results['signal_count']
    tt_meta = meta_results['filtered_signal_count']
    print(f"{'Total Trades':<20} {tt_baseline:<15} {tt_meta:<15} {tt_meta - tt_baseline:+}")

    # Filter Rate
    filter_rate = meta_results['filter_rate']
    print(f"{'Filter Rate':<20} {'N/A':<15} {filter_rate:<15.1%} {'N/A':<15}")


def main():
    """Run meta-labeling backtest A/B test."""

    print("🚀 META-LABELING BACKTEST")
    print("=" * 70)
    print("A/B Testing: Baseline vs ML-Filtered Signals")
    print("=" * 70)

    # Load data
    print("\n📊 Step 1: Loading data...")
    data = base_module.load_time_bars('2024-10-01', '2025-03-05')

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
    print(f"✅ Daily bias calculated: {daily_bias.sum():.0f} uptrend days, {(~daily_bias).sum():.0f} downtrend days")

    # Run pattern detection
    print("\n🎯 Step 3: Running Silver Bullet pattern detection...")
    backtester = SilverBulletBacktester(
        mss_lookback=3,
        fvg_min_gap=0.25,
        max_bar_distance=10,
        min_confidence=60.0,
        enable_time_windows=True,
        require_sweep=False,
    )

    signals_df = backtester.run_backtest(data)

    # Deduplicate
    signals_df = signals_df.sort_values('confidence', ascending=False)
    signals_df = signals_df[~signals_df.index.duplicated(keep='first')]

    # Apply existing filters
    signals_df = base_module.add_daily_bias_filter(signals_df, daily_bias)

    if len(signals_df) == 0:
        print("\n❌ No signals after daily bias filter!")
        return

    signals_df = base_module.add_volatility_filter(data, signals_df, min_atr_pct=0.003)

    if len(signals_df) == 0:
        print("\n❌ No signals after volatility filter!")
        return

    print(f"✅ Total signals after filters: {len(signals_df)}")

    # Run baseline backtest
    baseline_results = run_baseline_backtest(data, signals_df)

    print(f"\n📊 BASELINE Results:")
    print(f"   Trades: {len(baseline_results['trades'])}")
    print(f"   Win Rate: {baseline_results['metrics']['win_rate']:.2f}%")
    print(f"   Sharpe: {baseline_results['metrics']['sharpe_ratio']:.2f}")
    print(f"   Max DD: {baseline_results['metrics']['max_drawdown']:.2f}%")

    # Load ML model
    print("\n🤖 Step 4: Loading ML meta-model...")
    try:
        ml_inference = MLInference(model_dir='data/models/xgboost')
        print("✅ ML model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        print("   Run: python train_meta_model.py")
        return

    # Test different probability thresholds
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]

    print(f"\n⚡ Step 5: Testing different probability thresholds...")

    results_summary = []

    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"Testing Threshold: {threshold:.2f}")
        print(f"{'='*70}")

        try:
            meta_results = run_meta_filtered_backtest(
                data=data,
                signals_df=signals_df,
                ml_inference=ml_inference,
                probability_threshold=threshold
            )

            print_comparison_table(baseline_results, meta_results, threshold)

            # Store results for summary
            results_summary.append({
                'threshold': threshold,
                'win_rate': meta_results['metrics']['win_rate'],
                'sharpe': meta_results['metrics']['sharpe_ratio'],
                'max_dd': meta_results['metrics']['max_drawdown'],
                'trades': meta_results['filtered_signal_count'],
                'filter_rate': meta_results['filter_rate']
            })

        except Exception as e:
            print(f"❌ Error testing threshold {threshold:.2f}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary table
    if results_summary:
        print(f"\n{'='*70}")
        print("SUMMARY: All Thresholds")
        print(f"{'='*70}")

        print(f"\n{'Threshold':<12} {'Trades':<10} {'Filter%':<10} {'WinRate':<10} {'Sharpe':<10} {'MaxDD':<10}")
        print("-" * 70)

        baseline_str = f"Baseline"
        print(f"{baseline_str:<12} {baseline_results['signal_count']:<10} {0.0:<10.1%} {baseline_results['metrics']['win_rate']:<10.2f} {baseline_results['metrics']['sharpe_ratio']:<10.2f} {baseline_results['metrics']['max_drawdown']:<10.2f}")

        for r in results_summary:
            print(f"{r['threshold']:<12.2f} {r['trades']:<10} {r['filter_rate']:<10.1%} {r['win_rate']:<10.2f} {r['sharpe']:<10.2f} {r['max_dd']:<10.2f}")

    print(f"\n{'='*70}")
    print("✅ META-LABELING BACKTEST COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

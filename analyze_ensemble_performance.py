#!/usr/bin/env python
"""Analyze ensemble performance and signal distribution before adjusting thresholds."""

import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_ensemble_signals():
    """Analyze ensemble signal distribution and composite confidence scores."""
    logger.info("=" * 80)
    logger.info("ENSEMBLE SIGNAL DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)

    # Import required modules
    from src.research.ensemble_backtester import EnsembleBacktester, create_dollar_bar_from_series
    from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator
    from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

    # Initialize components
    logger.info("\n1. Initializing ensemble components...")
    aggregator = EnsembleSignalAggregator(max_lookback=10)

    # Create config
    config = {
        "ensemble": {
            "strategies": {
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20,
            },
            "confidence_threshold": 0.50,
            "minimum_strategies": 2,
        },
    }

    config_path = Path("/tmp/analysis_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    scorer = WeightedConfidenceScorer(config_path=str(config_path))
    logger.info("✓ Components initialized")

    # Load 2024 data
    logger.info("\n2. Loading 2024 data...")
    data_dir = Path("data/processed/dollar_bars")
    h5_files = sorted(data_dir.glob("MNQ_dollar_bars_2024*.h5"))

    all_bars = []
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            all_bars.append(f["dollar_bars"][:])

    combined_data = np.vstack(all_bars)
    timestamps_ms = combined_data[:, 0].astype(np.int64)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps_ms, unit="ms"),
        "open": combined_data[:, 1],
        "high": combined_data[:, 2],
        "low": combined_data[:, 3],
        "close": combined_data[:, 4],
        "volume": combined_data[:, 5].astype(int),
    })

    logger.info(f"✓ Loaded {len(df)} bars from {len(h5_files)} files")

    # Initialize strategies
    logger.info("\n3. Initializing strategies...")
    from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
    from src.detection.wolf_pack_strategy import WolfPackStrategy
    from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
    from src.detection.vwap_bounce_strategy import VWAPBounceStrategy
    from src.detection.opening_range_strategy import OpeningRangeStrategy

    strategies = []
    strategies.append(("triple_confluence", TripleConfluenceStrategy(config={})))
    strategies.append(("wolf_pack", WolfPackStrategy(tick_size=0.25, risk_ticks=20, min_confidence=0.8)))
    strategies.append(("adaptive_ema", AdaptiveEMAStrategy()))
    strategies.append(("vwap_bounce", VWAPBounceStrategy(config={})))
    strategies.append(("opening_range", OpeningRangeStrategy(config={})))
    logger.info(f"✓ Initialized {len(strategies)} strategies")

    # Track statistics
    logger.info("\n4. Processing bars and tracking signals...")

    stats = {
        "total_bars": 0,
        "strategy_signals": defaultdict(int),
        "aggregator_signals_per_bar": [],
        "unique_strategies_per_bar": [],
        "composite_confidence_scores": [],
        "signals_above_threshold": {0.25: 0, 0.30: 0, 0.40: 0, 0.50: 0},
        "strategy_combinations": defaultdict(int),
        "max_composite_confidence": 0.0,
        "bars_with_3_plus_strategies": 0,
        "bars_with_2_strategies": 0,
        "bars_with_1_strategy": 0,
        "bars_with_0_strategies": 0,
    }

    # Process sample of data (every 10th bar for speed)
    sample_bars = df.iloc[::10].reset_index(drop=True)
    logger.info(f"   Sampling {len(sample_bars)} bars (every 10th bar)")

    for idx, row in sample_bars.iterrows():
        stats["total_bars"] += 1

        # Convert to DollarBar
        dollar_bar = create_dollar_bar_from_series(row)

        # Process with each strategy
        for strategy_name, strategy in strategies:
            try:
                if hasattr(strategy, 'process_bar'):
                    signal = strategy.process_bar(dollar_bar)
                elif hasattr(strategy, 'process_bars'):
                    signal = strategy.process_bars([dollar_bar])
                else:
                    continue

                if signal is not None:
                    stats["strategy_signals"][strategy_name] += 1

                    # Normalize and add to aggregator
                    from src.detection.ensemble_signal_aggregator import normalize_signal
                    try:
                        normalized_signal = normalize_signal(signal)
                        aggregator.add_signal(normalized_signal)
                    except:
                        continue

            except Exception as e:
                continue

        # Get aggregated signals for current bar (window_bars=5)
        current_bar_signals = aggregator.get_signals_for_bar(
            dollar_bar.timestamp, window_bars=5
        )

        # Track unique strategies
        unique_strategies = set(s.strategy_name for s in current_bar_signals)
        num_unique = len(unique_strategies)

        stats["unique_strategies_per_bar"].append(num_unique)
        stats["aggregator_signals_per_bar"].append(len(current_bar_signals))

        if num_unique == 0:
            stats["bars_with_0_strategies"] += 1
        elif num_unique == 1:
            stats["bars_with_1_strategy"] += 1
        elif num_unique == 2:
            stats["bars_with_2_strategies"] += 1
        else:
            stats["bars_with_3_plus_strategies"] += 1

        # Track strategy combinations
        if num_unique >= 2:
            combo = tuple(sorted(unique_strategies))
            stats["strategy_combinations"][combo] += 1

        # Score signals if we have multiple
        if current_bar_signals and num_unique >= 2:
            ensemble_signal = scorer.score_signals(current_bar_signals)
            if ensemble_signal is not None:
                composite_conf = ensemble_signal.composite_confidence
                stats["composite_confidence_scores"].append(composite_conf)

                # Track max
                if composite_conf > stats["max_composite_confidence"]:
                    stats["max_composite_confidence"] = composite_conf

                # Count above thresholds
                for threshold in [0.25, 0.30, 0.40, 0.50]:
                    if composite_conf >= threshold:
                        stats["signals_above_threshold"][threshold] += 1

        # Progress
        if (idx + 1) % 1000 == 0:
            logger.info(f"   Processed {idx + 1}/{len(sample_bars)} bars")

    # Print results
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("ANALYSIS RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nTotal bars analyzed: {stats['total_bars']}")
    logger.info(f"\n--- Strategy Signal Generation ---")
    for strategy, count in stats["strategy_signals"].items():
        logger.info(f"  {strategy}: {count} signals")

    logger.info(f"\n--- Signals Per Bar ---")
    logger.info(f"  Mean aggregator signals: {np.mean(stats['aggregator_signals_per_bar']):.2f}")
    logger.info(f"  Mean unique strategies: {np.mean(stats['unique_strategies_per_bar']):.2f}")

    logger.info(f"\n--- Distribution of Unique Strategies Per Bar ---")
    logger.info(f"  0 strategies: {stats['bars_with_0_strategies']} ({stats['bars_with_0_strategies']/stats['total_bars']*100:.1f}%)")
    logger.info(f"  1 strategy:  {stats['bars_with_1_strategy']} ({stats['bars_with_1_strategy']/stats['total_bars']*100:.1f}%)")
    logger.info(f"  2 strategies: {stats['bars_with_2_strategies']} ({stats['bars_with_2_strategies']/stats['total_bars']*100:.1f}%)")
    logger.info(f"  3+ strategies: {stats['bars_with_3_plus_strategies']} ({stats['bars_with_3_plus_strategies']/stats['total_bars']*100:.1f}%)")

    logger.info(f"\n--- Composite Confidence Scores ---")
    if stats["composite_confidence_scores"]:
        logger.info(f"  Total ensemble signals: {len(stats['composite_confidence_scores'])}")
        logger.info(f"  Mean: {np.mean(stats['composite_confidence_scores']):.4f}")
        logger.info(f"  Median: {np.median(stats['composite_confidence_scores']):.4f}")
        logger.info(f"  Max: {stats['max_composite_confidence']:.4f}")
        logger.info(f"  Std Dev: {np.std(stats['composite_confidence_scores']):.4f}")

        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        logger.info(f"\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(stats['composite_confidence_scores'], p)
            logger.info(f"    {p}th percentile: {val:.4f}")

    logger.info(f"\n--- Signals Above Threshold ---")
    for threshold, count in stats["signals_above_threshold"].items():
        pct = count / stats['total_bars'] * 100 if stats['total_bars'] > 0 else 0
        logger.info(f"  ≥{threshold:.0%}: {count} signals ({pct:.2f}% of bars)")

    logger.info(f"\n--- Top 10 Strategy Combinations ---")
    sorted_combos = sorted(stats["strategy_combinations"].items(), key=lambda x: x[1], reverse=True)[:10]
    for combo, count in sorted_combos:
        logger.info(f"  {' + '.join(combo)}: {count} occurrences")

    logger.info(f"\n")
    logger.info("=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    if stats['bars_with_3_plus_strategies'] > 0:
        pct_3plus = stats['bars_with_3_plus_strategies'] / stats['total_bars'] * 100
        logger.info(f"✓ 3+ strategy confluence occurs: {pct_3plus:.2f}% of bars")
        logger.info(f"  → Current 50% threshold may be appropriate")
    else:
        logger.info(f"✗ 3+ strategy confluence: Rare (0 occurrences)")
        logger.info(f"  → Consider lowering threshold to 25-30%")

    if stats['bars_with_2_strategies'] > 0:
        pct_2 = stats['bars_with_2_strategies'] / stats['total_bars'] * 100
        logger.info(f"✓ 2-strategy confluence occurs: {pct_2:.2f}% of bars")

    if stats['max_composite_confidence'] > 0:
        logger.info(f"\nMax composite confidence: {stats['max_composite_confidence']:.4f}")

        if stats['max_composite_confidence'] < 0.30:
            logger.info(f"  → RECOMMENDATION: Lower threshold to 20-25%")
            logger.info(f"  → Reason: Even best signals below 30%")
        elif stats['max_composite_confidence'] < 0.40:
            logger.info(f"  → RECOMMENDATION: Lower threshold to 30-35%")
            logger.info(f"  → Reason: Best signals in 30-40% range")
        elif stats['max_composite_confidence'] < 0.50:
            logger.info(f"  → CONSIDER: Lower threshold to 40%")
            logger.info(f"  → Reason: Best signals approaching 50%")
        else:
            logger.info(f"  → Current 50% threshold may be appropriate")

    logger.info("")


if __name__ == "__main__":
    try:
        analyze_ensemble_signals()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

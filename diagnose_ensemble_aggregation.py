#!/usr/bin/env python
"""Diagnose why ensemble signals aren't becoming trades."""

import logging
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.DEBUG,  # Enable debug logging
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_small_test_dataset():
    """Create a small test dataset for quick debugging."""
    logger.info("Creating small test dataset...")

    # Load just one month of data
    data_dir = Path("data/processed/dollar_bars")
    h5_file = data_dir / "MNQ_dollar_bars_202401.h5"

    with h5py.File(h5_file, "r") as f:
        dollar_bars = f["dollar_bars"][:1000]  # Just 1000 bars for quick testing

    timestamps_ms = dollar_bars[:, 0].astype(np.int64)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps_ms, unit="ms"),
        "open": dollar_bars[:, 1],
        "high": dollar_bars[:, 2],
        "low": dollar_bars[:, 3],
        "close": dollar_bars[:, 4],
        "volume": dollar_bars[:, 5].astype(int),
    })

    # Save to HDF5
    temp_path = Path("/tmp/debug_ensemble_test.h5")
    with h5py.File(temp_path, "w") as f:
        timestamps_ns = df["timestamp"].astype("datetime64[ns]").astype(np.int64)
        f.create_dataset("timestamps", data=timestamps_ns.values)
        f.create_dataset("open", data=df["open"].values)
        f.create_dataset("high", data=df["high"].values)
        f.create_dataset("low", data=df["low"].values)
        f.create_dataset("close", data=df["close"].values)
        f.create_dataset("volume", data=df["volume"].values)

    logger.info(f"✓ Created test dataset: {len(df)} bars")
    return str(temp_path)


def create_config():
    """Create ensemble config."""
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
            "minimum_strategies": 1,
        },
        "risk": {
            "max_position_size": 5,
            "risk_reward_ratio": 2.0,
            "max_risk_per_trade": 0.02,
        },
    }

    config_path = Path("/tmp/debug_ensemble_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


def diagnose_ensemble_flow(config_path, data_path):
    """Diagnose the ensemble signal flow step by step."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("DIAGNOSING ENSEMBLE SIGNAL FLOW")
    logger.info("=" * 80)

    # Import required modules
    from src.research.ensemble_backtester import EnsembleBacktester, create_dollar_bar_from_series
    from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator
    from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

    # Initialize components
    logger.info("\n1. Initializing ensemble components...")
    aggregator = EnsembleSignalAggregator(max_lookback=10)
    scorer = WeightedConfidenceScorer(config_path=config_path)
    logger.info("✓ Components initialized")

    # Load data
    logger.info("\n2. Loading test data...")
    with h5py.File(data_path, "r") as f:
        timestamps = pd.to_datetime(f["timestamps"][:], unit="ns")
        open_prices = f["open"][:]
        high_prices = f["high"][:]
        low_prices = f["low"][:]
        close_prices = f["close"][:]
        volumes = f["volume"][:]

    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })

    logger.info(f"✓ Loaded {len(bars)} bars")

    # Initialize strategies
    logger.info("\n3. Initializing strategies...")
    from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
    from src.detection.wolf_pack_strategy import WolfPackStrategy
    from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
    from src.detection.vwap_bounce_strategy import VWAPBounceStrategy
    from src.detection.opening_range_strategy import OpeningRangeStrategy

    strategies = []

    tc = TripleConfluenceStrategy(config={})
    strategies.append(("triple_confluence", tc))
    logger.info("  ✓ Triple Confluence initialized")

    wp = WolfPackStrategy(tick_size=0.25, risk_ticks=20, min_confidence=0.8)
    strategies.append(("wolf_pack", wp))
    logger.info("  ✓ Wolf Pack initialized")

    ae = AdaptiveEMAStrategy()
    strategies.append(("adaptive_ema", ae))
    logger.info("  ✓ Adaptive EMA initialized")

    vb = VWAPBounceStrategy(config={})
    strategies.append(("vwap_bounce", vb))
    logger.info("  ✓ VWAP Bounce initialized")

    ob = OpeningRangeStrategy(config={})
    strategies.append(("opening_range", ob))
    logger.info("  ✓ Opening Range initialized")

    # Process bars and track signals
    logger.info("\n4. Processing bars and tracking signals...")

    strategy_signals_count = 0
    aggregator_signals_count = 0
    ensemble_signals_count = 0
    below_threshold_count = 0

    for idx, row in bars.iterrows():
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
                    strategy_signals_count += 1

                    # Normalize signal to Ensemble format before adding to aggregator
                    from src.detection.ensemble_signal_aggregator import normalize_signal

                    try:
                        normalized_signal = normalize_signal(signal)
                        aggregator.add_signal(normalized_signal)
                        aggregator_signals_count += 1
                        logger.debug(f"Added normalized signal to aggregator: {strategy_name}, confidence: {signal.confidence:.2f}")
                    except Exception as e:
                        logger.debug(f"Failed to normalize signal from {strategy_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Strategy {strategy_name} error: {e}")
                continue

        # Get aggregated signals for current bar (with wider window)
        current_bar_signals = aggregator.get_signals_for_bar(
            dollar_bar.timestamp, window_bars=5  # Include 5 bars before and after
        )

        if current_bar_signals:
            logger.info(f"\n  Bar {idx}: {len(current_bar_signals)} aggregated signals")

            # Debug: Show signal details
            for sig in current_bar_signals:
                logger.info(f"    Signal: strategy='{sig.strategy_name}', confidence={sig.confidence:.2f}, direction={sig.direction}")

            # Score signals
            ensemble_signal = scorer.score_signals(current_bar_signals)

            if ensemble_signal is not None:
                ensemble_signals_count += 1
                logger.info(f"    → Ensemble signal: confidence={ensemble_signal.composite_confidence:.2f}")
                logger.info(f"    → Contributing strategies: {ensemble_signal.contributing_strategies}")

                if ensemble_signal.composite_confidence < 0.50:
                    below_threshold_count += 1
                    logger.info(f"    ✗ Below threshold (0.50)")
                else:
                    logger.info(f"    ✓ Above threshold - should trade!")
            else:
                logger.debug(f"    ✗ Scorer returned None")

        # Print progress every 100 bars
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(bars)} bars")
            logger.info(f"    Strategy signals: {strategy_signals_count}")
            logger.info(f"    Aggregator signals: {aggregator_signals_count}")
            logger.info(f"    Ensemble signals: {ensemble_signals_count}")
            logger.info(f"    Below threshold: {below_threshold_count}")

    # Summary
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total bars processed: {len(bars)}")
    logger.info(f"Strategy signals generated: {strategy_signals_count}")
    logger.info(f"Signals added to aggregator: {aggregator_signals_count}")
    logger.info(f"Ensemble signals created: {ensemble_signals_count}")
    logger.info(f"Signals below threshold: {below_threshold_count}")
    logger.info("")

    if strategy_signals_count == 0:
        logger.error("❌ ISSUE: No strategy signals generated")
        logger.error("   Strategies may not be finding patterns in this data")
    elif aggregator_signals_count == 0:
        logger.error("❌ ISSUE: No signals in aggregator")
        logger.error("   Signals generated but not added to aggregator")
        logger.error("   Check aggregator.add_signal() method")
    elif ensemble_signals_count == 0:
        logger.error("❌ ISSUE: No ensemble signals created")
        logger.error("   Signals in aggregator but scorer returned None")
        logger.error("   Check scorer.score_signals() method")
    elif below_threshold_count == ensemble_signals_count:
        logger.error("❌ ISSUE: All ensemble signals below threshold")
        logger.error("   Composite confidence < 0.50 for all signals")
        logger.error("   Check how composite confidence is calculated")
    else:
        logger.info("✓ Ensemble system working - signals should generate trades")

    logger.info("")
    logger.info("RECOMMENDATIONS:")
    logger.info("1. Check if aggregator is storing signals correctly")
    logger.info("2. Check if scorer is combining signals correctly")
    logger.info("3. Check composite confidence calculation")
    logger.info("4. Verify signal timestamps match bar timestamps")
    logger.info("5. Check if strategies need specific bar lookback periods")


def main():
    """Main diagnostic."""
    logger.info("ENSEMBLE SIGNAL FLOW DIAGNOSTIC")
    logger.info("=" * 80)

    try:
        # Create test data
        data_path = create_small_test_dataset()

        # Create config
        config_path = create_config()

        # Diagnose
        diagnose_ensemble_flow(config_path, data_path)

        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

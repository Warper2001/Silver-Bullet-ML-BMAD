#!/usr/bin/env python
"""Diagnose why strategies are generating zero confidence signals."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_bars(n_bars=500):
    """Load sample bars for testing."""
    data_dir = Path("data/processed/dollar_bars")
    h5_file = data_dir / "MNQ_dollar_bars_202401.h5"

    with h5py.File(h5_file, "r") as f:
        dollar_bars = f["dollar_bars"][:n_bars]

    timestamps_ms = dollar_bars[:, 0].astype(np.int64)

    bars = []
    for i in range(len(dollar_bars)):
        bar = {
            "timestamp": datetime.fromtimestamp(timestamps_ms[i] / 1000, tz=timezone.utc),
            "open": float(dollar_bars[i, 1]),
            "high": float(dollar_bars[i, 2]),
            "low": float(dollar_bars[i, 3]),
            "close": float(dollar_bars[i, 4]),
            "volume": int(dollar_bars[i, 5]),
        }
        bars.append(bar)

    return bars


def test_triple_confluence(bars):
    """Test Triple Confluence strategy."""
    logger.info("=" * 80)
    logger.info("TESTING: Triple Confluence Strategy")
    logger.info("=" * 80)

    try:
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        strategy = TripleConfluenceStrategy(config={})

        signals = []
        for bar in bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        logger.info(f"✓ Generated {len(signals)} signals")

        if signals:
            logger.info(f"Sample signals:")
            for i, sig in enumerate(signals[:5], 1):
                logger.info(f"  Signal {i}:")
                logger.info(f"    Direction: {sig.direction}")
                logger.info(f"    Entry Price: {sig.entry_price}")
                logger.info(f"    Stop Loss: {sig.stop_loss}")
                logger.info(f"    Take Profit: {sig.take_profit}")
                logger.info(f"    Confidence: {sig.confidence}")

            # Check confidence values
            confidences = [s.confidence for s in signals]
            logger.info(f"Confidence stats:")
            logger.info(f"  Min: {min(confidences):.4f}")
            logger.info(f"  Max: {max(confidences):.4f}")
            logger.info(f"  Mean: {np.mean(confidences):.4f}")
            logger.info(f"  Non-zero: {sum(1 for c in confidences if c > 0)} / {len(confidences)}")

        return len(signals)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def test_ensemble_aggregation(bars):
    """Test ensemble signal aggregation."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TESTING: Ensemble Signal Aggregation")
    logger.info("=" * 80)

    try:
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator(max_lookback=10)

        # Create some fake signals to test aggregation
        from src.detection.models import TripleConfluenceSignal

        test_signals = []
        for i, bar in enumerate(bars[:10]):
            if i % 2 == 0:  # Create signal every other bar
                signal = TripleConfluenceSignal(
                    timestamp=bar["timestamp"],
                    direction="long" if i % 4 == 0 else "short",
                    entry_price=bar["close"],
                    stop_loss=bar["close"] * 0.998,
                    take_profit=bar["close"] * 1.004,
                    confidence=0.75,  # Non-zero confidence
                    level_sweep_detected=True,
                    fvg_detected=True,
                    vwap_alignment=True,
                )
                test_signals.append(signal)

        logger.info(f"Created {len(test_signals)} test signals")

        # Aggregate signals
        ensemble_signals = []
        for signal in test_signals:
            result = aggregator.aggregate_signal(signal, strategy_name="triple_confluence")
            if result:
                ensemble_signals.append(result)

        logger.info(f"✓ Aggregated to {len(ensemble_signals)} ensemble signals")

        if ensemble_signals:
            logger.info(f"Sample ensemble signals:")
            for i, sig in enumerate(ensemble_signals[:3], 1):
                logger.info(f"  Signal {i}:")
                logger.info(f"    Direction: {sig.direction}")
                logger.info(f"    Confidence: {sig.confidence}")

            # Check confidence values
            confidences = [s.confidence for s in ensemble_signals]
            logger.info(f"Confidence stats:")
            logger.info(f"  Min: {min(confidences):.4f}")
            logger.info(f"  Max: {max(confidences):.4f}")
            logger.info(f"  Mean: {np.mean(confidences):.4f}")

        return len(ensemble_signals)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Main diagnostic."""
    logger.info("STRATEGY SIGNAL DIAGNOSTIC")
    logger.info("=" * 80)

    # Load sample bars
    logger.info("Loading sample bars...")
    bars = load_sample_bars(n_bars=500)
    logger.info(f"✓ Loaded {len(bars)} bars")
    logger.info(f"  Date range: {bars[0]['timestamp']} to {bars[-1]['timestamp']}")
    logger.info(f"  Price range: ${min(b['close'] for b in bars):.2f} to ${max(b['close'] for b in bars):.2f}")

    # Test individual strategies
    tc_signals = test_triple_confluence(bars)

    # Test ensemble aggregation
    ensemble_signals = test_ensemble_aggregation(bars)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Triple Confluence signals: {tc_signals}")
    logger.info(f"Ensemble aggregated signals: {ensemble_signals}")
    logger.info("")

    if tc_signals == 0:
        logger.error("❌ ISSUE: Triple Confluence generated NO signals")
        logger.error("   This suggests the strategy logic may be too restrictive")
        logger.error("   OR the data doesn't match expected patterns")

    if ensemble_signals == 0:
        logger.error("❌ ISSUE: Ensemble aggregation produced NO signals")
        logger.error("   This suggests the aggregator may not be working correctly")

    logger.info("")
    logger.info("RECOMMENDATIONS:")
    logger.info("1. Check individual strategy signal generation")
    logger.info("2. Verify strategy parameters (lookback, thresholds)")
    logger.info("3. Test with different market conditions")
    logger.info("4. Consider relaxing strategy constraints for testing")
    logger.info("5. Add debug logging to understand signal generation")


if __name__ == "__main__":
    main()

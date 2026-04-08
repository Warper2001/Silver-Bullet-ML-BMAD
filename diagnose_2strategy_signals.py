#!/usr/bin/env python
"""Debug why 2-strategy confluence isn't generating ensemble signals."""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("DEBUGGING 2-STRATEGY SIGNALS")
    logger.info("=" * 80)

    from src.research.ensemble_backtester import create_dollar_bar_from_series
    from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator, normalize_signal
    from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer
    import h5py
    import numpy as np
    import pandas as pd
    import yaml

    # Initialize
    aggregator = EnsembleSignalAggregator(max_lookback=10)

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

    config_path = Path("/tmp/debug_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    scorer = WeightedConfidenceScorer(config_path=str(config_path))

    # Load small sample
    data_file = Path('data/processed/dollar_bars/MNQ_dollar_bars_202401.h5')
    with h5py.File(data_file, 'r') as f:
        dollar_bars = f['dollar_bars'][:2000]

    timestamps_ms = dollar_bars[:, 0].astype(np.int64)
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps_ms, unit='ms'),
        'open': dollar_bars[:, 1],
        'high': dollar_bars[:, 2],
        'low': dollar_bars[:, 3],
        'close': dollar_bars[:, 4],
        'volume': dollar_bars[:, 5].astype(int),
    })

    logger.info(f"Loaded {len(df)} bars")

    # Initialize strategies
    from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
    from src.detection.opening_range_strategy import OpeningRangeStrategy

    strategies = [
        ("triple_confluence", TripleConfluenceStrategy(config={})),
        ("opening_range", OpeningRangeStrategy(config={})),
    ]

    # Track 2-strategy events
    two_strategy_events = []

    for idx, row in df.iterrows():
        dollar_bar = create_dollar_bar_from_series(row)

        # Process strategies
        for strategy_name, strategy in strategies:
            try:
                signal = strategy.process_bar(dollar_bar)
                if signal is not None:
                    normalized = normalize_signal(signal)
                    aggregator.add_signal(normalized)
            except:
                pass

        # Check for 2-strategy confluence
        signals = aggregator.get_signals_for_bar(dollar_bar.timestamp, window_bars=5)
        unique_strategies = set(s.strategy_name for s in signals)

        if len(unique_strategies) >= 2:
            two_strategy_events.append({
                'bar': idx,
                'timestamp': dollar_bar.timestamp,
                'signals': signals,
                'unique_strategies': unique_strategies,
            })

            # Try scoring
            ensemble_signal = scorer.score_signals(signals)

            logger.info(f"\nBar {idx}: {len(unique_strategies)} strategies")
            for sig in signals:
                logger.info(f"  - {sig.strategy_name}: {sig.direction}, conf={sig.confidence:.2f}")

            if ensemble_signal is None:
                logger.info(f"  → Scorer returned None")
                # Check why
                directions = [s.direction for s in signals]
                if len(set(directions)) > 1:
                    logger.info(f"  → REASON: Direction mismatch ({directions})")
            else:
                logger.info(f"  → SUCCESS: composite_conf={ensemble_signal.composite_confidence:.4f}")

            if len(two_strategy_events) >= 5:  # Show first 5 examples
                break

    logger.info(f"\nTotal 2-strategy events found: {len(two_strategy_events)}")


if __name__ == "__main__":
    main()

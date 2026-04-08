"""Quick test of data loading and strategy signal generation."""

import sys
from pathlib import Path
from datetime import datetime, timezone
import logging

import h5py
import numpy as np

from src.data.models import DollarBar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dollar_bars_fixed(data_path: str = "data/processed/dollar_bars/") -> list:
    """Load dollar bars with correct structure.

    Columns are: [timestamp(ns), open, high, low, close, volume, notional]
    """
    logger.info(f"Loading bars from {data_path}")

    path = Path(data_path)
    h5_files = sorted(path.glob("*.h5"))

    all_bars = []

    # Load just one file for quick testing
    h5_file = h5_files[0]  # Use first file
    logger.info(f"Loading {h5_file.name}...")

    with h5py.File(h5_file, 'r') as f:
        ds = f['dollar_bars']

        # Columns: [timestamp(ns), open, high, low, close, volume, notional]
        for i in range(min(len(ds), 100)):  # Load first 100 bars for testing
            try:
                # Convert millisecond timestamp to datetime
                ts_ms = ds[i, 0]
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

                bar = DollarBar(
                    timestamp=ts,
                    open=float(ds[i, 1]),
                    high=float(ds[i, 2]),
                    low=float(ds[i, 3]),
                    close=float(ds[i, 4]),
                    volume=int(ds[i, 5]),
                    notional_value=float(ds[i, 6]),
                    is_forward_filled=False,  # Not stored in HDF5
                )
                all_bars.append(bar)

            except Exception as e:
                logger.warning(f"Error loading bar {i}: {e}")

    logger.info(f"Loaded {len(all_bars)} bars")
    return all_bars


def test_strategy():
    """Test strategy with loaded bars."""
    bars = load_dollar_bars_fixed()

    if not bars:
        logger.error("No bars loaded!")
        return

    logger.info(f"Date range: {bars[0].timestamp} to {bars[-1].timestamp}")

    # Test one strategy
    try:
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        logger.info("Testing Triple Confluence Scalper...")

        # Pass empty config dict for defaults
        strategy = TripleConfluenceStrategy(config={})

        signals = 0
        for i, bar in enumerate(bars[:50]):  # Test first 50 bars
            signal = strategy.process_bar(bar)
            if signal:
                signals += 1
                logger.info(f"Signal {signals}: {signal.direction} @ {bar.timestamp}")

        logger.info(f"Total signals: {signals}")

    except Exception as e:
        logger.error(f"Strategy error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_strategy()

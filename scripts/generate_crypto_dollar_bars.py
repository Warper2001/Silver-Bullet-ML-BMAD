#!/usr/bin/env python3
"""
Crypto Dollar Bar Generation Script

This script generates dollar bars from raw Binance kline data.

Dollar bars aggregate based on notional value traded rather than time.
This provides better signal quality for ML models by:
- Normalizing for volatility
- Reducing noise during low-volume periods
- Aligning bars with economic activity

Features:
- Configurable dollar bar threshold (default: $10M for crypto)
- State persistence (save every 10 seconds)
- Crash recovery (load from snapshot on restart)
- OHLCV aggregation

Usage:
    python scripts/generate_crypto_dollar_bars.py --input data/binance/historical/BTCUSDT_5m_365days.parquet

Reference: src/data/transformation.py
"""

import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DollarBarGenerator:
    """
    Generate dollar bars from kline data.

    Attributes:
        dollar_bar_threshold: Notional value threshold for bar completion
        state_file: Path to state persistence file
        save_interval: Seconds between state saves
    """

    def __init__(
        self,
        dollar_bar_threshold: float = 10_000_000.0,  # $10M for crypto
        state_file: Optional[str] = None,
        save_interval: int = 10,
    ) -> None:
        """
        Initialize dollar bar generator.

        Args:
            dollar_bar_threshold: Notional value threshold (USD)
            state_file: Path to state persistence file
            save_interval: Seconds between state saves
        """
        self.dollar_bar_threshold = dollar_bar_threshold
        self.state_file = state_file
        self.save_interval = save_interval

        # State for accumulation
        self.current_bar = None
        self.accumulated_notional = 0.0
        self.bars = []

        # State tracking for recovery
        self.last_save_time = None
        self.processed_rows = 0

        logger.info(f"Dollar bar threshold: ${dollar_bar_threshold:,.0f}")

    def generate_dollar_bars(
        self,
        df: pd.DataFrame,
        save_state: bool = True,
    ) -> pd.DataFrame:
        """
        Generate dollar bars from kline DataFrame.

        Args:
            df: DataFrame with kline data (columns: open_time, open, high, low, close, volume)
            save_state: Whether to save state for crash recovery

        Returns:
            DataFrame with dollar bars
        """
        logger.info(f"Generating dollar bars from {len(df)} klines...")

        # Load state if available
        if save_state and self.state_file:
            self._load_state()

        # Resume from last processed row
        start_row = self.processed_rows
        logger.info(f"Starting from row {start_row}")

        # Process each kline
        for idx, row in df.iloc[start_row:].iterrows():
            self._process_row(row)

            # Update processed count
            self.processed_rows += 1

            # Save state periodically
            if save_state and self.state_file:
                self._maybe_save_state()

        # Flush any remaining bar
        if self.current_bar is not None:
            self.bars.append(self.current_bar)

        # Create DataFrame from bars
        bars_df = pd.DataFrame(self.bars)

        logger.info(f"Generated {len(bars_df)} dollar bars from {len(df)} klines")
        logger.info(
            f"Average bars per day: {len(bars_df) / (df['close_time'].max() - df['open_time'].min()).days:.1f}"
        )

        # Clear state file after successful completion
        if save_state and self.state_file and Path(self.state_file).exists():
            Path(self.state_file).unlink()
            logger.info("Cleared state file after successful completion")

        return bars_df

    def _process_row(self, row: pd.Series) -> None:
        """
        Process a single kline row and update dollar bar state.

        Args:
            row: Kline row with OHLCV data
        """
        # Extract data
        timestamp = row["open_time"]
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]
        volume = row["volume"]

        # Calculate notional value (volume * close price)
        notional = volume * close_price

        # Initialize new bar if needed
        if self.current_bar is None:
            self.current_bar = {
                "timestamp": timestamp,
                "open": close_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "notional": notional,
            }
            self.accumulated_notional = notional
            return

        # Update current bar
        self.current_bar["high"] = max(self.current_bar["high"], high_price)
        self.current_bar["low"] = min(self.current_bar["low"], low_price)
        self.current_bar["close"] = close_price
        self.current_bar["volume"] += volume
        self.current_bar["notional"] += notional
        self.accumulated_notional += notional

        # Check if bar is complete
        if self.accumulated_notional >= self.dollar_bar_threshold:
            self.bars.append(self.current_bar)
            self.current_bar = None
            self.accumulated_notional = 0.0

    def _maybe_save_state(self) -> None:
        """
        Save state if enough time has elapsed since last save.

        State is saved every save_interval seconds.
        """
        now = datetime.now(timezone.utc)

        if self.last_save_time is None:
            self.last_save_time = now
            return

        elapsed = (now - self.last_save_time).total_seconds()

        if elapsed >= self.save_interval:
            self._save_state()
            self.last_save_time = now

    def _save_state(self) -> None:
        """
        Save current state to disk for crash recovery.

        State includes:
        - Current incomplete bar
        - Accumulated notional value
        - Processed row count
        - Completed bars
        """
        if not self.state_file:
            return

        state = {
            "current_bar": self.current_bar,
            "accumulated_notional": self.accumulated_notional,
            "processed_rows": self.processed_rows,
            "bars": self.bars,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Create state file directory if needed
            state_path = Path(self.state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)

            # Save state
            with open(state_path, "wb") as f:
                pickle.dump(state, f)

            logger.debug(f"State saved: {self.processed_rows} rows processed, {len(self.bars)} bars")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> bool:
        """
        Load state from disk for crash recovery.

        Returns:
            True if state was loaded, False otherwise
        """
        if not self.state_file or not Path(self.state_file).exists():
            return False

        try:
            with open(self.state_file, "rb") as f:
                state = pickle.load(f)

            self.current_bar = state["current_bar"]
            self.accumulated_notional = state["accumulated_notional"]
            self.processed_rows = state["processed_rows"]
            self.bars = state["bars"]

            logger.info(
                f"State loaded: {self.processed_rows} rows processed, "
                f"{len(self.bars)} bars completed, "
                f"${self.accumulated_notional:,.0f} accumulated in current bar"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: str,
    ) -> Path:
        """
        Save dollar bars to Parquet file.

        Args:
            df: DataFrame with dollar bars
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to Parquet
        df.to_parquet(output_file, index=False, compression="snappy")

        logger.info(f"Saved {len(df)} dollar bars to {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return output_file


def main():
    """Main entry point for dollar bar generation."""
    parser = argparse.ArgumentParser(
        description="Generate dollar bars from Binance kline data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Parquet file with kline data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Parquet file (default: <input>_dollar_bars.parquet)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10_000_000.0,
        help="Dollar bar threshold in USD (default: 10,000,000)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="State file for crash recovery (default: <input>.state)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="State save interval in seconds (default: 10)",
    )

    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_dollar_bars.parquet")

    # Generate state file path if not specified
    if args.state_file is None:
        args.state_file = f"{args.output}.state"

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"State file: {args.state_file}")

    # Load kline data
    logger.info("Loading kline data...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} klines")

    # Create generator
    generator = DollarBarGenerator(
        dollar_bar_threshold=args.threshold,
        state_file=args.state_file,
        save_interval=args.save_interval,
    )

    # Generate dollar bars
    bars_df = generator.generate_dollar_bars(df)

    # Save to Parquet
    generator.save_to_parquet(bars_df, args.output)

    logger.info("Dollar bar generation complete!")


if __name__ == "__main__":
    main()

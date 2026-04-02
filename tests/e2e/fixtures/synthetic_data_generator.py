"""Synthetic data generator for E2E testing.

Generates realistic MNQ dollar bar data with known signal patterns
for deterministic testing of ensemble components.
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple


class SyntheticDataGenerator:
    """Generate synthetic MNQ dollar bar data for testing.

    Creates realistic price data with controllable patterns:
    - Trending markets
    - Ranging markets
    - Known signal locations for validation
    - Edge cases (gaps, outliers, low volume)
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility.

        Args:
            seed: Random seed for numpy
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_trending_data(
        self,
        n_bars: int = 100,
        trend: str = "up",
        start_price: float = 15000.0,
        start_time: datetime = None,
    ) -> pd.DataFrame:
        """Generate trending market data.

        Args:
            n_bars: Number of bars to generate
            trend: "up" or "down"
            start_price: Starting price level
            start_time: Start timestamp (default: now)

        Returns:
            DataFrame with OHLCV data
        """
        if start_time is None:
            start_time = datetime(2024, 1, 1, 9, 30, 0)

        timestamps = pd.date_range(start=start_time, periods=n_bars, freq="5min")

        # Generate trending price with noise
        trend_factor = 0.0002 if trend == "up" else -0.0002
        returns = np.random.normal(trend_factor, 0.0003, n_bars)
        prices = start_price * (1 + returns).cumprod()

        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices * np.random.uniform(1.0001, 1.0005, n_bars),
            "low": prices * np.random.uniform(0.9995, 0.9999, n_bars),
            "close": prices,
            "volume": np.random.randint(100, 1000, n_bars),
        })

        return bars

    def generate_ranging_data(
        self,
        n_bars: int = 100,
        center_price: float = 15000.0,
        range_width: float = 0.002,  # 0.2% range
        start_time: datetime = None,
    ) -> pd.DataFrame:
        """Generate ranging/consolidation market data.

        Args:
            n_bars: Number of bars to generate
            center_price: Center of price range
            range_width: Width of range (as fraction of price)
            start_time: Start timestamp

        Returns:
            DataFrame with OHLCV data
        """
        if start_time is None:
            start_time = datetime(2024, 1, 1, 9, 30, 0)

        timestamps = pd.date_range(start=start_time, periods=n_bars, freq="5min")

        # Generate oscillating price within range
        t = np.arange(n_bars)
        oscillation = np.sin(2 * np.pi * t / 20)  # 20-bar cycle
        noise = np.random.normal(0, 0.0001, n_bars)

        prices = center_price * (1 + oscillation * range_width / 2 + noise)

        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices * np.random.uniform(1.0001, 1.0003, n_bars),
            "low": prices * np.random.uniform(0.9997, 0.9999, n_bars),
            "close": prices,
            "volume": np.random.randint(100, 1000, n_bars),
        })

        return bars

    def generate_with_embedded_signals(
        self,
        n_bars: int = 100,
        num_signals: int = 5,
        start_time: datetime = None,
    ) -> Tuple[pd.DataFrame, list[dict]]:
        """Generate data with known signal locations.

        Creates realistic price data with embedded signal patterns
        at known locations for validation.

        Args:
            n_bars: Number of bars to generate
            num_signals: Number of signals to embed
            start_time: Start timestamp

        Returns:
            Tuple of (DataFrame, list of signal locations)
            Each signal location dict contains:
            - bar_index: Index of signal bar
            - timestamp: Signal timestamp
            - direction: "long" or "short"
            - expected_patterns: List of expected patterns to fire
        """
        if start_time is None:
            start_time = datetime(2024, 1, 1, 9, 30, 0)

        timestamps = pd.date_range(start=start_time, periods=n_bars, freq="5min")
        base_price = 15000.0

        # Generate base price movement
        returns = np.random.normal(0, 0.0002, n_bars)
        prices = base_price * (1 + returns).cumprod()

        # Embed signals at specific locations
        signal_locations = []
        signal_indices = np.random.choice(
            range(20, n_bars - 20),  # Avoid edges
            size=num_signals,
            replace=False
        )
        signal_indices.sort()

        for idx in signal_indices:
            direction = "long" if np.random.random() > 0.5 else "short"

            # Create signal pattern (quick pullback then continuation)
            if direction == "long":
                # Bearish candle followed by strong bullish move
                prices[idx] *= 0.9995  # Dip
                if idx + 1 < n_bars:
                    prices[idx + 1] *= 1.0010  # Strong recovery
            else:
                # Bullish candle followed by strong bearish move
                prices[idx] *= 1.0005  # Pop
                if idx + 1 < n_bars:
                    prices[idx + 1] *= 0.9990  # Strong decline

            signal_locations.append({
                "bar_index": idx,
                "timestamp": timestamps[idx],
                "direction": direction,
                "expected_patterns": ["triple_confluence", "liquidity_sweep"],
            })

        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices * np.random.uniform(1.0001, 1.0005, n_bars),
            "low": prices * np.random.uniform(0.9995, 0.9999, n_bars),
            "close": prices,
            "volume": np.random.randint(100, 1000, n_bars),
        })

        return bars, signal_locations

    def generate_edge_cases(self, n_bars: int = 50) -> pd.DataFrame:
        """Generate data with edge cases for testing.

        Includes:
        - Price gaps
        - Volume spikes
        - Outliers
        - Low volume periods

        Args:
            n_bars: Number of bars to generate

        Returns:
            DataFrame with edge case OHLCV data
        """
        start_time = datetime(2024, 1, 1, 9, 30, 0)
        timestamps = pd.date_range(start=start_time, periods=n_bars, freq="5min")

        base_price = 15000.0
        prices = np.full(n_bars, base_price)

        # Add gap at bar 10
        if n_bars > 10:
            prices[10:] *= 1.002  # Gap up

        # Add outlier at bar 20
        if n_bars > 20:
            prices[20] *= 1.005  # Large spike

        # Add low volume period at bar 30-35
        # (handled in volume generation)

        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices * np.random.uniform(1.0001, 1.0005, n_bars),
            "low": prices * np.random.uniform(0.9995, 0.9999, n_bars),
            "close": prices,
            "volume": np.random.randint(100, 1000, n_bars),
        })

        # Low volume period
        if n_bars > 35:
            bars.loc[30:35, "volume"] = np.random.randint(10, 50, 6)

        return bars

    def save_to_hdf5(self, bars: pd.DataFrame, filepath: str) -> None:
        """Save bar data to HDF5 file.

        Args:
            bars: DataFrame with OHLCV data
            filepath: Path to save HDF5 file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "timestamps",
                data=bars["timestamp"].astype(np.int64).values
            )
            f.create_dataset("open", data=bars["open"].values)
            f.create_dataset("high", data=bars["high"].values)
            f.create_dataset("low", data=bars["low"].values)
            f.create_dataset("close", data=bars["close"].values)
            f.create_dataset("volume", data=bars["volume"].values)

    def load_sample_500bar(self) -> pd.DataFrame:
        """Generate and return 500-bar sample dataset.

        This is the standard dataset for E2E testing.

        Returns:
            DataFrame with 500 bars of realistic MNQ data
        """
        # Mix of trending and ranging
        trending_up = self.generate_trending_data(100, "up")
        ranging = self.generate_ranging_data(100)
        trending_down = self.generate_trending_data(100, "down")
        ranging_2 = self.generate_ranging_data(100)
        final_trend = self.generate_trending_data(100, "up")

        all_bars = pd.concat([
            trending_up,
            ranging,
            trending_down,
            ranging_2,
            final_trend,
        ], ignore_index=True)

        # Adjust timestamps to be continuous
        start_time = datetime(2024, 1, 1, 9, 30, 0)
        all_bars["timestamp"] = pd.date_range(
            start=start_time,
            periods=len(all_bars),
            freq="5min"
        )

        return all_bars


# Convenience function for quick usage
def generate_test_data(
    data_type: str = "sample_500",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate test data quickly.

    Args:
        data_type: Type of data to generate
            - "trending_up": 100 bars uptrend
            - "trending_down": 100 bars downtrend
            - "ranging": 100 bars ranging
            - "sample_500": 500 bars mixed (default)
            - "edge_cases": 50 bars with edge cases
        seed: Random seed

    Returns:
        DataFrame with generated data
    """
    gen = SyntheticDataGenerator(seed)

    if data_type == "trending_up":
        return gen.generate_trending_data(100, "up")
    elif data_type == "trending_down":
        return gen.generate_trending_data(100, "down")
    elif data_type == "ranging":
        return gen.generate_ranging_data(100)
    elif data_type == "sample_500":
        return gen.load_sample_500bar()
    elif data_type == "edge_cases":
        return gen.generate_edge_cases()
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

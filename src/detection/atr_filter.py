"""ATR (Average True Range) filter for FVG quality screening.

This module implements vectorized ATR calculation and gap size filtering
to eliminate noise FVGs that are unlikely to fill convincingly.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class ATRFilter:
    """Calculate ATR and filter FVGs based on gap size relative to ATR.

    ATR measures market volatility. Small gaps during high volatility are
    often noise, while gaps during low volatility are more significant.

    Filtering Logic:
        - Gap size >= atr_threshold * ATR → PASS (significant gap)
        - Gap size < atr_threshold * ATR → REJECT (noise gap)

    Args:
        lookback_period: Number of bars for ATR calculation (default: 14)
        atr_threshold: Minimum gap size as multiple of ATR (default: 0.5)
        min_history_bars: Minimum bars required for ATR calculation (default: 50)

    Attributes:
        noise_filter_count: Number of FVGs rejected due to small gap size
    """

    def __init__(
        self,
        lookback_period: int = 14,
        atr_threshold: float = 0.5,
        min_history_bars: int = 50,
    ):
        """Initialize ATR filter with configurable parameters."""
        self.lookback_period = lookback_period
        self.atr_threshold = atr_threshold
        self.min_history_bars = min_history_bars
        self.noise_filter_count = 0

    def calculate_atr(self, bars: list[DollarBar]) -> float:
        """Calculate Average True Range using vectorized pandas operations.

        Uses Wilder's smoothing method (exponential moving average) for ATR.
        Falls back to SMA if insufficient bars for EMA.

        Args:
            bars: List of Dollar Bars for ATR calculation

        Returns:
            ATR value in price points (dollars for MNQ)

        Raises:
            ValueError: If fewer than 3 bars available (minimum for TR calculation)
        """
        if len(bars) < 3:
            raise ValueError(
                f"Insufficient bars for ATR calculation: {len(bars)} < 3"
            )

        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame([
            {
                "timestamp": bar.timestamp,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
            }
            for bar in bars
        ])

        # Calculate True Range using vectorized operations
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        # Drop first row (no previous close)
        df = df.dropna(subset=["true_range"])

        # Check if we have enough data for lookback period
        available_bars = len(df)
        if available_bars < self.lookback_period:
            logger.warning(
                f"Insufficient bars for {self.lookback_period}-period ATR: "
                f"{available_bars} available. Using SMA with available data."
            )
            # Fall back to simple average of available bars
            atr = df["true_range"].mean()
        else:
            # Use Wilder's smoothing (EMA with alpha = 1/lookback)
            # This is the standard ATR calculation method
            atr = (
                df["true_range"]
                .tail(self.lookback_period)
                .ewm(alpha=1 / self.lookback_period, adjust=False)
                .mean()
                .iloc[-1]
            )

        # Warn if using insufficient history
        if available_bars < self.min_history_bars:
            logger.warning(
                f"ATR calculated with {available_bars} bars, "
                f"below recommended minimum of {self.min_history_bars}"
            )

        return float(atr)

    def check_gap_significance(
        self,
        gap_size_points: float,
        atr: float,
        direction: Literal["bullish", "bearish"],
    ) -> tuple[bool, float, str]:
        """Check if gap size is significant relative to ATR.

        Args:
            gap_size_points: Gap size in price points
            atr: Current ATR value
            direction: FVG direction (for logging)

        Returns:
            Tuple of (is_significant, atr_multiple, message)
                - is_significant: True if gap >= threshold * ATR
                - atr_multiple: Gap size as multiple of ATR
                - message: Human-readable explanation
        """
        if atr == 0:
            logger.warning("Zero ATR calculated, using absolute gap size check")
            # Default to 1.0 point minimum if ATR is zero
            atr_multiple = gap_size_points / 1.0
        else:
            atr_multiple = gap_size_points / atr

        is_significant = atr_multiple >= self.atr_threshold

        if is_significant:
            message = (
                f"{direction.upper()} FVG gap size {gap_size_points:.2f} points "
                f"is {atr_multiple:.2f}x ATR ({atr:.2f}), passes threshold "
                f"(>={self.atr_threshold:.2f}x ATR)"
            )
            logger.debug(message)
        else:
            message = (
                f"{direction.upper()} FVG gap size {gap_size_points:.2f} points "
                f"is {atr_multiple:.2f}x ATR ({atr:.2f}), below threshold "
                f"(>={self.atr_threshold:.2f}x ATR) - FILTERED AS NOISE"
            )
            logger.debug(message)
            self.noise_filter_count += 1

        return is_significant, atr_multiple, message

    def should_filter_fvg(
        self,
        gap_size_points: float,
        bars: list[DollarBar],
        direction: Literal["bullish", "bearish"],
    ) -> tuple[bool, float, str]:
        """Determine if FVG should be filtered based on ATR analysis.

        This is the main entry point for ATR-based filtering.

        Args:
            gap_size_points: Gap size in price points
            bars: Historical dollar bars for ATR calculation
            direction: FVG direction

        Returns:
            Tuple of (should_filter, atr_multiple, message)
                - should_filter: True if FVG should be rejected
                - atr_multiple: Gap size as multiple of ATR
                - message: Human-readable explanation
        """
        try:
            atr = self.calculate_atr(bars)
        except ValueError as e:
            logger.error(f"ATR calculation failed: {e}")
            # If ATR calculation fails, be conservative and don't filter
            return False, 0.0, f"ATR calculation failed, allowing FVG: {e}"

        is_significant, atr_multiple, message = self.check_gap_significance(
            gap_size_points, atr, direction
        )

        # should_filter = NOT is_significant
        return (not is_significant), atr_multiple, message

    def reset_metrics(self) -> None:
        """Reset filter metrics (e.g., for backtesting or new trading session)."""
        self.noise_filter_count = 0

    def get_metrics(self) -> dict[str, int]:
        """Get current filter metrics.

        Returns:
            Dictionary with filter statistics
        """
        return {
            "noise_filter_count": self.noise_filter_count,
            "lookback_period": self.lookback_period,
            "atr_threshold": self.atr_threshold,
        }

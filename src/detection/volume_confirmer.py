"""Volume confirmation for FVG quality screening.

This module implements volume directional ratio calculation to validate
gap conviction and filter false breakouts lacking institutional participation.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class VolumeConfirmer:
    """Calculate volume ratios and confirm FVG directional conviction.

    Volume Analysis Logic:
        - Bullish FVG: UpVolume / DownVolume >= volume_ratio_threshold
        - Bearish FVG: DownVolume / UpVolume >= volume_ratio_threshold

    Volume Classification:
        - Up Bar: Close > Open (buying pressure)
        - Down Bar: Close < Open (selling pressure)
        - Flat Bar: Close == Open (neutral, excluded from ratios)

    Args:
        lookback_period: Number of bars for volume ratio calculation (default: 20)
        volume_ratio_threshold: Minimum volume ratio to confirm (default: 1.5)
        min_history_bars: Minimum bars required for calculation (default: 30)

    Attributes:
        volume_filter_count: Number of FVGs rejected due to low volume conviction
    """

    def __init__(
        self,
        lookback_period: int = 20,
        volume_ratio_threshold: float = 1.5,
        min_history_bars: int = 30,
    ):
        """Initialize volume confirmer with configurable parameters."""
        self.lookback_period = lookback_period
        self.volume_ratio_threshold = volume_ratio_threshold
        self.min_history_bars = min_history_bars
        self.volume_filter_count = 0

    def calculate_volume_ratios(
        self, bars: list[DollarBar]
    ) -> dict[str, float | int]:
        """Calculate directional volume ratios over lookback period.

        Args:
            bars: List of Dollar Bars for volume analysis

        Returns:
            Dictionary with:
                - up_volume: Total volume on up bars
                - down_volume: Total volume on down bars
                - total_volume: Total volume (up + down)
                - up_bar_count: Number of up bars
                - down_bar_count: Number of down bars
                - flat_bar_count: Number of flat bars (excluded)
                - up_volume_ratio: UpVolume / DownVolume (or inf if DownVolume=0)
                - down_volume_ratio: DownVolume / UpVolume (or inf if UpVolume=0)

        Raises:
            ValueError: If fewer than 2 bars available
        """
        if len(bars) < 2:
            raise ValueError(
                f"Insufficient bars for volume analysis: {len(bars)} < 2"
            )

        # Use last N bars (lookback period)
        recent_bars = bars[-self.lookback_period:] if len(bars) >= self.lookback_period else bars

        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame([
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in recent_bars
        ])

        # Classify bars as up/down/flat
        df["bar_type"] = np.where(
            df["close"] > df["open"], "up",
            np.where(df["close"] < df["open"], "down", "flat")
        )

        # Calculate directional volumes
        up_bars = df[df["bar_type"] == "up"]
        down_bars = df[df["bar_type"] == "down"]
        flat_bars = df[df["bar_type"] == "flat"]

        up_volume = up_bars["volume"].sum()
        down_volume = down_bars["volume"].sum()
        total_volume = up_volume + down_volume

        # Calculate ratios (handle division by zero)
        up_volume_ratio = (
            float("inf") if down_volume == 0
            else up_volume / down_volume
        )
        down_volume_ratio = (
            float("inf") if up_volume == 0
            else down_volume / up_volume
        )

        # Warn if using insufficient history
        if len(recent_bars) < self.min_history_bars:
            logger.warning(
                f"Volume ratios calculated with {len(recent_bars)} bars, "
                f"below recommended minimum of {self.min_history_bars}"
            )

        return {
            "up_volume": int(up_volume),
            "down_volume": int(down_volume),
            "total_volume": int(total_volume),
            "up_bar_count": len(up_bars),
            "down_bar_count": len(down_bars),
            "flat_bar_count": len(flat_bars),
            "up_volume_ratio": up_volume_ratio,
            "down_volume_ratio": down_volume_ratio,
        }

    def check_volume_confirmation(
        self,
        direction: Literal["bullish", "bearish"],
        volume_metrics: dict[str, float | int],
    ) -> tuple[bool, float, str]:
        """Check if volume confirms FVG direction.

        Args:
            direction: FVG direction (bullish/bearish)
            volume_metrics: Volume metrics from calculate_volume_ratios()

        Returns:
            Tuple of (is_confirmed, volume_ratio, message)
                - is_confirmed: True if volume confirms direction
                - volume_ratio: Actual volume ratio (for logging)
                - message: Human-readable explanation
        """
        if direction == "bullish":
            volume_ratio = volume_metrics["up_volume_ratio"]
            is_confirmed = volume_ratio >= self.volume_ratio_threshold
            ratio_desc = f"UpVolume/DownVolume = {volume_ratio:.2f}"
        else:  # bearish
            volume_ratio = volume_metrics["down_volume_ratio"]
            is_confirmed = volume_ratio >= self.volume_ratio_threshold
            ratio_desc = f"DownVolume/UpVolume = {volume_ratio:.2f}"

        if is_confirmed:
            message = (
                f"{direction.upper()} FVG volume confirmation: {ratio_desc}, "
                f"passes threshold (>={self.volume_ratio_threshold:.2f})"
            )
            logger.debug(message)
        else:
            message = (
                f"{direction.upper()} FVG volume confirmation: {ratio_desc}, "
                f"below threshold (>={self.volume_ratio_threshold:.2f}) - FILTERED"
            )
            logger.debug(message)
            self.volume_filter_count += 1

        return is_confirmed, volume_ratio, message

    def should_filter_fvg(
        self,
        direction: Literal["bullish", "bearish"],
        bars: list[DollarBar],
    ) -> tuple[bool, float, str]:
        """Determine if FVG should be filtered based on volume analysis.

        This is the main entry point for volume-based filtering.

        Args:
            direction: FVG direction (bullish/bearish)
            bars: Historical dollar bars for volume analysis

        Returns:
            Tuple of (should_filter, volume_ratio, message)
                - should_filter: True if FVG should be rejected
                - volume_ratio: Actual volume ratio
                - message: Human-readable explanation
        """
        try:
            volume_metrics = self.calculate_volume_ratios(bars)
        except ValueError as e:
            logger.error(f"Volume calculation failed: {e}")
            # If volume calculation fails, be conservative and don't filter
            return False, 0.0, f"Volume calculation failed, allowing FVG: {e}"

        is_confirmed, volume_ratio, message = self.check_volume_confirmation(
            direction, volume_metrics
        )

        # should_filter = NOT is_confirmed
        return (not is_confirmed), volume_ratio, message

    def reset_metrics(self) -> None:
        """Reset filter metrics (e.g., for backtesting or new trading session)."""
        self.volume_filter_count = 0

    def get_metrics(self) -> dict[str, int]:
        """Get current filter metrics.

        Returns:
            Dictionary with filter statistics
        """
        return {
            "volume_filter_count": self.volume_filter_count,
            "lookback_period": self.lookback_period,
            "volume_ratio_threshold": self.volume_ratio_threshold,
        }

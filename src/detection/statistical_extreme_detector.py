"""Statistical Extreme Detector for Wolf Pack 3-Edge strategy.

This module implements statistical extreme detection, which identifies
when price deviates significantly from its mean (statistical edge).
"""

import logging
from collections import deque

import numpy as np

from src.data.models import DollarBar
from src.detection.models import StatisticalExtremeEvent

logger = logging.getLogger(__name__)


class StatisticalExtremeDetector:
    """Detects statistical extremes for Wolf Pack 3-Edge strategy.

    Computes rolling mean and standard deviation, then identifies when
    price exceeds 2 standard deviations from the mean.

    Attributes:
        _window: Rolling window for statistics (default: 20 bars)
        _sd_threshold: Standard deviation threshold (default: 2.0)
        _bars: History of recent Dollar Bars
    """

    DEFAULT_WINDOW = 20  # Bars for rolling statistics
    DEFAULT_SD_THRESHOLD = 2.0  # Standard deviations for extreme
    MAX_BAR_HISTORY = 100

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        sd_threshold: float = DEFAULT_SD_THRESHOLD,
    ) -> None:
        """Initialize Statistical Extreme detector.

        Args:
            window: Rolling window for statistics
            sd_threshold: Standard deviation threshold for extreme
        """
        self._window = window
        self._sd_threshold = sd_threshold

        self._bars: deque[DollarBar] = deque(maxlen=self.MAX_BAR_HISTORY)

    def process_bars(self, bars: list[DollarBar]) -> list[StatisticalExtremeEvent]:
        """Process new bars and detect statistical extremes.

        Args:
            bars: List of new Dollar Bars to process

        Returns:
            List of detected extreme events
        """
        extremes = []

        for bar in bars:
            if len(self._bars) == 0 or bar.timestamp > self._bars[-1].timestamp:
                self._bars.append(bar)

                # Check for extreme after adding new bar
                extreme = self._check_for_extreme()
                if extreme:
                    extremes.append(extreme)

        return extremes

    def _check_for_extreme(self) -> StatisticalExtremeEvent | None:
        """Check if current price represents a statistical extreme.

        Returns:
            StatisticalExtremeEvent if extreme detected, None otherwise
        """
        if len(self._bars) < self._window:
            return None

        bars_list = list(self._bars)
        recent_bars = bars_list[-self._window:]

        # Calculate rolling statistics
        prices = [bar.close for bar in recent_bars]
        rolling_mean = np.mean(prices)
        rolling_std = np.std(prices)

        # Get current price
        current_bar = bars_list[-1]
        current_price = current_bar.close

        # Calculate Z-score
        if rolling_std == 0:
            return None

        z_score = (current_price - rolling_mean) / rolling_std

        # Check if exceeds threshold (absolute value)
        if abs(z_score) < self._sd_threshold:
            return None

        # Determine direction
        direction = "high" if z_score > 0 else "low"
        magnitude = abs(z_score)

        # Create extreme event
        event = StatisticalExtremeEvent(
            timestamp=current_bar.timestamp,
            z_score=z_score,
            direction=direction,
            magnitude=magnitude,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            current_price=current_price,
        )

        logger.info(
            f"Statistical extreme detected: {direction}, z-score={z_score:.2f}, "
            f"price={current_price:.2f}, mean={rolling_mean:.2f}, std={rolling_std:.2f}"
        )

        return event

    def reset(self) -> None:
        """Reset detector state."""
        self._bars.clear()

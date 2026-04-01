"""RSI Calculator for Adaptive EMA Momentum strategy.

This module calculates the Relative Strength Index (RSI) with mid-band
emphasis and direction tracking for momentum analysis.
"""

import logging
from collections import deque
from datetime import datetime

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class RSICalculator:
    """Calculates Relative Strength Index (RSI) from Dollar Bars.

    RSI = 100 - (100 / (1 + RS))
    Where RS = Average Gain / Average Loss
    Uses Wilder's smoothing method.

    Attributes:
        _period: RSI period (default 14)
        _mid_band_lower: Lower bound of mid-band (default 40)
        _mid_band_upper: Upper bound of mid-band (default 60)
        _rsi: Current RSI value
        _rsi_history: Deque of RSI values
        _prev_close: Previous close price for change calculation
        _gains: Deque of gains for averaging
        _losses: Deque of losses for averaging
        _avg_gain: Current average gain
        _avg_loss: Current average loss
    """

    DEFAULT_PERIOD = 14
    DEFAULT_MID_BAND_LOWER = 40
    DEFAULT_MID_BAND_UPPER = 60
    MAX_HISTORY = 100  # Keep last 100 RSI values

    def __init__(
        self,
        period: int = DEFAULT_PERIOD,
        mid_band_lower: int = DEFAULT_MID_BAND_LOWER,
        mid_band_upper: int = DEFAULT_MID_BAND_UPPER,
    ) -> None:
        """Initialize RSI calculator.

        Args:
            period: RSI period (default 14)
            mid_band_lower: Lower bound of mid-band (default 40)
            mid_band_upper: Upper bound of mid-band (default 60)
        """
        self._period = period
        self._mid_band_lower = mid_band_lower
        self._mid_band_upper = mid_band_upper

        # Initialize values
        self._rsi: float | None = None
        self._prev_close: float | None = None
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None

        # Store history
        self._rsi_history: deque[float] = deque(maxlen=self.MAX_HISTORY)

        # Store gains and losses for initial average calculation
        self._gains: deque[float] = deque(maxlen=self._period)
        self._losses: deque[float] = deque(maxlen=self._period)

    def calculate_rsi(self, bars: list[DollarBar]) -> float | None:
        """Calculate RSI from a list of Dollar Bars.

        Args:
            bars: List of Dollar Bars to calculate RSI from

        Returns:
            Current RSI value, or None if insufficient data
        """
        if not bars:
            return None

        for bar in bars:
            self._update_rsi(bar.close)

        return self._rsi

    def _update_rsi(self, close_price: float) -> None:
        """Update RSI with new close price.

        Args:
            close_price: New close price to update RSI with
        """
        if self._prev_close is None:
            # First price, store and wait for next
            self._prev_close = close_price
            return

        # Calculate price change
        change = close_price - self._prev_close
        self._prev_close = close_price

        # Separate into gains and losses
        if change > 0:
            gain = change
            loss = 0.0
        else:
            gain = 0.0
            loss = abs(change)

        self._gains.append(gain)
        self._losses.append(loss)

        # Calculate RSI once we have enough data
        if len(self._gains) == self._period:
            if self._avg_gain is None or self._avg_loss is None:
                # First calculation - use simple average
                self._avg_gain = sum(self._gains) / self._period
                self._avg_loss = sum(self._losses) / self._period
            else:
                # Use Wilder's smoothing
                self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
                self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

            # Calculate RSI
            if self._avg_loss == 0:
                self._rsi = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                self._rsi = 100.0 - (100.0 / (1.0 + rs))

            self._rsi_history.append(self._rsi)

    def get_current_rsi(self) -> float | None:
        """Get current RSI value.

        Returns:
            Current RSI value, or None if not available
        """
        return self._rsi

    def get_rsi_history(self) -> list[float]:
        """Get RSI history for analysis.

        Returns:
            List of RSI values
        """
        return list(self._rsi_history)

    def is_in_mid_band(self) -> bool:
        """Check if RSI is in mid-band range.

        Returns:
            True if RSI is between mid_band_lower and mid_band_upper
        """
        if self._rsi is None:
            return False

        return self._mid_band_lower <= self._rsi <= self._mid_band_upper

    def is_rising(self) -> bool:
        """Check if RSI is rising.

        Returns:
            True if current RSI > previous RSI
        """
        if len(self._rsi_history) < 2:
            return False

        current = self._rsi_history[-1]
        previous = self._rsi_history[-2]

        return current > previous

    def is_falling(self) -> bool:
        """Check if RSI is falling.

        Returns:
            True if current RSI < previous RSI
        """
        if len(self._rsi_history) < 2:
            return False

        current = self._rsi_history[-1]
        previous = self._rsi_history[-2]

        return current < previous

    def is_mid_band_and_rising(self) -> bool:
        """Check if RSI is in mid-band and rising.

        Returns:
            True if RSI is in mid-band and rising
        """
        return self.is_in_mid_band() and self.is_rising()

    def is_mid_band_and_falling(self) -> bool:
        """Check if RSI is in mid-band and falling.

        Returns:
            True if RSI is in mid-band and falling
        """
        return self.is_in_mid_band() and self.is_falling()

    def reset(self) -> None:
        """Reset RSI calculator and clear all history."""
        self._rsi = None
        self._prev_close = None
        self._avg_gain = None
        self._avg_loss = None
        self._rsi_history.clear()
        self._gains.clear()
        self._losses.clear()
        logger.debug("RSI calculator reset")

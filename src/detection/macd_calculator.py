"""MACD Calculator for Adaptive EMA Momentum strategy.

This module calculates the Moving Average Convergence Divergence (MACD) indicator
including MACD line, signal line, and histogram for momentum analysis.
"""

import logging
from collections import deque
from datetime import datetime

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class MACDCalculator:
    """Calculates MACD (Moving Average Convergence Divergence) from Dollar Bars.

    MACD Line = 12-period EMA - 26-period EMA
    Signal Line = 9-period EMA of MACD Line
    Histogram = MACD Line - Signal Line

    Attributes:
        _fast_period: Fast EMA period for MACD (default 12)
        _slow_period: Slow EMA period for MACD (default 26)
        _signal_period: Signal line EMA period (default 9)
        _macd_line: Current MACD line value
        _signal_line: Current signal line value
        _histogram: Current histogram value
        _macd_history: Deque of MACD line values
        _signal_history: Deque of signal line values
        _histogram_history: Deque of histogram values
        _close_prices: Deque of close prices for calculation
    """

    DEFAULT_FAST_PERIOD = 12
    DEFAULT_SLOW_PERIOD = 26
    DEFAULT_SIGNAL_PERIOD = 9
    MAX_HISTORY = 100  # Keep last 100 MACD values

    def __init__(
        self,
        fast_period: int = DEFAULT_FAST_PERIOD,
        slow_period: int = DEFAULT_SLOW_PERIOD,
        signal_period: int = DEFAULT_SIGNAL_PERIOD,
    ) -> None:
        """Initialize MACD calculator.

        Args:
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
        """
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period

        # Initialize values
        self._macd_line: float | None = None
        self._signal_line: float | None = None
        self._histogram: float | None = None

        # EMA tracking for MACD components
        self._fast_ema: float | None = None
        self._slow_ema: float | None = None

        # Store history
        self._macd_history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._signal_history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._histogram_history: deque[float] = deque(maxlen=self.MAX_HISTORY)

        # Store close prices needed for calculation
        self._close_prices: deque[float] = deque(maxlen=self._slow_period + self._signal_period + 10)

        # Store MACD values for signal line calculation
        self._macd_values: deque[float] = deque(maxlen=self._signal_period + 10)

    def calculate_macd(self, bars: list[DollarBar]) -> dict[str, float | None]:
        """Calculate MACD from a list of Dollar Bars.

        Args:
            bars: List of Dollar Bars to calculate MACD from

        Returns:
            Dictionary with 'macd_line', 'signal_line', 'histogram' values
        """
        if not bars:
            return {'macd_line': None, 'signal_line': None, 'histogram': None}

        # Process each bar
        for bar in bars:
            self._update_macd(bar.close)

        return {
            'macd_line': self._macd_line,
            'signal_line': self._signal_line,
            'histogram': self._histogram
        }

    def _update_macd(self, close_price: float) -> None:
        """Update MACD values with new close price.

        Args:
            close_price: New close price to update MACD with
        """
        # Add to price history
        self._close_prices.append(close_price)

        # Calculate fast EMA
        if len(self._close_prices) >= self._fast_period:
            multiplier = 2.0 / (self._fast_period + 1)
            if self._fast_ema is None:
                self._fast_ema = sum(list(self._close_prices)[-self._fast_period:]) / self._fast_period
            else:
                self._fast_ema = close_price * multiplier + self._fast_ema * (1 - multiplier)

        # Calculate slow EMA
        if len(self._close_prices) >= self._slow_period:
            multiplier = 2.0 / (self._slow_period + 1)
            if self._slow_ema is None:
                self._slow_ema = sum(list(self._close_prices)[-self._slow_period:]) / self._slow_period
            else:
                self._slow_ema = close_price * multiplier + self._slow_ema * (1 - multiplier)

        # Calculate MACD line
        if self._fast_ema is not None and self._slow_ema is not None:
            self._macd_line = self._fast_ema - self._slow_ema
            self._macd_values.append(self._macd_line)
            self._macd_history.append(self._macd_line)

            # Calculate signal line (EMA of MACD)
            if len(self._macd_values) >= self._signal_period:
                multiplier = 2.0 / (self._signal_period + 1)
                if self._signal_line is None:
                    self._signal_line = sum(list(self._macd_values)[-self._signal_period:]) / self._signal_period
                else:
                    self._signal_line = self._macd_line * multiplier + self._signal_line * (1 - multiplier)
                self._signal_history.append(self._signal_line)

                # Calculate histogram
                self._histogram = self._macd_line - self._signal_line
                self._histogram_history.append(self._histogram)

    def get_current_macd(self) -> dict[str, float | None]:
        """Get current MACD values.

        Returns:
            Dictionary with 'macd_line', 'signal_line', 'histogram' values
        """
        return {
            'macd_line': self._macd_line,
            'signal_line': self._signal_line,
            'histogram': self._histogram
        }

    def get_macd_history(self) -> dict[str, list[float]]:
        """Get MACD history for momentum analysis.

        Returns:
            Dictionary with 'macd_line', 'signal_line', 'histogram' lists
        """
        return {
            'macd_line': list(self._macd_history),
            'signal_line': list(self._signal_history),
            'histogram': list(self._histogram_history)
        }

    def is_momentum_increasing(self) -> bool:
        """Check if momentum is increasing (histogram rising).

        Returns:
            True if histogram is positive and increasing
        """
        if len(self._histogram_history) < 2:
            return False

        # Check if histogram is positive and current > previous
        current = self._histogram_history[-1]
        previous = self._histogram_history[-2]

        return current > 0 and current > previous

    def is_momentum_decreasing(self) -> bool:
        """Check if momentum is decreasing (histogram falling).

        Returns:
            True if histogram is negative and decreasing
        """
        if len(self._histogram_history) < 2:
            return False

        # Check if histogram is negative and current < previous
        current = self._histogram_history[-1]
        previous = self._histogram_history[-2]

        return current < 0 and current < previous

    def reset(self) -> None:
        """Reset MACD calculator and clear all history."""
        self._macd_line = None
        self._signal_line = None
        self._histogram = None
        self._fast_ema = None
        self._slow_ema = None
        self._macd_history.clear()
        self._signal_history.clear()
        self._histogram_history.clear()
        self._close_prices.clear()
        self._macd_values.clear()
        logger.debug("MACD calculator reset")

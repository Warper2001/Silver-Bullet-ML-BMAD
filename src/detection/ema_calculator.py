"""EMA Calculator for Adaptive EMA Momentum strategy.

This module calculates exponential moving averages (EMAs) for 9, 55, and 100 periods
to determine trend direction for momentum trading.

Note: Using EMA100 instead of EMA200 for faster signal generation and reduced warmup period.
"""

import logging
from collections import deque
from datetime import datetime

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class EMACalculator:
    """Calculates Exponential Moving Averages (EMAs) from Dollar Bars.

    EMA = Price_t × multiplier + EMA_t-1 × (1 - multiplier)
    Where multiplier = 2 / (period + 1)

    Uses 9, 55, and 100 period EMAs for trend analysis.

    Attributes:
        _fast_period: Fast EMA period (default 9)
        _medium_period: Medium EMA period (default 55)
        _slow_period: Slow EMA period (default 100)
        _fast_ema_history: Deque of fast EMA values
        _medium_ema_history: Deque of medium EMA values
        _slow_ema_history: Deque of slow EMA values
        _close_prices: Deque of close prices for EMA calculation
    """

    DEFAULT_FAST_PERIOD = 9
    DEFAULT_MEDIUM_PERIOD = 55
    DEFAULT_SLOW_PERIOD = 100  # Changed from 200 to 100 for faster signal generation
    MAX_HISTORY = 150  # Keep last 150 EMA values

    def __init__(
        self,
        fast_period: int = DEFAULT_FAST_PERIOD,
        medium_period: int = DEFAULT_MEDIUM_PERIOD,
        slow_period: int = DEFAULT_SLOW_PERIOD,
    ) -> None:
        """Initialize EMA calculator.

        Args:
            fast_period: Fast EMA period (default 9)
            medium_period: Medium EMA period (default 55)
            slow_period: Slow EMA period (default 100)
        """
        self._fast_period = fast_period
        self._medium_period = medium_period
        self._slow_period = slow_period

        # Initialize EMAs as None (not enough data initially)
        self._fast_ema: float | None = None
        self._medium_ema: float | None = None
        self._slow_ema: float | None = None

        # Store history for trend analysis
        self._fast_ema_history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._medium_ema_history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._slow_ema_history: deque[float] = deque(maxlen=self.MAX_HISTORY)

        # Store close prices needed for EMA calculation
        self._close_prices: deque[float] = deque(maxlen=self._slow_period + 10)

    def calculate_emas(self, bars: list[DollarBar]) -> dict[str, float | None]:
        """Calculate EMAs from a list of Dollar Bars.

        Args:
            bars: List of Dollar Bars to calculate EMAs from

        Returns:
            Dictionary with 'fast_ema', 'medium_ema', 'slow_ema' values
        """
        if not bars:
            return {'fast_ema': None, 'medium_ema': None, 'slow_ema': None}

        # Process each bar
        for bar in bars:
            self._update_ema(bar.close)

        return {
            'fast_ema': self._fast_ema,
            'medium_ema': self._medium_ema,
            'slow_ema': self._slow_ema
        }

    def _update_ema(self, close_price: float) -> None:
        """Update EMA values with new close price.

        Args:
            close_price: New close price to update EMAs with
        """
        # Add to price history
        self._close_prices.append(close_price)

        # Calculate fast EMA
        if len(self._close_prices) >= self._fast_period:
            multiplier = 2.0 / (self._fast_period + 1)
            if self._fast_ema is None:
                # Initialize with SMA of first period values
                self._fast_ema = sum(list(self._close_prices)[-self._fast_period:]) / self._fast_period
            else:
                self._fast_ema = close_price * multiplier + self._fast_ema * (1 - multiplier)
            self._fast_ema_history.append(self._fast_ema)

        # Calculate medium EMA
        if len(self._close_prices) >= self._medium_period:
            multiplier = 2.0 / (self._medium_period + 1)
            if self._medium_ema is None:
                # Initialize with SMA of first period values
                self._medium_ema = sum(list(self._close_prices)[-self._medium_period:]) / self._medium_period
            else:
                self._medium_ema = close_price * multiplier + self._medium_ema * (1 - multiplier)
            self._medium_ema_history.append(self._medium_ema)

        # Calculate slow EMA
        if len(self._close_prices) >= self._slow_period:
            multiplier = 2.0 / (self._slow_period + 1)
            if self._slow_ema is None:
                # Initialize with SMA of first period values
                self._slow_ema = sum(list(self._close_prices)[-self._slow_period:]) / self._slow_period
            else:
                self._slow_ema = close_price * multiplier + self._slow_ema * (1 - multiplier)
            self._slow_ema_history.append(self._slow_ema)

    def get_trend_direction(self) -> str:
        """Determine trend direction based on EMA alignment.

        Returns:
            "bullish" if fast > medium > slow
            "bearish" if fast < medium < slow
            "neutral" otherwise (EMAs crossed or close together)
        """
        if None in (self._fast_ema, self._medium_ema, self._slow_ema):
            return "neutral"

        # Check if EMAs are close together (within 0.05%)
        avg_ema = (self._fast_ema + self._medium_ema + self._slow_ema) / 3
        threshold = avg_ema * 0.0005  # 0.05% threshold

        if (abs(self._fast_ema - self._medium_ema) < threshold and
            abs(self._medium_ema - self._slow_ema) < threshold):
            return "neutral"

        # Check bullish alignment
        if self._fast_ema > self._medium_ema > self._slow_ema:
            return "bullish"

        # Check bearish alignment
        if self._fast_ema < self._medium_ema < self._slow_ema:
            return "bearish"

        return "neutral"

    def get_ema_history(self) -> dict[str, list[float]]:
        """Get EMA history for trend analysis.

        Returns:
            Dictionary with 'fast_ema', 'medium_ema', 'slow_ema' lists
        """
        return {
            'fast_ema': list(self._fast_ema_history),
            'medium_ema': list(self._medium_ema_history),
            'slow_ema': list(self._slow_ema_history)
        }

    def get_current_emas(self) -> dict[str, float | None]:
        """Get current EMA values without processing new bars.

        Returns:
            Dictionary with 'fast_ema', 'medium_ema', 'slow_ema' values
        """
        return {
            'fast_ema': self._fast_ema,
            'medium_ema': self._medium_ema,
            'slow_ema': self._slow_ema
        }

    def reset(self) -> None:
        """Reset EMA calculator and clear all history."""
        self._fast_ema = None
        self._medium_ema = None
        self._slow_ema = None
        self._fast_ema_history.clear()
        self._medium_ema_history.clear()
        self._slow_ema_history.clear()
        self._close_prices.clear()
        logger.debug("EMA calculator reset")

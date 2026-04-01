"""ADX Calculator for VWAP Bounce strategy.

This module calculates the Average Directional Index (ADX) and
Directional Indicators (DI+/DI-) to determine trend strength and direction.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from math import sqrt

import numpy as np

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


@dataclass
class ADXResult:
    """Result of ADX calculation.

    Attributes:
        adx: Average Directional Index value
        di_plus: Positive Directional Indicator (+DI)
        di_minus: Negative Directional Indicator (-DI)
        trend_strength: "trending" if ADX > threshold, "ranging" otherwise
        trend_direction: "up" if DI+ > DI-, "down" if DI- > DI+, "neutral" if equal
        timestamp: When the ADX was calculated
    """

    adx: float
    di_plus: float
    di_minus: float
    trend_strength: str  # "trending" or "ranging"
    trend_direction: str  # "up", "down", or "neutral"
    timestamp: datetime


class ADXCalculator:
    """Calculates Average Directional Index (ADX) for trend strength.

    ADX measures trend strength regardless of direction:
    - ADX > 20: Trending market
    - ADX <= 20: Ranging market

    DI+ and DI- show trend direction:
    - DI+ > DI-: Uptrend
    - DI- > DI+: Downtrend

    Attributes:
        _period: ADX calculation period (default 14)
        _adx_threshold: Threshold for trending vs ranging (default 20)
    """

    DEFAULT_PERIOD = 14
    DEFAULT_ADX_THRESHOLD = 20

    def __init__(
        self,
        period: int = DEFAULT_PERIOD,
        adx_threshold: float = DEFAULT_ADX_THRESHOLD,
    ) -> None:
        """Initialize ADX Calculator.

        Args:
            period: ADX calculation period (default 14)
            adx_threshold: Threshold for trending vs ranging (default 20)
        """
        self._period = period
        self._adx_threshold = adx_threshold

    def calculate_adx(self, bars: list[DollarBar]) -> ADXResult | None:
        """Calculate ADX from dollar bars.

        Args:
            bars: List of dollar bars (minimum: period + 1 bars)

        Returns:
            ADXResult with ADX, DI+, DI-, and trend analysis, or None if insufficient bars
        """
        if len(bars) < self._period + 1:
            logger.debug(f"Insufficient bars for ADX: {len(bars)} < {self._period + 1}")
            return None

        # Extract price data
        highs = np.array([bar.high for bar in bars])
        lows = np.array([bar.low for bar in bars])
        closes = np.array([bar.close for bar in bars])

        # Calculate True Range
        tr = self._calculate_true_range(highs, lows, closes)

        # Calculate +DM and -DM
        plus_dm, minus_dm = self._calculate_directional_movement(highs, lows)

        # Smooth TR, +DM, -DM using Wilder's smoothing
        atr = self._wilder_smoothing(tr, self._period)
        smooth_plus_dm = self._wilder_smoothing(plus_dm, self._period)
        smooth_minus_dm = self._wilder_smoothing(minus_dm, self._period)

        # Calculate +DI and -DI
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            di_plus = 100 * smooth_plus_dm / atr
            di_minus = 100 * smooth_minus_dm / atr
            # Replace inf/nan with 0
            di_plus = np.nan_to_num(di_plus, nan=0.0, posinf=0.0, neginf=0.0)
            di_minus = np.nan_to_num(di_minus, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate DX and ADX
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)

        adx = self._wilder_smoothing(dx, self._period)

        # Determine trend strength and direction
        trend_strength = "trending" if adx[-1] > self._adx_threshold else "ranging"

        if di_plus[-1] > di_minus[-1]:
            trend_direction = "up"
        elif di_minus[-1] > di_plus[-1]:
            trend_direction = "down"
        else:
            trend_direction = "neutral"

        result = ADXResult(
            adx=adx[-1],
            di_plus=di_plus[-1],
            di_minus=di_minus[-1],
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            timestamp=bars[-1].timestamp,
        )

        logger.debug(
            f"ADX: {adx[-1]:.2f}, DI+: {di_plus[-1]:.2f}, DI-: {di_minus[-1]:.2f}, "
            f"Trend: {trend_strength} {trend_direction}"
        )

        return result

    def _calculate_true_range(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> np.ndarray:
        """Calculate True Range.

        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))

        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices

        Returns:
            Array of True Range values
        """
        # High - Low
        hl = highs[1:] - lows[1:]

        # Abs(High - Previous Close)
        h_pc = np.abs(highs[1:] - closes[:-1])

        # Abs(Low - Previous Close)
        l_pc = np.abs(lows[1:] - closes[:-1])

        # TR is the maximum of the three
        tr = np.maximum(np.maximum(hl, h_pc), l_pc)

        # Add 0 at the beginning to match array length
        tr = np.concatenate([[0], tr])

        return tr

    def _calculate_directional_movement(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate +DM and -DM.

        +DM: Current high - previous high (if upward movement)
        -DM: Previous low - current low (if downward movement)

        Args:
            highs: Array of high prices
            lows: Array of low prices

        Returns:
            Tuple of (+DM array, -DM array)
        """
        # Calculate up and down moves
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        # +DM: up move if up > down and up > 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)

        # -DM: down move if down > up and down > 0
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Add 0 at the beginning to match array length
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])

        return plus_dm, minus_dm

    def _wilder_smoothing(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing (exponential moving average).

        Uses alpha = 1 / period for Wilder's smoothing method.

        Args:
            data: Array of values to smooth
            period: Smoothing period

        Returns:
            Smoothed array
        """
        alpha = 1.0 / period
        smoothed = np.zeros_like(data)

        # Initialize with first value
        smoothed[0] = data[0]

        # Apply smoothing
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

        return smoothed

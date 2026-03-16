"""Swing point and Market Structure Shift (MSS) detection algorithms.

This module implements the core algorithms for identifying swing points
(pivot highs and lows) and detecting Market Structure Shifts when price
breaks through previous swing levels with volume confirmation.
"""

import collections
import logging

from src.data.models import DollarBar, MSSEvent, SwingPoint

logger = logging.getLogger(__name__)


class RollingVolumeAverage:
    """Efficient rolling average of volume with O(1) update time.

    Attributes:
        window: Number of bars to include in moving average
        buffer: Circular buffer storing recent volume values
        sum: Running sum of volume values in buffer
    """

    def __init__(self, window: int = 20) -> None:
        """Initialize rolling volume average.

        Args:
            window: Number of bars for moving average (default: 20)
        """
        self.window = window
        self.buffer: collections.deque[float] = collections.deque(maxlen=window)
        self.sum = 0.0

    def update(self, volume: float) -> float:
        """Update with new volume and return current average.

        Args:
            volume: New volume value to add to average

        Returns:
            Current rolling average after update
        """
        # Remove oldest value if buffer is full
        if len(self.buffer) == self.window:
            self.sum -= self.buffer[0]

        # Add new value
        self.buffer.append(volume)
        self.sum += volume

        return self.sum / len(self.buffer)

    @property
    def average(self) -> float:
        """Get current average.

        Returns:
            Current rolling average (0.0 if no data)
        """
        return self.sum / len(self.buffer) if self.buffer else 0.0


def detect_swing_high(bars: list[DollarBar], index: int, lookback: int = 3) -> bool:
    """Detect pivot high: highest high with at least N bars lower on each side.

    A swing high represents a potential resistance level and indicates
    that bulls were unable to push price higher at this point.

    Args:
        bars: List of Dollar Bars (OHLCV data)
        index: Current bar index to check
        lookback: Minimum bars on each side (default: 3)

    Returns:
        True if bar[index] is a swing high

    Examples:
        >>> bars = [DollarBar(...), ...]  # 7+ bars
        >>> detect_swing_high(bars, 3, lookback=3)  # Bar 3 is highest
        True
    """
    # Check if we have enough bars on both sides
    if index < lookback or index >= len(bars) - lookback:
        return False

    current_high = bars[index].high

    # Check if current high is the maximum in the range
    for i in range(index - lookback, index + lookback + 1):
        if i != index and bars[i].high >= current_high:
            return False

    return True


def detect_swing_low(bars: list[DollarBar], index: int, lookback: int = 3) -> bool:
    """Detect pivot low: lowest low with at least N bars higher on each side.

    A swing low represents a potential support level and indicates
    that bears were unable to push price lower at this point.

    Args:
        bars: List of Dollar Bars (OHLCV data)
        index: Current bar index to check
        lookback: Minimum bars on each side (default: 3)

    Returns:
        True if bar[index] is a swing low

    Examples:
        >>> bars = [DollarBar(...), ...]  # 7+ bars
        >>> detect_swing_low(bars, 3, lookback=3)
        True  # Bar 3 is lowest in range [0, 6]
    """
    # Check if we have enough bars on both sides
    if index < lookback or index >= len(bars) - lookback:
        return False

    current_low = bars[index].low

    # Check if current low is the minimum in the range
    for i in range(index - lookback, index + lookback + 1):
        if i != index and bars[i].low <= current_low:
            return False

    return True


def detect_bullish_mss(
    current_bar: DollarBar,
    swing_highs: list[SwingPoint],
    volume_ma_20: float,
    volume_confirmation_ratio: float = 1.5,
) -> MSSEvent | None:
    """Detect bullish MSS: price breaks above swing high with volume confirmation.

    A bullish Market Structure Shift indicates a potential trend reversal
    to the upside or continuation of an uptrend.

    Args:
        current_bar: Most recent Dollar Bar
        swing_highs: List of confirmed swing highs
        volume_ma_20: 20-bar moving average of volume
        volume_confirmation_ratio: Minimum volume ratio for confirmation (default: 1.5x)

    Returns:
        MSSEvent if bullish MSS detected, None otherwise

    Examples:
        >>> swing_highs = [SwingPoint(price=11800.0, ...)]
        >>> bar = DollarBar(high=11900.0, volume=1500, ...)
        >>> detect_bullish_mss(bar, swing_highs, volume_ma_20=1000.0)
        MSSEvent(direction='bullish', breakout_price=11900.0, volume_ratio=1.5, ...)
    """
    if not swing_highs:
        return None

    most_recent_swing_high = swing_highs[-1]
    breakout_price = current_bar.high

    # Check for breakout (price must exceed swing high)
    if breakout_price <= most_recent_swing_high.price:
        return None

    # Calculate volume ratio
    volume_ratio = current_bar.volume / volume_ma_20 if volume_ma_20 > 0 else 0.0

    # Volume confirmation (breakout must have above-average volume)
    if volume_ratio < volume_confirmation_ratio:
        return None

    # Bullish MSS detected!
    return MSSEvent(
        timestamp=current_bar.timestamp,
        direction="bullish",
        breakout_price=breakout_price,
        swing_point=most_recent_swing_high,
        volume_ratio=volume_ratio,
        bar_index=0,  # Will be set by MSSDetector
        confidence=0.0,  # Will be calculated in Story 2.6
    )


def detect_bearish_mss(
    current_bar: DollarBar,
    swing_lows: list[SwingPoint],
    volume_ma_20: float,
    volume_confirmation_ratio: float = 1.5,
) -> MSSEvent | None:
    """Detect bearish MSS: price breaks below recent swing low with volume confirmation.

    A bearish Market Structure Shift indicates a potential trend reversal
    to the downside or continuation of a downtrend.

    Args:
        current_bar: Most recent Dollar Bar
        swing_lows: List of confirmed swing lows
        volume_ma_20: 20-bar moving average of volume
        volume_confirmation_ratio: Minimum volume ratio for confirmation (default: 1.5x)

    Returns:
        MSSEvent if bearish MSS detected, None otherwise

    Examples:
        >>> swing_lows = [SwingPoint(price=11700.0, ...)]
        >>> bar = DollarBar(low=11600.0, volume=1500, ...)
        >>> detect_bearish_mss(bar, swing_lows, volume_ma_20=1000.0)
        MSSEvent(direction='bearish', breakout_price=11600.0, volume_ratio=1.5, ...)
    """
    if not swing_lows:
        return None

    most_recent_swing_low = swing_lows[-1]
    breakdown_price = current_bar.low

    # Check for breakdown (price must drop below swing low)
    if breakdown_price >= most_recent_swing_low.price:
        return None

    # Calculate volume ratio
    volume_ratio = current_bar.volume / volume_ma_20 if volume_ma_20 > 0 else 0.0

    # Volume confirmation (breakdown must have above-average volume)
    if volume_ratio < volume_confirmation_ratio:
        return None

    # Bearish MSS detected!
    return MSSEvent(
        timestamp=current_bar.timestamp,
        direction="bearish",
        breakout_price=breakdown_price,
        swing_point=most_recent_swing_low,
        volume_ratio=volume_ratio,
        bar_index=0,  # Will be set by MSSDetector
        confidence=0.0,  # Will be calculated in Story 2.6
    )

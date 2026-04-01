"""Breakout Detector for Opening Range Breakout strategy.

This module detects breakouts above/below the opening range with
volume confirmation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from src.data.models import DollarBar
from src.detection.opening_range_detector import OpeningRange

logger = logging.getLogger(__name__)


@dataclass
class BreakoutEvent:
    """Represents a breakout event.

    Attributes:
        timestamp: When the breakout occurred
        direction: "bullish" (above ORH) or "bearish" (below ORL)
        breakout_price: Price at which breakout occurred
        volume_ratio: Volume / opening range baseline
        or_high: Opening range high
        or_low: Opening range low
        bar: The bar that triggered the breakout
    """

    timestamp: datetime
    direction: str  # "bullish" or "bearish"
    breakout_price: float
    volume_ratio: float
    or_high: float
    or_low: float
    bar: DollarBar


class BreakoutDetector:
    """Detects breakouts from the opening range.

    A breakout is confirmed when:
    1. Price moves beyond ORH/ORL by threshold (default 1 tick)
    2. Volume is above baseline multiplier (default 1.5x)
    3. Bar closes beyond the breakout level (confirmation)

    Attributes:
        _tick_size: Tick size for futures contract (default 0.25 for MNQ)
        _volume_threshold: Volume multiplier threshold (default 1.5x)
        _breakout_threshold_ticks: Minimum ticks beyond ORH/ORL (default 1)
    """

    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size
    DEFAULT_VOLUME_THRESHOLD = 1.5  # 1.5x baseline volume
    DEFAULT_BREAKOUT_THRESHOLD_TICKS = 1  # 1 tick beyond ORH/ORL

    def __init__(
        self,
        tick_size: float = DEFAULT_TICK_SIZE,
        volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
        breakout_threshold_ticks: int = DEFAULT_BREAKOUT_THRESHOLD_TICKS,
    ) -> None:
        """Initialize Breakout Detector.

        Args:
            tick_size: Tick size for futures contract
            volume_threshold: Volume multiplier for confirmation
            breakout_threshold_ticks: Minimum ticks beyond ORH/ORL for breakout
        """
        self._tick_size = tick_size
        self._volume_threshold = volume_threshold
        self._breakout_threshold_ticks = breakout_threshold_ticks

    def detect_breakout(
        self, bar: DollarBar, opening_range: OpeningRange
    ) -> BreakoutEvent | None:
        """Detect if bar shows breakout from opening range.

        Args:
            bar: Current bar to check for breakout
            opening_range: Opening range data

        Returns:
            BreakoutEvent if breakout detected, None otherwise
        """
        if not opening_range.is_complete:
            logger.debug("Opening range not complete, skipping breakout detection")
            return None

        # Check for bullish breakout (above ORH)
        if self._check_bullish_breakout(bar, opening_range):
            return self._create_bullish_breakout(bar, opening_range)

        # Check for bearish breakout (below ORL)
        if self._check_bearish_breakout(bar, opening_range):
            return self._create_bearish_breakout(bar, opening_range)

        return None

    def _check_bullish_breakout(
        self, bar: DollarBar, opening_range: OpeningRange
    ) -> bool:
        """Check if bar shows bullish breakout above ORH.

        Args:
            bar: Bar to check
            opening_range: Opening range data

        Returns:
            True if bullish breakout detected
        """
        # Price must exceed ORH by threshold
        threshold_price = (
            opening_range.high
            + (self._breakout_threshold_ticks * self._tick_size)
        )

        if bar.high <= threshold_price:
            return False

        # Volume must be above threshold
        volume_ratio = bar.volume / opening_range.volume_baseline
        if volume_ratio < self._volume_threshold:
            logger.debug(
                f"Volume {bar.volume} below threshold "
                f"{opening_range.volume_baseline * self._volume_threshold:.0f}"
            )
            return False

        # Close must confirm (stay above ORH)
        if bar.close <= opening_range.high:
            logger.debug(f"Close {bar.close} did not confirm above ORH {opening_range.high}")
            return False

        return True

    def _check_bearish_breakout(
        self, bar: DollarBar, opening_range: OpeningRange
    ) -> bool:
        """Check if bar shows bearish breakout below ORL.

        Args:
            bar: Bar to check
            opening_range: Opening range data

        Returns:
            True if bearish breakout detected
        """
        # Price must go below ORL by threshold
        threshold_price = (
            opening_range.low - (self._breakout_threshold_ticks * self._tick_size)
        )

        if bar.low >= threshold_price:
            return False

        # Volume must be above threshold
        volume_ratio = bar.volume / opening_range.volume_baseline
        if volume_ratio < self._volume_threshold:
            logger.debug(
                f"Volume {bar.volume} below threshold "
                f"{opening_range.volume_baseline * self._volume_threshold:.0f}"
            )
            return False

        # Close must confirm (stay below ORL)
        if bar.close >= opening_range.low:
            logger.debug(f"Close {bar.close} did not confirm below ORL {opening_range.low}")
            return False

        return True

    def _create_bullish_breakout(
        self, bar: DollarBar, opening_range: OpeningRange
    ) -> BreakoutEvent:
        """Create bullish breakout event.

        Args:
            bar: Bar that triggered breakout
            opening_range: Opening range data

        Returns:
            BreakoutEvent for bullish breakout
        """
        volume_ratio = bar.volume / opening_range.volume_baseline

        breakout = BreakoutEvent(
            timestamp=bar.timestamp,
            direction="bullish",
            breakout_price=bar.high,
            volume_ratio=volume_ratio,
            or_high=opening_range.high,
            or_low=opening_range.low,
            bar=bar,
        )

        logger.info(
            f"BULLISH breakout detected: price {bar.high:.2f} > ORH {opening_range.high:.2f}, "
            f"volume ratio {volume_ratio:.2f}"
        )

        return breakout

    def _create_bearish_breakout(
        self, bar: DollarBar, opening_range: OpeningRange
    ) -> BreakoutEvent:
        """Create bearish breakout event.

        Args:
            bar: Bar that triggered breakout
            opening_range: Opening range data

        Returns:
            BreakoutEvent for bearish breakout
        """
        volume_ratio = bar.volume / opening_range.volume_baseline

        breakout = BreakoutEvent(
            timestamp=bar.timestamp,
            direction="bearish",
            breakout_price=bar.low,
            volume_ratio=volume_ratio,
            or_high=opening_range.high,
            or_low=opening_range.low,
            bar=bar,
        )

        logger.info(
            f"BEARISH breakout detected: price {bar.low:.2f} < ORL {opening_range.low:.2f}, "
            f"volume ratio {volume_ratio:.2f}"
        )

        return breakout

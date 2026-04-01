"""Rejection Detector for VWAP Bounce strategy.

This module detects rejection candles at VWAP levels, which indicate
potential support/resistance and trading opportunities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


@dataclass
class RejectionEvent:
    """Represents a rejection candle event at VWAP.

    A rejection occurs when price touches VWAP and closes away from it,
    indicating support/resistance at that level.
    """

    timestamp: datetime
    rejection_type: str  # "bullish" or "bearish"
    vwap_value: float
    touch_price: float
    close_price: float
    distance_ticks: float
    volume_ratio: float
    bar: DollarBar


class RejectionDetector:
    """Detects rejection candles at VWAP levels.

    A rejection candle is identified when:
    1. Price touches VWAP (within 2 ticks)
    2. Bar closes away from VWAP
    3. Volume is above average (confirmation)

    Attributes:
        _tick_size: Tick size for futures contract (default 0.25 for MNQ)
        _volume_threshold: Multiplier for average volume (default 1.2x)
        _max_ticks_away: Maximum ticks from VWAP to consider (default 2)
        _lookback_period: Bars to look back for average volume (default 20)
    """

    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size
    DEFAULT_VOLUME_THRESHOLD = 1.2  # 1.2x average volume
    DEFAULT_MAX_TICKS_AWAY = 2  # Within 2 ticks of VWAP
    DEFAULT_LOOKBACK_PERIOD = 20  # 20 bars for average volume

    def __init__(
        self,
        tick_size: float = DEFAULT_TICK_SIZE,
        volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
        max_ticks_away: float = DEFAULT_MAX_TICKS_AWAY,
        lookback_period: int = DEFAULT_LOOKBACK_PERIOD,
    ) -> None:
        """Initialize Rejection Detector.

        Args:
            tick_size: Tick size for futures contract
            volume_threshold: Volume multiplier threshold (default 1.2x avg)
            max_ticks_away: Maximum ticks from VWAP to consider rejection
            lookback_period: Number of bars to look back for average volume
        """
        self._tick_size = tick_size
        self._volume_threshold = volume_threshold
        self._max_ticks_away = max_ticks_away
        self._lookback_period = lookback_period

    def detect_rejection(
        self,
        bar: DollarBar,
        vwap_value: float,
        historical_bars: list[DollarBar],
    ) -> RejectionEvent | None:
        """Detect if a bar shows rejection at VWAP.

        Args:
            bar: Current bar to check for rejection
            vwap_value: Current VWAP value
            historical_bars: Historical bars for volume average calculation

        Returns:
            RejectionEvent if rejection detected, None otherwise
        """
        if vwap_value == 0:
            return None

        # Check if bar touched VWAP (within max_ticks_away)
        ticks_from_vwap_low = abs(bar.low - vwap_value) / self._tick_size
        ticks_from_vwap_high = abs(bar.high - vwap_value) / self._tick_size

        touched_vwap = (
            ticks_from_vwap_low <= self._max_ticks_away
            or ticks_from_vwap_high <= self._max_ticks_away
        )

        if not touched_vwap:
            return None

        # Determine which price touched VWAP
        if ticks_from_vwap_low <= self._max_ticks_away:
            touch_price = bar.low
        else:
            touch_price = bar.high

        # Check if bar closed away from VWAP (rejection)
        # For bullish rejection: touched from below, closed above
        # For bearish rejection: touched from above, closed below
        close_diff_ticks = (bar.close - vwap_value) / self._tick_size

        if abs(close_diff_ticks) <= 1:  # Closed too close to VWAP
            return None

        # Determine rejection type
        if bar.close > vwap_value and bar.low <= vwap_value:
            rejection_type = "bullish"  # Touched from below, rejected upward
        elif bar.close < vwap_value and bar.high >= vwap_value:
            rejection_type = "bearish"  # Touched from above, rejected downward
        else:
            return None  # No clear rejection

        # Check volume confirmation
        avg_volume = self._calculate_average_volume(historical_bars)
        if avg_volume == 0:
            volume_ratio = 1.0
        else:
            volume_ratio = bar.volume / avg_volume

        if volume_ratio < self._volume_threshold:
            logger.debug(
                f"Volume {bar.volume} below threshold "
                f"{avg_volume * self._volume_threshold:.0f}"
            )
            return None

        # Calculate distance from VWAP to touch point
        distance_ticks = abs(touch_price - vwap_value) / self._tick_size

        rejection = RejectionEvent(
            timestamp=bar.timestamp,
            rejection_type=rejection_type,
            vwap_value=vwap_value,
            touch_price=touch_price,
            close_price=bar.close,
            distance_ticks=distance_ticks,
            volume_ratio=volume_ratio,
            bar=bar,
        )

        logger.info(
            f"{rejection_type.upper()} rejection detected at VWAP {vwap_value:.2f}, "
            f"close {bar.close:.2f}, volume ratio {volume_ratio:.2f}"
        )

        return rejection

    def _calculate_average_volume(self, bars: list[DollarBar]) -> float:
        """Calculate average volume from historical bars.

        Args:
            bars: List of historical bars

        Returns:
            Average volume, or 0 if no bars
        """
        if not bars:
            return 0.0

        # Use last N bars for average
        lookback_bars = bars[-self._lookback_period :] if len(bars) > self._lookback_period else bars

        total_volume = sum(bar.volume for bar in lookback_bars)
        return total_volume / len(lookback_bars)

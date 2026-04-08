"""Opening Range Detector for Opening Range Breakout strategy.

This module tracks the opening range (first hour of trading) and
calculates key levels for breakout trading.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time

import pytz

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


@dataclass
class OpeningRange:
    """Represents the opening range data.

    Attributes:
        or_start: Start time of opening range
        or_end: End time of opening range
        high: Highest price during opening range
        low: Lowest price during opening range
        volume_baseline: Average volume per bar during opening range
        is_complete: Whether opening range period is complete
    """

    or_start: datetime
    or_end: datetime
    high: float
    low: float
    volume_baseline: float
    is_complete: bool


class OpeningRangeDetector:
    """Tracks and calculates the opening range (9:30 AM - 10:30 AM ET).

    The opening range is the first hour of trading and is used
    to identify key support/resistance levels for breakout trading.

    Attributes:
        _or_start: Opening range start time (default 9:30 AM)
        _or_end: Opening range end time (default 10:30 AM)
        _timezone: Timezone for time calculations (default America/New_York)
        _current_or_bars: Bars collected during current opening range
        _current_or_date: Date of current opening range
        _last_or: Last completed opening range data
    """

    DEFAULT_OR_START = time(9, 30)  # 9:30 AM
    DEFAULT_OR_END = time(10, 30)  # 10:30 AM
    DEFAULT_TIMEZONE = "America/New_York"

    def __init__(
        self,
        or_start: time = DEFAULT_OR_START,
        or_end: time = DEFAULT_OR_END,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        """Initialize Opening Range Detector.

        Args:
            or_start: Opening range start time (default 9:30 AM)
            or_end: Opening range end time (default 10:30 AM)
            timezone: Timezone for time calculations
        """
        self._or_start = or_start
        self._or_end = or_end
        self._timezone = pytz.timezone(timezone)

        self._current_or_bars: list[DollarBar] = []
        self._current_or_date: datetime | None = None
        self._last_or: OpeningRange | None = None

    def process_bar(self, bar: DollarBar) -> None:
        """Process a dollar bar and update opening range if applicable.

        Args:
            bar: Dollar bar to process
        """
        # Convert bar timestamp to ET
        # Handle naive datetime by making it timezone-aware
        if bar.timestamp.tzinfo is None:
            # Naive datetime - assume it's in the configured timezone
            bar_time_et = bar.timestamp.replace(tzinfo=self._timezone)
        else:
            bar_time_et = bar.timestamp.astimezone(self._timezone)

        # Check if we need to reset (new day detected)
        if self._current_or_date is None:
            # First bar - set date
            self._current_or_date = bar_time_et.date()
        elif bar_time_et.date() != self._current_or_date:
            # New day - reset and start fresh
            self._reset_opening_range()
            self._current_or_date = bar_time_et.date()

        # Check if bar is within opening range time window
        bar_time = bar_time_et.time()
        if self._or_start <= bar_time < self._or_end:
            # Bar is in opening range - collect it
            self._current_or_bars.append(bar)
            logger.debug(f"Added bar to opening range: {bar.timestamp}")
        elif bar_time >= self._or_end:
            # Opening range period has ended - finalize if we have bars and not yet finalized
            if self._current_or_bars and self._last_or is None:
                self._finalize_opening_range(bar.timestamp)
        # If we have collected bars and see a new bar after OR, finalize was already done
        # Don't collect bars after OR period

    def get_opening_range(self) -> OpeningRange | None:
        """Get the current or last completed opening range.

        Returns:
            OpeningRange data if available, None otherwise
        """
        # Return last completed OR if available
        if self._last_or:
            return self._last_or

        # Check if current OR period is complete
        # Only finalize if we have bars and the last bar is past OR end time
        if self._current_or_bars:
            last_bar_time = self._current_or_bars[-1].timestamp
            # Handle naive datetime
            if last_bar_time.tzinfo is None:
                last_bar_time_et = last_bar_time.replace(tzinfo=self._timezone)
            else:
                last_bar_time_et = last_bar_time.astimezone(self._timezone)

            # Only finalize if past OR end time
            if last_bar_time_et.time() >= self._or_end:
                self._finalize_opening_range(self._current_or_bars[-1].timestamp)
                return self._last_or

        return None

    def _finalize_opening_range(self, timestamp: datetime) -> None:
        """Finalize the opening range calculations.

        Args:
            timestamp: Timestamp to use as OR end time
        """
        if not self._current_or_bars:
            return

        # Calculate ORH and ORL
        high = max(bar.high for bar in self._current_or_bars)
        low = min(bar.low for bar in self._current_or_bars)

        # Calculate volume baseline (average per bar)
        total_volume = sum(bar.volume for bar in self._current_or_bars)
        volume_baseline = total_volume / len(self._current_or_bars)

        # Create opening range start datetime
        or_start = self._current_or_bars[0].timestamp

        # Create OpeningRange object
        self._last_or = OpeningRange(
            or_start=or_start,
            or_end=timestamp,
            high=high,
            low=low,
            volume_baseline=volume_baseline,
            is_complete=True,
        )

        logger.info(
            f"Opening Range finalized: ORH={high:.2f}, ORL={low:.2f}, "
            f"Volume Baseline={volume_baseline:.0f}"
        )

    def _reset_opening_range(self) -> None:
        """Reset for new trading day."""
        self._current_or_bars = []
        self._last_or = None
        logger.debug("Opening range detector reset for new day")

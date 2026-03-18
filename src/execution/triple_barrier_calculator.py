"""Triple barrier calculation for position exit strategy.

This module calculates the three barriers (upper, lower, time) for
systematic position management based on the triple barrier method
from Advances in Financial Machine Learning.

Features:
- Upper barrier: Take profit at ±$15 from entry
- Lower barrier: Stop loss at ±$7.50 from entry (1:2 risk-reward)
- Time barrier: Exit at 19:00 UTC (13:00 CT) same day
- Direction-specific calculations (bullish vs bearish)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TripleBarriers:
    """Triple barrier levels for position exit.

    Attributes:
        upper_barrier_price: Take profit price (entry + $15 for long)
        lower_barrier_price: Stop loss price (entry - $7.50 for long)
        time_barrier_utc: Max hold time (13:00 CT = 19:00 UTC)
        entry_price: Position entry price
        direction: Position direction (affects barrier calculation)
    """

    upper_barrier_price: float
    lower_barrier_price: float
    time_barrier_utc: datetime
    entry_price: float
    direction: str


class TripleBarrierCalculator:
    """Calculate triple barriers for position exit.

    Implements the triple barrier method with fixed take profit ($15)
    and stop loss ($7.50) levels, plus same-day time barrier.

    Attributes:
        _upper_barrier_points: Take profit in points (default $15.00)
        _lower_barrier_points: Stop loss in points (default $7.50)
        _time_barrier_hour_utc: Max hold hour UTC (default 19 for 13:00 CT)

    Example:
        >>> calculator = TripleBarrierCalculator()
        >>> barriers = calculator.calculate_barriers(
        ...     entry_price=11800.00,
        ...     direction="bullish",
        ...     entry_time=datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)
        ... )
        >>> barriers.upper_barrier_price
        11815.00  # Entry + $15
        >>> barriers.lower_barrier_price
        11792.50  # Entry - $7.50
    """

    UPPER_BARRIER_POINTS = 15.00  # $15.00 take profit
    LOWER_BARRIER_POINTS = 7.50  # $7.50 stop loss
    TIME_BARRIER_HOUR_UTC = 19  # 13:00 CT = 19:00 UTC

    def __init__(
        self,
        upper_barrier_points: float = UPPER_BARRIER_POINTS,
        lower_barrier_points: float = LOWER_BARRIER_POINTS,
        time_barrier_hour_utc: int = TIME_BARRIER_HOUR_UTC,
    ) -> None:
        """Initialize triple barrier calculator.

        Args:
            upper_barrier_points: Take profit in points (default $15.00)
            lower_barrier_points: Stop loss in points (default $7.50)
            time_barrier_hour_utc: Max hold hour UTC (default 19)
        """
        self._upper_barrier_points = upper_barrier_points
        self._lower_barrier_points = lower_barrier_points
        self._time_barrier_hour_utc = time_barrier_hour_utc

        logger.info(
            "TripleBarrierCalculator initialized: "
            "upper=${:.2f}, lower=${:.2f}, time_barrier={}UTC".format(
                upper_barrier_points,
                lower_barrier_points,
                time_barrier_hour_utc
            )
        )

    def calculate_barriers(
        self,
        entry_price: float,
        direction: str,
        entry_time: datetime
    ) -> TripleBarriers:
        """Calculate triple barriers for position.

        Args:
            entry_price: Position entry price
            direction: "bullish" (long) or "bearish" (short)
            entry_time: Position entry timestamp

        Returns:
            TripleBarriers with calculated levels

        Raises:
            ValueError: If direction not recognized

        Example:
            >>> calculator = TripleBarrierCalculator()
            >>> barriers = calculator.calculate_barriers(
            ...     entry_price=11800.00,
            ...     direction="bullish",
            ...     entry_time=datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)
            ... )
            >>> barriers.upper_barrier_price
            11815.00
        """
        if direction == "bullish":
            # Long position: profit when price goes up
            upper_barrier = entry_price + self._upper_barrier_points
            lower_barrier = entry_price - self._lower_barrier_points

        elif direction == "bearish":
            # Short position: profit when price goes down
            upper_barrier = entry_price - self._upper_barrier_points
            lower_barrier = entry_price + self._lower_barrier_points

        else:
            raise ValueError(
                "Invalid direction: {}. Expected 'bullish' or 'bearish'".format(
                    direction
                )
            )

        # Calculate time barrier
        time_barrier = self._calculate_time_barrier(entry_time)

        logger.debug(
            "Barriers calculated: entry=${:.2f}, upper=${:.2f}, "
            "lower=${:.2f}, time_barrier={}".format(
                entry_price,
                upper_barrier,
                lower_barrier,
                time_barrier.isoformat()
            )
        )

        return TripleBarriers(
            upper_barrier_price=upper_barrier,
            lower_barrier_price=lower_barrier,
            time_barrier_utc=time_barrier,
            entry_price=entry_price,
            direction=direction,
        )

    def _calculate_time_barrier(self, entry_time: datetime) -> datetime:
        """Calculate time barrier datetime.

        Args:
            entry_time: Position entry timestamp

        Returns:
            Time barrier datetime (19:00 UTC same day or next day)

        Logic:
            - If entry_time is before 19:00 UTC, barrier is 19:00 UTC same day
            - If entry_time is after 19:00 UTC, barrier is 19:00 UTC next day
        """
        barrier_time = entry_time.replace(
            hour=self._time_barrier_hour_utc,
            minute=0,
            second=0,
            microsecond=0
        )

        # If entry time is after barrier time, move to next day
        if entry_time >= barrier_time:
            # Add one day
            barrier_time = barrier_time + timedelta(days=1)

        return barrier_time

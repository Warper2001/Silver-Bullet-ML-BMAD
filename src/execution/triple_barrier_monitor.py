"""Triple barrier monitoring for position exit detection.

This module monitors active positions to detect when any of the three
barriers (upper, lower, time) have been hit and position should be exited.

Features:
- Upper barrier monitoring (take profit)
- Lower barrier monitoring (stop loss)
- Time barrier monitoring (max hold time)
- Direction-specific barrier checking
- Barrier priority handling (upper > lower > time)
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from src.execution.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


@dataclass
class BarrierCheckResult:
    """Result of checking if any barrier was hit.

    Attributes:
        barrier_hit: Which barrier was hit ("UPPER", "LOWER", "TIME", or None)
        hit_time: When barrier was hit
        current_price: Price at barrier hit
        exit_price: Calculated exit price
        exit_reason: Human-readable exit reason
        should_exit: Whether position should be exited
    """

    barrier_hit: str | None
    hit_time: datetime
    current_price: float
    exit_price: float
    exit_reason: str
    should_exit: bool


class TripleBarrierMonitor:
    """Monitor positions for triple barrier hits.

    Continuously checks if current price has hit upper/lower barriers
    or if time barrier has been reached.

    Attributes:
        _position_tracker: PositionTracker for active positions

    Example:
        >>> monitor = TripleBarrierMonitor(tracker)
        >>> result = monitor.check_barriers(
        ...     order_id="ORDER-123",
        ...     current_price=11815.50,
        ...     current_time=datetime.now(timezone.utc)
        ... )
        >>> if result.should_exit:
        ...     # Exit position
    """

    def __init__(self, position_tracker: PositionTracker) -> None:
        """Initialize triple barrier monitor.

        Args:
            position_tracker: PositionTracker instance

        Raises:
            ValueError: If position_tracker is None
        """
        if position_tracker is None:
            raise ValueError("Position tracker cannot be None")

        self._position_tracker = position_tracker

        logger.info("TripleBarrierMonitor initialized")

    def check_barriers(
        self,
        order_id: str,
        current_price: float,
        current_time: datetime
    ) -> BarrierCheckResult:
        """Check if any barrier has been hit.

        Args:
            order_id: Position order ID to check
            current_price: Current MNQ price
            current_time: Current timestamp

        Returns:
            BarrierCheckResult with exit recommendation

        Barrier Priority:
            1. Check price barriers first (upper/lower)
            2. Then check time barrier
            3. Return first barrier hit

        Example:
            >>> result = monitor.check_barriers(
            ...     order_id="ORDER-123",
            ...     current_price=11815.50,
            ...     current_time=datetime.now(timezone.utc)
            ... )
            >>> result.barrier_hit
            'UPPER'
        """
        # Get position
        position = self._position_tracker.get_position(order_id)

        if position is None:
            logger.warning("Position not found: {}".format(order_id))
            return BarrierCheckResult(
                barrier_hit=None,
                hit_time=current_time,
                current_price=current_price,
                exit_price=current_price,
                exit_reason="Position not found",
                should_exit=False
            )

        # Check if barriers are set
        if position.upper_barrier_price is None:
            logger.warning(
                "Barriers not set for order: {}".format(order_id)
            )
            return BarrierCheckResult(
                barrier_hit=None,
                hit_time=current_time,
                current_price=current_price,
                exit_price=current_price,
                exit_reason="Barriers not set",
                should_exit=False
            )

        # Check upper barrier (take profit)
        upper_result = self._check_upper_barrier(
            position, current_price, current_time
        )
        if upper_result.barrier_hit == "UPPER":
            return upper_result

        # Check lower barrier (stop loss)
        lower_result = self._check_lower_barrier(
            position, current_price, current_time
        )
        if lower_result.barrier_hit == "LOWER":
            return lower_result

        # Check time barrier
        time_result = self._check_time_barrier(
            position, current_price, current_time
        )
        if time_result.barrier_hit == "TIME":
            return time_result

        # No barrier hit
        return BarrierCheckResult(
            barrier_hit=None,
            hit_time=current_time,
            current_price=current_price,
            exit_price=current_price,
            exit_reason="No barrier hit",
            should_exit=False
        )

    def _check_upper_barrier(
        self,
        position,
        current_price: float,
        current_time: datetime
    ) -> BarrierCheckResult:
        """Check if upper barrier (take profit) has been hit.

        Args:
            position: Position object
            current_price: Current MNQ price
            current_time: Current timestamp

        Returns:
            BarrierCheckResult with UPPER barrier hit or None
        """
        upper_barrier = position.upper_barrier_price

        if position.direction == "bullish":
            # Long: upper barrier hit when price >= barrier
            hit = current_price >= upper_barrier
        else:  # bearish
            # Short: upper barrier hit when price <= barrier
            hit = current_price <= upper_barrier

        if hit:
            return BarrierCheckResult(
                barrier_hit="UPPER",
                hit_time=current_time,
                current_price=current_price,
                exit_price=current_price,
                exit_reason=(
                    "Upper barrier hit: price ${:.2f} >= ${:.2f}".format(
                        current_price, upper_barrier
                    )
                ),
                should_exit=True
            )

        return BarrierCheckResult(
            barrier_hit=None,
            hit_time=current_time,
            current_price=current_price,
            exit_price=current_price,
            exit_reason="",
            should_exit=False
        )

    def _check_lower_barrier(
        self,
        position,
        current_price: float,
        current_time: datetime
    ) -> BarrierCheckResult:
        """Check if lower barrier (stop loss) has been hit.

        Args:
            position: Position object
            current_price: Current MNQ price
            current_time: Current timestamp

        Returns:
            BarrierCheckResult with LOWER barrier hit or None
        """
        lower_barrier = position.lower_barrier_price

        if position.direction == "bullish":
            # Long: lower barrier hit when price <= barrier
            hit = current_price <= lower_barrier
        else:  # bearish
            # Short: lower barrier hit when price >= barrier
            hit = current_price >= lower_barrier

        if hit:
            return BarrierCheckResult(
                barrier_hit="LOWER",
                hit_time=current_time,
                current_price=current_price,
                exit_price=current_price,
                exit_reason=(
                    "Lower barrier hit: price ${:.2f} <= ${:.2f}".format(
                        current_price, lower_barrier
                    )
                ),
                should_exit=True
            )

        return BarrierCheckResult(
            barrier_hit=None,
            hit_time=current_time,
            current_price=current_price,
            exit_price=current_price,
            exit_reason="",
            should_exit=False
        )

    def _check_time_barrier(
        self,
        position,
        current_price: float,
        current_time: datetime
    ) -> BarrierCheckResult:
        """Check if time barrier (max hold time) has been hit.

        Args:
            position: Position object
            current_price: Current MNQ price
            current_time: Current timestamp

        Returns:
            BarrierCheckResult with TIME barrier hit or None
        """
        time_barrier = position.time_barrier_utc

        if time_barrier is None:
            return BarrierCheckResult(
                barrier_hit=None,
                hit_time=current_time,
                current_price=current_price,
                exit_price=current_price,
                exit_reason="",
                should_exit=False
            )

        hit = current_time >= time_barrier

        if hit:
            return BarrierCheckResult(
                barrier_hit="TIME",
                hit_time=current_time,
                current_price=current_price,
                exit_price=current_price,
                exit_reason=(
                    "Time barrier hit: {} >= {}".format(
                        current_time.isoformat(),
                        time_barrier.isoformat()
                    )
                ),
                should_exit=True
            )

        return BarrierCheckResult(
            barrier_hit=None,
            hit_time=current_time,
            current_price=current_price,
            exit_price=current_price,
            exit_reason="",
            should_exit=False
        )

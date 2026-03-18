"""Position monitoring service for automated exit execution.

This module orchestrates the monitoring of active positions and automatic
execution of exits when triple barriers are hit. It integrates barrier
calculation, monitoring, and execution into a unified service.

Features:
- Automatic barrier calculation on position entry
- Continuous monitoring on price updates
- Automatic exit execution when barriers hit
- Position status tracking and queries
- CSV audit trail logging
- Concurrent position handling
- Performance: <50ms to check all positions
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.execution.position_tracker import PositionTracker
from src.execution.triple_barrier_calculator import TripleBarrierCalculator
from src.execution.triple_barrier_monitor import (
    TripleBarrierMonitor,
    BarrierCheckResult,
)
from src.execution.triple_barrier_exit_executor import (
    TripleBarrierExitExecutor,
    ExitResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionStatus:
    """Real-time status of active position.

    Attributes:
        order_id: Position order ID
        entry_price: Position entry price
        current_price: Current market price
        upper_barrier: Upper barrier price
        lower_barrier: Lower barrier price
        time_barrier: Time barrier datetime
        distance_to_upper: Points to upper barrier (negative if hit)
        distance_to_lower: Points to lower barrier (negative if hit)
        time_to_barrier: Time remaining until time barrier
        status: Position status ("ACTIVE", "UPPER_HIT", "LOWER_HIT", "TIME_HIT")
    """

    order_id: str
    entry_price: float
    current_price: float
    upper_barrier: float
    lower_barrier: float
    time_barrier: datetime
    distance_to_upper: float
    distance_to_lower: float
    time_to_barrier: timedelta
    status: str  # "ACTIVE", "UPPER_HIT", "LOWER_HIT", or "TIME_HIT"


class PositionMonitoringService:
    """Continuously monitors active positions and executes exits.

    Integrates triple barrier calculation, monitoring, and execution
    into a unified service that runs during market hours.

    Attributes:
        _calculator: TripleBarrierCalculator for barrier levels
        _monitor: TripleBarrierMonitor for barrier checking
        _executor: TripleBarrierExitExecutor for exit execution
        _position_tracker: PositionTracker for active positions
        _api_client: TradeStation API client
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> service = PositionMonitoringService(
        ...     calculator=calculator,
        ...     monitor=monitor,
        ...     executor=executor,
        ...     position_tracker=tracker,
        ...     api_client=api_client
        ... )
        >>> service.on_position_entered(
        ...     order_id="ORDER-123",
        ...     entry_price=11800.00,
        ...     quantity=5,
        ...     direction="bullish",
        ...     entry_time=datetime.now(timezone.utc)
        ... )
        >>> exits = service.on_price_update(
        ...     current_price=11815.50,
        ...     current_time=datetime.now(timezone.utc)
        ... )
    """

    def __init__(
        self,
        calculator: TripleBarrierCalculator,
        monitor: TripleBarrierMonitor,
        executor: TripleBarrierExitExecutor,
        position_tracker: PositionTracker,
        api_client,
        audit_trail_path: Optional[str] = None,
    ) -> None:
        """Initialize position monitoring service.

        Args:
            calculator: TripleBarrierCalculator instance
            monitor: TripleBarrierMonitor instance
            executor: TripleBarrierExitExecutor instance
            position_tracker: PositionTracker instance
            api_client: TradeStation API client
            audit_trail_path: Path to CSV audit trail file (optional)

        Raises:
            ValueError: If required dependencies are None
        """
        if calculator is None:
            raise ValueError("Calculator cannot be None")
        if monitor is None:
            raise ValueError("Monitor cannot be None")
        if executor is None:
            raise ValueError("Executor cannot be None")
        if position_tracker is None:
            raise ValueError("Position tracker cannot be None")
        if api_client is None:
            raise ValueError("API client cannot be None")

        self._calculator = calculator
        self._monitor = monitor
        self._executor = executor
        self._position_tracker = position_tracker
        self._api_client = api_client
        self._audit_trail_path = audit_trail_path

        logger.info("PositionMonitoringService initialized")

    def on_position_entered(
        self,
        order_id: str,
        entry_price: float,
        quantity: int,
        direction: str,
        entry_time: datetime,
    ) -> None:
        """Calculate and store barriers when position is entered.

        Args:
            order_id: Position order ID
            entry_price: Position entry price
            quantity: Position size (contracts)
            direction: Position direction ("bullish" or "bearish")
            entry_time: Position entry timestamp

        Example:
            >>> service.on_position_entered(
            ...     order_id="ORDER-123",
            ...     entry_price=11800.00,
            ...     quantity=5,
            ...     direction="bullish",
            ...     entry_time=datetime.now(timezone.utc)
            ... )
        """
        # Calculate barriers
        barriers = self._calculator.calculate_barriers(
            entry_price=entry_price,
            direction=direction,
            entry_time=entry_time
        )

        # Get or create position
        position = self._position_tracker.get_position(order_id)

        if position is None:
            # Create position if it doesn't exist
            from src.execution.position_tracker import Position
            position = Position(
                order_id=order_id,
                signal_id="",  # Unknown at this point
                entry_price=entry_price,
                quantity=quantity,
                direction=direction,
                order_type="LIMIT",  # Assume LIMIT (most common for entries)
                timestamp=entry_time
            )
            self._position_tracker.add_position(position)
            logger.info("Position created: {}".format(order_id))

        # Store barriers in position
        position.upper_barrier_price = barriers.upper_barrier_price
        position.lower_barrier_price = barriers.lower_barrier_price
        position.time_barrier_utc = barriers.time_barrier_utc

        logger.info(
            "Position barriers stored: order_id={}, upper=${:.2f}, "
            "lower=${:.2f}, time_barrier={}".format(
                order_id,
                barriers.upper_barrier_price,
                barriers.lower_barrier_price,
                barriers.time_barrier_utc.isoformat()
            )
        )

        # Log to audit trail
        if self._audit_trail_path:
            self._log_position_entry(
                order_id=order_id,
                entry_price=entry_price,
                upper_barrier=barriers.upper_barrier_price,
                lower_barrier=barriers.lower_barrier_price,
                time_barrier=barriers.time_barrier_utc
            )

    def on_price_update(
        self,
        current_price: float,
        current_time: datetime
    ) -> list[ExitResult]:
        """Check all active positions for barrier hits on price update.

        Called when new dollar bar is received (every 30-60 seconds).

        Args:
            current_price: Current MNQ price
            current_time: Current timestamp

        Returns:
            List of ExitResult for positions that were exited

        Example:
            >>> exits = service.on_price_update(
            ...     current_price=11815.50,
            ...     current_time=datetime.now(timezone.utc)
            ... )
            >>> for exit_result in exits:
            ...     print(f"Exited {exit_result.order_id}: {exit_result.exit_barrier}")
        """
        # Get all active positions
        # Note: In production, you'd want a more efficient way to get all positions
        # For now, we'll iterate through the tracker's internal state
        all_order_ids = list(self._position_tracker._positions.keys())

        exits = []

        for order_id in all_order_ids:
            # Check if position should exit
            exit_result = self._executor.execute_exit(
                order_id=order_id,
                current_price=current_price,
                current_time=current_time
            )

            # If exit was executed (not just no barrier hit)
            if exit_result.success and exit_result.exit_order_id is not None:
                exits.append(exit_result)
                logger.info(
                    "Position exited: order_id={}, barrier={}, price=${:.2f}".format(
                        order_id,
                        exit_result.exit_barrier,
                        current_price
                    )
                )

        return exits

    def monitor_all_positions(
        self,
        current_price: float,
        current_time: datetime
    ) -> dict[str, BarrierCheckResult]:
        """Check all active positions for barrier hits.

        Args:
            current_price: Current MNQ price
            current_time: Current timestamp

        Returns:
            Dictionary mapping order_id to BarrierCheckResult

        Note:
            This checks barriers but does NOT execute exits.
            Use execute_exits() to execute exits for hit positions.

        Example:
            >>> results = service.monitor_all_positions(
            ...     current_price=11815.50,
            ...     current_time=datetime.now(timezone.utc)
            ... )
            >>> for order_id, result in results.items():
            ...     if result.should_exit:
            ...         print(f"{order_id}: {result.barrier_hit}")
        """
        # Get all active positions
        all_order_ids = list(self._position_tracker._positions.keys())

        results = {}

        for order_id in all_order_ids:
            # Check barriers
            barrier_result = self._monitor.check_barriers(
                order_id=order_id,
                current_price=current_price,
                current_time=current_time
            )
            results[order_id] = barrier_result

        return results

    def execute_exits(
        self,
        barrier_results: dict[str, BarrierCheckResult]
    ) -> list[ExitResult]:
        """Execute exits for positions that hit barriers.

        Args:
            barrier_results: Dictionary of order_id to BarrierCheckResult

        Returns:
            List of ExitResult for executed exits

        Example:
            >>> barrier_results = service.monitor_all_positions(price, time)
            >>> exits = service.execute_exits(barrier_results)
            >>> print(f"Executed {len(exits)} exits")
        """
        exits = []

        for order_id, barrier_result in barrier_results.items():
            # Only execute if barrier hit
            if barrier_result.should_exit:
                exit_result = self._executor.execute_exit(
                    order_id=order_id,
                    current_price=barrier_result.current_price,
                    current_time=barrier_result.hit_time
                )

                if exit_result.success:
                    exits.append(exit_result)

        return exits

    def get_position_status(
        self,
        order_id: str,
        current_price: float,
        current_time: datetime
    ) -> Optional[PositionStatus]:
        """Get real-time status of position.

        Args:
            order_id: Position order ID
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            PositionStatus with current state, or None if not found

        Example:
            >>> status = service.get_position_status(
            ...     order_id="ORDER-123",
            ...     current_price=11805.00,
            ...     current_time=datetime.now(timezone.utc)
            ... )
            >>> status.distance_to_upper
            10.0  # $10 away from upper barrier
        """
        position = self._position_tracker.get_position(order_id)

        if position is None:
            return None

        # Check barriers to determine status
        barrier_result = self._monitor.check_barriers(
            order_id=order_id,
            current_price=current_price,
            current_time=current_time
        )

        # Calculate distances
        if position.direction == "bullish":
            distance_to_upper = position.upper_barrier_price - current_price
            distance_to_lower = current_price - position.lower_barrier_price
        else:  # bearish
            distance_to_upper = current_price - position.upper_barrier_price
            distance_to_lower = position.lower_barrier_price - current_price

        # Calculate time to barrier
        if position.time_barrier_utc:
            time_to_barrier = position.time_barrier_utc - current_time
        else:
            time_to_barrier = timedelta(0)

        # Determine status
        if barrier_result.barrier_hit == "UPPER":
            status = "UPPER_HIT"
        elif barrier_result.barrier_hit == "LOWER":
            status = "LOWER_HIT"
        elif barrier_result.barrier_hit == "TIME":
            status = "TIME_HIT"
        else:
            status = "ACTIVE"

        return PositionStatus(
            order_id=order_id,
            entry_price=position.entry_price,
            current_price=current_price,
            upper_barrier=position.upper_barrier_price,
            lower_barrier=position.lower_barrier_price,
            time_barrier=position.time_barrier_utc,
            distance_to_upper=distance_to_upper,
            distance_to_lower=distance_to_lower,
            time_to_barrier=time_to_barrier,
            status=status
        )

    def get_all_positions_status(
        self,
        current_price: float,
        current_time: datetime
    ) -> list[PositionStatus]:
        """Get status of all active positions.

        Args:
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            List of PositionStatus for all active positions

        Example:
            >>> statuses = service.get_all_positions_status(
            ...     current_price=11805.00,
            ...     current_time=datetime.now(timezone.utc)
            ... )
            >>> for status in statuses:
            ...     print(f"{status.order_id}: {status.status}")
        """
        # Get all active positions
        all_order_ids = list(self._position_tracker._positions.keys())

        statuses = []

        for order_id in all_order_ids:
            status = self.get_position_status(
                order_id=order_id,
                current_price=current_price,
                current_time=current_time
            )
            if status is not None:
                statuses.append(status)

        return statuses

    def _log_position_entry(
        self,
        order_id: str,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float,
        time_barrier: datetime
    ) -> None:
        """Log position entry event to CSV audit trail.

        Args:
            order_id: Position order ID
            entry_price: Position entry price
            upper_barrier: Upper barrier price
            lower_barrier: Lower barrier price
            time_barrier: Time barrier datetime

        Example:
            >>> service._log_position_entry(
            ...     order_id="ORDER-123",
            ...     entry_price=11800.00,
            ...     upper_barrier=11815.00,
            ...     lower_barrier=11792.50,
            ...     time_barrier=datetime(2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc)
            ... )
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if file exists and has content
        file_exists = audit_path.exists() and audit_path.stat().st_size > 0

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "order_id",
                    "entry_price",
                    "upper_barrier",
                    "lower_barrier",
                    "time_barrier"
                ])

            # Write log entry
            writer.writerow([
                timestamp,
                "ENTER",
                order_id,
                "{:.2f}".format(entry_price),
                "{:.2f}".format(upper_barrier),
                "{:.2f}".format(lower_barrier),
                time_barrier.isoformat()
            ])

        logger.debug("Position monitoring audit trail updated: {}".format(
            audit_path
        ))

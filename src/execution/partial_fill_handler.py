"""Partial fill handling for limit orders.

This module detects and handles partial fills on limit orders by immediately
resubmitting unfilled quantities at updated limit prices, tracking cumulative
fill time, and handling timeouts.

Features:
- Partial fill detection (filled_quantity < target_quantity)
- Unfilled quantity calculation (target - filled)
- Immediate limit price recalculation (current_price ± 2 ticks)
- Order resubmission for unfilled quantity
- Cumulative fill time tracking
- 5-minute timeout handling with order cancellation
- CSV audit trail logging
- Performance monitoring (<100ms handling time)
"""

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.execution.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


@dataclass
class PartialFillResult:
    """Result of partial fill handling.

    Attributes:
        original_order_id: Original order ID that was partially filled
        filled_quantity: Quantity that has been filled
        unfilled_quantity: Quantity yet to be filled
        new_order_id: New order ID for unfilled quantity (None if fully filled)
        status: Final status (PARTIALLY_FILLED, FULLY_FILLED, UNFILLED_TIMEOUT)
        cumulative_time_seconds: Total time from first submission
        handling_time_ms: Time taken to handle partial fill
    """

    original_order_id: str
    filled_quantity: int
    unfilled_quantity: int
    new_order_id: Optional[str]
    status: str
    cumulative_time_seconds: float
    handling_time_ms: float


class PartialFillHandler:
    """Handles partial fills on limit orders.

    Detects partial fills, calculates unfilled quantities, recalculates
    limit prices, resubmits orders, tracks cumulative fill time, and
    handles timeouts.

    Attributes:
        _api_client: TradeStation API client (authenticated)
        _position_tracker: Position tracking storage
        _audit_trail_path: Path to CSV audit trail file
        _timeout_seconds: Timeout for unfilled orders (default 300s = 5 minutes)
        _tick_size: MNQ tick size ($0.25)
        _tick_offset: Number of ticks for limit price offset (2 ticks)

    Example:
        >>> api_client = TradeStationApiClient()
        >>> tracker = PositionTracker()
        >>> handler = PartialFillHandler(api_client, tracker)
        >>> result = handler.handle_partial_fill(
        ...     original_order_id="ORDER-123",
        ...     filled_quantity=2,
        ...     target_quantity=5,
        ...     current_price=11800.00,
        ...     direction="bullish"
        ... )
        >>> result.unfilled_quantity
        3
        >>> result.new_order_id
        'ORDER-789'
    """

    DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
    TICK_SIZE = 0.25  # MNQ tick size
    TICK_OFFSET = 2  # 2-tick offset for recalculated limit price

    def __init__(
        self,
        api_client,
        position_tracker: PositionTracker,
        audit_trail_path: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize partial fill handler.

        Args:
            api_client: Authenticated TradeStation API client
            position_tracker: Position tracking storage instance
            audit_trail_path: Path to CSV audit trail file (optional)
            timeout_seconds: Timeout for unfilled orders in seconds
                (default 300 = 5 minutes)

        Raises:
            ValueError: If api_client or position_tracker is None
        """
        if api_client is None:
            raise ValueError("API client cannot be None")
        if position_tracker is None:
            raise ValueError("Position tracker cannot be None")

        self._api_client = api_client
        self._position_tracker = position_tracker
        self._audit_trail_path = audit_trail_path
        self._timeout_seconds = timeout_seconds
        self._tick_size = self.TICK_SIZE
        self._tick_offset = self.TICK_OFFSET

        logger.info(
            "PartialFillHandler initialized: timeout={}s, audit_trail={}".format(
                timeout_seconds, audit_trail_path
            )
        )

    def calculate_unfilled_quantity(
        self, target_quantity: int, filled_quantity: int
    ) -> int:
        """Calculate unfilled quantity.

        Args:
            target_quantity: Target quantity (contracts)
            filled_quantity: Quantity filled so far

        Returns:
            Unfilled quantity (0 if fully filled)

        Example:
            >>> handler = PartialFillHandler(api_client, tracker)
            >>> unfilled = handler.calculate_unfilled_quantity(5, 2)
            >>> unfilled
            3
        """
        return target_quantity - filled_quantity

    def recalculate_limit_price(
        self, current_price: float, direction: str
    ) -> float:
        """Recalculate limit price with tick offset.

        Args:
            current_price: Current market price
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Recalculated limit price (current_price ± 2 × $0.25)

        Raises:
            ValueError: If direction not recognized

        Example:
            >>> handler = PartialFillHandler(api_client, tracker)
            >>> limit_price = handler.recalculate_limit_price(11800.00, "bullish")
            >>> limit_price
            11800.50
        """
        offset = self._tick_offset * self._tick_size

        if direction == "bullish":
            limit_price = current_price + offset
        elif direction == "bearish":
            limit_price = current_price - offset
        else:
            raise ValueError(
                "Invalid direction: {}. Expected 'bullish' or 'bearish'".format(
                    direction
                )
            )

        return limit_price

    def resubmit_unfilled_quantity(
        self,
        original_order_id: str,
        unfilled_quantity: int,
        direction: str,
        limit_price: float,
    ) -> str:
        """Resubmit unfilled quantity as new limit order.

        Args:
            original_order_id: Original order ID (for logging)
            unfilled_quantity: Quantity to resubmit
            direction: Signal direction ("bullish" or "bearish")
            limit_price: Recalculated limit price

        Returns:
            New order ID from TradeStation API

        Raises:
            Exception: If order submission fails

        Example:
            >>> handler = PartialFillHandler(api_client, tracker)
            >>> new_order_id = handler.resubmit_unfilled_quantity(
            ...     original_order_id="ORDER-123",
            ...     unfilled_quantity=3,
            ...     direction="bullish",
            ...     limit_price=11800.50
            ... )
            >>> new_order_id
            'ORDER-789'
        """
        # Convert direction to API side
        side = "BUY" if direction == "bullish" else "SELL"

        # Construct order payload
        payload = {
            "symbol": "MNQ",
            "quantity": unfilled_quantity,
            "side": side,
            "orderType": "LIMIT",
            "limitPrice": limit_price,
            "timeInForce": "DAY",
        }

        logger.info(
            "Resubmitting unfilled quantity: original_order={}, qty={}, "
            "limitPrice=${:.2f}".format(
                original_order_id, unfilled_quantity, limit_price
            )
        )

        # Submit order (API client handles retries)
        response = self._api_client.submit_order(payload)

        if not response.get("success", False):
            raise Exception(
                "Failed to resubmit order: {}".format(
                    response.get("error", "Unknown error")
                )
            )

        new_order_id = response.get("order_id")
        logger.info(
            "Order resubmitted: new_order_id={}, original_order_id={}".format(
                new_order_id, original_order_id
            )
        )

        return new_order_id

    def _calculate_elapsed_time(
        self, initial_time: Optional[datetime]
    ) -> float:
        """Calculate elapsed time since initial submission.

        Args:
            initial_time: Initial submission time (None returns 0)

        Returns:
            Elapsed time in seconds
        """
        if initial_time is None:
            return 0.0

        elapsed = (datetime.now(timezone.utc) - initial_time).total_seconds()
        return elapsed

    def _handle_timeout(
        self, original_order_id: str, unfilled_quantity: int
    ) -> None:
        """Handle timeout by canceling order and updating status.

        Args:
            original_order_id: Order ID to cancel
            unfilled_quantity: Unfilled quantity (for logging)

        Example:
            >>> handler = PartialFillHandler(api_client, tracker)
            >>> handler._handle_timeout("ORDER-123", 3)
            # Cancels order and updates status to UNFILLED_TIMEOUT
        """
        logger.warning(
            "Timeout reached for order: order_id={}, unfilled_qty={}".format(
                original_order_id, unfilled_quantity
            )
        )

        # Cancel original order
        self._api_client.cancel_order(original_order_id)

        # Update position status
        self._position_tracker.update_status(
            original_order_id, "UNFILLED_TIMEOUT"
        )

        logger.info(
            "Order cancelled due to timeout: order_id={}".format(
                original_order_id
            )
        )

    def handle_partial_fill(
        self,
        original_order_id: str,
        filled_quantity: int,
        target_quantity: int,
        current_price: float,
        direction: str,
    ) -> PartialFillResult:
        """Handle partial fill on limit order.

        Detects partial fills, calculates unfilled quantities, recalculates
        limit prices, resubmits orders, and handles timeouts.

        Args:
            original_order_id: Order ID that was partially filled
            filled_quantity: Quantity that has been filled
            target_quantity: Target quantity (contracts)
            current_price: Current market price
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            PartialFillResult with handling details

        Performance:
            Completes in < 100ms (excluding API submission time)

        Example:
            >>> handler = PartialFillHandler(api_client, tracker)
            >>> result = handler.handle_partial_fill(
            ...     original_order_id="ORDER-123",
            ...     filled_quantity=2,
            ...     target_quantity=5,
            ...     current_price=11800.00,
            ...     direction="bullish"
            ... )
            >>> result.status
            'PARTIALLY_FILLED'
            >>> result.unfilled_quantity
            3
        """
        start_time = time.perf_counter()

        # Get position to check cumulative time
        position = self._position_tracker.get_position(original_order_id)

        # Calculate cumulative fill time
        initial_time = None
        if position:
            # Handle both Position objects and mock dicts
            if isinstance(position, dict):
                initial_time = position.get("initial_submission_time")
            else:
                initial_time = position.initial_submission_time

        cumulative_time = self._calculate_elapsed_time(initial_time)

        # Calculate unfilled quantity
        unfilled_quantity = self.calculate_unfilled_quantity(
            target_quantity, filled_quantity
        )

        # Check if fully filled
        if unfilled_quantity == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update position status
            if position:
                self._position_tracker.update_status(
                    original_order_id, "FULLY_FILLED"
                )

            logger.info(
                "Order fully filled: order_id={}, qty={}".format(
                    original_order_id, filled_quantity
                )
            )

            return PartialFillResult(
                original_order_id=original_order_id,
                filled_quantity=filled_quantity,
                unfilled_quantity=0,
                new_order_id=None,
                status="FULLY_FILLED",
                cumulative_time_seconds=cumulative_time,
                handling_time_ms=elapsed_ms,
            )

        # Check for timeout
        if cumulative_time >= self._timeout_seconds:
            # Handle timeout
            self._handle_timeout(original_order_id, unfilled_quantity)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return PartialFillResult(
                original_order_id=original_order_id,
                filled_quantity=filled_quantity,
                unfilled_quantity=unfilled_quantity,
                new_order_id=None,
                status="UNFILLED_TIMEOUT",
                cumulative_time_seconds=cumulative_time,
                handling_time_ms=elapsed_ms,
            )

        # Recalculate limit price
        new_limit_price = self.recalculate_limit_price(current_price, direction)

        # Resubmit unfilled quantity
        new_order_id = self.resubmit_unfilled_quantity(
            original_order_id=original_order_id,
            unfilled_quantity=unfilled_quantity,
            direction=direction,
            limit_price=new_limit_price,
        )

        # Update position tracking
        if position:
            self._position_tracker.update_fill(
                order_id=original_order_id,
                filled_quantity=filled_quantity,
                new_order_id=new_order_id,
            )
            self._position_tracker.update_status(
                original_order_id, "PARTIALLY_FILLED"
            )

        # Calculate performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Log to audit trail if path provided
        if self._audit_trail_path:
            self.log_partial_fill_event(
                original_order_id=original_order_id,
                filled_quantity=filled_quantity,
                unfilled_quantity=unfilled_quantity,
                new_order_id=new_order_id,
                cumulative_time=cumulative_time,
            )

        logger.info(
            "Partial fill handled: original_order={}, filled={}, unfilled={}, "
            "new_order_id={}, time={:.2f}ms".format(
                original_order_id,
                filled_quantity,
                unfilled_quantity,
                new_order_id,
                elapsed_ms,
            )
        )

        return PartialFillResult(
            original_order_id=original_order_id,
            filled_quantity=filled_quantity,
            unfilled_quantity=unfilled_quantity,
            new_order_id=new_order_id,
            status="PARTIALLY_FILLED",
            cumulative_time_seconds=cumulative_time,
            handling_time_ms=elapsed_ms,
        )

    def log_partial_fill_event(
        self,
        original_order_id: str,
        filled_quantity: int,
        unfilled_quantity: int,
        new_order_id: str,
        cumulative_time: float,
    ) -> None:
        """Log partial fill event to CSV audit trail.

        Args:
            original_order_id: Original order ID
            filled_quantity: Quantity filled
            unfilled_quantity: Quantity unfilled
            new_order_id: New order ID for unfilled quantity
            cumulative_time: Cumulative fill time in seconds

        Example:
            >>> handler = PartialFillHandler(api_client, tracker, "audit.csv")
            >>> handler.log_partial_fill_event(
            ...     original_order_id="ORDER-123",
            ...     filled_quantity=2,
            ...     unfilled_quantity=3,
            ...     new_order_id="ORDER-789",
            ...     cumulative_time=30.5
            ... )
            # Logs to CSV audit trail
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
                writer.writerow(
                    [
                        "timestamp",
                        "event_type",
                        "original_order_id",
                        "filled_quantity",
                        "unfilled_quantity",
                        "new_order_id",
                        "cumulative_time_seconds",
                    ]
                )

            # Write log entry
            writer.writerow(
                [
                    timestamp,
                    "PARTIAL_FILL",
                    original_order_id,
                    filled_quantity,
                    unfilled_quantity,
                    new_order_id,
                    "{:.2f}".format(cumulative_time),
                ]
            )

        logger.debug("Partial fill audit trail updated: {}".format(audit_path))

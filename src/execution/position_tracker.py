"""Position tracking for partial fill handling.

This module tracks active positions with state management for partial fills,
including filled/unfilled quantities, cumulative fill time, and resubmission
tracking.

Features:
- Position state management (order_id, signal_id, entry_price, quantity)
- Partial fill tracking (filled_quantity, unfilled_quantity)
- Cumulative fill time tracking
- Order resubmission tracking (new_order_id)
- Status tracking (PENDING, PARTIALLY_FILLED, FULLY_FILLED, UNFILLED_TIMEOUT)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Active position tracking for partial fill handling.

    Attributes:
        order_id: TradeStation order ID
        signal_id: Signal identifier that triggered this position
        entry_price: Initial entry/limit price in dollars
        quantity: Target quantity (contracts)
        direction: Signal direction ("bullish" or "bearish")
        order_type: Order type ("MARKET" or "LIMIT")
        timestamp: Position creation timestamp
        filled_quantity: Quantity filled so far (defaults to quantity)
        unfilled_quantity: Quantity yet to be filled (defaults to 0)
        new_order_id: Resubmitted order ID for unfilled quantity
        status: Position status
        initial_submission_time: First submission time for timeout tracking
        cumulative_fill_time_seconds: Total time from first submission to fill
    """

    order_id: str
    signal_id: str
    entry_price: float
    quantity: int
    direction: str
    order_type: str
    timestamp: datetime
    filled_quantity: int = field(init=False)
    unfilled_quantity: int = field(init=False)
    new_order_id: Optional[str] = None
    status: str = "PENDING"
    initial_submission_time: Optional[datetime] = None
    cumulative_fill_time_seconds: float = 0.0

    def __post_init__(self):
        """Initialize derived fields after creation."""
        # Initially assume full fill
        self.filled_quantity = self.quantity
        self.unfilled_quantity = 0
        # Set initial submission time if not provided
        if self.initial_submission_time is None:
            self.initial_submission_time = self.timestamp


class PositionTracker:
    """Tracks active positions with partial fill state.

    Maintains in-memory storage of active positions with methods to
    add, update, and retrieve position state for partial fill handling.

    Attributes:
        _positions: Dictionary mapping order_id to Position

    Example:
        >>> tracker = PositionTracker()
        >>> position = Position(
        ...     order_id="ORDER-123",
        ...     signal_id="SIG-456",
        ...     entry_price=11800.0,
        ...     quantity=5,
        ...     direction="bullish",
        ...     order_type="LIMIT",
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        >>> tracker.add_position(position)
        >>> tracker.update_fill("ORDER-123", filled_quantity=2)
        >>> retrieved = tracker.get_position("ORDER-123")
        >>> retrieved.filled_quantity
        2
        >>> retrieved.unfilled_quantity
        3
    """

    def __init__(self) -> None:
        """Initialize position tracker with empty storage."""
        self._positions: Dict[str, Position] = {}
        logger.info("PositionTracker initialized")

    def add_position(self, position: Position) -> None:
        """Add a new position to tracking.

        Args:
            position: Position object to track

        Raises:
            ValueError: If position with order_id already exists

        Example:
            >>> tracker = PositionTracker()
            >>> position = Position(
            ...     order_id="ORDER-123",
            ...     signal_id="SIG-456",
            ...     entry_price=11800.0,
            ...     quantity=5,
            ...     direction="bullish",
            ...     order_type="LIMIT",
            ...     timestamp=datetime.now(timezone.utc)
            ... )
            >>> tracker.add_position(position)
        """
        if position.order_id in self._positions:
            raise ValueError(
                "Position with order_id {} already exists".format(
                    position.order_id
                )
            )

        self._positions[position.order_id] = position
        logger.debug(
            "Position added: order_id={}, signal_id={}, qty={}".format(
                position.order_id, position.signal_id, position.quantity
            )
        )

    def get_position(self, order_id: str) -> Optional[Position]:
        """Retrieve position by order ID.

        Args:
            order_id: TradeStation order ID

        Returns:
            Position object if found, None otherwise

        Example:
            >>> tracker = PositionTracker()
            >>> position = tracker.get_position("ORDER-123")
            >>> if position:
            ...     print(position.filled_quantity)
        """
        return self._positions.get(order_id)

    def update_fill(
        self,
        order_id: str,
        filled_quantity: int,
        new_order_id: Optional[str] = None,
    ) -> None:
        """Update position with partial fill information.

        Args:
            order_id: TradeStation order ID to update
            filled_quantity: Quantity that has been filled
            new_order_id: New order ID for unfilled quantity (if resubmitted)

        Raises:
            ValueError: If order_id not found or filled_quantity invalid

        Example:
            >>> tracker = PositionTracker()
            >>> tracker.update_fill(
            ...     order_id="ORDER-123",
            ...     filled_quantity=2,
            ...     new_order_id="ORDER-789"
            ... )
        """
        position = self._positions.get(order_id)
        if position is None:
            raise ValueError(
                "Position not found: {}".format(order_id)
            )

        if filled_quantity < 0:
            raise ValueError(
                "filled_quantity must be non-negative, got {}".format(
                    filled_quantity
                )
            )

        if filled_quantity > position.quantity:
            raise ValueError(
                "filled_quantity {} exceeds quantity {}".format(
                    filled_quantity, position.quantity
                )
            )

        # Update filled/unfilled quantities
        position.filled_quantity = filled_quantity
        position.unfilled_quantity = position.quantity - filled_quantity

        # Update new order ID if provided
        if new_order_id is not None:
            position.new_order_id = new_order_id

        logger.debug(
            "Position updated: order_id={}, filled={}, unfilled={}".format(
                order_id, filled_quantity, position.unfilled_quantity
            )
        )

    def update_status(self, order_id: str, status: str) -> None:
        """Update position status.

        Args:
            order_id: TradeStation order ID to update
            status: New status (PENDING, PARTIALLY_FILLED, FULLY_FILLED,
                UNFILLED_TIMEOUT)

        Raises:
            ValueError: If order_id not found

        Example:
            >>> tracker = PositionTracker()
            >>> tracker.update_status("ORDER-123", "PARTIALLY_FILLED")
        """
        position = self._positions.get(order_id)
        if position is None:
            raise ValueError(
                "Position not found: {}".format(order_id)
            )

        position.status = status
        logger.debug(
            "Position status updated: order_id={}, status={}".format(
                order_id, status
            )
        )

"""Maximum open position size limit enforcement.

This module implements a maximum open position size limit that caps
the total number of contracts that can be held simultaneously. This prevents
over-leveraging and ensures capital is not concentrated in too few positions.

Features:
- Track total open position size
- Enforce maximum position size limit
- Calculate available capacity
- CSV audit trail logging
- Integration with position tracker
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PositionSizeTracker:
    """Track total open position size and enforce maximum limit.

    Attributes:
        _max_position_size: Maximum number of contracts allowed open
        _current_position_size: Total contracts currently open
        _position_counts: Dictionary of order_id → contract count
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> tracker = PositionSizeTracker(
        ...     max_position_size=20
        ... )
        >>> tracker.add_position("ORDER-1", 5)
        >>> tracker.get_available_capacity()
        15
        >>> tracker.can_open_position(10)
        True
        >>> tracker.can_open_position(20)
        False  # Would exceed limit
    """

    def __init__(
        self,
        max_position_size: int,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize position size tracker.

        Args:
            max_position_size: Maximum contracts allowed open simultaneously
            audit_trail_path: Path to CSV audit trail file (optional)

        Raises:
            ValueError: If max_position_size <= 0

        Example:
            >>> tracker = PositionSizeTracker(max_position_size=20)
        """
        if max_position_size <= 0:
            raise ValueError(
                "Max position size must be positive: {}".format(
                    max_position_size
                )
            )

        self._max_position_size = max_position_size
        self._current_position_size = 0
        self._position_counts = {}
        self._audit_trail_path = audit_trail_path

        logger.info(
            "PositionSizeTracker initialized: max={} contracts".format(
                max_position_size
            )
        )

    def add_position(
        self,
        order_id: str,
        quantity: int
    ) -> None:
        """Add a position to tracking.

        Args:
            order_id: Order ID
            quantity: Number of contracts

        Example:
            >>> tracker.add_position("ORDER-123", 5)
            >>> tracker.get_current_position_size()
            5
        """
        if quantity <= 0:
            raise ValueError(
                "Quantity must be positive: {}".format(quantity)
            )

        # Check if order already exists (replace)
        if order_id in self._position_counts:
            old_quantity = self._position_counts[order_id]

            # Update current size (remove old, add new)
            self._current_position_size = (
                self._current_position_size - old_quantity + quantity
            )

            logger.debug(
                "Position replaced: {} ({} → {} contracts)".format(
                    order_id, old_quantity, quantity
                )
            )
        else:
            # Add to current size
            self._current_position_size += quantity

            logger.debug(
                "Position added: {} ({} contracts)".format(
                    order_id, quantity
                )
            )

        # Store position count
        self._position_counts[order_id] = quantity

        # Log to audit trail
        self._log_audit_event("ADD", order_id, quantity)

    def remove_position(
        self,
        order_id: str
    ) -> None:
        """Remove a position from tracking (position closed).

        Args:
            order_id: Order ID to remove

        Example:
            >>> tracker.remove_position("ORDER-123")
            >>> tracker.get_current_position_size()
            0
        """
        if order_id not in self._position_counts:
            # Position doesn't exist, nothing to remove
            logger.debug(
                "Attempted to remove non-existent position: {}".format(
                    order_id
                )
            )
            return

        # Get quantity before removal
        quantity = self._position_counts[order_id]

        # Remove from current size
        self._current_position_size -= quantity

        # Remove from tracking
        del self._position_counts[order_id]

        logger.debug(
            "Position removed: {} ({} contracts)".format(
                order_id, quantity
            )
        )

        # Log to audit trail
        self._log_audit_event("REMOVE", order_id, quantity)

    def can_open_position(
        self,
        quantity: int
    ) -> bool:
        """Check if position can be opened without exceeding limit.

        Args:
            quantity: Number of contracts to open

        Returns:
            True if within limit, False otherwise

        Example:
            >>> if not tracker.can_open_position(10):
            ...     return "Maximum position size reached"
        """
        if quantity <= 0:
            raise ValueError(
                "Quantity must be positive: {}".format(quantity)
            )

        new_size = self._current_position_size + quantity

        return new_size <= self._max_position_size

    def get_available_capacity(self) -> int:
        """Get remaining capacity for new positions.

        Returns:
            Number of contracts that can still be opened

        Example:
            >>> capacity = tracker.get_available_capacity()
            >>> print(f"Can open {capacity} more contracts")
        """
        return self._max_position_size - self._current_position_size

    def get_current_position_size(self) -> int:
        """Get current total open position size.

        Returns:
            Total contracts currently open

        Example:
            >>> current = tracker.get_current_position_size()
            >>> print(f"Currently holding {current} contracts")
        """
        return self._current_position_size

    def get_position_count(self) -> int:
        """Get number of distinct positions.

        Returns:
            Number of unique order IDs with positions

        Example:
            >>> count = tracker.get_position_count()
            >>> print(f"Holding {count} positions")
        """
        return len(self._position_counts)

    def is_at_capacity(self) -> bool:
        """Check if at maximum position size.

        Returns:
            True if at limit, False otherwise

        Example:
            >>> if tracker.is_at_capacity():
            ...     logger.warning("At maximum position size")
        """
        return self._current_position_size >= self._max_position_size

    def get_position_summary(self) -> dict:
        """Get position size summary.

        Returns:
            Dictionary with position statistics

        Example:
            >>> summary = tracker.get_position_summary()
            >>> print(f"Current: {summary['current_position_size']}")
            >>> print(f"Available: {summary['available_capacity']}")
        """
        return {
            "max_position_size": self._max_position_size,
            "current_position_size": self._current_position_size,
            "available_capacity": self.get_available_capacity(),
            "position_count": self.get_position_count(),
            "is_at_capacity": self.is_at_capacity()
        }

    def _log_audit_event(
        self,
        event_type: str,
        order_id: str,
        quantity: int
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (ADD, REMOVE, CHECK)
            order_id: Order ID
            quantity: Number of contracts
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if file exists
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
                    "quantity",
                    "current_position_size",
                    "available_capacity",
                    "is_at_capacity"
                ])

            # Write event
            writer.writerow([
                timestamp,
                event_type,
                order_id,
                quantity,
                self._current_position_size,
                self.get_available_capacity(),
                self.is_at_capacity()
            ])

        logger.debug("Position size audit event logged: {}".format(event_type))

"""Maximum drawdown limit enforcement for trading safety.

This module implements a maximum drawdown limit that stops trading
when the account experiences a significant decline from its peak value.
Drawdown measures peak-to-trough decline over multi-day periods.

Features:
- Track peak account value
- Calculate drawdown percentage
- Enforce maximum drawdown limit (default: 10%)
- Recovery threshold (hysteresis) to prevent oscillation
- CSV audit trail logging
- Alert on limit breach
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_current_time() -> datetime:
    """Get current time in UTC.

    Returns:
        Current datetime in UTC

    Note:
        Separate function to facilitate testing with mocks.
    """
    return datetime.now(timezone.utc)


class DrawdownTracker:
    """Track account drawdown from peak and enforce maximum drawdown limit.

    Attributes:
        _max_drawdown_percentage: Maximum allowed drawdown (e.g., 0.10 = 10%)
        _recovery_threshold_percentage: Recovery threshold (e.g., 0.95 = 95%)
        _peak_value: Highest account value observed
        _current_value: Current account value
        _is_halted: Whether trading is halted due to drawdown
        _halt_reason: Reason trading was halted
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> tracker = DrawdownTracker(
        ...     max_drawdown_percentage=0.10,  # 10%
        ...     recovery_threshold_percentage=0.95,  # 95%
        ...     initial_value=50000.00
        ... )
        >>> tracker.update_value(52000.00)  # New peak
        >>> tracker.update_value(48000.00)  # Drawdown
        >>> tracker.get_drawdown_percentage()
        0.0769  # 7.69% drawdown
        >>> tracker.is_trading_allowed()
        True  # Still under 10% limit
    """

    def __init__(
        self,
        max_drawdown_percentage: float,
        recovery_threshold_percentage: float,
        initial_value: float,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize drawdown tracker.

        Args:
            max_drawdown_percentage: Maximum allowed drawdown (0.10 = 10%)
            recovery_threshold_percentage: Recovery threshold (0.95 = 95%)
            initial_value: Initial account value
            audit_trail_path: Path to CSV audit trail file (optional)

        Raises:
            ValueError: If percentages not in valid range or initial_value <= 0

        Example:
            >>> tracker = DrawdownTracker(
            ...     max_drawdown_percentage=0.10,
            ...     recovery_threshold_percentage=0.95,
            ...     initial_value=50000.00
            ... )
        """
        if not (0.0 < max_drawdown_percentage < 1.0):
            raise ValueError(
                "Max drawdown percentage must be between 0 and 1: {}".format(
                    max_drawdown_percentage
                )
            )

        if not (0.0 < recovery_threshold_percentage <= 1.0):
            raise ValueError(
                "Recovery threshold percentage must be "
                "between 0 and 1: {}".format(
                    recovery_threshold_percentage
                )
            )

        if initial_value <= 0:
            raise ValueError(
                "Initial value must be positive: {}".format(initial_value)
            )

        self._max_drawdown_percentage = max_drawdown_percentage
        self._recovery_threshold_percentage = recovery_threshold_percentage
        self._peak_value = initial_value
        self._current_value = initial_value
        self._is_halted = False
        self._halt_reason: Optional[str] = None
        self._audit_trail_path = audit_trail_path

        logger.info(
            "DrawdownTracker initialized: max_dd={:.1%}, recovery={:.1%}, "
            "initial={:.2f}".format(
                max_drawdown_percentage,
                recovery_threshold_percentage,
                initial_value
            )
        )

    def update_value(
        self,
        current_value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update account value and check drawdown limits.

        Args:
            current_value: Current account value
            timestamp: Timestamp of update (optional, defaults to now)

        Example:
            >>> tracker.update_value(52000.00)  # New peak established
            >>> tracker.update_value(49000.00)  # Drawdown
            >>> if tracker.is_trading_halted():
            ...     print("Drawdown limit exceeded")
        """
        if current_value <= 0:
            raise ValueError(
                "Account value must be positive: {}".format(current_value)
            )

        # Update current value
        self._current_value = current_value

        # Update peak if new high
        if current_value > self._peak_value:
            old_peak = self._peak_value
            self._peak_value = current_value

            logger.debug(
                "New peak established: ${:.2f} → ${:.2f}".format(
                    old_peak, current_value
                )
            )

            # Log peak event
            self._log_audit_event("PEAK", timestamp)

        # Calculate drawdown
        drawdown_pct = self.get_drawdown_percentage()

        # Check if previously halted and now recovered
        if self._is_halted:
            recovery_value = self.get_recovery_value()
            if current_value >= recovery_value:
                # Recovered enough to resume trading
                self._is_halted = False
                self._halt_reason = None

                logger.info(
                    "Trading resumed: recovered to {} ({:.2f} needed)".format(
                        current_value, recovery_value
                    )
                )
            else:
                # Still below recovery threshold
                pass
        else:
            # Not halted, check if should halt
            if drawdown_pct >= self._max_drawdown_percentage:
                self._is_halted = True
                self._halt_reason = (
                    "Maximum drawdown limit breached: {:.2%} / {:.1%}".format(
                        drawdown_pct,
                        self._max_drawdown_percentage
                    )
                )

                logger.critical(
                    "MAXIMUM DRAWDOWN LIMIT BREACHED: {}".format(
                        self._halt_reason
                    )
                )

                # Log halt event
                self._log_audit_event("HALT", timestamp)
            else:
                # Still within limits
                pass

        # Log update event
        self._log_audit_event("UPDATE", timestamp)

    def get_drawdown_percentage(self) -> float:
        """Get current drawdown percentage.

        Returns:
            Drawdown percentage (0.0 to 1.0)

        Example:
            >>> dd = tracker.get_drawdown_percentage()
            >>> print(f"Drawdown: {dd * 100:.2f}%")
        """
        if self._peak_value == 0:
            return 0.0

        drawdown = (self._peak_value - self._current_value) / self._peak_value
        return max(0.0, drawdown)

    def get_peak_value(self) -> float:
        """Get current peak account value.

        Returns:
            Highest account value observed

        Example:
            >>> peak = tracker.get_peak_value()
            >>> print(f"Peak: ${peak:.2f}")
        """
        return self._peak_value

    def get_current_value(self) -> float:
        """Get current account value.

        Returns:
            Current account value

        Example:
            >>> current = tracker.get_current_value()
            >>> print(f"Current: ${current:.2f}")
        """
        return self._current_value

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed.

        Returns:
            True if trading allowed, False if halted

        Example:
            >>> if not tracker.is_trading_allowed():
            ...     return "Maximum drawdown limit exceeded"
        """
        return not self._is_halted

    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted.

        Returns:
            True if halted due to drawdown, False otherwise

        Example:
            >>> if tracker.is_trading_halted():
            ...     logger.warning("Trading halted due to drawdown")
        """
        return self._is_halted

    def get_halt_reason(self) -> Optional[str]:
        """Get reason trading was halted.

        Returns:
            Halt reason or None if not halted

        Example:
            >>> reason = tracker.get_halt_reason()
            >>> print(f"Halted: {reason}")
        """
        return self._halt_reason

    def get_recovery_value(self) -> float:
        """Get value needed to resume trading.

        Returns:
            Account value needed to resume trading

        Example:
            >>> recovery_value = tracker.get_recovery_value()
            >>> current = tracker.get_current_value()
            >>> print(f"Need ${recovery_value - current:.2f} more to resume")
        """
        return self._peak_value * self._recovery_threshold_percentage

    def get_drawdown_summary(self) -> dict:
        """Get drawdown statistics summary.

        Returns:
            Dictionary with drawdown statistics

        Example:
            >>> summary = tracker.get_drawdown_summary()
            >>> print(f"Peak: ${summary['peak_value']}")
            >>> print(f"Current: ${summary['current_value']}")
            >>> print(f"Drawdown: {summary['drawdown_percentage'] * 100:.2f}%")
        """
        return {
            "peak_value": self._peak_value,
            "current_value": self._current_value,
            "drawdown_percentage": self.get_drawdown_percentage(),
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "recovery_value": self.get_recovery_value(),
            "max_drawdown_percentage": self._max_drawdown_percentage,
            "recovery_threshold_percentage": self._recovery_threshold_percentage
        }

    def _log_audit_event(
        self,
        event_type: str,
        timestamp: Optional[datetime]
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (UPDATE, HALT, RECOVER, PEAK)
            timestamp: Timestamp of event (optional)
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        if timestamp is None:
            timestamp = _get_current_time()

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
                    "peak_value",
                    "current_value",
                    "drawdown_percentage",
                    "is_halted",
                    "recovery_value"
                ])

            # Write event
            writer.writerow([
                timestamp.isoformat(),
                event_type,
                f"{self._peak_value:.2f}",
                f"{self._current_value:.2f}",
                f"{self.get_drawdown_percentage():.4f}",
                self._is_halted,
                f"{self.get_recovery_value():.2f}"
            ])

        logger.debug("Drawdown audit event logged: {}".format(event_type))

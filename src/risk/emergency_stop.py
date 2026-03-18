"""Emergency stop button for immediate trading halt.

This module implements an emergency stop button that immediately halts
all trading activity and prevents new orders from being submitted.
This is a critical safety mechanism for manually intervening when
the system behaves unexpectedly or during extreme market conditions.

Features:
- Immediate trading halt (activate/deactivate)
- State persistence (survives restarts)
- CSV audit trail logging
- CLI interface for manual operation
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EmergencyStop:
    """Emergency stop button for immediate trading halt.

    Attributes:
        _is_stopped: Whether emergency stop is active
        _stop_time: When emergency stop was activated
        _stop_reason: Reason for emergency stop
        _audit_trail_path: Path to CSV audit trail file
        _state_path: Path to JSON state file

    Example:
        >>> stop = EmergencyStop()
        >>> stop.activate("Manual: System behaving erratically")
        >>> stop.is_trading_allowed()
        False
        >>> stop.deactivate()
        >>> stop.is_trading_allowed()
        True
    """

    def __init__(
        self,
        audit_trail_path: Optional[str] = None,
        state_path: Optional[str] = None
    ) -> None:
        """Initialize emergency stop button.

        Args:
            audit_trail_path: Path to CSV audit trail file (optional)
            state_path: Path to JSON state file (optional)

        Example:
            >>> stop = EmergencyStop()
        """
        self._is_stopped = False
        self._stop_time = None
        self._stop_reason = None
        self._audit_trail_path = audit_trail_path
        self._state_path = state_path

        # Load persisted state if available
        if self._state_path is not None:
            self._load_state()

        logger.info("EmergencyStop initialized")

    def activate(self, reason: str) -> None:
        """Activate emergency stop.

        Args:
            reason: Reason for activating emergency stop

        Example:
            >>> stop.activate("Manual: Excessive order submission detected")
        """
        if self._is_stopped:
            logger.warning("Emergency stop already active")
            return

        self._is_stopped = True
        self._stop_time = self._get_current_time()
        self._stop_reason = reason

        logger.error(
            "EMERGENCY STOP ACTIVATED: {}".format(reason)
        )

        # Save state
        if self._state_path is not None:
            self._save_state()

        # Log event
        self._log_audit_event("ACTIVATE")

    def deactivate(self) -> None:
        """Deactivate emergency stop and resume trading.

        Example:
            >>> stop.deactivate()
        """
        if not self._is_stopped:
            logger.warning("Emergency stop not active")
            return

        self._is_stopped = False
        stop_reason = self._stop_reason
        self._stop_time = None
        self._stop_reason = None

        logger.info(
            "Emergency stop deactivated (was: {})".format(
                stop_reason
            )
        )

        # Save state
        if self._state_path is not None:
            self._save_state()

        # Log event
        self._log_audit_event("DEACTIVATE")

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed.

        Returns:
            True if trading allowed, False if emergency stop active

        Example:
            >>> if not stop.is_trading_allowed():
            ...     logger.warning("Trading blocked: Emergency stop active")
        """
        # Log check
        self._log_audit_event("CHECK")

        return not self._is_stopped

    def get_status(self) -> dict:
        """Get current emergency stop status.

        Returns:
            Dictionary with status information:
            - is_stopped: Whether emergency stop is active
            - stop_time: When emergency stop was activated
            - stop_reason: Reason for emergency stop
            - time_stopped_seconds: Seconds since stop activated

        Example:
            >>> status = stop.get_status()
            >>> if status['is_stopped']:
            ...     print("Stopped for: {} seconds".format(
            ...         status['time_stopped_seconds']
            ...     ))
        """
        time_stopped_seconds = None
        if self._is_stopped and self._stop_time is not None:
            current_time = self._get_current_time()
            time_stopped_seconds = int(
                (current_time - self._stop_time).total_seconds()
            )

        return {
            "is_stopped": self._is_stopped,
            "stop_time": self._stop_time.isoformat() if self._stop_time else None,
            "stop_reason": self._stop_reason,
            "time_stopped_seconds": time_stopped_seconds
        }

    def _get_current_time(self) -> datetime:
        """Get current time (UTC).

        Returns:
            Current datetime in UTC

        Example:
            >>> now = stop._get_current_time()
        """
        return datetime.now(timezone.utc)

    def _save_state(self) -> None:
        """Save emergency stop state to file.

        Example:
            >>> stop._save_state()
        """
        if self._state_path is None:
            return

        state = {
            "is_stopped": self._is_stopped,
            "stop_time": (
                self._stop_time.isoformat() if self._stop_time else None
            ),
            "stop_reason": self._stop_reason,
            "last_updated": self._get_current_time().isoformat()
        }

        state_path = Path(self._state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.debug("Emergency stop state saved")

    def _load_state(self) -> None:
        """Load emergency stop state from file.

        Example:
            >>> stop._load_state()
        """
        if self._state_path is None:
            return

        state_path = Path(self._state_path)

        if not state_path.exists():
            logger.debug("No existing state file found")
            return

        try:
            with open(state_path, "r") as f:
                state = json.load(f)

            self._is_stopped = state.get("is_stopped", False)
            self._stop_reason = state.get("stop_reason")

            stop_time_str = state.get("stop_time")
            if stop_time_str:
                self._stop_time = datetime.fromisoformat(stop_time_str)
            else:
                self._stop_time = None

            logger.info(
                "Emergency stop state loaded: is_stopped={}".format(
                    self._is_stopped
                )
            )

        except Exception as e:
            logger.error(
                "Failed to load emergency stop state: {}".format(str(e))
            )

    def _log_audit_event(self, event_type: str) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (ACTIVATE, DEACTIVATE, CHECK)
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = self._get_current_time().isoformat()

        # Check if file exists
        file_exists = (
            audit_path.exists() and
            audit_path.stat().st_size > 0
        )

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "is_stopped",
                    "stop_reason",
                    "time_stopped_seconds"
                ])

            # Calculate time stopped
            time_stopped_seconds = None
            if self._is_stopped and self._stop_time is not None:
                current_time = self._get_current_time()
                time_stopped_seconds = int(
                    (current_time - self._stop_time).total_seconds()
                )

            # Write event
            writer.writerow([
                timestamp,
                event_type,
                self._is_stopped,
                self._stop_reason or "",
                time_stopped_seconds or ""
            ])

        logger.debug("Emergency stop audit logged: {}".format(event_type))

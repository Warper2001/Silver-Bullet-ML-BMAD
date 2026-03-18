"""Circuit breaker event detection for extreme market volatility.

This module implements circuit breaker detection to automatically halt
trading during extreme market volatility. Circuit breakers are triggered
by exchanges when prices move too rapidly, indicating market stress.

Features:
- Exchange API status checking
- Backup detection from price movement
- Halt period tracking and expiration
- Multi-level circuit breaker support (Level 1, 2, 3)
- CSV audit trail logging
- Integration with trade execution pipeline
"""

import csv
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_current_time() -> datetime:
    """Get current time in UTC.

    Note:
        Separate function to facilitate testing with mocks.

    Returns:
        Current datetime in UTC
    """
    return datetime.now(timezone.utc)


class CircuitBreakerDetector:
    """Detect circuit breaker events from exchange and market data.

    Attributes:
        _api_client: TradeStation API client
        _halted: Whether trading is currently halted
        _halt_level: Current circuit breaker level (1, 2, 3, or None)
        _halt_reason: Reason for halt
        _halt_start_time: When halt began
        _estimated_resume_time: When trading expected to resume
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> detector = CircuitBreakerDetector(api_client=api)
        >>> detector.check_circuit_breaker_status()
        >>> if detector.is_trading_halted():
        ...     print("Circuit breaker active: Level {}".format(
        ...         detector.get_halt_level()
        ...     ))
    """

    # Circuit breaker thresholds (CME Group)
    LEVEL_1_THRESHOLD = 0.07  # 7% decline
    LEVEL_2_THRESHOLD = 0.13  # 13% decline
    LEVEL_3_THRESHOLD = 0.20  # 20% decline

    def __init__(
        self,
        api_client,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize circuit breaker detector.

        Args:
            api_client: TradeStation API client
            audit_trail_path: Path to CSV audit trail file (optional)

        Example:
            >>> detector = CircuitBreakerDetector(api_client=api)
        """
        self._api_client = api_client
        self._halted = False
        self._halt_level = None
        self._halt_reason = None
        self._halt_start_time = None
        self._estimated_resume_time = None
        self._audit_trail_path = audit_trail_path

        logger.info("CircuitBreakerDetector initialized")

    def check_circuit_breaker_status(self) -> dict:
        """Check current circuit breaker status from exchange.

        Returns:
            Dictionary with status information:
            - is_halted: Whether trading is halted
            - halt_level: Circuit breaker level (1, 2, 3, or None)
            - reason: Halt reason
            - start_time: When halt began
            - estimated_resume: When trading resumes

        Example:
            >>> status = detector.check_circuit_breaker_status()
            >>> if status['is_halted']:
            ...     logger.warning(
            ...         "Circuit breaker Level {} active".format(
            ...             status['halt_level']
            ...         )
            ...     )
        """
        # Try primary detection method (exchange API)
        try:
            exchange_status = self._check_exchange_status()
        except Exception as e:
            logger.error(
                "Exchange API check failed: {}".format(str(e))
            )
            # Return safe default (not halted)
            exchange_status = {
                "halted": False,
                "level": None,
                "reason": None,
                "start_time": None,
                "estimated_resume": None
            }

        # Update internal state
        self._update_halt_status(exchange_status)

        # Check if halt expired
        if self._halted and self._check_if_halt_expired():
            logger.info("Circuit breaker halt period expired")
            self._halted = False
            self._halt_level = None
            self._halt_reason = None

        # Log check event
        self._log_audit_event(
            "CHECK",
            self._halt_level,
            self._halted
        )

        # Return status
        return {
            "is_halted": self._halted,
            "halt_level": self._halt_level,
            "reason": self._halt_reason,
            "start_time": self._halt_start_time,
            "estimated_resume": self._estimated_resume_time
        }

    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted due to circuit breaker.

        Returns:
            True if halted, False otherwise

        Example:
            >>> if detector.is_trading_halted():
            ...     return "Circuit breaker halt active"
        """
        return self._halted

    def get_halt_level(self) -> Optional[int]:
        """Get current circuit breaker halt level.

        Returns:
            Halt level (1, 2, 3) or None if not halted

        Example:
            >>> level = detector.get_halt_level()
            >>> if level == 3:
            ...     print("Level 3 circuit breaker - rest of day")
        """
        return self._halt_level

    def get_estimated_resume_time(self) -> Optional[datetime]:
        """Get estimated time when trading will resume.

        Returns:
            Resume time (UTC) or None if not halted

        Example:
            >>> resume_time = detector.get_estimated_resume_time()
            >>> print("Trading resumes at {}".format(resume_time))
        """
        return self._estimated_resume_time

    def get_halt_duration_remaining(self) -> Optional[timedelta]:
        """Get remaining time in current halt.

        Returns:
            Time remaining until resume (timedelta) or None

        Example:
            >>> remaining = detector.get_halt_duration_remaining()
            >>> print("{} minutes remaining".format(
            ...     remaining.seconds // 60
            ... ))
        """
        if not self._halted or self._estimated_resume_time is None:
            return None

        current_time = _get_current_time()
        remaining = self._estimated_resume_time - current_time

        # Return zero if already expired
        if remaining.total_seconds() < 0:
            return timedelta(seconds=0)

        return remaining

    def can_resume_trading(self) -> bool:
        """Check if trading can resume (halt period over).

        Returns:
            True if trading can resume, False otherwise

        Example:
            >>> if detector.can_resume_trading():
            ...     logger.info("Circuit breaker period ended")
        """
        if not self._halted:
            return True

        return self._check_if_halt_expired()

    def _check_exchange_status(self) -> dict:
        """Check exchange circuit breaker status via API.

        Returns:
            Dictionary with exchange status

        Note:
            This is the primary detection method. API provides
            real-time circuit breaker notifications.

        Exchange API Endpoints:
        - GET /market/circuit_breaker/status
        - Returns: { halted, level, start_time, estimated_resume }
        """
        # In production, this would call the TradeStation API
        # For now, return default (not halted)
        # TODO: Implement actual API call when endpoint available

        return {
            "halted": False,
            "level": None,
            "reason": None,
            "start_time": None,
            "estimated_resume": None
        }

    def _detect_from_price_movement(
        self,
        current_price: float,
        previous_close: float
    ) -> Optional[dict]:
        """Detect circuit breaker from rapid price movement.

        Args:
            current_price: Current MNQ price
            previous_close: Previous day's close price

        Returns:
            Detection result or None if no breaker triggered

        Note:
            This is a backup detection method if API unavailable.
            Calculates % decline from previous close.

        Calculation:
        Level 1: decline >= 7%
        Level 2: decline >= 13%
        Level 3: decline >= 20%
        """
        # Calculate decline percentage
        decline_pct = (previous_close - current_price) / previous_close

        # Check Level 3 (most severe)
        if decline_pct >= self.LEVEL_3_THRESHOLD:
            return {
                "halted": True,
                "level": 3,
                "reason": "Level 3 circuit breaker ({}% decline)".format(
                    round(decline_pct * 100, 1)
                ),
                "start_time": _get_current_time(),
                "estimated_resume": _get_current_time() + timedelta(hours=6)
            }

        # Check Level 2
        if decline_pct >= self.LEVEL_2_THRESHOLD:
            return {
                "halted": True,
                "level": 2,
                "reason": "Level 2 circuit breaker ({}% decline)".format(
                    round(decline_pct * 100, 1)
                ),
                "start_time": _get_current_time(),
                "estimated_resume": _get_current_time() + timedelta(minutes=15)
            }

        # Check Level 1
        if decline_pct >= self.LEVEL_1_THRESHOLD:
            return {
                "halted": True,
                "level": 1,
                "reason": "Level 1 circuit breaker ({}% decline)".format(
                    round(decline_pct * 100, 1)
                ),
                "start_time": _get_current_time(),
                "estimated_resume": _get_current_time() + timedelta(minutes=15)
            }

        # No breaker triggered
        return None

    def _check_if_halt_expired(self) -> bool:
        """Check if halt period has expired.

        Returns:
            True if halt expired, False otherwise

        Example:
            >>> if detector._check_if_halt_expired():
            ...     self._halted = False
            ...     self._halt_level = None
        """
        if not self._halted or self._estimated_resume_time is None:
            return False

        current_time = _get_current_time()
        return current_time >= self._estimated_resume_time

    def _update_halt_status(self, status: dict) -> None:
        """Update internal halt status from exchange API.

        Args:
            status: Status from exchange API

        Updates:
            - _halted: Trading halted flag
            - _halt_level: Circuit breaker level
            - _halt_reason: Human-readable reason
            - _halt_start_time: When halt began
            - _estimated_resume_time: When trading resumes

        Example:
            >>> status = {
            ...     "halted": True,
            ...     "level": 2,
            ...     "reason": "Level 2 circuit breaker",
            ...     "start_time": "2026-03-17T14:00:00Z",
            ...     "estimated_resume": "2026-03-17T14:15:00Z"
            ... }
            >>> detector._update_halt_status(status)
            >>> detector.is_trading_halted()
            True
        """
        self._halted = status.get("halted", False)
        self._halt_level = status.get("level")
        self._halt_reason = status.get("reason")
        self._halt_start_time = status.get("start_time")
        self._estimated_resume_time = status.get("estimated_resume")

    def _log_audit_event(
        self,
        event_type: str,
        halt_level: Optional[int],
        is_halted: bool
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (CHECK, DETECT, HALT, RESUME)
            halt_level: Circuit breaker level
            is_halted: Whether trading is halted
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = _get_current_time().isoformat()

        # Calculate time remaining
        time_remaining = None
        if is_halted and self._estimated_resume_time:
            remaining = self.get_halt_duration_remaining()
            if remaining:
                time_remaining = int(remaining.total_seconds())

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
                    "halt_level",
                    "is_halted",
                    "halt_reason",
                    "halt_start_time",
                    "estimated_resume_time",
                    "time_remaining"
                ])

            # Format times
            start_time_str = (
                self._halt_start_time.isoformat()
                if self._halt_start_time else ""
            )
            resume_time_str = (
                self._estimated_resume_time.isoformat()
                if self._estimated_resume_time else ""
            )

            # Write event
            writer.writerow([
                timestamp,
                event_type,
                halt_level,
                is_halted,
                self._halt_reason or "",
                start_time_str,
                resume_time_str,
                time_remaining or ""
            ])

        logger.debug("Circuit breaker audit event logged: {}".format(event_type))

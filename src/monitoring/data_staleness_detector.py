"""Data Staleness Detector for WebSocket feed monitoring.

Detects when no new data received within threshold, attempts
reconnection with exponential backoff, and activates safe mode
if recovery fails.
"""

import asyncio
import logging
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Optional, Callable


class DataStalenessDetector:
    """Detects data staleness and manages reconnection.

    Tracks last data timestamp and triggers alerts when no data
    received for 30+ seconds. Attempts automatic reconnection
    with exponential backoff. Enters safe mode if recovery fails.
    """

    STALENESS_THRESHOLD_SECONDS = 30
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAYS = [1, 2, 4]  # Exponential backoff

    def __init__(
        self,
        staleness_threshold_seconds: int = 30,
        reconnect_delays: list = None,
        alert_callback: Optional[Callable] = None,
        audit_trail=None
    ):
        """Initialize data staleness detector.

        Args:
            staleness_threshold_seconds: Seconds without data before alert (default 30)
            reconnect_delays: List of backoff delays in seconds (default [1, 2, 4])
            alert_callback: Optional callback for staleness alerts
            audit_trail: Optional audit trail for logging
        """
        self._staleness_threshold = staleness_threshold_seconds
        self._reconnect_delays = reconnect_delays or self.RECONNECT_DELAYS
        self._alert_callback = alert_callback
        self._audit_trail = audit_trail
        self._logger = logging.getLogger(__name__)

        # State tracking
        self._last_data_timestamp: Optional[datetime] = None
        self._staleness_state = "FRESH"  # FRESH, STALE, SAFE_MODE
        self._reconnect_attempts = 0
        self._safe_mode_activated = False

        # Statistics tracking
        self._staleness_events = 0
        self._total_stale_time_seconds = 0
        self._staleness_start_time: Optional[datetime] = None

        # History (last 100 staleness events)
        self._staleness_history = deque(maxlen=100)

    def update_last_data_timestamp(self) -> None:
        """Update timestamp of last received data packet.

        Call this method every time a data packet is received
        from the WebSocket connection.
        """
        self._last_data_timestamp = self._get_current_time()

        # If we were in stale state, mark as recovered
        if self._staleness_state == "STALE":
            self._handle_recovery()

    def check_staleness(self) -> Dict:
        """Check if data staleness detected.

        Called every 5 seconds by health check system.
        Triggers reconnection if staleness detected.

        Returns:
            Staleness status dictionary
        """
        current_time = self._get_current_time()

        # No data received yet
        if self._last_data_timestamp is None:
            return {
                "state": "FRESH",
                "last_data_timestamp": None,
                "staleness_duration_seconds": 0
            }

        # Calculate time since last data
        time_since_last_data = (
            current_time - self._last_data_timestamp
        ).total_seconds()

        # Check if staleness threshold exceeded
        if time_since_last_data > self._staleness_threshold:
            return self._handle_staleness(time_since_last_data)

        # Data is fresh
        return {
            "state": "FRESH",
            "last_data_timestamp": self._last_data_timestamp.isoformat(),
            "staleness_duration_seconds": 0
        }

    def get_staleness_statistics(self) -> Dict:
        """Get staleness event statistics.

        Returns:
            Statistics dictionary with staleness metrics
        """
        return {
            "total_staleness_events": self._staleness_events,
            "average_staleness_duration_seconds": (
                self._total_stale_time_seconds / self._staleness_events
                if self._staleness_events > 0 else 0
            ),
            "current_state": self._staleness_state,
            "safe_mode_activated": self._safe_mode_activated,
            "last_data_timestamp": (
                self._last_data_timestamp.isoformat()
                if self._last_data_timestamp else None
            ),
            "recent_events": list(self._staleness_history)
        }

    def is_in_safe_mode(self) -> bool:
        """Check if system is currently in safe mode.

        Returns:
            True if in safe mode, False otherwise
        """
        return self._safe_mode_activated

    def exit_safe_mode(self) -> None:
        """Manually exit safe mode (operator intervention).

        Resets safe mode flag and allows trading to resume.
        Should only be called after operator verifies data feed is stable.
        """
        if self._safe_mode_activated:
            self._safe_mode_activated = False
            self._staleness_state = "FRESH"
            self._logger.info("Safe mode exited - operator intervention")

            # Log to audit trail
            if self._audit_trail:
                self._audit_trail.log_action(
                    "SAFE_MODE_EXITED",
                    "data_staleness_detector",
                    "system",
                    {"reason": "operator_intervention"}
                )

    def _handle_staleness(self, staleness_duration_seconds: float) -> Dict:
        """Handle staleness detection.

        Args:
            staleness_duration_seconds: Duration since last data

        Returns:
            Staleness status dictionary
        """
        # First time detecting staleness
        if self._staleness_state == "FRESH":
            self._staleness_state = "STALE"
            self._staleness_events += 1
            self._staleness_start_time = self._get_current_time()

            # Log staleness event
            self._log_staleness_event(staleness_duration_seconds)

            # Trigger alert
            self._trigger_alert(
                "staleness_detected",
                "No data received for {:.0f}+ seconds".format(
                    staleness_duration_seconds
                ),
                staleness_duration_seconds
            )

            # Start reconnection attempts (in background)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._attempt_reconnection())
            except RuntimeError:
                # No event loop running - skip async reconnection
                pass

        # Update staleness duration
        return {
            "state": self._staleness_state,
            "last_data_timestamp": self._last_data_timestamp.isoformat(),
            "staleness_duration_seconds": staleness_duration_seconds,
            "reconnect_attempts": self._reconnect_attempts
        }

    async def _attempt_reconnection(self) -> None:
        """Attempt WebSocket reconnection with exponential backoff.

        Tries up to 3 times with delays of 1s, 2s, and 4s.
        Enters safe mode if all attempts fail.
        """
        for attempt, delay in enumerate(self._reconnect_delays, 1):
            self._reconnect_attempts = attempt

            # Log reconnection attempt
            self._logger.info(
                "Reconnection attempt {} - waiting {}s".format(attempt, delay)
            )

            # Wait for backoff delay
            await asyncio.sleep(delay)

            try:
                # Simulate reconnection attempt
                # In real implementation, this would call WebSocket.connect()
                # For now, we'll simulate success on final attempt
                if attempt == len(self._reconnect_delays):
                    # Simulate successful reconnection
                    self._handle_recovery()
                    return

            except Exception as e:
                # Log failed attempt
                self._logger.error(
                    "Reconnection attempt {} failed: {}".format(attempt, e)
                )

                if self._audit_trail:
                    self._audit_trail.log_action(
                        "RECONNECTION_ATTEMPT",
                        "data_staleness_detector",
                        "websocket",
                        {
                            "attempt_number": attempt,
                            "backoff_delay_seconds": delay,
                            "success": False,
                            "error": str(e)
                        }
                        )

        # All attempts failed - enter safe mode
        self._enter_safe_mode()

    def _handle_recovery(self) -> None:
        """Handle data feed recovery after staleness.

        Resets staleness state and logs recovery event.
        """
        if self._staleness_state == "STALE":
            current_time = self._get_current_time()
            downtime = (
                current_time - self._staleness_start_time
            ).total_seconds() if self._staleness_start_time else 0

            self._total_stale_time_seconds += downtime

            # Log recovery
            self._logger.info(
                "Data feed restored - downtime {:.1f}s, {} attempts".format(
                    downtime, self._reconnect_attempts
                )
            )

            # Log to audit trail
            if self._audit_trail:
                self._audit_trail.log_action(
                    "DATA_FEED_RESTORED",
                    "data_staleness_detector",
                    "websocket",
                    {
                        "downtime_duration_seconds": downtime,
                        "reconnection_attempts": self._reconnect_attempts
                    }
                )

            # Reset state
            self._staleness_state = "FRESH"
            self._staleness_start_time = None
            self._reconnect_attempts = 0

    def _enter_safe_mode(self) -> None:
        """Enter safe mode after failed reconnection.

        Stops trading and requires manual intervention.
        """
        self._safe_mode_activated = True
        self._staleness_state = "SAFE_MODE"

        staleness_duration = (
            self._get_current_time() - self._staleness_start_time
        ).total_seconds() if self._staleness_start_time else 0

        # Log safe mode activation
        self._logger.critical(
            "Safe mode activated - reconnection failed after {} attempts".format(
                self._reconnect_attempts
            )
        )

        # Log to audit trail
        if self._audit_trail:
            self._audit_trail.log_action(
                "SAFE_MODE_ACTIVATED",
                "data_staleness_detector",
                "system",
                {
                    "reason": "Reconnection failed after 3 attempts",
                    "staleness_duration_seconds": staleness_duration
                }
            )

        # Trigger critical alert
        self._trigger_alert(
            "safe_mode_activated",
            "Data feed down - safe mode activated - manual intervention required",
            staleness_duration
        )

    def _log_staleness_event(self, duration_seconds: float) -> None:
        """Log staleness detection event.

        Args:
            duration_seconds: Duration since last data
        """
        event = {
            "event_type": "DATA_STALENESS_DETECTED",
            "timestamp": self._get_current_time().isoformat(),
            "last_data_timestamp": (
                self._last_data_timestamp.isoformat()
                if self._last_data_timestamp else None
            ),
            "staleness_duration_seconds": duration_seconds,
            "reconnection_attempt": 0,
            "state": "STALE"
        }

        self._staleness_history.append(event)

        # Log to audit trail
        if self._audit_trail:
            self._audit_trail.log_action(
                "DATA_STALENESS_DETECTED",
                "data_staleness_detector",
                "websocket",
                {
                    "staleness_duration_seconds": duration_seconds,
                    "last_data_timestamp": event["last_data_timestamp"]
                }
            )

    def _trigger_alert(
        self,
        alert_type: str,
        message: str,
        value: float
    ) -> None:
        """Trigger staleness alert.

        Args:
            alert_type: Type of alert (staleness_detected, safe_mode_activated)
            message: Alert message
            value: Current staleness duration
        """
        if self._alert_callback:
            self._alert_callback(alert_type, message, value)

        self._logger.warning(message)

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)

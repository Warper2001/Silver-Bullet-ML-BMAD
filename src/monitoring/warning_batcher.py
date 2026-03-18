"""WarningBatcher for batching WARNING events.

Batches WARNING events to send together every 5 minutes,
preventing notification fatigue.
"""

import logging
from datetime import datetime, timezone
from typing import Dict


class WarningBatcher:
    """Batches WARNING events to send together every 5 minutes.

    Prevents notification fatigue by grouping non-critical warnings
    and sending them as a single notification.

    Attributes:
        _notification_manager: NotificationManager instance
        _batch_interval: Seconds between batches (default 300)
        _pending_warnings: List of pending warning events
        _last_batch_time: Timestamp of last batch sent
    """

    BATCH_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        notification_manager,
        batch_interval_seconds: int = 300
    ):
        """Initialize warning batcher.

        Args:
            notification_manager: NotificationManager instance
            batch_interval_seconds: Seconds between batches (default 300)
        """
        self._notification_manager = notification_manager
        self._batch_interval = batch_interval_seconds
        self._pending_warnings = []
        self._last_batch_time = None
        self._logger = logging.getLogger(__name__)

    def add_warning(self, event_data: Dict) -> None:
        """Add warning to batch.

        Args:
            event_data: Warning event data with keys:
                - event_type (str): Event type
                - message (str): Event message
                - timestamp (str): ISO 8601 timestamp
                - **kwargs: Additional event data
        """
        self._pending_warnings.append(event_data)
        self._check_and_send_batch()

    def _check_and_send_batch(self) -> None:
        """Check if batch interval expired and send."""
        current_time = self._get_current_time()

        # Only check interval if we've sent a batch before
        if self._last_batch_time is None:
            return

        # Check if batch interval expired
        time_since_last_batch = (
            current_time - self._last_batch_time
        ).total_seconds()

        if time_since_last_batch >= self._batch_interval:
            self._send_batched_warnings()
            self._last_batch_time = current_time

    def force_send_batch(self) -> None:
        """Force send batch immediately (for testing)."""
        self._send_batched_warnings()
        self._last_batch_time = self._get_current_time()

    def _send_batched_warnings(self) -> None:
        """Send all pending warnings as single notification."""
        if not self._pending_warnings:
            return

        # Format batched warnings
        message_lines = []
        for warning in self._pending_warnings:
            message_lines.append(
                "- {}: {}".format(
                    warning["event_type"],
                    warning["message"]
                )
            )

        batched_message = "System warnings ({} events):\n{}".format(
            len(self._pending_warnings),
            "\n".join(message_lines)
        )

        # Send batched notification
        success = self._notification_manager.send_notification(
            severity="WARNING",
            title="System Warnings ({})".format(
                len(self._pending_warnings)
            ),
            message=batched_message,
            events=self._pending_warnings
        )

        # Log result
        if success:
            self._logger.info(
                "Sent batched warning notification with {} events".format(
                    len(self._pending_warnings)
                )
            )
        else:
            self._logger.error(
                "Failed to send batched warning notification"
            )

        # Clear pending warnings
        self._pending_warnings = []

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)

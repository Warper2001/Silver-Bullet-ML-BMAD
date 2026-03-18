"""Notification integration for monitoring system events.

Integrates WarningBatcher with monitoring components to send
notifications for critical and warning events.
"""

import logging
from typing import Dict, Optional

from src.risk.notification_manager import NotificationManager
from src.monitoring.warning_batcher import WarningBatcher


class MonitoringNotificationIntegration:
    """Integrates notifications with monitoring system.

    Manages both immediate CRITICAL notifications and batched
    WARNING notifications for monitoring events.

    Attributes:
        _notification_manager: NotificationManager instance
        _warning_batcher: WarningBatcher instance
        _correlation_id_enabled: Whether to track correlation IDs
    """

    # Event type priorities
    CRITICAL_EVENTS = [
        "CRASH_DETECTED",
        "CRASH_RECOVERED",
        "SAFE_MODE_ACTIVATED",
        "EMERGENCY_STOP",
        "DATA_STALENESS_DETECTED",
        "HEALTH_CHECK_UNHEALTHY"
    ]

    WARNING_EVENTS = [
        "HIGH_CPU",
        "HIGH_MEMORY",
        "LOW_DISK",
        "HEALTH_CHECK_DEGRADED",
        "RECONNECTION_FAILED"
    ]

    def __init__(
        self,
        notification_manager: Optional[NotificationManager] = None,
        warning_batch_interval_seconds: int = 300,
        correlation_id_enabled: bool = True
    ):
        """Initialize monitoring notification integration.

        Args:
            notification_manager: NotificationManager instance
                (creates default if None)
            warning_batch_interval_seconds: Seconds between
                warning batches (default 300)
            correlation_id_enabled: Whether to track correlation IDs
        """
        self._notification_manager = (
            notification_manager or NotificationManager(enabled=True)
        )
        self._warning_batcher = WarningBatcher(
            self._notification_manager,
            batch_interval_seconds=warning_batch_interval_seconds
        )
        self._correlation_id_enabled = correlation_id_enabled
        self._logger = logging.getLogger(__name__)

    def send_notification(
        self,
        event_type: str,
        severity: str,
        title: str,
        message: str,
        **kwargs
    ) -> bool:
        """Send notification with proper routing.

        Routes CRITICAL events immediately and batches WARNING events.

        Args:
            event_type: Type of event (e.g., "HIGH_CPU")
            severity: Severity level (CRITICAL, WARNING, INFO)
            title: Notification title
            message: Notification message
            **kwargs: Additional event data

        Returns:
            True if notification sent, False otherwise

        Example:
            >>> integration = MonitoringNotificationIntegration()
            >>> integration.send_notification(
            ...     event_type="HIGH_MEMORY",
            ...     severity="WARNING",
            ...     title="High Memory Usage",
            ...     message="Memory usage at 85%"
            ... )
        """
        # Add correlation ID if enabled
        if self._correlation_id_enabled:
            kwargs["correlation_id"] = self._generate_correlation_id()

        # Route based on severity
        if severity == "CRITICAL":
            return self._send_critical_notification(
                event_type,
                title,
                message,
                **kwargs
            )
        elif severity == "WARNING":
            return self._add_warning_to_batch(
                event_type,
                title,
                message,
                **kwargs
            )
        else:
            # INFO events - send immediately
            return self._notification_manager.send_notification(
                severity=severity,
                title=title,
                message=message,
                notification_type=event_type
            )

    def _send_critical_notification(
        self,
        event_type: str,
        title: str,
        message: str,
        **kwargs
    ) -> bool:
        """Send critical notification immediately.

        Args:
            event_type: Type of event
            title: Notification title
            message: Notification message
            **kwargs: Additional event data

        Returns:
            True if sent successfully, False otherwise
        """
        self._logger.critical(
            "CRITICAL: {} - {}".format(event_type, message)
        )

        return self._notification_manager.send_notification(
            severity="CRITICAL",
            title=title,
            message=message,
            notification_type=event_type,
            **kwargs
        )

    def _add_warning_to_batch(
        self,
        event_type: str,
        title: str,
        message: str,
        **kwargs
    ) -> bool:
        """Add warning to batch.

        Args:
            event_type: Type of event
            title: Notification title
            message: Notification message
            **kwargs: Additional event data

        Returns:
            True (warnings are always added to batch)
        """
        self._logger.warning(
            "WARNING: {} - {}".format(event_type, message)
        )

        # Add to batch
        event_data = {
            "event_type": event_type,
            "message": message,
            "timestamp": self._get_current_time().isoformat(),
            **kwargs
        }

        self._warning_batcher.add_warning(event_data)
        return True

    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for event tracking.

        Returns:
            Correlation ID string
        """
        import uuid
        return "evt_{}".format(uuid.uuid4().hex[:12])

    def _get_current_time(self):
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)

    def get_notification_stats(self) -> Dict:
        """Get notification statistics.

        Returns:
            Dictionary with notification statistics
        """
        return self._notification_manager.get_notification_stats()

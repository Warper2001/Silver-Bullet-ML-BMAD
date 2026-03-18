"""Push notification manager for risk events.

This module implements a notification system that alerts traders
when risk limits are breached or critical events occur. This ensures
human operators are immediately informed of situations requiring
their attention.

Features:
- Push notification sending with severity levels
- Rate limiting to prevent notification spam
- Enable/disable functionality
- Statistics tracking
- Console output fallback
- CSV audit trail logging
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manage push notifications for risk events.

    Attributes:
        _enabled: Whether notifications are enabled
        _notification_queue: Queue of pending notifications
        _last_notification_times: Track recent notifications
        _rate_limit_seconds: Minimum seconds between similar notifications
        _total_sent: Total notifications sent
        _total_rate_limited: Total notifications rate limited
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> manager = NotificationManager(enabled=True)
        >>> manager.send_notification(
        ...     severity="CRITICAL",
        ...     title="Daily Loss Limit Breached",
        ...     message="Loss: $1,200 / $1,000 limit"
        ... )
    """

    # Severity levels
    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_INFO = "INFO"

    def __init__(
        self,
        enabled: bool = True,
        rate_limit_seconds: int = 300,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize notification manager.

        Args:
            enabled: Whether notifications are enabled
            rate_limit_seconds: Minimum seconds between similar
                notifications (default 300 = 5 minutes)
            audit_trail_path: Path to CSV audit trail file (optional)

        Example:
            >>> manager = NotificationManager(
            ...     enabled=True,
            ...     rate_limit_seconds=300
            ... )
        """
        self._enabled = enabled
        self._rate_limit_seconds = rate_limit_seconds
        self._audit_trail_path = audit_trail_path

        # Statistics
        self._total_sent = 0
        self._total_rate_limited = 0

        # Track last notification times per type
        self._last_notification_times = {}

        logger.info(
            "NotificationManager initialized: enabled={}, rate_limit={}s".format(
                enabled,
                rate_limit_seconds
            )
        )

    def send_notification(
        self,
        severity: str,
        title: str,
        message: str,
        notification_type: Optional[str] = None
    ) -> bool:
        """Send push notification.

        Args:
            severity: Notification severity (CRITICAL, WARNING, INFO)
            title: Notification title
            message: Notification message
            notification_type: Type of notification (optional,
                used for rate limiting)

        Returns:
            True if notification sent, False if rate limited or disabled

        Example:
            >>> manager.send_notification(
            ...     severity="CRITICAL",
            ...     title="Daily Loss Limit Breached",
            ...     message="Loss: $1,200 / $1,000 limit"
            ... )
        """
        # Check if notifications enabled
        if not self._enabled:
            logger.debug("Notifications disabled, skipping")
            return False

        # Check rate limiting
        if notification_type and self._should_rate_limit(notification_type):
            logger.debug(
                "Notification rate limited: {}".format(notification_type)
            )
            self._total_rate_limited += 1

            # Log rate limited event
            self._log_audit_event(
                severity=severity,
                title=title,
                message=message,
                notification_type=notification_type,
                sent_successfully=False,
                rate_limited=True
            )

            return False

        # Send notification
        success = self._send_to_external_service(severity, title, message)

        if success:
            self._total_sent += 1

            # Update last notification time
            if notification_type:
                self._last_notification_times[notification_type] = (
                    self._get_current_time()
                )

        # Log event
        self._log_audit_event(
            severity=severity,
            title=title,
            message=message,
            notification_type=notification_type,
            sent_successfully=success,
            rate_limited=False
        )

        return success

    def is_notification_enabled(self) -> bool:
        """Check if notifications are enabled.

        Returns:
            True if enabled, False otherwise

        Example:
            >>> if manager.is_notification_enabled():
            ...     manager.send_notification(...)
        """
        return self._enabled

    def enable_notifications(self) -> None:
        """Enable notifications.

        Example:
            >>> manager.enable_notifications()
        """
        self._enabled = True
        logger.info("Notifications enabled")

    def disable_notifications(self) -> None:
        """Disable notifications.

        Example:
            >>> manager.disable_notifications()
        """
        self._enabled = False
        logger.info("Notifications disabled")

    def get_notification_stats(self) -> dict:
        """Get notification statistics.

        Returns:
            Dictionary with stats:
            - total_sent: Total notifications sent
            - total_rate_limited: Total notifications rate limited
            - enabled: Whether notifications are enabled

        Example:
            >>> stats = manager.get_notification_stats()
            >>> print("Sent: {} notifications".format(
            ...     stats['total_sent']
            ... ))
        """
        return {
            "total_sent": self._total_sent,
            "total_rate_limited": self._total_rate_limited,
            "enabled": self._enabled
        }

    def _should_rate_limit(self, notification_type: str) -> bool:
        """Check if notification should be rate limited.

        Args:
            notification_type: Type of notification to check

        Returns:
            True if rate limited, False otherwise

        Example:
            >>> if manager._should_rate_limit("DAILY_LOSS_BREACH"):
            ...     return  # Skip this notification
        """
        if notification_type not in self._last_notification_times:
            return False

        last_time = self._last_notification_times[notification_type]
        current_time = self._get_current_time()

        time_since_last = (current_time - last_time).total_seconds()

        return time_since_last < self._rate_limit_seconds

    def _send_to_external_service(
        self,
        severity: str,
        title: str,
        message: str
    ) -> bool:
        """Send notification to console (fallback method).

        Args:
            severity: Notification severity
            title: Notification title
            message: Notification message

        Returns:
            True if sent successfully, False otherwise

        Example:
            >>> success = manager._send_to_external_service(
            ...     severity="CRITICAL",
            ...     title="Alert",
            ...     message="Something happened"
            ... )
        """
        try:
            # Format output with severity prefix
            prefix_map = {
                self.SEVERITY_CRITICAL: "🚨",
                self.SEVERITY_WARNING: "⚠️ ",
                self.SEVERITY_INFO: "ℹ️ "
            }

            prefix = prefix_map.get(severity, "📢")

            print("{} [{}] {}".format(prefix, title, message))

            return True

        except Exception as e:
            logger.error(
                "Failed to send notification: {}".format(str(e))
            )
            return False

    def _get_current_time(self) -> datetime:
        """Get current time (UTC).

        Returns:
            Current datetime in UTC

        Example:
            >>> now = manager._get_current_time()
        """
        return datetime.now(timezone.utc)

    def _log_audit_event(
        self,
        severity: str,
        title: str,
        message: str,
        notification_type: Optional[str],
        sent_successfully: bool,
        rate_limited: bool
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            severity: Notification severity
            title: Notification title
            message: Notification message
            notification_type: Type of notification
            sent_successfully: Whether notification was sent
            rate_limited: Whether notification was rate limited
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
                    "severity",
                    "title",
                    "message",
                    "notification_type",
                    "sent_successfully",
                    "rate_limited"
                ])

            # Write event
            writer.writerow([
                timestamp,
                severity,
                title,
                message,
                notification_type or "",
                sent_successfully,
                rate_limited
            ])

        logger.debug("Notification audit logged")

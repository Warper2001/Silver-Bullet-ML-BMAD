"""Unit tests for Notification Manager.

Tests notification sending, rate limiting, enable/disable,
statistics tracking, CSV logging, and console output.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from src.risk.notification_manager import NotificationManager


class TestNotificationManagerInit:
    """Test NotificationManager initialization."""

    def test_init_with_default_parameters(self):
        """Verify manager initializes with defaults."""
        manager = NotificationManager()

        assert manager._enabled is True
        assert manager._rate_limit_seconds == 300
        assert manager._total_sent == 0
        assert manager._total_rate_limited == 0

    def test_init_with_custom_parameters(self):
        """Verify manager initializes with custom parameters."""
        manager = NotificationManager(
            enabled=False,
            rate_limit_seconds=600
        )

        assert manager._enabled is False
        assert manager._rate_limit_seconds == 600

    def test_init_with_audit_trail(self):
        """Verify manager initializes with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "notifications.csv")

        manager = NotificationManager(audit_trail_path=audit_path)

        assert manager._audit_trail_path == audit_path


class TestSendNotification:
    """Test notification sending."""

    @pytest.fixture
    def manager(self):
        """Create notification manager."""
        return NotificationManager()

    def test_send_notification_when_enabled(self, manager, capsys):
        """Verify notification sent when enabled."""
        result = manager.send_notification(
            severity="CRITICAL",
            title="Test Alert",
            message="Test message"
        )

        assert result is True
        assert manager._total_sent == 1

        # Check console output
        captured = capsys.readouterr()
        assert "🚨" in captured.out
        assert "[Test Alert]" in captured.out
        assert "Test message" in captured.out

    def test_send_notification_when_disabled(self, manager):
        """Verify notification not sent when disabled."""
        manager.disable_notifications()

        result = manager.send_notification(
            severity="CRITICAL",
            title="Test Alert",
            message="Test message"
        )

        assert result is False
        assert manager._total_sent == 0

    def test_send_notification_with_type(self, manager):
        """Verify notification with type is sent."""
        result = manager.send_notification(
            severity="WARNING",
            title="Test Warning",
            message="Warning message",
            notification_type="TEST_WARNING"
        )

        assert result is True
        assert "TEST_WARNING" in manager._last_notification_times

    def test_send_notification_without_type(self, manager):
        """Verify notification without type is sent."""
        result = manager.send_notification(
            severity="INFO",
            title="Test Info",
            message="Info message"
        )

        assert result is True
        assert len(manager._last_notification_times) == 0

    def test_send_critical_notification(self, manager, capsys):
        """Verify critical notification formatted correctly."""
        manager.send_notification(
            severity="CRITICAL",
            title="Critical Alert",
            message="Critical issue"
        )

        captured = capsys.readouterr()
        assert "🚨" in captured.out

    def test_send_warning_notification(self, manager, capsys):
        """Verify warning notification formatted correctly."""
        manager.send_notification(
            severity="WARNING",
            title="Warning Alert",
            message="Warning issue"
        )

        captured = capsys.readouterr()
        assert "⚠️ " in captured.out

    def test_send_info_notification(self, manager, capsys):
        """Verify info notification formatted correctly."""
        manager.send_notification(
            severity="INFO",
            title="Info Alert",
            message="Info message"
        )

        captured = capsys.readouterr()
        assert "ℹ️" in captured.out


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def manager(self):
        """Create notification manager with short rate limit."""
        return NotificationManager(rate_limit_seconds=60)

    def test_first_notification_sent(self, manager):
        """Verify first notification of type is sent."""
        result = manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST_TYPE"
        )

        assert result is True
        assert manager._total_sent == 1

    def test_duplicate_notification_rate_limited(self, manager):
        """Verify duplicate notification is rate limited."""
        # First notification
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST_TYPE"
        )

        # Immediate duplicate (should be rate limited)
        result = manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST_TYPE"
        )

        assert result is False
        assert manager._total_sent == 1
        assert manager._total_rate_limited == 1

    def test_different_types_not_rate_limited(self, manager):
        """Verify different notification types not rate limited."""
        # First type
        manager.send_notification(
            severity="CRITICAL",
            title="Test1",
            message="Test1",
            notification_type="TYPE_1"
        )

        # Different type (should not be rate limited)
        result = manager.send_notification(
            severity="CRITICAL",
            title="Test2",
            message="Test2",
            notification_type="TYPE_2"
        )

        assert result is True
        assert manager._total_sent == 2

    def test_rate_limit_expires_after_timeout(self, manager):
        """Verify rate limit expires after timeout."""
        # First notification
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST_TYPE"
        )

        # Manually expire rate limit
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        manager._last_notification_times["TEST_TYPE"] = old_time

        # Should not be rate limited now
        result = manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST_TYPE"
        )

        assert result is True
        assert manager._total_sent == 2

    def test_no_type_no_rate_limiting(self, manager):
        """Verify notifications without type are not rate limited."""
        for i in range(5):
            manager.send_notification(
                severity="INFO",
                title="Test {}".format(i),
                message="Message {}".format(i)
            )

        assert manager._total_sent == 5
        assert manager._total_rate_limited == 0


class TestEnableDisable:
    """Test enable/disable functionality."""

    @pytest.fixture
    def manager(self):
        """Create notification manager."""
        return NotificationManager()

    def test_is_notification_enabled_when_enabled(self, manager):
        """Verify is_notification_enabled returns True."""
        assert manager.is_notification_enabled() is True

    def test_is_notification_enabled_when_disabled(self, manager):
        """Verify is_notification_enabled returns False."""
        manager.disable_notifications()
        assert manager.is_notification_enabled() is False

    def test_enable_notifications(self, manager):
        """Verify enable_notifications works."""
        manager.disable_notifications()
        assert manager.is_notification_enabled() is False

        manager.enable_notifications()
        assert manager.is_notification_enabled() is True

    def test_disable_notifications(self, manager):
        """Verify disable_notifications works."""
        manager.disable_notifications()
        assert manager.is_notification_enabled() is False

    def test_send_after_disable(self, manager):
        """Verify sending fails after disable."""
        manager.disable_notifications()

        result = manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test"
        )

        assert result is False

    def test_send_after_enable(self, manager):
        """Verify sending works after enable."""
        manager.disable_notifications()
        manager.enable_notifications()

        result = manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test"
        )

        assert result is True


class TestNotificationStats:
    """Test notification statistics."""

    @pytest.fixture
    def manager(self):
        """Create notification manager."""
        return NotificationManager()

    def test_stats_initially_zero(self, manager):
        """Verify stats start at zero."""
        stats = manager.get_notification_stats()

        assert stats["total_sent"] == 0
        assert stats["total_rate_limited"] == 0
        assert stats["enabled"] is True

    def test_stats_update_after_send(self, manager):
        """Verify stats update after sending."""
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test"
        )

        stats = manager.get_notification_stats()

        assert stats["total_sent"] == 1
        assert stats["total_rate_limited"] == 0

    def test_stats_include_rate_limited(self, manager):
        """Verify stats include rate limited count."""
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST"
        )

        # Duplicate (rate limited)
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST"
        )

        stats = manager.get_notification_stats()

        assert stats["total_sent"] == 1
        assert stats["total_rate_limited"] == 1

    def test_stats_reflect_disabled_state(self, manager):
        """Verify stats reflect enabled state."""
        manager.disable_notifications()

        stats = manager.get_notification_stats()

        assert stats["enabled"] is False


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def manager(self):
        """Create notification manager with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "notifications.csv")

        return NotificationManager(audit_trail_path=audit_path)

    def test_csv_file_created_on_send(self, manager):
        """Verify CSV file created on send."""
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test"
        )

        assert Path(manager._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, manager):
        """Verify CSV has all required columns."""
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test"
        )

        import csv
        with open(manager._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "severity",
            "title",
            "message",
            "notification_type",
            "sent_successfully",
            "rate_limited"
        ]

        assert headers == expected_headers

    def test_csv_logs_sent_notification(self, manager):
        """Verify CSV logs sent notification."""
        manager.send_notification(
            severity="CRITICAL",
            title="Test Alert",
            message="Test message",
            notification_type="TEST_TYPE"
        )

        import csv
        with open(manager._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["severity"] == "CRITICAL"
        assert rows[0]["title"] == "Test Alert"
        assert rows[0]["message"] == "Test message"
        assert rows[0]["notification_type"] == "TEST_TYPE"
        assert rows[0]["sent_successfully"] == "True"
        assert rows[0]["rate_limited"] == "False"

    def test_csv_logs_rate_limited_notification(self, manager):
        """Verify CSV logs rate limited notification."""
        # First notification
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST"
        )

        # Rate limited duplicate
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TEST"
        )

        import csv
        with open(manager._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Second row should be rate limited
        assert rows[1]["sent_successfully"] == "False"
        assert rows[1]["rate_limited"] == "True"


class TestMultipleNotifications:
    """Test multiple notifications in sequence."""

    @pytest.fixture
    def manager(self):
        """Create notification manager."""
        return NotificationManager()

    def test_multiple_notifications_sent(self, manager):
        """Verify multiple notifications sent successfully."""
        for i in range(5):
            manager.send_notification(
                severity="INFO",
                title="Alert {}".format(i),
                message="Message {}".format(i),
                notification_type="TYPE_{}".format(i)
            )

        assert manager._total_sent == 5

    def test_mixed_severities(self, manager, capsys):
        """Verify mixed severity notifications."""
        manager.send_notification(
            severity="CRITICAL",
            title="Critical",
            message="Critical message"
        )

        manager.send_notification(
            severity="WARNING",
            title="Warning",
            message="Warning message"
        )

        manager.send_notification(
            severity="INFO",
            title="Info",
            message="Info message"
        )

        assert manager._total_sent == 3

        captured = capsys.readouterr()
        assert "🚨" in captured.out
        assert "⚠️ " in captured.out
        assert "ℹ️" in captured.out

    def test_stats_track_all_notifications(self, manager):
        """Verify stats track all notifications."""
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TYPE"
        )

        # Rate limited
        manager.send_notification(
            severity="CRITICAL",
            title="Test",
            message="Test",
            notification_type="TYPE"
        )

        # Different type
        manager.send_notification(
            severity="WARNING",
            title="Test2",
            message="Test2",
            notification_type="TYPE2"
        )

        stats = manager.get_notification_stats()

        assert stats["total_sent"] == 2
        assert stats["total_rate_limited"] == 1

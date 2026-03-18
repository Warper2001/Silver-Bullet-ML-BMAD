"""Unit tests for MonitoringNotificationIntegration.

Tests notification routing, critical event handling,
warning batching, and correlation ID generation.
"""

from unittest.mock import MagicMock, patch

from src.monitoring.notification_integration import (
    MonitoringNotificationIntegration
)


class TestMonitoringNotificationIntegrationInit:
    """Test MonitoringNotificationIntegration initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        integration = MonitoringNotificationIntegration()

        assert integration._notification_manager is not None
        assert integration._warning_batcher is not None
        assert integration._correlation_id_enabled is True

    def test_init_with_custom_notification_manager(self):
        """Verify initialization with custom notification manager."""
        notification_manager = MagicMock()
        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager
        )

        assert integration._notification_manager == notification_manager

    def test_init_with_custom_batch_interval(self):
        """Verify initialization with custom batch interval."""
        integration = MonitoringNotificationIntegration(
            warning_batch_interval_seconds=600
        )

        assert integration._warning_batcher._batch_interval == 600

    def test_init_with_correlation_id_disabled(self):
        """Verify initialization with correlation ID disabled."""
        integration = MonitoringNotificationIntegration(
            correlation_id_enabled=False
        )

        assert integration._correlation_id_enabled is False


class TestSendNotification:
    """Test send_notification functionality."""

    def test_send_critical_notification_immediately(self):
        """Verify CRITICAL events are sent immediately."""
        notification_manager = MagicMock()
        notification_manager.send_notification.return_value = True
        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager
        )

        result = integration.send_notification(
            event_type="SAFE_MODE_ACTIVATED",
            severity="CRITICAL",
            title="Safe Mode Activated",
            message="Data feed down - safe mode activated"
        )

        assert result is True
        notification_manager.send_notification.assert_called_once()

        # Verify CRITICAL severity
        call_args = notification_manager.send_notification.call_args
        assert call_args[1]["severity"] == "CRITICAL"

    def test_send_warning_notification_batched(self):
        """Verify WARNING events are batched."""
        notification_manager = MagicMock()
        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager
        )

        # Initialize batch time
        integration._warning_batcher._last_batch_time = (
            integration._get_current_time()
        )

        result = integration.send_notification(
            event_type="HIGH_MEMORY",
            severity="WARNING",
            title="High Memory Usage",
            message="Memory usage at 85%"
        )

        assert result is True

        # Verify warning was added to batch
        assert len(integration._warning_batcher._pending_warnings) == 1

    def test_send_info_notification_immediately(self):
        """Verify INFO events are sent immediately."""
        notification_manager = MagicMock()
        notification_manager.send_notification.return_value = True
        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager
        )

        result = integration.send_notification(
            event_type="SYSTEM_STARTUP",
            severity="INFO",
            title="System Startup",
            message="System started successfully"
        )

        assert result is True
        notification_manager.send_notification.assert_called_once()

        # Verify INFO severity
        call_args = notification_manager.send_notification.call_args
        assert call_args[1]["severity"] == "INFO"

    @patch("uuid.uuid4")
    def test_send_notification_adds_correlation_id(
        self,
        mock_uuid4
    ):
        """Verify correlation ID is added when enabled."""
        mock_uuid4.return_value.hex = "abc123def456"

        notification_manager = MagicMock()
        notification_manager.send_notification.return_value = True
        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager,
            correlation_id_enabled=True
        )

        integration.send_notification(
            event_type="SAFE_MODE_ACTIVATED",
            severity="CRITICAL",
            title="Safe Mode Activated",
            message="Data feed down"
        )

        # Verify correlation ID in call
        call_args = notification_manager.send_notification.call_args
        assert "correlation_id" in call_args[1]
        assert call_args[1]["correlation_id"] == "evt_abc123def456"

    def test_send_notification_no_correlation_id_when_disabled(self):
        """Verify correlation ID not added when disabled."""
        notification_manager = MagicMock()
        notification_manager.send_notification.return_value = True
        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager,
            correlation_id_enabled=False
        )

        integration.send_notification(
            event_type="SAFE_MODE_ACTIVATED",
            severity="CRITICAL",
            title="Safe Mode Activated",
            message="Data feed down"
        )

        # Verify correlation ID not in call
        call_args = notification_manager.send_notification.call_args
        assert "correlation_id" not in call_args[1]


class TestGenerateCorrelationId:
    """Test _generate_correlation_id functionality."""

    @patch("uuid.uuid4")
    def test_generate_correlation_id_format(self, mock_uuid4):
        """Verify correlation ID format is correct."""
        mock_uuid4.return_value.hex = "abc123def456"

        integration = MonitoringNotificationIntegration()

        result = integration._generate_correlation_id()

        assert result == "evt_abc123def456"


class TestGetNotificationStats:
    """Test get_notification_stats functionality."""

    def test_get_notification_stats_returns_stats(self):
        """Verify get_notification_stats returns statistics."""
        notification_manager = MagicMock()
        notification_manager.get_notification_stats.return_value = {
            "total_sent": 10,
            "total_rate_limited": 2,
            "enabled": True
        }

        integration = MonitoringNotificationIntegration(
            notification_manager=notification_manager
        )

        stats = integration.get_notification_stats()

        assert stats["total_sent"] == 10
        assert stats["total_rate_limited"] == 2
        assert stats["enabled"] is True

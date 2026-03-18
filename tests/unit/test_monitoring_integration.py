"""Unit tests for MonitoringIntegration.

Tests integration of all monitoring components into asyncio data pipeline
with background tasks for health checks, resource monitoring, state persistence,
and daily report generation.
"""

from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime, timezone

from src.monitoring.monitoring_integration import MonitoringIntegration


class TestMonitoringIntegrationInit:
    """Test MonitoringIntegration initialization."""

    def test_init_with_all_components(self):
        """Verify initialization with all components."""
        health_check = MagicMock()
        resource_monitor = MagicMock()
        crash_recovery = MagicMock()
        audit_trail = MagicMock()
        staleness_detector = MagicMock()
        notification_integration = MagicMock()
        daily_report = MagicMock()
        error_logger = MagicMock()
        graceful_shutdown = MagicMock()

        monitoring = MonitoringIntegration(
            health_check_manager=health_check,
            resource_monitor=resource_monitor,
            crash_recovery=crash_recovery,
            audit_trail=audit_trail,
            staleness_detector=staleness_detector,
            notification_integration=notification_integration,
            daily_report_generator=daily_report,
            error_logger=error_logger,
            graceful_shutdown_manager=graceful_shutdown
        )

        assert monitoring._health_check_manager == health_check
        assert monitoring._resource_monitor == resource_monitor
        assert monitoring._crash_recovery == crash_recovery
        assert monitoring._audit_trail == audit_trail
        assert monitoring._staleness_detector == staleness_detector
        assert monitoring._notification_integration == notification_integration
        assert monitoring._daily_report_generator == daily_report
        assert monitoring._error_logger == error_logger
        assert monitoring._graceful_shutdown == graceful_shutdown

    def test_init_with_optional_components(self):
        """Verify initialization with optional components None."""
        monitoring = MonitoringIntegration(
            health_check_manager=None,
            resource_monitor=None,
            crash_recovery=None,
            audit_trail=None,
            staleness_detector=None,
            notification_integration=None,
            daily_report_generator=None,
            error_logger=None,
            graceful_shutdown_manager=None
        )

        assert monitoring._health_check_manager is None
        assert monitoring._resource_monitor is None
        assert monitoring._crash_recovery is None
        assert monitoring._audit_trail is None
        assert monitoring._staleness_detector is None
        assert monitoring._notification_integration is None
        assert monitoring._daily_report_generator is None
        assert monitoring._error_logger is None
        assert monitoring._graceful_shutdown is None

    def test_init_with_custom_state_file_path(self):
        """Verify initialization with custom state file path."""
        monitoring = MonitoringIntegration(
            state_file_path="custom/state.json"
        )

        assert monitoring._state_file_path == "custom/state.json"

    def test_monitoring_statistics_initialized(self):
        """Verify monitoring statistics initialized correctly."""
        monitoring = MonitoringIntegration()

        assert monitoring._health_check_count == 0
        assert monitoring._health_check_pass_count == 0
        assert monitoring._error_count == 0
        assert monitoring._recovery_count == 0
        assert monitoring._total_uptime_seconds == 0
        assert monitoring._start_time is None


class TestBackgroundTaskCreation:
    """Test background task creation."""

    def test_start_method_exists(self):
        """Verify start method exists and is callable."""
        monitoring = MonitoringIntegration()

        # Just verify the method exists and can be called
        assert hasattr(monitoring, 'start')
        assert callable(monitoring.start)

    def test_background_task_attributes_initialized(self):
        """Verify background task attributes initialized."""
        monitoring = MonitoringIntegration()

        assert monitoring._health_check_task is None
        assert monitoring._resource_monitor_task is None
        assert monitoring._state_persistence_task is None
        assert monitoring._daily_report_task is None


class TestLogAction:
    """Test log_action functionality."""

    def test_log_action_delegates_to_audit_trail(self):
        """Verify log_action delegates to audit trail."""
        audit_trail = MagicMock()
        monitoring = MonitoringIntegration(audit_trail=audit_trail)

        # Test that the method exists and works
        import asyncio
        asyncio.run(monitoring.log_action(
            "TEST_ACTION",
            "test_component",
            "test_target",
            {"key": "value"}
        ))

        audit_trail.log_action.assert_called_once()

    def test_log_action_handles_missing_audit_trail(self):
        """Verify log_action handles missing audit trail."""
        monitoring = MonitoringIntegration(audit_trail=None)

        # Should not raise exception
        import asyncio
        asyncio.run(monitoring.log_action(
            "TEST_ACTION",
            "test_component",
            "test_target"
        ))


class TestLogError:
    """Test log_error functionality."""

    def test_log_error_delegates_to_error_logger(self):
        """Verify log_error delegates to error logger."""
        error_logger = MagicMock()
        error_logger.log_error = MagicMock(return_value="ERR_123")
        monitoring = MonitoringIntegration(error_logger=error_logger)

        error = ValueError("Test error")

        import asyncio
        error_id = asyncio.run(monitoring.log_error(error))

        assert error_id == "ERR_123"
        assert monitoring._error_count == 1

    def test_log_error_increments_error_count(self):
        """Verify log_error increments error count."""
        error_logger = MagicMock()
        error_logger.log_error = MagicMock(return_value="ERR_123")
        monitoring = MonitoringIntegration(error_logger=error_logger)

        error = ValueError("Test error")

        import asyncio
        asyncio.run(monitoring.log_error(error))

        assert monitoring._error_count == 1

    def test_log_error_sends_notification_for_critical(self):
        """Verify log_error sends notification for CRITICAL."""
        error_logger = MagicMock()
        error_logger.log_error = MagicMock(return_value="ERR_123")
        notification_integration = MagicMock()
        monitoring = MonitoringIntegration(
            error_logger=error_logger,
            notification_integration=notification_integration
        )

        error = RuntimeError("Critical error")

        import asyncio
        asyncio.run(monitoring.log_error(error, severity="CRITICAL"))

        # Notification should be sent
        assert notification_integration.send_notification.called

    def test_log_error_returns_error_id(self):
        """Verify log_error returns error ID."""
        error_logger = MagicMock()
        error_logger.log_error = MagicMock(return_value="ERR_ABC123")
        monitoring = MonitoringIntegration(error_logger=error_logger)

        error = ValueError("Test")

        import asyncio
        error_id = asyncio.run(monitoring.log_error(error))

        assert error_id == "ERR_ABC123"


class TestCheckDataStaleness:
    """Test check_data_staleness functionality."""

    def test_check_data_staleness_delegates_to_detector(self):
        """Verify check_data_staleness delegates to detector."""
        staleness_detector = MagicMock()
        staleness_detector.is_data_stale = MagicMock(return_value=False)
        monitoring = MonitoringIntegration(staleness_detector=staleness_detector)

        timestamp = datetime.now(timezone.utc)

        import asyncio
        is_stale = asyncio.run(monitoring.check_data_staleness(timestamp))

        assert is_stale is False
        staleness_detector.is_data_stale.assert_called_once_with(timestamp)

    def test_check_data_staleness_logs_action_when_stale(self):
        """Verify check_data_staleness logs action when stale."""
        staleness_detector = MagicMock()
        staleness_detector.is_data_stale = MagicMock(return_value=True)
        audit_trail = MagicMock()
        monitoring = MonitoringIntegration(
            staleness_detector=staleness_detector,
            audit_trail=audit_trail
        )

        timestamp = datetime.now(timezone.utc)

        import asyncio
        is_stale = asyncio.run(monitoring.check_data_staleness(timestamp))

        assert is_stale is True
        audit_trail.log_action.assert_called_once()

    def test_check_data_staleness_handles_missing_detector(self):
        """Verify check_data_staleness handles missing detector."""
        monitoring = MonitoringIntegration(staleness_detector=None)

        timestamp = datetime.now(timezone.utc)

        import asyncio
        is_stale = asyncio.run(monitoring.check_data_staleness(timestamp))

        assert is_stale is False


class TestGetMonitoringStatistics:
    """Test get_monitoring_statistics functionality."""

    def test_get_monitoring_statistics_returns_all_fields(self):
        """Verify get_monitoring_statistics returns all fields."""
        monitoring = MonitoringIntegration()
        monitoring._start_time = datetime.now(timezone.utc)
        monitoring._health_check_count = 100
        monitoring._health_check_pass_count = 95
        monitoring._error_count = 3
        monitoring._recovery_count = 1

        stats = monitoring.get_monitoring_statistics()

        assert "uptime_seconds" in stats
        assert "health_check_count" in stats
        assert "health_check_pass_count" in stats
        assert "health_check_pass_rate" in stats
        assert "error_count" in stats
        assert "recovery_count" in stats
        assert "avg_cpu_usage" in stats
        assert "avg_memory_usage" in stats

    def test_health_check_pass_rate_calculated_correctly(self):
        """Verify health check pass rate calculated correctly."""
        monitoring = MonitoringIntegration()
        monitoring._health_check_count = 100
        monitoring._health_check_pass_count = 95

        stats = monitoring.get_monitoring_statistics()

        assert stats["health_check_pass_rate"] == 0.95

    def test_uptime_calculated_correctly(self):
        """Verify uptime calculated correctly."""
        monitoring = MonitoringIntegration()
        monitoring._start_time = datetime.now(timezone.utc)

        stats = monitoring.get_monitoring_statistics()

        assert stats["uptime_seconds"] >= 0


class TestHealthCheckLogic:
    """Test health check loop logic."""

    def test_health_check_components_exist(self):
        """Verify health check components are properly set."""
        health_check_manager = MagicMock()
        monitoring = MonitoringIntegration(
            health_check_manager=health_check_manager
        )
        monitoring._graceful_shutdown = MagicMock()
        monitoring._graceful_shutdown.is_shutdown_requested = MagicMock(
            return_value=True
        )

        # Verify components are set
        assert monitoring._health_check_manager == health_check_manager
        assert monitoring._graceful_shutdown is not None


class TestPersistState:
    """Test state persistence functionality."""

    @patch("src.monitoring.monitoring_integration.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_persist_state_creates_directory(self, mock_file, mock_path):
        """Verify persist_state creates directory."""
        monitoring = MonitoringIntegration()

        import asyncio
        asyncio.run(monitoring.persist_state())

        # Verify directory creation
        mock_path.return_value.parent.mkdir.assert_called_once()

    @patch("src.monitoring.monitoring_integration.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_persist_state_writes_file(self, mock_file, mock_path):
        """Verify persist_state writes file."""
        monitoring = MonitoringIntegration()

        import asyncio
        asyncio.run(monitoring.persist_state())

        # Verify file was written
        mock_file.assert_called_once()

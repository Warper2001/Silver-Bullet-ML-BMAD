"""Unit tests for Resource Monitor.

Tests CPU, memory, and disk monitoring with alert thresholds,
consecutive check tracking, history tracking, and trend analysis.
"""

from collections import deque
from unittest.mock import MagicMock, patch

from src.monitoring.resource_monitor import ResourceMonitor


class TestResourceMonitorInit:
    """Test ResourceMonitor initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        monitor = ResourceMonitor()

        assert monitor._check_interval == 10
        assert monitor._alert_callback is None
        assert monitor._audit_trail is None

    def test_init_with_custom_interval(self):
        """Verify initialization with custom check interval."""
        monitor = ResourceMonitor(check_interval_seconds=5)

        assert monitor._check_interval == 5

    def test_init_with_alert_callback(self):
        """Verify initialization with alert callback."""
        callback = MagicMock()
        monitor = ResourceMonitor(alert_callback=callback)

        assert monitor._alert_callback == callback

    def test_init_with_audit_trail(self):
        """Verify initialization with audit trail."""
        audit_trail = MagicMock()
        monitor = ResourceMonitor(audit_trail=audit_trail)

        assert monitor._audit_trail == audit_trail

    def test_init_initializes_tracking_variables(self):
        """Verify initialization of tracking variables."""
        monitor = ResourceMonitor()

        assert monitor._cpu_consecutive_high == 0
        assert monitor._memory_consecutive_high == 0
        assert isinstance(monitor._cpu_history, deque)
        assert isinstance(monitor._memory_history, deque)
        assert isinstance(monitor._disk_history, deque)


class TestCheckResources:
    """Test check_resources functionality."""

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_returns_all_metrics(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify check_resources returns all metrics."""
        mock_cpu.return_value = 45.5
        mock_memory.return_value = MagicMock(
            rss=1024 * 1024 * 512,  # 512 MB
            percent=25.3
        )
        mock_disk.return_value = MagicMock(
            percent=60.0,
            free=150 * (1024**3),  # 150 GB
            total=500 * (1024**3)   # 500 GB
        )

        monitor = ResourceMonitor()
        metrics = monitor.check_resources()

        assert "cpu_percent" in metrics
        assert "memory_mb" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert "disk_free_gb" in metrics
        assert "disk_total_gb" in metrics
        assert metrics["cpu_percent"] == 45.5
        assert metrics["memory_percent"] == 25.3
        assert metrics["disk_percent"] == 60.0

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_no_alerts_when_normal(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify no alerts when resources are normal."""
        mock_cpu.return_value = 45.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        monitor = ResourceMonitor()
        metrics = monitor.check_resources()

        assert metrics["cpu_alert"] is False
        assert metrics["memory_alert"] is False
        assert metrics["disk_alert"] is False

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_cpu_alert_after_3_consecutive(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify CPU alert triggers after 3 consecutive high checks."""
        mock_cpu.return_value = 85.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        monitor = ResourceMonitor()

        # First 2 checks - no alert
        monitor.check_resources()
        monitor.check_resources()
        metrics = monitor.check_resources()

        assert metrics["cpu_alert"] is True
        assert monitor._cpu_consecutive_high == 3

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_cpu_alert_resets_when_cpu_drops(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify CPU alert resets when CPU drops."""
        # First 3 checks with high CPU
        mock_cpu.return_value = 85.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        monitor = ResourceMonitor()
        monitor.check_resources()
        monitor.check_resources()
        monitor.check_resources()

        assert monitor._cpu_consecutive_high == 3

        # CPU drops
        mock_cpu.return_value = 50.0
        metrics = monitor.check_resources()

        assert metrics["cpu_alert"] is False
        assert monitor._cpu_consecutive_high == 0

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_memory_alert_after_3_consecutive(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify memory alert triggers after 3 consecutive high checks."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=85.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        monitor = ResourceMonitor()

        # First 2 checks - no alert
        monitor.check_resources()
        monitor.check_resources()
        metrics = monitor.check_resources()

        assert metrics["memory_alert"] is True
        assert monitor._memory_consecutive_high == 3

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_disk_alert_immediate(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify disk alert triggers immediately when threshold exceeded."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(
            percent=85.0,
            free=75 * (1024**3),  # 75 GB
            total=500 * (1024**3)   # 500 GB
        )

        monitor = ResourceMonitor()
        metrics = monitor.check_resources()

        assert metrics["disk_alert"] is True
        # Should trigger on first check, not require consecutive

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_updates_history(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify that check_resources updates history."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=50.0)

        monitor = ResourceMonitor()

        # Check resources multiple times
        for _ in range(5):
            monitor.check_resources()

        assert len(monitor._cpu_history) == 5
        assert len(monitor._memory_history) == 5
        assert len(monitor._disk_history) == 5


class TestGetResourceHistory:
    """Test get_resource_history functionality."""

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_get_resource_history_returns_statistics(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify get_resource_history returns statistics."""
        mock_cpu.return_value = 45.0
        mock_memory.return_value = MagicMock(percent=55.0)
        mock_disk.return_value = MagicMock(percent=65.0)

        monitor = ResourceMonitor()

        # Add some history
        for i in range(10):
            monitor._cpu_history.append(40.0 + i)
            monitor._memory_history.append(50.0 + i)
            monitor._disk_history.append(60.0 + i)

        history = monitor.get_resource_history()

        assert "cpu" in history
        assert "memory" in history
        assert "disk" in history
        assert "current" in history["cpu"]
        assert "average" in history["cpu"]
        assert "peak" in history["cpu"]
        assert "trend" in history["cpu"]

    def test_get_resource_history_no_history(self):
        """Verify get_resource_history handles no history."""
        monitor = ResourceMonitor()

        history = monitor.get_resource_history()

        assert "error" in history
        assert "No history available" in history["error"]


class TestCalculateTrend:
    """Test trend calculation functionality."""

    def test_calculate_trend_increasing(self):
        """Verify trend is increasing when values rise."""
        monitor = ResourceMonitor()

        # Create increasing history
        history = list(range(10, 30))  # 20 values, increasing

        trend = monitor._calculate_trend(history)

        assert trend == "increasing"

    def test_calculate_trend_decreasing(self):
        """Verify trend is decreasing when values fall."""
        monitor = ResourceMonitor()

        # Create decreasing history
        history = list(range(30, 10, -1))  # 20 values, decreasing

        trend = monitor._calculate_trend(history)

        assert trend == "decreasing"

    def test_calculate_trend_stable(self):
        """Verify trend is stable when values are constant."""
        monitor = ResourceMonitor()

        # Create stable history
        history = [50.0] * 20  # 20 constant values

        trend = monitor._calculate_trend(history)

        assert trend == "stable"

    def test_calculate_trend_insufficient_data(self):
        """Verify trend is stable with insufficient data."""
        monitor = ResourceMonitor()

        # Create small history
        history = [50.0, 55.0]  # Only 2 values

        trend = monitor._calculate_trend(history)

        assert trend == "stable"


class TestTriggerAlert:
    """Test alert triggering functionality."""

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_trigger_alert_calls_callback(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify alert triggers callback."""
        mock_cpu.return_value = 85.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        callback = MagicMock()
        monitor = ResourceMonitor(alert_callback=callback)

        # Trigger alert by checking 3 times
        monitor.check_resources()
        monitor.check_resources()
        monitor.check_resources()

        # Verify callback was called
        assert callback.call_count > 0

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_trigger_alert_with_cpu_message(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify alert message format for CPU."""
        mock_cpu.return_value = 85.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        callback = MagicMock()
        monitor = ResourceMonitor(alert_callback=callback)

        # Trigger alert
        monitor.check_resources()
        monitor.check_resources()
        monitor.check_resources()

        # Get last call arguments
        call_args = callback.call_args_list[-1]
        resource_type = call_args[0][0]
        message = call_args[0][1]
        value = call_args[0][2]

        assert resource_type == "cpu"
        assert "85.0%" in message
        assert value == 85.0

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_trigger_alert_with_memory_message(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify alert message format for memory."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=85.0)
        mock_disk.return_value = MagicMock(percent=60.0)

        callback = MagicMock()
        monitor = ResourceMonitor(alert_callback=callback)

        # Trigger alert
        monitor.check_resources()
        monitor.check_resources()
        monitor.check_resources()

        # Get last call arguments
        call_args = callback.call_args_list[-1]
        resource_type = call_args[0][0]
        message = call_args[0][1]

        assert resource_type == "memory"
        assert "85.0%" in message

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_trigger_alert_with_disk_message(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify alert message format for disk."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(
            percent=85.0,
            free=75 * (1024**3),  # 75 GB
            total=500 * (1024**3)   # 500 GB
        )

        callback = MagicMock()
        monitor = ResourceMonitor(alert_callback=callback)

        # Trigger alert
        _ = monitor.check_resources()

        # Get call arguments
        call_args = callback.call_args_list[-1]
        resource_type = call_args[0][0]
        message = call_args[0][1]

        assert resource_type == "disk"
        assert "85.0%" in message
        assert "75.0" in message or "75 GB" in message


class TestPerformance:
    """Test performance requirements."""

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_check_resources_performance_under_50ms(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify that resource checking overhead is < 50ms."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=50.0)

        monitor = ResourceMonitor()

        import time

        # Measure time to check resources
        start = time.perf_counter()
        _ = monitor.check_resources()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be < 50ms
        assert elapsed_ms < 50.0


class TestLogMetrics:
    """Test metrics logging functionality."""

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_log_metrics_with_audit_trail(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify logging to audit trail when configured."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(
            rss=1024 * 1024 * 512,
            percent=50.0
        )
        mock_disk.return_value = MagicMock(
            percent=50.0,
            free=250 * (1024**3),
            total=500 * (1024**3)
        )

        audit_trail = MagicMock()
        monitor = ResourceMonitor(audit_trail=audit_trail)

        _ = monitor.check_resources()

        # Verify log_action was called
        audit_trail.log_action.assert_called_once()

        # Check call arguments
        call_args = audit_trail.log_action.call_args
        assert call_args[0][0] == "RESOURCE_CHECK"
        assert call_args[0][1] == "resource_monitor"
        assert call_args[0][2] == "system"

    @patch('src.monitoring.resource_monitor.psutil.cpu_percent')
    @patch('src.monitoring.resource_monitor.psutil.virtual_memory')
    @patch('src.monitoring.resource_monitor.psutil.disk_usage')
    def test_log_metrics_without_audit_trail(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Verify no error when audit trail not configured."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=50.0)

        monitor = ResourceMonitor(audit_trail=None)

        # Should not raise exception
        metrics = monitor.check_resources()

        assert metrics is not None
        _ = metrics  # Mark as intentionally used for validation

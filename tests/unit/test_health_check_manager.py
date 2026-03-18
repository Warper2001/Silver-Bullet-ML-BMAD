"""Unit tests for Health Check Manager.

Tests health check registration, system health status,
component checking, critical vs non-critical handling,
response time tracking, and CSV logging.
"""

import tempfile
from pathlib import Path
import pytest

from src.monitoring.health_check_manager import (
    HealthCheckManager,
    check_system_resources
)


class TestHealthCheckManagerInit:
    """Test HealthCheckManager initialization."""

    def test_init_with_default_parameters(self):
        """Verify manager initializes with defaults."""
        manager = HealthCheckManager()

        assert manager._check_interval_seconds == 60
        assert len(manager._health_checks) == 0

    def test_init_with_custom_interval(self):
        """Verify manager initializes with custom interval."""
        manager = HealthCheckManager(check_interval_seconds=120)

        assert manager._check_interval_seconds == 120

    def test_init_with_audit_trail(self):
        """Verify manager initializes with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "health_checks.csv")

        manager = HealthCheckManager(audit_trail_path=audit_path)

        assert manager._audit_trail_path == audit_path


class TestRegisterCheck:
    """Test health check registration."""

    @pytest.fixture
    def manager(self):
        """Create health check manager."""
        return HealthCheckManager()

    def test_register_check_adds_component(self, manager):
        """Verify registration adds component."""
        def dummy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("test_component", dummy_check)

        assert "test_component" in manager._health_checks

    def test_register_check_stores_function(self, manager):
        """Verify registration stores check function."""
        def dummy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("test_component", dummy_check)

        assert (
            manager._health_checks["test_component"]["function"]
            == dummy_check
        )

    def test_register_check_default_critical(self, manager):
        """Verify default critical flag is True."""
        def dummy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("test_component", dummy_check)

        assert manager._health_checks["test_component"]["critical"] is True

    def test_register_check_non_critical(self, manager):
        """Verify non-critical can be set."""
        def dummy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check(
            "test_component",
            dummy_check,
            critical=False
        )

        assert manager._health_checks["test_component"]["critical"] is False


class TestGetSystemHealth:
    """Test system health status retrieval."""

    @pytest.fixture
    def manager(self):
        """Create health check manager."""
        manager = HealthCheckManager()

        # Register healthy components
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("component1", healthy_check, critical=True)
        manager.register_check("component2", healthy_check, critical=False)

        return manager

    def test_get_system_health_returns_all_fields(self, manager):
        """Verify system health returns all required fields."""
        status = manager.get_system_health()

        assert 'overall_status' in status
        assert 'timestamp' in status
        assert 'component_count' in status
        assert 'healthy_count' in status
        assert 'degraded_count' in status
        assert 'unhealthy_count' in status
        assert 'components' in status

    def test_get_system_health_all_healthy(self, manager):
        """Verify status when all components healthy."""
        status = manager.get_system_health()

        assert status['overall_status'] == "HEALTHY"
        assert status['healthy_count'] == 2
        assert status['degraded_count'] == 0
        assert status['unhealthy_count'] == 0

    def test_get_system_health_with_degraded(self, manager):
        """Verify status with degraded component."""
        def degraded_check():
            return {"status": "DEGRADED", "message": "Slowing down"}

        manager.register_check("component3", degraded_check, critical=False)

        status = manager.get_system_health()

        assert status['overall_status'] == "DEGRADED"
        assert status['degraded_count'] == 1

    def test_get_system_health_critical_unhealthy(self, manager):
        """Verify status when critical component unhealthy."""
        def unhealthy_check():
            return {"status": "UNHEALTHY", "message": "Failed"}

        manager.register_check("critical_component", unhealthy_check, critical=True)

        status = manager.get_system_health()

        assert status['overall_status'] == "UNHEALTHY"

    def test_get_system_health_non_critical_unhealthy(self, manager):
        """Verify status when non-critical unhealthy."""
        def unhealthy_check():
            return {"status": "UNHEALTHY", "message": "Failed"}

        manager.register_check("non_critical", unhealthy_check, critical=False)

        status = manager.get_system_health()

        assert status['overall_status'] == "DEGRADED"


class TestCheckComponent:
    """Test individual component checking."""

    @pytest.fixture
    def manager(self):
        """Create health check manager."""
        return HealthCheckManager()

    def test_check_component_healthy(self, manager):
        """Verify check returns healthy status."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("test", healthy_check)

        result = manager.check_component("test")

        assert result['status'] == "HEALTHY"
        assert result['message'] == "OK"
        assert 'timestamp' in result
        assert 'response_time_ms' in result

    def test_check_component_degraded(self, manager):
        """Verify check returns degraded status."""
        def degraded_check():
            return {"status": "DEGRADED", "message": "Warning"}

        manager.register_check("test", degraded_check)

        result = manager.check_component("test")

        assert result['status'] == "DEGRADED"

    def test_check_component_unhealthy(self, manager):
        """Verify check returns unhealthy status."""
        def unhealthy_check():
            return {"status": "UNHEALTHY", "message": "Failed"}

        manager.register_check("test", unhealthy_check)

        result = manager.check_component("test")

        assert result['status'] == "UNHEALTHY"

    def test_check_component_not_registered(self, manager):
        """Verify check handles unregistered component."""
        result = manager.check_component("nonexistent")

        assert result['status'] == "UNHEALTHY"
        assert "not registered" in result['message']

    def test_check_component_exception_handling(self, manager):
        """Verify check handles exceptions gracefully."""
        def failing_check():
            raise ValueError("Check failed")

        manager.register_check("failing", failing_check)

        result = manager.check_component("failing")

        assert result['status'] == "UNHEALTHY"
        assert "exception" in result['message'].lower()

    def test_check_component_invalid_result(self, manager):
        """Verify check handles invalid result."""
        def invalid_check():
            return {"message": "Missing status"}  # No status field

        manager.register_check("invalid", invalid_check)

        result = manager.check_component("invalid")

        assert result['status'] == "UNHEALTHY"

    def test_check_component_invalid_status(self, manager):
        """Verify check handles invalid status value."""
        def invalid_check():
            return {"status": "INVALID", "message": "Bad status"}

        manager.register_check("invalid", invalid_check)

        result = manager.check_component("invalid")

        assert result['status'] == "UNHEALTHY"

    def test_check_component_tracks_response_time(self, manager):
        """Verify check tracks response time."""
        import time

        def slow_check():
            time.sleep(0.01)  # 10ms
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("slow", slow_check)

        result = manager.check_component("slow")

        assert result['response_time_ms'] >= 10
        assert 'response_time_ms' in result

    def test_check_component_interval_enforcement(self, manager):
        """Verify check interval is enforced."""
        def fast_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager = HealthCheckManager(check_interval_seconds=60)
        manager.register_check("fast", fast_check)

        # First check
        result1 = manager.check_component("fast")
        assert result1['status'] == "HEALTHY"

        # Immediate second check should be rate limited
        result2 = manager.check_component("fast")
        assert result2['status'] == "UNHEALTHY"
        assert "too recent" in result2['message']


class TestIsSystemHealthy:
    """Test system health boolean check."""

    @pytest.fixture
    def manager(self):
        """Create health check manager."""
        manager = HealthCheckManager()

        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("component", healthy_check)

        return manager

    def test_is_system_healthy_when_healthy(self, manager):
        """Verify returns True when system healthy."""
        assert manager.is_system_healthy() is True

    def test_is_system_healthy_when_degraded(self, manager):
        """Verify returns False when system degraded."""
        def degraded_check():
            return {"status": "DEGRADED", "message": "Warning"}

        manager.register_check("degraded", degraded_check, critical=False)

        assert manager.is_system_healthy() is False

    def test_is_system_healthy_when_unhealthy(self, manager):
        """Verify returns False when system unhealthy."""
        def unhealthy_check():
            return {"status": "UNHEALTHY", "message": "Failed"}

        manager.register_check("unhealthy", unhealthy_check, critical=True)

        assert manager.is_system_healthy() is False


class TestGetUnhealthyComponents:
    """Test unhealthy components list."""

    @pytest.fixture
    def manager(self):
        """Create health check manager."""
        return HealthCheckManager()

    def test_get_unhealthy_empty_when_all_healthy(self, manager):
        """Verify empty list when all components healthy."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("comp1", healthy_check)
        manager.register_check("comp2", healthy_check)

        unhealthy = manager.get_unhealthy_components()

        assert len(unhealthy) == 0

    def test_get_unhealthy_includes_degraded(self, manager):
        """Verify list includes degraded components."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        def degraded_check():
            return {"status": "DEGRADED", "message": "Warning"}

        manager.register_check("healthy", healthy_check)
        manager.register_check("degraded", degraded_check)

        unhealthy = manager.get_unhealthy_components()

        assert "degraded" in unhealthy

    def test_get_unhealthy_includes_unhealthy(self, manager):
        """Verify list includes unhealthy components."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        def unhealthy_check():
            return {"status": "UNHEALTHY", "message": "Failed"}

        manager.register_check("healthy", healthy_check)
        manager.register_check("unhealthy", unhealthy_check)

        unhealthy = manager.get_unhealthy_components()

        assert "unhealthy" in unhealthy


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def manager(self):
        """Create manager with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "health_checks.csv")

        return HealthCheckManager(audit_trail_path=audit_path)

    def test_csv_file_created_on_check(self, manager):
        """Verify CSV file created on health check."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("test", healthy_check)
        manager.check_component("test")

        assert Path(manager._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, manager):
        """Verify CSV has all required columns."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "OK"}

        manager.register_check("test", healthy_check)
        manager.check_component("test")

        import csv
        with open(manager._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "component_name",
            "status",
            "message",
            "response_time_ms",
            "critical"
        ]

        assert headers == expected_headers

    def test_csv_logs_health_check_result(self, manager):
        """Verify CSV logs health check result."""
        def healthy_check():
            return {"status": "HEALTHY", "message": "All good"}

        manager.register_check("test", healthy_check, critical=True)
        manager.check_component("test")

        import csv
        with open(manager._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["component_name"] == "test"
        assert rows[0]["status"] == "HEALTHY"
        assert rows[0]["critical"] == "True"

    def test_csv_logs_error_result(self, manager):
        """Verify CSV logs error results."""
        def failing_check():
            raise RuntimeError("Check error")

        manager.register_check("failing", failing_check, critical=False)
        manager.check_component("failing")

        import csv
        with open(manager._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["status"] == "UNHEALTHY"
        assert "exception" in rows[0]["message"].lower()


class TestSystemResourcesCheck:
    """Test built-in system resources health check."""

    def test_system_resources_check_returns_dict(self):
        """Verify system resources check returns dict."""
        result = check_system_resources()

        assert isinstance(result, dict)
        assert 'status' in result
        assert 'message' in result
        assert 'metrics' in result

    def test_system_resources_check_valid_status(self):
        """Verify system resources check returns valid status."""
        result = check_system_resources()

        assert result['status'] in [
            HealthCheckManager.STATUS_HEALTHY,
            HealthCheckManager.STATUS_DEGRADED,
            HealthCheckManager.STATUS_UNHEALTHY
        ]

    def test_system_resources_check_includes_metrics(self):
        """Verify system resources check includes metrics."""
        result = check_system_resources()

        assert 'metrics' in result
        assert isinstance(result['metrics'], dict)

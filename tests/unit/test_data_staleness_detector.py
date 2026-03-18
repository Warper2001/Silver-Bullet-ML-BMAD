"""Unit tests for Data Staleness Detector.

Tests staleness detection, reconnection logic, safe mode activation,
statistics tracking, and performance requirements.
"""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.monitoring.data_staleness_detector import DataStalenessDetector


class TestDataStalenessDetectorInit:
    """Test DataStalenessDetector initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        detector = DataStalenessDetector()

        assert detector._staleness_threshold == 30
        assert detector._alert_callback is None
        assert detector._audit_trail is None
        assert detector._staleness_state == "FRESH"
        assert detector._reconnect_attempts == 0
        assert detector._safe_mode_activated is False

    def test_init_with_custom_threshold(self):
        """Verify initialization with custom staleness threshold."""
        detector = DataStalenessDetector(staleness_threshold_seconds=60)

        assert detector._staleness_threshold == 60

    def test_init_with_custom_reconnect_delays(self):
        """Verify initialization with custom reconnect delays."""
        delays = [2, 4, 8]
        detector = DataStalenessDetector(reconnect_delays=delays)

        assert detector._reconnect_delays == delays

    def test_init_with_alert_callback(self):
        """Verify initialization with alert callback."""
        callback = MagicMock()
        detector = DataStalenessDetector(alert_callback=callback)

        assert detector._alert_callback == callback

    def test_init_with_audit_trail(self):
        """Verify initialization with audit trail."""
        audit_trail = MagicMock()
        detector = DataStalenessDetector(audit_trail=audit_trail)

        assert detector._audit_trail == audit_trail

    def test_init_initializes_tracking_variables(self):
        """Verify initialization of tracking variables."""
        detector = DataStalenessDetector()

        assert detector._last_data_timestamp is None
        assert detector._staleness_events == 0
        assert detector._total_stale_time_seconds == 0
        assert detector._staleness_start_time is None
        assert len(detector._staleness_history) == 0


class TestUpdateLastDataTimestamp:
    """Test update_last_data_timestamp functionality."""

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_update_last_data_timestamp_updates_timestamp(self, mock_time):
        """Verify update_last_data_timestamp sets timestamp."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        detector.update_last_data_timestamp()

        expected = datetime(2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc)
        assert detector._last_data_timestamp == expected

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_update_last_data_timestamp_multiple_calls(self, mock_time):
        """Verify multiple calls update timestamp correctly."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        detector.update_last_data_timestamp()
        first_timestamp = detector._last_data_timestamp

        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 35, tzinfo=timezone.utc
        )
        detector.update_last_data_timestamp()

        expected = datetime(2026, 3, 18, 10, 15, 35, tzinfo=timezone.utc)
        assert detector._last_data_timestamp == expected
        assert detector._last_data_timestamp != first_timestamp


class TestCheckStaleness:
    """Test check_staleness functionality."""

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_check_staleness_no_data_yet_returns_fresh(self, mock_time):
        """Verify check_staleness returns fresh when no data received."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        result = detector.check_staleness()

        assert result["state"] == "FRESH"
        assert result["last_data_timestamp"] is None
        assert result["staleness_duration_seconds"] == 0

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_check_staleness_recent_data_returns_fresh(self, mock_time):
        """Verify check_staleness returns fresh when data recent."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        ts = datetime(2026, 3, 18, 10, 15, 25, tzinfo=timezone.utc)
        detector._last_data_timestamp = ts

        result = detector.check_staleness()

        assert result["state"] == "FRESH"
        assert result["staleness_duration_seconds"] == 0

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_check_staleness_stale_data_detected(self, mock_time):
        """Verify check_staleness detects stale data after 30 seconds."""
        detector = DataStalenessDetector()

        # Set last data timestamp to 31 seconds ago
        current_time = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )
        detector._last_data_timestamp = current_time - timedelta(seconds=31)

        mock_time.return_value = current_time
        result = detector.check_staleness()

        assert result["state"] == "STALE"
        assert result["staleness_duration_seconds"] == 31

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_check_staleness_triggers_alert_on_staleness(self, mock_time):
        """Verify check_staleness triggers alert when stale."""
        callback = MagicMock()
        detector = DataStalenessDetector(alert_callback=callback)

        # Set last data timestamp to 31 seconds ago
        current_time = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )
        detector._last_data_timestamp = current_time - timedelta(seconds=31)

        mock_time.return_value = current_time
        detector.check_staleness()

        # Verify alert was triggered
        assert callback.call_count > 0

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_check_staleness_logs_to_audit_trail(self, mock_time):
        """Verify check_staleness logs to audit trail."""
        audit_trail = MagicMock()
        detector = DataStalenessDetector(audit_trail=audit_trail)

        # Set last data timestamp to 31 seconds ago
        current_time = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )
        detector._last_data_timestamp = current_time - timedelta(seconds=31)

        mock_time.return_value = current_time
        detector.check_staleness()

        # Verify audit trail was called
        audit_trail.log_action.assert_called()


class TestGetStalenessStatistics:
    """Test get_staleness_statistics functionality."""

    def test_get_staleness_statistics_returns_metrics(self):
        """Verify get_staleness_statistics returns all metrics."""
        detector = DataStalenessDetector()
        detector._staleness_events = 5
        detector._total_stale_time_seconds = 150

        stats = detector.get_staleness_statistics()

        assert "total_staleness_events" in stats
        assert "average_staleness_duration_seconds" in stats
        assert "current_state" in stats
        assert "safe_mode_activated" in stats
        assert "last_data_timestamp" in stats
        assert "recent_events" in stats

    def test_get_staleness_statistics_calculates_average(self):
        """Verify get_staleness_statistics calculates average correctly."""
        detector = DataStalenessDetector()
        detector._staleness_events = 3
        detector._total_stale_time_seconds = 90

        stats = detector.get_staleness_statistics()

        assert stats["average_staleness_duration_seconds"] == 30

    def test_get_staleness_statistics_no_events(self):
        """Verify get_staleness_statistics handles no events."""
        detector = DataStalenessDetector()
        detector._staleness_events = 0
        detector._total_stale_time_seconds = 0

        stats = detector.get_staleness_statistics()

        assert stats["average_staleness_duration_seconds"] == 0


class TestIsInSafeMode:
    """Test is_in_safe_mode functionality."""

    def test_is_in_safe_mode_initially_false(self):
        """Verify is_in_safe_mode returns False initially."""
        detector = DataStalenessDetector()

        assert detector.is_in_safe_mode() is False

    def test_is_in_safe_mode_returns_true_when_activated(self):
        """Verify is_in_safe_mode returns True when activated."""
        detector = DataStalenessDetector()
        detector._safe_mode_activated = True

        assert detector.is_in_safe_mode() is True


class TestExitSafeMode:
    """Test exit_safe_mode functionality."""

    def test_exit_safe_mode_resets_safe_mode_flag(self):
        """Verify exit_safe_mode resets safe mode flag."""
        detector = DataStalenessDetector()
        detector._safe_mode_activated = True
        detector._staleness_state = "SAFE_MODE"

        detector.exit_safe_mode()

        assert detector.is_in_safe_mode() is False
        assert detector._staleness_state == "FRESH"

    def test_exit_safe_mode_logs_to_audit_trail(self):
        """Verify exit_safe_mode logs to audit trail."""
        audit_trail = MagicMock()
        detector = DataStalenessDetector(audit_trail=audit_trail)
        detector._safe_mode_activated = True

        detector.exit_safe_mode()

        audit_trail.log_action.assert_called_once()


class TestHandleRecovery:
    """Test _handle_recovery functionality."""

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_handle_recovery_resets_state(self, mock_time):
        """Verify _handle_recovery resets staleness state."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        detector._staleness_state = "STALE"
        ts = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
        detector._staleness_start_time = ts
        detector._reconnect_attempts = 2

        detector._handle_recovery()

        assert detector._staleness_state == "FRESH"
        assert detector._staleness_start_time is None
        assert detector._reconnect_attempts == 0

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_handle_recovery_updates_statistics(self, mock_time):
        """Verify _handle_recovery updates staleness statistics."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        detector._staleness_state = "STALE"
        detector._staleness_events = 1
        ts = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
        detector._staleness_start_time = ts

        detector._handle_recovery()

        assert detector._total_stale_time_seconds == 30

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_handle_recovery_logs_to_audit_trail(self, mock_time):
        """Verify _handle_recovery logs recovery to audit trail."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        audit_trail = MagicMock()
        detector = DataStalenessDetector(audit_trail=audit_trail)
        detector._staleness_state = "STALE"
        ts = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
        detector._staleness_start_time = ts

        detector._handle_recovery()

        audit_trail.log_action.assert_called()


class TestEnterSafeMode:
    """Test _enter_safe_mode functionality."""

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_enter_safe_mode_activates_safe_mode(self, mock_time):
        """Verify _enter_safe_mode activates safe mode."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        ts = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
        detector._staleness_start_time = ts

        detector._enter_safe_mode()

        assert detector._safe_mode_activated is True
        assert detector._staleness_state == "SAFE_MODE"

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_enter_safe_mode_triggers_alert(self, mock_time):
        """Verify _enter_safe_mode triggers alert."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        callback = MagicMock()
        detector = DataStalenessDetector(alert_callback=callback)
        ts = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
        detector._staleness_start_time = ts

        detector._enter_safe_mode()

        # Verify alert was triggered
        assert callback.call_count > 0

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_enter_safe_mode_logs_to_audit_trail(self, mock_time):
        """Verify _enter_safe_mode logs to audit trail."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        audit_trail = MagicMock()
        detector = DataStalenessDetector(audit_trail=audit_trail)
        ts = datetime(2026, 3, 18, 10, 15, 0, tzinfo=timezone.utc)
        detector._staleness_start_time = ts

        detector._enter_safe_mode()

        audit_trail.log_action.assert_called()


class TestPerformance:
    """Test performance requirements."""

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_update_last_data_timestamp_performance_under_5ms(
        self, mock_time
    ):
        """Verify that update_last_data_timestamp overhead is < 5ms."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()

        import time

        # Measure time to update timestamp
        start = time.perf_counter()
        for _ in range(100):
            detector.update_last_data_timestamp()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        # Should be < 5ms per update
        assert elapsed_ms < 5.0

    @patch(
        "src.monitoring.data_staleness_detector.DataStalenessDetector._get_current_time"
    )
    def test_check_staleness_performance_under_5ms(self, mock_time):
        """Verify that check_staleness overhead is < 5ms."""
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        detector = DataStalenessDetector()
        ts = datetime(2026, 3, 18, 10, 15, 25, tzinfo=timezone.utc)
        detector._last_data_timestamp = ts

        import time

        # Measure time to check staleness
        start = time.perf_counter()
        for _ in range(100):
            detector.check_staleness()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        # Should be < 5ms per check
        assert elapsed_ms < 5.0

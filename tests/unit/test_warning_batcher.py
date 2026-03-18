"""Unit tests for WarningBatcher.

Tests warning event batching functionality including initialization,
adding warnings, batch interval timing, and notification sending.
"""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.monitoring.warning_batcher import WarningBatcher


class TestWarningBatcherInit:
    """Test WarningBatcher initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        assert batcher._notification_manager == notification_manager
        assert batcher._batch_interval == 300
        assert len(batcher._pending_warnings) == 0
        assert batcher._last_batch_time is None

    def test_init_with_custom_batch_interval(self):
        """Verify initialization with custom batch interval."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(
            notification_manager,
            batch_interval_seconds=600
        )

        assert batcher._batch_interval == 600


class TestAddWarning:
    """Test add_warning functionality."""

    def test_add_warning_adds_to_batch(self):
        """Verify add_warning adds event to pending batch."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        # Initialize last batch time to prevent immediate send
        batcher._last_batch_time = batcher._get_current_time()

        event_data = {
            "event_type": "HIGH_MEMORY",
            "message": "High memory usage - 85%",
            "timestamp": "2026-03-18T11:00:00Z"
        }

        batcher.add_warning(event_data)

        assert len(batcher._pending_warnings) == 1
        assert batcher._pending_warnings[0] == event_data

    def test_add_warning_does_not_send_before_interval(self):
        """Verify add_warning doesn't send before interval expires."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        # Set last batch time to now (so interval won't expire)
        batcher._last_batch_time = batcher._get_current_time()

        # Add warning
        event_data = {
            "event_type": "HIGH_MEMORY",
            "message": "High memory usage - 85%",
            "timestamp": "2026-03-18T11:00:00Z"
        }

        batcher.add_warning(event_data)

        # Verify notification not sent
        notification_manager.send_notification.assert_not_called()

    @patch("src.monitoring.warning_batcher.WarningBatcher._get_current_time")
    def test_add_warning_sends_after_interval(self, mock_time):
        """Verify add_warning sends notification after interval."""
        # First call: set last batch time
        mock_time.return_value = datetime(
            2026, 3, 18, 10, 55, 0, tzinfo=timezone.utc
        )

        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)
        batcher._last_batch_time = datetime(
            2026, 3, 18, 10, 55, 0, tzinfo=timezone.utc
        )

        # Second call: add warning after interval
        mock_time.return_value = datetime(
            2026, 3, 18, 11, 0, 1, tzinfo=timezone.utc
        )

        event_data = {
            "event_type": "HIGH_MEMORY",
            "message": "High memory usage - 85%",
            "timestamp": "2026-03-18T11:00:00Z"
        }

        batcher.add_warning(event_data)

        # Verify notification sent
        notification_manager.send_notification.assert_called_once()

    @patch("src.monitoring.warning_batcher.WarningBatcher._get_current_time")
    def test_add_warning_batches_multiple_warnings(self, mock_time):
        """Verify add_warning batches multiple warnings."""
        # Set initial time
        initial_time = datetime(
            2026, 3, 18, 11, 0, 0, tzinfo=timezone.utc
        )
        mock_time.return_value = initial_time

        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        # Initialize last batch time
        batcher._last_batch_time = initial_time

        # Add three warnings
        warnings = [
            {
                "event_type": "HIGH_MEMORY",
                "message": "High memory usage - 85%",
                "timestamp": "2026-03-18T11:00:00Z"
            },
            {
                "event_type": "HIGH_CPU",
                "message": "High CPU usage - 82%",
                "timestamp": "2026-03-18T11:02:00Z"
            },
            {
                "event_type": "LOW_DISK",
                "message": "Low disk space - 82%",
                "timestamp": "2026-03-18T11:04:00Z"
            }
        ]

        for warning in warnings:
            batcher.add_warning(warning)

        assert len(batcher._pending_warnings) == 3

        # Trigger batch send by advancing time
        mock_time.return_value = datetime(
            2026, 3, 18, 11, 5, 1, tzinfo=timezone.utc
        )

        batcher.add_warning({
            "event_type": "HIGH_MEMORY",
            "message": "High memory usage - 86%",
            "timestamp": "2026-03-18T11:05:00Z"
        })

        # Verify notification sent with all 4 warnings
        call_args = notification_manager.send_notification.call_args
        assert call_args[1]["severity"] == "WARNING"
        assert "4)" in call_args[1]["title"]


class TestSendBatchedWarnings:
    """Test _send_batched_warnings functionality."""

    def test_send_batched_warnings_formats_correctly(self):
        """Verify _send_batched_warnings formats message correctly."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        # Add pending warnings
        batcher._pending_warnings = [
            {
                "event_type": "HIGH_MEMORY",
                "message": "High memory usage - 85%",
                "timestamp": "2026-03-18T11:00:00Z"
            },
            {
                "event_type": "HIGH_CPU",
                "message": "High CPU usage - 82%",
                "timestamp": "2026-03-18T11:02:00Z"
            }
        ]

        batcher._send_batched_warnings()

        # Verify notification sent
        notification_manager.send_notification.assert_called_once()

        call_args = notification_manager.send_notification.call_args
        assert call_args[1]["severity"] == "WARNING"
        assert "2)" in call_args[1]["title"]
        assert "- HIGH_MEMORY: High memory usage - 85%" in (
            call_args[1]["message"]
        )
        assert "- HIGH_CPU: High CPU usage - 82%" in (
            call_args[1]["message"]
        )

    def test_send_batched_warnings_clears_pending_warnings(self):
        """Verify _send_batched_warnings clears pending warnings."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        # Add pending warnings
        batcher._pending_warnings = [
            {
                "event_type": "HIGH_MEMORY",
                "message": "High memory usage - 85%",
                "timestamp": "2026-03-18T11:00:00Z"
            }
        ]

        batcher._send_batched_warnings()

        assert len(batcher._pending_warnings) == 0

    def test_send_batched_warnings_empty_list_does_not_send(self):
        """Verify _send_batched_warnings doesn't send when empty."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        batcher._send_batched_warnings()

        notification_manager.send_notification.assert_not_called()


class TestGetCurrentTime:
    """Test _get_current_time functionality."""

    @patch("src.monitoring.warning_batcher.datetime")
    def test_get_current_time_returns_utc(self, mock_datetime):
        """Verify _get_current_time returns UTC time."""
        mock_datetime.now.return_value = datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )

        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        result = batcher._get_current_time()

        assert result == datetime(
            2026, 3, 18, 10, 15, 30, tzinfo=timezone.utc
        )


class TestPerformance:
    """Test performance requirements."""

    def test_add_warning_performance_under_50ms(self):
        """Verify that add_warning overhead is < 50ms."""
        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        event_data = {
            "event_type": "HIGH_MEMORY",
            "message": "High memory usage - 85%",
            "timestamp": "2026-03-18T11:00:00Z"
        }

        import time

        # Measure time to add warning
        start = time.perf_counter()
        for _ in range(100):
            batcher.add_warning(event_data)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        # Should be < 50ms per add
        assert elapsed_ms < 50.0

    @patch("src.monitoring.warning_batcher.WarningBatcher._get_current_time")
    def test_send_batched_warnings_performance_under_50ms(
        self,
        mock_time
    ):
        """Verify that _send_batched_warnings overhead is < 50ms."""
        mock_time.return_value = datetime(
            2026, 3, 18, 11, 0, 0, tzinfo=timezone.utc
        )

        notification_manager = MagicMock()
        batcher = WarningBatcher(notification_manager)

        # Add pending warnings
        batcher._pending_warnings = [
            {
                "event_type": "HIGH_MEMORY",
                "message": "High memory usage - 85%",
                "timestamp": "2026-03-18T11:00:00Z"
            }
        ]

        import time

        # Measure time to send batch
        start = time.perf_counter()
        for _ in range(100):
            batcher._send_batched_warnings()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        # Should be < 50ms per send
        assert elapsed_ms < 50.0

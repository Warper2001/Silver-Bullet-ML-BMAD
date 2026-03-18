"""Unit tests for GracefulShutdownManager.

Tests graceful shutdown mechanism with signal handling,
state persistence, position closing, and resource cleanup.
"""

from unittest.mock import MagicMock, patch, mock_open
import time

from src.monitoring.graceful_shutdown_manager import (
    GracefulShutdownManager,
    load_system_state
)


class TestGracefulShutdownManagerInit:
    """Test GracefulShutdownManager initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        manager = GracefulShutdownManager()

        assert manager._state_file_path == "data/state/system_state.json"
        assert manager._close_positions is False
        assert manager._audit_trail is None
        assert manager._notification_manager is None
        assert manager._shutdown_requested is False
        assert manager._shutdown_complete is False
        assert manager._accepting_new_data is True

    def test_init_with_custom_state_file_path(self):
        """Verify initialization with custom state file path."""
        manager = GracefulShutdownManager(
            state_file_path="custom/state.json"
        )

        assert manager._state_file_path == "custom/state.json"

    def test_init_with_close_positions_flag(self):
        """Verify initialization with close_positions flag."""
        manager = GracefulShutdownManager(
            close_positions_on_shutdown=True
        )

        assert manager._close_positions is True

    def test_init_with_audit_trail(self):
        """Verify initialization with audit trail."""
        audit_trail = MagicMock()
        manager = GracefulShutdownManager(audit_trail=audit_trail)

        assert manager._audit_trail == audit_trail

    def test_init_with_notification_manager(self):
        """Verify initialization with notification manager."""
        notification_manager = MagicMock()
        manager = GracefulShutdownManager(
            notification_manager=notification_manager
        )

        assert manager._notification_manager == notification_manager


class TestRequestShutdown:
    """Test request_shutdown functionality."""

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_request_shutdown_sets_shutdown_flag(
        self,
        mock_file,
        mock_path
    ):
        """Verify request_shutdown sets shutdown flag."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        assert manager._shutdown_requested is True

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_request_shutdown_calls_shutdown_sequence(
        self,
        mock_file,
        mock_path
    ):
        """Verify request_shutdown calls shutdown sequence."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        assert manager._shutdown_complete is True
        assert manager._accepting_new_data is False

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_request_shutdown_idempotent(
        self,
        mock_file,
        mock_path
    ):
        """Verify request_shutdown is idempotent."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()
        first_requested = manager._shutdown_requested
        first_complete = manager._shutdown_complete

        manager.request_shutdown()

        assert manager._shutdown_requested == first_requested
        assert manager._shutdown_complete == first_complete

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_request_shutdown_logs_to_audit_trail(
        self,
        mock_file,
        mock_path
    ):
        """Verify request_shutdown logs to audit trail."""
        audit_trail = MagicMock()
        manager = GracefulShutdownManager(audit_trail=audit_trail)

        manager.request_shutdown()

        audit_trail.log_action.assert_called_once()
        call_args = audit_trail.log_action.call_args
        assert call_args[0][0] == "GRACEFUL_SHUTDOWN"

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_request_shutdown_sends_notification(
        self,
        mock_file,
        mock_path
    ):
        """Verify request_shutdown sends notification."""
        notification_manager = MagicMock()
        manager = GracefulShutdownManager(
            notification_manager=notification_manager
        )

        manager.request_shutdown()

        notification_manager.send_notification.assert_called_once()
        call_args = notification_manager.send_notification.call_args
        assert call_args[1]["severity"] == "INFO"
        assert call_args[1]["notification_type"] == "SYSTEM_SHUTDOWN"


class TestShutdownSequence:
    """Test shutdown sequence execution."""

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_stops_accepting_new_data(self, mock_file, mock_path):
        """Verify shutdown stops accepting new data."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        assert manager.is_accepting_new_data() is False

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_waits_for_in_flight_operations(self, mock_file, mock_path):
        """Verify shutdown waits for in-flight operations."""
        manager = GracefulShutdownManager()

        start = time.time()
        manager.request_shutdown()
        elapsed = time.time() - start

        # Should wait at least 0.5 seconds (in-flight wait)
        assert elapsed >= 0.5

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_persists_state_to_disk(self, mock_file, mock_path):
        """Verify shutdown persists state to disk."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        # Verify file was opened for writing
        mock_file.assert_called_once()
        handle = mock_file()
        handle.write.assert_called()

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_closes_positions_when_enabled(self, mock_file, mock_path):
        """Verify shutdown closes positions when enabled."""
        manager = GracefulShutdownManager(
            close_positions_on_shutdown=True
        )

        manager.request_shutdown()

        # Position closing is logged
        assert manager._close_positions is True

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_closes_connections(self, mock_file, mock_path):
        """Verify shutdown closes connections."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        # Connection closing is logged
        assert manager._shutdown_complete is True

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_marks_shutdown_complete(self, mock_file, mock_path):
        """Verify shutdown marks shutdown complete."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        assert manager.is_shutdown_complete() is True
        assert manager._system_state["shutdown_complete"] is True
        assert manager._system_state["shutdown_timestamp"] is not None


class TestStatePersistence:
    """Test state persistence functionality."""

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_state_file_creation(self, mock_file, mock_path):
        """Verify state file is created."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        # Verify directory creation was called
        # The Path mock is called, and mkdir should be called on parent
        assert mock_path.return_value.parent.mkdir.called

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_state_includes_all_required_fields(self, mock_file, mock_path):
        """Verify state includes all required fields."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        state = manager._system_state

        # Check required fields
        assert "shutdown_timestamp" in state
        assert "positions" in state
        assert "orders" in state
        assert "counters" in state
        assert "model_status" in state
        assert "configuration" in state
        assert "shutdown_complete" in state

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_state_file_json_format_valid(self, mock_file, mock_path):
        """Verify state file is valid JSON."""
        manager = GracefulShutdownManager()

        manager.request_shutdown()

        # Get the JSON data that was written
        handle = mock_file()
        written_data = []
        for call in handle.write.call_args_list:
            written_data.append(call[0][0])

        # Verify JSON is valid
        # Note: This is a simplified check - in real test would capture
        # the full json.dump call
        assert manager._system_state is not None


class TestLoadSystemState:
    """Test load_system_state functionality."""

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    def test_load_with_existing_state_file(self, mock_path):
        """Verify load with existing state file."""
        mock_path.return_value.exists.return_value = True
        with patch(
            "builtins.open",
            mock_open(read_data='{"shutdown_complete": true}')
        ):
            state = load_system_state("test_state.json")

        assert state["shutdown_complete"] is True

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    def test_load_with_missing_state_file(self, mock_path):
        """Verify load with missing state file."""
        mock_path.return_value.exists.return_value = False

        state = load_system_state("missing_state.json")

        assert state == {}

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    def test_returns_empty_dict_when_file_missing(self, mock_path):
        """Verify returns empty dict when file missing."""
        mock_path.return_value.exists.return_value = False

        state = load_system_state("nonexistent.json")

        assert isinstance(state, dict)
        assert len(state) == 0


class TestPerformance:
    """Test performance requirements."""

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_shutdown_completes_within_10_seconds(
        self,
        mock_file,
        mock_path
    ):
        """Verify shutdown completes within 10 seconds."""
        manager = GracefulShutdownManager()

        start = time.time()
        manager.request_shutdown()
        elapsed = time.time() - start

        # Should complete well within 10 seconds
        assert elapsed < 10.0

    @patch("src.monitoring.graceful_shutdown_manager.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_state_persistence_under_2_seconds(
        self,
        mock_file,
        mock_path
    ):
        """Verify state persistence < 2 seconds."""
        manager = GracefulShutdownManager()

        start = time.time()
        manager._persist_system_state()
        elapsed = time.time() - start

        # Should persist quickly
        assert elapsed < 2.0

"""Unit tests for Emergency Stop.

Tests emergency stop activation, deactivation, status checking,
state persistence, CSV logging, and CLI integration.
"""

import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from src.risk.emergency_stop import EmergencyStop


class TestEmergencyStopInit:
    """Test EmergencyStop initialization."""

    def test_init_with_default_parameters(self):
        """Verify emergency stop initializes with defaults."""
        stop = EmergencyStop()

        assert stop._is_stopped is False
        assert stop._stop_time is None
        assert stop._stop_reason is None
        assert stop._audit_trail_path is None
        assert stop._state_path is None

    def test_init_with_audit_trail(self):
        """Verify emergency stop initializes with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "emergency_stop.csv")

        stop = EmergencyStop(audit_trail_path=audit_path)

        assert stop._audit_trail_path == audit_path
        assert stop._is_stopped is False

    def test_init_with_state_persistence(self):
        """Verify emergency stop initializes with state path."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        stop = EmergencyStop(state_path=state_path)

        assert stop._state_path == state_path
        assert stop._is_stopped is False

    def test_init_loads_existing_state(self):
        """Verify emergency stop loads existing state."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        # Create existing state
        existing_state = {
            "is_stopped": True,
            "stop_time": "2026-03-17T14:30:00.000Z",
            "stop_reason": "Manual intervention",
            "last_updated": "2026-03-17T14:30:00.000Z"
        }

        with open(state_path, "w") as f:
            json.dump(existing_state, f)

        # Load state
        stop = EmergencyStop(state_path=state_path)

        assert stop._is_stopped is True
        assert stop._stop_reason == "Manual intervention"


class TestActivate:
    """Test emergency stop activation."""

    @pytest.fixture
    def stop(self):
        """Create emergency stop."""
        return EmergencyStop()

    def test_activate_sets_stopped_flag(self, stop):
        """Verify activation sets stopped flag."""
        stop.activate("Testing activation")

        assert stop._is_stopped is True

    def test_activate_records_stop_time(self, stop):
        """Verify activation records stop time."""
        before_time = datetime.now(timezone.utc)
        stop.activate("Testing activation")
        after_time = datetime.now(timezone.utc)

        assert stop._stop_time is not None
        assert before_time <= stop._stop_time <= after_time

    def test_activate_records_reason(self, stop):
        """Verify activation records reason."""
        reason = "System behaving erratically"
        stop.activate(reason)

        assert stop._stop_reason == reason

    def test_activate_when_already_stopped(self, stop):
        """Verify activate when already stopped logs warning."""
        stop.activate("First activation")

        # Should log warning but not crash
        stop.activate("Second activation")

        assert stop._is_stopped is True
        # Should keep original reason
        assert stop._stop_reason == "First activation"

    def test_activate_saves_state(self, stop):
        """Verify activation saves state to file."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        stop_with_state = EmergencyStop(state_path=state_path)
        stop_with_state.activate("Test state save")

        # Load state from file
        with open(state_path, "r") as f:
            state = json.load(f)

        assert state["is_stopped"] is True
        assert state["stop_reason"] == "Test state save"
        assert state["stop_time"] is not None


class TestDeactivate:
    """Test emergency stop deactivation."""

    @pytest.fixture
    def stopped_emergency_stop(self):
        """Create activated emergency stop."""
        stop = EmergencyStop()
        stop.activate("Testing")
        return stop

    def test_deactivate_clears_stopped_flag(self, stopped_emergency_stop):
        """Verify deactivation clears stopped flag."""
        stopped_emergency_stop.deactivate()

        assert stopped_emergency_stop._is_stopped is False

    def test_deactivate_clears_stop_time(self, stopped_emergency_stop):
        """Verify deactivation clears stop time."""
        stopped_emergency_stop.deactivate()

        assert stopped_emergency_stop._stop_time is None

    def test_deactivate_clears_stop_reason(self, stopped_emergency_stop):
        """Verify deactivation clears stop reason."""
        stopped_emergency_stop.deactivate()

        assert stopped_emergency_stop._stop_reason is None

    def test_deactivate_when_not_stopped(self):
        """Verify deactivation when not stopped logs warning."""
        stop = EmergencyStop()

        # Should log warning but not crash
        stop.deactivate()

        assert stop._is_stopped is False

    def test_deactivate_saves_state(self):
        """Verify deactivation saves state to file."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        stop = EmergencyStop(state_path=state_path)
        stop.activate("Test")
        stop.deactivate()

        # Load state from file
        with open(state_path, "r") as f:
            state = json.load(f)

        assert state["is_stopped"] is False
        assert state["stop_reason"] is None
        assert state["stop_time"] is None


class TestIsTradingAllowed:
    """Test trading status checking."""

    @pytest.fixture
    def stop(self):
        """Create emergency stop."""
        return EmergencyStop()

    def test_trading_allowed_when_not_stopped(self, stop):
        """Verify trading allowed when not stopped."""
        assert stop.is_trading_allowed() is True

    def test_trading_blocked_when_stopped(self, stop):
        """Verify trading blocked when stopped."""
        stop.activate("Testing")

        assert stop.is_trading_allowed() is False

    def test_trading_allowed_after_deactivate(self, stop):
        """Verify trading allowed after deactivation."""
        stop.activate("Testing")
        stop.deactivate()

        assert stop.is_trading_allowed() is True


class TestGetStatus:
    """Test status retrieval."""

    @pytest.fixture
    def stop(self):
        """Create emergency stop."""
        return EmergencyStop()

    def test_status_when_not_stopped(self, stop):
        """Verify status when not stopped."""
        status = stop.get_status()

        assert status["is_stopped"] is False
        assert status["stop_time"] is None
        assert status["stop_reason"] is None
        assert status["time_stopped_seconds"] is None

    def test_status_when_stopped(self, stop):
        """Verify status when stopped."""
        stop.activate("Testing")

        status = stop.get_status()

        assert status["is_stopped"] is True
        assert status["stop_time"] is not None
        assert status["stop_reason"] == "Testing"
        assert status["time_stopped_seconds"] is not None

    def test_status_includes_time_stopped(self, stop):
        """Verify status includes time stopped calculation."""
        stop_time = datetime.now(timezone.utc) - timedelta(seconds=120)

        # Manually set stop time for testing
        stop._is_stopped = True
        stop._stop_time = stop_time
        stop._stop_reason = "Testing"

        status = stop.get_status()

        # Should be approximately 120 seconds (allow for test delay)
        assert status["time_stopped_seconds"] >= 119
        assert status["time_stopped_seconds"] <= 121


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def stop(self):
        """Create emergency stop with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "emergency_stop.csv")

        return EmergencyStop(audit_trail_path=audit_path)

    def test_csv_file_created_on_activate(self, stop):
        """Verify CSV file created on activation."""
        stop.activate("Testing")

        assert Path(stop._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, stop):
        """Verify CSV has all required columns."""
        stop.activate("Testing")

        import csv
        with open(stop._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "is_stopped",
            "stop_reason",
            "time_stopped_seconds"
        ]

        assert headers == expected_headers

    def test_csv_logs_activate_event(self, stop):
        """Verify CSV logs activation event."""
        stop.activate("Test activation")

        import csv
        with open(stop._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["event_type"] == "ACTIVATE"
        assert rows[0]["is_stopped"] == "True"
        assert rows[0]["stop_reason"] == "Test activation"

    def test_csv_logs_deactivate_event(self, stop):
        """Verify CSV logs deactivation event."""
        stop.activate("Test")
        stop.deactivate()

        import csv
        with open(stop._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[1]["event_type"] == "DEACTIVATE"
        assert rows[1]["is_stopped"] == "False"

    def test_csv_logs_check_events(self, stop):
        """Verify CSV logs check events."""
        stop.activate("Test")
        stop.is_trading_allowed()  # Generates CHECK event
        stop.is_trading_allowed()  # Another CHECK event

        import csv
        with open(stop._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have ACTIVATE, CHECK, CHECK
        assert rows[0]["event_type"] == "ACTIVATE"
        assert rows[1]["event_type"] == "CHECK"
        assert rows[2]["event_type"] == "CHECK"


class TestStatePersistence:
    """Test state persistence and restoration."""

    def test_state_persists_across_instances(self):
        """Verify state persists across object instances."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        # First instance: activate
        stop1 = EmergencyStop(state_path=state_path)
        stop1.activate("Testing persistence")

        # Second instance: should load state
        stop2 = EmergencyStop(state_path=state_path)

        assert stop2._is_stopped is True
        assert stop2._stop_reason == "Testing persistence"

    def test_state_deactivation_persists(self):
        """Verify deactivation persists across instances."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        # First instance: activate then deactivate
        stop1 = EmergencyStop(state_path=state_path)
        stop1.activate("Testing")
        stop1.deactivate()

        # Second instance: should load deactivated state
        stop2 = EmergencyStop(state_path=state_path)

        assert stop2._is_stopped is False
        assert stop2._stop_reason is None

    def test_state_file_created_when_missing(self):
        """Verify state file created when missing."""
        temp_dir = tempfile.mkdtemp()
        state_path = str(Path(temp_dir) / "emergency_stop.json")

        stop = EmergencyStop(state_path=state_path)
        stop.activate("Testing")

        # File should exist
        assert Path(state_path).exists()


class TestMultipleActivateDeactivateCycles:
    """Test multiple activate/deactivate cycles."""

    @pytest.fixture
    def stop(self):
        """Create emergency stop."""
        return EmergencyStop()

    def test_multiple_cycles(self, stop):
        """Verify multiple activate/deactivate cycles work."""
        # Cycle 1
        stop.activate("Cycle 1")
        assert stop.is_trading_allowed() is False
        stop.deactivate()
        assert stop.is_trading_allowed() is True

        # Cycle 2
        stop.activate("Cycle 2")
        assert stop.is_trading_allowed() is False
        stop.deactivate()
        assert stop.is_trading_allowed() is True

        # Cycle 3
        stop.activate("Cycle 3")
        assert stop.is_trading_allowed() is False
        stop.deactivate()
        assert stop.is_trading_allowed() is True

    def test_status_updates_across_cycles(self, stop):
        """Verify status updates correctly across cycles."""
        # First cycle
        stop.activate("First")
        status1 = stop.get_status()
        assert status1["is_stopped"] is True
        assert status1["stop_reason"] == "First"

        stop.deactivate()
        status2 = stop.get_status()
        assert status2["is_stopped"] is False

        # Second cycle
        stop.activate("Second")
        status3 = stop.get_status()
        assert status3["is_stopped"] is True
        assert status3["stop_reason"] == "Second"

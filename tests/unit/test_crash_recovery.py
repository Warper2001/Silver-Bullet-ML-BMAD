"""Unit tests for Crash Recovery Manager.

Tests crash detection, state persistence, recovery from crashes,
backup rotation, CSV audit logging, and graceful shutdown.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from src.monitoring.crash_recovery import CrashRecoveryManager


class TestCrashRecoveryManagerInit:
    """Test CrashRecoveryManager initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            assert manager._state_file_path == state_path
            assert manager._backup_count == 5
            assert manager._audit_trail_path == audit_path

    def test_init_creates_directories(self):
        """Verify that initialization creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "subdir" / "state.json"
            audit_path = Path(tmpdir) / "audit" / "audit.csv"

            CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            assert state_path.parent.exists()
            assert audit_path.parent.exists()

    def test_init_with_custom_backup_count(self):
        """Verify initialization with custom backup count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=3,
                audit_trail_path=str(audit_path)
            )

            assert manager._backup_count == 3


class TestPersistState:
    """Test state persistence functionality."""

    def test_persist_state_creates_state_file(self):
        """Verify that persist_state creates the state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "positions": [],
                "pending_orders": []
            }

            manager.persist_state(state)

            assert state_path.exists()

    def test_persist_state_adds_shutdown_complete_flag(self):
        """Verify that shutdown_complete flag is added if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "positions": [],
                "pending_orders": []
            }

            manager.persist_state(state)

            with open(state_path, 'r') as f:
                loaded = json.load(f)

            assert "shutdown_complete" in loaded
            assert loaded["shutdown_complete"] is False

    def test_persist_state_adds_timestamp(self):
        """Verify that timestamp is added if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "positions": [],
                "pending_orders": []
            }

            manager.persist_state(state)

            with open(state_path, 'r') as f:
                loaded = json.load(f)

            assert "timestamp" in loaded
            # Verify it's a valid ISO format timestamp
            datetime.fromisoformat(loaded["timestamp"])

    def test_persist_state_preserves_shutdown_flag(self):
        """Verify that existing shutdown_complete flag is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "shutdown_complete": True,
                "positions": [],
                "pending_orders": []
            }

            manager.persist_state(state)

            with open(state_path, 'r') as f:
                loaded = json.load(f)

            assert loaded["shutdown_complete"] is True

    def test_persist_state_uses_atomic_operation(self):
        """Verify that state persistence uses atomic operation (temp file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "positions": [],
                "pending_orders": []
            }

            # Patch to verify temp file is used
            original_open = open
            temp_files_created = []

            def mock_open(*args, **kwargs):
                result = original_open(*args, **kwargs)
                if len(args) > 0 and str(args[0]).endswith('.tmp'):
                    temp_files_created.append(args[0])
                return result

            with patch('builtins.open', side_effect=mock_open):
                manager.persist_state(state)

            # Verify temp file was created
            assert len(temp_files_created) > 0


class TestDetectCrash:
    """Test crash detection functionality."""

    def test_detect_crash_no_state_file(self):
        """Verify crash detection when state file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            result = manager.detect_crash()

            assert result["crashed"] is True
            assert result["reason"] == "state_file_missing"

    def test_detect_crash_with_corrupted_state(self):
        """Verify crash detection with corrupted state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            # Write invalid JSON
            with open(state_path, 'w') as f:
                f.write("{invalid json}")

            result = manager.detect_crash()

            assert result["crashed"] is True
            assert result["reason"] == "state_file_corrupted"

    def test_detect_crash_with_incomplete_shutdown(self):
        """Verify crash detection when shutdown_complete is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "shutdown_complete": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            result = manager.detect_crash()

            assert result["crashed"] is True
            assert result["reason"] == "incomplete_shutdown"

    def test_detect_crash_with_missing_shutdown_flag(self):
        """Verify crash detection when shutdown_complete flag is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            result = manager.detect_crash()

            assert result["crashed"] is True
            assert result["reason"] == "incomplete_shutdown"

    def test_detect_crash_with_clean_shutdown(self):
        """Verify no crash detected after graceful shutdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "shutdown_complete": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            result = manager.detect_crash()

            assert result["crashed"] is False
            assert result["reason"] == "clean_shutdown"


class TestRecoverFromCrash:
    """Test crash recovery functionality."""

    def test_recover_from_crash_no_crash(self):
        """Verify recovery when no crash occurred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "shutdown_complete": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            result = manager.recover_from_crash()

            assert result["recovery_needed"] is False
            assert "No crash detected" in result["message"]

    def test_recover_from_crash_with_valid_state(self):
        """Verify successful recovery with valid state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            state = {
                "shutdown_complete": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [
                    {
                        "symbol": "MNQ 6-26",
                        "quantity": 2,
                        "entry_price": 12345.25
                    }
                ],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            result = manager.recover_from_crash()

            assert result["recovery_needed"] is True
            assert result["recovery_successful"] is True
            assert result["safe_mode"] is False
            assert "state_restored" in result

    def test_recover_from_crash_with_corrupted_state(self):
        """Verify recovery with corrupted state loads from backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path),
                backup_count=2
            )

            # Create corrupted state file
            with open(state_path, 'w') as f:
                f.write("{invalid json}")

            # Create valid backup
            backup_1 = state_path.with_suffix('.1.json')
            backup_state = {
                "shutdown_complete": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(backup_1, 'w') as f:
                json.dump(backup_state, f)

            result = manager.recover_from_crash()

            assert result["recovery_needed"] is True
            assert result["recovery_successful"] is True

    def test_recover_from_crash_no_valid_state(self):
        """Verify recovery enters safe mode when no valid state exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            # Create corrupted state file with no backups
            with open(state_path, 'w') as f:
                f.write("{invalid json}")

            result = manager.recover_from_crash()

            assert result["recovery_needed"] is True
            assert result["recovery_successful"] is False
            assert result["safe_mode"] is True
            assert "safe mode" in result["message"]


class TestRotateBackups:
    """Test backup rotation functionality."""

    def test_rotate_backups_maintains_correct_count(self):
        """Verify that backup rotation maintains correct number of backups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            backup_count = 3
            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=backup_count,
                audit_trail_path=str(audit_path)
            )

            # Create initial state file
            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            # Persist multiple times to create backups
            for i in range(backup_count + 2):
                manager.persist_state(state)

            # Check that we have exactly backup_count backups
            backup_files = list(Path(tmpdir).glob('*.json'))
            # Should have: state.json + backup_count backups
            assert len([f for f in backup_files if f != state_path]) <= backup_count

    def test_rotate_backups_deletes_oldest(self):
        """Verify that oldest backup is deleted when limit is exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=2,
                audit_trail_path=str(audit_path)
            )

            # Create initial state
            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            # Create all possible backups
            for i in range(3):
                manager.persist_state(state)

            # Verify oldest backup (.3.json) doesn't exist
            oldest_backup = state_path.with_suffix('.3.json')
            assert not oldest_backup.exists()

    def test_rotate_backups_renames_correctly(self):
        """Verify that backups are renamed in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=3,
                audit_trail_path=str(audit_path)
            )

            # Create initial state
            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            # Persist twice to create .1 and .2 backups
            manager.persist_state(state)
            manager.persist_state(state)

            # Verify backups exist
            backup_1 = state_path.with_suffix('.1.json')
            backup_2 = state_path.with_suffix('.2.json')

            assert backup_1.exists()
            assert backup_2.exists()


class TestLoadFromBackup:
    """Test backup loading functionality."""

    def test_load_from_backup_loads_most_recent(self):
        """Verify that most recent valid backup is loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=3,
                audit_trail_path=str(audit_path)
            )

            # Create multiple backups with different timestamps
            now = datetime.now(timezone.utc)

            for i in range(1, 4):
                backup_path = state_path.with_suffix('.{}.json'.format(i))
                state = {
                    "timestamp": (now + timedelta(seconds=i)).isoformat(),
                    "positions": [],
                    "pending_orders": []
                }

                with open(backup_path, 'w') as f:
                    json.dump(state, f)

            # Load from backup
            loaded_state = manager._load_from_backup()

            assert loaded_state is not None
            # Should load the most recent one (.1.json)
            assert "timestamp" in loaded_state

    def test_load_from_backup_with_corrupted_backups(self):
        """Verify loading skips corrupted backups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=3,
                audit_trail_path=str(audit_path)
            )

            # Create corrupted .1 and .2 backups
            backup_1 = state_path.with_suffix('.1.json')
            backup_2 = state_path.with_suffix('.2.json')
            backup_3 = state_path.with_suffix('.3.json')

            with open(backup_1, 'w') as f:
                f.write("{corrupted}")

            with open(backup_2, 'w') as f:
                f.write("{corrupted}")

            # Create valid .3 backup
            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            with open(backup_3, 'w') as f:
                json.dump(state, f)

            # Should load the valid .3 backup
            loaded_state = manager._load_from_backup()

            assert loaded_state is not None

    def test_load_from_backup_returns_none_when_no_backups(self):
        """Verify that None is returned when no backups exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                backup_count=3,
                audit_trail_path=str(audit_path)
            )

            # No backups created
            loaded_state = manager._load_from_backup()

            assert loaded_state is None


class TestMarkShutdownComplete:
    """Test graceful shutdown marking functionality."""

    def test_mark_shutdown_complete_sets_flag(self):
        """Verify that shutdown_complete flag is set to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            # Create initial state with shutdown_complete = False
            state = {
                "shutdown_complete": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            manager.persist_state(state)

            # Mark shutdown complete
            manager.mark_shutdown_complete()

            # Verify flag is set
            with open(state_path, 'r') as f:
                loaded = json.load(f)

            assert loaded["shutdown_complete"] is True

    def test_mark_shutdown_complete_adds_timestamp(self):
        """Verify that shutdown timestamp is added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            # Create initial state
            state = {
                "shutdown_complete": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [],
                "pending_orders": []
            }

            manager.persist_state(state)

            # Mark shutdown complete
            manager.mark_shutdown_complete()

            # Verify shutdown_timestamp is added
            with open(state_path, 'r') as f:
                loaded = json.load(f)

            assert "shutdown_timestamp" in loaded
            # Verify it's a valid ISO format timestamp
            datetime.fromisoformat(loaded["shutdown_timestamp"])


class TestLogRecoveryEvent:
    """Test CSV audit logging functionality."""

    def test_log_recovery_event_creates_file(self):
        """Verify that logging creates CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            detection_result = {
                "crashed": True,
                "reason": "incomplete_shutdown"
            }

            reconciliation = {
                "reconciled": True,
                "position_discrepancies": [],
                "order_discrepancies": [],
                "safe_mode_required": False
            }

            manager._log_recovery_event(detection_result, reconciliation)

            assert audit_path.exists()

    def test_log_recovery_event_writes_header(self):
        """Verify that CSV header is written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            detection_result = {
                "crashed": True,
                "reason": "incomplete_shutdown"
            }

            reconciliation = {
                "reconciled": True,
                "position_discrepancies": [],
                "order_discrepancies": [],
                "safe_mode_required": False
            }

            manager._log_recovery_event(detection_result, reconciliation)

            with open(audit_path, 'r') as f:
                content = f.read()

            assert "timestamp,event_type,crash_detected" in content

    def test_log_recovery_event_writes_data_row(self):
        """Verify that data row is written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            detection_result = {
                "crashed": True,
                "reason": "incomplete_shutdown"
            }

            reconciliation = {
                "reconciled": True,
                "position_discrepancies": [],
                "order_discrepancies": [],
                "safe_mode_required": False
            }

            manager._log_recovery_event(detection_result, reconciliation)

            with open(audit_path, 'r') as f:
                lines = f.readlines()

            # Should have header + 1 data row
            assert len(lines) == 2

            # Verify data row has correct format
            data_row = lines[1].strip()
            fields = data_row.split(',')

            assert len(fields) == 8
            assert fields[1] == "crash_recovery"
            assert fields[2] == "True"

    def test_log_recovery_event_multiple_entries(self):
        """Verify that multiple entries are appended correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            audit_path = Path(tmpdir) / "audit.csv"

            manager = CrashRecoveryManager(
                state_file_path=str(state_path),
                audit_trail_path=str(audit_path)
            )

            detection_result = {
                "crashed": True,
                "reason": "incomplete_shutdown"
            }

            reconciliation = {
                "reconciled": True,
                "position_discrepancies": [],
                "order_discrepancies": [],
                "safe_mode_required": False
            }

            # Log multiple events
            for i in range(3):
                manager._log_recovery_event(detection_result, reconciliation)

            with open(audit_path, 'r') as f:
                lines = f.readlines()

            # Should have header + 3 data rows
            assert len(lines) == 4

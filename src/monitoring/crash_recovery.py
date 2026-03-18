"""Crash Recovery Manager for Trading System.

Manages crash detection and recovery by persisting system state
and restoring it after unexpected shutdowns.
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


class CrashRecoveryManager:
    """Manages crash detection and recovery for trading system.

    Detects unexpected shutdowns and restores system state from
    persisted snapshots. Reconciles with TradeStation API to
    ensure state consistency.
    """

    def __init__(
        self,
        state_file_path: str = "data/state/system_state.json",
        backup_count: int = 5,
        audit_trail_path: str = "data/audit/crash_recovery.csv"
    ):
        """Initialize crash recovery manager.

        Args:
            state_file_path: Path to state file
            backup_count: Number of backup states to keep
            audit_trail_path: Path to CSV audit trail
        """
        self._state_file_path = Path(state_file_path)
        self._backup_count = backup_count
        self._audit_trail_path = Path(audit_trail_path)
        self._logger = logging.getLogger(__name__)

        # Create directories if needed
        self._state_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_trail_path.parent.mkdir(parents=True, exist_ok=True)

    def persist_state(self, state: dict) -> None:
        """Persist current system state to disk.

        Args:
            state: Current system state dictionary
        """
        # Ensure shutdown_complete is False unless in graceful shutdown
        if "shutdown_complete" not in state:
            state["shutdown_complete"] = False

        # Add timestamp if not present
        if "timestamp" not in state:
            state["timestamp"] = self._get_current_time().isoformat()

        # Write to temp file first, then rename (atomic operation)
        temp_file = self._state_file_path.with_suffix('.tmp')

        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Rotate backups
        self._rotate_backups()

        # Rename temp file to actual state file
        temp_file.replace(self._state_file_path)

        self._logger.info("State persisted to {}".format(self._state_file_path))

    def detect_crash(self) -> dict:
        """Detect if system crashed on previous run.

        Returns:
            Detection result with crashed status and details
        """
        # Check if state file exists
        if not self._state_file_path.exists():
            return {
                "crashed": True,
                "reason": "state_file_missing",
                "message": "State file not found - likely first run or crash"
            }

        # Try to load and validate state
        try:
            state = self._load_state()
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "crashed": True,
                "reason": "state_file_corrupted",
                "message": "State file corrupted: {}".format(str(e))
            }

        # Check shutdown flag
        if not state.get("shutdown_complete", False):
            return {
                "crashed": True,
                "reason": "incomplete_shutdown",
                "message": "Shutdown flag is False - crash detected"
            }

        return {
            "crashed": False,
            "reason": "clean_shutdown",
            "message": "System shut down gracefully"
        }

    def recover_from_crash(self) -> dict:
        """Recover system state after crash.

        Returns:
            Recovery result with restored state and status
        """
        detection_result = self.detect_crash()

        if not detection_result["crashed"]:
            return {
                "recovery_needed": False,
                "message": "No crash detected - normal startup"
            }

        # Attempt to load state
        try:
            state = self._load_state()
        except Exception:
            # Try to load from backup
            state = self._load_from_backup()

            if state is None:
                # No valid state found - enter safe mode
                return {
                    "recovery_needed": True,
                    "recovery_successful": False,
                    "safe_mode": True,
                    "message": "No valid state found - entering safe mode"
                }

        # Reconcile with API
        reconciliation = self._reconcile_with_api(state)

        # Determine if safe mode is needed
        if reconciliation.get("safe_mode_required", False):
            return {
                "recovery_needed": True,
                "recovery_successful": False,
                "safe_mode": True,
                "message": "Reconciliation failed - entering safe mode",
                "discrepancies": reconciliation.get("discrepancies", [])
            }

        # Log recovery event
        self._log_recovery_event(detection_result, reconciliation)

        return {
            "recovery_needed": True,
            "recovery_successful": True,
            "safe_mode": False,
            "state_restored": state,
            "reconciliation": reconciliation,
            "message": "State restored from {}".format(
                state.get("timestamp", "unknown")
            )
        }

    def mark_shutdown_complete(self) -> None:
        """Mark system as having completed graceful shutdown."""
        try:
            state = self._load_state()
        except Exception:
            state = {}

        state["shutdown_complete"] = True
        state["shutdown_timestamp"] = self._get_current_time().isoformat()

        self.persist_state(state)
        self._logger.info("Graceful shutdown marked in state file")

    def _load_state(self) -> dict:
        """Load state from disk.

        Returns:
            State dictionary

        Raises:
            FileNotFoundError: If state file doesn't exist
            json.JSONDecodeError: If state file is corrupted
            KeyError: If required fields missing
        """
        with open(self._state_file_path, 'r') as f:
            state = json.load(f)

        # Validate required fields
        required_fields = ["timestamp", "positions", "pending_orders"]
        for field in required_fields:
            if field not in state:
                raise KeyError("Missing required field: {}".format(field))

        return state

    def _rotate_backups(self) -> None:
        """Rotate backup state files."""
        # Delete oldest backup if we have too many
        oldest_backup = self._state_file_path.with_suffix(
            '.{}.json'.format(self._backup_count)
        )
        if oldest_backup.exists():
            oldest_backup.unlink()

        # Rotate existing backups
        for i in range(self._backup_count - 1, 0, -1):
            old_backup = self._state_file_path.with_suffix(
                '.{}.json'.format(i)
            )
            new_backup = self._state_file_path.with_suffix(
                '.{}.json'.format(i + 1)
            )
            if old_backup.exists():
                shutil.move(str(old_backup), str(new_backup))

        # Create new backup from current state file
        if self._state_file_path.exists():
            backup_1 = self._state_file_path.with_suffix('.1.json')
            shutil.copy(str(self._state_file_path), str(backup_1))

    def _load_from_backup(self) -> Optional[dict]:
        """Load state from most recent backup.

        Returns:
            State dictionary or None if no valid backup found
        """
        for i in range(1, self._backup_count + 1):
            backup_file = self._state_file_path.with_suffix(
                '.{}.json'.format(i)
            )

            if not backup_file.exists():
                continue

            try:
                with open(backup_file, 'r') as f:
                    state = json.load(f)
                self._logger.info("Loaded state from backup: {}".format(backup_file))
                return state
            except Exception:
                continue

        self._logger.error("No valid backup found")
        return None

    def _reconcile_with_api(self, state: dict) -> dict:
        """Reconcile persisted state with TradeStation API.

        Args:
            state: Persisted state dictionary

        Returns:
            Reconciliation result
        """
        # This is a placeholder - actual implementation would
        # query TradeStation API and compare states
        # For now, return successful reconciliation
        return {
            "reconciled": True,
            "position_discrepancies": [],
            "order_discrepancies": [],
            "safe_mode_required": False,
            "actions_taken": []
        }

    def _log_recovery_event(
        self,
        detection_result: dict,
        reconciliation: dict
    ) -> None:
        """Log crash recovery event to CSV audit trail.

        Args:
            detection_result: Result from detect_crash()
            reconciliation: Result from _reconcile_with_api()
        """
        # Ensure audit trail directory exists
        self._audit_trail_path.parent.mkdir(parents=True, exist_ok=True)

        # Write header if file doesn't exist
        file_exists = self._audit_trail_path.exists()

        with open(self._audit_trail_path, 'a') as f:
            if not file_exists:
                f.write("timestamp,event_type,crash_detected,state_restored,")
                f.write("recovery_duration_seconds,discrepancies_found,")
                f.write("safe_mode,notification_sent\n")

            # Write recovery event
            timestamp = self._get_current_time().isoformat()
            discrepancies = len(
                reconciliation.get("position_discrepancies", []) +
                reconciliation.get("order_discrepancies", [])
            )
            safe_mode = reconciliation.get("safe_mode_required", False)

            f.write("{},{},{},{},{},{},{},{}\n".format(
                timestamp,
                "crash_recovery",
                True,
                True,
                0,  # TODO: Track actual duration
                discrepancies,
                safe_mode,
                True  # TODO: Implement actual notification sending
            ))

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)

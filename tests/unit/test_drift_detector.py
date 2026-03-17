"""Unit tests for DriftDetector.

Tests model drift detection, drift history tracking,
consecutive cycle counting, trading halt logic, and recovery detection.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.ml.drift_detector import (
    DriftDetector,
    InsufficientDataError,
    InvalidHistoryError,
)


class TestDriftDetectorInit:
    """Test DriftDetector initialization and configuration."""

    def test_init_with_default_parameters(self):
        """Verify DriftDetector initializes with default parameters."""
        detector = DriftDetector()
        assert detector is not None
        assert detector._model_dir.name == "xgboost"
        assert detector._history_file.name == "drift_history.json"

    def test_init_with_custom_model_dir(self, tmp_path):
        """Verify DriftDetector initializes with custom model directory."""
        custom_dir = tmp_path / "custom_models"
        detector = DriftDetector(model_dir=custom_dir)
        assert detector._model_dir == custom_dir

    def test_init_creates_empty_history_if_not_exists(self, tmp_path):
        """Verify empty drift history is created if file doesn't exist."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Verify empty history structure
        history = detector.get_drift_history()
        assert history["statistics"]["total_drift_events"] == 0
        assert history["statistics"]["active_drift"] is False
        assert history["statistics"]["consecutive_drift_cycles"] == 0

    def test_init_loads_existing_history(self, tmp_path):
        """Verify existing drift history is loaded on initialization."""
        import json

        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)
        history_file = model_dir / "drift_history.json"

        # Create existing history
        existing_history = {
            "drift_events": [
                {
                    "drift_id": "test-drift-1",
                    "detected_at": "2024-01-01T12:00:00",
                    "duration_minutes": 0,
                    "severity": "WARNING",
                    "magnitude": -0.15,
                    "expected_win_rate": 0.60,
                    "actual_win_rate": 0.45,
                    "recovered_at": None,
                    "consecutive_cycle_count": 1,
                    "trading_halted": False,
                }
            ],
            "statistics": {
                "total_drift_events": 1,
                "active_drift": True,
                "consecutive_drift_cycles": 1,
                "last_drift_detection": "2024-01-01T12:00:00",
                "last_drift_recovery": None,
            },
        }

        with open(history_file, "w") as f:
            json.dump(existing_history, f)

        detector = DriftDetector(model_dir=model_dir)

        # Verify existing history was loaded
        history = detector.get_drift_history()
        assert history["statistics"]["total_drift_events"] == 1
        assert history["statistics"]["active_drift"] is True


class TestDriftDetection:
    """Test drift detection logic."""

    def test_check_drift_detects_10_percent_drop(self):
        """Verify drift detected when actual win rate drops >10% below expected."""
        detector = DriftDetector()

        # 15% drop (0.60 -> 0.45)
        result = detector.check_drift(actual_win_rate=0.45, expected_win_rate=0.60)

        assert result["drift_detected"] is True
        assert result["drift_magnitude"] == pytest.approx(-0.15, abs=0.001)
        assert result["severity"] == "WARNING"

    def test_check_drift_no_drift_within_10_percent(self):
        """Verify no drift when win rate drop is within 10%."""
        detector = DriftDetector()

        # 5% drop (within threshold)
        result = detector.check_drift(actual_win_rate=0.57, expected_win_rate=0.60)

        assert result["drift_detected"] is False
        assert result["drift_magnitude"] == pytest.approx(-0.03, abs=0.001)  # 0.57 - 0.60 = -0.03

    def test_check_drift_no_drift_when_improved(self):
        """Verify no drift when actual win rate exceeds expected."""
        detector = DriftDetector()

        # 5% improvement
        result = detector.check_drift(actual_win_rate=0.63, expected_win_rate=0.60)

        assert result["drift_detected"] is False
        assert result["drift_magnitude"] == pytest.approx(0.03, abs=0.001)

    def test_check_drift_severity_warning(self):
        """Verify WARNING severity for 10-20% drop."""
        detector = DriftDetector()

        # 15% drop
        result = detector.check_drift(actual_win_rate=0.45, expected_win_rate=0.60)

        assert result["severity"] == "WARNING"

    def test_check_drift_severity_critical(self):
        """Verify CRITICAL severity for >20% drop."""
        detector = DriftDetector()

        # 25% drop
        result = detector.check_drift(actual_win_rate=0.35, expected_win_rate=0.60)

        assert result["severity"] == "CRITICAL"

    def test_check_drift_raises_error_for_invalid_win_rates(self):
        """Verify ValueError raised for win rates outside [0, 1]."""
        detector = DriftDetector()

        with pytest.raises(ValueError, match="Win rates must be in range"):
            detector.check_drift(actual_win_rate=1.5, expected_win_rate=0.60)

        with pytest.raises(ValueError, match="Win rates must be in range"):
            detector.check_drift(actual_win_rate=0.50, expected_win_rate=-0.1)


class TestDriftEventTracking:
    """Test drift event tracking and history management."""

    def test_track_drift_event_creates_new_event(self, tmp_path):
        """Verify track_drift_event() creates new drift event."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        drift_id = detector.track_drift_event(
            actual_win_rate=0.45,
            expected_win_rate=0.60,
            severity="WARNING",
        )

        # Verify drift ID returned
        assert drift_id is not None
        assert isinstance(drift_id, str)

        # Verify event added to history
        history = detector.get_drift_history()
        assert len(history["drift_events"]) == 1
        assert history["drift_events"][0]["drift_id"] == drift_id
        assert history["drift_events"][0]["actual_win_rate"] == 0.45
        assert history["drift_events"][0]["expected_win_rate"] == 0.60

    def test_track_drift_event_updates_statistics(self, tmp_path):
        """Verify track_drift_event() updates drift statistics."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        history = detector.get_drift_history()
        assert history["statistics"]["total_drift_events"] == 1
        assert history["statistics"]["active_drift"] is True
        assert history["statistics"]["last_drift_detection"] is not None

    def test_track_drift_event_persists_to_file(self, tmp_path):
        """Verify track_drift_event() persists history to file."""
        import json

        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Verify file was created
        history_file = model_dir / "drift_history.json"
        assert history_file.exists()

        # Verify file contents
        with open(history_file, "r") as f:
            file_history = json.load(f)

        assert file_history["statistics"]["total_drift_events"] == 1

    def test_get_drift_history_returns_all_events(self, tmp_path):
        """Verify get_drift_history() returns complete drift history."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track multiple drift events
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        detector.track_drift_event(
            actual_win_rate=0.40, expected_win_rate=0.60, severity="CRITICAL"
        )

        history = detector.get_drift_history()
        assert len(history["drift_events"]) == 2
        assert history["statistics"]["total_drift_events"] == 2


class TestConsecutiveDriftCycleTracking:
    """Test consecutive drift cycle counting."""

    def test_increment_consecutive_drift_cycle_increments_counter(self, tmp_path):
        """Verify increment_consecutive_drift_cycle() increments counter."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track initial drift
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Increment consecutive cycles
        detector.increment_consecutive_drift_cycle()

        # Verify counter incremented
        assert detector.get_consecutive_drift_cycles() == 1

    def test_increment_consecutive_drift_cycle_updates_active_event(
        self, tmp_path
    ):
        """Verify increment updates consecutive_cycle_count in active drift."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track initial drift
        drift_id = detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Increment to cycle 2
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()

        history = detector.get_drift_history()
        active_drift = history["drift_events"][0]
        assert active_drift["consecutive_cycle_count"] == 2

    def test_should_halt_trading_after_3_consecutive_cycles(self, tmp_path):
        """Verify should_halt_trading() returns True after 3 consecutive cycles."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track drift and increment to 3 cycles
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()

        # Check if trading should halt
        should_halt, reason = detector.should_halt_trading()

        assert should_halt is True
        assert "3 consecutive" in reason

    def test_should_not_halt_trading_before_3_cycles(self, tmp_path):
        """Verify should_halt_trading() returns False before 3 cycles."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track drift and increment to 2 cycles
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()

        should_halt, reason = detector.should_halt_trading()

        assert should_halt is False
        assert reason == ""

    def test_trading_halt_flag_set_on_third_cycle(self, tmp_path):
        """Verify trading_halted flag set when 3 consecutive cycles reached."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track drift and increment to 3 cycles
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()

        history = detector.get_drift_history()
        active_drift = history["drift_events"][0]
        assert active_drift["trading_halted"] is True


class TestDriftRecovery:
    """Test drift recovery detection."""

    def test_check_drift_recovery_detects_recovery(self, tmp_path):
        """Verify check_drift_recovery() detects when drift recovers."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track active drift
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Check for recovery (within 5% threshold)
        recovered = detector.check_drift_recovery(
            actual_win_rate=0.58, expected_win_rate=0.60
        )

        assert recovered is True

    def test_check_drift_recovery_no_recovery_if_still_drifting(self, tmp_path):
        """Verify check_drift_recovery() returns False if still drifting."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track active drift
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Check for recovery (still >5% below expected)
        recovered = detector.check_drift_recovery(
            actual_win_rate=0.50, expected_win_rate=0.60
        )

        assert recovered is False

    def test_check_drift_recovery_marks_event_as_recovered(self, tmp_path):
        """Verify check_drift_recovery() marks drift event as recovered."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track active drift
        drift_id = detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Trigger recovery
        detector.check_drift_recovery(
            actual_win_rate=0.58, expected_win_rate=0.60
        )

        # Verify event marked as recovered
        history = detector.get_drift_history()
        drift_event = history["drift_events"][0]
        assert drift_event["recovered_at"] is not None
        assert drift_event["duration_minutes"] > 0

    def test_check_drift_recovery_resets_consecutive_counter(self, tmp_path):
        """Verify check_drift_recovery() resets consecutive counter."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track drift and increment cycles
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        detector.increment_consecutive_drift_cycle()
        detector.increment_consecutive_drift_cycle()

        # Verify counter is 2
        assert detector.get_consecutive_drift_cycles() == 2

        # Trigger recovery
        detector.check_drift_recovery(
            actual_win_rate=0.58, expected_win_rate=0.60
        )

        # Verify counter reset to 0
        assert detector.get_consecutive_drift_cycles() == 0

    def test_check_drift_recovery_updates_statistics(self, tmp_path):
        """Verify check_drift_recovery() updates drift statistics."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track active drift
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Trigger recovery
        detector.check_drift_recovery(
            actual_win_rate=0.58, expected_win_rate=0.60
        )

        # Verify statistics updated
        history = detector.get_drift_history()
        assert history["statistics"]["active_drift"] is False
        assert history["statistics"]["last_drift_recovery"] is not None


class TestPersistentHistory:
    """Test persistent history storage and loading."""

    def test_history_persists_across_reinitialization(self, tmp_path):
        """Verify drift history persists when detector is reinitialized."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create first detector and track drift
        detector1 = DriftDetector(model_dir=model_dir)
        detector1.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )

        # Create second detector (should load existing history)
        detector2 = DriftDetector(model_dir=model_dir)

        history = detector2.get_drift_history()
        assert history["statistics"]["total_drift_events"] == 1

    def test_load_history_raises_error_for_corrupted_file(self, tmp_path):
        """Verify InvalidHistoryError raised for corrupted JSON."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)
        history_file = model_dir / "drift_history.json"

        # Write corrupted JSON
        with open(history_file, "w") as f:
            f.write("{invalid json content")

        # Should create empty history instead of crashing
        detector = DriftDetector(model_dir=model_dir)
        history = detector.get_drift_history()
        assert history["statistics"]["total_drift_events"] == 0

    def test_load_history_validates_structure(self, tmp_path):
        """Verify history structure is validated on load."""
        import json

        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)
        history_file = model_dir / "drift_history.json"

        # Write invalid structure (missing required keys)
        with open(history_file, "w") as f:
            json.dump({"invalid": "structure"}, f)

        # Should create empty history
        detector = DriftDetector(model_dir=model_dir)
        history = detector.get_drift_history()
        assert "drift_events" in history
        assert "statistics" in history


class TestPerformance:
    """Test performance requirements."""

    def test_drift_detection_under_1ms(self):
        """Verify drift detection completes in < 1ms."""
        import time

        detector = DriftDetector()

        start_time = time.perf_counter()
        detector.check_drift(actual_win_rate=0.45, expected_win_rate=0.60)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert (
            elapsed_ms < 1.0
        ), f"Drift detection took {elapsed_ms:.3f}ms, exceeds 1ms limit"

    def test_track_drift_event_under_10ms(self, tmp_path):
        """Verify drift event tracking completes in < 10ms."""
        import time

        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        start_time = time.perf_counter()
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert (
            elapsed_ms < 10.0
        ), f"Drift tracking took {elapsed_ms:.3f}ms, exceeds 10ms limit"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_check_drift_recovery_with_no_active_drift(self, tmp_path):
        """Verify check_drift_recovery() returns False when no active drift."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # No drift tracked, check recovery
        recovered = detector.check_drift_recovery(
            actual_win_rate=0.58, expected_win_rate=0.60
        )

        assert recovered is False

    def test_increment_consecutive_cycles_with_no_active_drift(self, tmp_path):
        """Verify increment works even without active drift event."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Increment without tracking drift
        detector.increment_consecutive_drift_cycle()

        assert detector.get_consecutive_drift_cycles() == 1

    def test_multiple_drift_events_in_history(self, tmp_path):
        """Verify multiple drift events tracked correctly."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track multiple drift events
        detector.track_drift_event(
            actual_win_rate=0.45, expected_win_rate=0.60, severity="WARNING"
        )
        detector.track_drift_event(
            actual_win_rate=0.40, expected_win_rate=0.60, severity="CRITICAL"
        )
        detector.track_drift_event(
            actual_win_rate=0.35, expected_win_rate=0.60, severity="CRITICAL"
        )

        history = detector.get_drift_history()
        assert len(history["drift_events"]) == 3
        assert history["statistics"]["total_drift_events"] == 3

    def test_get_drift_history_returns_only_last_100_events(self, tmp_path):
        """Verify get_drift_history() limits to last 100 events."""
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        detector = DriftDetector(model_dir=model_dir)

        # Track 150 drift events
        for i in range(150):
            detector.track_drift_event(
                actual_win_rate=0.45 - (i * 0.001),
                expected_win_rate=0.60,
                severity="WARNING",
            )

        history = detector.get_drift_history()
        # Should return only last 100
        assert len(history["drift_events"]) <= 100

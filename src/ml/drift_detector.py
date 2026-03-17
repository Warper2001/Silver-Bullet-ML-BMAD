"""Model drift detection and monitoring.

This module provides continuous model drift detection with:
- Drift detection based on win rate degradation (>10% below expected)
- Drift history tracking (count, duration, recovery time)
- Consecutive drift cycle counting
- Trading halt recommendation after 3 consecutive failed cycles
- Drift recovery detection
- Persistent history storage
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class InvalidHistoryError(Exception):
    """Raised when drift history file is corrupted or invalid."""


class InsufficientDataError(Exception):
    """Raised when insufficient trade data for drift detection."""


class DriftDetector:
    """Continuous model drift detection and monitoring.

    Handles:
    - Drift detection based on win rate degradation (>10% below expected)
    - Drift history tracking (count, duration, recovery time)
    - Consecutive drift cycle counting
    - Trading halt recommendation after 3 consecutive failed cycles
    - Drift recovery detection
    - Persistent history storage

    Example:
        >>> detector = DriftDetector(model_dir="models/xgboost")
        >>> result = detector.check_drift(actual_win_rate=0.45, expected_win_rate=0.60)
        >>> if result["drift_detected"]:
        ...     drift_id = detector.track_drift_event(...)
        ...     detector.increment_consecutive_drift_cycle()
        ...     should_halt, reason = detector.should_halt_trading()
    """

    # Configuration constants
    DRIFT_THRESHOLD = 0.10  # 10% drop triggers drift
    RECOVERY_THRESHOLD = 0.05  # Within 5% is recovered
    MAX_CONSECUTIVE_CYCLES = 3  # Halt trading after 3 consecutive cycles
    MAX_HISTORY_EVENTS = 100  # Keep only last 100 events in memory

    def __init__(self, model_dir: str | Path = "models/xgboost"):
        """Initialize DriftDetector with persistent history.

        Args:
            model_dir: Directory containing model metadata and drift history
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._history_file = self._model_dir / "drift_history.json"
        self._file_lock = Lock()  # Thread-safe file operations

        # Load existing drift history or create new
        if self._history_file.exists():
            self._history = self._load_history()
        else:
            self._history = self._create_empty_history()

        # Statistics
        self._consecutive_drift_cycles = self._history["statistics"][
            "consecutive_drift_cycles"
        ]

        logger.info(
            f"DriftDetector initialized with "
            f"{self._history['statistics']['total_drift_events']} "
            f"drift events, {self._consecutive_drift_cycles} consecutive cycles"
        )

    def check_drift(self, actual_win_rate: float, expected_win_rate: float) -> dict:
        """Check for model drift based on win rate comparison.

        Args:
            actual_win_rate: Current actual win rate from live trading
            expected_win_rate: Expected win rate from model validation

        Returns:
            Dictionary with:
                - drift_detected (bool): Whether drift is detected
                - drift_magnitude (float): Win rate difference (actual - expected)
                - severity (str): "INFO", "WARNING", or "CRITICAL"

        Raises:
            ValueError: If win rates are not in [0, 1] range
        """
        # Validate inputs
        if not 0.0 <= actual_win_rate <= 1.0:
            raise ValueError(
                "Win rates must be in range [0, 1], "
                f"got actual_win_rate={actual_win_rate}"
            )
        if not 0.0 <= expected_win_rate <= 1.0:
            raise ValueError(
                "Win rates must be in range [0, 1], "
                f"got expected_win_rate={expected_win_rate}"
            )

        # Calculate drift magnitude
        drift_magnitude = actual_win_rate - expected_win_rate

        # Check if drift detected (>10% drop)
        drift_detected = drift_magnitude < -self.DRIFT_THRESHOLD

        # Determine severity
        if drift_magnitude < -0.20:
            severity = "CRITICAL"
        elif drift_magnitude < -self.DRIFT_THRESHOLD:
            severity = "WARNING"
        else:
            severity = "INFO"

        return {
            "drift_detected": drift_detected,
            "drift_magnitude": drift_magnitude,
            "severity": severity,
        }

    def track_drift_event(
        self,
        actual_win_rate: float,
        expected_win_rate: float,
        severity: str = "WARNING",
    ) -> str:
        """Record a drift detection event.

        Args:
            actual_win_rate: Current actual win rate
            expected_win_rate: Expected win rate from model
            severity: Severity level ("INFO", "WARNING", "CRITICAL")

        Returns:
            Drift event ID (UUID)
        """
        # Calculate drift magnitude
        magnitude = actual_win_rate - expected_win_rate

        # Create drift event
        drift_event = {
            "drift_id": str(uuid.uuid4()),
            "detected_at": datetime.now().isoformat(),
            "duration_minutes": 0,  # Active, not recovered yet
            "severity": severity,
            "magnitude": magnitude,
            "expected_win_rate": expected_win_rate,
            "actual_win_rate": actual_win_rate,
            "recovered_at": None,
            "consecutive_cycle_count": self._consecutive_drift_cycles,
            "trading_halted": False,
        }

        # Add to history
        with self._file_lock:
            self._history["drift_events"].append(drift_event)
            self._history["statistics"]["total_drift_events"] += 1
            self._history["statistics"]["active_drift"] = True
            self._history["statistics"][
                "last_drift_detection"
            ] = datetime.now().isoformat()

            # Limit history size
            if len(self._history["drift_events"]) > self.MAX_HISTORY_EVENTS:
                self._history["drift_events"] = self._history["drift_events"][
                    -self.MAX_HISTORY_EVENTS:
                ]

            # Save to disk
            self._save_history()

        logger.warning(
            f"Drift detected: {magnitude:.2%} drop "
            f"(actual={actual_win_rate:.2%}, expected={expected_win_rate:.2%})"
        )

        return drift_event["drift_id"]

    def get_drift_history(self) -> dict:
        """Get the complete drift history.

        Returns:
            Dictionary with:
                - drift_events (list): List of drift event dictionaries
                - statistics (dict): Aggregate statistics
        """
        return self._history

    def increment_consecutive_drift_cycle(self):
        """Increment consecutive drift cycle counter.

        Called by WalkForwardOptimizer after retraining if drift detected.
        Updates the active drift event's consecutive_cycle_count and checks
        if trading should be halted.
        """
        with self._file_lock:
            self._consecutive_drift_cycles += 1
            self._history["statistics"][
                "consecutive_drift_cycles"
            ] = self._consecutive_drift_cycles

            # Update active drift event
            active_drift = self._get_active_drift_event()
            if active_drift:
                active_drift["consecutive_cycle_count"] = self._consecutive_drift_cycles

                # Check if trading should be halted
                if self._consecutive_drift_cycles >= self.MAX_CONSECUTIVE_CYCLES:
                    active_drift["trading_halted"] = True
                    logger.critical(
                        f"TRADING HALT: {self._consecutive_drift_cycles} consecutive "
                        f"drift cycles detected. Manual review required."
                    )
                else:
                    logger.warning(
                        f"Consecutive drift cycles: {self._consecutive_drift_cycles}/"
                        f"{self.MAX_CONSECUTIVE_CYCLES}. "
                        f"Trading continues but monitored closely."
                    )

            # Save updated history
            self._save_history()

    def get_consecutive_drift_cycles(self) -> int:
        """Get current consecutive drift cycle count.

        Returns:
            Number of consecutive retraining cycles with drift detected
        """
        return self._consecutive_drift_cycles

    def should_halt_trading(self) -> tuple[bool, str]:
        """Check if trading should be halted due to persistent drift.

        Returns:
            Tuple of (should_halt: bool, reason: str)
        """
        # Check consecutive drift cycles
        if self._consecutive_drift_cycles >= self.MAX_CONSECUTIVE_CYCLES:
            return (
                True,
                f"Trading halted: {self._consecutive_drift_cycles} consecutive "
                f"drift cycles detected. Manual review required.",
            )

        # Check if current drift has trading_halted flag
        active_drift = self._get_active_drift_event()
        if active_drift and active_drift.get("trading_halted", False):
            return (
                True,
                "Trading halted: Previous drift event requires manual review.",
            )

        return False, ""

    def check_drift_recovery(
        self, actual_win_rate: float, expected_win_rate: float
    ) -> bool:
        """Check if drift has recovered.

        Args:
            actual_win_rate: Current actual win rate
            expected_win_rate: Expected win rate from model

        Returns:
            True if recovered (within 5% of expected), False otherwise
        """
        # Check for active drift
        active_drift = self._get_active_drift_event()
        if not active_drift:
            return False  # No drift to recover

        # Check if recovered (within 5% threshold)
        drift_magnitude = actual_win_rate - expected_win_rate

        if abs(drift_magnitude) <= self.RECOVERY_THRESHOLD:
            # Mark as recovered
            recovered_at = datetime.now().isoformat()
            active_drift["recovered_at"] = recovered_at
            active_drift["duration_minutes"] = self._calculate_duration(
                active_drift["detected_at"], recovered_at
            )

            # Reset consecutive counter
            old_count = self._consecutive_drift_cycles
            self._consecutive_drift_cycles = 0
            self._history["statistics"]["consecutive_drift_cycles"] = 0

            # Update statistics
            self._history["statistics"]["active_drift"] = False
            self._history["statistics"]["last_drift_recovery"] = recovered_at

            # Save history
            self._save_history()

            logger.info(
                f"Drift recovered after {active_drift['duration_minutes']} minutes. "
                f"Consecutive cycles reset from {old_count} to 0."
            )

            return True

        return False

    def _get_active_drift_event(self) -> dict | None:
        """Get the currently active drift event (not recovered).

        Returns:
            Active drift event dictionary or None if no active drift
        """
        for event in reversed(self._history["drift_events"]):
            if event["recovered_at"] is None:
                return event
        return None

    def _calculate_duration(self, detected_at: str, recovered_at: str) -> int:
        """Calculate drift duration in minutes.

        Args:
            detected_at: ISO format timestamp when drift was detected
            recovered_at: ISO format timestamp when drift was recovered

        Returns:
            Duration in minutes
        """
        detected = datetime.fromisoformat(detected_at)
        recovered = datetime.fromisoformat(recovered_at)
        duration_seconds = (recovered - detected).total_seconds()
        return int(duration_seconds / 60)

    def _save_history(self):
        """Save drift history to persistent storage."""
        try:
            with self._file_lock:
                with open(self._history_file, "w") as f:
                    json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save drift history: {e}")

    def _load_history(self) -> dict:
        """Load drift history from persistent storage.

        Returns:
            Drift history dictionary

        Raises:
            InvalidHistoryError: If history file is corrupted
        """
        try:
            with self._file_lock:
                with open(self._history_file, "r") as f:
                    history = json.load(f)

            # Validate structure
            if (
                "drift_events" not in history
                or "statistics" not in history
                or not isinstance(history["drift_events"], list)
            ):
                logger.warning("Invalid drift history file structure")
                return self._create_empty_history()

            return history

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Corrupted drift history file: {e}")
            return self._create_empty_history()
        except Exception as e:
            logger.error(f"Failed to load drift history: {e}")
            return self._create_empty_history()

    def _create_empty_history(self) -> dict:
        """Create empty drift history structure.

        Returns:
            Empty drift history dictionary
        """
        return {
            "drift_events": [],
            "statistics": {
                "total_drift_events": 0,
                "active_drift": False,
                "consecutive_drift_cycles": 0,
                "last_drift_detection": None,
                "last_drift_recovery": None,
            },
        }

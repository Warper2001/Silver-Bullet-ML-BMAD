"""Signal Filter for ML-Based Probability Thresholding.

This module implements signal filtering based on ML probability scores,
filtering out low-probability Silver Bullet setups before execution.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.data.models import SilverBulletSetup

logger = logging.getLogger(__name__)


class SignalFilter:
    """Filter Silver Bullet signals based on ML probability threshold.

    Handles:
    - Threshold-based filtering (default: P(Success) > 0.65)
    - Filtering statistics tracking
    - Win rate monitoring for allowed signals
    - Model drift detection (>10% win rate drop)

    Performance:
    - Filtering overhead: < 5ms per signal
    - Statistics logging overhead: < 1ms
    - Memory efficient: Keep only last 1000 allowed signals
    """

    def __init__(
        self, threshold: float = 0.65, model_dir: str | Path = "models/xgboost"
    ):
        """Initialize SignalFilter with probability threshold.

        Args:
            threshold: Minimum probability required for signal execution (0.0 to 1.0)
            model_dir: Directory containing model metadata for expected win rate

        Raises:
            ValueError: If threshold not in [0, 1] range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        self._threshold = threshold
        self._model_dir = Path(model_dir)

        # Filtering statistics
        self._stats = {
            "total_signals": 0,
            "filtered_signals": 0,
            "allowed_signals": 0,
            "filter_rate": 0.0,
            "session_start": datetime.now(),
            "last_update": datetime.now(),
            "drift_detected": False,
            "drift_magnitude": 0.0,
        }

        # Track allowed signals for win rate calculation
        self._allowed_signals = []  # List of {timestamp, probability, outcome?}

        logger.info(f"SignalFilter initialized with threshold: {self._threshold}")

    def filter_signal(
        self, signal: SilverBulletSetup, probability: float
    ) -> dict[str, object]:
        """Filter signal based on probability threshold.

        Args:
            signal: Silver Bullet setup
            probability: ML predicted probability of success (0.0 to 1.0)

        Returns:
            Dictionary with filtering decision and metadata:
            {
                "allowed": bool,  # True if signal passes threshold
                "reason": str,  # "allowed_by_ml_threshold" or "filtered_by_ml_threshold"
                "probability": float,  # Input probability score
                "threshold": float,  # Applied threshold
                "latency_ms": float  # Processing time
            }

        Raises:
            ValueError: If probability not in [0, 1] range
        """
        start_time = datetime.now()

        # Validate probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        # Apply threshold filter
        allowed = probability > self._threshold

        # Update statistics
        self._update_statistics(allowed, probability, signal)

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Create result
        result = {
            "allowed": allowed,
            "reason": "allowed_by_ml_threshold"
            if allowed
            else "filtered_by_ml_threshold",
            "probability": probability,
            "threshold": self._threshold,
            "latency_ms": latency_ms,
        }

        # Log result
        if allowed:
            logger.info(
                f"Signal ALLOWED by ML threshold: P={probability:.4f} > {self._threshold}"
            )
        else:
            logger.info(
                f"Signal FILTERED by ML threshold: P={probability:.4f} <= {self._threshold}"
            )

        return result

    def should_allow(self, probability: float) -> bool:
        """Check if probability passes threshold.

        Args:
            probability: ML predicted probability of success

        Returns:
            True if probability > threshold, False otherwise

        Raises:
            ValueError: If probability not in [0, 1] range
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        return probability > self._threshold

    def record_trade_outcome(self, signal: SilverBulletSetup, success: bool):
        """Record trade outcome for win rate tracking.

        Args:
            signal: The original signal that was allowed
            success: True if trade was profitable, False otherwise
        """
        # Find matching allowed signal by timestamp
        for allowed_signal in reversed(self._allowed_signals):
            if allowed_signal["timestamp"] == signal.timestamp:
                allowed_signal["outcome"] = success
                logger.debug(
                    f"Recorded trade outcome: {success} for signal at {signal.timestamp}"
                )
                return

        logger.warning(
            f"No matching allowed signal found for timestamp {signal.timestamp}"
        )

    def calculate_win_rate(self) -> float:
        """Calculate win rate over last 50 completed trades.

        Returns:
            Win rate (0.0 to 1.0), or 0.0 if no completed trades
        """
        # Get completed signals with outcomes
        completed_signals = [s for s in self._allowed_signals if "outcome" in s]

        if not completed_signals:
            return 0.0

        # Use last 50 trades
        recent_signals = completed_signals[-50:]
        outcomes = [s["outcome"] for s in recent_signals]

        win_rate = sum(outcomes) / len(outcomes)
        logger.debug(f"Calculated win rate: {win_rate:.4f} over {len(outcomes)} trades")

        return win_rate

    def check_drift_detection(self) -> bool:
        """Check for model drift and alert if detected.

        Returns:
            True if model drift detected (>10% below expected), False otherwise
        """
        # Get completed signals with outcomes
        completed_signals = [s for s in self._allowed_signals if "outcome" in s]

        if len(completed_signals) < 50:
            logger.debug("Not enough completed trades for drift detection (need 50)")
            return False

        # Calculate actual win rate
        outcomes = [s["outcome"] for s in completed_signals[-50:]]
        actual_win_rate = sum(outcomes) / len(outcomes)

        # Load expected win rate from model
        expected_win_rate = self._load_expected_win_rate()

        # Check for drift
        drift_magnitude = actual_win_rate - expected_win_rate

        if drift_magnitude < -0.10:  # More than 10% below expected
            logger.error(
                f"MODEL DRIFT DETECTED: "
                f"actual={actual_win_rate:.2%}, "
                f"expected={expected_win_rate:.2%}, "
                f"drift={drift_magnitude:.2%}"
            )
            self._stats["drift_detected"] = True
            self._stats["drift_magnitude"] = drift_magnitude
            return True
        else:
            self._stats["drift_detected"] = False
            self._stats["drift_magnitude"] = 0.0
            return False

    def get_statistics(self) -> dict[str, object]:
        """Get current filtering statistics.

        Returns:
            Dictionary with filtering statistics:
            {
                "total_signals": int,
                "filtered_signals": int,
                "allowed_signals": int,
                "filter_rate": float,
                "session_start": datetime,
                "last_update": datetime,
                "drift_detected": bool,
                "drift_magnitude": float,
                "win_rate": float
            }
        """
        stats = self._stats.copy()
        stats["win_rate"] = self.calculate_win_rate()
        return stats

    def _update_statistics(
        self, allowed: bool, probability: float, signal: SilverBulletSetup
    ):
        """Update filtering statistics.

        Args:
            allowed: Whether signal was allowed
            probability: Probability score for the signal
            signal: The Silver Bullet setup
        """
        self._stats["total_signals"] += 1

        if allowed:
            self._stats["allowed_signals"] += 1
            # Track allowed signals for win rate calculation
            self._allowed_signals.append(
                {
                    "timestamp": signal.timestamp,
                    "probability": probability,
                }
            )
        else:
            self._stats["filtered_signals"] += 1

        # Calculate filter rate
        if self._stats["total_signals"] > 0:
            self._stats["filter_rate"] = (
                self._stats["filtered_signals"] / self._stats["total_signals"]
            )

        self._stats["last_update"] = datetime.now()

        # Keep only last 1000 allowed signals
        if len(self._allowed_signals) > 1000:
            self._allowed_signals = self._allowed_signals[-1000:]

        # Log statistics every 100 signals
        if self._stats["total_signals"] % 100 == 0:
            self._log_statistics()

    def _log_statistics(self):
        """Log current filtering statistics."""
        stats = self.get_statistics()
        logger.info(
            f"Filtering Statistics: "
            f"total={stats['total_signals']}, "
            f"filtered={stats['filtered_signals']}, "
            f"allowed={stats['allowed_signals']}, "
            f"filter_rate={stats['filter_rate']:.2%}, "
            f"win_rate={stats['win_rate']:.2%}"
        )

    def _load_expected_win_rate(self) -> float:
        """Load expected win rate from model validation metrics.

        Returns:
            Expected win rate (0.0 to 1.0), or 0.65 if unavailable
        """
        # Get model directory (use first available horizon)
        model_dirs = (
            list(self._model_dir.glob("*_minute")) if self._model_dir.exists() else []
        )

        if not model_dirs:
            logger.warning("No model directories found, using default 0.65 win rate")
            return 0.65

        # Load metadata from first model
        model_dir = sorted(model_dirs)[0]
        metadata_file = model_dir / "metadata.json"

        if not metadata_file.exists():
            logger.warning("No model metadata found, using default 0.65 win rate")
            return 0.65

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Convert ROC-AUC to expected win rate
            # This is a simplified conversion - actual formula may differ
            roc_auc = metadata.get("metrics", {}).get("roc_auc", 0.65)
            expected_win_rate = (
                roc_auc - 0.5
            ) * 2 + 0.5  # Map AUC [0.5, 1.0] to [0.0, 1.0]

            logger.debug(f"Loaded expected win rate: {expected_win_rate:.4f}")
            return max(0.0, min(1.0, expected_win_rate))  # Clamp to [0, 1]

        except Exception as e:
            logger.warning(f"Failed to load expected win rate: {e}, using default 0.65")
            return 0.65

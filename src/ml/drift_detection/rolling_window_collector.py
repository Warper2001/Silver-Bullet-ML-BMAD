"""Rolling window data collector for drift detection.

This module collects and maintains a rolling window of recent predictions
and features for drift detection monitoring.
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.ml.drift_detection import StatisticalDriftDetector

logger = logging.getLogger(__name__)


class RollingWindowCollector:
    """Collects rolling window of recent data for drift detection.

    Maintains configurable-sized rolling windows of:
    - Recent predictions (last N predictions)
    - Recent features (last N feature vectors)
    - Timestamps for temporal tracking

    Example:
        >>> collector = RollingWindowCollector(window_hours=24)
        >>> collector.add_prediction(prediction=0.75, features={"f1": 1.0, "f2": 2.0})
        >>> if collector.has_sufficient_data():
        ...     recent_features, recent_predictions = collector.get_recent_data()
    """

    def __init__(
        self,
        window_hours: int = 24,
        min_samples: int = 100,
        max_samples: int = 10000,
    ):
        """Initialize rolling window collector.

        Args:
            window_hours: Window size in hours
            min_samples: Minimum samples required for drift detection
            max_samples: Maximum samples to keep in memory
        """
        self._window_hours = window_hours
        self._min_samples = min_samples
        self._max_samples = max_samples

        # Data storage (using deque for efficient append/pop)
        self._predictions = deque(maxlen=max_samples)
        self._features = {}  # feature_name -> deque of values
        self._timestamps = deque(maxlen=max_samples)

        # Feature set tracking
        self._feature_names = set()

        logger.info(
            f"RollingWindowCollector initialized: "
            f"window={window_hours}h, min_samples={min_samples}, "
            f"max_samples={max_samples}"
        )

    def add_prediction(
        self,
        prediction: float,
        features: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Add a prediction and its features to the rolling window.

        Args:
            prediction: Model prediction probability
            features: Dictionary of feature_name -> feature_value
            timestamp: Timestamp of prediction (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Add prediction
        self._predictions.append(prediction)
        self._timestamps.append(timestamp)

        # Add features
        for feature_name, feature_value in features.items():
            if feature_name not in self._features:
                self._features[feature_name] = deque(maxlen=self._max_samples)

            self._features[feature_name].append(feature_value)
            self._feature_names.add(feature_name)

        # Prune old data to maintain window size
        self._prune_old_data()

    def _prune_old_data(self) -> None:
        """Remove data older than the rolling window."""
        if not self._timestamps:
            return

        cutoff_time = datetime.now() - timedelta(hours=self._window_hours)
        cutoff_timestamp = cutoff_time.timestamp()

        # Find first index within window
        prune_count = 0
        for ts in self._timestamps:
            if ts.timestamp() >= cutoff_timestamp:
                break
            prune_count += 1

        # Remove old data
        if prune_count > 0:
            # Prune predictions and timestamps
            for _ in range(prune_count):
                if self._predictions:
                    self._predictions.popleft()
                if self._timestamps:
                    self._timestamps.popleft()

            # Prune features
            for feature_name in self._features:
                for _ in range(min(prune_count, len(self._features[feature_name]))):
                    self._features[feature_name].popleft()

            if prune_count > 100:
                logger.debug(f"Pruned {prune_count} old samples from rolling window")

    def has_sufficient_data(self) -> bool:
        """Check if sufficient data collected for drift detection.

        Returns:
            True if at least min_samples collected
        """
        return len(self._predictions) >= self._min_samples

    def get_recent_data(
        self,
        feature_names: list[str] | None = None,
    ) -> tuple[dict[str, list], list[float]]:
        """Get recent data for drift detection.

        Args:
            feature_names: List of feature names to extract (uses all if None)

        Returns:
            Tuple of (features_dict, predictions_list)
            - features_dict: Dictionary of feature_name -> values_list
            - predictions_list: List of prediction values
        """
        if not self.has_sufficient_data():
            raise ValueError(
                f"Insufficient data: {len(self._predictions)} "
                f"< {self._min_samples} required"
            )

        # Use all tracked features if none specified
        if feature_names is None:
            feature_names = list(self._features.keys())

        # Extract features as lists
        features_dict = {}
        for feature_name in feature_names:
            if feature_name in self._features:
                features_dict[feature_name] = list(self._features[feature_name])

        # Get predictions
        predictions_list = list(self._predictions)

        return features_dict, predictions_list

    def get_window_stats(self) -> dict:
        """Get statistics about the rolling window.

        Returns:
            Dictionary with window statistics
        """
        if not self._timestamps:
            return {
                "total_samples": 0,
                "window_hours": self._window_hours,
                "oldest_timestamp": None,
                "newest_timestamp": None,
            }

        return {
            "total_samples": len(self._predictions),
            "window_hours": self._window_hours,
            "oldest_timestamp": self._timestamps[0].isoformat(),
            "newest_timestamp": self._timestamps[-1].isoformat(),
            "features_count": len(self._feature_names),
            "feature_names": list(self._feature_names),
        }

    def clear(self) -> None:
        """Clear all collected data."""
        self._predictions.clear()
        self._features.clear()
        self._timestamps.clear()
        self._feature_names.clear()

        logger.info("Rolling window cleared")

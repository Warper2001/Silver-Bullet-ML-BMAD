"""Statistical drift detector using PSI and KS tests.

This module provides the main DriftDetector class that combines:
- PSI (Population Stability Index) for feature drift detection
- KS (Kolmogorov-Smirnov) test for prediction drift detection
- Continuous monitoring with configurable thresholds
- Event logging and alert triggering
"""

import csv
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.ml.drift_detection.ks_calculator import (
    calculate_ks_statistic,
    calculate_drift_magnitude,
    classify_prediction_drift,
)
from src.ml.drift_detection.models import (
    DriftDetectionResult,
    DriftDetectorConfig,
    DriftEvent,
    KSTestResult,
    PSIMetric,
)
from src.ml.drift_detection.psi_calculator import (
    calculate_psi,
    calculate_psi_for_multiple_features,
    classify_drift_severity,
)

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when insufficient data for drift detection."""

    pass


class InvalidBaselineError(Exception):
    """Raised when baseline data is invalid or missing."""

    pass


class StatisticalDriftDetector:
    """Statistical drift detector using PSI and KS tests.

    Detects model drift by monitoring:
    - Feature distribution changes (via PSI)
    - Prediction distribution changes (via KS test)

    Example:
        >>> detector = StatisticalDriftDetector(
        ...     baseline_features={"feature1": train_data},
        ...     baseline_predictions=train_predictions,
        ...     feature_names=["feature1", "feature2"]
        ... )
        >>> result = detector.detect_drift(
        ...     recent_features={"feature1": recent_data},
        ...     recent_predictions=recent_preds
        ... )
        >>> if result.drift_detected:
        ...     print(f"Drift detected in: {result.drifting_features}")
    """

    def __init__(
        self,
        baseline_features: dict[str, np.ndarray],
        baseline_predictions: np.ndarray,
        feature_names: list[str],
        config: DriftDetectorConfig | None = None,
        csv_log_path: str | None = None,
    ):
        """Initialize statistical drift detector.

        Args:
            baseline_features: Feature distributions from training data
            baseline_predictions: Prediction probabilities from training data
            feature_names: List of feature names to monitor
            config: Drift detection configuration (uses defaults if None)
            csv_log_path: Path to CSV audit trail file (default: logs/drift_events.csv)

        Raises:
            InvalidBaselineError: If baseline data is invalid
            InsufficientDataError: If insufficient baseline data
        """
        # Validate baseline data
        if not baseline_features:
            raise InvalidBaselineError("Baseline features cannot be empty")

        if len(baseline_predictions) < 100:
            raise InsufficientDataError(
                f"Baseline predictions insufficient: "
                f"{len(baseline_predictions)} < 100 minimum required"
            )

        # Store baseline
        self._baseline_features = {
            name: np.asarray(arr) for name, arr in baseline_features.items()
        }
        self._baseline_predictions = np.asarray(baseline_predictions)
        self._feature_names = feature_names or list(baseline_features.keys())

        # Configuration
        self._config = config or DriftDetectorConfig()

        # Event tracking
        self._drift_events: list[DriftEvent] = []
        self._last_check_time: datetime | None = None

        # CSV audit trail
        self._csv_log_path = csv_log_path or "logs/drift_events.csv"

        logger.info(
            f"StatisticalDriftDetector initialized with "
            f"{len(self._feature_names)} features, "
            f"{len(self._baseline_predictions)} baseline predictions, "
            f"CSV log: {self._csv_log_path}"
        )

    def detect_drift(
        self,
        recent_features: dict[str, np.ndarray],
        recent_predictions: np.ndarray,
        check_interval_hours: int | None = None,
    ) -> DriftDetectionResult:
        """Detect drift in recent data compared to baseline.

        Args:
            recent_features: Recent feature distributions (last 24 hours)
            recent_predictions: Recent prediction probabilities (last 24 hours)
            check_interval_hours: Override check interval from config

        Returns:
            DriftDetectionResult with PSI metrics, KS result, and drift status

        Raises:
            InsufficientDataError: If insufficient recent data
        """
        logger.info("Starting drift detection...")

        # Validate recent data
        if len(recent_predictions) < 100:
            raise InsufficientDataError(
                f"Recent predictions insufficient: "
                f"{len(recent_predictions)} < 100 minimum required"
            )

        # Calculate PSI for each feature
        psi_metrics = self._calculate_psi_metrics(recent_features)

        # Calculate KS test for predictions
        ks_result = self._calculate_ks_test(recent_predictions)

        # Determine which features are drifting
        drifting_features = [
            metric.feature_name
            for metric in psi_metrics
            if metric.drift_severity in ["moderate", "severe"]
        ]

        # Check if prediction drift detected
        prediction_drift_detected = ks_result.drift_detected if ks_result else False
        if prediction_drift_detected:
            drifting_features.append("predictions")

        # Overall drift detected?
        drift_detected = len(drifting_features) > 0

        # Create result
        result = DriftDetectionResult(
            psi_metrics=psi_metrics,
            ks_result=ks_result,
            drift_detected=drift_detected,
            drifting_features=drifting_features,
        )

        # Log drift event if detected
        if drift_detected:
            self._log_drift_event(result)

        # Update last check time
        self._last_check_time = datetime.now()

        logger.info(
            f"Drift detection complete: "
            f"drift_detected={drift_detected}, "
            f"drifting_features={drifting_features}"
        )

        return result

    def _calculate_psi_metrics(
        self, recent_features: dict[str, np.ndarray | list]
    ) -> list[PSIMetric]:
        """Calculate PSI metrics for all features.

        Args:
            recent_features: Recent feature distributions (can be lists or arrays)

        Returns:
            List of PSIMetric objects
        """
        psi_metrics = []

        # Filter to only features in baseline and convert to numpy arrays
        valid_features = {}
        for name, arr in recent_features.items():
            if name in self._baseline_features:
                # Convert lists to numpy arrays
                if isinstance(arr, list):
                    valid_features[name] = np.array(arr)
                else:
                    valid_features[name] = arr

        # Calculate PSI for each feature
        psi_scores = calculate_psi_for_multiple_features(
            expected_features={
                name: self._baseline_features[name] for name in valid_features.keys()
            },
            actual_features=valid_features,
            bins=self._config.psi_bins,
        )

        # Create PSIMetric objects
        for feature_name, psi_score in psi_scores.items():
            try:
                if not np.isnan(psi_score):
                    metric = PSIMetric.from_psi_value(feature_name, psi_score)
                    psi_metrics.append(metric)
            except ValidationError as e:
                logger.warning(f"Could not create PSIMetric for {feature_name}: {e}")

        return psi_metrics

    def _calculate_ks_test(
        self, recent_predictions: np.ndarray | list
    ) -> KSTestResult | None:
        """Calculate KS test for predictions.

        Args:
            recent_predictions: Recent prediction probabilities (can be list or array)

        Returns:
            KSTestResult or None if calculation fails
        """
        try:
            # Convert list to numpy array if needed
            if isinstance(recent_predictions, list):
                recent_predictions = np.array(recent_predictions)

            ks_statistic, p_value = calculate_ks_statistic(
                baseline_predictions=self._baseline_predictions,
                recent_predictions=recent_predictions,
            )

            result = KSTestResult.from_ks_test(
                ks_statistic=ks_statistic,
                p_value=p_value,
                p_value_threshold=self._config.ks_p_value_threshold,
            )

            logger.debug(
                f"KS test: statistic={ks_statistic:.4f}, "
                f"p_value={p_value:.4f}, "
                f"drift_detected={result.drift_detected}"
            )

            return result

        except (ValueError, InsufficientDataError) as e:
            logger.error(f"Could not calculate KS test: {e}")
            return None

    def _log_drift_event(self, result: DriftDetectionResult) -> None:
        """Log drift event to internal tracking and CSV.

        Args:
            result: Drift detection result
        """
        # Determine overall severity
        has_severe = any(
            metric.drift_severity == "severe" for metric in result.psi_metrics
        )
        has_severe_prediction = (
            result.ks_result.drift_detected
            if result.ks_result
            else False
        )
        severity: Literal["moderate", "severe"] = (
            "severe" if has_severe or has_severe_prediction else "moderate"
        )

        # Determine event type
        event_type: Literal["feature_drift", "prediction_drift"] = (
            "prediction_drift"
            if "predictions" in result.drifting_features
            else "feature_drift"
        )

        # Create event
        event = DriftEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            details={
                "drifting_features": result.drifting_features,
                "psi_scores": {
                    metric.feature_name: metric.psi_score for metric in result.psi_metrics
                },
                "ks_result": (
                    result.ks_result.model_dump() if result.ks_result else None
                ),
            },
            timestamp=datetime.now(),
        )

        # Track internally
        self._drift_events.append(event)

        # Log to CSV audit trail
        self._log_to_csv(event)

        # Log to logger
        logger.warning(
            f"Drift detected: event_type={event_type}, "
            f"severity={severity}, "
            f"drifting_features={result.drifting_features}"
        )

    def _log_to_csv(self, event: DriftEvent) -> None:
        """Append drift event to CSV audit trail.

        Args:
            event: Drift event to log
        """
        try:
            log_file = Path(self._csv_log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Prepare event data for CSV
            features_str = ';'.join(event.details.get('drifting_features', []))
            psi_scores = event.details.get('psi_scores', {})
            psi_str = ';'.join(f"{k}:{v:.4f}" for k, v in psi_scores.items())

            # Format KS result
            ks_result = event.details.get('ks_result')
            if ks_result:
                ks_str = f"statistic={ks_result.get('ks_statistic', 0):.4f}," \
                        f"p_value={ks_result.get('p_value', 1):.4f}," \
                        f"drift={ks_result.get('drift_detected', False)}"
            else:
                ks_str = "None"

            # Write to CSV
            file_exists = log_file.exists()
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write header
                    writer.writerow([
                        'event_id', 'timestamp', 'event_type', 'severity',
                        'drifting_features', 'psi_scores', 'ks_result'
                    ])

                # Write event row
                writer.writerow([
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.severity,
                    features_str,
                    psi_str,
                    ks_str
                ])

            logger.debug(f"Drift event logged to CSV: {log_file}")

        except Exception as e:
            # Don't crash on CSV logging errors
            logger.error(f"Failed to log drift event to CSV: {e}")

    def get_drift_events(self, last_n: int | None = None) -> list[DriftEvent]:
        """Get tracked drift events.

        Args:
            last_n: Return only last N events (if specified)

        Returns:
            List of drift events
        """
        if last_n is None:
            return self._drift_events.copy()
        return self._drift_events[-last_n:]

    def clear_old_events(self, retention_days: int | None = None) -> int:
        """Clear drift events older than retention period.

        Args:
            retention_days: Retention period (uses config default if None)

        Returns:
            Number of events cleared
        """
        retention = retention_days or self._config.retention_days
        cutoff_time = datetime.now() - timedelta(days=retention)

        initial_count = len(self._drift_events)
        self._drift_events = [
            event for event in self._drift_events if event.timestamp >= cutoff_time
        ]
        cleared_count = initial_count - len(self._drift_events)

        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old drift events")

        return cleared_count

    def get_baseline_info(self) -> dict:
        """Get information about baseline data.

        Returns:
            Dictionary with baseline statistics
        """
        return {
            "num_features": len(self._feature_names),
            "feature_names": self._feature_names,
            "baseline_predictions_mean": float(np.mean(self._baseline_predictions)),
            "baseline_predictions_std": float(np.std(self._baseline_predictions)),
            "baseline_predictions_min": float(np.min(self._baseline_predictions)),
            "baseline_predictions_max": float(np.max(self._baseline_predictions)),
        }

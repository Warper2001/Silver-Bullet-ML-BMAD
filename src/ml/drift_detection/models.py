"""Pydantic models for drift detection metrics and events.

This module provides data models for:
- PSI (Population Stability Index) metrics
- KS (Kolmogorov-Smirnov) test results
- Drift events and alerts
- Drift detection configuration
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PSIMetric(BaseModel):
    """Population Stability Index metric for a single feature.

    PSI measures the distribution shift between baseline (training) and recent data.
    Higher values indicate more drift.

    Attributes:
        feature_name: Name of the feature being monitored
        psi_score: PSI score (higher = more drift)
        drift_severity: Severity level based on PSI thresholds
        timestamp: When the PSI was calculated
    """

    feature_name: str = Field(..., description="Name of the feature being monitored")
    psi_score: float = Field(..., ge=0.0, description="PSI score (higher = more drift)")
    drift_severity: Literal["none", "moderate", "severe"] = Field(
        ..., description="Severity level based on PSI thresholds"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="When PSI was calculated")

    @classmethod
    def from_psi_value(cls, feature_name: str, psi_score: float) -> "PSIMetric":
        """Create PSIMetric from PSI score with automatic severity classification.

        Args:
            feature_name: Name of the feature
            psi_score: Calculated PSI score

        Returns:
            PSIMetric with appropriate severity level
        """
        if psi_score < 0.2:
            drift_severity = "none"
        elif psi_score < 0.5:
            drift_severity = "moderate"
        else:
            drift_severity = "severe"

        return cls(feature_name=feature_name, psi_score=psi_score, drift_severity=drift_severity)


class KSTestResult(BaseModel):
    """Kolmogorov-Smirnov test result for prediction distribution comparison.

    The KS test compares two distributions to detect significant differences.

    Attributes:
        ks_statistic: Maximum distance between cumulative distribution functions
        p_value: P-value for the test (lower = more significant drift)
        drift_detected: Whether drift was detected (p-value < threshold)
        timestamp: When the KS test was performed
    """

    ks_statistic: float = Field(
        ..., ge=0.0, le=1.0, description="Maximum distance between CDFs"
    )
    p_value: float = Field(..., ge=0.0, le=1.0, description="P-value for the test")
    drift_detected: bool = Field(
        ..., description="Whether drift was detected (p-value < 0.05)"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="When test was performed")

    @classmethod
    def from_ks_test(
        cls, ks_statistic: float, p_value: float, p_value_threshold: float = 0.05
    ) -> "KSTestResult":
        """Create KSTestResult from KS test values with automatic drift detection.

        Args:
            ks_statistic: KS statistic value
            p_value: P-value from KS test
            p_value_threshold: Threshold for drift detection (default: 0.05)

        Returns:
            KSTestResult with drift_detected flag set appropriately
        """
        drift_detected = p_value < p_value_threshold

        return cls(
            ks_statistic=ks_statistic, p_value=p_value, drift_detected=drift_detected
        )


class DriftEvent(BaseModel):
    """Drift detection event.

    Represents a drift detection occurrence with details about what drifted.

    Attributes:
        event_id: Unique identifier for the drift event
        event_type: Type of drift (feature or prediction)
        severity: Severity level of the drift
        details: Specific details about what drifted
        timestamp: When the drift was detected
    """

    event_id: str = Field(..., description="Unique identifier for the drift event")
    event_type: Literal["feature_drift", "prediction_drift"] = Field(
        ..., description="Type of drift detected"
    )
    severity: Literal["moderate", "severe"] = Field(..., description="Severity level")
    details: dict = Field(..., description="Specific details about what drifted")
    timestamp: datetime = Field(default_factory=datetime.now, description="When drift was detected")


class DriftDetectionResult(BaseModel):
    """Complete drift detection result.

    Aggregates PSI metrics for all features and KS test result for predictions.

    Attributes:
        psi_metrics: List of PSI metrics for all monitored features
        ks_result: KS test result for predictions
        drift_detected: Whether any drift was detected
        drifting_features: List of features with detected drift
        timestamp: When the drift detection was performed
    """

    psi_metrics: list[PSIMetric] = Field(
        default_factory=list, description="PSI metrics for all features"
    )
    ks_result: KSTestResult | None = Field(None, description="KS test result")
    drift_detected: bool = Field(False, description="Whether any drift was detected")
    drifting_features: list[str] = Field(
        default_factory=list, description="Features with detected drift"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When detection was performed"
    )

    def has_drift(self) -> bool:
        """Check if any drift was detected.

        Returns:
            True if any feature drift or prediction drift was detected
        """
        return self.drift_detected


class DriftDetectorConfig(BaseModel):
    """Configuration for drift detection.

    Attributes:
        enabled: Whether drift detection is enabled
        check_interval_hours: How often to check for drift (hours)
        rolling_window_hours: Rolling window size for recent data (hours)
        retention_days: How long to keep drift metrics (days)
        psi_bins: Number of bins for PSI calculation
        psi_threshold_moderate: PSI threshold for moderate drift
        psi_threshold_severe: PSI threshold for severe drift
        ks_p_value_threshold: P-value threshold for KS test drift detection
    """

    enabled: bool = Field(True, description="Whether drift detection is enabled")
    check_interval_hours: int = Field(1, description="Check interval in hours")
    rolling_window_hours: int = Field(24, description="Rolling window size in hours")
    retention_days: int = Field(30, description="Retention period in days")

    # PSI configuration
    psi_bins: int = Field(10, description="Number of bins for PSI calculation")
    psi_threshold_moderate: float = Field(0.2, description="PSI threshold for moderate drift")
    psi_threshold_severe: float = Field(0.5, description="PSI threshold for severe drift")

    # KS test configuration
    ks_p_value_threshold: float = Field(
        0.05, description="P-value threshold for KS test drift detection"
    )

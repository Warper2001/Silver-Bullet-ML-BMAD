"""Statistical drift detection for ML model monitoring.

This package provides statistical drift detection using:
- Population Stability Index (PSI) for feature distribution monitoring
- Kolmogorov-Smirnov (KS) test for prediction distribution monitoring
- Continuous drift monitoring with configurable thresholds
- Historical validation and threshold sensitivity analysis
"""

from src.ml.drift_detection.drift_detector import (
    InsufficientDataError,
    InvalidBaselineError,
    StatisticalDriftDetector,
)
from src.ml.drift_detection.models import (
    DriftDetectionResult,
    DriftDetectorConfig,
    DriftEvent,
    KSTestResult,
    PSIMetric,
)
from src.ml.drift_detection.rolling_window_collector import RollingWindowCollector

__all__ = [
    "PSIMetric",
    "KSTestResult",
    "DriftEvent",
    "DriftDetectionResult",
    "DriftDetectorConfig",
    "StatisticalDriftDetector",
    "RollingWindowCollector",
    "InsufficientDataError",
    "InvalidBaselineError",
]

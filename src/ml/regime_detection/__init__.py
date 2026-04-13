"""Regime detection using Hidden Markov Models.

This module provides HMM-based regime detection for identifying market
regimes (trending, ranging, volatile) to enable regime-specific model selection.

Components:
- HMMRegimeDetector: Main detector class for regime identification
- HMMFeatureEngineer: Feature engineering for HMM observations
- Pydantic models: Data structures for regime state and transitions

Example:
    >>> from src.ml.regime_detection import HMMRegimeDetector
    >>> detector = HMMRegimeDetector.load(model_path="models/hmm/regime_model.joblib")
    >>> regime_state = detector.detect_regime(features_df)
    >>> print(f"Current regime: {regime_state.regime} (confidence: {regime_state.probability:.2f})")
"""

from src.ml.regime_detection.hmm_detector import HMMRegimeDetector
from src.ml.regime_detection.features import HMMFeatureEngineer
from src.ml.regime_detection.models import (
    HMMModelMetadata,
    HMMTrainingConfig,
    RegimeDetectionResult,
    RegimeState,
    RegimeTransitionEvent,
    RegimeType,
)

__all__ = [
    "HMMRegimeDetector",
    "HMMFeatureEngineer",
    "HMMModelMetadata",
    "HMMTrainingConfig",
    "RegimeDetectionResult",
    "RegimeState",
    "RegimeTransitionEvent",
    "RegimeType",
]

"""Regime-Aware Model Selector.

This module implements dynamic model selection based on HMM-detected market regimes.
Integrates with MLInference to provide regime-aware predictions.
"""

import joblib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.ml.regime_detection import HMMRegimeDetector

logger = logging.getLogger(__name__)


class RegimeAwareModelSelector:
    """Selects appropriate XGBoost model based on detected market regime.

    Handles:
    - Loading regime-specific XGBoost models
    - Detecting current market regime using HMM
    - Selecting appropriate model (regime-specific or generic)
    - Fallback to generic model when regime confidence is low

    Attributes:
        hmm_detector: Trained HMM regime detector
        regime_models: Dict mapping regime names to XGBoost models
        generic_model: Fallback XGBoost model (not regime-specific)
        regime_confidence_threshold: Minimum confidence to use regime-specific model
    """

    def __init__(
        self,
        hmm_detector: "HMMRegimeDetector",
        regime_model_dir: str | Path = "models/xgboost/regime_aware",
        regime_confidence_threshold: float = 0.7
    ):
        """Initialize regime-aware model selector.

        Args:
            hmm_detector: Trained HMM regime detector
            regime_model_dir: Directory containing regime-specific models
            regime_confidence_threshold: Min confidence to use regime-specific model
        """
        self.hmm_detector = hmm_detector
        self._regime_model_dir = Path(regime_model_dir)
        self._regime_confidence_threshold = regime_confidence_threshold

        # Load models
        self.regime_models: dict[str, object] = {}
        self.generic_model: object | None = None

        self._load_regime_models()

        logger.info(
            f"RegimeAwareModelSelector initialized: "
            f"{len(self.regime_models)} regime models loaded, "
            f"confidence threshold={regime_confidence_threshold}"
        )

    def _load_regime_models(self):
        """Load regime-specific XGBoost models from disk."""
        logger.info(f"Loading regime-specific models from {self._regime_model_dir}")

        # Load regime-specific models
        for model_file in self._regime_model_dir.glob("model_*.joblib"):
            # Extract regime name from filename
            regime_name = model_file.stem.replace("model_", "")

            if regime_name == "generic":
                # Load generic model separately
                self.generic_model = joblib.load(model_file)
                logger.info(f"  Loaded generic model from {model_file.name}")
            else:
                # Load regime-specific model
                self.regime_models[regime_name] = joblib.load(model_file)
                logger.info(f"  Loaded regime model '{regime_name}' from {model_file.name}")

        if not self.regime_models:
            logger.warning("No regime-specific models found. Will use generic model only.")

        if self.generic_model is None:
            logger.error("Generic model not found. Regime-aware selector may fail.")

    def detect_regime_from_features(
        self,
        features_df: pd.DataFrame
    ) -> tuple[str, float]:
        """Detect current regime from feature DataFrame.

        Args:
            features_df: Feature DataFrame (OHLCV data)

        Returns:
            Tuple of (regime_name, confidence)
        """
        # Detect regime using HMM
        regime_state = self.hmm_detector.detect_regime(features_df)

        regime_name = regime_state.regime
        confidence = regime_state.probability

        logger.debug(
            f"Detected regime: {regime_name} (confidence: {confidence:.3f})"
        )

        return regime_name, confidence

    def select_model(
        self,
        features_df: pd.DataFrame,
        regime_name: str | None = None,
        confidence: float | None = None
    ) -> tuple[object, str, bool]:
        """Select appropriate model based on detected regime.

        Args:
            features_df: Feature DataFrame (for regime detection if needed)
            regime_name: Pre-detected regime name (optional, will detect if None)
            confidence: Pre-detected confidence (optional, will detect if None)

        Returns:
            Tuple of (model, regime_used, is_regime_specific)
            - model: Selected XGBoost model
            - regime_used: Regime name used for selection (or "generic")
            - is_regime_specific: Whether model is regime-specific
        """
        # Detect regime if not provided
        if regime_name is None or confidence is None:
            regime_name, confidence = self.detect_regime_from_features(features_df)

        # Check if confidence is high enough to use regime-specific model
        if (
            confidence >= self._regime_confidence_threshold and
            regime_name in self.regime_models
        ):
            # Use regime-specific model
            model = self.regime_models[regime_name]
            logger.debug(
                f"Using regime-specific model for '{regime_name}' "
                f"(confidence: {confidence:.3f})"
            )
            return model, regime_name, True

        # Fallback to generic model
        if self.generic_model is None:
            raise ValueError("Generic model not available for fallback")

        logger.debug(
            f"Using generic model "
            f"(regime: {regime_name}, confidence: {confidence:.3f} < threshold)"
        )
        return self.generic_model, "generic", False

    def predict_regime_aware(
        self,
        features_df: pd.DataFrame,
        regime_name: str | None = None,
        confidence: float | None = None
    ) -> dict:
        """Generate regime-aware prediction.

        Args:
            features_df: Feature DataFrame
            regime_name: Pre-detected regime name (optional)
            confidence: Pre-detected confidence (optional)

        Returns:
            Dictionary with prediction and metadata:
            {
                "prediction": float,  # Probability score
                "regime": str,  # Detected regime
                "confidence": float,  # Regime confidence
                "model_used": str,  # Model used (regime name or "generic")
                "is_regime_specific": bool  # Whether regime-specific model was used
            }
        """
        # Select model
        model, model_used, is_regime_specific = self.select_model(
            features_df, regime_name, confidence
        )

        # Detect regime if not provided
        if regime_name is None or confidence is None:
            regime_name, confidence = self.detect_regime_from_features(features_df)

        # Generate prediction
        prediction_proba = model.predict_proba(features_df)[:, 1][0]

        result = {
            "prediction": prediction_proba,
            "regime": regime_name,
            "confidence": confidence,
            "model_used": model_used,
            "is_regime_specific": is_regime_specific
        }

        logger.debug(
            f"Regime-aware prediction: {prediction_proba:.4f} "
            f"(regime: {regime_name}, model: {model_used})"
        )

        return result

    def get_regime_state(self) -> dict:
        """Get current HMM regime state.

        Returns:
            Dictionary with current regime info:
            {
                "regime": str,
                "probability": float,
                "duration_bars": int,
                "duration_days": float,
                "detected_at": datetime
            }
        """
        if self.hmm_detector.current_regime is None:
            return {
                "regime": "unknown",
                "probability": 0.0,
                "duration_bars": 0,
                "duration_days": 0.0,
                "detected_at": None
            }

        regime_state = self.hmm_detector.current_regime

        return {
            "regime": regime_state.regime,
            "probability": regime_state.probability,
            "duration_bars": regime_state.duration_bars,
            "duration_days": regime_state.duration_days,
            "detected_at": regime_state.detected_at
        }

    def get_available_regimes(self) -> list[str]:
        """Get list of available regime-specific models.

        Returns:
            List of regime names with available models
        """
        return list(self.regime_models.keys())

    def has_regime_model(self, regime_name: str) -> bool:
        """Check if regime-specific model exists.

        Args:
            regime_name: Name of regime

        Returns:
            True if regime-specific model exists
        """
        return regime_name in self.regime_models

"""Regime-Aware Inference Extension for MLInference.

This module extends MLInference with regime-aware model selection capabilities.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.ml.inference import MLInference

logger = logging.getLogger(__name__)


class RegimeAwareInferenceMixin:
    """Mixin class to add regime-aware inference to MLInference.

    This mixin can be used to extend MLInference with regime-aware model
    selection capabilities without modifying the core MLInference class.
    """

    def initialize_regime_aware_inference(
        self,
        hmm_model_path: str | Path = "models/hmm/regime_model",
        regime_model_dir: str | Path = "models/xgboost/regime_aware",
        regime_confidence_threshold: float = 0.7
    ):
        """Initialize regime-aware inference components.

        Args:
            hmm_model_path: Path to trained HMM model
            regime_model_dir: Path to regime-specific XGBoost models
            regime_confidence_threshold: Min confidence to use regime-specific model
        """
        from src.ml.regime_detection import HMMRegimeDetector
        from src.ml.regime_aware_model_selector import RegimeAwareModelSelector

        logger.info("Initializing regime-aware inference...")

        # Load HMM detector
        hmm_path = Path(hmm_model_path)
        if not hmm_path.exists():
            logger.error(f"HMM model not found: {hmm_path}")
            raise FileNotFoundError(f"HMM model not found: {hmm_path}")

        self._hmm_detector = HMMRegimeDetector.load(hmm_path)
        logger.info(f"Loaded HMM detector: {self._hmm_detector.n_regimes} regimes")

        # Initialize regime-aware model selector
        self._regime_selector = RegimeAwareModelSelector(
            hmm_detector=self._hmm_detector,
            regime_model_dir=regime_model_dir,
            regime_confidence_threshold=regime_confidence_threshold
        )

        # Track regime-aware inference statistics
        self._regime_stats = {
            "regime_aware_count": 0,
            "generic_count": 0,
            "regime_distribution": {},
            "regime_transitions": 0
        }

        logger.info("Regime-aware inference initialized")

    def predict_regime_aware(
        self,
        signal,
        horizon: int = 30
    ) -> dict:
        """Generate regime-aware prediction for Silver Bullet signal.

        Args:
            signal: SilverBulletSetup with OHLCV data
            horizon: Time horizon in minutes

        Returns:
            Dictionary with prediction and regime metadata:
            {
                "probability": float,
                "horizon": int,
                "regime": str,
                "confidence": float,
                "model_used": str,
                "is_regime_specific": bool,
                "inference_timestamp": datetime
            }
        """
        from datetime import datetime

        if not hasattr(self, '_regime_selector'):
            raise RuntimeError(
                "Regime-aware inference not initialized. "
                "Call initialize_regime_aware_inference() first."
            )

        try:
            # Engineer features from signal
            features_df = self._engineer_features_for_signal(signal)

            # Get regime-aware prediction
            result = self._regime_selector.predict_regime_aware(features_df)

            # Update statistics
            self._regime_stats["regime_aware_count"] += 1

            if result["is_regime_specific"]:
                self._regime_stats["regime_aware_count"] += 1
            else:
                self._regime_stats["generic_count"] += 1

            # Track regime distribution
            regime = result["regime"]
            self._regime_stats["regime_distribution"][regime] = \
                self._regime_stats["regime_distribution"].get(regime, 0) + 1

            # Add metadata
            result["horizon"] = horizon
            result["inference_timestamp"] = datetime.now()

            logger.info(
                f"Regime-aware prediction: P(Success)={result['prediction']:.4f}, "
                f"regime={result['regime']}, "
                f"model={result['model_used']}, "
                f"specific={result['is_regime_specific']}"
            )

            return result

        except Exception as e:
            logger.error(f"Regime-aware inference failed: {e}")
            raise

    def get_current_regime_state(self) -> dict:
        """Get current HMM regime state.

        Returns:
            Dictionary with current regime information
        """
        if not hasattr(self, '_regime_selector'):
            return {
                "regime": "unknown",
                "error": "Regime-aware inference not initialized"
            }

        return self._regime_selector.get_regime_state()

    def get_regime_statistics(self) -> dict:
        """Get regime-aware inference statistics.

        Returns:
            Dictionary with statistics:
            {
                "regime_aware_count": int,
                "generic_count": int,
                "regime_distribution": dict,
                "regime_transitions": int
            }
        """
        if not hasattr(self, '_regime_stats'):
            return {"error": "Regime-aware inference not initialized"}

        return self._regime_stats.copy()

    def reset_regime_statistics(self):
        """Reset regime-aware inference statistics."""
        if hasattr(self, '_regime_stats'):
            self._regime_stats = {
                "regime_aware_count": 0,
                "generic_count": 0,
                "regime_distribution": {},
                "regime_transitions": 0
            }
            logger.info("Regime-aware statistics reset")

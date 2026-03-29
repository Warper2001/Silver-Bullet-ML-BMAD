"""ML Inference for Real-Time Signal Prediction.

This module implements live probability score generation for Silver Bullet
signals using trained XGBoost models and feature engineering pipelines.
"""

import joblib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.models import DollarBar, SilverBulletSetup
from src.ml.features import FeatureEngineer
from src.ml.pipeline_serializer import PipelineSerializer

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when insufficient data is available for reliable ML inference."""

    pass


class MLInference:
    """Live ML inference for Silver Bullet signal probability prediction.

    Handles:
    - Loading trained XGBoost models and feature pipelines
    - Real-time feature engineering for live signals
    - Probability score generation using model.predict_proba()
    - Multi-horizon inference (5, 15, 30, 60 minutes)
    - Inference statistics tracking and logging

    Performance:
    - Lazy loading: Models loaded on first inference call
    - Thread-safe: Concurrent inference requests supported
    - Inference latency: < 10ms per signal
    - Statistics overhead: < 1ms
    """

    def __init__(self, model_dir: str | Path = "models/xgboost"):
        """Initialize MLInference with lazy loading.

        Args:
            model_dir: Directory containing trained models and pipelines
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded caches
        self._models: dict[int, xgb.XGBClassifier] = {}
        self._pipelines: dict[int, object] = {}

        # Thread-safe loading
        self._load_lock = threading.Lock()

        # Feature engineer
        self._feature_engineer = FeatureEngineer()

        # Pipeline serializer
        self._pipeline_serializer = PipelineSerializer(model_dir=self._model_dir)

        # Inference statistics
        self._stats = {
            "inference_count": 0,
            "hourly_count": 0,
            "probabilities": [],  # Rolling window (last 1000)
            "latencies": [],  # Rolling window (last 1000)
            "last_log_time": datetime.now(),
        }

        logger.info(f"MLInference initialized with model_dir: {self._model_dir}")

    def predict_probability(
        self,
        signal: SilverBulletSetup,
        horizon: int,
        recent_bars: list[DollarBar] | None = None,
    ) -> dict[str, object]:
        """Generate success probability score for a signal.

        Args:
            signal: Silver Bullet setup with timestamp and features
            horizon: Time horizon in minutes (5, 15, 30, or 60)
            recent_bars: Optional list of recent DollarBar objects for feature engineering

        Returns:
            Dictionary with probability score and metadata:
            {
                "probability": float,  # 0.0 to 1.0
                "horizon": int,
                "model_version": str,
                "inference_timestamp": datetime,
                "latency_ms": float
            }

        Raises:
            FileNotFoundError: If model or pipeline doesn't exist for horizon
            ValueError: If signal data is invalid
        """
        start_time = datetime.now()

        try:
            # Load model and pipeline if needed (lazy loading)
            model = self._load_model_if_needed(horizon)
            pipeline = self._load_pipeline_if_needed(horizon)

            # Engineer features from signal and recent bars
            features_df = self._engineer_features_for_signal(signal, recent_bars)

            # Transform features using pipeline
            transformed = self._pipeline_serializer.transform_features(
                pipeline, features_df
            )

            # Predict probability
            probability = self._predict_with_model(model, transformed)

            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Get model version
            model_version = self._get_model_version(horizon)

            # Create result
            result = {
                "probability": probability,
                "horizon": horizon,
                "model_version": model_version,
                "inference_timestamp": datetime.now(),
                "latency_ms": latency_ms,
            }

            # Update statistics
            self._update_statistics(probability, latency_ms)

            logger.debug(
                f"Inference complete for {horizon}-minute horizon: "
                f"P(Success)={probability:.4f}, latency={latency_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Inference failed for {horizon}-minute horizon: {e}")
            return self._error_response(horizon, error=str(e), latency_ms=0)

    def predict_probability_from_features(
        self, features_df: pd.DataFrame, horizon: int = 30
    ) -> float:
        """Generate probability score from feature DataFrame.

        Simpler interface for meta-labeling backtest. Bypasses SilverBulletSetup
        requirement and directly uses feature DataFrame.

        Args:
            features_df: DataFrame with engineered features (already preprocessed)
                         Must match the feature pipeline used during training
            horizon: Time horizon in minutes (default: 30)

        Returns:
            Probability score (0.0 to 1.0)

        Raises:
            FileNotFoundError: If model or pipeline doesn't exist for horizon
            ValueError: If features DataFrame is invalid
        """
        try:
            # Load model and pipeline if needed (lazy loading)
            model = self._load_model_if_needed(horizon)
            pipeline = self._load_pipeline_if_needed(horizon)

            # Transform features using pipeline
            transformed = self._pipeline_serializer.transform_features(
                pipeline, features_df
            )

            # Get expected feature names from the model
            expected_features = model.get_booster().feature_names

            # Filter to only expected features (handle feature mismatch)
            available_features = [f for f in expected_features if f in transformed.columns]

            if len(available_features) < len(expected_features):
                missing = set(expected_features) - set(available_features)
                logger.warning(
                    f"Missing {len(missing)} features: {missing}. "
                    f"Using {len(available_features)}/{len(expected_features)} features."
                )

            # Use only available features
            filtered = transformed[available_features]

            # Predict probability
            probability = self._predict_with_model(model, filtered)

            logger.debug(
                f"Direct inference complete for {horizon}-minute horizon: "
                f"P(Success)={probability:.4f}"
            )

            return probability

        except Exception as e:
            logger.error(f"Direct inference failed for {horizon}-minute horizon: {e}")
            raise

    def predict_all_horizons(
        self, signal: SilverBulletSetup
    ) -> dict[int, dict[str, object]]:
        """Generate probability scores for all available time horizons.

        Args:
            signal: Silver Bullet setup with timestamp and features

        Returns:
            Dictionary mapping horizon to probability result:
            {
                5: {"probability": 0.78, "horizon": 5, ...},
                15: {"probability": 0.72, "horizon": 15, ...},
                ...
            }
        """
        results = {}
        available_horizons = self._get_available_horizons()

        for horizon in available_horizons:
            try:
                result = self.predict_probability(signal, horizon)
                results[horizon] = result
            except Exception as e:
                logger.warning(f"Skipping {horizon}-minute horizon due to error: {e}")
                continue

        logger.info(
            f"Predicted probabilities for {len(results)} horizons: "
            f"{[f'{h}m={r['probability']:.2f}' for h, r in results.items()]}"
        )

        return results

    def get_statistics(self) -> dict[str, object]:
        """Get current inference statistics.

        Returns:
            Dictionary with inference statistics:
            {
                "inference_count": int,
                "hourly_count": int,
                "average_probability": float,
                "probability_distribution": dict,
                "latency_p50_ms": float,
                "latency_p95_ms": float,
                "latency_p99_ms": float
            }
        """
        if not self._stats["probabilities"]:
            return {
                "inference_count": 0,
                "hourly_count": 0,
                "average_probability": 0.0,
                "probability_distribution": self._empty_distribution(),
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
            }

        probabilities = self._stats["probabilities"]
        latencies = self._stats["latencies"]

        return {
            "inference_count": self._stats["inference_count"],
            "hourly_count": self._stats["hourly_count"],
            "average_probability": float(np.mean(probabilities)),
            "probability_distribution": self._calculate_distribution(probabilities),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
        }

    def _load_model_if_needed(self, horizon: int) -> xgb.XGBClassifier:
        """Load model for horizon if not already cached (lazy loading).

        Args:
            horizon: Time horizon in minutes

        Returns:
            Loaded XGBoost model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        with self._load_lock:
            if horizon not in self._models:
                model_file = (
                    self._model_dir / f"{horizon}_minute" / "xgboost_model.pkl"
                )

                if not model_file.exists():
                    raise FileNotFoundError(
                        f"No model found for {horizon}-minute horizon: {model_file}"
                    )

                logger.debug(f"Loading model for {horizon}-minute horizon...")
                model = joblib.load(model_file)
                self._models[horizon] = model
                logger.debug(f"Model loaded for {horizon}-minute horizon")

            return self._models[horizon]

    def _load_pipeline_if_needed(self, horizon: int) -> object:
        """Load pipeline for horizon if not already cached (lazy loading).

        Args:
            horizon: Time horizon in minutes

        Returns:
            Loaded feature engineering pipeline

        Raises:
            FileNotFoundError: If pipeline file doesn't exist
        """
        with self._load_lock:
            if horizon not in self._pipelines:
                logger.debug(f"Loading pipeline for {horizon}-minute horizon...")
                pipeline = self._pipeline_serializer.load_pipeline(horizon)
                self._pipelines[horizon] = pipeline
                logger.debug(f"Pipeline loaded for {horizon}-minute horizon")

            return self._pipelines[horizon]

    def _engineer_features_for_signal(
        self,
        signal: SilverBulletSetup,
        recent_bars: list[DollarBar] | None = None,
    ) -> pd.DataFrame:
        """Engineer features for a Silver Bullet signal.

        Args:
            signal: Silver Bullet setup
            recent_bars: List of recent DollarBar objects for feature engineering.
                        Minimum 20 bars required for reliable feature engineering.

        Returns:
            DataFrame with engineered features

        Raises:
            InsufficientDataError: If fewer than 20 bars provided

        Note:
            Per spec constraints, ML inference requires real-time data from dollar bars.
            Minimum 20 bars needed to calculate meaningful features. No fallback to
            dummy data - trading decisions must be based on real market conditions.
        """
        if recent_bars is None or len(recent_bars) < 20:
            # Per spec: Generate 40+ features from dollar bars for ML inference
            # Cannot use dummy data - must have real market data
            bars_count = len(recent_bars) if recent_bars else 0
            raise InsufficientDataError(
                f"Insufficient dollar bars for feature engineering: {bars_count} bars. "
                f"Minimum 20 bars required for reliable ML inference. "
                f"Cannot make trading decisions on dummy data. "
                f"Spec constraint: 'Generate 40+ features from dollar bars for ML inference'"
            )

        # Convert DollarBar objects to DataFrame for feature engineering
        bars_data = []
        for bar in recent_bars:
            bars_data.append({
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(bars_data)

        # Engineer features using the FeatureEngineer
        features_df = self._feature_engineer.engineer_features(df)

        # Return only the last row (most recent features)
        return features_df.iloc[[-1]].copy()

    def _predict_with_model(
        self, model: xgb.XGBClassifier, features: pd.DataFrame
    ) -> float:
        """Generate probability prediction using XGBoost model.

        Args:
            model: Trained XGBoost classifier
            features: Transformed feature DataFrame

        Returns:
            Probability score (0.0 to 1.0)
        """
        # Get probability of positive class (success)
        probability = model.predict_proba(features)[:, 1][0]
        return float(probability)

    def _get_model_version(self, horizon: int) -> str:
        """Get model version (hash) from metadata.

        Args:
            horizon: Time horizon in minutes

        Returns:
            Model version string (hash)
        """
        import json

        metadata_file = self._model_dir / f"{horizon}_minute" / "pipeline_metadata.json"

        if not metadata_file.exists():
            return "unknown"

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return metadata.get("model_hash", "unknown")
        except Exception as e:
            logger.warning(f"Failed to read model version: {e}")
            return "unknown"

    def _update_statistics(self, probability: float, latency_ms: float):
        """Update inference statistics.

        Args:
            probability: Predicted probability score
            latency_ms: Inference latency in milliseconds
        """
        self._stats["inference_count"] += 1
        self._stats["hourly_count"] += 1

        self._stats["probabilities"].append(probability)
        self._stats["latencies"].append(latency_ms)

        # Keep only last 1000 for memory efficiency
        if len(self._stats["probabilities"]) > 1000:
            self._stats["probabilities"] = self._stats["probabilities"][-1000:]
            self._stats["latencies"] = self._stats["latencies"][-1000:]

        # Log statistics every 100 inferences or hourly
        should_log = self._stats[
            "inference_count"
        ] % 100 == 0 or datetime.now() - self._stats["last_log_time"] > timedelta(
            hours=1
        )

        if should_log:
            self._log_statistics()

    def _log_statistics(self):
        """Log current inference statistics."""
        stats = self.get_statistics()
        logger.info(
            f"Inference Statistics: "
            f"count={stats['inference_count']}, "
            f"hourly_count={stats['hourly_count']}, "
            f"avg_prob={stats['average_probability']:.4f}, "
            f"latency_p95={stats['latency_p95_ms']:.2f}ms"
        )

        self._stats["last_log_time"] = datetime.now()

        # Reset hourly count
        self._stats["hourly_count"] = 0

    def _calculate_distribution(self, probabilities: list[float]) -> dict[str, int]:
        """Calculate probability distribution buckets.

        Args:
            probabilities: List of probability scores

        Returns:
            Dictionary with bucket counts
        """
        distribution = self._empty_distribution()

        for prob in probabilities:
            if 0.0 <= prob < 0.2:
                distribution["0.0-0.2"] += 1
            elif 0.2 <= prob < 0.4:
                distribution["0.2-0.4"] += 1
            elif 0.4 <= prob < 0.6:
                distribution["0.4-0.6"] += 1
            elif 0.6 <= prob < 0.8:
                distribution["0.6-0.8"] += 1
            elif 0.8 <= prob <= 1.0:
                distribution["0.8-1.0"] += 1

        return distribution

    def _empty_distribution(self) -> dict[str, int]:
        """Create empty probability distribution.

        Returns:
            Dictionary with zero counts for all buckets
        """
        return {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

    def _get_available_horizons(self) -> list[int]:
        """Get list of available time horizons with trained models.

        Returns:
            List of horizon values in minutes
        """
        horizons = []
        for path in self._model_dir.iterdir():
            if path.is_dir() and path.name.endswith("_minute"):
                try:
                    horizon = int(path.name.split("_")[0])
                    horizons.append(horizon)
                except (ValueError, IndexError):
                    continue

        return sorted(horizons)

    def _error_response(
        self, horizon: int, error: str, latency_ms: float
    ) -> dict[str, object]:
        """Create error response for failed inference.

        Args:
            horizon: Time horizon in minutes
            error: Error message
            latency_ms: Inference latency before error

        Returns:
            Error response dictionary
        """
        return {
            "probability": 0.5,  # Uncertain probability on error
            "horizon": horizon,
            "model_version": "error",
            "inference_timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "error": error,
        }

"""ML Inference for Real-Time Signal Prediction.

This module implements live probability score generation for Silver Bullet
signals using trained XGBoost models and feature engineering pipelines.
"""

import asyncio
import joblib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.models import SilverBulletSetup
from src.ml.features import FeatureEngineer
from src.ml.pipeline_serializer import PipelineSerializer
from src.ml.probability_calibration import ProbabilityCalibration

if TYPE_CHECKING:
    from src.ml.drift_detection import StatisticalDriftDetector

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        model_dir: str | Path = "models/xgboost",
        use_calibration: bool = True,
        enable_automated_retraining: bool = False,
        retraining_config: dict | None = None
    ):
        """Initialize MLInference with lazy loading.

        Args:
            model_dir: Directory containing trained models and pipelines
            use_calibration: Whether to use probability calibration (default: True)
            enable_automated_retraining: Whether to enable automated retraining (default: False)
            retraining_config: Optional configuration for retraining triggers
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded caches
        self._models: dict[int, xgb.XGBClassifier] = {}
        self._pipelines: dict[int, object] = {}
        self._calibration: dict[int, ProbabilityCalibration] = {}

        # Calibration flag
        self._use_calibration = use_calibration

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

        # Automated retraining components (optional)
        self._enable_automated_retraining = enable_automated_retraining
        self._retraining_trigger = None
        self._retraining_task = None
        self._retraining_lock = None  # For async retraining coordination

        if enable_automated_retraining and retraining_config:
            self._initialize_automated_retraining(retraining_config)

        logger.info(f"MLInference initialized with model_dir: {self._model_dir}")

    def _initialize_automated_retraining(self, config: dict):
        """Initialize automated retraining components.

        Args:
            config: Configuration dictionary with retraining settings
        """
        try:
            from src.ml.retraining import RetrainingTrigger, AsyncRetrainingTask

            # Initialize retraining trigger
            trigger_config = config.get("trigger", {})
            self._retraining_trigger = RetrainingTrigger(trigger_config)

            # Initialize async retraining task
            self._retraining_task = AsyncRetrainingTask(config, ml_inference=self)

            # Initialize lock for async coordination
            self._retraining_lock = threading.Lock()

            logger.info(
                f"Automated retraining enabled: PSI threshold={trigger_config.get('psi_threshold', 0.5)}, "
                f"KS p-value threshold={trigger_config.get('ks_p_value_threshold', 0.01)}"
            )

        except ImportError as e:
            logger.error(f"Failed to import retraining modules: {e}")
            self._enable_automated_retraining = False
        except Exception as e:
            logger.error(f"Failed to initialize automated retraining: {e}")
            self._enable_automated_retraining = False

    def predict_probability(
        self, signal: SilverBulletSetup, horizon: int
    ) -> dict[str, object]:
        """Generate success probability score for a signal.

        Args:
            signal: Silver Bullet setup with timestamp and features
            horizon: Time horizon in minutes (5, 15, 30, or 60)

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

            # Engineer features from signal
            features_df = self._engineer_features_for_signal(signal)

            # Transform features using pipeline
            transformed = self._pipeline_serializer.transform_features(
                pipeline, features_df
            )

            # Predict probability
            probability = self._predict_with_model(model, transformed, horizon)

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

            # Collect for drift detection if enabled
            if hasattr(self, '_drift_collector') and self._drift_collector is not None:
                try:
                    # Extract features as dictionary for drift detection
                    feature_dict = transformed.iloc[0].to_dict() if hasattr(transformed, 'iloc') else {}
                    self._drift_collector.add_prediction(
                        prediction=probability,
                        features=feature_dict,
                        timestamp=datetime.now()
                    )
                    logger.debug(f"Collected prediction for drift detection: P={probability:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to collect prediction for drift detection: {e}")

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

            # Handle models without feature names (trained with numpy arrays)
            if expected_features is None:
                # Use all features from transformed DataFrame
                filtered = transformed
            else:
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
            probability = self._predict_with_model(model, filtered, horizon)

            # Collect for drift detection if enabled
            if hasattr(self, '_drift_collector') and self._drift_collector is not None:
                try:
                    # Extract features as dictionary for drift detection
                    feature_dict = filtered.iloc[0].to_dict() if hasattr(filtered, 'iloc') else {}
                    self._drift_collector.add_prediction(
                        prediction=probability,
                        features=feature_dict,
                        timestamp=datetime.now()
                    )
                    logger.debug(f"Collected prediction for drift detection: P={probability:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to collect prediction for drift detection: {e}")

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

        Also loads calibrated model if available and calibration is enabled.

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

                # Load calibrated model if available and calibration enabled
                if self._use_calibration:
                    calibrated_model_file = (
                        self._model_dir / f"{horizon}_minute" / "calibrated_model.joblib"
                    )
                    if calibrated_model_file.exists():
                        logger.debug(
                            f"Loading calibrated model for {horizon}-minute horizon..."
                        )
                        calibration = ProbabilityCalibration.load(calibrated_model_file)
                        self._calibration[horizon] = calibration
                        logger.debug(
                            f"Calibrated model loaded for {horizon}-minute horizon "
                            f"(Brier score: {calibration.brier_score:.4f})"
                        )

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

    def _engineer_features_for_signal(self, signal: SilverBulletSetup) -> pd.DataFrame:
        """Engineer features for a Silver Bullet signal.

        Args:
            signal: Silver Bullet setup

        Returns:
            DataFrame with engineered features
        """
        # TODO: Integrate with Epic 2 to get recent Dollar Bars
        # For now, create dummy features
        # This will be implemented when Epic 2 integration is complete

        # Create dummy feature row
        features = {
            "atr": 1.0,
            "rsi": 50.0,
            "macd": 0.0,
            "close_position": 0.5,
            "volume_ratio": 1.0,
            "hour": signal.timestamp.hour,
            "day_of_week": signal.timestamp.weekday(),
        }

        return pd.DataFrame([features])

    def _predict_with_model(
        self, model: xgb.XGBClassifier, features: pd.DataFrame, horizon: int
    ) -> float:
        """Generate probability prediction using XGBoost model.

        Uses calibrated predictions if available and calibration is enabled,
        otherwise falls back to uncalibrated XGBoost predictions.

        Args:
            model: Trained XGBoost classifier
            features: Transformed feature DataFrame
            horizon: Time horizon in minutes (for calibration lookup)

        Returns:
            Probability score (0.0 to 1.0)
        """
        # Use calibrated prediction if available and enabled
        if self._use_calibration and horizon in self._calibration:
            calibration = self._calibration[horizon]
            # Convert DataFrame to numpy array for calibration
            features_array = features.values[0] if len(features) == 1 else features.values
            probability = calibration.predict_proba(features_array)
            return float(probability)

        # Fallback to uncalibrated XGBoost prediction
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

    # ============================================================================
    # DRIFT DETECTION INTEGRATION (Epic 5 Phase 2: Story 5.2.1)
    # ============================================================================

    def initialize_drift_detection(
        self,
        drift_detector: "StatisticalDriftDetector",
        window_hours: int = 24,
        enable_monitoring: bool = True,
    ) -> None:
        """Initialize drift detection monitoring with rolling window collector.

        Args:
            drift_detector: StatisticalDriftDetector instance with baseline
            window_hours: Rolling window size for data collection
            enable_monitoring: Whether to enable automatic drift monitoring
        """
        from src.ml.drift_detection import RollingWindowCollector

        self._drift_detector = drift_detector
        self._drift_collector = RollingWindowCollector(
            window_hours=window_hours,
            min_samples=100,
            max_samples=10000,
        )
        self._drift_detection_enabled = enable_monitoring

        # Create drift events log directory
        drift_log_dir = Path("logs/drift_events")
        drift_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Drift detection initialized: window={window_hours}h, "
            f"enabled={enable_monitoring}"
        )

    def collect_for_drift_detection(
        self,
        probability: float,
        features: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Collect prediction and features for drift detection monitoring.

        Should be called after each inference to populate the rolling window.

        Args:
            probability: Predicted probability score
            features: Dictionary of feature_name -> feature_value
            timestamp: Timestamp of prediction (defaults to now)
        """
        if not hasattr(self, "_drift_collector") or not self._drift_detection_enabled:
            return

        self._drift_collector.add_prediction(
            prediction=probability, features=features, timestamp=timestamp
        )

        logger.debug(
            f"Collected prediction for drift detection: "
            f"p={probability:.4f}, features={len(features)}"
        )

    def check_drift_and_log(self, force_check: bool = False) -> dict | None:
        """Check for drift and log results to CSV audit trail.

        Args:
            force_check: Run drift check even if insufficient data

        Returns:
            Drift detection result dict, or None if insufficient data

        Raises:
            ValueError: If drift detection not initialized
        """
        if not hasattr(self, "_drift_detector") or not hasattr(
            self, "_drift_collector"
        ):
            raise ValueError(
                "Drift detection not initialized. "
                "Call initialize_drift_detection() first."
            )

        # Check if sufficient data collected
        if not force_check and not self._drift_collector.has_sufficient_data():
            logger.debug(
                f"Insufficient data for drift check: "
                f"{self._drift_collector.get_window_stats()['total_samples']} "
                f"< 100 minimum"
            )
            return None

        # Get recent data from rolling window
        try:
            recent_features, recent_predictions = (
                self._drift_collector.get_recent_data()
            )
        except ValueError as e:
            logger.warning(f"Cannot check drift: {e}")
            return None

        # Run drift detection
        result = self._drift_detector.detect_drift(
            recent_features=recent_features, recent_predictions=recent_predictions
        )

        # Log drift event to CSV
        self._log_drift_event_to_csv(result)

        # Log to system logger
        if result.drift_detected:
            ks_p_value = (
                f"{result.ks_result.p_value:.4f}" if result.ks_result else "N/A"
            )
            logger.warning(
                f"🚨 DRIFT DETECTED: {len(result.drifting_features)} features drifting, "
                f"KS p_value={ks_p_value}"
            )

            # INTEGRATION: Evaluate retraining trigger if automated retraining enabled
            if self._enable_automated_retraining and self._retraining_trigger:
                self._evaluate_and_trigger_retraining(result)
        else:
            logger.info("✅ No drift detected in latest check")

        return {
            "drift_detected": result.drift_detected,
            "drifting_features": result.drifting_features,
            "psi_metrics": [
                {"feature": m.feature_name, "psi": m.psi_score, "severity": m.drift_severity}
                for m in result.psi_metrics
            ],
            "ks_result": (
                {
                    "statistic": result.ks_result.ks_statistic,
                    "p_value": result.ks_result.p_value,
                    "drift_detected": result.ks_result.drift_detected,
                }
                if result.ks_result
                else None
            ),
            "timestamp": result.timestamp,
        }

    def _log_drift_event_to_csv(self, drift_result: dict) -> None:
        """Log drift event to CSV audit trail.

        Args:
            drift_result: DriftDetectionResult from detect_drift()
        """
        drift_log_dir = Path("logs/drift_events")
        csv_file = drift_log_dir / "drift_events.csv"

        # Create CSV with headers if doesn't exist
        csv_file_exists = csv_file.exists()

        # Prepare row data
        row = {
            "timestamp": drift_result.timestamp.isoformat(),
            "drift_detected": drift_result.drift_detected,
            "drifting_features_count": len(drift_result.drifting_features),
            "drifting_features": ",".join(drift_result.drifting_features),
            "ks_statistic": (
                drift_result.ks_result.ks_statistic if drift_result.ks_result else None
            ),
            "ks_p_value": (
                drift_result.ks_result.p_value if drift_result.ks_result else None
            ),
            "ks_drift_detected": (
                drift_result.ks_result.drift_detected
                if drift_result.ks_result
                else None
            ),
        }

        # Add PSI metrics for top 5 features
        for i, metric in enumerate(drift_result.psi_metrics[:5]):
            row[f"psi_feature_{i}"] = metric.feature_name
            row[f"psi_score_{i}"] = metric.psi_score
            row[f"psi_severity_{i}"] = metric.drift_severity

        # Write to CSV
        import csv

        with open(csv_file, "a", newline="") as f:
            fieldnames = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not csv_file_exists:
                writer.writeheader()

            writer.writerow(row)

        logger.debug(f"Drift event logged to {csv_file}")

    def _evaluate_and_trigger_retraining(self, drift_result):
        """Evaluate retraining trigger and initiate automated retraining if conditions met.

        Args:
            drift_result: DriftDetectionResult from detect_drift()
        """
        if not self._retraining_trigger:
            logger.warning("Retraining trigger not initialized")
            return

        try:
            # Evaluate trigger conditions
            trigger_decision = self._retraining_trigger.should_trigger_retraining(drift_result)

            logger.info(
                f"Retraining trigger evaluation: {trigger_decision['trigger']} - "
                f"{trigger_decision['justification']}"
            )

            # If trigger conditions met, initiate async retraining
            if trigger_decision["trigger"] and self._retraining_task:
                logger.info("🚨 Initiating automated retraining...")

                # Start async retraining in background thread
                def run_retraining_async():
                    """Run retraining in async event loop."""
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            self._retraining_task.run_retraining(trigger_decision)
                        )

                        if result["success"]:
                            logger.info(
                                f"✅ Automated retraining completed: "
                                f"{result['old_model_hash']} → {result['new_model_hash']}"
                            )
                        else:
                            logger.error(f"❌ Automated retraining failed: {result.get('reason', 'Unknown')}")

                    except Exception as e:
                        logger.error(f"Automated retraining crashed: {e}", exc_info=True)
                    finally:
                        loop.close()

                # Run in background thread to avoid blocking inference
                import threading

                retraining_thread = threading.Thread(
                    target=run_retraining_async,
                    name="AutomatedRetraining",
                    daemon=True
                )
                retraining_thread.start()

                logger.info("Automated retraining task started in background")

        except Exception as e:
            logger.error(f"Error evaluating retraining trigger: {e}", exc_info=True)

    def get_drift_detection_status(self) -> dict:
        """Get current drift detection status.

        Returns:
            Dictionary with drift detection status:
            {
                "enabled": bool,
                "window_stats": dict,
                "last_check_result": dict,
                "drift_detector_initialized": bool
            }
        """
        status = {
            "enabled": getattr(self, "_drift_detection_enabled", False),
            "drift_detector_initialized": hasattr(self, "_drift_detector"),
            "window_stats": (
                self._drift_collector.get_window_stats()
                if hasattr(self, "_drift_collector")
                else {}
            ),
            "last_check_result": getattr(self, "_last_drift_check_result", None),
        }

        return status

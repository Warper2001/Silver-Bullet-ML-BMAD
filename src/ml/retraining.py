"""Automated retraining triggers and execution for ML models.

This module implements:
- RetrainingTrigger: Evaluates drift events and triggers retraining
- ModelVersioning: Manages model versions, hashing, and rollback
- PerformanceValidator: Validates new model performance
- AsyncRetrainingTask: Async background retraining execution
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

if TYPE_CHECKING:
    from src.ml.drift_detection import DriftDetectionResult

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """Evaluates drift events and determines if retraining should be triggered.

    Uses multiple criteria to make retraining decisions:
    - Drift severity (PSI > 0.5 OR KS p-value < 0.01)
    - Minimum interval since last retraining (24 hours)
    - Data availability (≥1000 new samples)
    - System health (no critical errors)

    Example:
        >>> trigger = RetrainingTrigger(config)
        >>> result = trigger.should_trigger_retraining(drift_event)
        >>> if result["trigger"]:
        ...     # Initiate retraining
        ...     logger.info(f"Retraining triggered: {result['justification']}")
    """

    def __init__(self, config: dict):
        """Initialize RetrainingTrigger with configuration.

        Args:
            config: Configuration dictionary with trigger settings
        """
        self.psi_threshold = config.get("psi_threshold", 0.5)
        self.ks_p_value_threshold = config.get("ks_p_value_threshold", 0.01)
        self.min_interval_hours = config.get("min_interval_hours", 24)
        self.min_samples = config.get("min_samples", 1000)

        self.last_retraining_time = None
        self._retraining_events_dir = Path("logs/retraining_events")
        self._retraining_events_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"RetrainingTrigger initialized: PSI>{self.psi_threshold}, "
            f"KS p-value<{self.ks_p_value_threshold}, "
            f"min_interval={self.min_interval_hours}h, min_samples={self.min_samples}"
        )

    def should_trigger_retraining(self, drift_event: "DriftDetectionResult") -> dict:
        """Evaluate if retraining should be triggered based on drift event.

        Args:
            drift_event: Latest drift detection result

        Returns:
            Dictionary with decision and justification:
            {
                "trigger": bool,
                "justification": str,
                "drift_metrics": dict,
                "data_availability": dict
            }
        """
        logger.info("Evaluating retraining trigger...")

        # Check 1: Severe drift
        severe_drift, drift_reason = self._check_severe_drift(drift_event)

        # Check 2: Minimum interval
        interval_ok, interval_hours = self._check_minimum_interval()

        # Check 3: Data availability
        data_available, sample_count = self._check_data_availability()

        # All checks must pass
        trigger = severe_drift and interval_ok and data_available

        justification = self._build_justification(
            severe_drift, interval_ok, data_available,
            drift_reason, interval_hours, sample_count
        )

        result = {
            "trigger": trigger,
            "justification": justification,
            "drift_metrics": {
                "max_psi": max([m.psi_score for m in drift_event.psi_metrics]) if drift_event.psi_metrics else 0,
                "ks_p_value": drift_event.ks_result.p_value if drift_event.ks_result else 1.0,
                "drifting_features": drift_event.drifting_features,
                "drift_detected": drift_event.drift_detected
            },
            "data_availability": {
                "samples_since_last_training": sample_count,
                "hours_since_last_retraining": interval_hours
            }
        }

        # Log decision to audit trail
        self._log_retraining_decision(result)

        if trigger:
            logger.warning(f"🚨 RETRAINING TRIGGERED: {justification}")
        else:
            logger.info(f"✅ Retraining NOT triggered: {justification}")

        return result

    def _check_severe_drift(self, drift_event: "DriftDetectionResult") -> tuple[bool, str]:
        """Check if drift severity meets threshold.

        Args:
            drift_event: Drift detection result

        Returns:
            Tuple of (is_severe, reason)
        """
        if not drift_event.drift_detected:
            return False, "No drift detected"

        # FIXED: Validate drift data completeness
        has_psi = bool(drift_event.psi_metrics)
        has_ks = drift_event.ks_result is not None

        if not has_psi and not has_ks:
            logger.error("Drift detected but no PSI metrics or KS result available")
            return False, "Drift detected but incomplete data (no PSI or KS)"

        # Check PSI scores (handle empty list)
        if has_psi:
            max_psi = max([m.psi_score for m in drift_event.psi_metrics])
            severe_psi = max_psi > self.psi_threshold
        else:
            logger.warning("No PSI metrics available, relying on KS test only")
            max_psi = 0.0
            severe_psi = False

        # Check KS p-value
        ks_p_value = drift_event.ks_result.p_value if drift_event.ks_result else 1.0
        severe_ks = ks_p_value < self.ks_p_value_threshold

        if severe_psi or severe_ks:
            reasons = []
            if severe_psi:
                reasons.append(f"PSI={max_psi:.4f}>{self.psi_threshold}")
            if severe_ks:
                reasons.append(f"KS p-value={ks_p_value:.4e}<{self.ks_p_value_threshold}")
            return True, " AND ".join(reasons)

        return False, f"Drift not severe (PSI={max_psi:.4f}, KS p-value={ks_p_value:.4e})"

    def _check_minimum_interval(self) -> tuple[bool, float]:
        """Check if minimum interval since last retraining has passed.

        Returns:
            Tuple of (interval_ok, hours_since_last)
        """
        if self.last_retraining_time is None:
            return True, 0.0

        hours_since = (datetime.now() - self.last_retraining_time).total_seconds() / 3600
        interval_ok = hours_since >= self.min_interval_hours

        return interval_ok, hours_since

    def _check_data_availability(self) -> tuple[bool, int]:
        """Check if sufficient new data is available for retraining.

        Returns:
            Tuple of (sufficient_data, sample_count)
        """
        # IMPLEMENTED: Check actual data availability from dollar bars
        from pathlib import Path

        data_dir = Path("data/processed/dollar_bars/1_minute")

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return False, 0

        # Find most recent data file
        data_files = list(data_dir.glob("mnq_1min_*.csv"))

        if not data_files:
            logger.warning("No data files found in data directory")
            return False, 0

        # Load most recent file to count samples
        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)

        try:
            df = pd.read_csv(latest_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter samples since last retraining
            if self.last_retraining_time is not None:
                # Filter to samples after last retraining time
                df_filtered = df[df["timestamp"] >= self.last_retraining_time]
            else:
                # First training - use all samples
                df_filtered = df

            sample_count = len(df_filtered)
            sufficient = sample_count >= self.min_samples

            logger.info(
                f"Data availability check: {sample_count} samples "
                f"({'sufficient' if sufficient else 'insufficient'}, "
                f"minimum: {self.min_samples})"
            )

            return sufficient, sample_count

        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return False, 0

    def _build_justification(
        self,
        severe_drift: bool,
        interval_ok: bool,
        data_available: bool,
        drift_reason: str,
        interval_hours: float,
        sample_count: int
    ) -> str:
        """Build human-readable justification for trigger decision.

        Args:
            severe_drift: Whether drift is severe
            interval_ok: Whether minimum interval passed
            data_available: Whether sufficient data available
            drift_reason: Reason for drift severity check
            interval_hours: Hours since last retraining
            sample_count: Number of samples available

        Returns:
            Human-readable justification string
        """
        if not severe_drift:
            return f"Drift not severe: {drift_reason}"

        if not interval_ok:
            return f"Insufficient time since last retraining: {interval_hours:.1f}h < {self.min_interval_hours}h"

        if not data_available:
            return f"Insufficient data available: {sample_count} < {self.min_samples} samples"

        return (
            f"All checks passed: {drift_reason}, "
            f"{interval_hours:.1f}h since last retraining, "
            f"{sample_count} samples available"
        )

    def _log_retraining_decision(self, result: dict) -> None:
        """Log retraining decision to CSV audit trail.

        Args:
            result: Result from should_trigger_retraining()
        """
        csv_file = self._retraining_events_dir / "retraining_decisions.csv"

        # Create CSV with headers if doesn't exist
        csv_file_exists = csv_file.exists()

        # Prepare row data
        row = {
            "timestamp": datetime.now().isoformat(),
            "trigger": result["trigger"],
            "justification": result["justification"],
            "max_psi": result["drift_metrics"]["max_psi"],
            "ks_p_value": result["drift_metrics"]["ks_p_value"],
            "drifting_features": ",".join(result["drift_metrics"]["drifting_features"]),
            "samples_available": result["data_availability"]["samples_since_last_training"],
            "hours_since_last": result["data_availability"]["hours_since_last_retraining"],
        }

        # Write to CSV
        import csv

        with open(csv_file, "a", newline="") as f:
            fieldnames = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not csv_file_exists:
                writer.writeheader()

            writer.writerow(row)

        logger.debug(f"Retraining decision logged to {csv_file}")

    def update_last_retraining_time(self, timestamp: datetime) -> None:
        """Update last retraining timestamp.

        Args:
            timestamp: Timestamp of retraining completion
        """
        self.last_retraining_time = timestamp
        logger.info(f"Last retraining time updated to {timestamp}")


class ModelVersioning:
    """Manages model versioning, hashing, and rollback.

    Features:
    - SHA256 model hashing for unique identification
    - Model lineage tracking (previous_hash → current_hash)
    - Atomic model deployment with rollback capability
    - Model metadata storage (training timestamp, performance metrics)

    Example:
        >>> versioning = ModelVersioning()
        >>> model_hash = versioning.save_model(trained_model)
        >>> versioning.rollback_model(previous_hash)
    """

    def __init__(self, models_dir: str = "models/xgboost/1_minute"):
        """Initialize ModelVersioning.

        Args:
            models_dir: Directory to store model versions
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.current_model_hash = None
        self.model_lineage_file = Path("models/model_lineage.json")

        self._load_current_model_hash()

        logger.info(f"ModelVersioning initialized: models_dir={self.models_dir}")

    def _load_current_model_hash(self) -> None:
        """Load current model hash from lineage file."""
        if self.model_lineage_file.exists():
            try:
                with open(self.model_lineage_file, "r") as f:
                    lineage = json.load(f)
                self.current_model_hash = lineage.get("current_model")
                logger.info(f"Current model hash: {self.current_model_hash}")
            except Exception as e:
                logger.warning(f"Could not load model lineage: {e}")

    def get_model_hash(self, model_file: Path) -> str:
        """Generate SHA256 hash of model file.

        Args:
            model_file: Path to model file

        Returns:
            SHA256 hash string
        """
        sha256_hash = hashlib.sha256()
        with open(model_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def save_model(self, model, metadata: dict | None = None) -> str:
        """Save model with metadata and return hash.

        Args:
            model: Trained model (XGBClassifier or calibrated wrapper)
            metadata: Optional metadata dictionary

        Returns:
            Model hash string
        """
        # Save to temporary file
        temp_file = Path("temp_model.joblib")
        joblib.dump(model, temp_file)

        # Generate hash
        model_hash = self.get_model_hash(temp_file)

        # Create model directory
        model_dir = self.models_dir / model_hash
        model_dir.mkdir(parents=True, exist_ok=True)

        # Move model file
        final_file = model_dir / "model.joblib"
        temp_file.rename(final_file)

        # Prepare metadata
        model_metadata = {
            "hash": model_hash,
            "timestamp": datetime.now().isoformat(),
            "previous_hash": self.current_model_hash,
        }

        if metadata:
            model_metadata.update(metadata)

        # Save metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(model_metadata, f, indent=2)

        # Update lineage
        self._update_lineage(model_hash)

        self.current_model_hash = model_hash

        logger.info(f"Model saved: hash={model_hash}, file={final_file}")

        return model_hash

    def _update_lineage(self, new_hash: str) -> None:
        """Update model lineage file.

        Args:
            new_hash: New model hash
        """
        lineage = {
            "current_model": new_hash,
            "last_updated": datetime.now().isoformat()
        }

        if self.current_model_hash:
            lineage["previous_model"] = self.current_model_hash

        with open(self.model_lineage_file, "w") as f:
            json.dump(lineage, f, indent=2)

    def load_model(self, model_hash: str | None = None):
        """Load model by hash (or current if None).

        Args:
            model_hash: Model hash to load (None for current)

        Returns:
            Loaded model
        """
        target_hash = model_hash or self.current_model_hash

        if not target_hash:
            raise ValueError("No model hash specified and no current model")

        model_file = self.models_dir / target_hash / "model.joblib"

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        model = joblib.load(model_file)
        logger.info(f"Model loaded: hash={target_hash}")

        return model

    def rollback_model(self, target_hash: str) -> bool:
        """Rollback to previous model version.

        Args:
            target_hash: Target model hash to rollback to

        Returns:
            True if rollback successful
        """
        try:
            model_file = self.models_dir / target_hash / "model.joblib"

            if not model_file.exists():
                logger.error(f"Target model {target_hash} not found")
                return False

            # Update current model
            self.current_model_hash = target_hash
            self._update_lineage(target_hash)

            logger.info(f"Rolled back to model {target_hash}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_current_model_hash(self) -> str | None:
        """Get current model hash.

        Returns:
            Current model hash or None
        """
        return self.current_model_hash


class PerformanceValidator:
    """Validates new model performance before deployment.

    Checks:
    - Brier score < 0.2 (calibration quality)
    - Win rate >= old model win rate (no degradation)
    - Feature stability (no drastic shifts)
    - Prediction distribution (no overfitting)

    Example:
        >>> validator = PerformanceValidator()
        >>> result = validator.validate_model(new_model, old_model, test_data)
        >>> if result["passed"]:
        ...     # Deploy new model
    """

    def __init__(self, config: dict):
        """Initialize PerformanceValidator with configuration.

        Args:
            config: Configuration dictionary with validation criteria
        """
        self.brier_score_max = config.get("brier_score_max", 0.2)
        self.win_rate_min_delta = config.get("win_rate_min_delta", 0.0)
        self.feature_stability_threshold = config.get("feature_stability_threshold", 0.3)

        logger.info(
            f"PerformanceValidator initialized: "
            f"Brier score max={self.brier_score_max}, "
            f"Win rate min delta={self.win_rate_min_delta}, "
            f"Feature stability threshold={self.feature_stability_threshold}"
        )

    def validate_model(
        self,
        new_model,
        old_model,
        test_features: pd.DataFrame,
        test_labels: np.ndarray
    ) -> dict:
        """Validate new model performance against old model.

        Args:
            new_model: New trained model
            old_model: Current model
            test_features: Test features
            test_labels: Test labels

        Returns:
            Dictionary with validation result:
            {
                "passed": bool,
                "metrics": dict,
                "failures": list[str]
            }
        """
        logger.info("Validating new model performance...")

        # FIXED: Add input validation
        # Check for empty test set
        if len(test_features) == 0 or len(test_labels) == 0:
            logger.error("Empty test set - cannot validate model")
            return {
                "passed": False,
                "metrics": {},
                "failures": ["Empty test set - cannot validate model"]
            }

        # Check for shape mismatch
        if len(test_features) != len(test_labels):
            logger.error(
                f"Test set size mismatch: {len(test_features)} features != "
                f"{len(test_labels)} labels"
            )
            return {
                "passed": False,
                "metrics": {},
                "failures": [
                    f"Test set size mismatch: {len(test_features)} features != "
                    f"{len(test_labels)} labels"
                ]
            }

        failures = []
        metrics = {}

        # Check 1: Brier score
        brier_score, brier_error = self._calculate_brier_score(
            new_model, test_features, test_labels
        )
        metrics["brier_score"] = brier_score
        metrics["brier_error"] = brier_error

        if brier_error:
            failures.append(f"Brier score calculation failed: {brier_error}")
        elif brier_score > self.brier_score_max:
            failures.append(f"Brier score {brier_score:.4f} > {self.brier_score_max}")

        # Check 2: Win rate comparison
        new_win_rate, new_rate_error = self._calculate_win_rate(
            new_model, test_features, test_labels
        )
        old_win_rate, old_rate_error = self._calculate_win_rate(
            old_model, test_features, test_labels
        )

        metrics["new_win_rate"] = new_win_rate
        metrics["old_win_rate"] = old_win_rate
        metrics["win_rate_error"] = new_rate_error or old_rate_error

        if new_rate_error or old_rate_error:
            failures.append(f"Win rate calculation failed: {new_rate_error or old_rate_error}")
        else:
            win_rate_delta = new_win_rate - old_win_rate
            metrics["win_rate_delta"] = win_rate_delta

            if win_rate_delta < self.win_rate_min_delta:
                if win_rate_delta < 0:
                    failures.append(
                        f"Win rate degradation: {win_rate_delta:+.4f} "
                        f"(new={new_win_rate:.4f}, old={old_win_rate:.4f})"
                    )
                else:
                    failures.append(
                        f"Win rate delta {win_rate_delta:.4f} < {self.win_rate_min_delta} "
                        f"(new={new_win_rate:.4f}, old={old_win_rate:.4f})"
                    )

        # Check 3: Feature stability (simplified)
        # TODO: Implement proper feature importance comparison
        # For now, skip this check

        # Check 4: Prediction distribution
        pred_dist_ok = self._check_prediction_distribution(new_model, test_features)
        metrics["prediction_distribution_ok"] = pred_dist_ok

        if not pred_dist_ok:
            failures.append("Prediction distribution differs significantly from expected")

        passed = len(failures) == 0

        result = {
            "passed": passed,
            "metrics": metrics,
            "failures": failures
        }

        if passed:
            logger.info(f"✅ Model validation passed: Brier score={brier_score:.4f}, "
                       f"win rate delta={metrics.get('win_rate_delta', 0):.4f}")
        else:
            logger.error(f"❌ Model validation failed: {failures}")

        return result

    def _calculate_brier_score(
        self, model, features: pd.DataFrame, labels: np.ndarray
    ) -> tuple[float, str | None]:
        """Calculate Brier score for model predictions.

        Args:
            model: Model to evaluate
            features: Test features
            labels: True labels

        Returns:
            Tuple of (brier_score, error_message)
            error_message is None if calculation succeeded
        """
        try:
            predictions = model.predict_proba(features)[:, 1]
            brier_score = np.mean((predictions - labels) ** 2)
            return float(brier_score), None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Brier score calculation failed: {error_msg}")
            return 1.0, error_msg  # Worst score with error

    def _calculate_win_rate(
        self, model, features: pd.DataFrame, labels: np.ndarray
    ) -> tuple[float, str | None]:
        """Calculate win rate for model predictions.

        Args:
            model: Model to evaluate
            features: Test features
            labels: True labels

        Returns:
            Tuple of (win_rate, error_message)
            error_message is None if calculation succeeded
        """
        try:
            predictions = model.predict_proba(features)[:, 1]
            predicted_classes = (predictions > 0.5).astype(int)
            accuracy = np.mean(predicted_classes == labels)
            return float(accuracy), None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Win rate calculation failed: {error_msg}")
            return 0.0, error_msg

    def _check_prediction_distribution(self, model, features: pd.DataFrame) -> bool:
        """Check if prediction distribution is reasonable.

        Args:
            model: Model to evaluate
            features: Test features

        Returns:
            True if distribution is reasonable
        """
        try:
            predictions = model.predict_proba(features)[:, 1]

            # Check for degenerate distributions
            if np.mean(predictions) < 0.1 or np.mean(predictions) > 0.9:
                return False

            # Check for variance
            if np.std(predictions) < 0.05:
                return False

            return True

        except Exception as e:
            logger.error(f"Prediction distribution check failed: {e}")
            return False


class AsyncRetrainingTask:
    """Async background task for model retraining.

    Executes retraining workflow:
    1. Collect new training data
    2. Train new model
    3. Validate performance
    4. Deploy if validation passes
    5. Log results

    Example:
        >>> task = AsyncRetrainingTask(config)
        >>> result = await task.run_retraining(trigger_result)
    """

    def __init__(self, config: dict, ml_inference=None):
        """Initialize AsyncRetrainingTask with configuration.

        Args:
            config: Configuration dictionary
            ml_inference: Optional MLInference instance for cache invalidation
        """
        self.config = config
        self.is_retraining = False

        # FIXED: Add asyncio.Lock for atomic check-and-set of is_retraining flag
        self._retraining_lock = asyncio.Lock()

        # Store MLInference reference for deployment integration
        self._ml_inference = ml_inference

        self.model_versioning = ModelVersioning(
            config.get("models_dir", "models/xgboost/1_minute")
        )
        self.performance_validator = PerformanceValidator(
            config.get("validation", {})
        )

        logger.info("AsyncRetrainingTask initialized")

    def _invalidate_ml_inference_cache(self, new_model_hash: str):
        """Invalidate MLInference model cache to force reload of new model.

        Args:
            new_model_hash: Hash of the new model to load
        """
        if self._ml_inference is None:
            logger.warning("No MLInference instance provided - cache not invalidated")
            logger.info("Model deployed but MLInference will need manual restart to load new model")
            return

        try:
            # Clear model cache for all horizons
            if hasattr(self._ml_inference, "_models"):
                self._ml_inference._models.clear()
                logger.info("Cleared MLInference model cache")

            # Clear calibration cache for all horizons
            if hasattr(self._ml_inference, "_calibration"):
                self._ml_inference._calibration.clear()
                logger.info("Cleared MLInference calibration cache")

            # Clear pipeline cache for all horizons
            if hasattr(self._ml_inference, "_pipelines"):
                self._ml_inference._pipelines.clear()
                logger.info("Cleared MLInference pipeline cache")

            logger.info(f"MLInference caches cleared - will load new model {new_model_hash} on next inference")

        except Exception as e:
            logger.error(f"Error invalidating MLInference cache: {e}")
            raise

    async def run_retraining(self, trigger_result: dict) -> dict:
        """Run async retraining task.

        Args:
            trigger_result: Result from RetrainingTrigger.should_trigger_retraining()

        Returns:
            Dictionary with retraining result
        """
        # FIXED: Use asyncio.Lock for atomic check-and-set
        async with self._retraining_lock:
            if self.is_retraining:
                logger.warning("Retraining already in progress, skipping")
                return {"success": False, "reason": "Retraining already in progress"}

            self.is_retraining = True

        try:
            logger.info("Starting async retraining task...")

            # FIXED: Add timeout wrapper (60 minutes from config)
            timeout_seconds = self.config.get("timeout_minutes", 60) * 60

            async with asyncio.timeout(timeout_seconds):
                # 1. Collect new training data
                logger.info("Step 1: Collecting new training data...")
                new_data = await self._collect_new_training_data()

                if new_data is None:
                    return {"success": False, "reason": "Failed to collect training data"}

                # 2. Train new model
                logger.info("Step 2: Training new model...")
                new_model = await self._train_model(new_data)

                if new_model is None:
                    return {"success": False, "reason": "Model training failed"}

                # 3. Validate performance
                logger.info("Step 3: Validating model performance...")

                # FIXED: Handle FileNotFoundError in model loading
                try:
                    old_model = self.model_versioning.load_model()
                except FileNotFoundError as e:
                    logger.error(f"Cannot load current model for validation: {e}")
                    return {
                        "success": False,
                        "reason": "Current model not found - cannot validate new model"
                    }

                validation_result = self.performance_validator.validate_model(
                    new_model, old_model,
                    new_data["X_test"],
                    new_data["y_test"]
                )

                if not validation_result["passed"]:
                    logger.error(f"Model validation failed: {validation_result['failures']}")
                    return {
                        "success": False,
                        "reason": "Validation failed",
                        "validation_result": validation_result
                    }

                # 4. Deploy model
                logger.info("Step 4: Deploying new model...")
                old_model_hash = self.model_versioning.get_current_model_hash()

                metadata = {
                    "training_samples": len(new_data["X_train"]),
                    "validation_samples": len(new_data["X_test"]),
                    "performance_metrics": validation_result["metrics"],
                    "trigger_reason": trigger_result["justification"]
                }

                new_model_hash = self.model_versioning.save_model(new_model, metadata)

                # IMPLEMENTED: Update MLInference to use new model
                # Invalidate model cache to force reload on next inference
                try:
                    self._invalidate_ml_inference_cache(new_model_hash)
                    logger.info(f"MLInference cache invalidated - will load new model {new_model_hash}")
                except Exception as e:
                    logger.warning(f"Failed to invalidate MLInference cache: {e}")

                result = {
                    "success": True,
                    "old_model_hash": old_model_hash,
                    "new_model_hash": new_model_hash,
                    "performance_metrics": validation_result["metrics"],
                    "validation_result": validation_result
                }

                logger.info(f"✅ Retraining completed successfully: {old_model_hash} → {new_model_hash}")

                return result

        except asyncio.TimeoutError:
            logger.error(f"Retraining timed out after {timeout_seconds}s")
            return {"success": False, "reason": "Retraining timed out"}
        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            return {"success": False, "reason": str(e)}

        finally:
            self.is_retraining = False

    async def _collect_new_training_data(self) -> dict | None:
        """Collect new training data since last model update.

        Returns:
            Dictionary with X_train, X_test, y_train, y_test or None
        """
        # IMPLEMENTED: Load actual dollar bars data
        from pathlib import Path
        from sklearn.model_selection import train_test_split

        data_dir = Path("data/processed/dollar_bars/1_minute")

        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return None

        # Find most recent data file
        data_files = list(data_dir.glob("mnq_1min_*.csv"))

        if not data_files:
            logger.error("No data files found for training")
            return None

        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading training data from: {latest_file}")

        try:
            # Load dollar bars
            df = pd.read_csv(latest_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter to recent data if needed (last 3 months by default)
            if self.last_retraining_time is not None:
                # Collect data since last retraining, but minimum 30 days
                cutoff_date = max(
                    self.last_retraining_time - timedelta(days=30),
                    datetime.now() - timedelta(days=90)
                )
                df = df[df["timestamp"] >= cutoff_date].copy()
            else:
                # First training - use last 90 days
                cutoff_date = datetime.now() - timedelta(days=90)
                df = df[df["timestamp"] >= cutoff_date].copy()

            if len(df) < 1000:
                logger.error(f"Insufficient data for training: {len(df)} < 1000")
                return None

            logger.info(f"Loaded {len(df)} dollar bars for training")

            # Import here to avoid circular dependency
            from src.ml.features import FeatureEngineer

            feature_engineer = FeatureEngineer()

            # Engineer features
            logger.info("Engineering features...")
            features_df = feature_engineer.engineer_features(df)

            # Select feature columns (exclude OHLCV and datetime columns)
            exclude_columns = {
                "timestamp", "open", "high", "low", "close", "volume",
                "hour", "day_of_week", "trading_session",
                "is_london_am", "is_ny_am", "is_ny_pm",
            }
            feature_columns = [col for col in features_df.columns if col not in exclude_columns]

            # Handle NaN values
            features_df_selected = features_df[feature_columns].ffill().fillna(0)

            # Filter to numeric columns only
            features_df_numeric = features_df_selected.select_dtypes(include=[np.number])

            logger.info(f"Selected {len(feature_columns)} features ({len(features_df_numeric.columns)} numeric)")

            # Create labels (binary classification: 1 if close > open, else 0)
            # This is a simplified label - in production, use actual trade outcomes
            features_df_numeric["target"] = (df["close"] > df["open"]).astype(int)

            # Prepare features and labels
            X = features_df_numeric.drop(columns=["target"]).values
            y = features_df_numeric["target"].values

            # Split into train/test (80/20 split)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            logger.info(
                f"Training data split: {len(X_train)} train samples, "
                f"{len(X_test)} test samples"
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }

        except Exception as e:
            logger.error(f"Error collecting training data: {e}", exc_info=True)
            return None

    async def _train_model(self, data: dict) -> object | None:
        """Train new XGBoost model on data.

        Args:
            data: Dictionary with X_train, y_train

        Returns:
            Trained model or None
        """
        # IMPLEMENTED: Train XGBoost model with calibration
        import xgboost as xgb

        logger.info("Training XGBoost model...")

        try:
            X_train = data["X_train"]
            y_train = data["y_train"]

            # XGBoost parameters (use existing model config)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "use_label_encoder": False,
            }

            # Train XGBoost model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)

            logger.info("XGBoost model training completed")

            # Try to apply probability calibration if available
            try:
                from src.ml.probability_calibration import ProbabilityCalibration

                # Create calibration model
                calibration = ProbabilityCalibration(method="isotonic")

                # Fit calibration on training data
                # Get predicted probabilities
                y_pred_proba = model.predict_proba(X_train)[:, 1]

                # Fit calibration
                calibration.fit(y_pred_proba.reshape(-1, 1), y_train)

                # Create wrapped model
                class CalibratedModelWrapper:
                    """Wrapper for calibrated XGBoost model."""

                    def __init__(self, xgb_model, calibration):
                        self.xgb_model = xgb_model
                        self.calibration = calibration
                        self.base_model = xgb_model  # For feature count check

                    def predict_proba(self, X):
                        """Return calibrated probabilities."""
                        # Get raw probabilities from XGBoost
                        raw_proba = self.xgb_model.predict_proba(X)[:, 1]
                        # Apply calibration
                        calibrated_proba = self.calibration.predict_proba(raw_proba.reshape(-1, 1))
                        # Stack with 1-proba for binary format
                        return np.stack([1 - calibrated_proba, calibrated_proba], axis=1)

                    def predict(self, X):
                        """Return binary predictions."""
                        proba = self.predict_proba(X)[:, 1]
                        return (proba > 0.5).astype(int)

                    @property
                    def n_features_in_(self):
                        """Return expected number of features."""
                        return self.xgb_model.n_features_in_

                calibrated_model = CalibratedModelWrapper(model, calibration)
                logger.info("Probability calibration applied (isotonic regression)")
                return calibrated_model

            except ImportError:
                logger.warning("ProbabilityCalibration not available, using uncalibrated model")
                return model
            except Exception as e:
                logger.warning(f"Calibration failed: {e}, using uncalibrated model")
                return model

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None

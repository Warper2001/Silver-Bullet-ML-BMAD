"""Walk-Forward Optimization for Adaptive ML Models.

This module implements automated weekly model retraining on a 6-month
rolling window to maintain predictive edge in changing market conditions.
"""

import asyncio
import csv
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.metrics import roc_auc_score

from src.ml.features import FeatureEngineer
from src.ml.pipeline_serializer import PipelineSerializer
from src.ml.training_data import TrainingDataPipeline
from src.ml.xgboost_trainer import XGBoostTrainer

logger = logging.getLogger(__name__)


class DataInsufficientError(Exception):
    """Raised when training data is insufficient (< 95% completeness)."""

    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""

    pass


class ValidationError(Exception):
    """Raised when validation set is too small."""

    pass


class WalkForwardOptimizer:
    """Automated walk-forward optimization for ML model retraining.

    Handles:
    - Weekly automated retraining (Sunday 8 PM EST)
    - 6-month rolling window for training data
    - 2-week validation window for model comparison
    - ROC-AUC comparison for deployment decisions
    - Model archiving with timestamps
    - Model registry tracking
    - Retry logic with exponential backoff

    Performance:
    - Total retraining time: < 5 minutes
    - Data loading: < 60 seconds
    - Feature engineering: < 90 seconds
    - Model training: < 120 seconds
    - Validation: < 30 seconds
    - Deployment: < 10 seconds
    """

    def __init__(
        self,
        model_dir: str | Path = "models/xgboost",
        retraining_interval: str = "weekly",
    ):
        """Initialize WalkForwardOptimizer with scheduling configuration.

        Args:
            model_dir: Directory containing trained models
            retraining_interval: Retraining frequency (weekly, bi-weekly, monthly)
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Create horizon-specific directories
        for horizon in [5, 15, 30, 60]:
            horizon_dir = self._model_dir / f"{horizon}_minute"
            horizon_dir.mkdir(parents=True, exist_ok=True)

        # Retraining configuration
        self._retraining_interval = retraining_interval
        self._train_months = 6
        self._validation_weeks = 2
        self._min_data_completeness = 0.95

        # Scheduler for automated retraining
        self._scheduler = AsyncIOScheduler()

        # Configure schedule based on interval
        if retraining_interval == "weekly":
            self._scheduler.add_job(
                self._run_retraining_async,
                "cron",
                day_of_week="sun",
                hour=20,  # 8 PM EST
                minute=0,
                timezone="America/New_York",
                id="weekly_retraining",
            )
        elif retraining_interval == "bi-weekly":
            self._scheduler.add_job(
                self._run_retraining_async,
                "cron",
                day_of_week="sun",
                week="2,4",  # Every 2 weeks
                hour=20,
                minute=0,
                timezone="America/New_York",
                id="bi_weekly_retraining",
            )
        elif retraining_interval == "monthly":
            self._scheduler.add_job(
                self._run_retraining_async,
                "cron",
                day=1,  # 1st of each month
                hour=20,
                minute=0,
                timezone="America/New_York",
                id="monthly_retraining",
            )

        # ML components
        self._feature_engineer = FeatureEngineer()
        self._training_pipeline = TrainingDataPipeline()
        self._xgboost_trainer = XGBoostTrainer()
        self._pipeline_serializer = PipelineSerializer(model_dir=self._model_dir)

        logger.info(
            f"WalkForwardOptimizer initialized with "
            f"model_dir: {self._model_dir}, "
            f"interval: {self._retraining_interval}"
        )

    def load_training_window(
        self, end_time: datetime | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load 6-month training window and 2-week validation window.

        Args:
            end_time: End of validation window (default: now)

        Returns:
            (train_data, val_data, train_signals, val_signals)

        Raises:
            DataInsufficientError: If data completeness < 95%
        """
        if end_time is None:
            end_time = datetime.now()

        # Calculate time windows
        val_start = end_time - timedelta(weeks=self._validation_weeks)
        train_start = val_start - timedelta(weeks=self._train_months * 4)

        logger.info(
            f"Loading training window: {train_start.date()} to {val_start.date()} "
            f"(6 months), validation: {val_start.date()} to {end_time.date()} (2 weeks)"
        )

        # Load Dollar Bars
        bars_data = self._load_dollar_bars(train_start, end_time)

        # Load Silver Bullet signals
        signals_data = self._load_signals(train_start, end_time)

        # Validate data completeness
        expected_bars = (
            self._train_months * 30 * 24
        )  # Approximate (months * days * hours)
        actual_bars = len(bars_data)
        completeness = actual_bars / expected_bars if expected_bars > 0 else 0

        logger.info(
            f"Data completeness: {completeness:.2%} "
            f"({actual_bars} / {expected_bars} expected bars)"
        )

        if completeness < self._min_data_completeness:
            raise DataInsufficientError(
                f"Data completeness {completeness:.2%} below threshold "
                f"{self._min_data_completeness:.2%}"
            )

        # Split into training and validation
        train_data = bars_data.loc[train_start:val_start]
        val_data = bars_data.loc[val_start:]

        # Split signals
        train_signals = signals_data[
            (signals_data["timestamp"] >= train_start)
            & (signals_data["timestamp"] < val_start)
        ]
        val_signals = signals_data[(signals_data["timestamp"] >= val_start)]

        logger.info(
            f"Loaded {len(train_data)} training bars, {len(val_data)} validation bars, "
            f"{len(train_signals)} training signals, {len(val_signals)} validation signals"
        )

        return train_data, val_data, train_signals, val_signals

    def _load_dollar_bars(
        self, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """Load Dollar Bars from HDF5 storage.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with Dollar Bars data
        """
        # TODO: Integrate with Epic 1 data pipeline
        # For now, create dummy data
        dates = pd.date_range(start=start_time, end=end_time, freq="H")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(11700, 11900, len(dates)),
                "high": np.random.uniform(11800, 12000, len(dates)),
                "low": np.random.uniform(11600, 11800, len(dates)),
                "close": np.random.uniform(11700, 11900, len(dates)),
                "volume": np.random.uniform(1000, 5000, len(dates)),
            }
        )
        data.set_index("timestamp", inplace=True)
        return data

    def _load_signals(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Load Silver Bullet signals from storage.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with signals data
        """
        # TODO: Integrate with Epic 2 signal detection
        # For now, create dummy signals
        dates = pd.date_range(start=start_time, end=end_time, freq="D")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "direction": np.random.choice(["bullish", "bearish"], len(dates)),
                "outcome": np.random.choice([0, 1], len(dates)),
            }
        )
        return data

    def _prepare_features(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        train_signals: pd.DataFrame,
        val_signals: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare features and labels for training.

        Args:
            train_data: Training Dollar Bars
            val_data: Validation Dollar Bars
            train_signals: Training signals
            val_signals: Validation signals

        Returns:
            (train_features, val_features, train_labels, val_labels)
        """
        logger.info("Engineering features for training and validation sets")

        # Engineer features using FeatureEngineer
        train_features_df = self._feature_engineer.engineer_features(train_data)
        val_features_df = self._feature_engineer.engineer_features(val_data)

        # Use TrainingDataPipeline for feature selection and labeling
        # Note: This is simplified - actual implementation would merge signals with bars
        train_features_array = train_features_df.values
        val_features_array = val_features_df.values

        # Generate labels from signal outcomes
        train_labels = train_signals["outcome"].values[: len(train_features_array)]
        val_labels = val_signals["outcome"].values[: len(val_features_array)]

        # Ensure equal lengths
        min_train_len = min(len(train_features_array), len(train_labels))
        min_val_len = min(len(val_features_array), len(val_labels))

        train_features = train_features_array[:min_train_len]
        train_labels = train_labels[:min_train_len]
        val_features = val_features_array[:min_val_len]
        val_labels = val_labels[:min_val_len]

        logger.info(
            f"Prepared {len(train_features)} training samples, "
            f"{len(val_features)} validation samples"
        )

        return train_features, val_features, train_labels, val_labels

    def _train_model(
        self, train_features: pd.DataFrame | np.ndarray, train_labels: np.ndarray
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Train new XGBoost model.

        Args:
            train_features: Training feature matrix
            train_labels: Training labels

        Returns:
            (model, pipeline, metadata)

        Raises:
            ModelTrainingError: If training fails
        """
        logger.info("Training new XGBoost model on 6-month window")

        try:
            # Convert to pandas DataFrame if needed
            if isinstance(train_features, np.ndarray):
                train_features = pd.DataFrame(train_features)

            # Train model using XGBoostTrainer
            model, pipeline, metadata = self._xgboost_trainer.train_xgboost(
                train_features, train_labels
            )

            logger.info(
                f"Model training complete. ROC-AUC: {metadata['metrics']['roc_auc']:.4f}"
            )

            return model, pipeline, metadata

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelTrainingError(f"Failed to train model: {e}")

    def compare_models(
        self,
        new_model: Any,
        current_model: Any,
        val_features: pd.DataFrame,
        val_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare new model against current model on validation set.

        Args:
            new_model: Newly trained XGBoost model
            current_model: Currently deployed model
            val_features: Validation set features
            val_labels: Validation set labels

        Returns:
            Dictionary with comparison results and deployment decision
        """
        logger.info("Comparing new model against current model")

        # Get predictions
        new_probs = new_model.predict_proba(val_features)[:, 1]
        current_probs = current_model.predict_proba(val_features)[:, 1]

        # Calculate ROC-AUC
        new_roc_auc = roc_auc_score(val_labels, new_probs)
        current_roc_auc = roc_auc_score(val_labels, current_probs)

        logger.info(
            f"New model ROC-AUC: {new_roc_auc:.4f}, "
            f"Current model ROC-AUC: {current_roc_auc:.4f}"
        )

        # Compare and make decision
        if new_roc_auc >= current_roc_auc:
            return {
                "deploy": True,
                "reason": "roc_auc_improved_or_equal",
                "new_roc_auc": new_roc_auc,
                "current_roc_auc": current_roc_auc,
                "difference": new_roc_auc - current_roc_auc,
            }
        else:
            return {
                "deploy": False,
                "reason": "roc_auc_degraded",
                "new_roc_auc": new_roc_auc,
                "current_roc_auc": current_roc_auc,
                "difference": current_roc_auc - new_roc_auc,
            }

    def deploy_model(
        self,
        new_model_path: str | Path,
        new_pipeline_path: str | Path,
        metadata: Dict[str, Any] | None = None,
        deployed: bool = True,
    ):
        """Deploy new model atomically with archiving.

        Args:
            new_model_path: Path to new model file
            new_pipeline_path: Path to new pipeline file
            metadata: Model metadata dictionary
            deployed: Whether model was deployed (vs. rejected)
        """
        new_model_path = Path(new_model_path)
        new_pipeline_path = Path(new_pipeline_path)

        # Deploy to all horizon directories (simplified - using 5-minute as primary)
        for horizon in [5, 15, 30, 60]:
            horizon_dir = self._model_dir / f"{horizon}_minute"

            # Archive current model if it exists
            current_model = horizon_dir / "xgboost_model.json"
            current_pipeline = horizon_dir / "feature_pipeline.pkl"

            if current_model.exists():
                # Create archive directory
                archive_dir = (
                    self._model_dir
                    / "archive"
                    / datetime.now().strftime("%Y-%m-%d")
                    / f"{horizon}_minute"
                )
                archive_dir.mkdir(parents=True, exist_ok=True)

                # Move current model to archive
                shutil.move(str(current_model), str(archive_dir / "xgboost_model.json"))
                shutil.move(
                    str(current_pipeline), str(archive_dir / "feature_pipeline.pkl")
                )

                # Archive metadata too
                metadata_file = horizon_dir / "metadata.json"
                if metadata_file.exists():
                    shutil.move(str(metadata_file), str(archive_dir / "metadata.json"))

                logger.info(f"Archived previous model to {archive_dir}")

            # Copy new model to production location
            shutil.copy(str(new_model_path), str(current_model))
            shutil.copy(str(new_pipeline_path), str(current_pipeline))

            # Save metadata
            if metadata:
                metadata_file = horizon_dir / "metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Model deployed to {current_model}")

        # Update model registry
        self._update_model_registry(new_model_path, metadata, deployed)

    def _update_model_registry(
        self, model_path: Path, metadata: Dict[str, Any] | None, deployed: bool
    ):
        """Update model registry CSV with new model entry.

        Args:
            model_path: Path to model file
            metadata: Model metadata dictionary
            deployed: Whether model was deployed
        """
        registry_file = self._model_dir / "model_registry.csv"
        file_exists = registry_file.exists()

        with open(registry_file, "a", newline="") as f:
            fieldnames = [
                "date",
                "model_path",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "deployed",
                "deployment_time",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            if metadata and "metrics" in metadata:
                metrics = metadata["metrics"]
            else:
                metrics = {"roc_auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

            writer.writerow(
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "model_path": str(model_path),
                    "roc_auc": metrics.get("roc_auc", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "deployed": deployed,
                    "deployment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        logger.info(f"Model registry updated: {model_path}")

    def run_retraining(self) -> Dict[str, Any]:
        """Run the complete walk-forward optimization pipeline.

        Returns:
            Dictionary with retraining results and performance metrics
        """
        import time

        start_time = time.time()
        timings = {}

        logger.info("Starting walk-forward optimization retraining")

        try:
            # Load data
            load_start = time.time()
            (
                train_data,
                val_data,
                train_signals,
                val_signals,
            ) = self.load_training_window()
            timings["data_loading"] = time.time() - load_start

            # Engineer features
            feature_start = time.time()
            (
                train_features,
                val_features,
                train_labels,
                val_labels,
            ) = self._prepare_features(train_data, val_data, train_signals, val_signals)
            timings["feature_engineering"] = time.time() - feature_start

            # Train model
            train_start = time.time()
            new_model, new_pipeline, new_metadata = self._train_model(
                train_features, train_labels
            )
            timings["model_training"] = time.time() - train_start

            # Validate and compare
            compare_start = time.time()
            comparison = self._compare_and_deploy(
                new_model, new_pipeline, new_metadata, val_features, val_labels
            )
            timings["validation_deployment"] = time.time() - compare_start

            # Total time
            total_time = time.time() - start_time
            timings["total"] = total_time

            logger.info(f"Retraining completed in {total_time:.2f} seconds")
            logger.info(f"Timings: {timings}")

            # Check performance requirement
            if total_time > 300:  # 5 minutes
                logger.warning(
                    f"Retraining exceeded 5-minute threshold: {total_time:.2f}s"
                )

            return {
                "success": True,
                "comparison": comparison,
                "timings": timings,
                "total_time_seconds": total_time,
            }

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timings": timings,
            }

    def _compare_and_deploy(
        self,
        new_model: Any,
        new_pipeline: Any,
        new_metadata: Dict[str, Any],
        val_features: pd.DataFrame,
        val_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare models and deploy if improved.

        Args:
            new_model: Newly trained model
            new_pipeline: New feature pipeline
            new_metadata: New model metadata
            val_features: Validation features
            val_labels: Validation labels

        Returns:
            Comparison results
        """
        # Try to load current model for comparison
        try:
            current_model = self._pipeline_serializer.load_model(
                5
            )  # Load 5-minute model
        except FileNotFoundError:
            logger.warning("No current model found, deploying new model")
            current_model = None

        if current_model is None:
            # No current model, deploy new model
            comparison = {"deploy": True, "reason": "first_model"}
        else:
            # Compare with current model
            comparison = self.compare_models(
                new_model, current_model, val_features, val_labels
            )

        # Deploy or reject based on comparison
        if comparison["deploy"]:
            # Save new model and pipeline to temp files first
            temp_dir = self._model_dir / "temp"
            temp_dir.mkdir(exist_ok=True)

            temp_model_path = temp_dir / "xgboost_model.json"
            temp_pipeline_path = temp_dir / "feature_pipeline.pkl"

            new_model.save_model(str(temp_model_path))
            # Note: Pipeline saving would use joblib
            # joblib.dump(new_pipeline, temp_pipeline_path)

            self.deploy_model(
                temp_model_path,
                temp_pipeline_path,
                metadata=new_metadata,
                deployed=True,
            )

            logger.info(f"New model deployed: {comparison['reason']}")

        else:
            logger.info(f"New model rejected: {comparison['reason']}")

        return comparison

    async def _run_retraining_async(self):
        """Async wrapper for run_retraining (for scheduler)."""
        try:
            result = self.run_retraining()
            if result["success"]:
                logger.info("Scheduled retraining completed successfully")
            else:
                logger.error(f"Scheduled retraining failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"Scheduled retraining failed with exception: {e}")

    async def run_retraining_with_retry(self, max_retries: int = 3) -> Dict[str, Any]:
        """Run retraining with retry logic.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary with retraining results
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Retrying attempt {attempt + 1}/{max_retries}")
                result = self.run_retraining()

                if result["success"]:
                    logger.info("Retraining completed successfully")
                    return result

            except Exception as e:
                logger.error(f"Retrying failed (attempt {attempt + 1}): {e}")

                if attempt == max_retries - 1:
                    logger.critical(
                        "All retraining attempts failed. Keeping current model."
                    )
                    logger.critical("Manual intervention required.")
                    # Send alert notification
                    await self._send_alert_notification(
                        "walk_forward_failure", error=str(e)
                    )
                    return {"success": False, "error": str(e)}

                # Wait before retry (exponential backoff)
                await asyncio.sleep(2**attempt * 60)  # 2min, 4min, 8min

        return {"success": False, "error": "Max retries exhausted"}

    async def _send_alert_notification(self, alert_type: str, **kwargs):
        """Send alert notification for model events.

        Args:
            alert_type: Type of alert
            **kwargs: Additional context
        """
        # TODO: Integrate with notification system (Epic 6, Story 6.6)
        # For now, just log the notification
        if alert_type == "model_deployed":
            logger.info(
                f"NOTIFICATION: New ML model deployed. "
                f"ROC-AUC: {kwargs.get('roc_auc', 'N/A')}, "
                f"Improvement: {kwargs.get('improvement', 'N/A')}"
            )
        elif alert_type == "model_rejected":
            logger.warning(
                f"NOTIFICATION: New model rejected. "
                f"New ROC-AUC: {kwargs.get('new_roc_auc', 'N/A')}, "
                f"Current ROC-AUC: {kwargs.get('current_roc_auc', 'N/A')}, "
                f"Degradation: {kwargs.get('degradation', 'N/A')}"
            )
        elif alert_type == "walk_forward_failure":
            logger.critical(
                f"NOTIFICATION: Walk-forward optimization FAILED. "
                f"Error: {kwargs.get('error', 'Unknown error')}. "
                f"Manual intervention required."
            )

    def start_scheduler(self):
        """Start the scheduler for automatic retraining."""
        try:
            self._scheduler.start()
            logger.info("Walk-forward optimization scheduler started")
            job = self._scheduler.get_job("weekly_retraining")
            if job:
                logger.info(f"Next retraining scheduled for: {job.next_run_time}")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    def stop_scheduler(self):
        """Stop the scheduler gracefully."""
        try:
            self._scheduler.shutdown()
            logger.info("Walk-forward optimization scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")

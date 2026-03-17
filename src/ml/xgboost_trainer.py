"""XGBoost Model Training for ML Meta-Labeling.

This module implements XGBoost classifier training, hyperparameter tuning,
model evaluation, and persistence for ML meta-labeling.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ============================================================================
# Model Training
# ============================================================================


def train_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> tuple[XGBClassifier, dict]:
    """Train XGBoost classifier for binary classification.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_estimators: Number of trees (default: 100)
        max_depth: Maximum tree depth (default: 6)
        learning_rate: Learning rate (default: 0.1)
        min_child_weight: Minimum child weight (default: 1)
        subsample: Subsample ratio for training data (default: 0.8)
        colsample_bytree: Subsample ratio for columns (default: 0.8)
        random_state: Random seed (default: 42)

    Returns:
        Tuple of (trained_model, validation_metrics)
    """
    # Initialize model
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predict on validation set
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    metrics = evaluate_model(y_val, y_pred, y_proba)

    logger.info(f"Model trained - Validation AUC: {metrics['roc_auc']:.4f}")
    return model, metrics


# ============================================================================
# Model Evaluation
# ============================================================================


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Calculate evaluation metrics for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (positive class)

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return metrics


# ============================================================================
# Hyperparameter Tuning
# ============================================================================


def tune_hyperparameters(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    n_iter: int = 10,
    cv_folds: int = 3,
) -> dict:
    """Tune XGBoost hyperparameters using randomized search.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_iter: Number of parameter settings sampled (default: 10)
        cv_folds: Number of cross-validation folds (default: 3)

    Returns:
        Dictionary with best hyperparameters
    """
    from sklearn.model_selection import RandomizedSearchCV

    # Define parameter grid
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # Initialize base model
    base_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    # Perform randomized search
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    logger.info(f"Hyperparameter tuning complete - Best AUC: {search.best_score_:.4f}")
    return search.best_params_


# ============================================================================
# XGBoost Trainer Class
# ============================================================================


class XGBoostTrainer:
    """Orchestrates XGBoost model training and evaluation.

    Handles:
    - Training models for multiple time horizons
    - Hyperparameter tuning
    - Model evaluation
    - Feature importance tracking
    - Model persistence

    Performance:
    - Training < 30 seconds per horizon
    - Memory efficient for large datasets
    """

    DEFAULT_HYPERPARAMETERS = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    def __init__(self, model_dir: str | Path = "models/xgboost"):
        """Initialize XGBoost trainer.

        Args:
            model_dir: Directory to save trained models
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"XGBoostTrainer initialized with model_dir: {self._model_dir}")

    def train_models(  # noqa: E501
        self,
        datasets: dict,
        time_horizons: list[int],
        perform_tuning: bool = False,
        n_iter: int = 10,
        **hyperparameters,
    ) -> dict[int, dict]:
        """Train XGBoost models for multiple time horizons.

        Args:
            datasets: Dictionary mapping horizon -> {train, val} DataFrames
            time_horizons: List of time horizons to train
            perform_tuning: Whether to perform hyperparameter tuning
            n_iter: Number of hyperparameter iterations
            **hyperparameters: Additional hyperparameters to override defaults

        Returns:
            Dictionary mapping horizon -> model data including metrics,
            feature importance, and hyperparameters
        """
        import time

        models = {}
        hyperparameters = {**self.DEFAULT_HYPERPARAMETERS, **hyperparameters}

        for horizon in time_horizons:
            logger.info(f"Training model for {horizon}-minute horizon...")
            start_time = time.perf_counter()

            if horizon not in datasets:
                logger.warning(f"No data for {horizon}-minute horizon, skipping")
                continue

            # Get data
            data = datasets[horizon]
            train_df = data["train"]
            val_df = data["val"]

            # Separate features and labels
            # Exclude non-feature columns
            exclude_cols = {
                "label",
                "time_horizon",
                "timestamp",
                "signal_direction",
                "trading_session",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]

            X_train = train_df[feature_cols]
            y_train = train_df["label"]
            X_val = val_df[feature_cols]
            y_val = val_df["label"]

            # Tune hyperparameters if requested
            if perform_tuning:
                best_params = tune_hyperparameters(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_iter=n_iter,
                )
                params = {**hyperparameters, **best_params}
            else:
                params = hyperparameters

            # Train model
            model, metrics = train_xgboost(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **params,
            )

            # Get feature importance
            importance = self._get_feature_importance(model, feature_cols)

            # Save model
            self._save_model(horizon, model, params, metrics, importance)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"{horizon}-minute model trained in {elapsed:.2f}s - "
                f"AUC: {metrics['roc_auc']:.4f}"
            )

            models[horizon] = {
                "model": model,
                "metrics": metrics,
                "feature_importance": importance,
                "hyperparameters": params,
            }

        return models

    def _get_feature_importance(
        self, model: XGBClassifier, feature_names: list[str]
    ) -> dict[str, float]:
        """Extract feature importance from trained model.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature -> importance score
        """
        importance_scores = model.feature_importances_
        importance = dict(zip(feature_names, importance_scores))

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance

    def _save_model(
        self,
        horizon: int,
        model: XGBClassifier,
        hyperparameters: dict,
        metrics: dict,
        feature_importance: dict,
    ) -> None:
        """Save trained model and metadata.

        Args:
            horizon: Time horizon in minutes
            model: Trained model
            hyperparameters: Model hyperparameters
            metrics: Validation metrics
            feature_importance: Feature importance scores
        """
        # Create horizon directory
        horizon_dir = self._model_dir / f"{horizon}_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = horizon_dir / "xgboost_model.json"
        model.save_model(str(model_file))

        # Convert numpy arrays and types to JSON-serializable format
        metrics_copy = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_copy[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                metrics_copy[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                metrics_copy[key] = int(value)
            else:
                metrics_copy[key] = value

        # Convert feature importance values to Python floats
        importance_copy = {k: float(v) for k, v in feature_importance.items()}

        # Save metadata
        metadata = {
            "horizon": horizon,
            "hyperparameters": hyperparameters,
            "metrics": metrics_copy,
            "feature_importance": importance_copy,
            "trained_at": datetime.now().isoformat(),
        }

        metadata_file = horizon_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved model for {horizon}-minute horizon")

    def load_model(self, horizon: int) -> XGBClassifier:
        """Load trained model for specific time horizon.

        Args:
            horizon: Time horizon in minutes

        Returns:
            Loaded XGBoost model
        """
        model_file = self._model_dir / f"{horizon}_minute" / "xgboost_model.json"

        if not model_file.exists():
            raise FileNotFoundError(f"No model found for {horizon}-minute horizon")

        model = XGBClassifier()
        model.load_model(str(model_file))

        logger.debug(f"Loaded model for {horizon}-minute horizon")
        return model

    def load_metadata(self, horizon: int) -> dict:
        """Load model metadata for specific time horizon.

        Args:
            horizon: Time horizon in minutes

        Returns:
            Metadata dictionary
        """
        metadata_file = self._model_dir / f"{horizon}_minute" / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata found for {horizon}-minute horizon")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        return metadata

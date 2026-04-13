"""Probability Calibration for ML Model Predictions.

This module implements probability calibration using Platt scaling and
isotonic regression to ensure model predictions reflect actual probabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# Wrapper classes for pickling compatibility (must be at module level)
class _PlattWrapper:
    """Wrapper for Platt scaled model."""

    def __init__(self, base_model, platt_model):
        self.base_model = base_model
        self.platt_model = platt_model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        # Get calibrated probabilities from Platt scaling
        raw_reshaped = raw.reshape(-1, 1)
        calibrated = self.platt_model.predict_proba(raw_reshaped)
        return calibrated


class _IsotonicWrapper:
    """Wrapper for isotonic calibrated model."""

    def __init__(self, base_model, isotonic_model):
        self.base_model = base_model
        self.isotonic_model = isotonic_model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.isotonic_model.predict(raw)

        # Convert to 2D probability array
        probs = np.zeros((len(calibrated), 2))
        probs[:, 0] = 1 - calibrated
        probs[:, 1] = calibrated
        return probs


class ProbabilityCalibration:
    """Probability calibration wrapper for XGBoost models.

    Implements two calibration methods:
    - Platt Scaling (sigmoid): Parametric, good for small datasets
    - Isotonic Regression: Non-parametric, requires more data

    Attributes:
        method: Calibration method ('platt' or 'isotonic')
        model_dir: Directory for model storage
        calibrated_model: Fitted CalibratedClassifierCV instance
        brier_score: Calibration quality metric (< 0.15 target)
        calibration_metadata: Dictionary with calibration details
    """

    def __init__(
        self,
        method: Literal["platt", "isotonic"] = "platt",
        model_dir: str | Path = "data/models/xgboost/1_minute",
    ):
        """Initialize calibration with specified method.

        Args:
            method: Calibration method ('platt' or 'isotonic')
            model_dir: Directory for model storage

        Raises:
            ValueError: If method is not 'platt' or 'isotonic'
        """
        if method not in ["platt", "isotonic"]:
            raise ValueError(
                f"Invalid calibration method: {method}. "
                "Must be 'platt' or 'isotonic'."
            )

        self.method = method
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Calibration model (fitted after fit() is called)
        self.calibrated_model: CalibratedClassifierCV | None = None

        # Calibration metrics
        self.brier_score: float | None = None
        self.calibration_metadata: dict[str, object] = {}

        logger.info(f"ProbabilityCalibration initialized with method={method}")

    def fit(
        self,
        model: XGBClassifier,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "ProbabilityCalibration":
        """Fit calibration on validation data.

        Args:
            model: Trained XGBoost model
            validation_features: Validation set features (n_samples, n_features)
            validation_labels: Validation set true labels (n_samples,)

        Returns:
            Self (fitted calibration instance)

        Raises:
            ValueError: If validation data is empty or features/labels mismatch
        """
        if validation_features.shape[0] == 0:
            raise ValueError("Validation features cannot be empty")

        if validation_features.shape[0] != validation_labels.shape[0]:
            raise ValueError(
                f"Features and labels size mismatch: "
                f"{validation_features.shape[0]} != {validation_labels.shape[0]}"
            )

        logger.info(
            f"Fitting {self.method} calibration on "
            f"{validation_features.shape[0]} validation samples"
        )

        # Manual calibration for sklearn 1.8 / xgboost 2.0 compatibility
        if self.method == "platt":
            self._fit_platt_scaling(model, validation_features, validation_labels)
        else:  # isotonic
            self._fit_isotonic_regression(
                model, validation_features, validation_labels
            )

        # Calculate calibration metrics
        self._calculate_and_store_metrics(validation_features, validation_labels)

        logger.info(
            f"Calibration fitting complete. Brier score: {self.brier_score:.4f}"
        )

        return self

    def _fit_platt_scaling(
        self,
        model: XGBClassifier,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ):
        """Fit Platt scaling calibration manually.

        Platt scaling uses logistic regression to map model outputs to calibrated
        probabilities.

        Args:
            model: XGBoost model
            validation_features: Validation set features
            validation_labels: Validation set true labels
        """
        from sklearn.linear_model import LogisticRegression

        # Get raw predictions from XGBoost
        raw_preds = model.predict_proba(validation_features)[:, 1]

        # Fit Platt scaling (logistic regression) on raw predictions
        X_raw = raw_preds.reshape(-1, 1)

        # Platt scaling: logistic regression on raw scores
        self.platt_model = LogisticRegression()
        self.platt_model.fit(X_raw, validation_labels)

        # Store base model for prediction
        self._base_model = model

        # Create module-level wrapper for compatibility
        self.calibrated_model = _PlattWrapper(model, self.platt_model)

    def _fit_isotonic_regression(
        self,
        model: XGBClassifier,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ):
        """Fit isotonic regression calibration manually.

        Isotonic regression is a non-parametric approach that fits a monotonic
        function to map raw probabilities to calibrated probabilities.

        Args:
            model: XGBoost model
            validation_features: Validation set features
            validation_labels: Validation set true labels
        """
        from sklearn.isotonic import IsotonicRegression

        # Get raw predictions
        raw_preds = model.predict_proba(validation_features)[:, 1]

        # Fit isotonic regression
        self.isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self.isotonic_model.fit(raw_preds, validation_labels)

        # Store base model for prediction
        self._base_model = model

        # Create module-level wrapper for compatibility
        self.calibrated_model = _IsotonicWrapper(model, self.isotonic_model)

    def predict_proba(self, features: np.ndarray) -> float:
        """Generate calibrated probability prediction.

        Args:
            features: Feature array for prediction (n_features,)

        Returns:
            Calibrated probability (0.0 to 1.0)

        Raises:
            ValueError: If calibration not fitted or features invalid
        """
        if self.calibrated_model is None:
            raise ValueError("Calibration not fitted. Call fit() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Predict probability of positive class
        probability = self.calibrated_model.predict_proba(features)[:, 1][0]

        return float(probability)

    def calculate_calibration_metrics(
        self, validation_features: np.ndarray, validation_labels: np.ndarray
    ) -> dict[str, float]:
        """Calculate calibration quality metrics.

        Args:
            validation_features: Validation set features
            validation_labels: Validation set true labels

        Returns:
            Dictionary with Brier score, calibration deviation, etc.
        """
        if self.calibrated_model is None:
            raise ValueError("Calibration not fitted. Call fit() first.")

        # Get calibrated predictions
        calibrated_probs = self.calibrated_model.predict_proba(
            validation_features
        )[:, 1]

        # Calculate Brier score
        brier = brier_score_loss(validation_labels, calibrated_probs)

        # Calculate calibration curve deviation
        max_deviation = self._calculate_calibration_deviation(
            calibrated_probs, validation_labels
        )

        # Calculate mean predicted probability vs actual win rate
        mean_predicted_prob = float(np.mean(calibrated_probs))
        actual_win_rate = float(np.mean(validation_labels))

        return {
            "brier_score": brier,
            "max_calibration_deviation": max_deviation,
            "mean_predicted_probability": mean_predicted_prob,
            "actual_win_rate": actual_win_rate,
        }

    def _calculate_and_store_metrics(
        self, validation_features: np.ndarray, validation_labels: np.ndarray
    ):
        """Calculate and store calibration metrics.

        Args:
            validation_features: Validation set features
            validation_labels: Validation set true labels
        """
        metrics = self.calculate_calibration_metrics(
            validation_features, validation_labels
        )

        self.brier_score = metrics["brier_score"]
        self.calibration_metadata = {
            "brier_score": self.brier_score,
            "max_calibration_deviation": metrics["max_calibration_deviation"],
            "mean_predicted_probability": metrics["mean_predicted_probability"],
            "actual_win_rate": metrics["actual_win_rate"],
        }

    def _calculate_calibration_deviation(
        self, predicted_probs: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """Calculate maximum deviation from perfect calibration.

        Args:
            predicted_probs: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Maximum deviation from perfect calibration (0.0 to 1.0)
        """
        # Bin probabilities into deciles
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)

        max_deviation = 0.0

        for i in range(num_bins):
            mask = (predicted_probs >= bin_edges[i]) & (
                predicted_probs < bin_edges[i + 1]
            )

            if i == num_bins - 1:
                # Include 1.0 in last bin
                mask = mask | (predicted_probs == 1.0)

            if np.sum(mask) > 0:
                # Actual frequency in this bin
                actual_freq = np.mean(true_labels[mask])

                # Expected frequency (bin midpoint)
                expected_freq = (bin_edges[i] + bin_edges[i + 1]) / 2

                # Deviation from perfect calibration
                deviation = abs(actual_freq - expected_freq)
                max_deviation = max(max_deviation, deviation)

        return float(max_deviation)

    def save(self) -> Path:
        """Persist calibrated model and metadata.

        Returns:
            Path to saved calibrated model
        """
        if self.calibrated_model is None:
            raise ValueError("No calibration model to save. Call fit() first.")

        # Create metadata
        metadata = {
            "method": self.method,
            "brier_score": self.brier_score,
            "max_calibration_deviation": self.calibration_metadata.get(
                "max_calibration_deviation"
            ),
            "mean_predicted_probability": self.calibration_metadata.get(
                "mean_predicted_probability"
            ),
            "actual_win_rate": self.calibration_metadata.get("actual_win_rate"),
            "timestamp": datetime.now().isoformat(),
        }

        # Save calibrated model
        model_path = self.model_dir / "calibrated_model.joblib"
        joblib.dump(self.calibrated_model, model_path)

        # Save metadata
        metadata_path = self.model_dir / "calibration_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Calibration model saved to {model_path}")
        logger.info(f"Calibration metadata saved to {metadata_path}")

        return model_path

    @classmethod
    def load(cls, model_path: Path) -> "ProbabilityCalibration":
        """Load calibrated model from disk.

        Args:
            model_path: Path to calibrated_model.joblib

        Returns:
            Loaded ProbabilityCalibration instance

        Raises:
            FileNotFoundError: If model or metadata files don't exist
        """
        model_path = Path(model_path)
        model_dir = model_path.parent

        if not model_path.exists():
            raise FileNotFoundError(f"Calibration model not found: {model_path}")

        # Load calibrated model
        calibrated_model = joblib.load(model_path)

        # Load metadata
        metadata_path = model_dir / "calibration_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Calibration metadata not found: {metadata_path}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create instance
        method = metadata["method"]
        calibration = cls(method=method, model_dir=model_dir)
        calibration.calibrated_model = calibrated_model
        calibration.brier_score = metadata.get("brier_score")
        calibration.calibration_metadata = metadata

        logger.info(f"Calibration model loaded from {model_path}")

        return calibration

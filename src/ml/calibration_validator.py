"""Calibration Validator for Historical MNQ Dataset.

This module validates probability calibration on historical MNQ data,
specifically the March 2025 ranging market failure case.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.ml.features import FeatureEngineer
from src.ml.probability_calibration import ProbabilityCalibration

logger = logging.getLogger(__name__)


class CalibrationValidator:
    """Validate probability calibration on historical MNQ dataset.

    Validates that the calibration layer fixes the March 2025 overconfidence
    issue where the model was 99.25% confident but only achieved 28.4% win rate.
    """

    def __init__(self, data_path: str = "data/processed/dollar_bars/1_minute"):
        """Initialize validator with MNQ dataset path.

        Args:
            data_path: Path to dollar bar data directory
        """
        self.data_path = Path(data_path)
        self.feature_engineer = FeatureEngineer()
        self.march_2025_data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.labels: pd.Series | None = None

        logger.info(f"CalibrationValidator initialized with data_path: {data_path}")

    def load_march_2025_data(self, feature_filter: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
        """Load March 2025 features and labels.

        Loads the March 2025 dollar bar data, extracts features, and generates
        labels based on 5-minute forward returns.

        Args:
            feature_filter: Optional list of feature names to filter to.

        Returns:
            (features_df, labels_series) for March 2025 period
        """
        logger.info("Loading March 2025 data...")

        # Load 2025 dollar bar data
        data_file = self.data_path / "mnq_1min_2025.csv"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}. "
                f"Expected path: {data_file.absolute()}"
            )

        # Load data
        df = pd.read_csv(data_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Handle timezone-aware and naive timestamps
        if df["timestamp"].dt.tz is not None:
            # Data has timezone, use UTC for filtering
            march_start = pd.Timestamp("2025-03-01", tz="UTC")
            march_end = pd.Timestamp("2025-03-31 23:59:59", tz="UTC")
        else:
            # Data is timezone-naive, use naive timestamps for filtering
            march_start = pd.Timestamp("2025-03-01")
            march_end = pd.Timestamp("2025-03-31 23:59:59")

        # Filter to March 2025
        df_march = df.loc[(df["timestamp"] >= march_start) & (df["timestamp"] <= march_end)]

        # Reset index to ensure timestamp is available as a column for FeatureEngineer
        df_march = df_march.reset_index(drop=True)

        logger.info(f"Loaded {len(df_march)} bars for March 2025")

        if len(df_march) == 0:
            raise ValueError(
                f"No data found for March 2025. "
                f"Available date range: {df.index.min()} to {df.index.max()}"
            )

        self.march_2025_data = df_march

        # Extract features using FeatureEngineer
        logger.info("Extracting features...")

        # Engineer features using sliding window
        features_df = self.feature_engineer.engineer_features(df_march)

        # Drop non-numeric columns (timestamp) and keep only numeric features
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]

        # Remove rows with NaN values (from feature calculation windows)
        features_df = features_df.dropna()

        # Filter to selected features if provided
        if feature_filter:
            available_features = [f for f in feature_filter if f in features_df.columns]
            if len(available_features) < len(feature_filter):
                missing_features = set(feature_filter) - set(available_features)
                logger.warning(
                    f"Missing {len(missing_features)} features: {missing_features}"
                )
            features_df = features_df[available_features]
            logger.info(f"Filtered to {len(available_features)} selected features")

        # Generate labels based on 5-minute forward returns
        logger.info("Generating labels...")

        # Get close prices and calculate forward returns
        close_prices = df_march["close"].iloc[len(df_march) - len(features_df):]
        forward_returns = close_prices.pct_change(5).shift(-5)

        # Create labels: 1 if positive return, 0 otherwise
        labels_series = (forward_returns > 0).astype(int)

        # Remove last 5 rows (no forward returns available)
        self.labels = labels_series.iloc[:-5]
        self.features = features_df.iloc[: len(self.labels)]

        # Ensure alignment
        min_length = min(len(self.features), len(self.labels))
        self.features = self.features.iloc[:min_length]
        self.labels = self.labels.iloc[:min_length]

        logger.info(
            f"Extracted {len(self.features)} features, "
            f"win rate: {self.labels.mean():.2%}"
        )

        return self.features, self.labels

    def train_calibration(
        self,
        model: xgb.XGBClassifier,
        method: Literal["platt", "isotonic"] = "platt",
        train_split: float = 0.7,
    ) -> ProbabilityCalibration:
        """Train calibration on March 2025 data.

        Splits data into training and validation sets, then fits calibration
        on the validation set.

        Args:
            model: Trained XGBoost model
            method: Calibration method (platt or isotonic)
            train_split: Training data proportion (default: 0.7)

        Returns:
            Trained ProbabilityCalibration instance
        """
        if self.features is None or self.labels is None:
            raise ValueError(
                "Must call load_march_2025_data() before train_calibration()"
            )

        logger.info(f"Training {method} calibration...")

        # Split data
        n_train = int(len(self.features) * train_split)
        X_train = self.features.iloc[:n_train].values
        y_train = self.labels.iloc[:n_train].values
        X_val = self.features.iloc[n_train:].values
        y_val = self.labels.iloc[n_train:].values

        logger.info(
            f"Train set: {len(X_train)} samples, "
            f"Validation set: {len(X_val)} samples"
        )

        # Initialize and fit calibration
        calibration = ProbabilityCalibration(method=method)
        calibration.fit(model, X_val, y_val)

        logger.info(
            f"Calibration trained. Brier score: {calibration.brier_score:.4f}"
        )

        return calibration

    def validate_calibration_quality(
        self, calibration: ProbabilityCalibration
    ) -> dict[str, float]:
        """Validate calibration quality metrics.

        Calculates Brier score, calibration deviation, and probability match
        to assess calibration quality.

        Args:
            calibration: Trained ProbabilityCalibration instance

        Returns:
            Dictionary with calibration quality metrics
        """
        if self.features is None or self.labels is None:
            raise ValueError(
                "Must call load_march_2025_data() before validate_calibration_quality()"
            )

        logger.info("Validating calibration quality...")

        # Split data (use same split as training)
        n_train = int(len(self.features) * 0.7)
        X_val = self.features.iloc[n_train:].values
        y_val = self.labels.iloc[n_train:].values

        # Calculate metrics using calibration's built-in method
        metrics = calibration.calculate_calibration_metrics(X_val, y_val)

        # Calculate probability match
        mean_predicted_prob = metrics["mean_predicted_probability"]
        actual_win_rate = metrics["actual_win_rate"]
        probability_match = abs(mean_predicted_prob - actual_win_rate)

        metrics["probability_match"] = probability_match

        logger.info(
            f"Calibration quality - "
            f"Brier score: {metrics['brier_score']:.4f}, "
            f"Deviation: {metrics['max_calibration_deviation']:.4f}, "
            f"Match: {probability_match:.4f}"
        )

        return metrics

    def compare_uncalibrated_vs_calibrated(
        self,
        model: xgb.XGBClassifier,
        calibration: ProbabilityCalibration,
    ) -> dict[str, dict[str, float]]:
        """Compare uncalibrated vs calibrated predictions.

        Args:
            model: Trained XGBoost model
            calibration: Trained ProbabilityCalibration instance

        Returns:
            Dictionary with uncalibrated and calibrated metrics
        """
        if self.features is None or self.labels is None:
            raise ValueError(
                "Must call load_march_2025_data() before compare_uncalibrated_vs_calibrated()"
            )

        logger.info("Comparing uncalibrated vs calibrated...")

        # Split data
        n_train = int(len(self.features) * 0.7)
        X_val = self.features.iloc[n_train:].values
        y_val = self.labels.iloc[n_train:].values

        # Get uncalibrated predictions
        uncalibrated_probs = model.predict_proba(X_val)[:, 1]

        # Get calibrated predictions
        calibrated_probs = np.array(
            [calibration.predict_proba(x.reshape(1, -1)) for x in X_val]
        )

        # Calculate metrics
        actual_win_rate = np.mean(y_val)

        comparison = {
            "uncalibrated": {
                "mean_prob": float(np.mean(uncalibrated_probs)),
                "actual_win_rate": float(actual_win_rate),
                "brier_score": float(
                    brier_score_loss(y_val, uncalibrated_probs)
                ),
            },
            f"calibrated_{calibration.method}": {
                "mean_prob": float(np.mean(calibrated_probs)),
                "actual_win_rate": float(actual_win_rate),
                "brier_score": float(brier_score_loss(y_val, calibrated_probs)),
            },
        }

        logger.info(
            f"Uncalibrated: {comparison['uncalibrated']['mean_prob']:.2%} "
            f"vs actual {actual_win_rate:.2%}"
        )
        logger.info(
            f"Calibrated ({calibration.method}): "
            f"{comparison[f'calibrated_{calibration.method}']['mean_prob']:.2%} "
            f"vs actual {actual_win_rate:.2%}"
        )

        return comparison

    def generate_calibration_curve(
        self,
        model: xgb.XGBClassifier,
        calibration: ProbabilityCalibration,
        save_path: str = "docs/calibration_curve_march_2025.png",
    ) -> None:
        """Generate calibration curve visualization.

        Creates a reliability diagram showing uncalibrated vs calibrated
        predictions compared to perfect calibration.

        Args:
            model: Trained XGBoost model
            calibration: Trained ProbabilityCalibration instance
            save_path: Path to save the visualization
        """
        if self.features is None or self.labels is None:
            raise ValueError(
                "Must call load_march_2025_data() before generate_calibration_curve()"
            )

        logger.info("Generating calibration curve visualization...")

        # Split data
        n_train = int(len(self.features) * 0.7)
        X_val = self.features.iloc[n_train:].values
        y_val = self.labels.iloc[n_train:].values

        # Get predictions
        uncalibrated_probs = model.predict_proba(X_val)[:, 1]
        calibrated_probs = np.array(
            [calibration.predict_proba(x.reshape(1, -1)) for x in X_val]
        )

        # Create calibration curves
        n_bins = 10
        (
            uncalibrated_true_prob,
            uncalibrated_pred_prob,
        ) = calibration_curve(y_val, uncalibrated_probs, n_bins=n_bins)
        (
            calibrated_true_prob,
            calibrated_pred_prob,
        ) = calibration_curve(y_val, calibrated_probs, n_bins=n_bins)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot reliability diagram
        ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        ax1.plot(
            uncalibrated_pred_prob,
            uncalibrated_true_prob,
            "s-",
            label=f"Uncalibrated (Brier={brier_score_loss(y_val, uncalibrated_probs):.3f})",
        )
        ax1.plot(
            calibrated_pred_prob,
            calibrated_true_prob,
            "^-",
            label=f"Calibrated {calibration.method.title()} "
            f"(Brier={brier_score_loss(y_val, calibrated_probs):.3f})",
        )

        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Actual Win Rate")
        ax1.set_title("Calibration Curve - March 2025")
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot probability distribution
        ax2.hist(
            uncalibrated_probs,
            bins=50,
            alpha=0.5,
            label=f"Uncalibrated (mean={np.mean(uncalibrated_probs):.2%})",
        )
        ax2.hist(
            calibrated_probs,
            bins=50,
            alpha=0.5,
            label=f"Calibrated {calibration.method.title()} "
            f"(mean={np.mean(calibrated_probs):.2%})",
        )

        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Probability Distribution - March 2025")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Calibration curve saved to {save_path}")

    def generate_validation_report(
        self,
        model: xgb.XGBClassifier,
        calibration: ProbabilityCalibration,
        output_path: str = "data/models/xgboost/1_minute/validation_report_march_2025.json",
    ) -> dict:
        """Generate comprehensive validation report.

        Args:
            model: Trained XGBoost model
            calibration: Trained ProbabilityCalibration instance
            output_path: Path to save the validation report

        Returns:
            Dictionary with validation results
        """
        logger.info("Generating validation report...")

        # Get metrics
        metrics = self.validate_calibration_quality(calibration)
        comparison = self.compare_uncalibrated_vs_calibrated(model, calibration)

        # Create report
        report = {
            "validation_date": datetime.now().isoformat(),
            "period": "March 2025",
            "market_regime": "Ranging market (low volatility, mean reversion)",
            "calibration_method": calibration.method,
            "metrics": metrics,
            "comparison": comparison,
            "success_criteria": {
                "brier_score_target": "< 0.15",
                "brier_score_actual": metrics["brier_score"],
                "brier_score_passed": metrics["brier_score"] < 0.15,
                "calibration_deviation_target": "< 0.05",
                "calibration_deviation_actual": metrics["max_calibration_deviation"],
                "calibration_deviation_passed": metrics["max_calibration_deviation"]
                < 0.05,
                "probability_match_target": "< 0.05",
                "probability_match_actual": metrics["probability_match"],
                "probability_match_passed": metrics["probability_match"] < 0.05,
            },
            "overconfidence_fix": {
                "uncalibrated_mean_prob": comparison["uncalibrated"]["mean_prob"],
                "calibrated_mean_prob": comparison[f"calibrated_{calibration.method}"][
                    "mean_prob"
                ],
                "actual_win_rate": comparison["uncalibrated"]["actual_win_rate"],
                "overconfidence_fixed": abs(
                    comparison[f"calibrated_{calibration.method}"]["mean_prob"]
                    - comparison["uncalibrated"]["actual_win_rate"]
                )
                < 0.05,
            },
        }

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")

        return report

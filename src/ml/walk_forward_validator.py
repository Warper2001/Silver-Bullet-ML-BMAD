"""Walk-Forward Validation for Trading Models.

This module implements rolling window backtesting to simulate real trading
conditions and prevent overfitting. Unlike traditional train/test splits,
walk-forward validation uses time-based windows that roll forward through
historical data.

Example:
    Training: [Jan-Mar] → Test: [Apr]
    Training: [Feb-Apr] → Test: [May]
    Training: [Mar-May] → Test: [Jun]
    ...

This prevents look-ahead bias and provides realistic performance estimates.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.ml.inference import MLInference

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a single validation window.

    Attributes:
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date
        test_end: Test period end date
        train_metrics: Performance on training data
        test_metrics: Performance on test data
        generalization_gap: Difference between train and test performance
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    train_metrics: dict[str, float]
    test_metrics: dict[str, float]

    generalization_gap: float
    n_train_samples: int
    n_test_samples: int

    def is_overfit(self, threshold: float = 0.1) -> bool:
        """Check if model is overfitting.

        Args:
            threshold: Maximum allowed gap between train and test performance

        Returns:
            True if generalization gap exceeds threshold
        """
        return abs(self.generalization_gap) > threshold


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation.

    Attributes:
        validations: List of individual validation results
        mean_test_performance: Average performance across all test windows
        std_test_performance: Standard deviation of test performance
        best_window: Best performing validation window
        worst_window: Worst performing validation window
        performance_decay: Performance trend over time
    """

    validations: list[ValidationResult]
    mean_test_performance: dict[str, float]
    std_test_performance: dict[str, float]
    best_window: ValidationResult
    worst_window: ValidationResult
    performance_decay: dict[str, float]

    def get_realistic_win_rate(self) -> float:
        """Get realistic win rate expectation from walk-forward validation.

        Returns:
            Mean win rate across all test windows
        """
        return self.mean_test_performance.get("win_rate", 0.0)

    def get_performance_stability(self) -> float:
        """Get performance stability (inverse of standard deviation).

        Returns:
            Stability score (0-1, higher is more stable)
        """
        std_win_rate = self.std_test_performance.get("win_rate", 1.0)
        return max(0.0, 1.0 - std_win_rate)


class WalkForwardValidator:
    """Walk-forward validation for time-series trading models.

    Implements rolling window validation to simulate real trading conditions
    and detect overfitting.

    Usage:
        >>> validator = WalkForwardValidator(
        ...     train_months=3,
        ...     test_months=1,
        ...     step_months=1
        ... )
        >>> results = validator.validate(
        ...     data=historical_data,
        ...     model_trainer=train_xgboost_model
        ... )
        >>> print(f"Realistic win rate: {results.get_realistic_win_rate():.2%}")
    """

    def __init__(
        self,
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1,
        min_train_samples: int = 100,
    ):
        """Initialize walk-forward validator.

        Args:
            train_months: Number of months in training window
            test_months: Number of months in test window
            step_months: Number of months to roll forward each iteration
            min_train_samples: Minimum samples required for training
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_train_samples = min_train_samples

        logger.info(
            f"WalkForwardValidator initialized: "
            f"{train_months}mo train, {test_months}mo test, {step_months}mo step"
        )

    def validate(
        self,
        data: pd.DataFrame,
        model_trainer: Callable,
        feature_cols: list[str],
        target_col: str = "success",
    ) -> WalkForwardResults:
        """Run walk-forward validation.

        Args:
            data: Historical data with datetime index
            model_trainer: Function that trains a model and returns predictions
                           Signature: (X_train, y_train, X_test) -> (y_pred, y_prob)
            feature_cols: List of feature column names
            target_col: Name of target column

        Returns:
            WalkForwardResults with aggregated performance metrics
        """
        logger.info("Starting walk-forward validation...")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")

        # Generate validation windows
        windows = self._generate_windows(data)
        logger.info(f"Generated {len(windows)} validation windows")

        # Run validation on each window
        validations = []
        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"Validating window {i + 1}/{len(windows)}...")

            result = self._validate_window(
                train_data=train_data,
                test_data=test_data,
                model_trainer=model_trainer,
                feature_cols=feature_cols,
                target_col=target_col,
            )
            validations.append(result)

            # Log results
            logger.info(
                f"  Train win rate: {result.train_metrics['win_rate']:.2%}, "
                f"Test win rate: {result.test_metrics['win_rate']:.2%}, "
                f"Gap: {result.generalization_gap:.2%}"
            )

        # Aggregate results
        aggregated = self._aggregate_results(validations)

        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Mean test win rate: {aggregated.get_realistic_win_rate():.2%}")
        logger.info(f"Std test win rate: {aggregated.std_test_performance['win_rate']:.2%}")
        logger.info(f"Performance stability: {aggregated.get_performance_stability():.2%}")
        logger.info(f"Best window: {aggregated.best_window.test_start.strftime('%Y-%m')}")
        logger.info(f"  Win rate: {aggregated.best_window.test_metrics['win_rate']:.2%}")
        logger.info(f"Worst window: {aggregated.worst_window.test_start.strftime('%Y-%m')}")
        logger.info(f"  Win rate: {aggregated.worst_window.test_metrics['win_rate']:.2%}")
        logger.info("=" * 70)

        return aggregated

    def _generate_windows(
        self, data: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate rolling train/test windows.

        Args:
            data: Full dataset with datetime index

        Returns:
            List of (train_data, test_data) tuples
        """
        windows = []
        current_start = data.index.min()

        while True:
            # Define window boundaries
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Check if we have enough data
            if test_end > data.index.max():
                break

            # Split data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            # Check minimum sample size
            if len(train_data) < self.min_train_samples or len(test_data) < 10:
                current_start = train_start + pd.DateOffset(months=self.step_months)
                continue

            windows.append((train_data, test_data))

            # Roll forward
            current_start = train_start + pd.DateOffset(months=self.step_months)

        return windows

    def _validate_window(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_trainer: Callable,
        feature_cols: list[str],
        target_col: str,
    ) -> ValidationResult:
        """Validate a single train/test window.

        Args:
            train_data: Training data
            test_data: Test data
            model_trainer: Model training function
            feature_cols: Feature column names
            target_col: Target column name

        Returns:
            ValidationResult with metrics
        """
        # Prepare data
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Train model and get predictions
        y_pred_train, y_prob_train = model_trainer(
            X_train, y_train, X_train, return_prob=True
        )
        y_pred_test, y_prob_test = model_trainer(
            X_train, y_train, X_test, return_prob=True
        )

        # Calculate metrics
        train_metrics = self._calculate_metrics(
            y_train, y_pred_train, y_prob_train
        )
        test_metrics = self._calculate_metrics(y_test, y_pred_test, y_prob_test)

        # Calculate generalization gap
        generalization_gap = train_metrics["win_rate"] - test_metrics["win_rate"]

        return ValidationResult(
            train_start=train_data.index.min(),
            train_end=train_data.index.max(),
            test_start=test_data.index.min(),
            test_end=test_data.index.max(),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            generalization_gap=generalization_gap,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
        )

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, float]:
        """Calculate performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "win_rate": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        }

        return metrics

    def _aggregate_results(
        self, validations: list[ValidationResult]
    ) -> WalkForwardResults:
        """Aggregate results from all validation windows.

        Args:
            validations: List of validation results

        Returns:
            WalkForwardResults with aggregated metrics
        """
        # Extract test metrics
        test_metrics = [v.test_metrics for v in validations]

        # Calculate mean and std
        metric_names = test_metrics[0].keys()
        mean_metrics = {}
        std_metrics = {}

        for metric in metric_names:
            values = [m[metric] for m in test_metrics]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)

        # Find best and worst windows
        best_window = max(validations, key=lambda v: v.test_metrics["win_rate"])
        worst_window = min(validations, key=lambda v: v.test_metrics["win_rate"])

        # Calculate performance decay (trend over time)
        win_rates = [v.test_metrics["win_rate"] for v in validations]
        if len(win_rates) >= 2:
            # Simple linear regression to find trend
            x = np.arange(len(win_rates))
            slope = np.polyfit(x, win_rates, 1)[0]
            performance_decay = {"win_rate_trend": slope}
        else:
            performance_decay = {"win_rate_trend": 0.0}

        return WalkForwardResults(
            validations=validations,
            mean_test_performance=mean_metrics,
            std_test_performance=std_metrics,
            best_window=best_window,
            worst_window=worst_window,
            performance_decay=performance_decay,
        )

    def save_results(self, results: WalkForwardResults, output_path: Path) -> None:
        """Save validation results to disk.

        Args:
            results: WalkForwardResults to save
            output_path: Path to save results (JSON format)
        """
        import json

        # Convert to serializable format
        results_dict = {
            "mean_test_performance": results.mean_test_performance,
            "std_test_performance": results.std_test_performance,
            "best_window": {
                "period": f"{results.best_window.test_start.strftime('%Y-%m')}",
                "win_rate": results.best_window.test_metrics["win_rate"],
            },
            "worst_window": {
                "period": f"{results.worst_window.test_start.strftime('%Y-%m')}",
                "win_rate": results.worst_window.test_metrics["win_rate"],
            },
            "performance_decay": results.performance_decay,
            "realistic_win_rate": results.get_realistic_win_rate(),
            "performance_stability": results.get_performance_stability(),
            "validations": [
                {
                    "period": f"{v.test_start.strftime('%Y-%m')}",
                    "train_win_rate": v.train_metrics["win_rate"],
                    "test_win_rate": v.test_metrics["win_rate"],
                    "generalization_gap": v.generalization_gap,
                    "is_overfit": v.is_overfit(),
                }
                for v in results.validations
            ],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_path}")

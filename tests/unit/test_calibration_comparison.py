"""Unit tests for calibration comparison module.

Tests the side-by-side comparison of uncalibrated vs calibrated models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.research.backtest_engine import BacktestEngine, CalibrationComparison


class TestCalibrationComparison:
    """Test calibration comparison functionality."""

    def test_calibration_comparison_initialization(self):
        """Test CalibrationComparison can be initialized with models."""
        # Arrange
        uncalibrated_model = Mock()
        calibrated_model = Mock()
        calibration = Mock()

        # Act
        comparison = CalibrationComparison(
            uncalibrated_model=uncalibrated_model,
            calibrated_model=calibrated_model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Assert
        assert comparison.uncalibrated_model == uncalibrated_model
        assert comparison.calibrated_model == calibrated_model
        assert comparison.calibration == calibration
        assert comparison.data_path == Path("data/processed/dollar_bars/1_minute")

    def test_run_uncalibrated_backtest(self):
        """Test running backtest with uncalibrated model."""
        # Arrange
        model = Mock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        calibration = Mock()
        comparison = CalibrationComparison(
            uncalibrated_model=model,
            calibrated_model=Mock(),
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Mock data loading
        comparison._load_data = Mock(return_value=(pd.DataFrame(), pd.Series()))

        # Act
        result = comparison.run_uncalibrated_backtest(
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        # Assert
        assert "win_rate" in result
        assert "mean_predicted_probability" in result
        assert "trade_count" in result
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_run_calibrated_backtest(self):
        """Test running backtest with calibrated model."""
        # Arrange
        model = Mock()
        calibration = Mock()
        calibration.predict_proba.return_value = np.array([0.5, 0.6])
        comparison = CalibrationComparison(
            uncalibrated_model=Mock(),
            calibrated_model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Mock data loading
        comparison._load_data = Mock(return_value=(pd.DataFrame(), pd.Series()))

        # Act
        result = comparison.run_calibrated_backtest(
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        # Assert
        assert "win_rate" in result
        assert "mean_predicted_probability" in result
        assert "trade_count" in result
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_generate_comparison_metrics(self):
        """Test generating side-by-side comparison metrics."""
        # Arrange
        uncalibrated_result = {
            "win_rate": 0.60,
            "mean_predicted_probability": 0.75,
            "brier_score": 0.28,
            "trade_count": 100
        }
        calibrated_result = {
            "win_rate": 0.62,
            "mean_predicted_probability": 0.61,
            "brier_score": 0.14,
            "trade_count": 98
        }

        comparison = CalibrationComparison(
            uncalibrated_model=Mock(),
            calibrated_model=Mock(),
            calibration=Mock(),
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Act
        metrics = comparison.generate_comparison_metrics(
            uncalibrated_result, calibrated_result
        )

        # Assert
        assert "win_rate_improvement" in metrics
        assert "brier_score_improvement" in metrics
        assert "probability_match_improvement" in metrics
        assert metrics["win_rate_improvement"] == pytest.approx(0.02, abs=0.01)
        assert metrics["brier_score_improvement"] == pytest.approx(0.14, abs=0.01)

    def test_calculate_brier_score(self):
        """Test Brier score calculation."""
        # Arrange
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.3])

        comparison = CalibrationComparison(
            uncalibrated_model=Mock(),
            calibrated_model=Mock(),
            calibration=Mock(),
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Act
        brier = comparison._calculate_brier_score(y_true, y_prob)

        # Assert
        assert brier >= 0.0
        assert brier < 1.0  # Brier score is between 0 and 1

    def test_side_by_side_comparison_full_workflow(self):
        """Test complete side-by-side comparison workflow."""
        # Arrange
        model = Mock()
        model.predict_proba.return_value = np.array([[0.4, 0.6], [0.7, 0.3]])
        calibration = Mock()
        calibration.predict_proba.return_value = np.array([0.55, 0.45])

        comparison = CalibrationComparison(
            uncalibrated_model=model,
            calibrated_model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Mock data loading
        mock_features = pd.DataFrame({
            "feature1": [1.0, 2.0],
            "feature2": [3.0, 4.0]
        })
        mock_labels = pd.Series([1, 0])
        comparison._load_data = Mock(return_value=(mock_features, mock_labels))

        # Act
        result = comparison.run_side_by_side_comparison(
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        # Assert
        assert "uncalibrated" in result
        assert "calibrated" in result
        assert "comparison" in result
        assert "win_rate" in result["uncalibrated"]
        assert "win_rate" in result["calibrated"]
        assert "win_rate_improvement" in result["comparison"]
        assert "brier_score_improvement" in result["comparison"]

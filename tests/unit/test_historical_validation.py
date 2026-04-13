"""Unit tests for historical validation script.

Tests the comprehensive validation script for calibration on 2-year MNQ dataset.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from scripts.validate_historical_calibration import HistoricalValidationRunner


class TestHistoricalValidationRunner:
    """Test historical validation runner."""

    def test_initialization(self):
        """Test HistoricalValidationRunner can be initialized."""
        # Arrange
        model = Mock()
        calibration = Mock()

        # Act
        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Assert
        assert runner.model == model
        assert runner.calibration == calibration
        assert runner.data_path == Path("data/processed/dollar_bars/1_minute")

    def test_load_data_from_csv(self):
        """Test loading MNQ data from CSV files."""
        # Arrange
        model = Mock()
        calibration = Mock()
        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Act
        features, labels = runner.load_data(
            start_date="2025-01-01",
            end_date="2025-03-31"
        )

        # Assert
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert len(features) > 0
        assert len(labels) > 0
        assert len(features) == len(labels)

    def test_generate_comparison_report(self):
        """Test generating comparison report."""
        # Arrange
        model = Mock()
        calibration = Mock()
        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

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

        # Act
        report = runner.generate_comparison_report(
            uncalibrated_result,
            calibrated_result,
            output_path="data/reports/test_comparison.csv"
        )

        # Assert
        assert Path("data/reports/test_comparison.csv").exists()
        assert report["win_rate_uncalibrated"] == 0.60
        assert report["win_rate_calibrated"] == 0.62
        assert report["brier_score_uncalibrated"] == 0.28
        assert report["brier_score_calibrated"] == 0.14

    def test_run_full_validation(self):
        """Test running full historical validation."""
        # Arrange
        model = Mock()
        # Mock that returns predictions based on input size
        def mock_predict_proba(X):
            n_samples = X.shape[0]
            # Return n_samples predictions
            return np.array([[0.4, 0.6]] * n_samples)

        model.predict_proba = mock_predict_proba

        calibration = Mock()
        # Mock calibration that returns predictions based on input size
        def mock_calibrate(X):
            n_samples = X.shape[0]
            return np.array([0.55] * n_samples)

        calibration.predict_proba = mock_calibrate

        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Act
        results = runner.run_validation(
            start_date="2025-03-01",
            end_date="2025-03-31",
            output_dir="data/reports"
        )

        # Assert
        assert "uncalibrated" in results
        assert "calibrated" in results
        assert "comparison" in results
        assert "comparison_report_path" in results
        assert "validation_report_path" in results

    def test_save_validation_report(self):
        """Test saving validation report to markdown."""
        # Arrange
        model = Mock()
        calibration = Mock()
        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        results = {
            "uncalibrated": {"win_rate": 0.60, "mean_predicted_probability": 0.75},
            "calibrated": {"win_rate": 0.62, "mean_predicted_probability": 0.61},
            "comparison": {"win_rate_improvement": 0.02}
        }

        # Act
        report_path = runner.save_validation_report(
            results=results,
            output_path="data/reports/test_validation_report.md"
        )

        # Assert
        assert Path(report_path).exists()
        assert report_path.endswith("test_validation_report.md")

        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
        assert "# Historical Validation Report" in content
        assert "Win Rate" in content
        assert "Brier Score" in content

    def test_validate_march_2025_failure_case(self):
        """Test March 2025 specific failure case analysis."""
        # Arrange
        model = Mock()

        def mock_predict_proba(X):
            n_samples = X.shape[0]
            return np.array([[0.4, 0.6]] * n_samples)

        model.predict_proba = mock_predict_proba

        calibration = Mock()

        def mock_calibrate(X):
            n_samples = X.shape[0]
            return np.array([0.55] * n_samples)

        calibration.predict_proba = mock_calibrate

        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Act
        results = runner.validate_march_2025_failure_case(
            output_dir="data/reports"
        )

        # Assert
        assert "march_analysis" in results
        assert "march_report_path" in results
        assert results["march_analysis"]["period"] == "March 2025"
        assert results["march_analysis"]["original_loss_percent"] == pytest.approx(-8.56, abs=0.01)
        assert "loss_prevented" in results["march_analysis"]
        assert results["march_analysis"]["uncalibrated_brier_score"] >= 0
        assert results["march_analysis"]["calibrated_brier_score"] >= 0
        assert Path(results["march_report_path"]).exists()

    def test_march_2025_report_content(self):
        """Test March 2025 report contains required sections."""
        # Arrange
        model = Mock()

        def mock_predict_proba(X):
            n_samples = X.shape[0]
            return np.array([[0.4, 0.6]] * n_samples)

        model.predict_proba = mock_predict_proba

        calibration = Mock()

        def mock_calibrate(X):
            n_samples = X.shape[0]
            return np.array([0.55] * n_samples)

        calibration.predict_proba = mock_calibrate

        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Act
        results = runner.validate_march_2025_failure_case(
            output_dir="data/reports"
        )

        # Assert report content
        with open(results["march_report_path"], 'r') as f:
            content = f.read()

        assert "# March 2025 Failure Case Analysis" in content
        assert "Original Failure" in content
        assert "Calibrated Model Results" in content
        assert "Ranging Market Analysis" in content
        assert "Conclusion" in content
        assert "-8.56%" in content or "8.56%" in content

    def test_detect_market_regimes(self):
        """Test market regime detection."""
        # Arrange
        runner = HistoricalValidationRunner(
            model=Mock(),
            calibration=Mock(),
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Create sample price data
        sample_prices = pd.Series([
            100.0, 101.0, 102.0, 103.0, 104.0,  # Trending up
            105.0, 104.5, 104.0, 103.5, 103.0,  # Trending down
            103.0, 103.5, 104.0, 104.5, 105.0,  # Ranging
        ])

        # Act
        regimes = runner.detect_market_regimes(
            prices=sample_prices,
            returns=pd.Series([0.01] * 15)
        )

        # Assert
        assert "regime_labels" in regimes
        assert len(regimes["regime_labels"]) == len(sample_prices)
        assert "trending_periods" in regimes
        assert "ranging_periods" in regimes

    def test_calculate_regime_specific_metrics(self):
        """Test calculating metrics per regime."""
        # Arrange
        model = Mock()

        def mock_predict_proba(X):
            n_samples = X.shape[0] if hasattr(X, 'shape') else 1
            return np.array([[0.4, 0.6]] * n_samples)

        model.predict_proba = mock_predict_proba

        calibration = Mock()

        def mock_calibrate(X):
            n_samples = X.shape[0] if hasattr(X, 'shape') else 1
            return np.array([0.55] * n_samples)

        calibration.predict_proba = mock_calibrate

        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Mock regime labels
        regime_labels = pd.Series(['trending'] * 10 + ['ranging'] * 10)

        # Act
        metrics = runner.calculate_regime_specific_metrics(
            labels=pd.Series([1] * 10 + [0] * 10),
            regime_labels=regime_labels
        )

        # Assert
        assert "trending" in metrics
        assert "ranging" in metrics
        assert "win_rate" in metrics["trending"]
        assert "win_rate" in metrics["ranging"]
        assert "brier_score" in metrics["trending"]
        assert "brier_score" in metrics["ranging"]

    def test_volatility_classification(self):
        """Test volatility classification (high vs low)."""
        # Arrange
        runner = HistoricalValidationRunner(
            model=Mock(),
            calibration=Mock(),
            data_path="data/processed/dollar_bars/1_minute"
        )

        # Create sample volatility data
        atr_values = pd.Series([0.5, 0.6, 0.4, 0.8, 1.2, 0.3, 0.4, 0.5])

        # Act
        volatility_regime = runner.classify_volatility(
            atr_values=atr_values,
            threshold_percentile=0.5
        )

        # Assert
        assert "regime" in volatility_regime
        assert len(volatility_regime["regime"]) == len(atr_values)
        assert "high_volatility_periods" in volatility_regime
        assert "low_volatility_periods" in volatility_regime

    def test_generate_final_validation_report(self):
        """Test generating comprehensive final validation report."""
        # Arrange
        model = Mock()
        calibration = Mock()
        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        overall_results = {
            "uncalibrated": {
                "win_rate": 0.60,
                "mean_predicted_probability": 0.75,
                "brier_score": 0.28,
                "trade_count": 100
            },
            "calibrated": {
                "win_rate": 0.62,
                "mean_predicted_probability": 0.61,
                "brier_score": 0.14,
                "trade_count": 98
            },
            "comparison": {
                "win_rate_improvement": 0.02,
                "brier_score_improvement": 0.14,
                "uncalibrated_probability_match": 0.15,
                "calibrated_probability_match": 0.01
            }
        }

        regime_results = {
            "trending": {
                "win_rate": 0.65,
                "brier_score": 0.12,
                "sample_count": 50
            },
            "ranging": {
                "win_rate": 0.58,
                "brier_score": 0.16,
                "sample_count": 48
            }
        }

        march_results = {
            "period": "March 2025",
            "original_loss_percent": -8.56,
            "uncalibrated_win_rate": 0.284,
            "calibrated_win_rate": 0.52,
            "uncalibrated_brier_score": 0.45,
            "calibrated_brier_score": 0.18,
            "loss_prevented": True,
            "improvement_percentage": 23.6
        }

        # Act
        report_path = runner.generate_final_validation_report(
            overall_results=overall_results,
            regime_results=regime_results,
            march_results=march_results,
            output_path="data/reports/test_final_validation_report.md"
        )

        # Assert
        assert Path(report_path).exists()
        assert report_path.endswith("test_final_validation_report.md")

        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()

        assert "# Final Historical Validation Report" in content
        assert "Executive Summary" in content
        assert "GO - PROCEED TO PHASE 2" in content
        assert "Overall Performance Metrics" in content
        assert "Success Criteria Validation" in content
        assert "Regime-Specific Analysis" in content
        assert "March 2025 Failure Case Analysis" in content
        assert "Go/No-Go Recommendation" in content
        assert "62.00%" in content  # Calibrated win rate
        assert "-8.56%" in content  # March loss
        assert "✅ PASS" in content  # Success criteria met

    def test_save_calibrated_model(self):
        """Test saving calibrated model and metadata."""
        # Arrange
        model = Mock()
        calibration = Mock()
        runner = HistoricalValidationRunner(
            model=model,
            calibration=calibration,
            data_path="data/processed/dollar_bars/1_minute"
        )

        validation_results = {
            "calibrated": {
                "win_rate": 0.62,
                "mean_predicted_probability": 0.61,
                "brier_score": 0.14,
                "trade_count": 98
            },
            "comparison": {
                "calibrated_probability_match": 0.01,
                "brier_score_improvement": 0.14
            }
        }

        # Create a mock source model file
        source_model_path = Path("models/xgboost/1_minute/xgboost_model.pkl")
        source_model_path.parent.mkdir(parents=True, exist_ok=True)
        source_model_path.touch()

        # Act
        result = runner.save_calibrated_model(
            validation_results=validation_results,
            model_output_path="data/models/xgboost/1_minute/model_calibrated.joblib",
            metadata_output_path="data/models/xgboost/1_minute/metadata_calibrated.json"
        )

        # Assert
        assert "model_path" in result
        assert "metadata_path" in result
        assert "deployment_ready" in result
        assert result["deployment_ready"] == True

        # Check model file exists
        assert Path(result["model_path"]).exists()

        # Check metadata file exists and has correct content
        import json
        with open(result["metadata_path"], 'r') as f:
            metadata = json.load(f)

        assert metadata["model_type"] == "XGBoostClassifier"
        assert metadata["calibration_method"] == "ProbabilityCalibration"
        assert metadata["deployment_ready"] == True
        assert metadata["performance_metrics"]["win_rate"] == 0.62
        assert metadata["performance_metrics"]["brier_score"] == 0.14
        assert metadata["success_criteria"]["brier_score_target_met"] == True
        assert metadata["success_criteria"]["probability_match_target_met"] == True

        # Cleanup
        Path(result["model_path"]).unlink(missing_ok=True)
        Path(result["metadata_path"]).unlink(missing_ok=True)
        source_model_path.unlink(missing_ok=True)

"""Unit tests for CalibrationValidator.

Tests the calibration validation functionality for historical MNQ data.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.ml.calibration_validator import CalibrationValidator
from src.ml.probability_calibration import ProbabilityCalibration


class TestCalibrationValidator:
    """Test CalibrationValidator class."""

    @pytest.fixture
    def validator(self):
        """Create CalibrationValidator instance."""
        return CalibrationValidator(
            data_path="data/processed/dollar_bars/1_minute"
        )

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample March 2025 data for testing."""
        # Create sample dollar bar data with proper structure
        np.random.seed(42)
        dates = pd.date_range("2025-03-01 00:00:00", periods=2000, freq="1min")

        # Create realistic price data (trending up then down)
        base_price = 21000
        trend = np.concatenate([
            np.linspace(0, 100, 1000),  # Trend up
            np.linspace(100, 0, 1000),  # Trend down
        ])
        noise = np.random.normal(0, 5, 2000)

        close_prices = base_price + trend + noise
        open_prices = close_prices + np.random.normal(0, 2, 2000)
        high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(5, 2, 2000))
        low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(5, 2, 2000))

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.randint(4000, 6000, 2000),
                "notional_value": close_prices * np.random.randint(4000, 6000, 2000),
            }
        )

        # Save to file
        data_dir = tmp_path / "1_minute"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "mnq_1min_2025.csv"
        df.to_csv(data_file, index=False)

        return tmp_path

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = CalibrationValidator(
            data_path="data/processed/dollar_bars/1_minute"
        )

        assert validator.data_path == Path("data/processed/dollar_bars/1_minute")
        assert validator.march_2025_data is None
        assert validator.features is None
        assert validator.labels is None

    @patch("src.ml.calibration_validator.pd.read_csv")
    def test_load_march_2025_data_file_not_found(self, mock_read_csv, validator):
        """Test loading data when file doesn't exist."""
        # Mock read_csv to raise FileNotFoundError
        mock_read_csv.side_effect = FileNotFoundError("No such file")

        with pytest.raises(FileNotFoundError):
            validator.load_march_2025_data()

    def test_load_march_2025_data_success(self, validator, sample_data):
        """Test successful data loading."""
        # Update validator path
        validator.data_path = sample_data / "1_minute"

        # Load data
        features, labels = validator.load_march_2025_data()

        # Verify
        assert features is not None
        assert labels is not None
        assert len(features) > 0
        assert len(labels) == len(features)
        assert validator.march_2025_data is not None

    def test_load_march_2025_data_stores_attributes(self, validator, sample_data):
        """Test that data loading stores features and labels."""
        validator.data_path = sample_data / "1_minute"

        features, labels = validator.load_march_2025_data()

        # Verify stored in validator
        assert validator.features is not None
        assert validator.labels is not None
        assert validator.march_2025_data is not None

    def test_train_calibration_without_data(self, validator):
        """Test training calibration before loading data raises error."""
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)

        with pytest.raises(ValueError, match="Must call load_march_2025_data"):
            validator.train_calibration(model)

    @patch.object(CalibrationValidator, "load_march_2025_data")
    def test_train_calibration_platt(self, mock_load, validator, sample_data):
        """Test training Platt calibration."""
        # Setup mock data
        validator.data_path = sample_data / "1_minute"
        validator.features = pd.DataFrame(np.random.rand(100, 5))
        validator.labels = pd.Series(np.random.randint(0, 2, 100))

        # Create model
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(validator.features.values, validator.labels.values)

        # Train calibration
        calibration = validator.train_calibration(
            model=model, method="platt", train_split=0.7
        )

        # Verify
        assert isinstance(calibration, ProbabilityCalibration)
        assert calibration.method == "platt"
        assert calibration.brier_score is not None

    @patch.object(CalibrationValidator, "load_march_2025_data")
    def test_train_calibration_isotonic(self, mock_load, validator, sample_data):
        """Test training isotonic calibration."""
        # Setup mock data
        validator.data_path = sample_data / "1_minute"
        validator.features = pd.DataFrame(np.random.rand(100, 5))
        validator.labels = pd.Series(np.random.randint(0, 2, 100))

        # Create model
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(validator.features.values, validator.labels.values)

        # Train calibration
        calibration = validator.train_calibration(
            model=model, method="isotonic", train_split=0.7
        )

        # Verify
        assert isinstance(calibration, ProbabilityCalibration)
        assert calibration.method == "isotonic"
        assert calibration.brier_score is not None

    def test_validate_calibration_quality_without_data(self, validator):
        """Test validation before loading data raises error."""
        calibration = Mock()

        with pytest.raises(ValueError, match="Must call load_march_2025_data"):
            validator.validate_calibration_quality(calibration)

    @patch.object(CalibrationValidator, "load_march_2025_data")
    def test_validate_calibration_quality_success(
        self, mock_load, validator, sample_data
    ):
        """Test successful calibration quality validation."""
        # Setup
        validator.data_path = sample_data / "1_minute"
        validator.features = pd.DataFrame(np.random.rand(100, 5))
        validator.labels = pd.Series(np.random.randint(0, 2, 100))

        # Create and train model
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(validator.features.values, validator.labels.values)

        # Train calibration
        calibration = validator.train_calibration(
            model=model, method="platt", train_split=0.7
        )

        # Validate quality
        metrics = validator.validate_calibration_quality(calibration)

        # Verify metrics
        assert "brier_score" in metrics
        assert "max_calibration_deviation" in metrics
        assert "mean_predicted_probability" in metrics
        assert "actual_win_rate" in metrics
        assert "probability_match" in metrics

        # Verify types
        assert isinstance(metrics["brier_score"], float)
        assert isinstance(metrics["max_calibration_deviation"], float)
        assert isinstance(metrics["mean_predicted_probability"], float)
        assert isinstance(metrics["actual_win_rate"], float)
        assert isinstance(metrics["probability_match"], float)

    def test_compare_uncalibrated_vs_calibrated_without_data(self, validator):
        """Test comparison before loading data raises error."""
        model = Mock()
        calibration = Mock()

        with pytest.raises(ValueError, match="Must call load_march_2025_data"):
            validator.compare_uncalibrated_vs_calibrated(model, calibration)

    @patch.object(CalibrationValidator, "load_march_2025_data")
    def test_compare_uncalibrated_vs_calibrated_success(
        self, mock_load, validator, sample_data
    ):
        """Test successful comparison of uncalibrated vs calibrated."""
        # Setup
        validator.data_path = sample_data / "1_minute"
        validator.features = pd.DataFrame(np.random.rand(100, 5))
        validator.labels = pd.Series(np.random.randint(0, 2, 100))

        # Create and train model
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(validator.features.values, validator.labels.values)

        # Train calibration
        calibration = validator.train_calibration(
            model=model, method="platt", train_split=0.7
        )

        # Compare
        comparison = validator.compare_uncalibrated_vs_calibrated(model, calibration)

        # Verify structure
        assert "uncalibrated" in comparison
        assert "calibrated_platt" in comparison

        # Verify uncalibrated metrics
        assert "mean_prob" in comparison["uncalibrated"]
        assert "actual_win_rate" in comparison["uncalibrated"]
        assert "brier_score" in comparison["uncalibrated"]

        # Verify calibrated metrics
        assert "mean_prob" in comparison["calibrated_platt"]
        assert "actual_win_rate" in comparison["calibrated_platt"]
        assert "brier_score" in comparison["calibrated_platt"]

    def test_generate_calibration_curve_without_data(self, validator):
        """Test calibration curve generation before loading data raises error."""
        model = Mock()
        calibration = Mock()

        with pytest.raises(ValueError, match="Must call load_march_2025_data"):
            validator.generate_calibration_curve(
                model, calibration, save_path="test.png"
            )

    @patch.object(CalibrationValidator, "load_march_2025_data")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_calibration_curve_success(
        self, mock_close, mock_savefig, mock_load, validator, sample_data, tmp_path
    ):
        """Test successful calibration curve generation."""
        # Setup
        validator.data_path = sample_data / "1_minute"
        validator.features = pd.DataFrame(np.random.rand(100, 5))
        validator.labels = pd.Series(np.random.randint(0, 2, 100))

        # Create and train model
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(validator.features.values, validator.labels.values)

        # Train calibration
        calibration = validator.train_calibration(
            model=model, method="platt", train_split=0.7
        )

        # Generate curve
        save_path = tmp_path / "calibration_curve.png"
        validator.generate_calibration_curve(
            model=model, calibration=calibration, save_path=str(save_path)
        )

        # Verify matplotlib was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_generate_validation_report_without_data(self, validator):
        """Test report generation before loading data raises error."""
        model = Mock()
        calibration = Mock()

        with pytest.raises(ValueError, match="Must call load_march_2025_data"):
            validator.generate_validation_report(model, calibration)

    @patch.object(CalibrationValidator, "load_march_2025_data")
    def test_generate_validation_report_success(
        self, mock_load, validator, sample_data, tmp_path
    ):
        """Test successful validation report generation."""
        # Setup
        validator.data_path = sample_data / "1_minute"
        validator.features = pd.DataFrame(np.random.rand(100, 5))
        validator.labels = pd.Series(np.random.randint(0, 2, 100))

        # Create and train model
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(validator.features.values, validator.labels.values)

        # Train calibration
        calibration = validator.train_calibration(
            model=model, method="platt", train_split=0.7
        )

        # Generate report
        report_path = tmp_path / "validation_report.json"
        report = validator.generate_validation_report(
            model=model, calibration=calibration, output_path=str(report_path)
        )

        # Verify report structure
        assert "validation_date" in report
        assert "period" in report
        assert "market_regime" in report
        assert "calibration_method" in report
        assert "metrics" in report
        assert "comparison" in report
        assert "success_criteria" in report
        assert "overconfidence_fix" in report

        # Verify file was created
        assert report_path.exists()

        # Verify JSON is valid
        with open(report_path, "r") as f:
            loaded_report = json.load(f)
        assert loaded_report == report

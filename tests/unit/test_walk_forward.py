"""Unit tests for walk-forward validator components.

Tests window calculation, data isolation validation, execution logic,
and results persistence.
"""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

from src.research.walk_forward_config import (
    WalkForwardConfig,
    WalkForwardStep,
    WalkForwardStepResult,
    WalkForwardSummary,
    WalkForwardResults,
)
from src.research.walk_forward_validator import (
    WindowManager,
    DataIsolationValidator,
    LeakageReport,
    WalkForwardExecutor,
    WalkForwardResultsStorage,
)


class TestWindowManager:
    """Test WindowManager class."""

    @pytest.fixture
    def config_2_years(self):
        """Create config for 2-year dataset."""
        return WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

    @pytest.fixture
    def window_manager(self, config_2_years):
        """Create WindowManager instance."""
        return WindowManager(config_2_years)

    def test_initialization(self, window_manager):
        """Test WindowManager initialization."""
        assert window_manager.config.training_window_months == 6
        assert window_manager.config.testing_window_months == 1

    def test_calculate_steps_2_years(self, window_manager):
        """Test step calculation for 2-year dataset."""
        steps = window_manager.calculate_steps()

        # Should have at least 12 steps
        assert len(steps) >= 12

        # First step should be Jan-Jun train, Jul test
        first_step = steps[0]
        assert first_step.step_number == 1
        assert first_step.train_start == date(2024, 1, 1)
        assert first_step.train_end == date(2024, 6, 30)
        assert first_step.test_start == date(2024, 7, 1)

    def test_calculate_steps_temporal_ordering(self, window_manager):
        """Test that all steps maintain temporal ordering."""
        steps = window_manager.calculate_steps()

        for step in steps:
            # Train must end before test starts
            assert step.train_end < step.test_start

    def test_validate_data_sufficiency_sufficient(self, window_manager):
        """Test data sufficiency validation with sufficient data."""
        is_sufficient = window_manager.validate_data_sufficiency()

        # 2 years should provide sufficient data
        assert is_sufficient is True

    def test_validate_data_sufficiency_insufficient(self):
        """Test data sufficiency validation with insufficient data."""
        config = WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2024, 12, 31),  # Only 12 months
        )

        manager = WindowManager(config)
        is_sufficient = manager.validate_data_sufficiency()

        assert is_sufficient is False

    def test_get_max_possible_steps(self, window_manager):
        """Test max possible steps calculation."""
        max_steps = window_manager.get_max_possible_steps()

        # Should be at least 12
        assert max_steps >= 12

    def test_add_months_positive(self):
        """Test adding positive months."""
        result = WindowManager._add_months(date(2024, 1, 15), 3)
        assert result == date(2024, 4, 15)

    def test_add_months_negative(self):
        """Test subtracting months."""
        result = WindowManager._add_months(date(2024, 4, 15), -3)
        assert result == date(2024, 1, 15)

    def test_add_months_year_boundary(self):
        """Test adding months across year boundary."""
        result = WindowManager._add_months(date(2024, 11, 15), 3)
        assert result == date(2025, 2, 15)

    def test_add_months_month_overflow(self):
        """Test adding months that overflows month."""
        result = WindowManager._add_months(date(2024, 1, 31), 1)
        # Jan 31 + 1 month should be Feb 29 (2024 is leap year) or Feb 28
        assert result.month == 2
        assert result.year == 2024


class TestDataIsolationValidator:
    """Test DataIsolationValidator class."""

    @pytest.fixture
    def validator(self):
        """Create DataIsolationValidator instance."""
        return DataIsolationValidator()

    @pytest.fixture
    def sample_step(self):
        """Create sample walk-forward step."""
        return WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
        )

    @pytest.fixture
    def clean_train_data(self):
        """Create clean training data."""
        dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")
        return pd.DataFrame({"close": 100.0}, index=dates)

    @pytest.fixture
    def clean_test_data(self):
        """Create clean testing data."""
        dates = pd.date_range(start="2024-07-01", end="2024-07-31", freq="D")
        return pd.DataFrame({"close": 100.0}, index=dates)

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None

    def test_validate_isolation_clean_data(
        self, validator, sample_step, clean_train_data, clean_test_data
    ):
        """Test isolation validation with clean data."""
        is_valid = validator.validate_isolation(
            sample_step, clean_train_data, clean_test_data
        )

        assert is_valid is True

    def test_validate_isolation_temporal_violation(
        self, validator, sample_step
    ):
        """Test isolation validation with temporal violation."""
        # Create train data that extends into test period
        dates = pd.date_range(start="2024-01-01", end="2024-07-15", freq="D")
        train_data = pd.DataFrame({"close": 100.0}, index=dates)

        dates = pd.date_range(start="2024-07-01", end="2024-07-31", freq="D")
        test_data = pd.DataFrame({"close": 100.0}, index=dates)

        is_valid = validator.validate_isolation(sample_step, train_data, test_data)

        assert is_valid is False

    def test_validate_isolation_overlap(
        self, validator, sample_step, clean_train_data
    ):
        """Test isolation validation with overlapping data."""
        # Create test data that overlaps with training
        dates = pd.date_range(start="2024-06-15", end="2024-07-31", freq="D")
        test_data = pd.DataFrame({"close": 100.0}, index=dates)

        is_valid = validator.validate_isolation(sample_step, clean_train_data, test_data)

        assert is_valid is False

    def test_check_temporal_ordering_valid(self, validator, clean_train_data, clean_test_data):
        """Check temporal ordering with valid data."""
        is_valid = validator.check_temporal_ordering(clean_train_data, clean_test_data)

        assert is_valid is True

    def test_check_temporal_ordering_invalid(self, validator):
        """Check temporal ordering with invalid data."""
        dates = pd.date_range(start="2024-01-01", end="2024-07-15", freq="D")
        train_data = pd.DataFrame({"close": 100.0}, index=dates)

        dates = pd.date_range(start="2024-07-01", end="2024-07-31", freq="D")
        test_data = pd.DataFrame({"close": 100.0}, index=dates)

        is_valid = validator.check_temporal_ordering(train_data, test_data)

        assert is_valid is False

    def test_detect_leakage_clean(self, validator, clean_train_data, clean_test_data):
        """Test leakage detection with clean data."""
        report = validator.detect_leakage(clean_train_data, clean_test_data)

        assert report.has_leakage is False
        assert report.severity == "none"
        assert len(report.leakage_details) == 0

    def test_detect_leakage_overlap(self, validator, clean_train_data):
        """Test leakage detection with overlapping data."""
        # Create overlapping test data
        dates = pd.date_range(start="2024-06-15", end="2024-07-31", freq="D")
        test_data = pd.DataFrame({"close": 100.0}, index=dates)

        report = validator.detect_leakage(clean_train_data, test_data)

        assert report.has_leakage is True
        assert report.severity == "major"
        assert len(report.leakage_details) > 0

    def test_detect_leakage_duplicates(self, validator):
        """Test leakage detection with duplicate timestamps."""
        # Create train data with duplicates
        dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
        train_data = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)

        dates = pd.date_range(start="2024-07-01", end="2024-07-03", freq="D")
        test_data = pd.DataFrame({"close": 100.0}, index=dates)

        report = validator.detect_leakage(train_data, test_data)

        assert report.has_leakage is True
        assert "duplicate" in report.leakage_details[0].lower()


class TestWalkForwardExecutor:
    """Test WalkForwardExecutor class."""

    @pytest.fixture
    def mock_backtester(self):
        """Create mock backtester."""
        backtester = Mock()
        backtester.config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        # Mock _load_dollar_bars to return sample data
        def mock_load_bars(start_date, end_date):
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            df = pd.DataFrame({
                "timestamp": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            })
            df.set_index("timestamp", inplace=True)
            return df

        backtester._load_dollar_bars = mock_load_bars

        # Mock run_backtest to return sample results
        from src.research.ensemble_backtester import BacktestResults

        mock_result = BacktestResults(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=100.0,
            average_loss=-75.0,
            largest_win=200.0,
            largest_loss=-100.0,
            max_drawdown=0.1,
            max_drawdown_duration=5,
            sharpe_ratio=1.2,
            average_hold_time=30.0,
            trade_frequency=0.5,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.5,
            total_pnl=300.0,
        )
        backtester.run_backtest = Mock(return_value=mock_result)

        return backtester

    @pytest.fixture
    def executor(self, mock_backtester):
        """Create WalkForwardExecutor instance."""
        return WalkForwardExecutor(mock_backtester)

    @pytest.fixture
    def sample_steps(self):
        """Create sample walk-forward steps."""
        return [
            WalkForwardStep(
                step_number=1,
                train_start=date(2024, 1, 1),
                train_end=date(2024, 1, 31),
                test_start=date(2024, 2, 1),
                test_end=date(2024, 2, 29),
            ),
            WalkForwardStep(
                step_number=2,
                train_start=date(2024, 2, 1),
                train_end=date(2024, 2, 29),
                test_start=date(2024, 3, 1),
                test_end=date(2024, 3, 31),
            ),
        ]

    def test_initialization(self, executor):
        """Test executor initialization."""
        assert executor.backtester is not None
        assert executor.validator is not None

    def test_execute_step(self, executor, sample_steps):
        """Test executing a single step."""
        step = sample_steps[0]
        parameters = {"confidence_threshold": 0.5}

        result = executor.execute_step(step, parameters)

        assert result.step.step_number == 1
        assert result.in_sample_metrics["win_rate"] == 0.6
        assert result.out_of_sample_metrics["win_rate"] == 0.6
        assert result.trade_counts["in_sample"] == 10
        assert result.trade_counts["out_of_sample"] == 10

    def test_execute_all_steps(self, executor, sample_steps):
        """Test executing all steps."""
        parameters = {"confidence_threshold": 0.5}

        results = executor.execute_all_steps(sample_steps, parameters)

        assert len(results.steps) == 2
        assert results.summary.total_steps == 2
        assert results.summary.average_win_rate > 0

    def test_extract_metrics(self, executor):
        """Test metric extraction."""
        from src.research.ensemble_backtester import BacktestResults

        mock_result = BacktestResults(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=100.0,
            average_loss=-75.0,
            largest_win=200.0,
            largest_loss=-100.0,
            max_drawdown=0.1,
            max_drawdown_duration=5,
            sharpe_ratio=1.2,
            average_hold_time=30.0,
            trade_frequency=0.5,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.5,
            total_pnl=300.0,
        )

        metrics = executor._extract_metrics(mock_result)

        assert metrics["win_rate"] == 0.6
        assert metrics["profit_factor"] == 1.5
        assert metrics["sharpe_ratio"] == 1.2

    def test_calculate_summary(self, executor, sample_steps):
        """Test summary calculation."""
        from src.research.walk_forward_config import WalkForwardStepResult

        # Create mock step results with varied data to avoid NaN correlation
        step_results = [
            WalkForwardStepResult(
                step=sample_steps[0],
                in_sample_metrics={"win_rate": 0.6},
                out_of_sample_metrics={"win_rate": 0.55, "profit_factor": 1.3},
                trade_counts={"out_of_sample": 10},
            ),
            WalkForwardStepResult(
                step=sample_steps[1],
                in_sample_metrics={"win_rate": 0.65},
                out_of_sample_metrics={"win_rate": 0.58, "profit_factor": 1.4},
                trade_counts={"out_of_sample": 12},
            ),
        ]

        summary = executor._calculate_summary(step_results)

        assert summary.total_steps == 2
        assert summary.average_win_rate == 0.565  # (0.55 + 0.58) / 2
        assert summary.average_profit_factor == 1.35  # (1.3 + 1.4) / 2


class TestWalkForwardResultsStorage:
    """Test WalkForwardResultsStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create WalkForwardResultsStorage instance."""
        return WalkForwardResultsStorage(base_dir=temp_dir)

    @pytest.fixture
    def sample_results(self):
        """Create sample WalkForwardResults."""
        config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        step1 = WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
            train_bars_count=1000,
            test_bars_count=200,
        )

        step2 = WalkForwardStep(
            step_number=2,
            train_start=date(2024, 2, 1),
            train_end=date(2024, 7, 31),
            test_start=date(2024, 8, 1),
            test_end=date(2024, 8, 31),
            train_bars_count=1100,
            test_bars_count=220,
        )

        step_results = [
            WalkForwardStepResult(
                step=step1,
                in_sample_metrics={"win_rate": 0.6, "profit_factor": 1.5},
                out_of_sample_metrics={"win_rate": 0.55, "profit_factor": 1.3},
                parameters={"confidence_threshold": 0.5},
                trade_counts={"in_sample": 100, "out_of_sample": 20},
            ),
            WalkForwardStepResult(
                step=step2,
                in_sample_metrics={"win_rate": 0.65, "profit_factor": 1.6},
                out_of_sample_metrics={"win_rate": 0.58, "profit_factor": 1.4},
                parameters={"confidence_threshold": 0.5},
                trade_counts={"in_sample": 110, "out_of_sample": 22},
            ),
        ]

        summary = WalkForwardSummary(
            total_steps=2,
            average_win_rate=0.565,
            std_win_rate=0.015,
            average_profit_factor=1.35,
            best_step=2,
            worst_step=1,
            in_sample_out_of_sample_correlation=1.0,
            total_trades=42,
        )

        return WalkForwardResults(
            config=config,
            steps=step_results,
            summary=summary,
            timestamp="2024-01-01",
        )

    def test_initialization(self, storage, temp_dir):
        """Test storage initialization."""
        assert storage.base_dir == temp_dir
        assert temp_dir.exists()

    def test_save_results(self, storage, temp_dir, sample_results):
        """Test saving results to HDF5."""
        path = storage.save_results(sample_results)

        assert path.exists()
        assert path.name.startswith("walkforward_")
        assert path.suffix == ".h5"

    def test_load_results(self, storage, sample_results):
        """Test loading results from HDF5."""
        # Save results
        saved_path = storage.save_results(sample_results)

        # Load results
        loaded_results = storage.load_results(saved_path)

        # Verify config
        assert loaded_results.config.data_start_date == sample_results.config.data_start_date
        assert loaded_results.config.data_end_date == sample_results.config.data_end_date

        # Verify steps
        assert len(loaded_results.steps) == 2
        assert loaded_results.steps[0].step.step_number == 1
        assert loaded_results.steps[1].step.step_number == 2

        # Verify metrics
        assert loaded_results.steps[0].in_sample_metrics["win_rate"] == 0.6
        assert loaded_results.steps[0].out_of_sample_metrics["win_rate"] == 0.55

        # Verify summary
        assert loaded_results.summary.total_steps == 2
        assert loaded_results.summary.average_win_rate == 0.565

    def test_append_step_result(self, storage, sample_results):
        """Test appending step results to existing file."""
        # Save initial results with first step only
        initial_results = WalkForwardResults(
            config=sample_results.config,
            steps=[sample_results.steps[0]],
            summary=None,
        )
        path = storage.save_results(initial_results)

        # Append second step
        storage.append_step_result(sample_results.steps[1], path)

        # Load and verify both steps exist
        loaded_results = storage.load_results(path)
        assert len(loaded_results.steps) == 2
        assert loaded_results.steps[0].step.step_number == 1
        assert loaded_results.steps[1].step.step_number == 2

    def test_append_duplicate_step_skips(self, storage, sample_results):
        """Test that appending duplicate step is skipped."""
        # Save initial results
        path = storage.save_results(sample_results)

        # Try to append same step again
        storage.append_step_result(sample_results.steps[0], path)

        # Verify no duplicate
        loaded_results = storage.load_results(path)
        assert len(loaded_results.steps) == 2

    def test_calculate_data_checksum(self, storage):
        """Test data checksum calculation."""
        data = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        })

        checksum1 = storage.calculate_data_checksum(data)
        checksum2 = storage.calculate_data_checksum(data)

        # Same data should produce same checksum
        assert checksum1 == checksum2

        # Different data should produce different checksum
        data_different = pd.DataFrame({
            "close": [100.0, 101.0, 103.0],  # Changed last value
            "volume": [1000, 1100, 1200],
        })
        checksum3 = storage.calculate_data_checksum(data_different)

        assert checksum1 != checksum3

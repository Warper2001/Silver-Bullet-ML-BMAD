"""Integration tests for walk-forward validation framework.

Tests the complete walk-forward workflow end-to-end.
"""

import pytest
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
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
    WalkForwardExecutor,
    WalkForwardResultsStorage,
)


@pytest.fixture
def sample_dollar_bars():
    """Create sample dollar bar data for testing."""
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="h")
    np = pytest.importorskip("numpy")
    data = pd.DataFrame({
        "timestamp": dates,
        "open": 11700 + np.random.randn(len(dates)) * 50,
        "high": 11750 + np.random.randn(len(dates)) * 50,
        "low": 11650 + np.random.randn(len(dates)) * 50,
        "close": 11700 + np.random.randn(len(dates)) * 50,
        "volume": 1000 + np.random.randint(0, 500, len(dates)),
    })
    return data


class TestWalkForwardEndToEnd:
    """Test complete walk-forward workflow end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_complete_walk_forward_workflow(self, sample_dollar_bars, temp_dir):
        """Test complete walk-forward validation from config to results persistence."""
        # Step 1: Create configuration
        config = WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        # Step 2: Create window manager and calculate steps
        window_manager = WindowManager(config)
        steps = window_manager.calculate_steps()

        # Should have multiple steps
        assert len(steps) >= 12

        # Step 3: Validate data sufficiency
        is_sufficient = window_manager.validate_data_sufficiency()
        assert is_sufficient is True

        # Step 4: Create mock backtester
        mock_backtester = Mock()
        mock_backtester.config = config

        def mock_load_bars(start_date, end_date):
            # Filter data by date range
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + timedelta(days=1)
            filtered = sample_dollar_bars[
                (sample_dollar_bars["timestamp"] >= start_dt) &
                (sample_dollar_bars["timestamp"] < end_dt)
            ].copy()
            filtered.set_index("timestamp", inplace=True)
            return filtered

        mock_backtester._load_dollar_bars = mock_load_bars

        # Mock backtest results
        from src.research.ensemble_backtester import BacktestResults

        def mock_run_backtest(start_date, end_date, confidence_threshold=0.5):
            # Generate realistic-looking results
            np = pytest.importorskip("numpy")
            total_trades = np.random.randint(10, 50)
            winning_trades = int(total_trades * 0.55)
            losing_trades = total_trades - winning_trades

            return BacktestResults(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=winning_trades / total_trades,
                profit_factor=1.0 + np.random.random() * 0.8,
                average_win=100.0 + np.random.random() * 50,
                average_loss=-(75.0 + np.random.random() * 25),
                largest_win=200.0 + np.random.random() * 100,
                largest_loss=-(100.0 + np.random.random() * 50),
                max_drawdown=0.05 + np.random.random() * 0.15,
                max_drawdown_duration=np.random.randint(3, 15),
                sharpe_ratio=0.5 + np.random.random() * 1.5,
                average_hold_time=20.0 + np.random.random() * 20,
                trade_frequency=total_trades / max(1, (end_date - start_date).days),
                start_date=start_date,
                end_date=end_date,
                confidence_threshold=confidence_threshold,
                total_pnl=np.random.randn() * 1000,
            )

        mock_backtester.run_backtest = mock_run_backtest

        # Step 5: Execute walk-forward (limit to 3 steps for integration test)
        executor = WalkForwardExecutor(mock_backtester)
        test_steps = steps[:3]  # Only test first 3 steps
        parameters = {"confidence_threshold": 0.50}

        results = executor.execute_all_steps(test_steps, parameters)

        # Validate results structure
        assert results.config is not None
        assert len(results.steps) == 3
        assert results.summary is not None
        assert results.summary.total_steps == 3

        # Step 6: Persist results
        storage = WalkForwardResultsStorage(base_dir=temp_dir)
        saved_path = storage.save_results(results)

        # Verify file was created
        assert saved_path.exists()

        # Step 7: Load results and verify integrity
        loaded_results = storage.load_results(saved_path)

        # Verify all data preserved
        assert loaded_results.config.training_window_months == config.training_window_months
        assert len(loaded_results.steps) == 3
        assert loaded_results.summary.total_steps == 3

        # Verify step data preserved
        for i, (original, loaded) in enumerate(zip(results.steps, loaded_results.steps)):
            assert original.step.step_number == loaded.step.step_number
            assert original.step.train_start == loaded.step.train_start
            assert original.in_sample_metrics["win_rate"] == loaded.in_sample_metrics["win_rate"]
            assert original.out_of_sample_metrics["win_rate"] == loaded.out_of_sample_metrics["win_rate"]


class TestWindowManagerIntegration:
    """Integration tests for WindowManager."""

    def test_window_manager_with_real_config(self):
        """Test WindowManager with actual config values from config-sim.yaml."""
        config = WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2026, 3, 31),
        )

        manager = WindowManager(config)
        steps = manager.calculate_steps()

        # Verify steps calculated correctly
        assert len(steps) > 0

        # All steps should maintain temporal ordering
        for step in steps:
            assert step.train_end < step.test_start

        # Steps should be sequential
        for i in range(1, len(steps)):
            assert steps[i].step_number == steps[i - 1].step_number + 1


class TestDataIsolationValidatorIntegration:
    """Integration tests for DataIsolationValidator."""

    @pytest.fixture
    def validator(self):
        """Create DataIsolationValidator instance."""
        return DataIsolationValidator()

    @pytest.fixture
    def real_step(self):
        """Create realistic walk-forward step."""
        return WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
        )

    def test_validate_with_realistic_data(self, validator, real_step):
        """Test isolation validation with realistic market data."""
        # Create training data
        train_dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="h")
        train_data = pd.DataFrame({
            "close": 11700.0,
        }, index=train_dates)

        # Create testing data
        test_dates = pd.date_range(start="2024-07-01", end="2024-07-31", freq="h")
        test_data = pd.DataFrame({
            "close": 11800.0,
        }, index=test_dates)

        # Validate isolation
        is_valid = validator.validate_isolation(real_step, train_data, test_data)

        # Should be valid
        assert is_valid is True


class TestCheckpointResume:
    """Test checkpoint/resume functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_checkpoint_and_resume(self, temp_dir):
        """Test saving intermediate results and resuming."""
        config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2024, 12, 31),
        )

        storage = WalkForwardResultsStorage(base_dir=temp_dir)

        # Create first step result
        step1 = WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
        )

        result1 = WalkForwardStepResult(
            step=step1,
            in_sample_metrics={"win_rate": 0.6},
            out_of_sample_metrics={"win_rate": 0.55},
        )

        # Save initial results with first step
        initial_results = WalkForwardResults(
            config=config,
            steps=[result1],
            summary=None,
        )
        path = storage.save_results(initial_results)

        # Append second step
        step2 = WalkForwardStep(
            step_number=2,
            train_start=date(2024, 2, 1),
            train_end=date(2024, 7, 31),
            test_start=date(2024, 8, 1),
            test_end=date(2024, 8, 31),
        )

        result2 = WalkForwardStepResult(
            step=step2,
            in_sample_metrics={"win_rate": 0.65},
            out_of_sample_metrics={"win_rate": 0.58},
        )

        storage.append_step_result(result2, path)

        # Load and verify both steps present
        loaded = storage.load_results(path)
        assert len(loaded.steps) == 2
        assert loaded.steps[0].step.step_number == 1
        assert loaded.steps[1].step.step_number == 2


class TestReproducibility:
    """Test that walk-forward results are reproducible."""

    def test_reproducible_window_calculation(self):
        """Test that window calculations are deterministic."""
        config = WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        # Calculate steps twice
        manager1 = WindowManager(config)
        steps1 = manager1.calculate_steps()

        manager2 = WindowManager(config)
        steps2 = manager2.calculate_steps()

        # Should be identical
        assert len(steps1) == len(steps2)

        for s1, s2 in zip(steps1, steps2):
            assert s1.step_number == s2.step_number
            assert s1.train_start == s2.train_start
            assert s1.train_end == s2.train_end
            assert s1.test_start == s2.test_start
            assert s1.test_end == s2.test_end

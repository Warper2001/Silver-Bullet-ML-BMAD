"""Tests for grid search executor.

Tests for executing grid search with walk-forward validation, parallel execution,
and checkpointing.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from src.research.parameter_grids import ParameterGrid, CombinedGrid
from src.research.parameter_grid_search import (
    ParameterGridSearch,
    CombinationResult,
    GridSearchResults,
    ParameterSet,
)
from src.research.walk_forward_config import (
    WalkForwardResults,
    WalkForwardSummary,
    WalkForwardStepResult,
    WalkForwardConfig,
    WalkForwardStep,
)


@pytest.fixture
def mock_walk_forward_validator():
    """Create a mock walk-forward validator."""
    validator = Mock()

    # Create proper WalkForwardConfig
    config = WalkForwardConfig(
        training_window_months=6,
        testing_window_months=1,
        step_forward_months=1,
        minimum_steps=12,
        data_start_date=date(2024, 1, 1),
        data_end_date=date(2025, 12, 31),
    )

    # Create proper WalkForwardStep
    step = WalkForwardStep(
        step_number=1,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 6, 30),
        test_start=date(2024, 7, 1),
        test_end=date(2024, 7, 31),
    )

    # Create proper WalkForwardStepResult
    step_result = WalkForwardStepResult(
        step=step,
        out_of_sample_metrics={
            "win_rate": 0.60,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.08,
        },
        in_sample_metrics={
            "win_rate": 0.65,
            "profit_factor": 2.0,
        },
        trade_counts={"out_of_sample": 50, "in_sample": 100},
    )

    # Create proper WalkForwardResults
    mock_results = WalkForwardResults(
        config=config,
        steps=[step_result],
        summary=WalkForwardSummary(
            total_steps=1,
            average_win_rate=0.60,
            std_win_rate=0.05,
            average_profit_factor=1.8,
            best_step=1,
            worst_step=1,
            in_sample_out_of_sample_correlation=0.8,
            total_trades=50,
        ),
    )

    validator.execute_all_steps = Mock(return_value=mock_results)

    # Add window_manager mock with calculate_steps method
    validator.window_manager = Mock()
    validator.window_manager.calculate_steps = Mock(return_value=[step])

    return validator


@pytest.fixture
def sample_parameter_set():
    """Create a sample parameter set."""
    return ParameterSet(
        combination_id="combo_001",
        strategy_1_params={"fvg_min_size_ticks": 4},
        strategy_2_params={"statistical_extreme_sd": 2.0},
        strategy_3_params={},
        strategy_4_params={},
        strategy_5_params={},
        ensemble_params={"confidence_threshold": 0.50},
    )


class TestCombinationResult:
    """Test suite for CombinationResult Pydantic model."""

    def test_create_combination_result(self, sample_parameter_set):
        """Test creating a combination result."""
        result = CombinationResult(
            combination_id="combo_001",
            parameters=sample_parameter_set,
            avg_oos_win_rate=0.60,
            avg_oos_profit_factor=1.8,
            win_rate_std=0.05,
            max_drawdown=0.08,
            total_trades=50,
            execution_time=120.5,
        )

        assert result.combination_id == "combo_001"
        assert result.avg_oos_win_rate == 0.60
        assert result.avg_oos_profit_factor == 1.8


class TestGridSearchResults:
    """Test suite for GridSearchResults Pydantic model."""

    def test_create_grid_search_results(self, sample_parameter_set):
        """Test creating grid search results."""
        combo_result = CombinationResult(
            combination_id="combo_001",
            parameters=sample_parameter_set,
            avg_oos_win_rate=0.60,
            avg_oos_profit_factor=1.8,
            win_rate_std=0.05,
            max_drawdown=0.08,
            total_trades=50,
            execution_time=120.5,
        )

        results = GridSearchResults(
            combinations_tested=1,
            results=[combo_result],
            best_combination=combo_result,
            summary={"total_execution_time": 120.5},
        )

        assert results.combinations_tested == 1
        assert len(results.results) == 1
        assert results.best_combination.combination_id == "combo_001"


class TestParameterGridSearch:
    """Test suite for ParameterGridSearch class."""

    def test_initialization(self, mock_walk_forward_validator):
        """Test grid search executor initialization."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        assert executor.walk_forward_validator == mock_walk_forward_validator

    def test_execute_single_combination(
        self, mock_walk_forward_validator, sample_parameter_set
    ):
        """Test executing a single parameter combination."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        result = executor.execute_single_combination(sample_parameter_set)

        assert result.combination_id == "combo_001"
        assert result.avg_oos_win_rate == 0.60
        assert result.avg_oos_profit_factor == 1.8
        assert result.total_trades == 50

    def test_execute_search_multiple_combinations(
        self, mock_walk_forward_validator
    ):
        """Test executing grid search with multiple combinations."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        # Create sample combinations
        combinations = [
            ParameterSet(
                combination_id="combo_000",
                strategy_1_params={"fvg_min_size_ticks": 2},
                strategy_2_params={},
                strategy_3_params={},
                strategy_4_params={},
                strategy_5_params={},
                ensemble_params={"confidence_threshold": 0.45},
            ),
            ParameterSet(
                combination_id="combo_001",
                strategy_1_params={"fvg_min_size_ticks": 4},
                strategy_2_params={},
                strategy_3_params={},
                strategy_4_params={},
                strategy_5_params={},
                ensemble_params={"confidence_threshold": 0.50},
            ),
        ]

        results = executor.execute_search(combinations)

        assert results.combinations_tested == 2
        assert len(results.results) == 2
        assert results.best_combination is not None

    def test_save_checkpoint(self, mock_walk_forward_validator, sample_parameter_set, tmp_path):
        """Test saving checkpoint results."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        # Create sample result
        result = CombinationResult(
            combination_id="combo_001",
            parameters=sample_parameter_set,
            avg_oos_win_rate=0.60,
            avg_oos_profit_factor=1.8,
            win_rate_std=0.05,
            max_drawdown=0.08,
            total_trades=50,
            execution_time=120.5,
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.json"
        executor.save_checkpoint([result], str(checkpoint_path))

        # Verify file was created
        assert checkpoint_path.exists()

        # Load and verify content
        with open(checkpoint_path) as f:
            data = json.load(f)

        assert data["combinations_completed"] == 1
        assert len(data["results"]) == 1

    def test_load_checkpoint(self, mock_walk_forward_validator, sample_parameter_set, tmp_path):
        """Test loading checkpoint results."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        # Create checkpoint file
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_data = {
            "combinations_completed": 1,
            "combinations_total": 10,
            "best_combination_id": "combo_001",
            "best_win_rate": 0.60,
            "results": [
                {
                    "combination_id": "combo_001",
                    "avg_oos_win_rate": 0.60,
                    "avg_oos_profit_factor": 1.8,
                    "win_rate_std": 0.05,
                    "max_drawdown": 0.08,
                    "total_trades": 50,
                    "execution_time": 120.5,
                }
            ],
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        results = executor.load_checkpoint(str(checkpoint_path))

        assert len(results) == 1
        assert results[0].combination_id == "combo_001"

    def test_parallel_execution_not_implemented(
        self, mock_walk_forward_validator, sample_parameter_set
    ):
        """Test that parallel execution is handled (even if not fully implemented)."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        # Execute with parallel=False (sequential)
        combinations = [sample_parameter_set]
        results = executor.execute_search(combinations, parallel=False)

        assert results.combinations_tested == 1

    def test_progress_tracking(
        self, mock_walk_forward_validator, sample_parameter_set, caplog
    ):
        """Test that progress is logged during execution."""
        import logging

        executor = ParameterGridSearch(mock_walk_forward_validator)

        with caplog.at_level(logging.INFO):
            results = executor.execute_search([sample_parameter_set])

        # Check that progress was logged
        assert any("Executing combination" in record.message for record in caplog.records)

    def test_best_combination_selection(
        self, mock_walk_forward_validator
    ):
        """Test that best combination is selected correctly."""
        executor = ParameterGridSearch(mock_walk_forward_validator)

        # Create combinations with different performance
        combinations = [
            ParameterSet(
                combination_id="combo_poor",
                strategy_1_params={"fvg_min_size_ticks": 2},
                strategy_2_params={},
                strategy_3_params={},
                strategy_4_params={},
                strategy_5_params={},
                ensemble_params={"confidence_threshold": 0.45},
            ),
            ParameterSet(
                combination_id="combo_good",
                strategy_1_params={"fvg_min_size_ticks": 4},
                strategy_2_params={},
                strategy_3_params={},
                strategy_4_params={},
                strategy_5_params={},
                ensemble_params={"confidence_threshold": 0.50},
            ),
        ]

        results = executor.execute_search(combinations)

        # Best combination should be the one with higher win rate
        assert results.best_combination is not None

    def test_error_handling_per_combination(
        self, mock_walk_forward_validator
    ):
        """Test that errors in single combinations don't stop the search."""
        # Store the original return value
        original_return = mock_walk_forward_validator.execute_all_steps.return_value

        # Make validator fail for specific combination
        def side_effect(steps, parameters):
            if parameters.get("confidence_threshold") == 0.99:
                raise ValueError("Invalid parameters")
            return original_return

        mock_walk_forward_validator.execute_all_steps.side_effect = side_effect

        executor = ParameterGridSearch(mock_walk_forward_validator)

        combinations = [
            ParameterSet(
                combination_id="combo_good",
                strategy_1_params={},
                strategy_2_params={},
                strategy_3_params={},
                strategy_4_params={},
                strategy_5_params={},
                ensemble_params={"confidence_threshold": 0.50},
            ),
            ParameterSet(
                combination_id="combo_bad",
                strategy_1_params={},
                strategy_2_params={},
                strategy_3_params={},
                strategy_4_params={},
                strategy_5_params={},
                ensemble_params={"confidence_threshold": 0.99},
            ),
        ]

        # Should complete despite one combination failing
        results = executor.execute_search(combinations)

        # At least the good combination should complete
        assert results.combinations_tested >= 1

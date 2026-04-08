"""Integration tests for Walk-Forward Execution orchestrator.

Tests the complete walk-forward execution workflow including:
- Parallel execution across parameter combinations
- Checkpoint save/resume functionality
- Result aggregation
- Progress tracking
"""

import json
import logging
from datetime import date
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
import h5py
import numpy as np
import pandas as pd

from src.research.walk_forward_executor import (
    WalkForwardExecutorOrchestrator,
    ParameterStabilityTracker,
    CheckpointManager,
    ResultAggregator,
    WalkForwardExecutionResults,
)
from src.research.parameter_grid_search import ParameterSet
from src.research.walk_forward_config import (
    WalkForwardConfig,
    WalkForwardResults,
    WalkForwardSummary,
)


logger = logging.getLogger(__name__)


@pytest.fixture
def sample_config(tmp_path):
    """Create sample walk-forward configuration."""
    return WalkForwardConfig(
        training_window_months=6,
        testing_window_months=1,
        step_forward_months=1,
        minimum_steps=12,
        data_start_date=date(2024, 1, 1),
        data_end_date=date(2025, 12, 31),
    )


@pytest.fixture
def sample_parameter_sets():
    """Create sample parameter combinations for testing."""
    return [
        ParameterSet(
            combination_id="combo_001",
            strategy_1_params={"confidence_threshold": 0.50},
            strategy_2_params={"liquidity_sweep_threshold": 0.30},
            strategy_3_params={"ema_period": 20},
            strategy_4_params={"vwap_period": 30},
            strategy_5_params={"lookback_bars": 10},
            ensemble_params={"confidence_threshold": 0.65},
        ),
        ParameterSet(
            combination_id="combo_002",
            strategy_1_params={"confidence_threshold": 0.55},
            strategy_2_params={"liquidity_sweep_threshold": 0.35},
            strategy_3_params={"ema_period": 25},
            strategy_4_params={"vwap_period": 35},
            strategy_5_params={"lookback_bars": 15},
            ensemble_params={"confidence_threshold": 0.70},
        ),
    ]


@pytest.fixture
def mock_walk_forward_results():
    """Create mock walk-forward results."""
    config = WalkForwardConfig(
        training_window_months=6,
        testing_window_months=1,
        step_forward_months=1,
        minimum_steps=12,
        data_start_date=date(2024, 1, 1),
        data_end_date=date(2025, 12, 31),
    )

    # Create mock step results
    from src.research.walk_forward_config import WalkForwardStep, WalkForwardStepResult

    steps = []
    for i in range(1, 13):  # 12 steps
        step = WalkForwardStep(
            step_number=i,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
            train_bars_count=1000,
            test_bars_count=150,
        )

        step_result = WalkForwardStepResult(
            step=step,
            in_sample_metrics={
                "win_rate": 0.60 + (i * 0.01),
                "profit_factor": 1.5 + (i * 0.1),
            },
            out_of_sample_metrics={
                "win_rate": 0.55 + (i * 0.01),
                "profit_factor": 1.4 + (i * 0.1),
            },
            parameters={"confidence_threshold": 0.65},
            trade_counts={"in_sample": 50, "out_of_sample": 10},
        )
        steps.append(step_result)

    summary = WalkForwardSummary(
        total_steps=12,
        average_win_rate=0.60,
        std_win_rate=0.05,
        average_profit_factor=1.8,
        best_step=12,
        worst_step=1,
        in_sample_out_of_sample_correlation=0.75,
        total_trades=120,
    )

    return WalkForwardResults(
        config=config,
        steps=steps,
        summary=summary,
        timestamp="2026-04-01",
    )


class TestParameterStabilityTracker:
    """Tests for ParameterStabilityTracker."""

    def test_calculate_weight_variance(self, sample_config):
        """Test weight variance calculation across walk-forward steps."""
        tracker = ParameterStabilityTracker()

        # Mock weight evolution across steps
        weight_history = {
            "triple_confluence_scaler": [0.20, 0.22, 0.21, 0.20, 0.19, 0.20],
            "wolf_pack_3_edge": [0.20, 0.18, 0.19, 0.20, 0.21, 0.20],
            "adaptive_ema_momentum": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
            "vwap_bounce": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
            "opening_range_breakout": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        }

        variance = tracker.calculate_weight_variance(weight_history)

        # Check that variance is calculated
        assert isinstance(variance, float)
        assert variance >= 0.0

        # Triple confluence and wolf pack should have non-zero variance
        assert variance > 0.0

    def test_calculate_stability_score(self):
        """Test stability score calculation."""
        tracker = ParameterStabilityTracker()

        # Low variance = high stability
        low_variance = 0.001
        high_variance = 0.1

        low_score = tracker.calculate_stability_score(low_variance)
        high_score = tracker.calculate_stability_score(high_variance)

        # Lower variance should produce higher stability score
        assert low_score > high_score
        assert 0.0 <= low_score <= 1.0
        assert 0.0 <= high_score <= 1.0

    def test_track_parameter_adaptation(self):
        """Test tracking parameter adaptation rates."""
        tracker = ParameterStabilityTracker()

        # Simulate parameter changes across rebalancing
        parameters_list = [
            {"confidence_threshold": 0.65},
            {"confidence_threshold": 0.66},
            {"confidence_threshold": 0.65},
            {"confidence_threshold": 0.67},
            {"confidence_threshold": 0.66},
        ]

        adaptation_rate = tracker.track_parameter_adaptation(parameters_list)

        # Should calculate how often parameters change
        assert isinstance(adaptation_rate, float)
        assert 0.0 <= adaptation_rate <= 1.0


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_checkpoint(self, tmp_path, sample_parameter_sets):
        """Test saving checkpoint after N combinations."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=2,
        )

        # Mock results
        results = [
            {
                "combination_id": "combo_001",
                "avg_oos_win_rate": 0.60,
                "avg_oos_profit_factor": 1.8,
                "win_rate_std": 0.05,
                "max_drawdown": 0.10,
            }
        ]

        checkpoint_path = manager.save_checkpoint(
            checkpoint_id=1,
            results=results,
            best_combination_id="combo_001",
        )

        # Verify checkpoint file exists
        assert checkpoint_path.exists()

        # Verify checkpoint content
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        assert checkpoint_data["checkpoint_id"] == 1
        assert checkpoint_data["combinations_completed"] == ["combo_001"]
        assert checkpoint_data["best_combination"] == "combo_001"

    def test_load_checkpoint(self, tmp_path):
        """Test loading existing checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=2,
        )

        # Create a checkpoint file
        checkpoint_data = {
            "checkpoint_id": 1,
            "timestamp": "2026-04-01T12:00:00Z",
            "combinations_completed": ["combo_001", "combo_002"],
            "best_combination": "combo_001",
            "best_metrics": {"avg_oos_win_rate": 0.60},
            "remaining": ["combo_003", "combo_004"],
        }

        checkpoint_path = tmp_path / "walkforward_checkpoint_1.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_path)

        assert loaded["checkpoint_id"] == 1
        assert loaded["combinations_completed"] == ["combo_001", "combo_002"]
        assert loaded["best_combination"] == "combo_001"

    def test_detect_existing_checkpoint(self, tmp_path):
        """Test detecting existing checkpoint on restart."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=2,
        )

        # Create a checkpoint file
        checkpoint_path = tmp_path / "walkforward_checkpoint_5.json"
        with open(checkpoint_path, "w") as f:
            json.dump({"checkpoint_id": 5}, f)

        # Detect checkpoint
        detected = manager.detect_existing_checkpoint()

        assert detected is not None
        assert detected["checkpoint_id"] == 5

    def test_skip_completed_combinations(self, tmp_path):
        """Test skipping combinations that were already completed."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=2,
        )

        # Create checkpoint with completed combinations
        checkpoint_data = {
            "checkpoint_id": 1,
            "combinations_completed": ["combo_001", "combo_002"],
            "best_combination": "combo_001",
        }

        checkpoint_path = tmp_path / "walkforward_checkpoint_1.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        all_combinations = ["combo_001", "combo_002", "combo_003", "combo_004"]

        # Should skip completed combinations
        remaining = manager.get_remaining_combinations(all_combinations, checkpoint_path)

        assert remaining == ["combo_003", "combo_004"]


class TestResultAggregator:
    """Tests for ResultAggregator."""

    def test_aggregate_metrics_across_steps(self, mock_walk_forward_results):
        """Test aggregating metrics across all walk-forward steps."""
        aggregator = ResultAggregator()

        aggregated = aggregator.aggregate_results(mock_walk_forward_results)

        # Check key metrics
        assert "avg_oos_win_rate" in aggregated
        assert "avg_oos_profit_factor" in aggregated
        assert "oos_consistency" in aggregated
        assert "max_drawdown" in aggregated
        assert "performance_stability" in aggregated
        assert "trade_frequency" in aggregated

        # Verify values
        assert aggregated["avg_oos_win_rate"] == pytest.approx(0.60, rel=0.1)
        assert aggregated["avg_oos_profit_factor"] == pytest.approx(1.8, rel=0.1)

    def test_calculate_stability_scores(self):
        """Test calculating stability scores."""
        aggregator = ResultAggregator()

        # Mock step results with VARYING stability levels (not constant)
        # Use slight variations to avoid constant array NaN issue
        in_sample_metrics = [
            {"win_rate": 0.60 + (i * 0.01), "profit_factor": 1.8 + (i * 0.1)}
            for i in range(12)
        ]
        out_of_sample_metrics = [
            {"win_rate": 0.58 + (i * 0.01), "profit_factor": 1.7 + (i * 0.1)}
            for i in range(12)
        ]

        stability = aggregator.calculate_stability_scores(
            in_sample_metrics, out_of_sample_metrics
        )

        assert "win_rate_stability" in stability
        assert "profit_factor_stability" in stability
        # Stability scores should be valid (not NaN)
        assert stability["win_rate_stability"] == stability["win_rate_stability"]  # Check not NaN
        assert stability["profit_factor_stability"] == stability["profit_factor_stability"]  # Check not NaN
        assert 0.0 <= stability["win_rate_stability"] <= 1.0
        assert 0.0 <= stability["profit_factor_stability"] <= 1.0

    def test_rank_combinations_by_performance(self):
        """Test ranking combinations by out-of-sample performance."""
        aggregator = ResultAggregator()

        results = [
            {
                "combination_id": "combo_001",
                "avg_oos_win_rate": 0.60,
                "oos_consistency": 0.05,
            },
            {
                "combination_id": "combo_002",
                "avg_oos_win_rate": 0.65,
                "oos_consistency": 0.03,
            },
            {
                "combination_id": "combo_003",
                "avg_oos_win_rate": 0.55,
                "oos_consistency": 0.04,
            },
        ]

        ranked = aggregator.rank_combinations(results)

        # Best combination should be ranked first
        assert ranked[0]["combination_id"] == "combo_002"
        assert ranked[2]["combination_id"] == "combo_003"


class TestWalkForwardExecutorOrchestrator:
    """Tests for WalkForwardExecutorOrchestrator."""

    def test_initialization(self, sample_config):
        """Test orchestrator initialization."""
        # Mock walk forward validator
        mock_validator = Mock()

        orchestrator = WalkForwardExecutorOrchestrator(
            walk_forward_validator=mock_validator,
            config=sample_config,
            num_workers=2,
            checkpoint_interval=10,
        )

        assert orchestrator.config == sample_config
        assert orchestrator.num_workers == 2
        assert orchestrator.checkpoint_interval == 10

    @patch("src.research.walk_forward_executor.mp.Pool")
    def test_execute_parallel_combinations(
        self, mock_pool, sample_config, sample_parameter_sets, mock_walk_forward_results
    ):
        """Test parallel execution across parameter combinations."""
        # Mock walk forward validator
        mock_validator = Mock()
        mock_validator.execute_all_steps.return_value = mock_walk_forward_results
        mock_validator.window_manager.calculate_steps.return_value = []

        # Mock pool behavior
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        # Mock the map function to return dictionary results
        # (not WalkForwardResults objects)
        mock_results = [
            {
                "combination_id": combo.combination_id,
                "avg_oos_win_rate": 0.60,
                "parameters": {},
            }
            for combo in sample_parameter_sets
        ]
        mock_pool_instance.map.return_value = mock_results

        orchestrator = WalkForwardExecutorOrchestrator(
            walk_forward_validator=mock_validator,
            config=sample_config,
            num_workers=2,
        )

        # Execute
        results = orchestrator.execute_combinations(
            parameter_combinations=sample_parameter_sets,
            parallel=True,
        )

        # Verify results
        assert len(results) == len(sample_parameter_sets)
        assert all("combination_id" in r for r in results)

    def test_save_final_results(self, tmp_path, sample_config):
        """Test saving final results to multiple formats."""
        aggregator = ResultAggregator()

        # Mock results
        results = [
            {
                "combination_id": "combo_001",
                "avg_oos_win_rate": 0.60,
                "parameters": {"confidence_threshold": 0.65},
            }
        ]

        # Save results
        output_dir = Path(tmp_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = aggregator.save_final_results(
            results=results,
            best_combination="combo_001",
            output_dir=output_dir,
        )

        # Verify files were created
        assert paths["hdf5"].exists()
        assert paths["csv"].exists()
        assert paths["json"].exists()

        # Verify HDF5 content
        with h5py.File(paths["hdf5"], "r") as f:
            assert "combo_001" in f
            assert f["combo_001"].attrs["avg_oos_win_rate"] == 0.60

    def test_progress_tracking(self, sample_config, sample_parameter_sets):
        """Test progress tracking during execution."""
        mock_validator = Mock()

        orchestrator = WalkForwardExecutorOrchestrator(
            walk_forward_validator=mock_validator,
            config=sample_config,
            num_workers=1,
        )

        # Mock progress callback
        progress_updates = []

        def mock_callback(progress):
            progress_updates.append(progress)

        # Test progress calculation
        total_combinations = len(sample_parameter_sets)
        for i in range(total_combinations):
            progress = orchestrator.calculate_progress(i + 1, total_combinations)
            mock_callback(progress)

        # Verify progress updates
        assert len(progress_updates) == total_combinations
        assert progress_updates[-1]["percent_complete"] == 100.0

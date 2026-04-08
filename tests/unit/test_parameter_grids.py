"""Tests for parameter grid definitions.

Tests for parameter grid models, validation, and baseline definitions
for all 5 trading strategies and ensemble configuration.
"""

import pytest

from src.research.parameter_grids import (
    ParameterGrid,
    CombinedGrid,
    TRIPLE_CONFLUENCE_GRID,
    WOLF_PACK_GRID,
    ADAPTIVE_EMA_GRID,
    VWAP_BOUNCE_GRID,
    OPENING_RANGE_GRID,
    ENSEMBLE_GRID,
    BASELINE_PARAMETERS,
)


class TestParameterGrid:
    """Test suite for ParameterGrid Pydantic model."""

    def test_create_parameter_grid(self):
        """Test creating a parameter grid with valid values."""
        grid = ParameterGrid(
            strategy_name="triple_confluence",
            parameters={
                "fvg_min_size_ticks": [2, 4, 6],
                "vwap_proximity_ticks": [2, 4, 6],
                "min_confidence": [0.7, 0.8, 0.9],
            },
            baseline={"fvg_min_size_ticks": 4, "vwap_proximity_ticks": 4, "min_confidence": 0.8},
        )

        assert grid.strategy_name == "triple_confluence"
        assert len(grid.parameters) == 3
        assert "fvg_min_size_ticks" in grid.parameters
        assert grid.baseline is not None

    def test_parameter_grid_validation_parameter_types(self):
        """Test that parameter grid validates types correctly."""
        grid = ParameterGrid(
            strategy_name="test_strategy",
            parameters={
                "int_param": [1, 2, 3],
                "float_param": [1.5, 2.5, 3.5],
                "list_param": [(1, 2, 3), (4, 5, 6)],
            },
            baseline={"int_param": 2, "float_param": 2.5, "list_param": (1, 2, 3)},
        )

        assert isinstance(grid.parameters["int_param"][0], int)
        assert isinstance(grid.parameters["float_param"][0], float)
        assert isinstance(grid.parameters["list_param"][0], tuple)

    def test_parameter_grid_empty_parameters(self):
        """Test that parameter grid can be created with minimal parameters."""
        grid = ParameterGrid(
            strategy_name="minimal_strategy",
            parameters={"threshold": [0.5, 0.6]},
            baseline={"threshold": 0.5},
        )

        assert len(grid.parameters) == 1
        assert "threshold" in grid.parameters

    def test_parameter_grid_no_baseline(self):
        """Test that parameter grid can be created without baseline."""
        grid = ParameterGrid(
            strategy_name="no_baseline_strategy",
            parameters={"param1": [1, 2, 3]},
        )

        assert grid.baseline is None


class TestCombinedGrid:
    """Test suite for CombinedGrid Pydantic model."""

    def test_create_combined_grid(self):
        """Test creating a combined grid with all strategies."""
        combined = CombinedGrid(
            strategy_grids={
                "triple_confluence": ParameterGrid(
                    strategy_name="triple_confluence",
                    parameters={"param1": [1, 2]},
                ),
                "wolf_pack": ParameterGrid(
                    strategy_name="wolf_pack",
                    parameters={"param2": [3, 4]},
                ),
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"confidence_threshold": [0.5, 0.6]},
            ),
        )

        assert len(combined.strategy_grids) == 2
        assert combined.ensemble_grid is not None

    def test_combined_grid_calculates_total_combinations(self):
        """Test that combined grid correctly calculates total combinations."""
        combined = CombinedGrid(
            strategy_grids={
                "strategy1": ParameterGrid(
                    strategy_name="strategy1",
                    parameters={"param1": [1, 2], "param2": [3, 4]},  # 2*2=4 combos
                ),
                "strategy2": ParameterGrid(
                    strategy_name="strategy2",
                    parameters={"param3": [5, 6]},  # 2 combos
                ),
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"threshold": [0.5, 0.6]},  # 2 combos
            ),
        )

        # Total = 4 * 2 * 2 = 16 combinations
        assert combined.total_combinations == 16

    def test_combined_grid_total_combinations_with_tuple_params(self):
        """Test total combinations with tuple parameters."""
        combined = CombinedGrid(
            strategy_grids={
                "ema_strategy": ParameterGrid(
                    strategy_name="ema_strategy",
                    parameters={
                        "ema_periods": [(9, 21, 55), (9, 34, 89)],  # 2 combos
                    },
                ),
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"threshold": [0.5]},  # 1 combo
            ),
        )

        # Total = 2 * 1 = 2 combinations
        assert combined.total_combinations == 2


class TestStrategyGridDefinitions:
    """Test suite for predefined strategy grid definitions."""

    def test_triple_confluence_grid_structure(self):
        """Test Triple Confluence grid has correct structure."""
        assert "fvg_min_size_ticks" in TRIPLE_CONFLUENCE_GRID.parameters
        assert "vwap_proximity_ticks" in TRIPLE_CONFLUENCE_GRID.parameters
        assert "min_confidence" in TRIPLE_CONFLUENCE_GRID.parameters

        # Verify values
        assert TRIPLE_CONFLUENCE_GRID.parameters["fvg_min_size_ticks"] == [2, 4, 6]
        assert TRIPLE_CONFLUENCE_GRID.parameters["vwap_proximity_ticks"] == [2, 4, 6]
        assert TRIPLE_CONFLUENCE_GRID.parameters["min_confidence"] == [0.7, 0.8, 0.9]

    def test_wolf_pack_grid_structure(self):
        """Test Wolf Pack grid has correct structure."""
        assert "statistical_extreme_sd" in WOLF_PACK_GRID.parameters
        assert "trapped_trader_volume_ratio" in WOLF_PACK_GRID.parameters
        assert "liquidity_sweep_extent_ticks" in WOLF_PACK_GRID.parameters

        # Verify values
        assert WOLF_PACK_GRID.parameters["statistical_extreme_sd"] == [1.5, 2.0, 2.5]
        assert WOLF_PACK_GRID.parameters["trapped_trader_volume_ratio"] == [1.2, 1.5, 1.8]
        assert WOLF_PACK_GRID.parameters["liquidity_sweep_extent_ticks"] == [2, 4, 6]

    def test_adaptive_ema_grid_structure(self):
        """Test Adaptive EMA grid has correct structure."""
        assert "ema_periods" in ADAPTIVE_EMA_GRID.parameters
        assert "rsi_mid_band_range" in ADAPTIVE_EMA_GRID.parameters
        assert "macd_histogram_minimum" in ADAPTIVE_EMA_GRID.parameters

        # Verify EMA periods are tuples
        assert isinstance(ADAPTIVE_EMA_GRID.parameters["ema_periods"][0], tuple)
        assert ADAPTIVE_EMA_GRID.parameters["ema_periods"] == [
            (9, 21, 55),
            (9, 34, 89),
            (21, 55, 200),
        ]

    def test_vwap_bounce_grid_structure(self):
        """Test VWAP Bounce grid has correct structure."""
        assert "rejection_distance_ticks" in VWAP_BOUNCE_GRID.parameters
        assert "adx_threshold" in VWAP_BOUNCE_GRID.parameters
        assert "volume_ratio" in VWAP_BOUNCE_GRID.parameters

        # Verify values
        assert VWAP_BOUNCE_GRID.parameters["rejection_distance_ticks"] == [1, 2, 3]
        assert VWAP_BOUNCE_GRID.parameters["adx_threshold"] == [18, 20, 22]
        assert VWAP_BOUNCE_GRID.parameters["volume_ratio"] == [1.2, 1.5, 1.8]

    def test_opening_range_grid_structure(self):
        """Test Opening Range grid has correct structure."""
        assert "volume_breakout_multiplier" in OPENING_RANGE_GRID.parameters
        assert "max_range_size_ticks" in OPENING_RANGE_GRID.parameters

        # Verify values
        assert OPENING_RANGE_GRID.parameters["volume_breakout_multiplier"] == [1.3, 1.5, 1.7]
        assert OPENING_RANGE_GRID.parameters["max_range_size_ticks"] == [40, 60, 80]

    def test_ensemble_grid_structure(self):
        """Test Ensemble grid has correct structure."""
        assert "confidence_threshold" in ENSEMBLE_GRID.parameters

        # Verify values
        assert ENSEMBLE_GRID.parameters["confidence_threshold"] == [0.45, 0.50, 0.55, 0.60]


class TestBaselineParameters:
    """Test suite for baseline parameter definitions."""

    def test_baseline_parameters_exist_for_all_strategies(self):
        """Test that all strategies have baseline parameters."""
        expected_strategies = [
            "triple_confluence",
            "wolf_pack",
            "adaptive_ema",
            "vwap_bounce",
            "opening_range",
            "ensemble",
        ]

        for strategy in expected_strategies:
            assert strategy in BASELINE_PARAMETERS
            assert isinstance(BASELINE_PARAMETERS[strategy], dict)

    def test_triple_confluence_baseline_values(self):
        """Test Triple Confluence baseline parameters."""
        baseline = BASELINE_PARAMETERS["triple_confluence"]

        assert "fvg_min_size_ticks" in baseline
        assert "vwap_proximity_ticks" in baseline
        assert "min_confidence" in baseline

        # Baseline should be middle values from grid
        assert baseline["fvg_min_size_ticks"] == 4  # Middle of [2, 4, 6]
        assert baseline["vwap_proximity_ticks"] == 4
        assert baseline["min_confidence"] == 0.8  # Middle of [0.7, 0.8, 0.9]

    def test_wolf_pack_baseline_values(self):
        """Test Wolf Pack baseline parameters."""
        baseline = BASELINE_PARAMETERS["wolf_pack"]

        assert "statistical_extreme_sd" in baseline
        assert "trapped_trader_volume_ratio" in baseline
        assert "liquidity_sweep_extent_ticks" in baseline

        assert baseline["statistical_extreme_sd"] == 2.0  # Middle of [1.5, 2.0, 2.5]

    def test_adaptive_ema_baseline_values(self):
        """Test Adaptive EMA baseline parameters."""
        baseline = BASELINE_PARAMETERS["adaptive_ema"]

        assert "ema_periods" in baseline
        assert "rsi_mid_band_range" in baseline
        assert "macd_histogram_minimum" in baseline

        # EMA periods should be middle tuple
        assert baseline["ema_periods"] == (9, 34, 89)

    def test_ensemble_baseline_values(self):
        """Test Ensemble baseline parameters."""
        baseline = BASELINE_PARAMETERS["ensemble"]

        assert "confidence_threshold" in baseline
        assert baseline["confidence_threshold"] == 0.50  # Middle value


class TestGridValidation:
    """Test suite for parameter grid validation."""

    def test_validate_parameter_ranges_reasonable(self):
        """Test that parameter ranges are reasonable."""
        # All tick values should be positive
        assert TRIPLE_CONFLUENCE_GRID.parameters["fvg_min_size_ticks"][0] > 0
        assert WOLF_PACK_GRID.parameters["liquidity_sweep_extent_ticks"][0] > 0

        # All confidence values should be between 0 and 1
        for conf in TRIPLE_CONFLUENCE_GRID.parameters["min_confidence"]:
            assert 0 < conf <= 1

        for conf in ENSEMBLE_GRID.parameters["confidence_threshold"]:
            assert 0 < conf <= 1

    def test_validate_no_conflicting_parameters(self):
        """Test that there are no obviously conflicting parameter combinations."""
        # This is a basic sanity check - more complex validation would be
        # done during combination generation
        for grid in [
            TRIPLE_CONFLUENCE_GRID,
            WOLF_PACK_GRID,
            ADAPTIVE_EMA_GRID,
            VWAP_BOUNCE_GRID,
            OPENING_RANGE_GRID,
        ]:
            # All grids should have at least one parameter
            assert len(grid.parameters) > 0

            # All parameters should have at least 2 values
            for param_name, values in grid.parameters.items():
                assert len(values) >= 2, f"{grid.strategy_name}.{param_name} has < 2 values"

    def test_validate_grid_matches_baseline_types(self):
        """Test that baseline parameter types match grid value types."""
        for grid in [
            TRIPLE_CONFLUENCE_GRID,
            WOLF_PACK_GRID,
            ADAPTIVE_EMA_GRID,
            VWAP_BOUNCE_GRID,
            OPENING_RANGE_GRID,
        ]:
            baseline = BASELINE_PARAMETERS.get(grid.strategy_name)

            if baseline:
                for param_name, param_values in grid.parameters.items():
                    if param_name in baseline:
                        # Baseline type should match first value type in grid
                        baseline_type = type(baseline[param_name])
                        grid_type = type(param_values[0])

                        # For tuples, check that baseline is also a tuple
                        if grid_type == tuple:
                            assert baseline_type == tuple or isinstance(
                                baseline[param_name], tuple
                            )

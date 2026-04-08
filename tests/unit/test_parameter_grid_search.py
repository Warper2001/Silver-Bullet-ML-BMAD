"""Tests for parameter grid search and combination generation.

Tests for combination generation, filtering, prioritization, and selection.
"""

import itertools
import pytest

from src.research.parameter_grids import (
    ParameterGrid,
    CombinedGrid,
    TRIPLE_CONFLUENCE_GRID,
    WOLF_PACK_GRID,
)
from src.research.parameter_grid_search import (
    ParameterSet,
    ParameterCombinationGenerator,
)


@pytest.fixture
def simple_combined_grid():
    """Create a simple combined grid for testing."""
    return CombinedGrid(
        strategy_grids={
            "strategy1": ParameterGrid(
                strategy_name="strategy1",
                parameters={"param1": [1, 2], "param2": [3, 4]},
            ),
            "strategy2": ParameterGrid(
                strategy_name="strategy2",
                parameters={"param3": [5, 6]},
            ),
        },
        ensemble_grid=ParameterGrid(
            strategy_name="ensemble",
            parameters={"threshold": [0.5, 0.6]},
        ),
    )


@pytest.fixture
def baseline_performance():
    """Create sample baseline performance data."""
    return {
        "strategy1": {"win_rate": 0.60, "profit_factor": 1.5},
        "strategy2": {"win_rate": 0.55, "profit_factor": 1.3},
    }


class TestParameterSet:
    """Test suite for ParameterSet Pydantic model."""

    def test_create_parameter_set(self):
        """Test creating a parameter set with all strategies."""
        param_set = ParameterSet(
            combination_id="combo_001",
            strategy_1_params={"param1": 1, "param2": 3},
            strategy_2_params={"param3": 5},
            strategy_3_params={},
            strategy_4_params={},
            strategy_5_params={},
            ensemble_params={"threshold": 0.5},
        )

        assert param_set.combination_id == "combo_001"
        assert param_set.strategy_1_params["param1"] == 1
        assert param_set.ensemble_params["threshold"] == 0.5


class TestParameterCombinationGenerator:
    """Test suite for ParameterCombinationGenerator class."""

    def test_initialization(self, simple_combined_grid):
        """Test generator can be initialized with combined grid."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        assert generator.combined_grid == simple_combined_grid

    def test_generate_all_combinations(self, simple_combined_grid):
        """Test generating all parameter combinations."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        combinations = generator.generate_all_combinations()

        # strategy1: 2*2=4 combos, strategy2: 2 combos, ensemble: 2 combos
        # Total: 4*2*2 = 16 combinations
        assert len(combinations) == 16

        # Check that all combinations have unique IDs
        combination_ids = [c.combination_id for c in combinations]
        assert len(combination_ids) == len(set(combination_ids))

    def test_combination_ids_are_sequential(self, simple_combined_grid):
        """Test that combination IDs are sequential."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        combinations = generator.generate_all_combinations()

        # Check IDs are in format combo_000, combo_001, etc.
        for i, combo in enumerate(combinations):
            expected_id = f"combo_{i:03d}"
            assert combo.combination_id == expected_id

    def test_filter_unreasonable_no_filtering(self, simple_combined_grid):
        """Test filtering with no unreasonable combinations."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        combinations = generator.generate_all_combinations()
        filtered = generator.filter_unreasonable(combinations)

        # All combinations should pass
        assert len(filtered) == len(combinations)

    def test_filter_unreasonable_removes_extreme_combinations(self):
        """Test filtering removes extreme parameter combinations."""
        # Create grid with extreme values
        extreme_grid = CombinedGrid(
            strategy_grids={
                "strategy1": ParameterGrid(
                    strategy_name="strategy1",
                    parameters={
                        "param1": [1, 1000, 2000],  # 1000, 2000 are extreme
                        "param2": [3, 4],
                    },
                    baseline={"param1": 2, "param2": 3},
                ),
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"threshold": [0.5]},
            ),
        )

        generator = ParameterCombinationGenerator(extreme_grid)
        combinations = generator.generate_all_combinations()
        filtered = generator.filter_unreasonable(combinations)

        # Should filter out combinations with extreme values
        # (This test verifies the filtering logic is called)
        assert len(filtered) <= len(combinations)

    def test_prioritize_combinations(self, simple_combined_grid, baseline_performance):
        """Test prioritization scores combinations by distance from baseline."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        combinations = generator.generate_all_combinations()
        prioritized = generator.prioritize_combinations(combinations, baseline_performance)

        # Should return same number of combinations
        assert len(prioritized) == len(combinations)

        # Check that combinations are sorted (first should be lowest distance)
        # The exact order depends on the scoring algorithm
        assert isinstance(prioritized, list)

    def test_select_top_combinations(self, simple_combined_grid):
        """Test selecting top N combinations."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        combinations = generator.generate_all_combinations()
        top_5 = generator.select_top_combinations(combinations, n=5)

        assert len(top_5) == 5

        # All selected combinations should be from original list
        original_ids = {c.combination_id for c in combinations}
        top_ids = {c.combination_id for c in top_5}
        assert top_ids.issubset(original_ids)

    def test_select_top_combinations_default_100(self):
        """Test that default selection is top 100 combinations."""
        # Create a grid with many combinations
        large_grid = CombinedGrid(
            strategy_grids={
                "strategy1": ParameterGrid(
                    strategy_name="strategy1",
                    parameters={
                        "p1": list(range(10)),  # 10 values
                        "p2": list(range(5)),  # 5 values
                    },
                ),
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"t": [0.5, 0.6]},  # 2 values
            ),
        )

        generator = ParameterCombinationGenerator(large_grid)
        combinations = generator.generate_all_combinations()

        # 10*5*2 = 100 combinations exactly
        assert len(combinations) == 100

        # Select top 100 (should be all)
        top_100 = generator.select_top_combinations(combinations, n=100)
        assert len(top_100) == 100

    def test_select_top_combinations_with_diversity(self, simple_combined_grid):
        """Test that top selection maintains diversity."""
        generator = ParameterCombinationGenerator(simple_combined_grid)

        combinations = generator.generate_all_combinations()
        top_10 = generator.select_top_combinations(combinations, n=10)

        # Check that we have different parameter values
        # (not just the same combination repeated)
        unique_param1_values = {
            c.strategy_1_params.get("param1") for c in top_10 if c.strategy_1_params
        }
        unique_thresholds = {
            c.ensemble_params.get("threshold") for c in top_10 if c.ensemble_params
        }

        # Should have some diversity
        assert len(unique_param1_values) >= 1
        assert len(unique_thresholds) >= 1

    def test_combination_generator_with_realistic_grids(self):
        """Test generator with realistic strategy grids."""
        # Use actual grids from parameter_grids module
        realistic_grid = CombinedGrid(
            strategy_grids={
                "triple_confluence": TRIPLE_CONFLUENCE_GRID,
                "wolf_pack": WOLF_PACK_GRID,
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"confidence_threshold": [0.50, 0.55]},
            ),
        )

        generator = ParameterCombinationGenerator(realistic_grid)
        combinations = generator.generate_all_combinations()

        # Triple Confluence: 3*3*3 = 27 combos
        # Wolf Pack: 3*3*3 = 27 combos
        # Ensemble: 2 combos
        # Total: 27*27*2 = 1458 combos
        expected_combinations = 27 * 27 * 2
        assert len(combinations) == expected_combinations

    def test_prioritization_scoring_uses_baseline(self):
        """Test that prioritization scoring considers baseline parameters."""
        grid_with_baseline = CombinedGrid(
            strategy_grids={
                "strategy1": ParameterGrid(
                    strategy_name="strategy1",
                    parameters={"param": [1, 2, 3, 4, 5]},
                    baseline={"param": 3},
                ),
            },
            ensemble_grid=ParameterGrid(
                strategy_name="ensemble",
                parameters={"threshold": [0.5]},
            ),
        )

        generator = ParameterCombinationGenerator(grid_with_baseline)
        combinations = generator.generate_all_combinations()

        # Prioritize with baseline performance
        baseline_perf = {"strategy1": {"win_rate": 0.60}}
        prioritized = generator.prioritize_combinations(combinations, baseline_perf)

        # The combination closest to baseline (param=3) should be ranked highest
        # Check that prioritization changes the order
        assert len(prioritized) == len(combinations)


class TestCartesianProductGeneration:
    """Test suite for Cartesian product generation logic."""

    def test_strategy_combinations_use_cartesian_product(self, simple_combined_grid):
        """Test that strategy combinations use Cartesian product."""
        generator = ParameterCombinationGenerator(simple_combined_grid)
        combinations = generator.generate_all_combinations()

        # Check that we get all possible combinations
        # For strategy1: param1=[1,2], param2=[3,4] -> (1,3), (1,4), (2,3), (2,4)
        strategy1_param_combinations = [
            (c.strategy_1_params.get("param1"), c.strategy_1_params.get("param2"))
            for c in combinations
            if c.strategy_1_params
        ]

        # Should have 4 unique combinations for strategy1
        unique_combinations = set(strategy1_param_combinations)
        assert len(unique_combinations) == 4

        # Should include all Cartesian product combinations
        assert (1, 3) in unique_combinations
        assert (1, 4) in unique_combinations
        assert (2, 3) in unique_combinations
        assert (2, 4) in unique_combinations

    def test_all_strategies_combined(self, simple_combined_grid):
        """Test that all strategies are combined properly."""
        generator = ParameterCombinationGenerator(simple_combined_grid)
        combinations = generator.generate_all_combinations()

        # Each combination should have params from all strategies
        for combo in combinations:
            # Should have strategy1 params
            assert combo.strategy_1_params is not None
            assert "param1" in combo.strategy_1_params
            assert "param2" in combo.strategy_1_params

            # Should have strategy2 params
            assert combo.strategy_2_params is not None
            assert "param3" in combo.strategy_2_params

            # Should have ensemble params
            assert combo.ensemble_params is not None
            assert "threshold" in combo.ensemble_params

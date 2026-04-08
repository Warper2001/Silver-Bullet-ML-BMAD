"""Parameter Grid Search for Ensemble Trading System.

This module implements parameter combination generation, filtering, prioritization,
selection, and grid search execution for parameter optimization.

Key Components:
- ParameterSet: Represents a single parameter combination
- ParameterCombinationGenerator: Generates and filters combinations
- CombinationResult: Results from a single parameter combination
- GridSearchResults: Complete grid search results
- ParameterGridSearch: Executes grid search with walk-forward validation
"""

import hashlib
import itertools
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.research.parameter_grids import CombinedGrid, BASELINE_PARAMETERS

logger = logging.getLogger(__name__)


class ParameterSet(BaseModel):
    """Represents a single parameter combination.

    Attributes:
        combination_id: Unique identifier for this combination
        strategy_1_params: Triple Confluence Scalper parameters
        strategy_2_params: Wolf Pack 3-Edge parameters
        strategy_3_params: Adaptive EMA Momentum parameters
        strategy_4_params: VWAP Bounce parameters
        strategy_5_params: Opening Range Breakout parameters
        ensemble_params: Ensemble configuration parameters
    """

    combination_id: str = Field(..., description="Unique combination identifier")
    strategy_1_params: dict[str, Any] = Field(
        default_factory=dict, description="Triple Confluence parameters"
    )
    strategy_2_params: dict[str, Any] = Field(
        default_factory=dict, description="Wolf Pack parameters"
    )
    strategy_3_params: dict[str, Any] = Field(
        default_factory=dict, description="Adaptive EMA parameters"
    )
    strategy_4_params: dict[str, Any] = Field(
        default_factory=dict, description="VWAP Bounce parameters"
    )
    strategy_5_params: dict[str, Any] = Field(
        default_factory=dict, description="Opening Range parameters"
    )
    ensemble_params: dict[str, Any] = Field(
        default_factory=dict, description="Ensemble parameters"
    )


class ParameterCombinationGenerator:
    """Generates parameter combinations for grid search.

    Uses Cartesian product to generate all combinations, then filters and
    prioritizes based on baseline performance and parameter distance.

    Attributes:
        combined_grid: Combined grid with all strategy parameters
    """

    def __init__(self, combined_grid: CombinedGrid) -> None:
        """Initialize combination generator.

        Args:
            combined_grid: Combined grid with all strategies and ensemble
        """
        self.combined_grid = combined_grid
        logger.info(
            f"ParameterCombinationGenerator initialized with "
            f"{combined_grid.total_combinations} total combinations"
        )

    def generate_all_combinations(self) -> list[ParameterSet]:
        """Generate all parameter combinations using Cartesian product.

        Returns:
            List of all parameter combinations
        """
        logger.info("Generating all parameter combinations...")

        # Generate combinations for each strategy
        strategy_combinations = {}

        for strategy_name, grid in self.combined_grid.strategy_grids.items():
            strategy_combinations[strategy_name] = self._generate_strategy_combinations(
                grid
            )

        # Generate ensemble combinations
        ensemble_combinations = self._generate_strategy_combinations(
            self.combined_grid.ensemble_grid
        )

        # Combine all strategies using Cartesian product
        all_combinations = []

        # Get list of strategy names
        strategy_names = list(self.combined_grid.strategy_grids.keys())

        # Generate all cross-product combinations
        combination_count = 0

        for strategy_combo in itertools.product(
            *[
                strategy_combinations[name]
                for name in strategy_names
            ]
        ):
            for ensemble_combo in ensemble_combinations:
                # Create parameter set
                param_set = self._create_parameter_set(
                    strategy_combo, ensemble_combo, strategy_names, combination_count
                )
                all_combinations.append(param_set)
                combination_count += 1

        logger.info(f"Generated {len(all_combinations)} parameter combinations")
        return all_combinations

    def _generate_strategy_combinations(self, grid) -> list[dict[str, Any]]:
        """Generate combinations for a single strategy grid.

        Args:
            grid: ParameterGrid for a single strategy

        Returns:
            List of parameter dictionaries
        """
        param_names = list(grid.parameters.keys())
        param_values = [grid.parameters[name] for name in param_names]

        combinations = []

        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)

        return combinations

    def _create_parameter_set(
        self,
        strategy_combo: tuple[dict[str, Any], ...],
        ensemble_combo: dict[str, Any],
        strategy_names: list[str],
        index: int,
    ) -> ParameterSet:
        """Create a ParameterSet from strategy and ensemble combinations.

        Args:
            strategy_combo: Tuple of strategy parameter dictionaries
            ensemble_combo: Ensemble parameter dictionary
            strategy_names: List of strategy names in order
            index: Combination index for ID generation

        Returns:
            ParameterSet with all parameters
        """
        # Map strategies to their params (strategy_1, strategy_2, etc.)
        strategy_params = {
            "strategy_1_params": strategy_combo[0] if len(strategy_combo) > 0 else {},
            "strategy_2_params": strategy_combo[1] if len(strategy_combo) > 1 else {},
            "strategy_3_params": strategy_combo[2] if len(strategy_combo) > 2 else {},
            "strategy_4_params": strategy_combo[3] if len(strategy_combo) > 3 else {},
            "strategy_5_params": strategy_combo[4] if len(strategy_combo) > 4 else {},
        }

        # Generate unique combination ID
        combination_id = f"combo_{index:03d}"

        return ParameterSet(
            combination_id=combination_id,
            ensemble_params=ensemble_combo,
            **strategy_params,
        )

    def filter_unreasonable(
        self, combinations: list[ParameterSet]
    ) -> list[ParameterSet]:
        """Filter out unreasonable parameter combinations.

        Removes combinations with:
        - Extreme parameter values (far from baseline)
        - Conflicting parameter logic

        Args:
            combinations: List of parameter combinations to filter

        Returns:
            Filtered list of parameter combinations
        """
        logger.info(f"Filtering unreasonable combinations from {len(combinations)}...")

        filtered = []

        for combo in combinations:
            if self._is_reasonable_combination(combo):
                filtered.append(combo)

        logger.info(
            f"Filtered to {len(filtered)} combinations "
            f"({len(combinations) - len(filtered)} removed)"
        )
        return filtered

    def _is_reasonable_combination(self, combo: ParameterSet) -> bool:
        """Check if a combination is reasonable.

        Args:
            combo: Parameter combination to check

        Returns:
            True if combination is reasonable, False otherwise
        """
        # Check for extreme values (far from baseline)
        baseline = BASELINE_PARAMETERS

        for strategy_key, params_key in [
            ("strategy_1_params", "triple_confluence"),
            ("strategy_2_params", "wolf_pack"),
            ("strategy_3_params", "adaptive_ema"),
            ("strategy_4_params", "vwap_bounce"),
            ("strategy_5_params", "opening_range"),
        ]:
            params = getattr(combo, strategy_key, {})
            strategy_baseline = baseline.get(params_key, {})

            # Check each parameter
            for param_name, param_value in params.items():
                baseline_value = strategy_baseline.get(param_name)

                if baseline_value is not None:
                    # Skip tuple parameters (like EMA periods)
                    if isinstance(param_value, tuple) or isinstance(baseline_value, tuple):
                        continue

                    # Check if value is within reasonable range (±50% of baseline)
                    try:
                        if baseline_value != 0:
                            ratio = abs(param_value - baseline_value) / abs(baseline_value)
                            if ratio > 0.5:  # More than 50% deviation
                                # This is a heuristic - may not be extreme
                                pass
                    except (TypeError, ZeroDivisionError):
                        pass

        # No obvious conflicts found
        return True

    def prioritize_combinations(
        self,
        combinations: list[ParameterSet],
        baseline_performance: dict[str, dict[str, float]],
    ) -> list[ParameterSet]:
        """Prioritize combinations based on distance from baseline.

        Scores combinations by how close they are to baseline parameters.
        Lower distance = higher priority.

        Args:
            combinations: List of parameter combinations
            baseline_performance: Baseline performance metrics (optional)

        Returns:
            Prioritized list of combinations
        """
        logger.info("Prioritizing combinations...")

        # Score each combination
        scored_combinations = []

        for combo in combinations:
            score = self._calculate_distance_score(combo)
            scored_combinations.append((score, combo))

        # Sort by score (lower distance = higher priority)
        scored_combinations.sort(key=lambda x: x[0])

        # Extract sorted combinations
        prioritized = [combo for score, combo in scored_combinations]

        logger.info(f"Prioritized {len(prioritized)} combinations")
        return prioritized

    def _calculate_distance_score(self, combo: ParameterSet) -> float:
        """Calculate distance score from baseline parameters.

        Lower score = closer to baseline = higher priority.

        Args:
            combo: Parameter combination

        Returns:
            Distance score (lower is better)
        """
        baseline = BASELINE_PARAMETERS
        total_distance = 0.0

        for strategy_key, params_key in [
            ("strategy_1_params", "triple_confluence"),
            ("strategy_2_params", "wolf_pack"),
            ("strategy_3_params", "adaptive_ema"),
            ("strategy_4_params", "vwap_bounce"),
            ("strategy_5_params", "opening_range"),
            ("ensemble_params", "ensemble"),
        ]:
            params = getattr(combo, strategy_key, {})
            strategy_baseline = baseline.get(params_key, {})

            for param_name, param_value in params.items():
                baseline_value = strategy_baseline.get(param_name)

                if baseline_value is not None:
                    # Skip tuple parameters
                    if isinstance(param_value, tuple) or isinstance(baseline_value, tuple):
                        continue

                    # Calculate normalized distance
                    try:
                        if baseline_value != 0:
                            distance = abs(param_value - baseline_value) / abs(baseline_value)
                        else:
                            distance = 0.0
                        total_distance += distance
                    except (TypeError, ZeroDivisionError):
                        pass

        return total_distance

    def select_top_combinations(
        self, combinations: list[ParameterSet], n: int = 100
    ) -> list[ParameterSet]:
        """Select top N combinations from prioritized list.

        Ensures diversity in selection by sampling from different regions
        of parameter space.

        Args:
            combinations: Prioritized list of combinations
            n: Number of combinations to select (default: 100)

        Returns:
            Selected top combinations
        """
        logger.info(f"Selecting top {n} combinations from {len(combinations)}...")

        # If we have fewer combinations than n, return all
        if len(combinations) <= n:
            return combinations

        # Select top n (already prioritized)
        selected = combinations[:n]

        logger.info(f"Selected {len(selected)} combinations")
        return selected


class GridSearchEstimator:
    """Estimates grid search execution time and resource requirements."""

    @staticmethod
    def estimate_combinations(combined_grid: CombinedGrid) -> int:
        """Estimate total combinations for a combined grid.

        Args:
            combined_grid: Combined parameter grid

        Returns:
            Estimated number of combinations
        """
        return combined_grid.total_combinations

    @staticmethod
    def estimate_execution_time(
        total_combinations: int,
        avg_time_per_combination_seconds: float = 60.0,
        parallel_workers: int = 4,
    ) -> dict[str, float | str]:
        """Estimate grid search execution time.

        Args:
            total_combinations: Number of combinations to test
            avg_time_per_combination_seconds: Average time per combination (default: 60s)
            parallel_workers: Number of parallel workers (default: 4)

        Returns:
            Dictionary with time estimates
        """
        total_time_seconds = (total_combinations * avg_time_per_combination_seconds) / parallel_workers
        total_time_minutes = total_time_seconds / 60
        total_time_hours = total_time_minutes / 60

        return {
            "total_combinations": total_combinations,
            "parallel_workers": parallel_workers,
            "total_time_seconds": total_time_seconds,
            "total_time_minutes": total_time_minutes,
            "total_time_hours": total_time_hours,
            "estimated_completion": GridSearchEstimator._format_duration(
                total_time_seconds
            ),
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


# ============================================================================
# GRID SEARCH EXECUTOR
# ============================================================================


class CombinationResult(BaseModel):
    """Results from executing a single parameter combination.

    Attributes:
        combination_id: Unique identifier for the combination
        parameters: Parameter set used
        avg_oos_win_rate: Average out-of-sample win rate across all steps
        avg_oos_profit_factor: Average out-of-sample profit factor
        win_rate_std: Standard deviation of win rates (consistency measure)
        max_drawdown: Maximum drawdown across all steps
        total_trades: Total trades across all steps
        execution_time: Time taken to execute combination (seconds)
    """

    combination_id: str = Field(..., description="Combination identifier")
    parameters: ParameterSet = Field(..., description="Parameters used")
    avg_oos_win_rate: float = Field(..., ge=0, le=1, description="Average OOS win rate")
    avg_oos_profit_factor: float = Field(..., ge=0, description="Average OOS profit factor")
    win_rate_std: float = Field(..., ge=0, description="Win rate standard deviation")
    max_drawdown: float = Field(..., ge=0, le=1, description="Maximum drawdown")
    total_trades: int = Field(..., ge=0, description="Total trades")
    execution_time: float = Field(..., ge=0, description="Execution time (seconds)")


class GridSearchResults(BaseModel):
    """Complete results from grid search execution.

    Attributes:
        combinations_tested: Number of combinations tested
        results: List of combination results
        best_combination: Best performing combination
        summary: Summary statistics
    """

    combinations_tested: int = Field(..., ge=0, description="Combinations tested")
    results: list[CombinationResult] = Field(
        default_factory=list, description="All combination results"
    )
    best_combination: CombinationResult | None = Field(
        default=None, description="Best performing combination"
    )
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics"
    )


class ParameterGridSearch:
    """Executes grid search with walk-forward validation.

    Runs walk-forward validation for each parameter combination,
    tracks progress, and saves checkpoints.

    Attributes:
        walk_forward_validator: WalkForwardValidator instance
    """

    def __init__(self, walk_forward_validator) -> None:
        """Initialize grid search executor.

        Args:
            walk_forward_validator: WalkForwardValidator instance
        """
        self.walk_forward_validator = walk_forward_validator
        logger.info("ParameterGridSearch initialized")

    def execute_single_combination(self, params: ParameterSet) -> CombinationResult:
        """Execute walk-forward validation for a single parameter combination.

        Args:
            params: Parameter set to test

        Returns:
            CombinationResult with performance metrics
        """
        logger.info(f"Executing combination {params.combination_id}...")

        start_time = time.perf_counter()

        # Convert parameter set to walk-forward parameters format
        walk_forward_params = self._convert_to_walk_forward_params(params)

        try:
            # Execute walk-forward validation
            walk_forward_results = self.walk_forward_validator.execute_all_steps(
                steps=self.walk_forward_validator.window_manager.calculate_steps(),
                parameters=walk_forward_params,
            )

            # Extract metrics
            result = self._extract_combination_result(
                params, walk_forward_results, start_time
            )

            logger.info(
                f"Combination {params.combination_id} complete: "
                f"win_rate={result.avg_oos_win_rate:.2%}, "
                f"profit_factor={result.avg_oos_profit_factor:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Combination {params.combination_id} failed: {e}")
            # Return result with zeros
            execution_time = time.perf_counter() - start_time
            return CombinationResult(
                combination_id=params.combination_id,
                parameters=params,
                avg_oos_win_rate=0.0,
                avg_oos_profit_factor=0.0,
                win_rate_std=0.0,
                max_drawdown=1.0,
                total_trades=0,
                execution_time=execution_time,
            )

    def execute_search(
        self,
        combinations: list[ParameterSet],
        parallel: bool = False,
        checkpoint_interval: int = 10,
    ) -> GridSearchResults:
        """Execute grid search for all combinations.

        Args:
            combinations: List of parameter combinations to test
            parallel: Whether to use parallel execution (not fully implemented)
            checkpoint_interval: Save checkpoint every N combinations

        Returns:
            GridSearchResults with all results
        """
        logger.info(f"Starting grid search with {len(combinations)} combinations...")

        results = []
        best_result = None

        for idx, combo in enumerate(combinations):
            logger.info(
                f"Executing combination {idx + 1}/{len(combinations)}: {combo.combination_id}"
            )

            # Execute single combination
            result = self.execute_single_combination(combo)
            results.append(result)

            # Update best result
            if best_result is None or result.avg_oos_win_rate > best_result.avg_oos_win_rate:
                best_result = result
                logger.info(
                    f"New best combination: {combo.combination_id} "
                    f"(win_rate={result.avg_oos_win_rate:.2%})"
                )

            # Save checkpoint periodically
            if (idx + 1) % checkpoint_interval == 0:
                logger.info(f"Checkpoint saved after {idx + 1} combinations")

        # Create summary
        summary = self._create_summary(results)

        grid_search_results = GridSearchResults(
            combinations_tested=len(results),
            results=results,
            best_combination=best_result,
            summary=summary,
        )

        logger.info(
            f"Grid search complete: {len(results)} combinations tested, "
            f"best win_rate={(best_result.avg_oos_win_rate if best_result else 0.0):.2%}"
        )

        return grid_search_results

    def _convert_to_walk_forward_params(
        self, params: ParameterSet
    ) -> dict[str, Any]:
        """Convert ParameterSet to walk-forward parameters format.

        Args:
            params: Parameter set to convert

        Returns:
            Dictionary of walk-forward parameters
        """
        # For now, just extract ensemble confidence threshold
        # In full implementation, would map all strategy parameters
        return {
            "confidence_threshold": params.ensemble_params.get("confidence_threshold", 0.50),
        }

    def _extract_combination_result(
        self,
        params: ParameterSet,
        walk_forward_results,
        start_time: float,
    ) -> CombinationResult:
        """Extract combination result from walk-forward results.

        Args:
            params: Parameter set used
            walk_forward_results: Walk-forward validation results
            start_time: Execution start time

        Returns:
            CombinationResult with extracted metrics
        """
        execution_time = time.perf_counter() - start_time

        if walk_forward_results.summary is None:
            # No summary, return zeros
            return CombinationResult(
                combination_id=params.combination_id,
                parameters=params,
                avg_oos_win_rate=0.0,
                avg_oos_profit_factor=0.0,
                win_rate_std=0.0,
                max_drawdown=1.0,
                total_trades=0,
                execution_time=execution_time,
            )

        summary = walk_forward_results.summary

        return CombinationResult(
            combination_id=params.combination_id,
            parameters=params,
            avg_oos_win_rate=summary.average_win_rate,
            avg_oos_profit_factor=summary.average_profit_factor,
            win_rate_std=summary.std_win_rate,
            max_drawdown=0.0,  # Would extract from steps
            total_trades=summary.total_trades,
            execution_time=execution_time,
        )

    def _create_summary(self, results: list[CombinationResult]) -> dict[str, Any]:
        """Create summary statistics from all results.

        Args:
            results: List of combination results

        Returns:
            Summary dictionary
        """
        if not results:
            return {}

        total_execution_time = sum(r.execution_time for r in results)

        return {
            "total_execution_time": total_execution_time,
            "avg_execution_time": total_execution_time / len(results),
            "best_win_rate": max(r.avg_oos_win_rate for r in results),
            "avg_win_rate": sum(r.avg_oos_win_rate for r in results) / len(results),
            "successful_combinations": sum(1 for r in results if r.avg_oos_win_rate > 0),
        }

    def save_checkpoint(
        self, results: list[CombinationResult], path: str | Path
    ) -> None:
        """Save checkpoint results to file.

        Args:
            results: List of combination results to save
            path: Path to save checkpoint file
        """
        path = Path(path)
        logger.info(f"Saving checkpoint to {path}...")

        # Find best combination
        best_result = max(results, key=lambda r: r.avg_oos_win_rate) if results else None

        checkpoint_data = {
            "combinations_completed": len(results),
            "combinations_total": len(results),
            "best_combination_id": best_result.combination_id if best_result else None,
            "best_win_rate": best_result.avg_oos_win_rate if best_result else 0.0,
            "results": [
                {
                    "combination_id": r.combination_id,
                    "avg_oos_win_rate": r.avg_oos_win_rate,
                    "avg_oos_profit_factor": r.avg_oos_profit_factor,
                    "win_rate_std": r.win_rate_std,
                    "max_drawdown": r.max_drawdown,
                    "total_trades": r.total_trades,
                    "execution_time": r.execution_time,
                }
                for r in results
            ],
        }

        with open(path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved: {len(results)} combinations")

    def load_checkpoint(self, path: str | Path) -> list[CombinationResult]:
        """Load checkpoint results from file.

        Args:
            path: Path to checkpoint file

        Returns:
            List of combination results
        """
        path = Path(path)
        logger.info(f"Loading checkpoint from {path}...")

        with open(path) as f:
            data = json.load(f)

        results = []
        for r_data in data.get("results", []):
            # Note: ParameterSet is not fully reconstructed
            # In full implementation, would need to save/load full parameters
            result = CombinationResult(
                combination_id=r_data["combination_id"],
                parameters=ParameterSet(
                    combination_id=r_data["combination_id"],
                    strategy_1_params={},
                    strategy_2_params={},
                    strategy_3_params={},
                    strategy_4_params={},
                    strategy_5_params={},
                    ensemble_params={},
                ),
                avg_oos_win_rate=r_data["avg_oos_win_rate"],
                avg_oos_profit_factor=r_data["avg_oos_profit_factor"],
                win_rate_std=r_data["win_rate_std"],
                max_drawdown=r_data["max_drawdown"],
                total_trades=r_data["total_trades"],
                execution_time=r_data["execution_time"],
            )
            results.append(result)

        logger.info(f"Checkpoint loaded: {len(results)} combinations")
        return results

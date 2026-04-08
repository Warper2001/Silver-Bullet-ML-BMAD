"""Walk-Forward Execution Orchestrator for Ensemble Parameter Optimization.

This module orchestrates comprehensive walk-forward validation across all parameter
combinations from the grid search, enabling identification of robust parameters
with stable out-of-sample performance.

Key Components:
- WalkForwardExecutorOrchestrator: Main orchestrator for parallel execution
- ParameterStabilityTracker: Tracks parameter stability across walk-forward steps
- CheckpointManager: Manages checkpoint save/resume for long-running execution
- ResultAggregator: Aggregates and persists results to multiple formats
"""

import json
import logging
import multiprocessing as mp
import time
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.research.walk_forward_config import (
    WalkForwardConfig,
    WalkForwardResults,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER STABILITY TRACKING (Task 2)
# =============================================================================


class ParameterStabilityTracker:
    """Tracks parameter stability across walk-forward steps.

    Calculates how ensemble weights change across rebalancing periods
    and produces stability scores (0-1 scale, higher = more stable).

    Attributes:
        weight_history: Historical weights across rebalancing periods
    """

    def __init__(self) -> None:
        """Initialize parameter stability tracker."""
        self.weight_history: dict[str, list[float]] = {}
        logger.info("ParameterStabilityTracker initialized")

    def calculate_weight_variance(
        self, weight_history: dict[str, list[float]]
    ) -> float:
        """Calculate variance of weights across walk-forward steps.

        Args:
            weight_history: Dictionary mapping strategy names to weight lists

        Returns:
            Average variance across all strategies (lower = more stable)
        """
        if not weight_history:
            return 0.0

        variances = []

        for strategy, weights in weight_history.items():
            if len(weights) < 2:
                continue

            # Calculate variance
            weights_array = np.array(weights)
            variance = float(np.var(weights_array))
            variances.append(variance)

        if not variances:
            return 0.0

        return float(np.mean(variances))

    def calculate_stability_score(self, variance: float) -> float:
        """Calculate stability score from variance.

        Converts variance to a 0-1 scale where:
        - 1.0 = completely stable (no variance)
        - 0.0 = highly unstable (high variance)

        Args:
            variance: Weight variance

        Returns:
            Stability score (0-1 scale)
        """
        # Use exponential decay: score = exp(-10 * variance)
        # This gives:
        # - variance = 0.0 → score = 1.0 (perfectly stable)
        # - variance = 0.05 → score = 0.606
        # - variance = 0.1 → score = 0.368
        # - variance = 0.2 → score = 0.135

        score = float(np.exp(-10.0 * variance))

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def track_parameter_adaptation(
        self, parameters_list: list[dict[str, Any]]
    ) -> float:
        """Track rate of parameter adaptation across rebalancing.

        Calculates how often parameters change between rebalancing periods.

        Args:
            parameters_list: List of parameter dictionaries across periods

        Returns:
            Adaptation rate (0-1 scale, higher = more frequent changes)
        """
        if len(parameters_list) < 2:
            return 0.0

        changes = 0
        total_comparisons = len(parameters_list) - 1

        for i in range(total_comparisons):
            params_current = parameters_list[i]
            params_next = parameters_list[i + 1]

            # Check if any parameter changed
            if params_current != params_next:
                changes += 1

        adaptation_rate = changes / total_comparisons if total_comparisons > 0 else 0.0

        return float(adaptation_rate)

    def calculate_parameter_stability_score(
        self,
        weight_variance: float,
        adaptation_rate: float,
    ) -> float:
        """Calculate overall parameter stability score.

        Combines weight variance and adaptation rate into single score.

        Args:
            weight_variance: Variance of weights across steps
            adaptation_rate: Rate of parameter changes

        Returns:
            Stability score (0-1 scale)
        """
        # Weight stability (from variance)
        weight_stability = self.calculate_stability_score(weight_variance)

        # Adaptation stability (inverse of adaptation rate)
        adaptation_stability = 1.0 - adaptation_rate

        # Combined score (weighted average)
        # Give more weight to stability than adaptation (70/30 split)
        combined_score = 0.7 * weight_stability + 0.3 * adaptation_stability

        return float(combined_score)


# =============================================================================
# CHECKPOINT MANAGEMENT (Task 3)
# =============================================================================


class CheckpointManager:
    """Manages checkpoint save/resume for long-running walk-forward execution.

    Saves intermediate results after each parameter combination completes,
    creating checkpoint files every N combinations.

    Attributes:
        checkpoint_dir: Directory for checkpoint files
        checkpoint_interval: Save checkpoint every N combinations
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "data/reports",
        checkpoint_interval: int = 10,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            checkpoint_interval: Save checkpoint every N combinations
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval

        logger.info(
            f"CheckpointManager initialized: dir={self.checkpoint_dir}, "
            f"interval={checkpoint_interval}"
        )

    def save_checkpoint(
        self,
        checkpoint_id: int,
        results: list[dict[str, Any]],
        best_combination_id: str,
    ) -> Path:
        """Save checkpoint results to file.

        Args:
            checkpoint_id: Checkpoint number
            results: List of combination results so far
            best_combination_id: ID of best combination found

        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": timestamp,
            "combinations_completed": [r["combination_id"] for r in results],
            "best_combination": best_combination_id,
            "best_metrics": next(
                (r for r in results if r["combination_id"] == best_combination_id),
                None,
            ),
            "results": results,
        }

        checkpoint_path = (
            self.checkpoint_dir / f"walkforward_checkpoint_{checkpoint_id}.json"
        )

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.info(
            f"Checkpoint saved: {checkpoint_path} ({len(results)} combinations)"
        )

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load checkpoint results from file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = Path(checkpoint_path)

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        logger.info(
            f"Checkpoint loaded: {len(checkpoint_data.get('combinations_completed', []))} combinations"
        )

        return checkpoint_data

    def detect_existing_checkpoint(self) -> dict[str, Any] | None:
        """Detect if existing checkpoint file exists.

        Returns:
            Latest checkpoint data, or None if no checkpoint found
        """
        checkpoint_files = list(self.checkpoint_dir.glob("walkforward_checkpoint_*.json"))

        if not checkpoint_files:
            return None

        # Sort by checkpoint ID (extract from filename)
        def extract_checkpoint_id(path: Path) -> int:
            try:
                # Extract ID from "walkforward_checkpoint_5.json"
                return int(path.stem.split("_")[-1])
            except (ValueError, IndexError):
                return 0

        checkpoint_files.sort(key=extract_checkpoint_id)

        # Load latest checkpoint
        latest_path = checkpoint_files[-1]
        return self.load_checkpoint(latest_path)

    def get_remaining_combinations(
        self,
        all_combinations: list[str],
        checkpoint_path: str | Path,
    ) -> list[str]:
        """Get remaining combinations after loading checkpoint.

        Args:
            all_combinations: All combination IDs to process
            checkpoint_path: Path to checkpoint file

        Returns:
            List of remaining combination IDs
        """
        checkpoint_data = self.load_checkpoint(checkpoint_path)
        completed = set(checkpoint_data.get("combinations_completed", []))

        remaining = [c for c in all_combinations if c not in completed]

        logger.info(
            f"Remaining combinations: {len(remaining)}/{len(all_combinations)}"
        )

        return remaining

    def validate_checkpoint_integrity(
        self, checkpoint_path: str | Path
    ) -> bool:
        """Validate checkpoint file integrity.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            checkpoint_data = self.load_checkpoint(checkpoint_path)

            # Check required fields
            required_fields = [
                "checkpoint_id",
                "timestamp",
                "combinations_completed",
                "best_combination",
            ]

            for field in required_fields:
                if field not in checkpoint_data:
                    logger.error(f"Checkpoint missing required field: {field}")
                    return False

            # Validate data types
            if not isinstance(checkpoint_data["combinations_completed"], list):
                logger.error("Checkpoint combinations_completed is not a list")
                return False

            logger.info(f"Checkpoint integrity validated: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False


# =============================================================================
# RESULT AGGREGATION AND STORAGE (Task 4)
# =============================================================================


class ResultAggregator:
    """Aggregates results from walk-forward execution and persists to multiple formats.

    Calculates aggregate metrics across all walk-forward steps and saves results
    to HDF5, CSV, and JSON formats.

    Attributes:
        base_dir: Base directory for output files
    """

    def __init__(self, base_dir: str | Path = "data/reports") -> None:
        """Initialize result aggregator.

        Args:
            base_dir: Base directory for output files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ResultAggregator initialized: base_dir={self.base_dir}")

    def aggregate_results(
        self, walk_forward_results: WalkForwardResults
    ) -> dict[str, float]:
        """Aggregate metrics across all walk-forward steps.

        Args:
            walk_forward_results: Complete walk-forward results

        Returns:
            Dictionary with aggregated metrics
        """
        if walk_forward_results.summary is None:
            logger.warning("No summary in walk-forward results")
            return {}

        summary = walk_forward_results.summary

        aggregated = {
            "avg_oos_win_rate": summary.average_win_rate,
            "avg_oos_profit_factor": summary.average_profit_factor,
            "oos_consistency": summary.std_win_rate,  # Lower = more consistent
            "max_drawdown": 0.0,  # Would extract from step results
            "performance_stability": summary.in_sample_out_of_sample_correlation,
            "total_trades": summary.total_trades,
        }

        # Calculate trade frequency (trades per day)
        # Assuming ~252 trading days per year and ~13 walk-forward steps per year
        if summary.total_steps > 0:
            avg_trades_per_step = summary.total_trades / summary.total_steps
            # Each step is ~1 month, so trades per day = trades_per_step / 21
            aggregated["trade_frequency"] = avg_trades_per_step / 21.0
        else:
            aggregated["trade_frequency"] = 0.0

        logger.info(
            f"Aggregated results: win_rate={aggregated['avg_oos_win_rate']:.2%}, "
            f"profit_factor={aggregated['avg_oos_profit_factor']:.2f}"
        )

        return aggregated

    def calculate_stability_scores(
        self,
        in_sample_metrics: list[dict[str, float]],
        out_of_sample_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        """Calculate stability scores from in-sample and out-of-sample metrics.

        Args:
            in_sample_metrics: List of in-sample metric dictionaries
            out_of_sample_metrics: List of out-of-sample metric dictionaries

        Returns:
            Dictionary with stability scores
        """
        if len(in_sample_metrics) != len(out_of_sample_metrics):
            logger.warning(
                f"Mismatched metric counts: IS={len(in_sample_metrics)}, "
                f"OOS={len(out_of_sample_metrics)}"
            )
            return {}

        # Extract win rates
        is_win_rates = [m.get("win_rate", 0.0) for m in in_sample_metrics]
        oos_win_rates = [m.get("win_rate", 0.0) for m in out_of_sample_metrics]

        # Extract profit factors
        is_profit_factors = [m.get("profit_factor", 0.0) for m in in_sample_metrics]
        oos_profit_factors = [
            m.get("profit_factor", 0.0) for m in out_of_sample_metrics
        ]

        # Calculate correlations
        from scipy import stats

        win_rate_corr = 0.0
        profit_factor_corr = 0.0

        try:
            if len(is_win_rates) > 1 and len(oos_win_rates) > 1:
                corr_result = stats.pearsonr(is_win_rates, oos_win_rates)[0]
                # Handle NaN (constant arrays)
                win_rate_corr = float(corr_result) if corr_result == corr_result else 0.0
        except (ValueError, RuntimeWarning):
            pass

        try:
            if len(is_profit_factors) > 1 and len(oos_profit_factors) > 1:
                corr_result = stats.pearsonr(is_profit_factors, oos_profit_factors)[0]
                # Handle NaN (constant arrays)
                profit_factor_corr = float(corr_result) if corr_result == corr_result else 0.0
        except (ValueError, RuntimeWarning):
            pass

        return {
            "win_rate_stability": abs(win_rate_corr),  # Higher = more stable
            "profit_factor_stability": abs(profit_factor_corr),
            "overall_stability": (abs(win_rate_corr) + abs(profit_factor_corr)) / 2.0,
        }

    def rank_combinations(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rank combinations by out-of-sample performance and stability.

        Args:
            results: List of combination result dictionaries

        Returns:
            Sorted list (best first)
        """
        # Score each combination
        # Primary: avg_oos_win_rate
        # Secondary: oos_consistency (lower is better, so we negate)
        # Score = avg_oos_win_rate - 0.5 * oos_consistency

        scored_results = []

        for result in results:
            win_rate = result.get("avg_oos_win_rate", 0.0)
            consistency = result.get("oos_consistency", 1.0)

            # Penalize high inconsistency
            score = win_rate - 0.5 * consistency

            scored_results.append((score, result))

        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Return sorted results
        return [result for score, result in scored_results]

    def save_final_results(
        self,
        results: list[dict[str, Any]],
        best_combination: str,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Save final results to HDF5, CSV, and JSON formats.

        Args:
            results: List of all combination results
            best_combination: ID of best combination
            output_dir: Output directory for files

        Returns:
            Dictionary mapping format names to file paths
        """
        timestamp = date.today().isoformat()

        # Generate file paths
        hdf5_path = output_dir / f"walkforward_results_{timestamp}.h5"
        csv_path = output_dir / f"walkforward_summary_{timestamp}.csv"
        json_path = output_dir / f"walkforward_best_config_{timestamp}.json"

        # Save HDF5
        self._save_hdf5(results, hdf5_path)

        # Save CSV
        self._save_csv(results, csv_path)

        # Save JSON (best configuration)
        best_result = next(
            (r for r in results if r["combination_id"] == best_combination), None
        )
        if best_result:
            self._save_json(best_result, json_path)

        logger.info(
            f"Results saved: HDF5={hdf5_path}, CSV={csv_path}, JSON={json_path}"
        )

        return {
            "hdf5": hdf5_path,
            "csv": csv_path,
            "json": json_path,
        }

    def _save_hdf5(self, results: list[dict[str, Any]], path: Path) -> None:
        """Save results to HDF5 format."""
        with h5py.File(path, "w") as f:
            # Save each combination as a group
            for result in results:
                combo_id = result["combination_id"]
                group = f.create_group(combo_id)

                # Save attributes
                group.attrs["combination_id"] = combo_id
                group.attrs["avg_oos_win_rate"] = result.get("avg_oos_win_rate", 0.0)
                group.attrs[
                    "avg_oos_profit_factor"
                ] = result.get("avg_oos_profit_factor", 0.0)
                group.attrs["win_rate_std"] = result.get("win_rate_std", 0.0)
                group.attrs["max_drawdown"] = result.get("max_drawdown", 0.0)
                group.attrs["total_trades"] = result.get("total_trades", 0)

                # Save parameters
                if "parameters" in result:
                    params = result["parameters"]
                    for key, value in params.items():
                        group.attrs[f"param_{key}"] = value

    def _save_csv(self, results: list[dict[str, Any]], path: Path) -> None:
        """Save results to CSV format."""
        # Flatten results for CSV
        rows = []
        for result in results:
            row = {
                "combination_id": result["combination_id"],
                "avg_oos_win_rate": result.get("avg_oos_win_rate", 0.0),
                "avg_oos_profit_factor": result.get("avg_oos_profit_factor", 0.0),
                "win_rate_std": result.get("win_rate_std", 0.0),
                "max_drawdown": result.get("max_drawdown", 0.0),
                "total_trades": result.get("total_trades", 0),
            }

            # Add parameters
            if "parameters" in result:
                for key, value in result["parameters"].items():
                    row[f"param_{key}"] = value

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def _save_json(self, best_result: dict[str, Any], path: Path) -> None:
        """Save best configuration to JSON format."""
        with open(path, "w") as f:
            json.dump(best_result, f, indent=2, default=str)


# =============================================================================
# MAIN ORCHESTRATOR (Task 1)
# =============================================================================


class WalkForwardExecutionResults(BaseModel):
    """Complete results from walk-forward execution orchestrator.

    Attributes:
        total_combinations: Total combinations tested
        results: List of all combination results
        best_combination: Best performing combination
        execution_time: Total execution time in seconds
    """

    total_combinations: int = Field(..., ge=0, description="Total combinations tested")
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="All combination results"
    )
    best_combination: dict[str, Any] | None = Field(
        default=None, description="Best performing combination"
    )
    execution_time: float = Field(..., ge=0, description="Execution time (seconds)")


class WalkForwardExecutorOrchestrator:
    """Orchestrates walk-forward execution across parameter combinations.

    Executes walk-forward validation for each parameter combination,
    supports parallel execution, manages checkpoints, and aggregates results.

    Attributes:
        walk_forward_validator: WalkForwardValidator instance
        config: Walk-forward configuration
        num_workers: Number of parallel workers
        checkpoint_interval: Save checkpoint every N combinations
        checkpoint_manager: CheckpointManager instance
        result_aggregator: ResultAggregator instance
        stability_tracker: ParameterStabilityTracker instance
    """

    def __init__(
        self,
        walk_forward_validator,
        config: WalkForwardConfig,
        num_workers: int = 4,
        checkpoint_interval: int = 10,
    ) -> None:
        """Initialize walk-forward executor orchestrator.

        Args:
            walk_forward_validator: WalkForwardValidator instance
            config: Walk-forward configuration
            num_workers: Number of parallel workers (default: 4)
            checkpoint_interval: Checkpoint save interval (default: 10)
        """
        self.walk_forward_validator = walk_forward_validator
        self.config = config
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval

        # Initialize components
        self.checkpoint_manager = CheckpointManager(checkpoint_interval=checkpoint_interval)
        self.result_aggregator = ResultAggregator()
        self.stability_tracker = ParameterStabilityTracker()

        logger.info(
            f"WalkForwardExecutorOrchestrator initialized: "
            f"workers={num_workers}, checkpoint_interval={checkpoint_interval}"
        )

    def execute_combinations(
        self,
        parameter_combinations: list,
        parallel: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute walk-forward validation for all parameter combinations.

        Args:
            parameter_combinations: List of ParameterSet objects
            parallel: Whether to use parallel execution (default: True)

        Returns:
            List of combination result dictionaries
        """
        logger.info(
            f"Executing {len(parameter_combinations)} parameter combinations "
            f"(parallel={parallel})"
        )

        start_time = time.perf_counter()

        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.detect_existing_checkpoint()
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint['checkpoint_id']}")
            # TODO: Load existing results and skip completed combinations

        results = []

        if parallel and len(parameter_combinations) > 1:
            # Parallel execution
            results = self._execute_parallel(parameter_combinations)
        else:
            # Sequential execution
            results = self._execute_sequential(parameter_combinations)

        execution_time = time.perf_counter() - start_time

        logger.info(
            f"Execution complete: {len(results)} combinations, "
            f"time={execution_time:.2f}s"
        )

        return results

    def _execute_sequential(
        self, parameter_combinations: list
    ) -> list[dict[str, Any]]:
        """Execute combinations sequentially.

        Args:
            parameter_combinations: List of ParameterSet objects

        Returns:
            List of combination result dictionaries
        """
        results = []

        for idx, combo in enumerate(parameter_combinations):
            logger.info(
                f"Combination {idx + 1}/{len(parameter_combinations)}: {combo.combination_id}"
            )

            result = self._execute_single_combination(combo)
            results.append(result)

            # Save checkpoint periodically
            if (idx + 1) % self.checkpoint_interval == 0:
                best_id = max(results, key=lambda r: r.get("avg_oos_win_rate", 0.0))[
                    "combination_id"
                ]
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_id=(idx + 1) // self.checkpoint_interval,
                    results=results,
                    best_combination_id=best_id,
                )

        return results

    def _execute_parallel(
        self, parameter_combinations: list
    ) -> list[dict[str, Any]]:
        """Execute combinations in parallel using multiprocessing.

        Args:
            parameter_combinations: List of ParameterSet objects

        Returns:
            List of combination result dictionaries
        """
        # Create partial function with validator
        execute_func = partial(self._execute_single_combination)

        # Use multiprocessing pool
        with mp.Pool(processes=self.num_workers) as pool:
            # Map combinations to workers
            results = pool.map(execute_func, parameter_combinations)

        return results

    def _execute_single_combination(self, params) -> dict[str, Any]:
        """Execute walk-forward validation for a single parameter combination.

        Args:
            params: ParameterSet object

        Returns:
            Dictionary with combination results
        """
        logger.info(f"Executing combination: {params.combination_id}")

        start_time = time.perf_counter()

        # Convert parameter set to walk-forward parameters format
        walk_forward_params = self._convert_to_walk_forward_params(params)

        try:
            # Execute walk-forward validation
            walk_forward_results = self.walk_forward_validator.execute_all_steps(
                steps=self.walk_forward_validator.window_manager.calculate_steps(),
                parameters=walk_forward_params,
            )

            # Extract metrics using result aggregator
            aggregated = self.result_aggregator.aggregate_results(walk_forward_results)

            execution_time = time.perf_counter() - start_time

            result = {
                "combination_id": params.combination_id,
                "parameters": walk_forward_params,
                "avg_oos_win_rate": aggregated.get("avg_oos_win_rate", 0.0),
                "avg_oos_profit_factor": aggregated.get("avg_oos_profit_factor", 0.0),
                "win_rate_std": aggregated.get("oos_consistency", 0.0),
                "max_drawdown": aggregated.get("max_drawdown", 0.0),
                "total_trades": aggregated.get("total_trades", 0),
                "execution_time": execution_time,
            }

            logger.info(
                f"Combination {params.combination_id} complete: "
                f"win_rate={result['avg_oos_win_rate']:.2%}"
            )

            return result

        except Exception as e:
            logger.error(f"Combination {params.combination_id} failed: {e}")
            execution_time = time.perf_counter() - start_time

            # Return result with zeros
            return {
                "combination_id": params.combination_id,
                "parameters": walk_forward_params,
                "avg_oos_win_rate": 0.0,
                "avg_oos_profit_factor": 0.0,
                "win_rate_std": 0.0,
                "max_drawdown": 1.0,
                "total_trades": 0,
                "execution_time": execution_time,
            }

    def _convert_to_walk_forward_params(self, params) -> dict[str, Any]:
        """Convert ParameterSet to walk-forward parameters format.

        Args:
            params: ParameterSet object

        Returns:
            Dictionary of walk-forward parameters
        """
        # Extract ensemble confidence threshold
        # In full implementation, would map all strategy parameters
        return {
            "confidence_threshold": params.ensemble_params.get(
                "confidence_threshold", 0.50
            ),
        }

    def calculate_progress(
        self, completed: int, total: int
    ) -> dict[str, float | str]:
        """Calculate progress metrics.

        Args:
            completed: Number of combinations completed
            total: Total number of combinations

        Returns:
            Dictionary with progress metrics
        """
        percent_complete = (completed / total * 100) if total > 0 else 0.0

        return {
            "completed": completed,
            "total": total,
            "percent_complete": percent_complete,
        }

    def save_results(
        self,
        results: list[dict[str, Any]],
        output_dir: str | Path = "data/reports",
    ) -> dict[str, Path]:
        """Save final results to multiple formats.

        Args:
            results: List of combination results
            output_dir: Output directory for files

        Returns:
            Dictionary mapping format names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find best combination
        ranked = self.result_aggregator.rank_combinations(results)
        best_combination = ranked[0]["combination_id"] if ranked else None

        # Save results
        paths = self.result_aggregator.save_final_results(
            results=results,
            best_combination=best_combination,
            output_dir=output_dir,
        )

        return paths

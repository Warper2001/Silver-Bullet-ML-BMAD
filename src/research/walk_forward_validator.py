"""Walk-Forward Validation Framework for Ensemble Backtesting.

This module implements walk-forward validation with rolling training/testing windows
to prevent overfitting and validate out-of-sample performance.
"""

import hashlib
import h5py
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.research.walk_forward_config import (
    WalkForwardConfig,
    WalkForwardStep,
    WalkForwardStepResult,
    WalkForwardSummary,
    WalkForwardResults,
)

logger = logging.getLogger(__name__)


class WindowManager:
    """Manages walk-forward window calculation and validation.

    Calculates rolling training/testing windows according to walk-forward methodology:
    - Training window: N months of historical data
    - Testing window: M months of forward out-of-sample data
    - Step-forward: Move both windows forward by K months after each test
    """

    def __init__(self, config: WalkForwardConfig):
        """Initialize window manager with configuration.

        Args:
            config: Walk-forward configuration
        """
        self.config = config
        logger.info(
            f"WindowManager initialized: train={config.training_window_months}mo, "
            f"test={config.testing_window_months}mo, "
            f"step={config.step_forward_months}mo"
        )

    def calculate_steps(self) -> list[WalkForwardStep]:
        """Calculate all walk-forward steps.

        Returns:
            List of WalkForwardStep objects with date ranges

        Example:
            For 2-year data with 6-month train, 1-month test, 1-month step:
            Step 1: Train [Jan-Jun], Test [Jul]
            Step 2: Train [Feb-Jul], Test [Aug]
            ...
            Step 13: Train [Jul-Dec], Test [Jan next year]
        """
        steps = []
        current_step = 1

        # Calculate initial test window start
        test_start = self._add_months(
            self.config.data_start_date,
            self.config.training_window_months,
        )

        # Keep generating steps while we have data
        while True:
            # Calculate test window end
            test_end = self._add_months(
                test_start,
                self.config.testing_window_months,
            ) - timedelta(days=1)

            # Check if test window exceeds data
            if test_end > self.config.data_end_date:
                logger.info(
                    f"Test window {test_start} to {test_end} exceeds data end "
                    f"{self.config.data_end_date}. Stopping at step {current_step - 1}."
                )
                break

            # Calculate training window for this step
            train_end = test_start - timedelta(days=1)
            train_start = self._add_months(
                train_end,
                -self.config.training_window_months,
            )

            # Check if training window starts before data
            if train_start < self.config.data_start_date:
                logger.warning(
                    f"Training window for step {current_step} would start before data. "
                    f"Adjusting train_start from {train_start} to {self.config.data_start_date}"
                )
                train_start = self.config.data_start_date

            # Create step
            step = WalkForwardStep(
                step_number=current_step,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            steps.append(step)

            logger.debug(
                f"Step {current_step}: Train [{train_start} to {train_end}], "
                f"Test [{test_start} to {test_end}]"
            )

            # Step forward for next iteration
            test_start = self._add_months(
                test_start,
                self.config.step_forward_months,
            )
            current_step += 1

            # Safety check to prevent infinite loop
            if current_step > 1000:
                logger.error("Exceeded maximum steps (1000). Breaking.")
                break

        logger.info(f"Calculated {len(steps)} walk-forward steps")
        return steps

    def validate_data_sufficiency(self) -> bool:
        """Validate that enough data exists for minimum steps.

        Returns:
            True if sufficient data, False otherwise
        """
        max_steps = self.config.calculate_max_steps()

        if max_steps < self.config.minimum_steps:
            logger.warning(
                f"Insufficient data: have {max_steps} steps, "
                f"need {self.config.minimum_steps} minimum steps"
            )
            return False

        logger.info(
            f"Data sufficiency validated: {max_steps} steps available, "
            f"{self.config.minimum_steps} required"
        )
        return True

    def get_max_possible_steps(self) -> int:
        """Calculate maximum number of steps possible with available data.

        Returns:
            Maximum number of walk-forward steps
        """
        return self.config.calculate_max_steps()

    @staticmethod
    def _add_months(start_date: date, months: int) -> date:
        """Add months to a date, handling month/year boundaries.

        Args:
            start_date: Starting date
            months: Number of months to add (can be negative)

        Returns:
            New date with months added
        """
        # Calculate year and month changes
        year_change = months // 12
        month_change = months % 12

        new_year = start_date.year + year_change
        new_month = start_date.month + month_change

        # Handle month overflow/underflow
        if new_month > 12:
            new_year += 1
            new_month -= 12
        elif new_month < 1:
            new_year -= 1
            new_month += 12

        # Keep same day of month
        new_day = min(start_date.day, [31, 29 if (new_year % 4 == 0 and new_year % 100 != 0) or (new_year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][new_month - 1])

        return date(new_year, new_month, new_day)


class LeakageReport:
    """Report of data leakage detection.

    Attributes:
        has_leakage: Whether leakage was detected
        leakage_details: List of leakage descriptions
        severity: Severity level (none, minor, major)
    """

    def __init__(
        self,
        has_leakage: bool = False,
        leakage_details: list[str] | None = None,
        severity: Literal["none", "minor", "major"] = "none",
    ):
        self.has_leakage = has_leakage
        self.leakage_details = leakage_details or []
        self.severity = severity

    def __repr__(self) -> str:
        return (
            f"LeakageReport(has_leakage={self.has_leakage}, "
            f"severity={self.severity}, details={len(self.leakage_details)})"
        )


class DataIsolationValidator:
    """Validates data isolation between training and testing windows.

    Ensures no data leakage occurs between train and test sets.
    """

    def __init__(self):
        """Initialize data isolation validator."""
        logger.info("DataIsolationValidator initialized")

    def validate_isolation(
        self,
        step: WalkForwardStep,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> bool:
        """Validate that training and testing data are properly isolated.

        Args:
            step: Walk-forward step configuration
            train_data: Training data DataFrame
            test_data: Testing data DataFrame

        Returns:
            True if isolation is valid, False otherwise
        """
        # Check temporal ordering
        if not self.check_temporal_ordering(train_data, test_data):
            logger.error(f"Step {step.step_number}: Temporal ordering violation")
            return False

        # Check for leakage
        leakage = self.detect_leakage(train_data, test_data)
        if leakage.has_leakage:
            logger.error(
                f"Step {step.step_number}: Data leakage detected - {leakage.leakage_details}"
            )
            return False

        logger.info(f"Step {step.step_number}: Data isolation validated")
        return True

    def check_temporal_ordering(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> bool:
        """Check that all training data timestamps are before testing data.

        Args:
            train_data: Training data DataFrame with datetime index
            test_data: Testing data DataFrame with datetime index

        Returns:
            True if temporal ordering is valid, False otherwise
        """
        # Ensure DataFrames have datetime indices
        if not isinstance(train_data.index, pd.DatetimeIndex):
            logger.warning("Train data does not have DatetimeIndex")
            return False

        if not isinstance(test_data.index, pd.DatetimeIndex):
            logger.warning("Test data does not have DatetimeIndex")
            return False

        # Check that max train timestamp < min test timestamp
        train_max = train_data.index.max()
        test_min = test_data.index.min()

        if train_max >= test_min:
            logger.error(
                f"Temporal ordering violation: train max ({train_max}) >= test min ({test_min})"
            )
            return False

        logger.debug(f"Temporal ordering valid: train < {train_max} < {test_min} < test")
        return True

    def detect_leakage(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> LeakageReport:
        """Detect data leakage between training and testing sets.

        Args:
            train_data: Training data DataFrame
            test_data: Testing data DataFrame

        Returns:
            LeakageReport with detection results
        """
        details = []
        has_leakage = False
        severity: Literal["none", "minor", "major"] = "none"

        # Check for overlapping indices
        train_indices = set(train_data.index)
        test_indices = set(test_data.index)
        overlap = train_indices.intersection(test_indices)

        if overlap:
            has_leakage = True
            severity = "major"
            details.append(f"Found {len(overlap)} overlapping timestamps")

        # Check for duplicate timestamps within each set
        train_duplicates = train_data.index[train_data.index.duplicated()]
        test_duplicates = test_data.index[test_data.index.duplicated()]

        if len(train_duplicates) > 0:
            has_leakage = True
            severity = "minor" if severity == "none" else severity
            details.append(f"Training data has {len(train_duplicates)} duplicate timestamps")

        if len(test_duplicates) > 0:
            has_leakage = True
            severity = "minor" if severity == "none" else severity
            details.append(f"Testing data has {len(test_duplicates)} duplicate timestamps")

        return LeakageReport(
            has_leakage=has_leakage, leakage_details=details, severity=severity
        )


class WalkForwardExecutor:
    """Executes walk-forward validation using ensemble backtester.

    Runs backtests on each walk-forward step and collects results.
    """

    def __init__(self, ensemble_backtester):
        """Initialize walk-forward executor.

        Args:
            ensemble_backtester: EnsembleBacktester instance
        """
        self.backtester = ensemble_backtester
        self.validator = DataIsolationValidator()
        logger.info("WalkForwardExecutor initialized")

    def execute_step(
        self,
        step: WalkForwardStep,
        parameters: dict,
    ) -> WalkForwardStepResult:
        """Execute a single walk-forward step.

        Args:
            step: Walk-forward step configuration
            parameters: Strategy parameters to use

        Returns:
            WalkForwardStepResult with in-sample and out-of-sample metrics
        """
        logger.info(
            f"Executing step {step.step_number}: "
            f"Train [{step.train_start} to {step.train_end}], "
            f"Test [{step.test_start} to {step.test_end}]"
        )

        # Load train data
        train_data = self._load_dollar_bars(step.train_start, step.train_end)

        # Load test data
        test_data = self._load_dollar_bars(step.test_start, step.test_end)

        # Update bar counts
        step.train_bars_count = len(train_data)
        step.test_bars_count = len(test_data)

        # Validate data isolation
        if not self.validator.validate_isolation(step, train_data, test_data):
            raise ValueError(f"Data isolation validation failed for step {step.step_number}")

        # Extract in-sample metrics from train period
        in_sample_results = self.backtester.run_backtest(
            start_date=step.train_start,
            end_date=step.train_end,
            confidence_threshold=parameters.get("confidence_threshold", 0.50),
        )

        # Extract out-of-sample metrics from test period
        out_of_sample_results = self.backtester.run_backtest(
            start_date=step.test_start,
            end_date=step.test_end,
            confidence_threshold=parameters.get("confidence_threshold", 0.50),
        )

        # Create step result
        result = WalkForwardStepResult(
            step=step,
            in_sample_metrics=self._extract_metrics(in_sample_results),
            out_of_sample_metrics=self._extract_metrics(out_of_sample_results),
            parameters=parameters,
            trade_counts={
                "in_sample": in_sample_results.total_trades,
                "out_of_sample": out_of_sample_results.total_trades,
            },
        )

        logger.info(
            f"Step {step.step_number} complete: "
            f"In-sample win_rate={result.in_sample_metrics.get('win_rate', 0):.2%}, "
            f"Out-of-sample win_rate={result.out_of_sample_metrics.get('win_rate', 0):.2%}"
        )

        return result

    def execute_all_steps(
        self,
        steps: list[WalkForwardStep],
        parameters: dict,
    ) -> WalkForwardResults:
        """Execute all walk-forward steps.

        Args:
            steps: List of walk-forward steps
            parameters: Strategy parameters to use

        Returns:
            WalkForwardResults with all step results and summary
        """
        logger.info(f"Executing {len(steps)} walk-forward steps")

        step_results = []

        for idx, step in enumerate(steps):
            try:
                logger.info(f"Step {idx + 1}/{len(steps)}")

                result = self.execute_step(step, parameters)
                step_results.append(result)

            except Exception as e:
                logger.error(f"Step {step.step_number} failed: {e}")
                # Continue with remaining steps
                continue

        # Calculate summary statistics
        summary = self._calculate_summary(step_results)

        # Create complete results
        from src.research.walk_forward_config import WalkForwardConfig

        # Create config if backtester doesn't have one
        if not hasattr(self.backtester, 'config') or self.backtester.config is None:
            config = WalkForwardConfig(
                data_start_date=date(2024, 1, 1),
                data_end_date=date(2025, 12, 31),
            )
        else:
            config = self.backtester.config

        results = WalkForwardResults(
            config=config,
            steps=step_results,
            summary=summary,
        )

        logger.info(
            f"Walk-forward complete: {len(step_results)} steps, "
            f"avg win_rate={summary.average_win_rate:.2%}, "
            f"avg profit_factor={summary.average_profit_factor:.2f}"
        )

        return results

    def _load_dollar_bars(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Load dollar bars for date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with dollar bars
        """
        return self.backtester._load_dollar_bars(start_date, end_date)

    @staticmethod
    def _extract_metrics(backtest_results) -> dict[str, float]:
        """Extract performance metrics from backtest results.

        Args:
            backtest_results: BacktestResults object

        Returns:
            Dictionary with performance metrics
        """
        return {
            "win_rate": backtest_results.win_rate,
            "profit_factor": backtest_results.profit_factor,
            "average_win": backtest_results.average_win,
            "average_loss": backtest_results.average_loss,
            "sharpe_ratio": backtest_results.sharpe_ratio,
            "max_drawdown": backtest_results.max_drawdown,
            "total_trades": backtest_results.total_trades,
            "total_pnl": backtest_results.total_pnl,
        }

    @staticmethod
    def _calculate_summary(step_results: list[WalkForwardStepResult]) -> WalkForwardSummary:
        """Calculate summary statistics across all steps.

        Args:
            step_results: List of step results

        Returns:
            WalkForwardSummary with aggregate statistics
        """
        if not step_results:
            return WalkForwardSummary(total_steps=0)

        # Extract out-of-sample win rates
        oos_win_rates = [
            s.out_of_sample_metrics.get("win_rate", 0.0) for s in step_results
        ]

        # Extract out-of-sample profit factors
        oos_profit_factors = [
            s.out_of_sample_metrics.get("profit_factor", 0.0) for s in step_results
        ]

        # Extract in-sample metrics for correlation
        is_win_rates = [
            s.in_sample_metrics.get("win_rate", 0.0) for s in step_results
        ]

        # Calculate statistics
        import numpy as np

        avg_win_rate = float(np.mean(oos_win_rates))
        std_win_rate = float(np.std(oos_win_rates))
        avg_profit_factor = float(np.mean(oos_profit_factors))

        # Find best and worst steps
        best_step_idx = int(np.argmax(oos_win_rates))
        worst_step_idx = int(np.argmin(oos_win_rates))

        # Calculate correlation between in-sample and out-of-sample
        if len(is_win_rates) > 1 and len(oos_win_rates) > 1:
            try:
                correlation_result = stats.pearsonr(is_win_rates, oos_win_rates)
                correlation = float(correlation_result[0]) if not correlation_result[0] != correlation_result[0] else 0.0  # Check for NaN
            except (ValueError, RuntimeWarning):
                # Handle case where arrays are constant or have other issues
                correlation = 0.0
        else:
            correlation = 0.0

        # Total trades
        total_trades = sum(
            s.trade_counts.get("out_of_sample", 0) for s in step_results
        )

        return WalkForwardSummary(
            total_steps=len(step_results),
            average_win_rate=avg_win_rate,
            std_win_rate=std_win_rate,
            average_profit_factor=avg_profit_factor,
            best_step=step_results[best_step_idx].step.step_number,
            worst_step=step_results[worst_step_idx].step.step_number,
            in_sample_out_of_sample_correlation=float(correlation),
            total_trades=total_trades,
        )


class WalkForwardResultsStorage:
    """Handles persistence of walk-forward results to HDF5 format.

    Provides save/load functionality with checkpoint/resume support.
    """

    def __init__(self, base_dir: str | Path = "data/reports"):
        """Initialize results storage.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WalkForwardResultsStorage initialized with base_dir: {self.base_dir}")

    def save_results(self, results: WalkForwardResults, path: str | Path | None = None) -> Path:
        """Save walk-forward results to HDF5 file.

        Args:
            results: WalkForwardResults to save
            path: Optional file path (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if path is None:
            timestamp = results.timestamp or date.today().isoformat()
            path = self.base_dir / f"walkforward_{timestamp}.h5"
        else:
            path = Path(path)

        logger.info(f"Saving walk-forward results to {path}")

        with h5py.File(path, "w") as f:
            # Save configuration as attributes
            config_group = f.create_group("config")
            config_group.attrs["training_window_months"] = results.config.training_window_months
            config_group.attrs["testing_window_months"] = results.config.testing_window_months
            config_group.attrs["step_forward_months"] = results.config.step_forward_months
            config_group.attrs["minimum_steps"] = results.config.minimum_steps
            config_group.attrs["data_start_date"] = results.config.data_start_date.isoformat()
            config_group.attrs["data_end_date"] = results.config.data_end_date.isoformat()

            # Save metadata
            f.attrs["timestamp"] = results.timestamp
            f.attrs["total_steps"] = len(results.steps)

            # Save each step
            steps_group = f.create_group("steps")
            for step_result in results.steps:
                step_number = step_result.step.step_number
                step_group = steps_group.create_group(f"step_{step_number:03d}")

                # Save step info
                step_group.attrs["step_number"] = step_result.step.step_number
                step_group.attrs["train_start"] = step_result.step.train_start.isoformat()
                step_group.attrs["train_end"] = step_result.step.train_end.isoformat()
                step_group.attrs["test_start"] = step_result.step.test_start.isoformat()
                step_group.attrs["test_end"] = step_result.step.test_end.isoformat()
                step_group.attrs["train_bars_count"] = step_result.step.train_bars_count
                step_group.attrs["test_bars_count"] = step_result.step.test_bars_count

                # Save in-sample metrics
                if step_result.in_sample_metrics:
                    is_metrics = step_result.in_sample_metrics
                    step_group.create_dataset(
                        "in_sample_metrics",
                        data=np.array(list(is_metrics.values())),
                    )
                    step_group["in_sample_metrics"].attrs["keys"] = json.dumps(list(is_metrics.keys()))

                # Save out-of-sample metrics
                if step_result.out_of_sample_metrics:
                    oos_metrics = step_result.out_of_sample_metrics
                    step_group.create_dataset(
                        "out_of_sample_metrics",
                        data=np.array(list(oos_metrics.values())),
                    )
                    step_group["out_of_sample_metrics"].attrs["keys"] = json.dumps(list(oos_metrics.keys()))

                # Save parameters
                if step_result.parameters:
                    params_json = json.dumps(step_result.parameters, default=str)
                    step_group.attrs["parameters"] = params_json

                # Save trade counts
                if step_result.trade_counts:
                    tc = step_result.trade_counts
                    step_group.attrs["in_sample_trades"] = tc.get("in_sample", 0)
                    step_group.attrs["out_of_sample_trades"] = tc.get("out_of_sample", 0)

            # Save summary
            if results.summary:
                summary_group = f.create_group("summary")
                summary_group.attrs["total_steps"] = results.summary.total_steps
                summary_group.attrs["average_win_rate"] = results.summary.average_win_rate
                summary_group.attrs["std_win_rate"] = results.summary.std_win_rate
                summary_group.attrs["average_profit_factor"] = results.summary.average_profit_factor
                summary_group.attrs["best_step"] = results.summary.best_step
                summary_group.attrs["worst_step"] = results.summary.worst_step
                summary_group.attrs["in_sample_out_of_sample_correlation"] = results.summary.in_sample_out_of_sample_correlation
                summary_group.attrs["total_trades"] = results.summary.total_trades

        logger.info(f"Results saved successfully to {path}")
        return path

    def load_results(self, path: str | Path) -> WalkForwardResults:
        """Load walk-forward results from HDF5 file.

        Args:
            path: Path to HDF5 file

        Returns:
            WalkForwardResults object
        """
        path = Path(path)
        logger.info(f"Loading walk-forward results from {path}")

        with h5py.File(path, "r") as f:
            # Load configuration
            config_group = f["config"]
            config = WalkForwardConfig(
                training_window_months=config_group.attrs["training_window_months"],
                testing_window_months=config_group.attrs["testing_window_months"],
                step_forward_months=config_group.attrs["step_forward_months"],
                minimum_steps=config_group.attrs["minimum_steps"],
                data_start_date=date.fromisoformat(config_group.attrs["data_start_date"]),
                data_end_date=date.fromisoformat(config_group.attrs["data_end_date"]),
            )

            # Load steps
            steps = []
            steps_group = f["steps"]
            for step_name in sorted(steps_group.keys()):
                step_group = steps_group[step_name]

                # Recreate step
                step = WalkForwardStep(
                    step_number=step_group.attrs["step_number"],
                    train_start=date.fromisoformat(step_group.attrs["train_start"]),
                    train_end=date.fromisoformat(step_group.attrs["train_end"]),
                    test_start=date.fromisoformat(step_group.attrs["test_start"]),
                    test_end=date.fromisoformat(step_group.attrs["test_end"]),
                    train_bars_count=step_group.attrs["train_bars_count"],
                    test_bars_count=step_group.attrs["test_bars_count"],
                )

                # Load in-sample metrics
                in_sample_metrics = {}
                if "in_sample_metrics" in step_group:
                    keys = json.loads(step_group["in_sample_metrics"].attrs["keys"])
                    values = step_group["in_sample_metrics"][:]
                    in_sample_metrics = dict(zip(keys, values))

                # Load out-of-sample metrics
                out_of_sample_metrics = {}
                if "out_of_sample_metrics" in step_group:
                    keys = json.loads(step_group["out_of_sample_metrics"].attrs["keys"])
                    values = step_group["out_of_sample_metrics"][:]
                    out_of_sample_metrics = dict(zip(keys, values))

                # Load parameters
                parameters = {}
                if "parameters" in step_group.attrs:
                    parameters = json.loads(step_group.attrs["parameters"])

                # Load trade counts
                trade_counts = {
                    "in_sample": step_group.attrs.get("in_sample_trades", 0),
                    "out_of_sample": step_group.attrs.get("out_of_sample_trades", 0),
                }

                step_result = WalkForwardStepResult(
                    step=step,
                    in_sample_metrics=in_sample_metrics,
                    out_of_sample_metrics=out_of_sample_metrics,
                    parameters=parameters,
                    trade_counts=trade_counts,
                )
                steps.append(step_result)

            # Load summary
            summary = None
            if "summary" in f:
                summary_group = f["summary"]
                summary = WalkForwardSummary(
                    total_steps=summary_group.attrs["total_steps"],
                    average_win_rate=summary_group.attrs["average_win_rate"],
                    std_win_rate=summary_group.attrs["std_win_rate"],
                    average_profit_factor=summary_group.attrs["average_profit_factor"],
                    best_step=summary_group.attrs["best_step"],
                    worst_step=summary_group.attrs["worst_step"],
                    in_sample_out_of_sample_correlation=summary_group.attrs["in_sample_out_of_sample_correlation"],
                    total_trades=summary_group.attrs["total_trades"],
                )

            timestamp = f.attrs.get("timestamp", date.today().isoformat())

        results = WalkForwardResults(
            config=config,
            steps=steps,
            summary=summary,
            timestamp=timestamp,
        )

        logger.info(f"Loaded {len(steps)} steps from {path}")
        return results

    def append_step_result(
        self, step_result: WalkForwardStepResult, path: str | Path
    ) -> None:
        """Append a single step result to existing HDF5 file (checkpointing).

        Args:
            step_result: Step result to append
            path: Path to HDF5 file
        """
        path = Path(path)
        logger.info(f"Appending step {step_result.step.step_number} to {path}")

        with h5py.File(path, "a") as f:
            # Get or create steps group
            if "steps" not in f:
                steps_group = f.create_group("steps")
            else:
                steps_group = f["steps"]

            # Create step group
            step_number = step_result.step.step_number
            step_name = f"step_{step_number:03d}"

            # Skip if already exists
            if step_name in steps_group:
                logger.warning(f"Step {step_number} already exists in {path}, skipping")
                return

            step_group = steps_group.create_group(step_name)

            # Save step data (same logic as save_results)
            step_group.attrs["step_number"] = step_result.step.step_number
            step_group.attrs["train_start"] = step_result.step.train_start.isoformat()
            step_group.attrs["train_end"] = step_result.step.train_end.isoformat()
            step_group.attrs["test_start"] = step_result.step.test_start.isoformat()
            step_group.attrs["test_end"] = step_result.step.test_end.isoformat()
            step_group.attrs["train_bars_count"] = step_result.step.train_bars_count
            step_group.attrs["test_bars_count"] = step_result.step.test_bars_count

            if step_result.in_sample_metrics:
                is_metrics = step_result.in_sample_metrics
                step_group.create_dataset(
                    "in_sample_metrics",
                    data=np.array(list(is_metrics.values())),
                )
                step_group["in_sample_metrics"].attrs["keys"] = json.dumps(list(is_metrics.keys()))

            if step_result.out_of_sample_metrics:
                oos_metrics = step_result.out_of_sample_metrics
                step_group.create_dataset(
                    "out_of_sample_metrics",
                    data=np.array(list(oos_metrics.values())),
                )
                step_group["out_of_sample_metrics"].attrs["keys"] = json.dumps(list(oos_metrics.keys()))

            if step_result.parameters:
                params_json = json.dumps(step_result.parameters, default=str)
                step_group.attrs["parameters"] = params_json

            if step_result.trade_counts:
                tc = step_result.trade_counts
                step_group.attrs["in_sample_trades"] = tc.get("in_sample", 0)
                step_group.attrs["out_of_sample_trades"] = tc.get("out_of_sample", 0)

            # Update total steps count
            f.attrs["total_steps"] = len(steps_group)

        logger.info(f"Step {step_number} appended successfully")

    def calculate_data_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data integrity validation.

        Args:
            data: DataFrame to calculate checksum for

        Returns:
            Hexadecimal checksum string
        """
        # Use hash of DataFrame bytes
        data_bytes = data.to_csv().encode()
        return hashlib.sha256(data_bytes).hexdigest()


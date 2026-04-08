"""Walk-Forward Validation Configuration Models.

This module defines Pydantic models for configuring and storing walk-forward
validation results for ensemble backtesting.
"""

import logging
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation windows.

    Attributes:
        training_window_months: Size of training window in months (default: 6)
        testing_window_months: Size of testing window in months (default: 1)
        step_forward_months: Months to step forward after each test (default: 1)
        minimum_steps: Minimum number of walk-forward steps required (default: 12)
        data_start_date: Start date of available data
        data_end_date: End date of available data
    """

    training_window_months: int = Field(
        default=6, gt=0, description="Training window size in months"
    )
    testing_window_months: int = Field(
        default=1, gt=0, description="Testing window size in months"
    )
    step_forward_months: int = Field(
        default=1, gt=0, description="Step-forward size in months"
    )
    minimum_steps: int = Field(
        default=12, gt=0, description="Minimum number of walk-forward steps"
    )
    data_start_date: date = Field(..., description="Start date of available data")
    data_end_date: date = Field(..., description="End date of available data")

    @field_validator("step_forward_months")
    @classmethod
    def step_forward_not_greater_than_testing(
        cls, v: int, info
    ) -> int:
        """Validate that step_forward_months is not greater than testing_window_months."""
        if "testing_window_months" in info.data and v > info.data["testing_window_months"]:
            raise ValueError(
                f"step_forward_months ({v}) must be <= testing_window_months "
                f"({info.data['testing_window_months']})"
            )
        return v

    @model_validator(mode="after")
    def validate_data_sufficiency(self) -> "WalkForwardConfig":
        """Validate that enough data exists for minimum steps."""
        total_months = self._calculate_total_months()

        # Required months = (training + testing) * minimum_steps
        required_months = (
            self.training_window_months + self.testing_window_months
        ) * self.minimum_steps

        if total_months < required_months:
            logger.warning(
                f"Insufficient data for {self.minimum_steps} steps: "
                f"have {total_months} months, need {required_months} months. "
                f"Maximum possible steps: {self.calculate_max_steps()}"
            )

        return self

    def _calculate_total_months(self) -> int:
        """Calculate total months in dataset."""
        years = self.data_end_date.year - self.data_start_date.year
        months = self.data_end_date.month - self.data_start_date.month
        return years * 12 + months + 1

    def calculate_max_steps(self) -> int:
        """Calculate maximum number of walk-forward steps possible with available data."""
        total_months = self._calculate_total_months()

        # After initial training window, we can step forward
        # Each step moves us forward by step_forward_months
        # We need training_window_months + testing_window_months for first step
        available_for_steps = total_months - self.training_window_months

        if available_for_steps < self.testing_window_months:
            return 0

        max_steps = (
            available_for_steps - self.testing_window_months
        ) // self.step_forward_months + 1

        return max_steps

    def calculate_initial_test_start(self) -> date:
        """Calculate the start date of the first testing window."""
        from datetime import timedelta

        # Add training_window_months to data_start_date
        years = self.training_window_months // 12
        months = self.training_window_months % 12

        test_start = self.data_start_date
        test_start = test_start.replace(year=test_start.year + years)
        test_start = test_start.replace(
            month=min(12, test_start.month + months)
            if test_start.month + months <= 12
            else (test_start.month + months) % 12 or 12
        )

        # Handle month overflow
        if self.data_start_date.month + months > 12:
            test_start = test_start.replace(year=test_start.year + 1)

        return test_start


class WalkForwardStep(BaseModel):
    """Represents a single walk-forward step with train/test windows.

    Attributes:
        step_number: Step number (1-indexed)
        train_start: Training window start date
        train_end: Training window end date (inclusive)
        test_start: Testing window start date
        test_end: Testing window end date (inclusive)
        train_bars_count: Number of bars in training window
        test_bars_count: Number of bars in testing window
    """

    step_number: int = Field(..., ge=1, description="Step number (1-indexed)")
    train_start: date = Field(..., description="Training window start date")
    train_end: date = Field(..., description="Training window end date")
    test_start: date = Field(..., description="Testing window start date")
    test_end: date = Field(..., description="Testing window end date")
    train_bars_count: int = Field(
        default=0, ge=0, description="Number of bars in training window"
    )
    test_bars_count: int = Field(
        default=0, ge=0, description="Number of bars in testing window"
    )

    @model_validator(mode="after")
    def validate_temporal_ordering(self) -> "WalkForwardStep":
        """Validate that train_end < test_start (no overlap)."""
        if self.train_end >= self.test_start:
            raise ValueError(
                f"Train window ({self.train_start} to {self.train_end}) "
                f"must end before test window ({self.test_start} to {self.test_end}) starts"
            )
        return self


class WalkForwardStepResult(BaseModel):
    """Results from a single walk-forward step.

    Attributes:
        step: The walk-forward step configuration
        in_sample_metrics: Performance metrics on training data (win_rate, profit_factor, etc.)
        out_of_sample_metrics: Performance metrics on testing data
        parameters: Strategy parameters used in this step
        trade_counts: Dictionary with in_sample and out_of_sample trade counts
    """

    step: WalkForwardStep = Field(..., description="Walk-forward step configuration")
    in_sample_metrics: dict[str, float] = Field(
        default_factory=dict, description="In-sample performance metrics"
    )
    out_of_sample_metrics: dict[str, float] = Field(
        default_factory=dict, description="Out-of-sample performance metrics"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters used"
    )
    trade_counts: dict[str, int] = Field(
        default_factory=lambda: {"in_sample": 0, "out_of_sample": 0},
        description="Trade counts for in-sample and out-of-sample",
    )


class WalkForwardSummary(BaseModel):
    """Summary statistics across all walk-forward steps.

    Attributes:
        total_steps: Total number of steps executed
        average_win_rate: Average out-of-sample win rate
        std_win_rate: Standard deviation of win rates
        average_profit_factor: Average out-of-sample profit factor
        best_step: Step number with best out-of-sample performance
        worst_step: Step number with worst out-of-sample performance
        in_sample_out_of_sample_correlation: Correlation between in-sample and out-of-sample performance
        total_trades: Total trades across all steps
    """

    total_steps: int = Field(..., ge=0, description="Total number of steps executed")
    average_win_rate: float = Field(
        default=0.0, ge=0, le=1, description="Average out-of-sample win rate"
    )
    std_win_rate: float = Field(
        default=0.0, ge=0, description="Standard deviation of win rates"
    )
    average_profit_factor: float = Field(
        default=0.0, ge=0, description="Average out-of-sample profit factor"
    )
    best_step: int = Field(
        default=1, ge=1, description="Step number with best performance"
    )
    worst_step: int = Field(
        default=1, ge=1, description="Step number with worst performance"
    )
    in_sample_out_of_sample_correlation: float = Field(
        default=0.0, ge=-1, le=1, description="In-sample vs out-of-sample correlation"
    )
    total_trades: int = Field(
        default=0, ge=0, description="Total trades across all steps"
    )


class WalkForwardResults(BaseModel):
    """Complete results from walk-forward validation.

    Attributes:
        config: Walk-forward configuration used
        steps: List of results from each step
        summary: Aggregate statistics across all steps
        timestamp: When the walk-forward was run
    """

    config: WalkForwardConfig = Field(..., description="Walk-forward configuration")
    steps: list[WalkForwardStepResult] = Field(
        default_factory=list, description="Results from each step"
    )
    summary: WalkForwardSummary | None = Field(
        default=None, description="Aggregate statistics"
    )
    timestamp: str = Field(
        default_factory=lambda: date.today().isoformat(),
        description="When results were generated",
    )

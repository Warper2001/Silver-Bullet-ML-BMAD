"""Unit tests for walk-forward configuration models.

Tests configuration validation, window calculations, and step boundaries.
"""

import pytest
from datetime import date

from src.research.walk_forward_config import (
    WalkForwardConfig,
    WalkForwardStep,
    WalkForwardStepResult,
    WalkForwardSummary,
    WalkForwardResults,
)


class TestWalkForwardConfig:
    """Test WalkForwardConfig model."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        assert config.training_window_months == 6
        assert config.testing_window_months == 1
        assert config.step_forward_months == 1
        assert config.minimum_steps == 12

    def test_calculate_total_months_1_year(self):
        """Test total months calculation for 1 year."""
        config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2024, 12, 31),
        )

        total_months = config._calculate_total_months()
        assert total_months == 12

    def test_calculate_total_months_2_years(self):
        """Test total months calculation for 2 years."""
        config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        total_months = config._calculate_total_months()
        assert total_months == 24

    def test_calculate_max_steps_2_years(self):
        """Test max steps calculation for 2-year dataset."""
        config = WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        max_steps = config.calculate_max_steps()
        # 24 months - 6 months training = 18 months available
        # First test starts at month 7, then step 1 month each time
        # Should be able to do approximately 13-18 steps
        assert max_steps >= 13

    def test_calculate_max_steps_insufficient_data(self):
        """Test max steps with insufficient data."""
        config = WalkForwardConfig(
            training_window_months=6,
            testing_window_months=1,
            step_forward_months=1,
            minimum_steps=12,
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2024, 6, 30),  # Only 6 months
        )

        max_steps = config.calculate_max_steps()
        assert max_steps == 0

    def test_step_forward_not_greater_than_testing(self):
        """Test validation that step_forward <= testing_window."""
        with pytest.raises(ValueError, match="step_forward_months.*must be <="):
            WalkForwardConfig(
                training_window_months=6,
                testing_window_months=1,
                step_forward_months=2,  # Greater than testing
                minimum_steps=12,
                data_start_date=date(2024, 1, 1),
                data_end_date=date(2025, 12, 31),
            )

    def test_negative_window_months_raises_error(self):
        """Test that negative window months are rejected."""
        with pytest.raises(ValueError):
            WalkForwardConfig(
                training_window_months=-1,
                data_start_date=date(2024, 1, 1),
                data_end_date=date(2025, 12, 31),
            )

    def test_zero_minimum_steps_raises_error(self):
        """Test that zero minimum steps is rejected."""
        with pytest.raises(ValueError):
            WalkForwardConfig(
                minimum_steps=0,
                data_start_date=date(2024, 1, 1),
                data_end_date=date(2025, 12, 31),
            )

    def test_data_sufficiency_warning_logged(self, caplog):
        """Test that insufficient data logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = WalkForwardConfig(
                training_window_months=6,
                testing_window_months=1,
                step_forward_months=1,
                minimum_steps=12,
                data_start_date=date(2024, 1, 1),
                data_end_date=date(2024, 12, 31),  # Only 12 months
            )

        # Should have warning about insufficient data
        assert "Insufficient data" in caplog.text or "have 12 months" in caplog.text


class TestWalkForwardStep:
    """Test WalkForwardStep model."""

    def test_valid_step(self):
        """Test creating a valid walk-forward step."""
        step = WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
            train_bars_count=1000,
            test_bars_count=200,
        )

        assert step.step_number == 1
        assert step.train_bars_count == 1000
        assert step.test_bars_count == 200

    def test_temporal_ordering_validation_fails_on_overlap(self):
        """Test that overlapping train/test windows are rejected."""
        with pytest.raises(ValueError, match="must end before test window"):
            WalkForwardStep(
                step_number=1,
                train_start=date(2024, 1, 1),
                train_end=date(2024, 7, 15),  # Overlaps with test
                test_start=date(2024, 7, 1),
                test_end=date(2024, 7, 31),
            )

    def test_temporal_ordering_validation_fails_on_train_after_test(self):
        """Test that train ending after test starts is rejected."""
        with pytest.raises(ValueError, match="must end before test window"):
            WalkForwardStep(
                step_number=1,
                train_start=date(2024, 8, 1),
                train_end=date(2024, 9, 1),
                test_start=date(2024, 7, 1),
                test_end=date(2024, 7, 31),
            )

    def test_temporal_ordering_passes_on_valid_separation(self):
        """Test that valid temporal ordering passes validation."""
        step = WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),  # Strictly after train_end
            test_end=date(2024, 7, 31),
        )

        assert step.train_end < step.test_start

    def test_negative_bar_counts_rejected(self):
        """Test that negative bar counts are rejected."""
        with pytest.raises(ValueError):
            WalkForwardStep(
                step_number=1,
                train_start=date(2024, 1, 1),
                train_end=date(2024, 6, 30),
                test_start=date(2024, 7, 1),
                test_end=date(2024, 7, 31),
                train_bars_count=-1,
            )


class TestWalkForwardStepResult:
    """Test WalkForwardStepResult model."""

    def test_step_result_creation(self):
        """Test creating a step result with metrics."""
        step = WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
        )

        result = WalkForwardStepResult(
            step=step,
            in_sample_metrics={"win_rate": 0.6, "profit_factor": 1.5},
            out_of_sample_metrics={"win_rate": 0.55, "profit_factor": 1.3},
            parameters={"confidence_threshold": 0.5},
            trade_counts={"in_sample": 100, "out_of_sample": 20},
        )

        assert result.in_sample_metrics["win_rate"] == 0.6
        assert result.out_of_sample_metrics["win_rate"] == 0.55
        assert result.trade_counts["in_sample"] == 100
        assert result.trade_counts["out_of_sample"] == 20

    def test_default_values(self):
        """Test default values for optional fields."""
        step = WalkForwardStep(
            step_number=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 7, 31),
        )

        result = WalkForwardStepResult(step=step)

        assert result.in_sample_metrics == {}
        assert result.out_of_sample_metrics == {}
        assert result.parameters == {}
        assert result.trade_counts == {"in_sample": 0, "out_of_sample": 0}


class TestWalkForwardSummary:
    """Test WalkForwardSummary model."""

    def test_summary_creation(self):
        """Test creating a summary with statistics."""
        summary = WalkForwardSummary(
            total_steps=12,
            average_win_rate=0.58,
            std_win_rate=0.05,
            average_profit_factor=1.4,
            best_step=3,
            worst_step=7,
            in_sample_out_of_sample_correlation=0.3,
            total_trades=500,
        )

        assert summary.total_steps == 12
        assert summary.average_win_rate == 0.58
        assert summary.best_step == 3
        assert summary.worst_step == 7

    def test_default_values(self):
        """Test default values for optional fields."""
        summary = WalkForwardSummary(total_steps=0)

        assert summary.average_win_rate == 0.0
        assert summary.std_win_rate == 0.0
        assert summary.best_step == 1
        assert summary.worst_step == 1
        assert summary.total_trades == 0


class TestWalkForwardResults:
    """Test WalkForwardResults model."""

    def test_results_creation(self):
        """Test creating complete walk-forward results."""
        config = WalkForwardConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2025, 12, 31),
        )

        summary = WalkForwardSummary(total_steps=12)

        results = WalkForwardResults(
            config=config,
            steps=[],
            summary=summary,
        )

        assert results.config == config
        assert results.steps == []
        assert results.summary.total_steps == 12

    def test_default_timestamp(self):
        """Test that timestamp defaults to today."""
        results = WalkForwardResults(
            config=WalkForwardConfig(
                data_start_date=date(2024, 1, 1),
                data_end_date=date(2025, 12, 31),
            ),
        )

        # Timestamp should be ISO format date string
        assert "-" in results.timestamp

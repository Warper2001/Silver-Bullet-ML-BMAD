"""Tests for OptimalConfigurationSelector.

Tests the multi-criteria configuration selection system for choosing
optimal ensemble parameters from walk-forward validation results.
"""

import logging
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.research.optimal_config_selector import (
    CompositeScore,
    ConfigurationComparison,
    OptimalConfigurationSelector,
    PrimaryCriteria,
    SelectionCriteria,
    SelectionReport,
    TopConfigurations,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_walk_forward_results(tmp_path):
    """Create sample walk-forward results for testing.

    Returns:
        Path to HDF5 file with sample results
    """
    # Create HDF5 file with sample walk-forward results
    hdf5_path = tmp_path / "walkforward_results.h5"

    with h5py.File(hdf5_path, "w") as f:
        # Configuration 1: Good performance, moderate stability
        combo1 = f.create_group("config_001")
        combo1.attrs["combination_id"] = "config_001"
        combo1.attrs["avg_oos_win_rate"] = 0.62
        combo1.attrs["avg_oos_profit_factor"] = 2.1
        combo1.attrs["win_rate_std"] = 0.08
        combo1.attrs["max_drawdown"] = 0.12
        combo1.attrs["total_trades"] = 150
        combo1.attrs["param_confidence_threshold"] = 0.55
        combo1.attrs["parameter_stability_score"] = 0.75
        combo1.attrs["performance_stability"] = 0.72

        # Configuration 2: Excellent performance, low stability
        combo2 = f.create_group("config_002")
        combo2.attrs["combination_id"] = "config_002"
        combo2.attrs["avg_oos_win_rate"] = 0.68
        combo2.attrs["avg_oos_profit_factor"] = 2.5
        combo2.attrs["win_rate_std"] = 0.15
        combo2.attrs["max_drawdown"] = 0.18
        combo2.attrs["total_trades"] = 180
        combo2.attrs["param_confidence_threshold"] = 0.60
        combo2.attrs["parameter_stability_score"] = 0.55
        combo2.attrs["performance_stability"] = 0.50

        # Configuration 3: Moderate performance, high stability
        combo3 = f.create_group("config_003")
        combo3.attrs["combination_id"] = "config_003"
        combo3.attrs["avg_oos_win_rate"] = 0.58
        combo3.attrs["avg_oos_profit_factor"] = 1.8
        combo3.attrs["win_rate_std"] = 0.05
        combo3.attrs["max_drawdown"] = 0.08
        combo3.attrs["total_trades"] = 120
        combo3.attrs["param_confidence_threshold"] = 0.50
        combo3.attrs["parameter_stability_score"] = 0.85
        combo3.attrs["performance_stability"] = 0.80

        # Configuration 4: Fails primary criteria (low win rate)
        combo4 = f.create_group("config_004")
        combo4.attrs["combination_id"] = "config_004"
        combo4.attrs["avg_oos_win_rate"] = 0.52  # Below 55% threshold
        combo4.attrs["avg_oos_profit_factor"] = 1.6
        combo4.attrs["win_rate_std"] = 0.10
        combo4.attrs["max_drawdown"] = 0.14
        combo4.attrs["total_trades"] = 100
        combo4.attrs["param_confidence_threshold"] = 0.45
        combo4.attrs["parameter_stability_score"] = 0.70
        combo4.attrs["performance_stability"] = 0.65

    return hdf5_path


@pytest.fixture
def sample_summary_csv(tmp_path, sample_walk_forward_results):
    """Create sample summary CSV with walk-forward results.

    Returns:
        Path to CSV file with summary metrics
    """
    csv_path = tmp_path / "walkforward_summary.csv"

    data = {
        "combination_id": ["config_001", "config_002", "config_003", "config_004"],
        "avg_oos_win_rate": [0.62, 0.68, 0.58, 0.52],
        "avg_oos_profit_factor": [2.1, 2.5, 1.8, 1.6],
        "win_rate_std": [0.08, 0.15, 0.05, 0.10],
        "max_drawdown": [0.12, 0.18, 0.08, 0.14],
        "total_trades": [150, 180, 120, 100],
        "parameter_stability_score": [0.75, 0.55, 0.85, 0.70],
        "performance_stability": [0.72, 0.50, 0.80, 0.65],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def default_criteria():
    """Get default selection criteria."""
    return SelectionCriteria(
        min_win_rate=0.55,
        min_profit_factor=1.5,
        max_drawdown=0.15,
        min_trade_frequency=3.0,
        max_win_rate_std=0.10,
    )


class TestOptimalConfigurationSelector:
    """Test suite for OptimalConfigurationSelector."""

    def test_initialization(self, sample_walk_forward_results, sample_summary_csv):
        """Test selector initialization with valid paths."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
        )

        assert selector.hdf5_path == sample_walk_forward_results
        assert selector.summary_csv_path == sample_summary_csv
        assert selector.criteria is not None
        assert len(selector.configurations) == 0  # Not loaded yet

    def test_load_results(self, sample_walk_forward_results, sample_summary_csv):
        """Test loading walk-forward results from files."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
        )

        selector.load_results()

        # Should load all 4 configurations
        assert len(selector.configurations) == 4

        # Check first configuration
        config = selector.configurations["config_001"]
        assert config["avg_oos_win_rate"] == 0.62
        assert config["avg_oos_profit_factor"] == 2.1
        assert config["max_drawdown"] == 0.12

    def test_validate_primary_criteria_pass(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test primary criteria validation for passing configuration."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Config 001 passes all criteria
        result = selector.validate_primary_criteria(
            selector.configurations["config_001"]
        )

        assert result.passes is True
        assert result.win_rate_pass is True
        assert result.profit_factor_pass is True
        assert result.drawdown_pass is True
        assert result.trade_frequency_pass is True
        assert result.consistency_pass is True

    def test_validate_primary_criteria_fail_win_rate(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test primary criteria validation fails on low win rate."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Config 004 fails win rate threshold (52% < 55%)
        result = selector.validate_primary_criteria(
            selector.configurations["config_004"]
        )

        assert result.passes is False
        assert result.win_rate_pass is False
        assert result.profit_factor_pass is True
        assert result.drawdown_pass is True

    def test_validate_primary_criteria_fail_drawdown(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test primary criteria validation fails on high drawdown."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Config 002 fails drawdown threshold (18% > 15%)
        result = selector.validate_primary_criteria(
            selector.configurations["config_002"]
        )

        assert result.passes is False
        assert result.win_rate_pass is True
        assert result.profit_factor_pass is True
        assert result.drawdown_pass is False

    def test_calculate_secondary_criteria_scores(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test calculation of normalized secondary criteria scores."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Calculate scores for config 001
        scores = selector.calculate_secondary_scores(
            selector.configurations["config_001"]
        )

        # All scores should be in [0, 1]
        assert 0.0 <= scores.performance_score <= 1.0
        assert 0.0 <= scores.stability_score <= 1.0
        assert 0.0 <= scores.risk_score <= 1.0
        assert 0.0 <= scores.frequency_score <= 1.0

        # Composite score should be weighted sum
        expected_composite = (
            0.4 * scores.performance_score
            + 0.3 * scores.stability_score
            + 0.2 * scores.risk_score
            + 0.1 * scores.frequency_score
        )
        assert abs(scores.composite_score - expected_composite) < 0.01

    def test_calculate_composite_score_all_configs(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test composite score calculation across all configurations."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Calculate composite scores for all configurations
        composite_scores = {}
        for config_id, config in selector.configurations.items():
            scores = selector.calculate_secondary_scores(config)
            composite_scores[config_id] = scores.composite_score

        # All scores should be valid
        for config_id, score in composite_scores.items():
            assert 0.0 <= score <= 1.0, f"{config_id}: {score}"

        # Config 003 should have high score (stable + good risk)
        assert composite_scores["config_003"] > 0.5

    def test_filter_by_primary_criteria(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test filtering configurations by primary criteria."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Filter by primary criteria
        passing_configs = selector.filter_by_primary_criteria()

        # Should exclude config_002 (high drawdown) and config_004 (low win rate)
        assert len(passing_configs) == 2
        assert "config_001" in passing_configs
        assert "config_003" in passing_configs
        assert "config_002" not in passing_configs
        assert "config_004" not in passing_configs

    def test_rank_configurations(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test ranking configurations by composite score."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Rank configurations
        ranked = selector.rank_configurations()

        # Should only include configurations passing primary criteria
        assert len(ranked) == 2

        # Should be sorted by composite score (descending)
        assert ranked[0].composite_score >= ranked[1].composite_score

        # Top configuration should be config_001 or config_003
        assert ranked[0].combination_id in ["config_001", "config_003"]

    def test_select_top_configurations(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test selection of top N configurations."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        # Select top 10 (should return 2 since only 2 pass criteria)
        top_configs = selector.select_top_configurations(n=10)

        assert isinstance(top_configs, TopConfigurations)
        assert len(top_configs.configurations) == 2
        assert top_configs.total_evaluated == 4
        assert top_configs.passing_primary_criteria == 2

        # Check configuration fields
        for config_rank in top_configs.configurations:
            assert hasattr(config_rank, "rank")
            assert hasattr(config_rank, "combination_id")
            assert hasattr(config_rank, "composite_score")
            assert hasattr(config_rank, "individual_scores")

    def test_generate_comparison_table(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test generation of comparison table."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        top_configs = selector.select_top_configurations(n=10)
        comparison = selector.generate_comparison_table(top_configs)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "rank" in comparison.columns
        assert "combination_id" in comparison.columns
        assert "composite_score" in comparison.columns
        assert "avg_oos_win_rate" in comparison.columns

    def test_select_optimal_configuration(
        self, sample_walk_forward_results, sample_summary_csv, default_criteria
    ):
        """Test selection of single optimal configuration."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        optimal = selector.select_optimal_configuration()

        assert optimal is not None
        assert optimal in ["config_001", "config_003"]
        # Should prefer config_003 for stability or config_001 for balance

    def test_generate_config_file(
        self,
        sample_walk_forward_results,
        sample_summary_csv,
        default_criteria,
        tmp_path,
    ):
        """Test generation of YAML configuration file."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        optimal_config = selector.select_optimal_configuration()
        config_path = tmp_path / "config-optimal.yaml"

        selector.generate_config_file(
            optimal_config_id=optimal_config,
            output_path=config_path,
        )

        assert config_path.exists()

        # Check file contains expected sections
        content = config_path.read_text()
        assert "ensemble:" in content
        assert "strategies:" in content
        assert "risk:" in content

    def test_generate_selection_report(
        self,
        sample_walk_forward_results,
        sample_summary_csv,
        default_criteria,
        tmp_path,
    ):
        """Test generation of selection report."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results,
            summary_csv_path=sample_summary_csv,
            criteria=default_criteria,
        )
        selector.load_results()

        optimal_config = selector.select_optimal_configuration()
        report_path = tmp_path / "optimal_config_selection.md"

        report = selector.generate_selection_report(
            optimal_config_id=optimal_config,
            output_path=report_path,
        )

        assert report_path.exists()

        # Check report structure
        content = report_path.read_text()
        assert "# Optimal Configuration Selection Report" in content
        assert "## Executive Summary" in content
        assert "## Top Configurations" in content
        assert "## Optimal Configuration" in content


class TestSelectionCriteria:
    """Test suite for SelectionCriteria model."""

    def test_default_criteria(self):
        """Test default selection criteria values."""
        criteria = SelectionCriteria()

        assert criteria.min_win_rate == 0.55
        assert criteria.min_profit_factor == 1.5
        assert criteria.max_drawdown == 0.15
        assert criteria.min_trade_frequency == 3.0
        assert criteria.max_win_rate_std == 0.10

    def test_custom_criteria(self):
        """Test custom selection criteria."""
        criteria = SelectionCriteria(
            min_win_rate=0.60,
            min_profit_factor=2.0,
            max_drawdown=0.10,
        )

        assert criteria.min_win_rate == 0.60
        assert criteria.min_profit_factor == 2.0
        assert criteria.max_drawdown == 0.10


class TestPrimaryCriteria:
    """Test suite for PrimaryCriteria model."""

    def test_primary_criteria_validation(self):
        """Test primary criteria validation result."""
        criteria = PrimaryCriteria(
            passes=True,
            win_rate_pass=True,
            profit_factor_pass=True,
            drawdown_pass=True,
            trade_frequency_pass=True,
            consistency_pass=True,
        )

        assert criteria.passes is True
        assert criteria.win_rate_pass is True
        assert criteria.profit_factor_pass is True

    def test_primary_criteria_partial_fail(self):
        """Test primary criteria with some failures."""
        criteria = PrimaryCriteria(
            passes=False,
            win_rate_pass=False,
            profit_factor_pass=True,
            drawdown_pass=True,
            trade_frequency_pass=True,
            consistency_pass=True,
        )

        assert criteria.passes is False
        assert criteria.win_rate_pass is False


class TestCompositeScore:
    """Test suite for CompositeScore model."""

    def test_composite_score_range(self):
        """Test composite score is in valid range."""
        score = CompositeScore(
            performance_score=0.8,
            stability_score=0.7,
            risk_score=0.9,
            frequency_score=0.6,
            composite_score=0.75,
        )

        assert 0.0 <= score.performance_score <= 1.0
        assert 0.0 <= score.stability_score <= 1.0
        assert 0.0 <= score.risk_score <= 1.0
        assert 0.0 <= score.frequency_score <= 1.0
        assert 0.0 <= score.composite_score <= 1.0


class TestConfigurationComparison:
    """Test suite for ConfigurationComparison model."""

    def test_configuration_comparison_fields(self):
        """Test configuration comparison has required fields."""
        comparison = ConfigurationComparison(
            rank=1,
            combination_id="config_001",
            composite_score=0.75,
            individual_scores=CompositeScore(
                performance_score=0.8,
                stability_score=0.7,
                risk_score=0.9,
                frequency_score=0.6,
                composite_score=0.75,
            ),
            avg_oos_win_rate=0.62,
            avg_oos_profit_factor=2.1,
            max_drawdown=0.12,
            trade_frequency=7.1,
            parameter_stability_score=0.75,
            performance_stability=0.72,
        )

        assert comparison.rank == 1
        assert comparison.combination_id == "config_001"
        assert comparison.composite_score == 0.75
        assert comparison.avg_oos_win_rate == 0.62


class TestTopConfigurations:
    """Test suite for TopConfigurations model."""

    def test_top_configurations_validation(self):
        """Test top configurations model."""
        top = TopConfigurations(
            configurations=[],
            total_evaluated=10,
            passing_primary_criteria=5,
        )

        assert len(top.configurations) == 0
        assert top.total_evaluated == 10
        assert top.passing_primary_criteria == 5


class TestSelectionReport:
    """Test suite for SelectionReport model."""

    def test_selection_report_fields(self):
        """Test selection report has required fields."""
        report = SelectionReport(
            optimal_config_id="config_001",
            optimal_config_metrics={},
            top_configurations=TopConfigurations(
                configurations=[], total_evaluated=10, passing_primary_criteria=5
            ),
            comparison_table=pd.DataFrame(),
            sensitivity_analysis={},
            confidence_level="high",
        )

        assert report.optimal_config_id == "config_001"
        assert report.confidence_level == "high"

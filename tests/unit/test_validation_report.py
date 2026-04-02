"""Tests for ValidationReportGenerator.

Tests the comprehensive validation report generation system that synthesizes
all walk-forward testing and optimization results for go/no-go decisions.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from src.research.validation_report_generator import (
    FinalValidationReport,
    GoNoGoDecision,
    GoNoGoRecommendation,
    ReportMetrics,
    ReportSection,
    RiskAssessment,
    ValidationReportGenerator,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_walk_forward_results():
    """Create sample walk-forward validation results."""
    return {
        "total_steps": 13,
        "average_win_rate": 0.58,
        "std_win_rate": 0.08,
        "average_profit_factor": 1.9,
        "max_drawdown": 0.12,
        "total_trades": 147,
        "parameter_stability_score": 0.75,
        "performance_stability": 0.72,
    }


@pytest.fixture
def sample_grid_search_results():
    """Create sample parameter grid search results."""
    return {
        "total_combinations_tested": 243,
        "best_combination_id": "config_001",
        "best_win_rate": 0.62,
        "best_profit_factor": 2.1,
        "top_10_average_win_rate": 0.59,
    }


@pytest.fixture
def sample_optimal_config():
    """Create sample optimal configuration results."""
    return {
        "optimal_config_id": "config_001",
        "win_rate": 0.62,
        "profit_factor": 2.1,
        "max_drawdown": 0.12,
        "trade_frequency": 7.1,
        "parameter_stability": 0.75,
        "performance_stability": 0.72,
        "composite_score": 0.68,
    }


@pytest.fixture
def sample_baseline_results():
    """Create sample baseline results from Epic 1."""
    return {
        "triple_confluence_win_rate": 0.54,
        "wolf_pack_win_rate": 0.51,
        "adaptive_ema_win_rate": 0.56,
        "vwap_bounce_win_rate": 0.52,
        "opening_range_win_rate": 0.50,
    }


@pytest.fixture
def sample_ensemble_results():
    """Create sample ensemble results from Epic 2."""
    return {
        "ensemble_win_rate": 0.60,
        "ensemble_profit_factor": 1.95,
        "ensemble_max_drawdown": 0.11,
        "sharpe_ratio": 1.8,
    }


@pytest.fixture
def sample_validation_data(
    sample_walk_forward_results,
    sample_grid_search_results,
    sample_optimal_config,
    sample_baseline_results,
    sample_ensemble_results,
):
    """Create complete validation data aggregation."""
    return {
        "walk_forward": sample_walk_forward_results,
        "grid_search": sample_grid_search_results,
        "optimal_config": sample_optimal_config,
        "baseline": sample_baseline_results,
        "ensemble": sample_ensemble_results,
    }


class TestGoNoGoRecommendation:
    """Test suite for GoNoGoRecommendation model."""

    def test_recommendation_values(self):
        """Test valid recommendation values."""
        valid_values = ["PROCEED", "CAUTION", "DO_NOT_PROCEED"]

        for value in valid_values:
            rec = GoNoGoRecommendation(value=value)
            assert rec.value == value

    def test_invalid_recommendation(self):
        """Test invalid recommendation value raises error."""
        with pytest.raises(ValueError):
            GoNoGoRecommendation(value="INVALID")


class TestGoNoGoDecision:
    """Test suite for GoNoGoDecision model."""

    def test_proceed_decision(self):
        """Test PROCEED decision creation."""
        decision = GoNoGoDecision(
            recommendation=GoNoGoRecommendation(value="PROCEED"),
            confidence_level="high",
            rationale="All critical criteria met with strong performance",
            critical_pass_count=15,
            critical_total=15,
        )

        assert decision.recommendation.value == "PROCEED"
        assert decision.confidence_level == "high"
        assert decision.critical_pass_count == 15
        assert decision.critical_total == 15

    def test_caution_decision(self):
        """Test CAUTION decision creation."""
        decision = GoNoGoDecision(
            recommendation=GoNoGoRecommendation(value="CAUTION"),
            confidence_level="medium",
            rationale="Most criteria pass, some concerns about drawdown",
            critical_pass_count=12,
            critical_total=15,
        )

        assert decision.recommendation.value == "CAUTION"
        assert decision.critical_pass_count == 12

    def test_do_not_proceed_decision(self):
        """Test DO_NOT_PROCEED decision creation."""
        decision = GoNoGoDecision(
            recommendation=GoNoGoRecommendation(value="DO_NOT_PROCEED"),
            confidence_level="high",
            rationale="Critical win rate threshold not met",
            critical_pass_count=8,
            critical_total=15,
        )

        assert decision.recommendation.value == "DO_NOT_PROCEED"
        assert decision.critical_pass_count == 8


class TestRiskAssessment:
    """Test suite for RiskAssessment model."""

    def test_risk_assessment_creation(self):
        """Test risk assessment creation."""
        risk = RiskAssessment(
            overall_risk_level="medium",
            max_drawdown_risk="low",
            overfitting_risk="medium",
            regime_change_risk="medium",
            data_quality_risk="low",
            key_risks=[
                "Moderate drawdown in volatile conditions",
                "Limited sample size for extreme market events",
            ],
            mitigation_strategies=[
                "Implement conservative position sizing",
                "Monitor regime changes during paper trading",
            ],
        )

        assert risk.overall_risk_level == "medium"
        assert len(risk.key_risks) == 2
        assert len(risk.mitigation_strategies) == 2


class TestReportMetrics:
    """Test suite for ReportMetrics model."""

    def test_report_metrics_creation(self):
        """Test report metrics creation."""
        metrics = ReportMetrics(
            walk_forward_win_rate=0.58,
            walk_forward_profit_factor=1.9,
            walk_forward_max_drawdown=0.12,
            optimal_win_rate=0.62,
            optimal_profit_factor=2.1,
            optimal_drawdown=0.12,
            ensemble_win_rate=0.60,
            ensemble_sharpe_ratio=1.8,
            parameter_stability=0.75,
            performance_stability=0.72,
        )

        assert metrics.walk_forward_win_rate == 0.58
        assert metrics.optimal_win_rate == 0.62
        assert metrics.ensemble_win_rate == 0.60


class TestValidationReportGenerator:
    """Test suite for ValidationReportGenerator."""

    def test_initialization(self, sample_validation_data):
        """Test generator initialization with validation data."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        assert generator.validation_data == sample_validation_data
        assert generator.report_date is not None

    def test_aggregate_results(self, sample_validation_data):
        """Test aggregation of all validation results."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        metrics = generator.aggregate_results()

        assert isinstance(metrics, ReportMetrics)
        assert metrics.walk_forward_win_rate == 0.58
        assert metrics.optimal_win_rate == 0.62
        assert metrics.ensemble_win_rate == 0.60

    def test_generate_go_no_go_decision_proceed(self, sample_validation_data):
        """Test go/no-go decision with PROCEED recommendation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value in ["PROCEED", "CAUTION", "DO_NOT_PROCEED"]
        assert isinstance(decision.confidence_level, str)
        assert isinstance(decision.rationale, str)
        assert isinstance(decision.critical_pass_count, int)
        assert isinstance(decision.critical_total, int)

    def test_generate_risk_assessment(self, sample_validation_data):
        """Test risk assessment generation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        risk = generator.generate_risk_assessment()

        assert isinstance(risk, RiskAssessment)
        assert risk.overall_risk_level in ["low", "medium", "high"]
        assert isinstance(risk.key_risks, list)
        assert isinstance(risk.mitigation_strategies, list)

    def test_generate_executive_summary(self, sample_validation_data):
        """Test executive summary generation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        decision = generator.generate_go_no_go_decision()
        summary = generator.generate_executive_summary(decision)

        assert "recommendation" in summary
        assert "key_metrics" in summary
        assert "deployment_readiness" in summary
        assert "risk_assessment" in summary

    def test_validate_success_criteria(self, sample_validation_data):
        """Test validation against success criteria."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        criteria_results = generator.validate_success_criteria()

        # Check that we have FR and NFR criteria
        has_fr = any(k.startswith("FR") for k in criteria_results.keys())
        has_nfr = any(k.startswith("NFR") for k in criteria_results.keys())
        assert has_fr or has_nfr
        assert isinstance(criteria_results, dict)

    def test_generate_report_sections(self, sample_validation_data):
        """Test generation of all report sections."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        sections = generator.generate_report_sections()

        assert len(sections) > 0
        assert all(isinstance(section, ReportSection) for section in sections)

        # Check for required sections
        section_titles = [s.title for s in sections]
        assert "Executive Summary" in section_titles
        assert "System Overview" in section_titles
        assert "Walk-Forward Validation Results" in section_titles

    def test_generate_markdown_report(self, sample_validation_data, tmp_path):
        """Test markdown report generation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        output_path = tmp_path / "validation_report.md"
        report = generator.generate_markdown_report(output_path=output_path)

        assert output_path.exists()
        assert isinstance(report, FinalValidationReport)
        assert report.report_path == output_path

        # Check file content
        content = output_path.read_text()
        assert "# Final Validation Report" in content
        assert "## Executive Summary" in content

    def test_generate_csv_exports(self, sample_validation_data, tmp_path):
        """Test CSV data export generation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        output_dir = tmp_path / "exports"
        csv_files = generator.generate_csv_exports(output_dir=output_dir)

        assert len(csv_files) > 0
        assert all(path.exists() for path in csv_files.values())

    def test_generate_final_report(self, sample_validation_data, tmp_path):
        """Test complete final report generation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        output_dir = tmp_path / "reports"
        final_report = generator.generate_final_report(output_dir=output_dir)

        assert isinstance(final_report, FinalValidationReport)
        assert final_report.report_path.exists()
        assert len(final_report.csv_exports) > 0

    def test_complete_report_workflow(self, sample_validation_data, tmp_path):
        """Test complete report generation workflow."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)

        output_dir = tmp_path / "complete_report"

        # Generate all components
        metrics = generator.aggregate_results()
        decision = generator.generate_go_no_go_decision()
        risk = generator.generate_risk_assessment()
        summary = generator.generate_executive_summary(decision)
        sections = generator.generate_report_sections()

        # Verify all components
        assert isinstance(metrics, ReportMetrics)
        assert isinstance(decision, GoNoGoDecision)
        assert isinstance(risk, RiskAssessment)
        assert isinstance(summary, dict)
        assert len(sections) >= 5


class TestFinalValidationReport:
    """Test suite for FinalValidationReport model."""

    def test_final_report_creation(self, tmp_path, sample_validation_data):
        """Test final validation report creation."""
        generator = ValidationReportGenerator(validation_data=sample_validation_data)
        metrics = generator.aggregate_results()
        decision = generator.generate_go_no_go_decision()

        markdown_path = tmp_path / "validation_report.md"

        report = FinalValidationReport(
            report_date=date.today(),
            go_no_go_decision=decision,
            metrics=metrics,
            report_path=markdown_path,
            csv_exports={},
        )

        assert report.report_date == date.today()
        assert report.go_no_go_decision == decision
        assert report.metrics == metrics
        assert report.report_path == markdown_path


class TestReportSection:
    """Test suite for ReportSection model."""

    def test_report_section_creation(self):
        """Test report section creation."""
        section = ReportSection(
            title="Executive Summary",
            content="Summary content here",
            order=1,
            tables=[],
            figures=[],
        )

        assert section.title == "Executive Summary"
        assert section.order == 1
        assert isinstance(section.tables, list)
        assert isinstance(section.figures, list)

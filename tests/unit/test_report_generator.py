"""Tests for comprehensive ensemble analysis report generator.

Tests for generating go/no-go decisions, aggregating all analysis components,
and creating baseline CSV for before/after comparison.
"""

import pytest
from datetime import date, datetime

from src.research.ensemble_backtester import BacktestResults
from src.research.report_generator import (
    EnsembleAnalysisReportGenerator,
    GoNoGoDecision,
)


@pytest.fixture
def sample_backtest_results():
    """Create sample ensemble backtest results."""
    return BacktestResults(
        total_trades=200,
        winning_trades=130,
        losing_trades=70,
        win_rate=0.65,
        profit_factor=1.8,
        average_win=120.0,
        average_loss=-75.0,
        largest_win=300.0,
        largest_loss=-200.0,
        max_drawdown=0.08,
        max_drawdown_duration=15,
        sharpe_ratio=1.5,
        average_hold_time=8.5,
        trade_frequency=10.0,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        confidence_threshold=0.50,
        trades=[],
        total_pnl=4500.0,
    )


@pytest.fixture
def mock_analyzers(sample_backtest_results):
    """Create mock analyzers for testing."""
    # This is a simplified mock - in real tests would use actual analyzers
    return {
        "ensemble_analyzer": None,  # Would be EnsembleAnalyzer
        "optimal_config_analyzer": None,  # Would be OptimalConfigAnalyzer
        "weight_evolution_simulator": None,  # Would be WeightEvolutionSimulator
        "regime_analyzer": None,  # Would be RegimeAnalyzer
    }


class TestGoNoGoDecision:
    """Test suite for GoNoGoDecision model."""

    def test_go_decision(self):
        """Test GO decision creation."""
        decision = GoNoGoDecision(
            decision="GO",
            rationale="Ensemble meets all criteria with robust performance",
            grade="A",
            key_metrics={
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "sharpe_ratio": 1.5,
            },
            strengths=["High win rate", "Positive risk-adjusted returns", "Robust across regimes"],
            weaknesses=["None significant"],
        )

        assert decision.decision == "GO"
        assert decision.grade == "A"
        assert len(decision.strengths) == 3
        assert len(decision.weaknesses) == 1

    def test_nogo_decision(self):
        """Test NO-GO decision creation."""
        decision = GoNoGoDecision(
            decision="NO-GO",
            rationale="Critical NFRs fail - win rate below 60%",
            grade="D",
            key_metrics={
                "win_rate": 0.52,
                "profit_factor": 1.1,
                "sharpe_ratio": 0.3,
            },
            strengths=["Good trade frequency"],
            weaknesses=["Win rate below minimum", "Low profit factor", "Poor Sharpe ratio"],
        )

        assert decision.decision == "NO-GO"
        assert decision.grade == "D"
        assert len(decision.weaknesses) == 3


class TestEnsembleAnalysisReportGenerator:
    """Test suite for EnsembleAnalysisReportGenerator class."""

    def test_initialization(self, mock_analyzers):
        """Test report generator can be initialized with analyzers."""
        # For testing, we can pass None and mock methods
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        assert generator is not None
        assert generator.ensemble_analyzer is None
        assert generator.optimal_config_analyzer is None

    def test_generate_executive_summary(self, mock_analyzers):
        """Test executive summary generation."""
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        summary = generator.generate_executive_summary()

        assert isinstance(summary, dict)
        assert "go_no_go_decision" in summary
        assert "grade" in summary
        assert "key_metrics" in summary
        assert "strengths" in summary
        assert "weaknesses" in summary

    def test_calculate_ensemble_grade(self, mock_analyzers):
        """Test ensemble grade calculation."""
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        # Test with simulated component grades
        component_grades = {
            "ensemble_profile": 0.85,  # B
            "optimal_config": 0.92,   # A
            "weight_evolution": 0.78,  # B
            "regime_analysis": 0.88,   # B
        }

        grade = generator.calculate_ensemble_grade(component_grades)

        assert grade in ["A", "B", "C", "D", "F"]
        assert isinstance(grade, str)

    def test_make_go_no_go_decision(self, mock_analyzers):
        """Test go/no-go decision logic."""
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        # Test GO scenario
        go_decision = generator.make_go_no_go_decision(
            grade="A",
            nfr_pass=True,
            regime_robust=True,
        )

        assert go_decision.decision in ["GO", "CAUTION", "NO-GO"]
        assert isinstance(go_decision.rationale, str)
        assert len(go_decision.rationale) > 0

        # Test NO-GO scenario
        nogo_decision = generator.make_go_no_go_decision(
            grade="D",
            nfr_pass=False,
            regime_robust=False,
        )

        assert nogo_decision.decision == "NO-GO"

    def test_generate_report(self, mock_analyzers):
        """Test comprehensive report generation."""
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        report = generator.generate_report()

        assert isinstance(report, str)
        assert len(report) > 0

        # Check for key sections
        assert "# Executive Summary" in report
        assert "# Ensemble Profile" in report or "## Ensemble Profile" in report
        assert "# Recommendations" in report or "## Recommendations" in report

    def test_create_baseline_csv(self, mock_analyzers, tmp_path):
        """Test baseline CSV creation."""
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        csv_path = tmp_path / "ensemble_baseline.csv"

        generator.create_baseline_csv(str(csv_path))

        # Check file was created
        assert csv_path.exists()

        # Read and validate CSV content
        import pandas as pd

        df = pd.read_csv(csv_path)

        assert "Metric" in df.columns or "metric" in df.columns
        assert "Value" in df.columns or "value" in df.columns
        assert len(df) > 0

    def test_report_completeness(self, mock_analyzers):
        """Test that report has all required sections."""
        generator = EnsembleAnalysisReportGenerator(
            ensemble_analyzer=None,
            optimal_config_analyzer=None,
            weight_evolution_simulator=None,
            regime_analyzer=None,
        )

        report = generator.generate_report()

        # Check for required sections
        required_sections = [
            "Executive Summary",
            "Ensemble Profile",
            "Recommendations",
        ]

        for section in required_sections:
            assert section in report, f"Missing section: {section}"

        # Check for no placeholder text
        assert "TODO" not in report
        assert "coming soon" not in report.lower()

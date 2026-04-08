"""Integration tests for performance documentation and analysis."""

import pytest
from datetime import datetime

from src.research.backtest_engine import Trade
from src.research.report_generator import ReportGenerator, StrategyProfile


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        return {
            "Triple Confluence Scalper": {
                "total_trades": 150,
                "win_rate": 0.75,
                "profit_factor": 2.5,
                "expectancy": 45.0,
                "max_drawdown_percent": 8.5,
                "sharpe_ratio": 1.8,
            },
            "VWAP Bounce": {
                "total_trades": 180,
                "win_rate": 0.68,
                "profit_factor": 2.1,
                "expectancy": 35.0,
                "max_drawdown_percent": 10.2,
                "sharpe_ratio": 1.5,
            },
            "Opening Range Breakout": {
                "total_trades": 120,
                "win_rate": 0.62,
                "profit_factor": 1.8,
                "expectancy": 28.0,
                "max_drawdown_percent": 12.0,
                "sharpe_ratio": 1.2,
            },
        }

    def test_report_generator_initialization(self, sample_metrics_data):
        """Test report generator initialization."""
        generator = ReportGenerator(sample_metrics_data)

        assert generator is not None
        assert generator.metrics_data == sample_metrics_data

    def test_generate_strategy_profile(self, sample_metrics_data):
        """Test strategy profile generation."""
        generator = ReportGenerator(sample_metrics_data)

        profile = generator.generate_strategy_profile("Triple Confluence Scalper")

        assert profile is not None
        assert profile.name == "Triple Confluence Scalper"
        assert profile.total_trades == 150
        assert profile.win_rate == 0.75

    def test_identify_strengths_and_weaknesses(self, sample_metrics_data):
        """Test identification of key strengths and weaknesses."""
        generator = ReportGenerator(sample_metrics_data)

        profile = generator.generate_strategy_profile("Triple Confluence Scalper")

        # Should have 3 strengths and 3 weaknesses
        assert len(profile.strengths) == 3
        assert len(profile.weaknesses) == 3

        # Check that strengths contain high win rate
        assert any("win rate" in s.lower() for s in profile.strengths)

    def test_generate_ranking_table(self, sample_metrics_data):
        """Test ranking table generation."""
        generator = ReportGenerator(sample_metrics_data)

        ranking = generator.generate_ranking_table()

        assert len(ranking) == 3
        # Should be ranked by performance (best first)
        assert ranking[0][0] == "Triple Confluence Scalper"
        assert ranking[0][1] >= ranking[1][1]  # Scores should descend

    def test_generate_markdown_report(self, sample_metrics_data):
        """Test full markdown report generation."""
        generator = ReportGenerator(sample_metrics_data)

        report = generator.generate_markdown_report()

        # Check all sections are present
        assert "# Strategy Baseline Performance Report" in report
        assert "## Executive Summary" in report
        assert "## Detailed Strategy Profiles" in report
        assert "## Comparison Analysis" in report
        assert "## Recommendations" in report

        # Check strategy names are present
        assert "Triple Confluence Scalper" in report
        assert "VWAP Bounce" in report
        assert "Opening Range Breakout" in report


class TestStrategyProfile:
    """Tests for StrategyProfile dataclass."""

    def test_strategy_profile_creation(self):
        """Test strategy profile object creation."""
        profile = StrategyProfile(
            name="Test Strategy",
            total_trades=100,
            win_rate=0.65,
            profit_factor=2.0,
            expectancy=30.0,
            max_drawdown_percent=10.0,
            sharpe_ratio=1.5,
            strengths=["High win rate", "Good risk management"],
            weaknesses=["Low trade frequency"],
            recommendation="Use in trending markets",
        )

        assert profile.name == "Test Strategy"
        assert profile.total_trades == 100
        assert profile.win_rate == 0.65
        assert len(profile.strengths) == 2
        assert len(profile.weaknesses) == 1

    def test_overall_recommendation_generation(self):
        """Test that recommendation is generated based on metrics."""
        generator = ReportGenerator({})

        # High performance strategy
        recommendation = generator._generate_recommendation(
            name="High Performer",
            win_rate=0.80,
            profit_factor=3.0,
            max_dd=5.0,
        )

        assert "recommended" in recommendation.lower() or "highly" in recommendation.lower()

        # Low performance strategy
        recommendation = generator._generate_recommendation(
            name="Low Performer",
            win_rate=0.40,
            profit_factor=0.8,
            max_dd=25.0,
        )

        assert "avoid" in recommendation.lower() or "not recommended" in recommendation.lower()


class TestEndToEndDocumentation:
    """End-to-end tests for documentation generation."""

    def test_full_documentation_workflow(self):
        """Test complete workflow from metrics to markdown report."""
        # Create sample metrics
        metrics_data = {
            "Test Strategy 1": {
                "total_trades": 50,
                "win_rate": 0.70,
                "profit_factor": 2.2,
                "expectancy": 40.0,
                "max_drawdown_percent": 9.0,
                "sharpe_ratio": 1.6,
            },
            "Test Strategy 2": {
                "total_trades": 60,
                "win_rate": 0.60,
                "profit_factor": 1.9,
                "expectancy": 25.0,
                "max_drawdown_percent": 11.0,
                "sharpe_ratio": 1.3,
            },
        }

        # Generate report
        generator = ReportGenerator(metrics_data)
        report = generator.generate_markdown_report()

        # Verify report structure
        lines = report.split("\n")

        # Should have multiple sections
        assert len(lines) > 50

        # Should contain headers
        assert any(line.startswith("# ") for line in lines)
        assert any(line.startswith("## ") for line in lines)

        # Should contain strategy names
        assert "Test Strategy 1" in report
        assert "Test Strategy 2" in report

        # Should contain metrics
        assert "70.00%" in report  # Win rate for strategy 1
        assert "60.00%" in report  # Win rate for strategy 2

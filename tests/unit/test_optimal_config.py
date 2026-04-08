"""Tests for optimal configuration analyzer.

Tests for sensitivity analysis, optimal threshold selection, and
configuration recommendation generation.
"""

import pytest
from datetime import date

from src.research.ensemble_backtester import BacktestResults, CompletedTrade
from src.research.optimal_config_analyzer import (
    OptimalConfigAnalyzer,
    ConfigRecommendation,
    TradeQualityAnalysis,
    ConfigValidator,
    ValidationReport,
)


@pytest.fixture
def sample_sensitivity_results():
    """Create sample sensitivity analysis results (5 thresholds)."""
    def create_results(win_rate, trades, pf=1.8, sharpe=1.5):
        return BacktestResults(
            total_trades=trades,
            winning_trades=int(trades * win_rate),
            losing_trades=trades - int(trades * win_rate),
            win_rate=win_rate,
            profit_factor=pf,
            average_win=120.0,
            average_loss=-75.0,
            largest_win=300.0,
            largest_loss=-200.0,
            max_drawdown=0.08,
            max_drawdown_duration=15,
            sharpe_ratio=sharpe,
            average_hold_time=8.5,
            trade_frequency=trades / 20,  # Assume 20 trading days
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4500.0,
        )

    return {
        0.40: create_results(win_rate=0.58, trades=300, pf=1.6, sharpe=1.3),
        0.45: create_results(win_rate=0.62, trades=250, pf=1.7, sharpe=1.4),
        0.50: create_results(win_rate=0.65, trades=200, pf=1.8, sharpe=1.5),
        0.55: create_results(win_rate=0.68, trades=150, pf=1.9, sharpe=1.6),
        0.60: create_results(win_rate=0.72, trades=100, pf=2.1, sharpe=1.7),
    }


class TestOptimalConfigAnalyzer:
    """Test suite for OptimalConfigAnalyzer class."""

    def test_initialization(self, sample_sensitivity_results):
        """Test analyzer can be initialized with sensitivity results."""
        analyzer = OptimalConfigAnalyzer(
            sensitivity_results=sample_sensitivity_results
        )

        assert analyzer.sensitivity_results == sample_sensitivity_results

    def test_find_optimal_threshold(self, sample_sensitivity_results):
        """Test optimal threshold selection."""
        analyzer = OptimalConfigAnalyzer(
            sensitivity_results=sample_sensitivity_results
        )

        optimal = analyzer.find_optimal_threshold()

        assert optimal in [0.40, 0.45, 0.50, 0.55, 0.60]
        assert isinstance(optimal, float)

    def test_analyze_trade_frequency_vs_quality(self, sample_sensitivity_results):
        """Test trade frequency vs quality analysis."""
        analyzer = OptimalConfigAnalyzer(
            sensitivity_results=sample_sensitivity_results
        )

        analysis = analyzer.analyze_trade_frequency_vs_quality()

        assert isinstance(analysis, TradeQualityAnalysis)
        assert hasattr(analysis, "threshold_analysis")
        assert hasattr(analysis, "trade_frequency_curve")
        assert hasattr(analysis, "win_rate_curve")
        assert hasattr(analysis, "sweet_spot")

    def test_generate_config_recommendation(self, sample_sensitivity_results):
        """Test configuration recommendation generation."""
        analyzer = OptimalConfigAnalyzer(
            sensitivity_results=sample_sensitivity_results
        )

        recommendation = analyzer.generate_config_recommendation()

        assert isinstance(recommendation, ConfigRecommendation)
        assert hasattr(recommendation, "recommended_threshold")
        assert hasattr(recommendation, "expected_performance")
        assert hasattr(recommendation, "reasoning")


class TestConfigRecommendation:
    """Test suite for ConfigRecommendation model."""

    def test_config_recommendation_creation(self):
        """Test ConfigRecommendation can be created."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.50,
            expected_performance={
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "sharpe_ratio": 1.5,
                "trades_per_day": 10.0,
            },
            trade_frequency_at_threshold=10.0,
            risk_adjusted_returns=1.5,
            reasoning="Balanced trade frequency and quality",
            comparison_to_default={
                "win_rate_delta": 0.0,
                "sharpe_delta": 0.0,
            },
        )

        assert recommendation.recommended_threshold == 0.50
        assert recommendation.expected_performance["win_rate"] == 0.65
        assert len(recommendation.reasoning) > 0


class TestTradeQualityAnalysis:
    """Test suite for TradeQualityAnalysis model."""

    def test_trade_quality_analysis_creation(self):
        """Test TradeQualityAnalysis can be created."""
        analysis = TradeQualityAnalysis(
            threshold_analysis={
                0.40: {"win_rate": 0.58, "trades": 300},
                0.50: {"win_rate": 0.65, "trades": 200},
            },
            trade_frequency_curve=[(0.40, 15.0), (0.50, 10.0)],
            win_rate_curve=[(0.40, 0.58), (0.50, 0.65)],
            sweet_spot=(0.50, 0.85),
        )

        assert analysis.sweet_spot == (0.50, 0.85)
        assert len(analysis.threshold_analysis) == 2
        assert len(analysis.trade_frequency_curve) == 2


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    def test_validate_trade_frequency_pass(self):
        """Test trade frequency validation (pass case)."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.50,
            expected_performance={},
            trade_frequency_at_threshold=10.0,  # Within 5-15 range
            risk_adjusted_returns=1.5,
            reasoning="",
            comparison_to_default={},
        )

        validator = ConfigValidator(recommendation)
        assert validator.validate_trade_frequency() is True

    def test_validate_trade_frequency_fail_low(self):
        """Test trade frequency validation (fail - too low)."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.60,
            expected_performance={},
            trade_frequency_at_threshold=3.0,  # Below 5
            risk_adjusted_returns=1.5,
            reasoning="",
            comparison_to_default={},
        )

        validator = ConfigValidator(recommendation)
        assert validator.validate_trade_frequency() is False

    def test_validate_trade_frequency_fail_high(self):
        """Test trade frequency validation (fail - too high)."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.40,
            expected_performance={},
            trade_frequency_at_threshold=20.0,  # Above 15
            risk_adjusted_returns=1.5,
            reasoning="",
            comparison_to_default={},
        )

        validator = ConfigValidator(recommendation)
        assert validator.validate_trade_frequency() is False

    def test_validate_win_rate_pass(self):
        """Test win rate validation (pass case)."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.50,
            expected_performance={"win_rate": 0.65},  # Above 60%
            trade_frequency_at_threshold=10.0,
            risk_adjusted_returns=1.5,
            reasoning="",
            comparison_to_default={},
        )

        validator = ConfigValidator(recommendation)
        assert validator.validate_win_rate() is True

    def test_check_red_flags(self):
        """Test red flag detection."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.50,
            expected_performance={"profit_factor": 1.8, "trades": 200},
            trade_frequency_at_threshold=10.0,
            risk_adjusted_returns=1.5,
            reasoning="",
            comparison_to_default={},
        )

        validator = ConfigValidator(recommendation)
        red_flags = validator.check_red_flags()

        assert isinstance(red_flags, list)

    def test_generate_validation_report(self):
        """Test validation report generation."""
        recommendation = ConfigRecommendation(
            recommended_threshold=0.50,
            expected_performance={"win_rate": 0.65},
            trade_frequency_at_threshold=10.0,
            risk_adjusted_returns=1.5,
            reasoning="Good performance",
            comparison_to_default={},
        )

        validator = ConfigValidator(recommendation)
        report = validator.generate_validation_report()

        assert isinstance(report, ValidationReport)
        assert hasattr(report, "is_valid")
        assert hasattr(report, "trade_frequency_pass")
        assert hasattr(report, "win_rate_pass")

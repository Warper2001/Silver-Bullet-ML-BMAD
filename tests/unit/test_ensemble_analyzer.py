"""Tests for ensemble analyzer - comprehensive performance analysis.

Tests for ensemble profile generation, grading system, strength/weakness
identification, and criteria comparison.
"""

import pytest
from datetime import date

from src.research.ensemble_backtester import BacktestResults, CompletedTrade
from src.research.ensemble_analyzer import (
    EnsembleAnalyzer,
    EnsembleProfile,
    CriteriaComparison,
    GradingSystem,
)


@pytest.fixture
def sample_ensemble_results():
    """Create sample ensemble backtest results."""
    trades = [
        CompletedTrade(
            entry_time="2024-01-01 10:00",
            exit_time="2024-01-01 10:08",
            direction="long",
            entry_price=15000.0,
            exit_price=15010.0,
            stop_loss=14990.0,
            take_profit=15020.0,
            pnl=50.0,
            bars_held=2,
            contracts=1,
            confidence=0.75,
            contributing_strategies=["Triple Confluence", "Wolf Pack"],
        ),
    ]

    return BacktestResults(
        total_trades=100,
        winning_trades=65,
        losing_trades=35,
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
        trade_frequency=12.5,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        confidence_threshold=0.50,
        trades=trades,
        total_pnl=4500.0,
    )


@pytest.fixture
def sample_individual_results():
    """Create sample individual strategy results."""
    return {
        "Triple Confluence": BacktestResults(
            total_trades=80,
            winning_trades=56,
            losing_trades=24,
            win_rate=0.70,
            profit_factor=2.0,
            average_win=100.0,
            average_loss=-60.0,
            largest_win=250.0,
            largest_loss=-150.0,
            max_drawdown=0.10,
            max_drawdown_duration=20,
            sharpe_ratio=1.3,
            average_hold_time=7.0,
            trade_frequency=10.0,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4000.0,
        ),
        "Wolf Pack": BacktestResults(
            total_trades=90,
            winning_trades=54,
            losing_trades=36,
            win_rate=0.60,
            profit_factor=1.5,
            average_win=130.0,
            average_loss=-90.0,
            largest_win=350.0,
            largest_loss=-250.0,
            max_drawdown=0.12,
            max_drawdown_duration=25,
            sharpe_ratio=1.1,
            average_hold_time=10.0,
            trade_frequency=11.25,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=3000.0,
        ),
    }


class TestEnsembleAnalyzer:
    """Test suite for EnsembleAnalyzer class."""

    def test_initialization(self, sample_ensemble_results, sample_individual_results):
        """Test analyzer can be initialized with results."""
        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        assert analyzer.backtest_results == sample_ensemble_results
        assert analyzer.individual_results == sample_individual_results

    def test_generate_profile(self, sample_ensemble_results, sample_individual_results):
        """Test ensemble profile generation."""
        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        profile = analyzer.generate_profile()

        assert isinstance(profile, EnsembleProfile)
        assert profile.name == "5-Strategy Weighted Ensemble"
        assert profile.grade in ["A", "B", "C", "D", "F"]
        assert len(profile.strengths) == 3
        assert len(profile.weaknesses) == 3

    def test_calculate_grade_a(self, sample_ensemble_results, sample_individual_results):
        """Test A grade calculation (excellent performance)."""
        # Modify results to get A grade
        sample_ensemble_results.win_rate = 0.70
        sample_ensemble_results.profit_factor = 2.2
        sample_ensemble_results.max_drawdown = 0.06
        sample_ensemble_results.sharpe_ratio = 2.5

        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        grade = analyzer.calculate_grade()
        assert grade == "A"

    def test_calculate_grade_b(self, sample_ensemble_results, sample_individual_results):
        """Test B grade calculation (good performance)."""
        sample_ensemble_results.win_rate = 0.58  # Below 60% target
        sample_ensemble_results.profit_factor = 1.7
        sample_ensemble_results.max_drawdown = 0.09
        sample_ensemble_results.sharpe_ratio = 1.4

        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        grade = analyzer.calculate_grade()
        assert grade == "B"

    def test_calculate_grade_c(self, sample_ensemble_results, sample_individual_results):
        """Test C grade calculation (acceptable performance)."""
        sample_ensemble_results.win_rate = 0.55
        sample_ensemble_results.profit_factor = 1.4
        sample_ensemble_results.max_drawdown = 0.10
        sample_ensemble_results.sharpe_ratio = 1.2

        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        grade = analyzer.calculate_grade()
        assert grade == "C"

    def test_calculate_grade_f(self, sample_ensemble_results, sample_individual_results):
        """Test F grade calculation (poor performance)."""
        sample_ensemble_results.win_rate = 0.45
        sample_ensemble_results.profit_factor = 1.0
        sample_ensemble_results.max_drawdown = 0.15
        sample_ensemble_results.sharpe_ratio = 0.8

        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        grade = analyzer.calculate_grade()
        assert grade in ["D", "F"]

    def test_identify_strengths_weaknesses(self, sample_ensemble_results, sample_individual_results):
        """Test strengths and weaknesses identification."""
        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        strengths, weaknesses = analyzer.identify_strengths_weaknesses()

        assert len(strengths) == 3
        assert len(weaknesses) == 3
        assert all(isinstance(s, str) for s in strengths)
        assert all(isinstance(w, str) for w in weaknesses)

    def test_compare_to_criteria(self, sample_ensemble_results, sample_individual_results):
        """Test comparison to success criteria."""
        analyzer = EnsembleAnalyzer(
            backtest_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        criteria = analyzer.compare_to_criteria()

        assert isinstance(criteria, CriteriaComparison)
        assert hasattr(criteria, "win_rate_pass")
        assert hasattr(criteria, "profit_factor_pass")
        assert hasattr(criteria, "max_drawdown_pass")
        assert hasattr(criteria, "trade_frequency_pass")
        assert hasattr(criteria, "overall_pass")


class TestEnsembleProfile:
    """Test suite for EnsembleProfile model."""

    def test_ensemble_profile_creation(self, sample_ensemble_results, sample_individual_results):
        """Test EnsembleProfile can be created with all fields."""
        profile = EnsembleProfile(
            name="5-Strategy Weighted Ensemble",
            description="Ensemble of 5 ICT-based strategies with weighted confidence scoring",
            performance_metrics={
                "total_trades": 100,
                "win_rate": 0.65,
                "profit_factor": 1.8,
            },
            comparison_vs_individuals={
                "win_rate_improvement": -0.05,
                "profit_factor_improvement": -0.2,
            },
            comparison_vs_criteria={
                "win_rate_pass": True,
                "max_drawdown_pass": True,
            },
            grade="B",
            strengths=["High win rate", "Good risk management", "Consistent performance"],
            weaknesses=["Lower profit factor than individuals", "Moderate drawdown", "Room for optimization"],
        )

        assert profile.name == "5-Strategy Weighted Ensemble"
        assert profile.grade == "B"
        assert len(profile.strengths) == 3
        assert profile.comparison_vs_criteria["win_rate_pass"] is True


class TestGradingSystem:
    """Test suite for GradingSystem."""

    def test_weighted_score_calculation(self):
        """Test weighted score calculation for grading."""
        system = GradingSystem()

        # A-grade performance
        score = system.calculate_weighted_score(
            win_rate=0.70,
            profit_factor=2.2,
            max_drawdown=0.06,
            sharpe_ratio=2.5,
        )

        assert score >= 0.90  # Should be A (90-100%)

    def test_score_to_grade_mapping(self):
        """Test score to letter grade mapping."""
        system = GradingSystem()

        assert system.score_to_grade(0.95) == "A"
        assert system.score_to_grade(0.85) == "B"
        assert system.score_to_grade(0.75) == "C"
        assert system.score_to_grade(0.65) == "D"
        assert system.score_to_grade(0.55) == "F"


class TestCriteriaComparison:
    """Test suite for CriteriaComparison model."""

    def test_criteria_comparison_creation(self):
        """Test CriteriaComparison can be created."""
        comparison = CriteriaComparison(
            win_rate_pass=True,
            profit_factor_pass=True,
            max_drawdown_pass=True,
            sharpe_ratio_pass=True,
            trade_frequency_pass=True,
            overall_pass=True,
            failed_criteria=[],
        )

        assert comparison.overall_pass is True
        assert len(comparison.failed_criteria) == 0

    def test_criteria_comparison_with_failures(self):
        """Test CriteriaComparison with some failures."""
        comparison = CriteriaComparison(
            win_rate_pass=False,  # Below 60%
            profit_factor_pass=True,
            max_drawdown_pass=True,
            sharpe_ratio_pass=True,
            trade_frequency_pass=False,  # Outside 5-15 range
            overall_pass=False,
            failed_criteria=["Win rate below 60% threshold", "Trade frequency outside target range"],
        )

        assert comparison.overall_pass is False
        assert len(comparison.failed_criteria) == 2

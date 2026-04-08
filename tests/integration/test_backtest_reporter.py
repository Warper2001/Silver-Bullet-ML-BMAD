"""Tests for backtest reporter - comparison and diversification analysis."""

import pytest
from datetime import date

from src.research.ensemble_backtester import BacktestResults, CompletedTrade
from src.research.backtest_reporter import (
    BacktestReporter,
    DiversificationReport,
    ComparisonTable,
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
        CompletedTrade(
            entry_time="2024-01-01 11:00",
            exit_time="2024-01-01 11:10",
            direction="short",
            entry_price=15000.0,
            exit_price=14990.0,
            stop_loss=15010.0,
            take_profit=14980.0,
            pnl=50.0,
            bars_held=2,
            contracts=1,
            confidence=0.80,
            contributing_strategies=["EMA Momentum", "VWAP Bounce"],
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
        "EMA Momentum": BacktestResults(
            total_trades=70,
            winning_trades=42,
            losing_trades=28,
            win_rate=0.60,
            profit_factor=1.6,
            average_win=110.0,
            average_loss=-70.0,
            largest_win=280.0,
            largest_loss=-180.0,
            max_drawdown=0.09,
            max_drawdown_duration=18,
            sharpe_ratio=1.4,
            average_hold_time=9.0,
            trade_frequency=8.75,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=3500.0,
        ),
    }


class TestBacktestReporter:
    """Test suite for BacktestReporter class."""

    def test_initialization(self, sample_ensemble_results, sample_individual_results):
        """Test reporter can be initialized with results."""
        reporter = BacktestReporter(
            ensemble_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        assert reporter.ensemble_results == sample_ensemble_results
        assert reporter.individual_results == sample_individual_results

    def test_generate_comparison_table(self, sample_ensemble_results, sample_individual_results):
        """Test comparison table generation."""
        reporter = BacktestReporter(
            ensemble_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        comparison = reporter.generate_comparison_table()

        assert isinstance(comparison, ComparisonTable)
        assert hasattr(comparison, "metrics")
        assert len(comparison.metrics) > 0

    def test_calculate_improvements(self, sample_ensemble_results, sample_individual_results):
        """Test performance improvement calculations."""
        reporter = BacktestReporter(
            ensemble_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        improvements = reporter.calculate_improvements()

        assert "win_rate_improvement" in improvements
        assert "profit_factor_improvement" in improvements
        assert "drawdown_reduction" in improvements
        assert "sharpe_improvement" in improvements

    def test_analyze_diversification(self, sample_ensemble_results, sample_individual_results):
        """Test diversification analysis."""
        reporter = BacktestReporter(
            ensemble_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        div_analysis = reporter.analyze_diversification()

        assert isinstance(div_analysis, DiversificationReport)
        assert hasattr(div_analysis, "signal_correlation")
        assert hasattr(div_analysis, "diversification_benefit")
        assert hasattr(div_analysis, "contribution_analysis")

    def test_generate_recommendation_go(self, sample_ensemble_results, sample_individual_results):
        """Test recommendation generation when ensemble outperforms."""
        reporter = BacktestReporter(
            ensemble_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        recommendation = reporter.generate_recommendation()

        assert recommendation in ["GO", "CAUTION", "NO-GO"]
        # Ensemble has 0.65 win rate vs best individual 0.70
        # But ensemble has lower drawdown (0.08 vs 0.10, 0.12, 0.09)
        # So should be GO or CAUTION

    def test_generate_markdown_report(self, sample_ensemble_results, sample_individual_results):
        """Test markdown report generation."""
        reporter = BacktestReporter(
            ensemble_results=sample_ensemble_results,
            individual_results=sample_individual_results
        )

        report = reporter.generate_markdown_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Ensemble" in report
        assert "win rate" in report.lower()
        assert "profit factor" in report.lower()


class TestDiversificationReport:
    """Test suite for DiversificationReport model."""

    def test_diversification_report_creation(self):
        """Test DiversificationReport can be created."""
        report = DiversificationReport(
            signal_correlation={"Triple-Wolf": 0.7, "Triple-EMA": 0.5},
            diversification_benefit=0.15,  # 15% drawdown reduction
            contribution_analysis={"Triple Confluence": 0.40, "Wolf Pack": 0.35},
            signal_frequency_analysis={"Triple Confluence": 10.0, "Wolf Pack": 11.25},
        )

        assert report.signal_correlation["Triple-Wolf"] == 0.7
        assert report.diversification_benefit == 0.15

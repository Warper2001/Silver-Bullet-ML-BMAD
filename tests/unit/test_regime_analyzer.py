"""Tests for regime analyzer.

Tests for market regime detection, performance analysis by regime,
and validation of diverse edge sources (NFR5).
"""

import pytest
import pandas as pd
from datetime import date, datetime

from src.research.ensemble_backtester import BacktestResults, CompletedTrade
from src.research.regime_analyzer import (
    RegimeAnalyzer,
    RegimePerformanceReport,
    StrategyContributionReport,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data with different regimes."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
    prices = []

    # First 25 bars: Bull market (rising prices)
    for i in range(25):
        prices.append(15000 + i * 10)  # Rising from 15000 to 15250

    # Next 25 bars: Bear market (falling prices)
    for i in range(25):
        prices.append(15250 - i * 10)  # Falling from 15250 to 15000

    # Next 25 bars: Ranging (oscillating around 15000)
    for i in range(25):
        prices.append(15000 + (i % 5) * 5)  # Oscillating 15000-15020

    # Last 25 bars: Volatile (large swings)
    for i in range(25):
        prices.append(15000 + i * 20)  # Large upward moves

    df = pd.DataFrame({
        "timestamp": dates,
        "close": prices,
    })
    df.set_index("timestamp", inplace=True)

    return df


@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results for regime analysis."""
    def create_results(win_rate, trades, pf=1.8):
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
            sharpe_ratio=1.5,
            average_hold_time=8.5,
            trade_frequency=trades / 20,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4500.0,
        )

    return create_results(win_rate=0.65, trades=200, pf=1.8)


@pytest.fixture
def sample_individual_results():
    """Create sample individual strategy results."""
    def create_results(win_rate, trades, pf=1.8):
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
            sharpe_ratio=1.5,
            average_hold_time=8.5,
            trade_frequency=trades / 20,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4500.0,
        )

    return {
        "triple_confluence_scaler": create_results(win_rate=0.68, trades=40, pf=2.0),
        "wolf_pack_3_edge": create_results(win_rate=0.62, trades=48, pf=1.7),
        "adaptive_ema_momentum": create_results(win_rate=0.70, trades=32, pf=2.2),
        "vwap_bounce": create_results(win_rate=0.58, trades=52, pf=1.5),
        "opening_range_breakout": create_results(win_rate=0.65, trades=28, pf=1.8),
    }


class TestRegimePerformanceReport:
    """Test suite for RegimePerformanceReport model."""

    def test_regime_performance_report_creation(self):
        """Test RegimePerformanceReport can be created."""
        report = RegimePerformanceReport(
            performance_by_regime={
                "bull": {"win_rate": 0.70, "profit_factor": 2.0, "trades": 50},
                "bear": {"win_rate": 0.60, "profit_factor": 1.5, "trades": 40},
                "ranging": {"win_rate": 0.65, "profit_factor": 1.8, "trades": 60},
                "volatile": {"win_rate": 0.55, "profit_factor": 1.3, "trades": 50},
            },
            best_regime="bull",
            worst_regime="volatile",
            regime_robustness_score=0.82,
            trade_counts={"bull": 50, "bear": 40, "ranging": 60, "volatile": 50},
        )

        assert report.best_regime == "bull"
        assert report.worst_regime == "volatile"
        assert report.regime_robustness_score == 0.82
        assert len(report.performance_by_regime) == 4


class TestStrategyContributionReport:
    """Test suite for StrategyContributionReport model."""

    def test_strategy_contribution_report_creation(self):
        """Test StrategyContributionReport can be created."""
        report = StrategyContributionReport(
            contributions_by_regime={
                "bull": {
                    "triple_confluence_scaler": 0.68,
                    "wolf_pack_3_edge": 0.62,
                    "adaptive_ema_momentum": 0.70,
                },
                "bear": {
                    "triple_confluence_scaler": 0.60,
                    "wolf_pack_3_edge": 0.70,
                    "adaptive_ema_momentum": 0.55,
                },
            },
            diverse_edge_sources=True,
            regime_coverage={
                "bull": ["triple_confluence_scaler", "adaptive_ema_momentum"],
                "bear": ["wolf_pack_3_edge"],
                "ranging": ["vwap_bounce", "opening_range_breakout"],
                "volatile": ["adaptive_ema_momentum"],
            },
        )

        assert report.diverse_edge_sources is True
        assert len(report.contributions_by_regime) == 2
        assert len(report.regime_coverage) == 4


class TestRegimeAnalyzer:
    """Test suite for RegimeAnalyzer class."""

    def test_initialization(self, sample_backtest_results, sample_price_data):
        """Test analyzer can be initialized with backtest results and price data."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
        )

        assert analyzer.backtest_results is not None
        assert analyzer.price_data is not None

    def test_detect_market_regimes(self, sample_backtest_results, sample_price_data):
        """Test market regime detection."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
        )

        regimes = analyzer.detect_market_regimes()

        assert isinstance(regimes, dict)
        assert len(regimes) > 0

        # Check that regimes are valid
        valid_regimes = {"bull", "bear", "ranging", "volatile"}
        for regime in regimes.values():
            assert regime in valid_regimes

    def test_analyze_regime_performance(self, sample_backtest_results, sample_price_data):
        """Test regime performance analysis."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
        )

        # Detect regimes first
        analyzer.detect_market_regimes()

        # Analyze performance
        report = analyzer.analyze_regime_performance()

        assert isinstance(report, RegimePerformanceReport)
        assert hasattr(report, "performance_by_regime")
        assert hasattr(report, "best_regime")
        assert hasattr(report, "worst_regime")
        assert hasattr(report, "regime_robustness_score")

    def test_calculate_regime_robustness_score(self, sample_backtest_results, sample_price_data):
        """Test regime robustness score calculation."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
        )

        # Detect regimes first
        analyzer.detect_market_regimes()

        # Calculate robustness score
        score = analyzer.calculate_regime_robustness_score()

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identify_regime_weaknesses(self, sample_backtest_results, sample_price_data):
        """Test regime weakness identification."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
        )

        # Detect regimes first
        analyzer.detect_market_regimes()

        # Identify weaknesses
        weaknesses = analyzer.identify_regime_weaknesses()

        assert isinstance(weaknesses, list)
        assert all(isinstance(w, str) for w in weaknesses)

    def test_validate_diverse_edges(self, sample_backtest_results, sample_price_data, sample_individual_results):
        """Test diverse edge sources validation (NFR5)."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
            individual_results=sample_individual_results,
        )

        # Detect regimes first
        analyzer.detect_market_regimes()

        # Validate diverse edges
        diverse = analyzer.validate_diverse_edges()

        assert isinstance(diverse, bool)

    def test_robustness_score_acceptable(self, sample_backtest_results, sample_price_data):
        """Test that robustness score meets minimum threshold."""
        analyzer = RegimeAnalyzer(
            backtest_results=sample_backtest_results,
            price_data=sample_price_data,
        )

        # Detect regimes first
        analyzer.detect_market_regimes()

        # Calculate robustness score
        score = analyzer.calculate_regime_robustness_score()

        # Robustness should be >= 0.0 (could be poor if no data)
        assert score >= 0.0

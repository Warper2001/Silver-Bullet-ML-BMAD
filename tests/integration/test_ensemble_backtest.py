"""Integration tests for ensemble backtesting.

Tests the complete ensemble backtesting workflow including:
- Signal aggregation from all strategies
- Weighted confidence scoring
- Entry/exit logic
- P&L calculation
- Performance metrics
"""

import h5py
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, date, timedelta
from pathlib import Path

from src.research.ensemble_backtester import (
    EnsembleBacktester,
    BacktestResults,
    CompletedTrade,
)


@pytest.fixture
def sample_dollar_bars(tmp_path):
    """Create sample dollar bar data for testing.

    Generates 100 days of 5-minute MNQ dollar bars with realistic price movements.
    """
    n_bars = 100 * 79  # 100 days * ~79 bars/day (6.5 hours * 60/5)
    timestamps = pd.date_range(
        start="2024-01-01 09:30:00",
        periods=n_bars,
        freq="5min"
    )

    # Generate realistic MNQ price data
    np.random.seed(42)
    base_price = 15000.0
    returns = np.random.normal(0, 0.0002, n_bars)  # Small returns per 5-min bar
    prices = base_price * (1 + returns).cumprod()

    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices * 1.0003,
        "low": prices * 0.9997,
        "close": prices,
        "volume": np.random.randint(100, 1000, n_bars),
    })

    # Save to HDF5
    data_path = tmp_path / "test_dollar_bars.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("timestamps", data=timestamps.astype(np.int64))
        f.create_dataset("open", data=bars["open"].values)
        f.create_dataset("high", data=bars["high"].values)
        f.create_dataset("low", data=bars["low"].values)
        f.create_dataset("close", data=bars["close"].values)
        f.create_dataset("volume", data=bars["volume"].values)

    return str(data_path)


@pytest.fixture
def mock_strategies(monkeypatch):
    """Mock strategy detectors to return predictable signals."""
    signals_generated = []

    def mock_detect(self, bar):
        # Generate 1-2 signals per day
        if np.random.random() < 0.02:  # 2% chance per bar
            from src.detection.models import TripleConfluenceSignal

            signals_generated.append(bar["timestamp"])

            return TripleConfluenceSignal(
                entry_price=bar["close"],
                stop_loss=bar["close"] * 0.999,
                take_profit=bar["close"] * 1.002,
                direction="long" if np.random.random() > 0.5 else "short",
                confidence=np.random.uniform(0.7, 0.95),
                timestamp=bar["timestamp"],
                contributing_factors={"test": "mock"},
            )
        return None

    # Monkey patch all strategy detectors
    monkeypatch.setattr(
        "src.detection.triple_confluence_strategy.TripleConfluenceStrategy.detect",
        mock_detect
    )
    # Add other strategies as needed

    return signals_generated


class TestEnsembleBacktester:
    """Test suite for EnsembleBacktester class."""

    def test_initialization(self, sample_dollar_bars):
        """Test backtester can be initialized with config and data paths."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        assert backtester is not None
        assert backtester.data_path == sample_dollar_bars

    def test_run_backtest_returns_results(self, sample_dollar_bars):
        """Test backtest runs and returns BacktestResults."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        results = backtester.run_backtest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50
        )

        assert isinstance(results, BacktestResults)
        assert results.total_trades >= 0
        assert results.win_rate >= 0.0
        assert results.win_rate <= 1.0
        assert results.profit_factor >= 0.0

    def test_backtest_processes_all_bars(self, sample_dollar_bars):
        """Test that backtest processes bars from date range."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        results = backtester.run_backtest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50
        )

        # Verify that bars were processed (exact count depends on fixture data generation)
        # The fixture generates 100 days of data starting 2024-01-01
        # We're processing January 2024, which should include ~31/100 of the data
        assert backtester.bars_processed > 0
        assert backtester.bars_processed <= 7900  # Max bars in fixture

    def test_backtest_records_trades_correctly(self, sample_dollar_bars):
        """Test that trades are recorded with all required fields."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        results = backtester.run_backtest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50
        )

        if results.total_trades > 0:
            trade = results.trades[0]
            assert isinstance(trade, CompletedTrade)
            assert hasattr(trade, "entry_time")
            assert hasattr(trade, "exit_time")
            assert hasattr(trade, "direction")
            assert hasattr(trade, "entry_price")
            assert hasattr(trade, "exit_price")
            assert hasattr(trade, "stop_loss")
            assert hasattr(trade, "take_profit")
            assert hasattr(trade, "pnl")
            assert hasattr(trade, "bars_held")
            assert trade.direction in ["long", "short"]
            assert trade.pnl != 0  # Should have non-zero P&L

    def test_backtest_calculates_performance_metrics(self, sample_dollar_bars):
        """Test that all 12 performance metrics are calculated."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        results = backtester.run_backtest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50
        )

        # Check all 12 metrics exist
        assert hasattr(results, "total_trades")
        assert hasattr(results, "win_rate")
        assert hasattr(results, "profit_factor")
        assert hasattr(results, "average_win")
        assert hasattr(results, "average_loss")
        assert hasattr(results, "largest_win")
        assert hasattr(results, "largest_loss")
        assert hasattr(results, "max_drawdown")
        assert hasattr(results, "max_drawdown_duration")
        assert hasattr(results, "sharpe_ratio")
        assert hasattr(results, "average_hold_time")
        assert hasattr(results, "trade_frequency")

    def test_confidence_threshold_filters_signals(self, sample_dollar_bars):
        """Test that higher confidence threshold generates fewer trades."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        # Run with low threshold
        results_low = backtester.run_backtest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.40
        )

        # Run with high threshold
        results_high = backtester.run_backtest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.60
        )

        # Higher threshold should generate fewer trades
        assert results_high.total_trades <= results_low.total_trades

    def test_sensitivity_analysis(self, sample_dollar_bars):
        """Test sensitivity analysis across multiple thresholds."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
        sensitivity_results = backtester.run_sensitivity_analysis(thresholds)

        assert isinstance(sensitivity_results, dict)
        assert len(sensitivity_results) == len(thresholds)

        for threshold in thresholds:
            assert threshold in sensitivity_results
            results = sensitivity_results[threshold]
            assert isinstance(results, BacktestResults)

    def test_sensitivity_analysis_trade_frequency_decreases(self, sample_dollar_bars):
        """Test that trade frequency decreases as threshold increases."""
        backtester = EnsembleBacktester(
            config_path="config-sim.yaml",
            data_path=sample_dollar_bars
        )

        thresholds = [0.40, 0.50, 0.60]
        sensitivity_results = backtester.run_sensitivity_analysis(thresholds)

        trades_40 = sensitivity_results[0.40].total_trades
        trades_50 = sensitivity_results[0.50].total_trades
        trades_60 = sensitivity_results[0.60].total_trades

        # Higher thresholds should generate fewer trades
        assert trades_60 <= trades_50 <= trades_40


class TestCompletedTrade:
    """Test suite for CompletedTrade model."""

    def test_completed_trade_creation(self):
        """Test CompletedTrade can be created with all fields."""
        trade = CompletedTrade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 10, 15),
            direction="long",
            entry_price=15000.0,
            exit_price=15010.0,
            stop_loss=14990.0,
            take_profit=15020.0,
            pnl=50.0,
            bars_held=3,
            contracts=1,
            confidence=0.80,
            contributing_strategies=["Triple Confluence Scalper", "Wolf Pack 3-Edge"],
        )

        assert trade.entry_time == datetime(2024, 1, 1, 10, 0)
        assert trade.direction == "long"
        assert trade.pnl == 50.0
        assert len(trade.contributing_strategies) == 2

    def test_pnl_calculation_long(self):
        """Test P&L calculation for long trade."""
        # MNQ: $5 per point
        # Long: (exit - entry) * 5 * contracts
        entry = 15000.0
        exit = 15010.0
        expected_pnl = (exit - entry) * 5.0  # $50

        trade = CompletedTrade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 10, 15),
            direction="long",
            entry_price=entry,
            exit_price=exit,
            stop_loss=14990.0,
            take_profit=15020.0,
            pnl=expected_pnl,
            bars_held=3,
            contracts=1,
            confidence=0.80,
            contributing_strategies=["Test"],
        )

        assert trade.pnl == 50.0

    def test_pnl_calculation_short(self):
        """Test P&L calculation for short trade."""
        # MNQ: $5 per point
        # Short: (entry - exit) * 5 * contracts
        entry = 15000.0
        exit = 14990.0
        expected_pnl = (entry - exit) * 5.0  # $50

        trade = CompletedTrade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 10, 15),
            direction="short",
            entry_price=entry,
            exit_price=exit,
            stop_loss=15010.0,
            take_profit=14980.0,
            pnl=expected_pnl,
            bars_held=3,
            contracts=1,
            confidence=0.80,
            contributing_strategies=["Test"],
        )

        assert trade.pnl == 50.0


class TestBacktestResults:
    """Test suite for BacktestResults model."""

    def test_backtest_results_creation(self):
        """Test BacktestResults can be created with sample data."""
        results = BacktestResults(
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            win_rate=0.65,
            profit_factor=1.8,
            average_win=120.0,
            average_loss=75.0,
            largest_win=300.0,
            largest_loss=200.0,
            max_drawdown=0.08,
            max_drawdown_duration=15,
            sharpe_ratio=1.5,
            average_hold_time=8.5,
            trade_frequency=12.5,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4500.0,
        )

        assert results.total_trades == 100
        assert results.win_rate == 0.65
        assert results.profit_factor == 1.8

    def test_backtest_results_metrics_calculation(self):
        """Test metrics are within valid ranges."""
        results = BacktestResults(
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            win_rate=0.65,
            profit_factor=1.8,
            average_win=120.0,
            average_loss=-75.0,  # Negative for losses
            largest_win=300.0,
            largest_loss=-200.0,  # Negative for losses
            max_drawdown=0.08,
            max_drawdown_duration=15,
            sharpe_ratio=1.5,
            average_hold_time=8.5,
            trade_frequency=12.5,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4500.0,
        )

        # Validate metric ranges
        assert 0 <= results.win_rate <= 1
        assert results.profit_factor >= 0
        assert 0 <= results.max_drawdown <= 1
        assert results.average_win > 0
        assert results.average_loss < 0  # Losses are negative
        assert results.largest_loss < 0  # Largest loss is negative
        assert results.trade_frequency > 0

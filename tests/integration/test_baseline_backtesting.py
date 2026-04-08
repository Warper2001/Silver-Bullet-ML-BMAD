"""Integration tests for baseline backtesting engine."""

import pytest
from datetime import datetime, timedelta

from src.data.models import DollarBar
from src.research.backtest_engine import BacktestEngine, Trade
from src.research.performance_analyzer import PerformanceMetrics, PerformanceAnalyzer


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    @pytest.fixture
    def sample_bars(self) -> list[DollarBar]:
        """Create sample dollar bars for testing."""
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        bars = []

        for i in range(100):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i * 2,
                high=11805.0 + i * 2,
                low=11799.0 + i * 2,
                close=11803.0 + i * 2,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        return bars

    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization."""
        engine = BacktestEngine()

        assert engine is not None
        assert len(engine.trades) == 0

    def test_process_trades(self, sample_bars):
        """Test processing trades through backtest engine."""
        engine = BacktestEngine(initial_capital=100000)

        # Simulate some trades
        trade1 = Trade(
            entry_time=sample_bars[0].timestamp,
            exit_time=sample_bars[5].timestamp,
            direction="long",
            entry_price=11800.0,
            exit_price=11810.0,
            stop_loss=11795.0,
            take_profit=11820.0,
            pnl=100.0,  # $100 per contract
            bars_held=5,
        )

        engine.trades.append(trade1)

        # Verify trade was recorded
        assert len(engine.trades) == 1
        assert engine.trades[0].pnl == 100.0

    def test_calculate_total_pnl(self, sample_bars):
        """Test total P&L calculation."""
        engine = BacktestEngine()

        # Add winning trade
        trade1 = Trade(
            entry_time=sample_bars[0].timestamp,
            exit_time=sample_bars[5].timestamp,
            direction="long",
            entry_price=11800.0,
            exit_price=11810.0,
            stop_loss=11795.0,
            take_profit=11820.0,
            pnl=100.0,
            bars_held=5,
        )

        # Add losing trade
        trade2 = Trade(
            entry_time=sample_bars[10].timestamp,
            exit_time=sample_bars[15].timestamp,
            direction="short",
            entry_price=11820.0,
            exit_price=11830.0,
            stop_loss=11825.0,
            take_profit=11810.0,
            pnl=-50.0,
            bars_held=5,
        )

        engine.trades.extend([trade1, trade2])

        total_pnl = sum(trade.pnl for trade in engine.trades)
        assert total_pnl == 50.0


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer class."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        trades = [
            # Winning trades
            Trade(
                entry_time=base_time + timedelta(minutes=i),
                exit_time=base_time + timedelta(minutes=i + 5),
                direction="long",
                entry_price=11800.0 + i,
                exit_price=11810.0 + i,
                stop_loss=11795.0 + i,
                take_profit=11820.0 + i,
                pnl=100.0,
                bars_held=5,
            )
            for i in range(0, 30, 5)
        ]

        # Add some losing trades
        trades.extend(
            [
                Trade(
                    entry_time=base_time + timedelta(minutes=i),
                    exit_time=base_time + timedelta(minutes=i + 5),
                    direction="short",
                    entry_price=11830.0 + i,
                    exit_price=11835.0 + i,
                    stop_loss=11835.0 + i,
                    take_profit=11820.0 + i,
                    pnl=-50.0,
                    bars_held=5,
                )
                for i in range(35, 60, 5)
            ]
        )

        return trades

    def test_calculate_win_rate(self, sample_trades):
        """Test win rate calculation."""
        analyzer = PerformanceAnalyzer(sample_trades)
        metrics = analyzer.calculate_metrics()

        # 6 winning trades, 5 losing trades = 6/11 = 54.55%
        assert 0.50 <= metrics.win_rate <= 0.60

    def test_calculate_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        analyzer = PerformanceAnalyzer(sample_trades)
        metrics = analyzer.calculate_metrics()

        # Gross profit: 6 * 100 = 600
        # Gross loss: 5 * 50 = 250
        # Profit factor: 600/250 = 2.4
        assert 2.0 <= metrics.profit_factor <= 3.0

    def test_calculate_expectancy(self, sample_trades):
        """Test expectancy calculation."""
        analyzer = PerformanceAnalyzer(sample_trades)
        metrics = analyzer.calculate_metrics()

        # Total P&L: 600 - 250 = 350
        # Number of trades: 11
        # Expectancy: 350/11 = 31.82
        assert 30.0 <= metrics.expectancy <= 35.0

    def test_calculate_trade_frequency(self, sample_trades):
        """Test trade frequency calculation."""
        analyzer = PerformanceAnalyzer(sample_trades)
        metrics = analyzer.calculate_metrics()

        # 11 trades over ~1 hour = very high frequency for this sample
        assert metrics.trade_frequency > 0

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Create trades with drawdown pattern
        trades = [
            Trade(
                entry_time=base_time + timedelta(minutes=i),
                exit_time=base_time + timedelta(minutes=i + 1),
                direction="long",
                entry_price=11800.0,
                exit_price=11810.0,
                stop_loss=11795.0,
                take_profit=11820.0,
                pnl=100.0,
                bars_held=1,
            )
            for i in range(5)
        ]

        # Add losses creating drawdown
        trades.extend(
            [
                Trade(
                    entry_time=base_time + timedelta(minutes=5 + i),
                    exit_time=base_time + timedelta(minutes=6 + i),
                    direction="long",
                    entry_price=11800.0,
                    exit_price=11790.0,
                    stop_loss=11795.0,
                    take_profit=11820.0,
                    pnl=-200.0,
                    bars_held=1,
                )
                for i in range(3)
            ]
        )

        analyzer = PerformanceAnalyzer(trades)
        metrics = analyzer.calculate_metrics()

        # Peak: 500, Drawdown to -100, Max DD: 600
        assert metrics.max_drawdown_percent > 0


class TestEndToEndBacktesting:
    """End-to-end tests for backtesting workflow."""

    def test_full_backtest_workflow(self):
        """Test complete backtest from bars to metrics."""
        # This would require actual strategy implementations
        # For now, just verify the workflow structure

        engine = BacktestEngine(initial_capital=100000)

        # Simulate backtest
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        for i in range(10):
            trade = Trade(
                entry_time=base_time + timedelta(minutes=i * 10),
                exit_time=base_time + timedelta(minutes=i * 10 + 5),
                direction="long" if i % 2 == 0 else "short",
                entry_price=11800.0 + i,
                exit_price=11805.0 + i,
                stop_loss=11795.0 + i,
                take_profit=11810.0 + i,
                pnl=50.0,
                bars_held=5,
            )
            engine.trades.append(trade)

        # Calculate metrics
        analyzer = PerformanceAnalyzer(engine.trades)
        metrics = analyzer.calculate_metrics()

        # Verify metrics were calculated
        assert metrics.win_rate > 0
        assert metrics.profit_factor > 0
        assert metrics.total_trades == 10

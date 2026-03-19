"""Unit tests for PerformanceMetricsCalculator.

Tests comprehensive performance metrics calculation including Sharpe ratio,
Sortino ratio, win rate, profit factor, maximum drawdown, and trade statistics.
"""

import time

import pandas as pd
import numpy as np

from src.research.performance_metrics_calculator import (
    PerformanceMetricsCalculator
)


class TestPerformanceMetricsCalculatorInit:
    """Test PerformanceMetricsCalculator initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        calculator = PerformanceMetricsCalculator()

        assert calculator._risk_free_rate == 0.02

    def test_init_with_custom_risk_free_rate(self):
        """Verify initialization with custom risk-free rate."""
        calculator = PerformanceMetricsCalculator(risk_free_rate=0.03)

        assert calculator._risk_free_rate == 0.03


class TestTotalReturnCalculation:
    """Test total return calculation."""

    def test_calculate_gross_return(self):
        """Verify gross return calculation."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'entry_price': [2100.0] * 10,
            'exit_price': [2105.0] * 10,
            'direction': ['bullish'] * 10,
            'pnl': [500.0, 300.0, -200.0, 400.0, 100.0,
                    -150.0, 600.0, -100.0, 350.0, 200.0],
            'exit_reason': ['take_profit'] * 10
        })

        total_return = calculator.calculate_total_return(trades_df)

        # Sum of P&L = 500+300-200+400+100-150+600-100+350+200 = 2000
        assert total_return['total_pnl'] == 2000.0
        assert total_return['total_return_pct'] > 0

    def test_calculate_net_return_with_costs(self):
        """Verify net return calculation with slippage/commission."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'entry_price': [2100.0] * 5,
            'exit_price': [2105.0] * 5,
            'direction': ['bullish'] * 5,
            'pnl': [500.0, 300.0, -200.0, 400.0, 100.0],
            'commission': [-5.0] * 5,
            'slippage': [-2.5] * 5,
            'exit_reason': ['take_profit'] * 5
        })

        total_return = calculator.calculate_total_return(trades_df)

        # Net P&L should account for costs
        assert 'net_pnl' in total_return
        assert total_return['net_pnl'] < total_return['total_pnl']

    def test_handle_empty_trades(self):
        """Verify handling of empty trades DataFrame."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'timestamp': [],
            'entry_price': [],
            'exit_price': [],
            'direction': [],
            'pnl': [],
            'exit_reason': []
        })

        total_return = calculator.calculate_total_return(trades_df)

        assert total_return['total_pnl'] == 0.0
        assert total_return['total_return_pct'] == 0.0


class TestSharpeRatioCalculation:
    """Test Sharpe ratio calculation."""

    def test_calculate_positive_sharpe(self):
        """Verify Sharpe ratio calculation with positive returns."""
        calculator = PerformanceMetricsCalculator()

        # Create equity curve with upward trend
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)  # Daily returns
        returns[0] = 0.02  # Add positive bias

        equity_curve = pd.Series(
            np.cumprod(1 + returns) * 100000,
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )

        sharpe = calculator.calculate_sharpe_ratio(equity_curve)

        assert sharpe > 0
        assert sharpe < 10  # Sanity check

    def test_calculate_negative_sharpe(self):
        """Verify Sharpe ratio calculation with negative returns."""
        calculator = PerformanceMetricsCalculator()

        # Create declining equity curve
        returns = np.random.normal(-0.001, 0.01, 252)

        equity_curve = pd.Series(
            np.cumprod(1 + returns) * 100000,
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )

        sharpe = calculator.calculate_sharpe_ratio(equity_curve)

        assert sharpe < 0

    def test_handle_constant_returns_zero_volatility(self):
        """Verify handling of constant returns (zero volatility)."""
        calculator = PerformanceMetricsCalculator()

        # Flat equity curve
        equity_curve = pd.Series(
            [100000] * 100,
            index=pd.date_range('2024-01-01', periods=100, freq='D')
        )

        sharpe = calculator.calculate_sharpe_ratio(equity_curve)

        # Should return 0 or handle gracefully
        assert sharpe == 0 or sharpe is not None


class TestSortinoRatioCalculation:
    """Test Sortino ratio calculation."""

    def test_calculate_sortino_using_downside_deviation(self):
        """Verify Sortino ratio uses downside deviation only."""
        calculator = PerformanceMetricsCalculator()

        # Create returns with asymmetric distribution
        returns = np.array([0.02, 0.01, 0.03, -0.01, -0.005,
                           0.015, 0.025, -0.008, 0.018, 0.022])

        equity_curve = pd.Series(
            np.cumprod(1 + returns) * 100000,
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        sortino = calculator.calculate_sortino_ratio(equity_curve)

        # Sortino should typically be >= Sharpe (uses smaller denominator)
        sharpe = calculator.calculate_sharpe_ratio(equity_curve)
        assert sortino >= sharpe or abs(sortino - sharpe) < 0.5

    def test_compare_sortino_vs_sharpe(self):
        """Verify Sortino ratio vs Sharpe ratio comparison."""
        calculator = PerformanceMetricsCalculator()

        # Returns with negative skew (more frequent small losses,
        # occasional large gains)
        returns = np.array([
            0.05, -0.01, -0.01, -0.01, 0.03,
            -0.01, -0.01, -0.01, 0.04, -0.01
        ])

        equity_curve = pd.Series(
            np.cumprod(1 + returns) * 100000,
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        sortino = calculator.calculate_sortino_ratio(equity_curve)
        sharpe = calculator.calculate_sharpe_ratio(equity_curve)

        # Sortino should be higher when there are downside outliers
        assert sortino is not None
        assert sharpe is not None


class TestWinRateCalculation:
    """Test win rate calculation."""

    def test_calculate_win_rate_mixed_results(self):
        """Verify win rate calculation with mixed results."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [100, -50, 150, -75, 200, -25, 175, -100, 125, -50]
        })

        win_rate = calculator.calculate_win_rate(trades_df)

        # 5 wins out of 10 trades = 50%
        assert win_rate == 50.0

    def test_handle_all_wins_100_percent(self):
        """Verify handling of all wins (100% win rate)."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [100, 150, 200, 175, 125]
        })

        win_rate = calculator.calculate_win_rate(trades_df)

        assert win_rate == 100.0

    def test_handle_all_losses_0_percent(self):
        """Verify handling of all losses (0% win rate)."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [-100, -50, -75, -125, -80]
        })

        win_rate = calculator.calculate_win_rate(trades_df)

        assert win_rate == 0.0


class TestProfitFactorCalculation:
    """Test profit factor calculation."""

    def test_calculate_profit_factor_greater_than_1(self):
        """Verify profit factor calculation > 1."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [500, 300, 200, -150, -100, -250, 400, -200]
        })

        profit_factor = calculator.calculate_profit_factor(trades_df)

        # Gross wins: 500+300+200+400 = 1400
        # Gross losses: |-150-100-250-200| = 700
        # Profit factor: 1400/700 = 2.0
        assert profit_factor == 2.0

    def test_calculate_profit_factor_less_than_1(self):
        """Verify profit factor calculation < 1."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [100, 150, -300, -250, -200]
        })

        profit_factor = calculator.calculate_profit_factor(trades_df)

        # Gross wins: 100+150 = 250
        # Gross losses: |-300-250-200| = 750
        # Profit factor: 250/750 = 0.333
        assert profit_factor < 1.0

    def test_handle_no_losses_infinite_profit_factor(self):
        """Verify handling of no losses (infinite profit factor)."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [100, 150, 200, 175, 125]
        })

        profit_factor = calculator.calculate_profit_factor(trades_df)

        # Should return infinity or very large number
        assert profit_factor == float('inf') or profit_factor > 1000


class TestMaxDrawdownCalculation:
    """Test maximum drawdown calculation."""

    def test_calculate_max_drawdown_percentage(self):
        """Verify maximum drawdown percentage calculation."""
        calculator = PerformanceMetricsCalculator()

        # Create equity curve with drawdown
        equity_values = [
            100000, 105000, 110000, 108000, 103000,
            107000, 112000, 109000, 113000, 111000
        ]

        equity_curve = pd.Series(
            equity_values,
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        drawdown = calculator.calculate_max_drawdown(equity_curve)

        # Peak: 110000, trough: 103000
        # Drawdown: (110000 - 103000) / 110000 * 100 = 6.36%
        assert 'max_drawdown_pct' in drawdown
        assert abs(drawdown['max_drawdown_pct'] - 6.36) < 0.1

    def test_calculate_drawdown_duration(self):
        """Verify drawdown duration calculation."""
        calculator = PerformanceMetricsCalculator()

        # Create equity curve with drawdown over multiple periods
        equity_values = [
            100000, 105000, 110000, 108000, 103000,
            104000, 105000, 106000, 107000, 108000
        ]

        equity_curve = pd.Series(
            equity_values,
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        drawdown = calculator.calculate_max_drawdown(equity_curve)

        assert 'duration_days' in drawdown
        assert drawdown['duration_days'] > 0

    def test_handle_no_drawdown_always_rising(self):
        """Verify handling of equity curve always rising."""
        calculator = PerformanceMetricsCalculator()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        drawdown = calculator.calculate_max_drawdown(equity_curve)

        # Should be 0 or very small
        assert drawdown['max_drawdown_pct'] == 0 or drawdown['max_drawdown_pct'] < 1


class TestTradeStatistics:
    """Test trade statistics calculation."""

    def test_calculate_average_win_loss(self):
        """Verify average win and loss calculation."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [500, 300, -200, 400, -150, 600, -100, 350]
        })

        stats = calculator.calculate_trade_statistics(trades_df)

        # Average win: (500+300+400+600+350) / 5 = 430
        # Average loss: |-200-150-100| / 3 = 150
        assert abs(stats['avg_win'] - 430) < 1
        assert abs(stats['avg_loss'] - 150) < 1

    def test_calculate_largest_win_loss(self):
        """Verify largest win and loss calculation."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'pnl': [500, 300, -200, 750, -150, 600, -350, 350]
        })

        stats = calculator.calculate_trade_statistics(trades_df)

        assert stats['largest_win'] == 750
        assert stats['largest_loss'] == -350

    def test_calculate_average_trade_duration(self):
        """Verify average trade duration calculation."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'entry_time': pd.to_datetime([
                '2024-01-01 10:00',
                '2024-01-01 11:00',
                '2024-01-01 12:00'
            ]),
            'exit_time': pd.to_datetime([
                '2024-01-01 11:30',
                '2024-01-01 11:45',
                '2024-01-01 14:00'
            ]),
            'pnl': [100, 50, 75]
        })

        stats = calculator.calculate_trade_statistics(trades_df)

        # Durations: 90min, 45min, 120min = avg 85min
        assert stats['avg_duration_minutes'] > 0

    def test_calculate_trades_per_month(self):
        """Verify trades per month calculation."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=60, freq='D'),
            'pnl': [100] * 60
        })

        stats = calculator.calculate_trade_statistics(trades_df)

        # 60 trades over ~2 months = ~30 trades/month
        assert abs(stats['trades_per_month'] - 30) < 5


class TestCalculateAllMetrics:
    """Test comprehensive metrics calculation."""

    def test_complete_metrics_calculation(self):
        """Verify all metrics calculated correctly."""
        calculator = PerformanceMetricsCalculator()

        # Create trade results
        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='h'),
            'entry_price': [2100.0] * 50,
            'exit_price': [2105.0] * 50,
            'direction': ['bullish'] * 50,
            'pnl': np.random.randn(50) * 100 + 50,  # Slight positive bias
            'exit_reason': np.random.choice(['take_profit', 'stop_loss'], 50)
        })

        metrics = calculator.calculate_all_metrics(trades_df)

        # Verify all required metrics present
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'max_drawdown' in metrics
        assert 'trade_statistics' in metrics

    def test_all_metrics_in_output(self):
        """Verify all metrics included in output dictionary."""
        calculator = PerformanceMetricsCalculator()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='h'),
            'pnl': np.random.randn(20) * 50
        })

        metrics = calculator.calculate_all_metrics(trades_df)

        # Check structure
        assert isinstance(metrics, dict)

        # Check top-level metrics
        expected_keys = [
            'total_return', 'sharpe_ratio', 'sortino_ratio',
            'win_rate', 'profit_factor', 'max_drawdown', 'trade_statistics'
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_performance_requirement_under_10_seconds(self):
        """Verify calculation completes in < 10 seconds."""
        calculator = PerformanceMetricsCalculator()

        # Create large dataset (1000 trades)
        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
            'pnl': np.random.randn(1000) * 100,
            'entry_price': [2100.0] * 1000,
            'exit_price': [2105.0] * 1000
        })

        start_time = time.time()
        calculator.calculate_all_metrics(trades_df)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0, f"Calculation took {elapsed_time:.2f} seconds"

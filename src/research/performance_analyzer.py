"""Performance Analyzer for backtesting results.

This module calculates comprehensive performance metrics from
backtest trades.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.research.backtest_engine import Trade

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics.

    Attributes:
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: Win rate as percentage (0-1)
        profit_factor: Gross profit / gross loss
        avg_risk_reward: Average risk-reward ratio
        expectancy: Average profit/loss per trade in dollars
        trade_frequency: Average trades per day
        avg_hold_time_bars: Average bars held per trade
        max_drawdown_percent: Maximum drawdown as percentage
        sharpe_ratio: Sharpe ratio
        total_pnl: Total profit/loss in dollars
        final_capital: Final capital after all trades
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_risk_reward: float
    expectancy: float
    trade_frequency: float
    avg_hold_time_bars: float
    max_drawdown_percent: float
    sharpe_ratio: float
    total_pnl: float
    final_capital: float


class PerformanceAnalyzer:
    """Analyzes backtest results and calculates performance metrics."""

    def __init__(self, trades: list[Trade], initial_capital: float = 100000.0) -> None:
        """Initialize performance analyzer.

        Args:
            trades: List of completed trades
            initial_capital: Starting capital
        """
        self.trades = trades
        self.initial_capital = initial_capital

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics.

        Returns:
            PerformanceMetrics object with all metrics
        """
        if not self.trades:
            return self._empty_metrics()

        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl < 0)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average risk-reward
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        avg_risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Expectancy
        total_pnl = sum(t.pnl for t in self.trades)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0.0

        # Trade frequency (trades per day)
        if len(self.trades) >= 2:
            time_span = self.trades[-1].entry_time - self.trades[0].entry_time
            days = max(time_span.total_seconds() / 86400, 1)  # At least 1 day
            trade_frequency = total_trades / days
        else:
            trade_frequency = 0.0

        # Average hold time
        avg_hold_time_bars = (
            sum(t.bars_held for t in self.trades) / total_trades if total_trades > 0 else 0.0
        )

        # Maximum drawdown
        max_drawdown_percent = self._calculate_max_drawdown()

        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Final capital
        final_capital = self.initial_capital + total_pnl

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_risk_reward=avg_risk_reward,
            expectancy=expectancy,
            trade_frequency=trade_frequency,
            avg_hold_time_bars=avg_hold_time_bars,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            total_pnl=total_pnl,
            final_capital=final_capital,
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage.

        Returns:
            Maximum drawdown as percentage (0-100)
        """
        if not self.trades:
            return 0.0

        peak = self.initial_capital
        max_drawdown = 0.0
        current_capital = self.initial_capital

        for trade in self.trades:
            current_capital += trade.pnl

            # Update peak
            if current_capital > peak:
                peak = current_capital

            # Calculate drawdown
            drawdown = (peak - current_capital) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate simplified Sharpe ratio.

        Returns:
            Sharpe ratio (annualized)
        """
        if len(self.trades) < 2:
            return 0.0

        # Calculate returns
        returns = [t.pnl / self.initial_capital for t in self.trades]

        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance**0.5

        if std_dev == 0:
            return 0.0

        # Simplified Sharpe (not annualized for simplicity)
        sharpe = avg_return / std_dev

        return sharpe

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics object.

        Returns:
            PerformanceMetrics with all zeros
        """
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_risk_reward=0.0,
            expectancy=0.0,
            trade_frequency=0.0,
            avg_hold_time_bars=0.0,
            max_drawdown_percent=0.0,
            sharpe_ratio=0.0,
            total_pnl=0.0,
            final_capital=self.initial_capital,
        )

    def generate_summary(self) -> str:
        """Generate a human-readable summary of performance.

        Returns:
            Formatted summary string
        """
        metrics = self.calculate_metrics()

        summary = f"""
Backtest Performance Summary
============================
Total Trades: {metrics.total_trades}
Winning Trades: {metrics.winning_trades}
Losing Trades: {metrics.losing_trades}

Performance Metrics:
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Expectancy: ${metrics.expectancy:.2f} per trade
- Total P&L: ${metrics.total_pnl:.2f}
- Final Capital: ${metrics.final_capital:.2f}

Risk Metrics:
- Max Drawdown: {metrics.max_drawdown_percent:.2f}%
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}

Trading Characteristics:
- Trade Frequency: {metrics.trade_frequency:.2f} trades/day
- Avg Hold Time: {metrics.avg_hold_time_bars:.1f} bars
- Avg Risk-Reward: {metrics.avg_risk_reward:.2f}
"""
        return summary

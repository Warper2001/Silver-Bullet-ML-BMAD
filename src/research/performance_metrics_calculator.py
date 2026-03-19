"""Performance Metrics Calculator for backtest results.

Calculates comprehensive performance metrics including Sharpe ratio,
Sortino ratio, win rate, profit factor, maximum drawdown, and trade
statistics.

Performance: Completes in < 10 seconds for typical backtest results.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """Calculate comprehensive performance metrics for backtest results.

    Calculates Sharpe ratio, Sortino ratio, win rate, profit factor,
    maximum drawdown, trade statistics, and more.

    Performance: Completes in < 10 seconds for typical backtest results.
    """

    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self._risk_free_rate = risk_free_rate
        logger.debug(
            f"PerformanceMetricsCalculator initialized: "
            f"risk_free_rate={risk_free_rate}"
        )

    def calculate_all_metrics(self, trades_df: pd.DataFrame) -> dict[str, Any]:
        """Calculate all performance metrics for trade results.

        Args:
            trades_df: DataFrame with trade results (columns: timestamp,
                entry_price, exit_price, direction, pnl, exit_reason)

        Returns:
            Dictionary with all performance metrics
        """
        logger.info(f"Calculating all metrics for {len(trades_df)} trades...")

        # Calculate all metrics
        total_return = self.calculate_total_return(trades_df)

        # Build equity curve for ratio calculations
        equity_curve = self._build_equity_curve(trades_df)

        sharpe_ratio = self.calculate_sharpe_ratio(equity_curve)
        sortino_ratio = self.calculate_sortino_ratio(equity_curve)
        win_rate = self.calculate_win_rate(trades_df)
        profit_factor = self.calculate_profit_factor(trades_df)
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        trade_stats = self.calculate_trade_statistics(trades_df)

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trade_statistics': trade_stats
        }

        logger.info("All metrics calculated successfully")
        return metrics

    def calculate_total_return(self, trades_df: pd.DataFrame) -> dict[str, float]:
        """Calculate total gross and net return.

        Args:
            trades_df: DataFrame with trade results

        Returns:
            Dictionary with total_pnl, net_pnl, total_return_pct
        """
        if len(trades_df) == 0:
            return {
                'total_pnl': 0.0,
                'net_pnl': 0.0,
                'total_return_pct': 0.0
            }

        # Calculate total P&L
        total_pnl = trades_df['pnl'].sum()

        # Calculate net P&L (including commission and slippage if present)
        net_pnl = total_pnl
        if 'commission' in trades_df.columns:
            net_pnl += trades_df['commission'].sum()
        if 'slippage' in trades_df.columns:
            net_pnl += trades_df['slippage'].sum()

        # Calculate return percentage (assuming 100K initial capital)
        initial_capital = 100000.0
        total_return_pct = (net_pnl / initial_capital) * 100

        return {
            'total_pnl': float(total_pnl),
            'net_pnl': float(net_pnl),
            'total_return_pct': float(total_return_pct)
        }

    def calculate_sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """Calculate Sharpe ratio (annualized).

        Formula: (annual_return - risk_free_rate) / annual_std_dev

        Args:
            equity_curve: Series of cumulative equity values

        Returns:
            Sharpe ratio
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize returns (assuming daily data)
        trading_days_per_year = 252
        annual_return = returns.mean() * trading_days_per_year
        annual_std = returns.std() * np.sqrt(trading_days_per_year)

        # Calculate Sharpe ratio
        sharpe = (annual_return - self._risk_free_rate) / annual_std

        return float(sharpe)

    def calculate_sortino_ratio(self, equity_curve: pd.Series) -> float:
        """Calculate Sortino ratio (annualized, downside deviation only).

        Formula: (annual_return - risk_free_rate) / annual_downside_deviation

        Args:
            equity_curve: Series of cumulative equity values

        Returns:
            Sortino ratio
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        # Filter downside returns (negative returns only)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            # No downside volatility, return high value
            annual_return = returns.mean() * 252
            if annual_return > self._risk_free_rate:
                return float('inf')
            return 0.0

        # Calculate downside deviation
        trading_days_per_year = 252
        annual_return = returns.mean() * trading_days_per_year
        annual_downside_dev = (
            downside_returns.std() * np.sqrt(trading_days_per_year)
        )

        if annual_downside_dev == 0:
            return 0.0

        # Calculate Sortino ratio
        sortino = (annual_return - self._risk_free_rate) / annual_downside_dev

        return float(sortino)

    def calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate percentage.

        Args:
            trades_df: DataFrame with trade results

        Returns:
            Win rate as percentage
        """
        if len(trades_df) == 0:
            return 0.0

        # Count wins (trades with P&L > 0)
        wins = (trades_df['pnl'] > 0).sum()
        total_trades = len(trades_df)

        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

        return float(win_rate)

    def calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross wins / gross losses).

        Args:
            trades_df: DataFrame with trade results

        Returns:
            Profit factor
        """
        if len(trades_df) == 0:
            return 0.0

        # Separate wins and losses
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']

        gross_wins = wins.sum()
        gross_losses = abs(losses.sum()) if len(losses) > 0 else 0

        if gross_losses == 0:
            # No losses, infinite profit factor
            return float('inf') if gross_wins > 0 else 0.0

        profit_factor = gross_wins / gross_losses

        return float(profit_factor)

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> dict[str, float]:
        """Calculate maximum drawdown and duration.

        Args:
            equity_curve: Series of cumulative equity values

        Returns:
            Dictionary with max_drawdown_pct and duration_days
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown_pct': 0.0,
                'duration_days': 0
            }

        # Calculate running maximum (peak values)
        running_max = equity_curve.cummax()

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max * 100

        # Find maximum drawdown
        max_drawdown_pct = float(drawdown.min())

        # Find duration of max drawdown
        min_idx = drawdown.idxmin()
        peak_idx = equity_curve[:min_idx].idxmax()

        # Calculate duration in days
        if isinstance(min_idx, pd.Timestamp) and isinstance(
            peak_idx, pd.Timestamp
        ):
            duration_days = (min_idx - peak_idx).days
        else:
            # Use index difference if not timestamps
            duration_days = min_idx - peak_idx

        return {
            'max_drawdown_pct': abs(max_drawdown_pct),
            'duration_days': int(duration_days)
        }

    def calculate_trade_statistics(self, trades_df: pd.DataFrame) -> dict[str, float]:
        """Calculate trade statistics.

        Args:
            trades_df: DataFrame with trade results

        Returns:
            Dictionary with trade statistics
        """
        if len(trades_df) == 0:
            return {
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration_minutes': 0.0,
                'trades_per_month': 0.0
            }

        # Separate wins and losses
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']

        # Average win/loss
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0

        # Largest win/loss
        largest_win = float(trades_df['pnl'].max())
        largest_loss = float(trades_df['pnl'].min())

        # Average trade duration
        avg_duration_minutes = 0.0
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            durations = (
                trades_df['exit_time'] - trades_df['entry_time']
            ).dt.total_seconds() / 60
            avg_duration_minutes = float(durations.mean())

        # Trades per month
        if 'timestamp' in trades_df.columns:
            # Calculate time span in months
            time_span = (
                trades_df['timestamp'].max() - trades_df['timestamp'].min()
            )
            months = max(time_span.days / 30.44, 1)  # Avoid division by zero
            trades_per_month = len(trades_df) / months
        else:
            trades_per_month = 0.0

        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration_minutes': avg_duration_minutes,
            'trades_per_month': trades_per_month
        }

    def _build_equity_curve(self, trades_df: pd.DataFrame) -> pd.Series:
        """Build equity curve from trade results.

        Args:
            trades_df: DataFrame with trade results

        Returns:
            Series of cumulative equity values
        """
        if len(trades_df) == 0:
            return pd.Series([100000.0])

        # Sort by timestamp if available
        if 'timestamp' in trades_df.columns:
            trades_df = trades_df.sort_values('timestamp')

        # Calculate cumulative P&L
        cumulative_pnl = trades_df['pnl'].cumsum()

        # Add initial capital
        initial_capital = 100000.0
        equity_curve = initial_capital + cumulative_pnl

        return equity_curve

    def format_metrics_dict(self, metrics: dict[str, Any]) -> dict[str, str]:
        """Format metrics with human-readable descriptions.

        Args:
            metrics: Raw metrics dictionary

        Returns:
            Formatted metrics dictionary
        """
        formatted = {}

        # Total return
        total_return = metrics['total_return']
        formatted['Total Return'] = (
            f"${total_return['net_pnl']:,.2f} "
            f"({total_return['total_return_pct']:.2f}%)"
        )

        # Sharpe ratio
        sharpe = metrics['sharpe_ratio']
        formatted['Sharpe Ratio'] = f"{sharpe:.2f}"

        # Sortino ratio
        sortino = metrics['sortino_ratio']
        if sortino == float('inf'):
            formatted['Sortino Ratio'] = "∞ (no downside volatility)"
        else:
            formatted['Sortino Ratio'] = f"{sortino:.2f}"

        # Win rate
        win_rate = metrics['win_rate']
        formatted['Win Rate'] = f"{win_rate:.1f}%"

        # Profit factor
        profit_factor = metrics['profit_factor']
        if profit_factor == float('inf'):
            formatted['Profit Factor'] = "∞ (no losses)"
        else:
            formatted['Profit Factor'] = f"{profit_factor:.2f}"

        # Max drawdown
        drawdown = metrics['max_drawdown']
        formatted['Max Drawdown'] = (
            f"{drawdown['max_drawdown_pct']:.2f}% "
            f"({drawdown['duration_days']} days)"
        )

        # Trade statistics
        stats = metrics['trade_statistics']
        formatted['Average Win'] = f"${stats['avg_win']:,.2f}"
        formatted['Average Loss'] = f"${stats['avg_loss']:,.2f}"
        formatted['Largest Win'] = f"${stats['largest_win']:,.2f}"
        formatted['Largest Loss'] = f"${stats['largest_loss']:,.2f}"
        formatted['Avg Trade Duration'] = f"{stats['avg_duration_minutes']:.0f} min"
        formatted['Trades Per Month'] = f"{stats['trades_per_month']:.1f}"

        return formatted

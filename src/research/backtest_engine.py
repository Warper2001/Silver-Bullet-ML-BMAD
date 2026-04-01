"""Backtest Engine for strategy testing.

This module provides a backtesting engine to test trading strategies
on historical data and track trade performance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade.

    Attributes:
        entry_time: When the trade was entered
        exit_time: When the trade was exited
        direction: "long" or "short"
        entry_price: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        take_profit: Take profit price
        pnl: Profit/loss in dollars
        bars_held: Number of bars held
    """

    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    bars_held: int


class BacktestEngine:
    """Backtesting engine for strategy testing.

    Attributes:
        initial_capital: Starting capital for backtest
        trades: List of completed trades
        current_capital: Current capital after trades
    """

    MNQ_TICK_VALUE = 0.25  # $0.25 per tick for MNQ
    CONTRACT_MULTIPLIER = 20  # MNQ contract multiplier

    def __init__(self, initial_capital: float = 100000.0) -> None:
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital (default $100,000)
        """
        self.initial_capital = initial_capital
        self.trades: list[Trade] = []
        self.current_capital = initial_capital

    def add_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        bars_held: int,
    ) -> None:
        """Add a completed trade to the backtest.

        Args:
            entry_time: Trade entry time
            exit_time: Trade exit time
            direction: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            bars_held: Number of bars held
        """
        # Calculate P&L
        if direction == "long":
            price_diff = exit_price - entry_price
        else:  # short
            price_diff = entry_price - exit_price

        # MNQ: $0.25 per tick * 20 ticks per point = $5 per point
        pnl = price_diff * 5.0  # $5 per point for MNQ

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl,
            bars_held=bars_held,
        )

        self.trades.append(trade)
        self.current_capital += pnl

        logger.debug(
            f"Trade added: {direction} {entry_price:.2f} -> {exit_price:.2f}, "
            f"P&L: ${pnl:.2f}"
        )

    def get_total_pnl(self) -> float:
        """Get total P&L from all trades.

        Returns:
            Total P&L in dollars
        """
        return sum(trade.pnl for trade in self.trades)

    def get_win_count(self) -> int:
        """Get number of winning trades.

        Returns:
            Count of winning trades
        """
        return sum(1 for trade in self.trades if trade.pnl > 0)

    def get_loss_count(self) -> int:
        """Get number of losing trades.

        Returns:
            Count of losing trades
        """
        return sum(1 for trade in self.trades if trade.pnl < 0)

    def get_total_trades(self) -> int:
        """Get total number of trades.

        Returns:
            Total trade count
        """
        return len(self.trades)

    def get_all_trades(self) -> list[Trade]:
        """Get all completed trades.

        Returns:
            List of all trades
        """
        return self.trades

    def reset(self) -> None:
        """Reset the backtest engine."""
        self.trades = []
        self.current_capital = self.initial_capital
        logger.debug("Backtest engine reset")

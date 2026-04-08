"""Position tracking with mark-to-market P&L for SIM paper trading.

This module enhances position tracking with real-time P&L calculation
using SIM quote data for position valuation.

Features:
- Mark-to-market P&L calculation
- Real-time position valuation
- Unrealized P&L tracking
- Position state management
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from src.execution.tradestation.market_data.streaming import StreamPosition

logger = logging.getLogger(__name__)


@dataclass
class PositionWithPnL:
    """Active position with mark-to-market P&L tracking.

    Attributes:
        order_id: TradeStation order ID
        signal_id: Signal identifier that triggered this position
        symbol: Trading symbol (e.g., 'MNQH26')
        entry_price: Initial entry price in dollars
        quantity: Target quantity (contracts)
        direction: Signal direction ("bullish" or "bearish")
        timestamp: Position creation timestamp
        filled_quantity: Quantity filled so far
        current_price: Current market price for mark-to-market
        unrealized_pnl: Unrealized P&L in dollars
        unrealized_pnl_percent: Unrealized P&L as percentage
        status: Position status
    """
    order_id: str
    signal_id: str
    symbol: str
    entry_price: float
    quantity: int
    direction: str
    timestamp: datetime
    filled_quantity: int = field(init=False)
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    status: str = "OPEN"

    # MNQ contract multiplier (0.5 points per dollar)
    CONTRACT_MULTIPLIER: float = 0.5

    def __post_init__(self):
        """Initialize derived fields after creation."""
        self.filled_quantity = self.quantity
        self.current_price = self.entry_price

    def update_mark_to_market(self, current_price: float) -> None:
        """Update mark-to-market P&L based on current price.

        Args:
            current_price: Current market price
        """
        self.current_price = current_price

        # Calculate unrealized P&L
        price_diff = current_price - self.entry_price

        if self.direction.lower() in ["bullish", "long", "buy"]:
            # Long position
            self.unrealized_pnl = price_diff * self.quantity * self.CONTRACT_MULTIPLIER
        else:
            # Short position
            self.unrealized_pnl = -price_diff * self.quantity * self.CONTRACT_MULTIPLIER

        # Calculate P&L percentage
        if self.entry_price > 0:
            self.unrealized_pnl_percent = (self.unrealized_pnl / (self.entry_price * self.quantity * self.CONTRACT_MULTIPLIER)) * 100

        logger.debug(
            f"Position {self.order_id} mark-to-market: "
            f"P&L=${self.unrealized_pnl:.2f} ({self.unrealized_pnl_percent:.2f}%)"
        )

    def close_position(self, exit_price: float) -> float:
        """Close position and calculate final P&L.

        Args:
            exit_price: Exit price

        Returns:
            Realized P&L in dollars
        """
        # Update to exit price first
        self.update_mark_to_market(exit_price)

        # Calculate realized P&L
        realized_pnl = self.unrealized_pnl

        # Update status
        self.status = "CLOSED"

        logger.info(
            f"Position {self.order_id} closed: "
            f"Realized P&L=${realized_pnl:.2f}"
        )

        return realized_pnl


class PositionTrackerWithPnL:
    """Tracks active positions with mark-to-market P&L.

    Maintains in-memory storage of active positions with real-time
    P&L calculation using current market prices.

    Attributes:
        _positions: Dictionary mapping order_id to PositionWithPnL
        _total_unrealized_pnl: Total unrealized P&L across all positions
        _total_realized_pnl: Total realized P&L from closed positions

    Example:
        >>> tracker = PositionTrackerWithPnL()
        >>> position = PositionWithPnL(...)
        >>> tracker.add_position(position)
        >>> tracker.update_from_quote(StreamPosition(...))
        >>> print(tracker.total_unrealized_pnl)
    """

    def __init__(self) -> None:
        """Initialize position tracker."""
        self._positions: Dict[str, PositionWithPnL] = {}
        self._total_unrealized_pnl = 0.0
        self._total_realized_pnl = 0.0

    def add_position(self, position: PositionWithPnL) -> None:
        """Add a new position to tracking.

        Args:
            position: PositionWithPnL to track
        """
        self._positions[position.order_id] = position
        # Recalculate total P&L to include this position
        self._recalculate_total_pnl()
        logger.info(f"Added position {position.order_id} to tracking")

    def update_from_quote(self, quote: StreamPosition) -> None:
        """Update all positions for a symbol with current quote.

        Args:
            quote: StreamPosition with current market data
        """
        symbol_positions = [
            pos for pos in self._positions.values()
            if pos.symbol == quote.symbol and pos.status == "OPEN"
        ]

        for position in symbol_positions:
            position.update_mark_to_market(quote.last_price)

        # Recalculate total unrealized P&L
        self._recalculate_total_pnl()

    def _recalculate_total_pnl(self) -> None:
        """Recalculate total unrealized P&L across all positions."""
        self._total_unrealized_pnl = sum(
            pos.unrealized_pnl
            for pos in self._positions.values()
            if pos.status == "OPEN"
        )

    def close_position(self, order_id: str, exit_price: float) -> float:
        """Close a position and record realized P&L.

        Args:
            order_id: Order ID to close
            exit_price: Exit price

        Returns:
            Realized P&L in dollars

        Raises:
            KeyError: If order_id not found
        """
        if order_id not in self._positions:
            raise KeyError(f"Position {order_id} not found")

        position = self._positions[order_id]
        realized_pnl = position.close_position(exit_price)

        # Add to total realized P&L
        self._total_realized_pnl += realized_pnl

        # Recalculate total unrealized P&L
        self._recalculate_total_pnl()

        return realized_pnl

    def get_position(self, order_id: str) -> Optional[PositionWithPnL]:
        """Get position by order ID.

        Args:
            order_id: Order ID to retrieve

        Returns:
            PositionWithPnL or None if not found
        """
        return self._positions.get(order_id)

    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all open positions."""
        return self._total_unrealized_pnl

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L from closed positions."""
        return self._total_realized_pnl

    @property
    def open_position_count(self) -> int:
        """Get number of open positions."""
        return sum(
            1 for pos in self._positions.values()
            if pos.status == "OPEN"
        )

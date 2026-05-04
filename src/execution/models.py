"""Pydantic models for trade execution, entry decisions, and position tracking."""

import pytz
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

from src.detection.models import EnsembleTradeSignal

# Target timezone for all ensemble processing
NY_TZ = pytz.timezone("America/New_York")

def _now() -> datetime:
    """Get current time in New York timezone."""
    return datetime.now(NY_TZ)

class TradeOrder(BaseModel):
    """Trade order for execution with position lifecycle tracking."""

    trade_id: str = Field(..., description="Unique identifier for this trade")
    symbol: str = Field(default="MNQ", description="Trading symbol")
    direction: Literal["long", "short"] = Field(..., description="Trade direction")
    quantity: int = Field(..., ge=1, le=5, description="Number of contracts (1-5)")
    order_type: Literal["market", "limit"] = Field(..., description="Order type")
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price (if limit order)")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profit: float = Field(..., gt=0, description="Take profit price level")
    timestamp: datetime = Field(default_factory=_now, description="Order creation timestamp")
    status: Literal["pending", "submitted", "filled", "rejected", "cancelled"] = Field(
        default="pending", description="Order status"
    )
    ensemble_signal: EnsembleTradeSignal = Field(
        ..., description="Reference to ensemble signal"
    )
    position_size: int = Field(..., ge=1, le=5, description="Position size in contracts")

    # Exit tracking fields
    entry_time: datetime = Field(default_factory=_now, description="Time position was entered")
    exit_time: Optional[datetime] = Field(None, description="Time position was exited")
    exit_price: Optional[float] = Field(None, gt=0, description="Price at exit")
    exit_reason: Optional[str] = Field(None, description="Reason for exit")
    hold_time_seconds: Optional[int] = Field(None, ge=0, description="Hold time in seconds")
    realized_pnl: Optional[float] = Field(None, description="Realized P&L in USD")
    realized_pnl_ticks: Optional[float] = Field(None, description="Realized P&L in ticks")
    rr_achieved: Optional[float] = Field(None, description="R:R achieved")
    position_state: Literal["open", "partially_closed", "closed"] = Field(
        default="open", description="Current position state"
    )
    original_quantity: int = Field(default=1, ge=1, le=5, description="Original quantity")
    remaining_quantity: int = Field(default=1, ge=0, le=5, description="Remaining quantity")

    @model_validator(mode="after")
    def sync_quantities(self) -> "TradeOrder":
        """Ensure quantities are initialized correctly if omitted."""
        if self.original_quantity == 1 and self.quantity > 1:
            self.original_quantity = self.quantity
            self.remaining_quantity = self.quantity
        return self

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_respect_direction(cls, v: float, info) -> float:
        """Validate stop loss position (Allows breakeven Decision 1A)."""
        direction = info.data.get("direction")
        entry = info.data.get("entry_price")

        if entry is not None and direction is not None:
            # Relaxed: Allow SL == entry for breakeven trailing stops
            if direction == "long" and v > entry:
                raise ValueError("stop_loss must be <= entry_price for long positions")
            if direction == "short" and v < entry:
                raise ValueError("stop_loss must be >= entry_price for short positions")
        return v

    @field_validator("remaining_quantity")
    @classmethod
    def remaining_quantity_cannot_exceed_original(cls, v: int, info) -> int:
        """Validate remaining quantity doesn't exceed original."""
        original = info.data.get("original_quantity")
        if original is not None and v > original:
            raise ValueError("remaining_quantity cannot exceed original_quantity")
        return v

    def notional_value(self) -> float:
        """Calculate contract notional value."""
        # Patch 8: Parameterize multipliers (default to MNQ $0.50)
        multipliers = {"MNQ": 0.50, "NQ": 20.0, "MES": 5.0, "ES": 50.0}
        multiplier = multipliers.get(self.symbol, 0.50)
        return self.entry_price * multiplier * self.quantity

    def risk_per_contract(self) -> float:
        """Calculate risk per contract in points."""
        return abs(self.entry_price - self.stop_loss)

    def hold_time_minutes(self, current_time: Optional[datetime] = None) -> float:
        """Calculate hold time in minutes (supports aware datetimes)."""
        if self.exit_time:
            delta = self.exit_time - self.entry_time
        else:
            # Patch 4 & 11: Use provided time or aware now
            now = current_time or _now()
            delta = now - self.entry_time
        return delta.total_seconds() / 60.0

    def is_held_max_time(self, max_hold_minutes: float = 120.0, current_time: Optional[datetime] = None) -> bool:
        """Check if position has exceeded maximum hold time."""
        return self.hold_time_minutes(current_time) >= max_hold_minutes

    def is_at_take_profit(self, current_price: float) -> bool:
        """Check if current price has hit take profit level.

        Args:
            current_price: Current market price

        Returns:
            True if take profit level has been hit
        """
        if self.direction == "long":
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit

    def is_at_stop_loss(self, current_price: float) -> bool:
        """Check if current price has hit stop loss level.

        Args:
            current_price: Current market price

        Returns:
            True if stop loss level has been hit
        """
        if self.direction == "long":
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss

    def is_at_hybrid_partial(self, current_price: float, partial_rr: float = 1.5) -> bool:
        """Check if current price has hit hybrid partial exit level (1.5R).

        Args:
            current_price: Current market price
            partial_rr: Partial R:R ratio (default 1.5)

        Returns:
            True if hybrid partial level has been hit
        """
        risk = abs(self.entry_price - self.stop_loss)
        partial_target = self.entry_price + (risk * partial_rr) if self.direction == "long" else self.entry_price - (risk * partial_rr)

        if self.direction == "long":
            return current_price >= partial_target
        else:  # short
            return current_price <= partial_target


class EntryDecision(BaseModel):
    """Entry decision with risk validation results.

    This model represents the decision to accept or reject a trade signal
    based on risk checks. It includes detailed information about the
    validation process.

    Attributes:
        signal: Ensemble trade signal being evaluated
        position_size: Position size in contracts (0 if rejected)
        risk_checks_passed: Whether all risk checks passed
        risk_check_details: Dictionary with detailed results of each risk check
        decision: Entry decision (ACCEPT or REJECT)
        rejection_reason: Reason for rejection (if applicable)
        timestamp: Decision timestamp
    """

    signal: EnsembleTradeSignal = Field(..., description="Ensemble trade signal being evaluated")
    position_size: int = Field(..., ge=0, le=5, description="Position size in contracts (0 if rejected)")
    risk_checks_passed: bool = Field(..., description="Whether all risk checks passed")
    risk_check_details: dict = Field(..., description="Detailed risk check results")
    decision: Literal["ACCEPT", "REJECT"] = Field(..., description="Entry decision")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection")
    timestamp: datetime = Field(..., description="Decision timestamp")

    @field_validator("position_size")
    @classmethod
    def position_size_must_match_decision(cls, v: int, info) -> int:
        """Validate position size matches decision."""
        decision = info.data.get("decision")
        if decision == "REJECT" and v != 0:
            raise ValueError("position_size must be 0 for REJECT decisions")
        if decision == "ACCEPT" and v == 0:
            raise ValueError("position_size must be >= 1 for ACCEPT decisions")
        return v


class ExitOrder(BaseModel):
    """Exit order for closing positions."""

    position_id: str = Field(..., description="Position ID to exit")
    exit_type: Literal["full", "partial"] = Field(..., description="Exit type")
    quantity: int = Field(..., ge=1, le=5, description="Number of contracts to close")
    exit_price: float = Field(..., gt=0, description="Exit price")
    exit_reason: str = Field(..., description="Reason for exit (Human-readable Decision 2A)")
    timestamp: datetime = Field(default_factory=_now, description="Exit order timestamp")
    pnl: float = Field(..., description="Profit/loss for this exit in USD")
    pnl_ticks: float = Field(default=0.0, description="Profit/loss for this exit in ticks")
    rr_ratio: float = Field(..., description="Reward-risk ratio achieved")

class PositionMonitoringState(BaseModel):
    """Position state for exit monitoring."""

    position: TradeOrder = Field(..., description="Position being monitored")
    current_price: float = Field(..., gt=0, description="Current market price")
    current_time: datetime = Field(default_factory=_now, description="Current evaluation time (Aware)")
    unrealized_pnl: float = Field(..., description="Current unrealized P&L in USD")
    unrealized_pnl_ticks: float = Field(default=0.0, description="Current unrealized P&L in ticks")
    time_since_entry_seconds: int = Field(..., ge=0, description="Time since entry in seconds")
    distance_to_tp: float = Field(..., description="Distance to take profit")
    distance_to_sl: float = Field(..., description="Distance to stop loss")
    rr_achieved: float = Field(..., description="Current R:R achieved")

    def hold_time_minutes(self) -> float:
        """Calculate hold time in minutes."""
        return self.time_since_entry_seconds / 60.0

    def is_at_max_hold_time(self, max_hold_minutes: float = 120.0) -> bool:
        """Check if position has exceeded maximum hold time."""
        return self.hold_time_minutes() >= max_hold_minutes


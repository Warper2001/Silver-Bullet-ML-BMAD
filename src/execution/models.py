"""Pydantic models for trade execution, entry decisions, and position tracking."""

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, computed_field

from src.detection.models import EnsembleTradeSignal


class TradeOrder(BaseModel):
    """Trade order for execution with position lifecycle tracking.

    This model represents a complete trade order with all metadata needed
    for execution, monitoring, and exit management. Orders are created by
    entry logic, monitored for exit conditions, and passed to execution.

    Attributes:
        trade_id: Unique identifier for this trade (UUID)
        symbol: Trading symbol (e.g., 'MNQ')
        direction: Trade direction (long or short)
        quantity: Number of contracts (1-5)
        order_type: Order type (market or limit)
        entry_price: Entry price for the trade
        limit_price: Limit price (if limit order)
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        timestamp: Order creation timestamp
        status: Order status (pending, submitted, filled, rejected, cancelled)
        ensemble_signal: Reference to ensemble signal that triggered this order
        position_size: Position size in contracts (1-5)
        entry_time: Time position was entered
        exit_time: Time position was exited (None if still open)
        exit_price: Price at exit (None if not exited)
        exit_reason: Reason for exit (None if not exited)
        hold_time_seconds: Time held in seconds (None if not exited)
        realized_pnl: Realized profit/loss (None if not exited)
        rr_achieved: Reward-risk ratio achieved (None if not exited)
        position_state: Current position state (open/partially_closed/closed)
        original_quantity: Original quantity for partial exits tracking
        remaining_quantity: Remaining quantity after partial exits
    """

    trade_id: str = Field(..., description="Unique identifier for this trade")
    symbol: str = Field(default="MNQ", description="Trading symbol")
    direction: Literal["long", "short"] = Field(..., description="Trade direction")
    quantity: int = Field(..., ge=1, le=5, description="Number of contracts (1-5)")
    order_type: Literal["market", "limit"] = Field(..., description="Order type")
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price (if limit order)")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profit: float = Field(..., gt=0, description="Take profit price level")
    timestamp: datetime = Field(..., description="Order creation timestamp")
    status: Literal["pending", "submitted", "filled", "rejected", "cancelled"] = Field(
        default="pending", description="Order status"
    )
    ensemble_signal: EnsembleTradeSignal = Field(
        ..., description="Reference to ensemble signal"
    )
    position_size: int = Field(..., ge=1, le=5, description="Position size in contracts")

    # Exit tracking fields
    entry_time: datetime = Field(..., description="Time position was entered")
    exit_time: Optional[datetime] = Field(None, description="Time position was exited")
    exit_price: Optional[float] = Field(None, gt=0, description="Price at exit")
    exit_reason: Optional[str] = Field(None, description="Reason for exit")
    hold_time_seconds: Optional[int] = Field(None, ge=0, description="Hold time in seconds")
    realized_pnl: Optional[float] = Field(None, description="Realized P&L in USD")
    rr_achieved: Optional[float] = Field(None, description="R:R achieved")
    position_state: Literal["open", "partially_closed", "closed"] = Field(
        default="open", description="Current position state"
    )
    original_quantity: int = Field(..., ge=1, le=5, description="Original quantity")
    remaining_quantity: int = Field(..., ge=0, le=5, description="Remaining quantity")

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_respect_direction(cls, v: float, info) -> float:
        """Validate stop loss position based on direction."""
        direction = info.data.get("direction")
        entry = info.data.get("entry_price")

        if entry is not None and direction is not None:
            if direction == "long" and v >= entry:
                raise ValueError("stop_loss must be below entry_price for long positions")
            if direction == "short" and v <= entry:
                raise ValueError("stop_loss must be above entry_price for short positions")
        return v

    @field_validator("take_profit")
    @classmethod
    def take_profit_must_respect_rr_ratio(cls, v: float, info) -> float:
        """Validate take profit respects at least 2:1 reward-risk ratio."""
        entry = info.data.get("entry_price")
        stop_loss = info.data.get("stop_loss")
        direction = info.data.get("direction")

        if entry is not None and stop_loss is not None and direction is not None:
            risk = abs(entry - stop_loss)
            if risk > 0:
                min_reward = risk * 2.0
                if direction == "long" and v < entry + min_reward:
                    raise ValueError(f"take_profit must be at least 2:1 reward-risk (>= {entry + min_reward})")
                if direction == "short" and v > entry - min_reward:
                    raise ValueError(f"take_profit must be at least 2:1 reward-risk (<= {entry - min_reward})")
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
        """Calculate contract notional value.

        MNQ contract multiplier is $0.50 per point.

        Returns:
            Notional value in USD
        """
        multiplier = 0.50  # MNQ contract multiplier
        return self.entry_price * multiplier * self.quantity

    def risk_per_contract(self) -> float:
        """Calculate risk per contract in points.

        Returns:
            Risk in points from entry to stop loss
        """
        return abs(self.entry_price - self.stop_loss)

    def hold_time_minutes(self) -> float:
        """Calculate hold time in minutes.

        Returns:
            Hold time in minutes (0 if position not exited)
        """
        if self.hold_time_seconds is None:
            return 0.0
        return self.hold_time_seconds / 60.0

    def is_held_max_time(self, max_hold_minutes: float = 10.0) -> bool:
        """Check if position has exceeded maximum hold time.

        Args:
            max_hold_minutes: Maximum hold time in minutes (default 10)

        Returns:
            True if position has exceeded max hold time
        """
        if self.exit_time is None:
            # Check current hold time if still open
            from datetime import datetime
            current_hold = (datetime.now() - self.entry_time).total_seconds() / 60.0
            return current_hold >= max_hold_minutes
        else:
            # Check final hold time if closed
            return self.hold_time_minutes() >= max_hold_minutes

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
    """Exit order for closing positions.

    This model represents an order to exit a position, either fully or partially.
    Exit orders are generated by exit logic strategies and passed to execution.

    Attributes:
        position_id: Position ID (references TradeOrder.trade_id)
        exit_type: Exit type (full or partial)
        quantity: Number of contracts to close
        exit_price: Exit price
        exit_reason: Reason for exit (take_profit, stop_loss, time_stop, hybrid_partial, hybrid_trail)
        timestamp: Exit order timestamp
        pnl: Profit/loss for this exit in USD
        rr_ratio: Reward-risk ratio achieved
    """

    position_id: str = Field(..., description="Position ID to exit")
    exit_type: Literal["full", "partial"] = Field(..., description="Exit type")
    quantity: int = Field(..., ge=1, le=5, description="Number of contracts to close")
    exit_price: float = Field(..., gt=0, description="Exit price")
    exit_reason: Literal["take_profit", "stop_loss", "time_stop", "hybrid_partial", "hybrid_trail"] = Field(
        ..., description="Reason for exit"
    )
    timestamp: datetime = Field(..., description="Exit order timestamp")
    pnl: float = Field(..., description="Profit/loss for this exit in USD")
    rr_ratio: float = Field(..., description="Reward-risk ratio achieved")

    @field_validator("exit_reason")
    @classmethod
    def exit_reason_must_match_exit_type(cls, v: str, info) -> str:
        """Validate exit reason matches exit type."""
        exit_type = info.data.get("exit_type")
        if v == "hybrid_partial" and exit_type != "partial":
            raise ValueError("hybrid_partial exit_reason requires partial exit_type")
        if v in ["take_profit", "stop_loss", "time_stop", "hybrid_trail"] and exit_type != "full":
            raise ValueError(f"{v} exit_reason requires full exit_type")
        return v


class PositionMonitoringState(BaseModel):
    """Position state for exit monitoring.

    This model captures the current state of a position being monitored
    for exit conditions. Includes all information needed by exit strategies.

    Attributes:
        position: TradeOrder being monitored
        current_price: Current market price
        unrealized_pnl: Current unrealized P&L
        time_since_entry_seconds: Time since position entry in seconds
        distance_to_tp: Distance to take profit (price units)
        distance_to_sl: Distance to stop loss (price units)
        rr_achieved: Current R:R achieved
    """

    position: TradeOrder = Field(..., description="Position being monitored")
    current_price: float = Field(..., gt=0, description="Current market price")
    unrealized_pnl: float = Field(..., description="Current unrealized P&L in USD")
    time_since_entry_seconds: int = Field(..., ge=0, description="Time since entry in seconds")
    distance_to_tp: float = Field(..., description="Distance to take profit")
    distance_to_sl: float = Field(..., description="Distance to stop loss")
    rr_achieved: float = Field(..., description="Current R:R achieved")

    def hold_time_minutes(self) -> float:
        """Calculate hold time in minutes.

        Returns:
            Hold time in minutes
        """
        return self.time_since_entry_seconds / 60.0

    def is_at_max_hold_time(self, max_hold_minutes: float = 10.0) -> bool:
        """Check if position has exceeded maximum hold time.

        Args:
            max_hold_minutes: Maximum hold time in minutes (default 10)

        Returns:
            True if max hold time has been reached
        """
        return self.hold_time_minutes() >= max_hold_minutes


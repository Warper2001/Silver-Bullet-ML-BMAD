"""Pydantic models for TradeStation market data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, FieldValidationInfo, field_validator


class MarketData(BaseModel):
    """Real-time market data from TradeStation WebSocket."""

    symbol: str = Field(..., description="Trading symbol (e.g., 'MNQ')")
    timestamp: datetime = Field(..., description="Data timestamp")
    bid: Optional[float] = Field(None, ge=0, description="Current bid price")
    ask: Optional[float] = Field(None, gt=0, description="Current ask price")
    last: Optional[float] = Field(None, gt=0, description="Last trade price")
    volume: int = Field(..., ge=0, description="Trade volume")

    @field_validator("ask")
    @classmethod
    def ask_must_be_greater_than_bid(cls, v: float | None, info) -> float | None:  # type: ignore[no-untyped-def]
        """Validate ask price is greater than bid price."""
        if v is not None and info.data.get("bid") is not None:
            if v <= info.data["bid"]:
                raise ValueError("ask must be greater than bid")
        return v

    def has_required_fields(self) -> bool:
        """Check if all required market data fields are present."""
        return (
            self.bid is not None
            and self.ask is not None
            and self.last is not None
            and self.volume >= 0
        )


class WebSocketMessage(BaseModel):
    """Raw WebSocket message from TradeStation API."""

    symbol: str
    timestamp: str  # ISO 8601 format
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: int = 0

    def to_market_data(self) -> MarketData:
        """Convert to MarketData model with validation."""
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.fromisoformat(self.timestamp.replace("Z", "+00:00")),
            bid=self.bid,
            ask=self.ask,
            last=self.last,
            volume=self.volume,
        )


class DollarBar(BaseModel):
    """Dollar Bar with fixed $50M notional value threshold."""

    timestamp: datetime = Field(..., description="Bar completion timestamp")
    open: float = Field(..., gt=0, description="Open price (first trade)")
    high: float = Field(..., gt=0, description="High price (max trade)")
    low: float = Field(..., gt=0, description="Low price (min trade)")
    close: float = Field(..., gt=0, description="Close price (last trade)")
    volume: int = Field(..., ge=0, description="Total volume in bar")
    notional_value: float = Field(..., ge=0, description="Notional value ($)")

    @field_validator("high")
    @classmethod
    def high_gte_open_and_close(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate high is >= open and close."""
        # Only validate against open (close not set yet when high is validated)
        open_val = info.data.get("open")
        if open_val is not None and v < open_val:
            raise ValueError("high must be >= open")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_and_close(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate low is <= open and close."""
        # Only validate against open (close not set yet when low is validated)
        open_val = info.data.get("open")
        if open_val is not None and v > open_val:
            raise ValueError("low must be <= open")
        return v

    @field_validator("close")
    @classmethod
    def close_within_high_low(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate close is within high and low."""
        high_val = info.data.get("high")
        low_val = info.data.get("low")
        if high_val is not None and v > high_val:
            raise ValueError("close must be <= high")
        if low_val is not None and v < low_val:
            raise ValueError("close must be >= low")
        return v

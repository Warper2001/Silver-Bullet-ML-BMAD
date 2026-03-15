"""Pydantic models for TradeStation market data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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

"""
BYDFI API Data Models

Pydantic models for BYDFI API request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BYDFIQuote(BaseModel):
    """
    Current price quote for a trading symbol.

    Attributes:
        symbol: Trading symbol (e.g., BTC-USDT)
        price: Current price
        timestamp: Quote timestamp
    """

    symbol: str = Field(..., description="Trading symbol")
    price: str = Field(..., description="Current price")
    volume: Optional[str] = Field(None, description="24h volume")
    timestamp: Optional[datetime] = Field(None, description="Quote timestamp")


class BYDFIKline(BaseModel):
    """
    Kline (candlestick) data.

    Attributes:
        symbol: Trading symbol
        timestamp: Kline timestamp
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Trading volume
    """

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Kline timestamp")
    open: str = Field(..., description="Open price")
    high: str = Field(..., description="High price")
    low: str = Field(..., description="Low price")
    close: str = Field(..., description="Close price")
    volume: str = Field(..., description="Trading volume")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class BYDFIOrderBook(BaseModel):
    """
    Order book depth data.

    Attributes:
        symbol: Trading symbol
        bids: Bid orders (price, quantity)
        asks: Ask orders (price, quantity)
        timestamp: Order book timestamp
    """

    symbol: str = Field(..., description="Trading symbol")
    bids: list[tuple[str, str]] = Field(
        ...,
        description="Bid orders as [(price, quantity), ...]",
    )
    asks: list[tuple[str, str]] = Field(
        ...,
        description="Ask orders as [(price, quantity), ...]",
    )
    timestamp: Optional[datetime] = Field(None, description="Order book timestamp")


class BYDFIAccount(BaseModel):
    """
    Account information.

    Attributes:
        currency: Currency code (e.g., BTC, USDT)
        balance: Available balance
        frozen: Frozen balance
        total: Total balance
    """

    currency: str = Field(..., description="Currency code")
    balance: str = Field(..., description="Available balance")
    frozen: str = Field(default="0", description="Frozen balance")
    total: Optional[str] = Field(None, description="Total balance")


class BYDFIOrder(BaseModel):
    """
    Order information.

    Attributes:
        order_id: Order ID
        client_order_id: Client order ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        order_type: Order type (market/limit)
        price: Order price
        quantity: Order quantity
        filled_quantity: Filled quantity
        status: Order status
        timestamp: Order timestamp
    """

    order_id: str = Field(..., description="Order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    order_type: str = Field(..., description="Order type")
    price: Optional[str] = Field(None, description="Order price")
    quantity: str = Field(..., description="Order quantity")
    filled_quantity: str = Field(default="0", description="Filled quantity")
    status: str = Field(..., description="Order status")
    timestamp: Optional[datetime] = Field(None, description="Order timestamp")


class BYDFIErrorResponse(BaseModel):
    """
    BYDFI API error response.

    Attributes:
        code: Error code
        message: Error message
    """

    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")

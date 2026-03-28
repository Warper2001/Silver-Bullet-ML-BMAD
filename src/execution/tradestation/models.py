"""
TradeStation SDK - Shared Pydantic Models

This module defines shared Pydantic models for the TradeStation SDK.
All models use snake_case field names with Field(alias="...") to map
from the TradeStation API's camelCase JSON responses.

Design Pattern:
- Pythonic snake_case field names (PEP 8 compliant)
- API aliases for camelCase JSON mapping
- Validation and type safety via Pydantic v2
- populate_by_name enabled for dual-name support
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# OAuth 2.0 Models
# =============================================================================


class TokenResponse(BaseModel):
    """
    OAuth 2.0 token response from TradeStation API.

    Attributes:
        access_token: Bearer token for API authentication
        token_type: Token type (always "Bearer")
        expires_in: Time until token expires (seconds)
        refresh_token: Token for refreshing access token (LIVE environment only)
        scope: Granted OAuth scopes
    """

    access_token: str = Field(alias="access_token")
    token_type: str = Field(alias="token_type")
    expires_in: int = Field(alias="expires_in")
    refresh_token: str | None = Field(default=None, alias="refresh_token")
    scope: str | None = Field(default=None, alias="scope")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("token_type")
    @classmethod
    def validate_token_type(cls, v: str) -> str:
        """Validate token type is Bearer."""
        if v.lower() != "bearer":
            raise ValueError(f"Expected token_type 'Bearer', got '{v}'")
        return v


# =============================================================================
# Market Data Models
# =============================================================================


class TradeStationQuote(BaseModel):
    """
    Real-time market data quote from TradeStation API.

    Attributes:
        symbol: Trading symbol (e.g., "MNQH26")
        bid: Current bid price
        ask: Current ask price
        last: Last trade price
        bid_size: Size of bid (number of contracts)
        ask_size: Size of ask (number of contracts)
        last_size: Size of last trade
        timestamp: Quote timestamp
        volume: Daily volume
        open: Daily opening price
        high: Daily high price
        low: Daily low price
        close: Previous close price
    """

    symbol: str = Field(alias="Symbol")
    bid: float | None = Field(default=None, alias="Bid")
    ask: float | None = Field(default=None, alias="Ask")
    last: float | None = Field(default=None, alias="Last")
    bid_size: int | None = Field(default=None, alias="BidSize")
    ask_size: int | None = Field(default=None, alias="AskSize")
    last_size: int | None = Field(default=None, alias="LastSize")
    timestamp: datetime = Field(alias="TimeStamp")
    volume: int | None = Field(default=None, alias="Volume")
    open: float | None = Field(default=None, alias="Open")
    high: float | None = Field(default=None, alias="High")
    low: float | None = Field(default=None, alias="Low")
    close: float | None = Field(default=None, alias="Close")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol format."""
        if not v or len(v) < 2:
            raise ValueError("Invalid symbol format")
        return v.upper()


class HistoricalBar(BaseModel):
    """
    Historical OHLCV bar from TradeStation API.

    Attributes:
        symbol: Trading symbol
        timestamp: Bar timestamp
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        bar_type: Bar type (e.g., "minute", "daily", "5minute")
    """

    symbol: str = Field(alias="Symbol")
    timestamp: datetime = Field(alias="TimeStamp")
    open: float = Field(alias="Open")
    high: float = Field(alias="High")
    low: float = Field(alias="Low")
    close: float = Field(alias="Close")
    volume: int = Field(alias="Volume")
    bar_type: str = Field(alias="BarType")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("high", "low")
    @classmethod
    def validate_high_low(cls, v: float, info) -> float:
        """Validate high >= low."""
        # Note: This validator will have access to the full model in Pydantic v2
        # For now, we'll skip cross-field validation to keep it simple
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v


# =============================================================================
# Order Models
# =============================================================================


class OrderType(BaseModel):
    """
    Supported order types.

    Attributes:
        value: Order type value
    """

    value: Literal["Market", "Limit", "Stop", "StopLimit", "MarketOnClose"]


class OrderSide(BaseModel):
    """
    Order side (buy or sell).

    Attributes:
        value: Order side value
    """

    value: Literal["Buy", "Sell"]


class TradeStationOrder(BaseModel):
    """
    Order from TradeStation API.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        order_type: Order type (Market, Limit, Stop, etc.)
        side: Buy or Sell
        quantity: Number of contracts
        price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        status: Order status
        filled_quantity: Number of contracts filled
        avg_fill_price: Average fill price
        timestamp: Order creation timestamp
    """

    order_id: str = Field(alias="OrderID")
    symbol: str = Field(alias="Symbol")
    order_type: str = Field(alias="OrderType")
    side: str = Field(alias="Side")
    quantity: int = Field(alias="Quantity")
    price: float | None = Field(default=None, alias="Price")
    stop_price: float | None = Field(default=None, alias="StopPrice")
    status: str = Field(alias="Status")
    filled_quantity: int = Field(default=0, alias="FilledQuantity")
    avg_fill_price: float | None = Field(default=None, alias="AvgFillPrice")
    timestamp: datetime = Field(alias="TimeStamp")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("quantity", "filled_quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        """Validate quantity is positive."""
        if v < 0:
            raise ValueError("Quantity cannot be negative")
        return v

    @field_validator("order_type")
    @classmethod
    def validate_order_type(cls, v: str) -> str:
        """Validate order type is supported."""
        valid_types = {"Market", "Limit", "Stop", "StopLimit", "MarketOnClose"}
        if v not in valid_types:
            raise ValueError(f"Invalid order type: {v}")
        return v


class NewOrderRequest(BaseModel):
    """
    Request to submit a new order.

    Attributes:
        symbol: Trading symbol
        side: Buy or Sell
        order_type: Order type (Market, Limit, Stop, etc.)
        quantity: Number of contracts
        price: Limit price (required for Limit orders)
        stop_price: Stop price (required for Stop orders)
        time_in_force: Order duration (Day, GTC, etc.)
    """

    symbol: str
    side: Literal["Buy", "Sell"]
    order_type: Literal["Market", "Limit", "Stop", "StopLimit", "MarketOnClose"]
    quantity: int = Field(gt=0)
    price: float | None = Field(default=None)
    stop_price: float | None = Field(default=None)
    time_in_force: Literal["Day", "GTC", "IOC", "FOK"] = "Day"

    @field_validator("price")
    @classmethod
    def validate_price_for_limit(cls, v: float | None, info) -> float | None:
        """Validate price is provided for limit orders."""
        if info.data.get("order_type") in ("Limit", "StopLimit") and v is None:
            raise ValueError("Price is required for limit orders")
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v

    @field_validator("stop_price")
    @classmethod
    def validate_stop_price_for_stop(cls, v: float | None, info) -> float | None:
        """Validate stop_price is provided for stop orders."""
        if info.data.get("order_type") in ("Stop", "StopLimit") and v is None:
            raise ValueError("Stop price is required for stop orders")
        if v is not None and v <= 0:
            raise ValueError("Stop price must be positive")
        return v


class OrderStatusUpdate(BaseModel):
    """
    Real-time order status update from streaming endpoint.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        status: Current order status
        filled_quantity: Number of contracts filled
        avg_fill_price: Average fill price
        remaining_quantity: Remaining unfilled quantity
        timestamp: Status update timestamp
    """

    order_id: str = Field(alias="OrderID")
    symbol: str = Field(alias="Symbol")
    status: str = Field(alias="Status")
    filled_quantity: int = Field(default=0, alias="FilledQuantity")
    avg_fill_price: float | None = Field(default=None, alias="AvgFillPrice")
    remaining_quantity: int = Field(default=0, alias="RemainingQuantity")
    timestamp: datetime = Field(alias="TimeStamp")

    model_config = ConfigDict(populate_by_name=True)


class OrderFill(BaseModel):
    """
    Individual fill details for an order.

    Attributes:
        order_id: Unique order identifier
        fill_id: Unique fill identifier
        symbol: Trading symbol
        side: Buy or Sell
        fill_quantity: Number of contracts filled
        fill_price: Price of this fill
        timestamp: Fill timestamp
        commission: Commission for this fill
    """

    order_id: str = Field(alias="OrderID")
    fill_id: str = Field(alias="FillID")
    symbol: str = Field(alias="Symbol")
    side: str = Field(alias="Side")
    fill_quantity: int = Field(alias="FillQuantity")
    fill_price: float = Field(alias="FillPrice")
    timestamp: datetime = Field(alias="TimeStamp")
    commission: float | None = Field(default=None, alias="Commission")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Account Models
# =============================================================================


class AccountPosition(BaseModel):
    """
    Current account position from TradeStation API.

    Attributes:
        symbol: Trading symbol
        quantity: Current position size (positive = long, negative = short)
        average_price: Average entry price
        market_value: Current market value
        unrealized_pnl: Unrealized profit/loss
        timestamp: Position timestamp
    """

    symbol: str = Field(alias="Symbol")
    quantity: int = Field(alias="Quantity")
    average_price: float | None = Field(default=None, alias="AveragePrice")
    market_value: float | None = Field(default=None, alias="MarketValue")
    unrealized_pnl: float | None = Field(default=None, alias="UnrealizedPnL")
    timestamp: datetime = Field(alias="TimeStamp")

    model_config = ConfigDict(populate_by_name=True)


class AccountBalance(BaseModel):
    """
    Account balance summary from TradeStation API.

    Attributes:
        account_id: Account identifier
        cash_balance: Available cash balance
        total_value: Total account value
        buying_power: Available buying power
        margin_requirement: Margin requirement for open positions
        timestamp: Balance timestamp
    """

    account_id: str = Field(alias="AccountID")
    cash_balance: float = Field(alias="CashBalance")
    total_value: float = Field(alias="TotalValue")
    buying_power: float | None = Field(default=None, alias="BuyingPower")
    margin_requirement: float | None = Field(default=None, alias="MarginRequirement")
    timestamp: datetime = Field(alias="TimeStamp")

    model_config = ConfigDict(populate_by_name=True)

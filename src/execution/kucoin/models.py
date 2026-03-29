"""
KuCoin SDK - Shared Pydantic Models

This module defines shared Pydantic models for the KuCoin SDK.
All models use snake_case field names with Field(alias="...") to map
from the KuCoin API's camelCase JSON responses.

API Documentation: https://docs.kucoin.com/

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
# Market Data Models
# =============================================================================


class KuCoinSymbolPrice(BaseModel):
    """
    Symbol price ticker from KuCoin API.

    API Docs: https://docs.kucoin.com/#get-ticker

    Attributes:
        symbol: Trading symbol (e.g., "BTC-USDT")
        price: Current price
        sequence: Sequence number
    """

    symbol: str = Field(alias="symbol")
    price: float = Field(alias="price")
    sequence: str = Field(alias="sequence")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol format."""
        if not v or len(v) < 7:
            raise ValueError("Invalid symbol format (e.g., BTC-USDT)")
        return v.upper()

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price is positive."""
        if v <= 0:
            raise ValueError("Price must be positive")
        return v


class KuCoinQuote(BaseModel):
    """
    Ticker price change statistics from KuCoin API.

    API Docs: https://docs.kucoin.com/#get-ticker

    Attributes:
        symbol: Trading symbol
        price_change: Price change in last 24h
        price_change_percent: Price change percent in last 24h
        best_bid: Best bid price
        best_ask: Best ask price
        last_price: Last trade price
        volume: Total trade volume in last 24h (base asset)
        quote_volume: Total trade volume in last 24h (quote asset)
    """

    symbol: str = Field(alias="symbol")
    price_change: float | None = Field(default=None, alias="priceChange")
    price_change_percent: float | None = Field(default=None, alias="priceChangePercent")
    best_bid: float | None = Field(default=None, alias="bestBid")
    best_ask: float | None = Field(default=None, alias="bestAsk")
    last_price: float | None = Field(default=None, alias="last")
    volume: float | None = Field(default=None, alias="vol")
    quote_volume: float | None = Field(default=None, alias="volValue")

    model_config = ConfigDict(populate_by_name=True)


class KuCoinKline(BaseModel):
    """
    Kline (candlestick) data from KuCoin API.

    API Docs: https://docs.kucoin.com/#get-klines

    Attributes:
        symbol: Trading symbol
        interval: Kline interval (e.g., "1min", "5min", "1hour", "1day")
        open_time: Kline open time (milliseconds)
        close_time: Kline close time (milliseconds)
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume (base asset)
        quote_volume: Volume (quote asset)
    """

    symbol: str = Field(alias="symbol")
    interval: str = Field(alias="type")
    open_time: int = Field(alias="startTime")  # milliseconds
    close_time: int = Field(alias="endTime")  # milliseconds
    open: float = Field(alias="open")
    high: float = Field(alias="high")
    low: float = Field(alias="low")
    close: float = Field(alias="close")
    volume: float = Field(alias="vol")
    quote_volume: float = Field(alias="volQuote")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("high", "low")
    @classmethod
    def validate_prices(cls, v: float) -> float:
        """Validate price is not negative."""
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v


class KuCoinOrderBook(BaseModel):
    """
    Order book depth from KuCoin API.

    API Docs: https://docs.kucoin.com/#get-part-order-book-aggregated

    Attributes:
        symbol: Trading symbol
        bids: Bid levels (price, quantity)
        asks: Ask levels (price, quantity)
        sequence: Sequence number
    """

    symbol: str = Field(alias="symbol")
    bids: list[tuple[str, str]] = Field(alias="bids")  # [(price, quantity), ...]
    asks: list[tuple[str, str]] = Field(alias="asks")  # [(price, quantity), ...]
    sequence: str | None = Field(default=None, alias="sequence")

    model_config = ConfigDict(populate_by_name=True)


class KuCoinTrade(BaseModel):
    """
    Trade data from KuCoin API.

    API Docs: https://docs.kucoin.com/#get-trade-histories

    Attributes:
        symbol: Trading symbol
        price: Trade price
        quantity: Trade quantity
        time: Trade timestamp (milliseconds)
        trade_id: Trade sequence ID
    """

    symbol: str = Field(default="BTC-USDT", alias="symbol")
    price: float = Field(alias="price")
        quantity: float = Field(alias="size")
        time: int = Field(alias="tradeTime")  # milliseconds
        trade_id: str = Field(alias="tradeId")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Account Models
# =============================================================================


class KuCoinAccount(BaseModel):
    """
    Account information from KuCoin API.

    API Docs: https://docs.kucoin.com/#get-account-list-spot-margin-trade-hf

    Attributes:
        currency: Currency code (e.g., "USDT")
        balance: Available balance
        available: Available for trading
        holds: Funds on hold (in orders)
    """

    id: str = Field(alias="id")
    currency: str = Field(alias="currency")
    balance: float = Field(alias="balance")
    available: float = Field(alias="available")
    holds: float = Field(alias="holds")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Order Models
# =============================================================================


class KuCoinOrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP = "stop"


class KuCoinOrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class KuCoinOrder(BaseModel):
    """
    Order from KuCoin API.

    API Docs: https://docs.kucoin.com/#orders

    Attributes:
        id: Unique order identifier
        symbol: Trading symbol
        order_type: Order type (market, limit, etc)
        side: Buy or Sell
        price: Limit price
        size: Order quantity
        deal_size: Filled quantity
        deal_funds: Filled funds (quote currency)
        fee_fees: Trading fees
        fee_currency: Fee currency
        status: Order status
        created_at: Order creation time
        updated_at: Order update time
    """

    id: str = Field(alias="id")
    symbol: str = Field(alias="symbol")
    order_type: str = Field(alias="type")
    side: str = Field(alias="side")
    price: float | None = Field(default=None, alias="price")
    size: float = Field(alias="size")
    deal_size: float | None = Field(default=None, alias="dealSize")
    deal_funds: float | None = Field(default=None, alias="dealFunds")
    fee_fees: float | None = Field(default=0.0, alias="fee")
    fee_currency: str | None = Field(default=None, alias="feeCurrency")
    status: str = Field(alias="status")
    created_at: int = Field(alias="createdAt")  # milliseconds
    updated_at: int = Field(alias="updatedAt")  # milliseconds

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate order status is valid."""
        valid_statuses = {
            "open",
            "done",
            "match",
            "canceled",
            "postponed",
        }
        v_lower = v.lower()
        if v_lower not in valid_statuses:
            raise ValueError(f"Invalid order status: {v}")
        return v_lower

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side is BUY or SELL."""
        v_lower = v.lower()
        if v_lower not in ["buy", "sell"]:
            raise ValueError(f"Invalid order side: {v}")
        return v_lower


# =============================================================================
# WebSocket Models
# =============================================================================


class KuCoinWebSocketTrade(BaseModel):
    """
    Trade stream message from KuCoin WebSocket.

    API Docs: https://docs.kucoin.com/#match-execution-data

    Attributes:
        symbol: Trading symbol
        price: Trade price
        quantity: Trade quantity
        trade_id: Trade ID
        timestamp: Trade timestamp (milliseconds)
        sequence: Sequence number
    """

    symbol: str = Field(alias="symbol")
    price: float = Field(alias="price")
    quantity: float = Field(alias="size")
    trade_id: str = Field(alias="tradeId")
    timestamp: int = Field(alias="ts")  # milliseconds
    sequence: str = Field(alias="sequence")

    model_config = ConfigDict(populate_by_name=True)


class KuCoinWebSocketOrderUpdate(BaseModel):
    """
    Order update from KuCoin WebSocket user data stream.

    API Docs: https://docs.kucoin.com/#private-order-change

    Attributes:
        order_id: Order ID
        symbol: Trading symbol
        order_type: Order type
        side: Order side
        price: Order price
        size: Order size
        deal_size: Filled quantity
        status: Order status
        timestamp: Update timestamp (milliseconds)
    """

    order_id: str = Field(alias="orderId")
    symbol: str = Field(alias="symbol")
    order_type: str = Field(alias="type")
    side: str = Field(alias="side")
    price: float | None = Field(default=None, alias="price")
    size: float = Field(alias="size")
    deal_size: float | None = Field(default=None, alias="dealSize")
    status: str = Field(alias="status")
    timestamp: int = Field(alias="ts")  # milliseconds

    model_config = ConfigDict(populate_by_name=True)

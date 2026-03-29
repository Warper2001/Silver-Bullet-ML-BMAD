"""
Binance SDK - Shared Pydantic Models

This module defines shared Pydantic models for the Binance SDK.
All models use snake_case field names with Field(alias="...") to map
from the Binance API's camelCase JSON responses.

API Documentation: https://binance-docs.github.io/apidocs/

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


class BinanceSymbolPrice(BaseModel):
    """
    Symbol price ticker from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#symbol-price-ticker

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT")
        price: Current price
    """

    symbol: str = Field(alias="symbol")
    price: float = Field(alias="price")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol format."""
        if not v or len(v) < 6:
            raise ValueError("Invalid symbol format (e.g., BTCUSDT)")
        return v.upper()

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price is positive."""
        if v <= 0:
            raise ValueError("Price must be positive")
        return v


class BinanceQuote(BaseModel):
    """
    Real-time 24hr ticker price change statistics from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#24hr-ticker-price-change-statistics

    Attributes:
        symbol: Trading symbol
        price_change: Price change in last 24h
        price_change_percent: Price change percent in last 24h
        weighted_avg_price: Weighted average price in last 24h
        prev_close_price: Previous day's close price
        last_price: Last trade price
        bid_price: Best bid price
        ask_price: Best ask price
        bid_qty: Best bid quantity
        ask_qty: Best ask quantity
        open_price: Open price in last 24h
        high_price: High price in last 24h
        low_price: Low price in last 24h
        volume: Total trade volume in last 24h (base asset)
        quote_volume: Total trade volume in last 24h (quote asset)
        open_time: Open time in last 24h
        close_time: Close time in last 24h
        trades: Number of trades in last 24h
    """

    symbol: str = Field(alias="symbol")
    price_change: float = Field(alias="priceChange")
    price_change_percent: float = Field(alias="priceChangePercent")
    weighted_avg_price: float = Field(alias="weightedAvgPrice")
    prev_close_price: float = Field(alias="prevClosePrice")
    last_price: float = Field(alias="lastPrice")
    bid_price: float = Field(alias="bidPrice")
    ask_price: float = Field(alias="askPrice")
    bid_qty: float = Field(alias="bidQty")
    ask_qty: float = Field(alias="askQty")
    open_price: float = Field(alias="openPrice")
    high_price: float = Field(alias="highPrice")
    low_price: float = Field(alias="lowPrice")
    volume: float = Field(alias="volume")
    quote_volume: float = Field(alias="quoteVolume")
    open_time: int = Field(alias="openTime")  # Unix timestamp in milliseconds
    close_time: int = Field(alias="closeTime")  # Unix timestamp in milliseconds
    trades: int = Field(alias="count")

    model_config = ConfigDict(populate_by_name=True)


class BinanceKline(BaseModel):
    """
    Kline (candlestick) data from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#kline-candlestick-data

    Attributes:
        symbol: Trading symbol
        interval: Kline interval (e.g., "1m", "5m", "1h", "1d")
        open_time: Kline open time (milliseconds)
        close_time: Kline close time (milliseconds)
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume (base asset)
        quote_volume: Volume (quote asset)
        trades: Number of trades
        taker_buy_base_volume: Taker buy base asset volume
        taker_buy_quote_volume: Taker buy quote asset volume
    """

    symbol: str = Field(alias="symbol")
    interval: str = Field(alias="interval")
    open_time: int = Field(alias="openTime")  # milliseconds
    close_time: int = Field(alias="closeTime")  # milliseconds
    open: float = Field(alias="open")
    high: float = Field(alias="high")
    low: float = Field(alias="low")
    close: float = Field(alias="close")
    volume: float = Field(alias="volume")
    quote_volume: float = Field(alias="quoteVolume")
    trades: int = Field(alias="trades")
    taker_buy_base_volume: float = Field(alias="takerBuyBaseVolume")
    taker_buy_quote_volume: float = Field(alias="takerBuyQuoteVolume")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("high", "low")
    @classmethod
    def validate_prices(cls, v: float) -> float:
        """Validate price is not negative."""
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v


class BinanceOrderBook(BaseModel):
    """
    Order book depth from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#order-book-depth

    Attributes:
        symbol: Trading symbol
        last_update_id: Last update ID
        bids: Bid levels (price, quantity)
        asks: Ask levels (price, quantity)
    """

    symbol: str = Field(alias="symbol")
    last_update_id: int = Field(alias="lastUpdateId")
    bids: list[tuple[str, str]] = Field(alias="bids")  # [(price, quantity), ...]
    asks: list[tuple[str, str]] = Field(alias="asks")  # [(price, quantity), ...]

    model_config = ConfigDict(populate_by_name=True)


class BinanceTrade(BaseModel):
    """
    Recent trade data from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#recent-trades-list

    Attributes:
        symbol: Trading symbol
        id: Trade ID
        price: Trade price
        qty: Trade quantity
        quote_qty: Trade quantity in quote asset
        time: Trade timestamp (milliseconds)
        is_buyer_maker: Was the buyer the maker?
    """

    symbol: str = Field(default="BTCUSDT", alias="symbol")
    id: int = Field(alias="id")
    price: float = Field(alias="price")
    qty: float = Field(alias="qty")
    quote_qty: float = Field(alias="quoteQty")
    time: int = Field(alias="time")  # milliseconds
    is_buyer_maker: bool = Field(alias="isBuyerMaker")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Account Models
# =============================================================================


class BinanceAccount(BaseModel):
    """
    Account information from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#account-information-user_data

    Attributes:
        maker_commission: Maker commission rate (bips)
        taker_commission: Taker commission rate (bips)
        buyer_commission: Buyer commission rate (bips)
        seller_commission: Seller commission rate (bips)
        can_trade: Can this account trade?
        can_withdraw: Can this account withdraw?
        can_deposit: Can this account deposit?
        update_time: Last account update time (milliseconds)
        account_type: Account type (SPOT, MARGIN, FUTURES)
        balances: List of asset balances
        permissions: Account permissions
    """

    maker_commission: int = Field(alias="makerCommission")
    taker_commission: int = Field(alias="takerCommission")
    buyer_commission: int = Field(alias="buyerCommission")
    seller_commission: int = Field(alias="sellerCommission")
    can_trade: bool = Field(alias="canTrade")
    can_withdraw: bool = Field(alias="canWithdraw")
    can_deposit: bool = Field(alias="canDeposit")
    update_time: int = Field(alias="updateTime")
    account_type: str = Field(alias="accountType")
    balances: list["BinanceBalance"] = Field(alias="balances")
    permissions: list[str] = Field(alias="permissions")

    model_config = ConfigDict(populate_by_name=True)


class BinanceBalance(BaseModel):
    """
    Asset balance from Binance account.

    Attributes:
        asset: Asset symbol (e.g., "BTC", "USDT")
        free: Available balance (not locked in orders)
        locked: Locked balance (in open orders)
    """

    asset: str = Field(alias="asset")
    free: float = Field(alias="free")
    locked: float = Field(alias="locked")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Order Models
# =============================================================================


class BinanceOrderType(BaseModel):
    """
    Supported order types.

    API Docs: https://binance-docs.github.io/apidocs/#enums

    Attributes:
        value: Order type value
    """

    value: Literal["MARKET", "LIMIT", "STOP_LOSS_LIMIT"]


class BinanceOrderSide(BaseModel):
    """
    Order side (buy or sell).

    API Docs: https://binance-docs.github.io/apidocs/#enums

    Attributes:
        value: Order side value
    """

    value: Literal["BUY", "SELL"]


class BinanceOrderStatus(BaseModel):
    """
    Order status values.

    API Docs: https://binance-docs.github.io/apidocs/#enums

    Attributes:
        value: Order status value
    """

    value: Literal["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "PENDING_CANCEL", "REJECTED", "EXPIRED"]


class BinanceTimeInForce(BaseModel):
    """
    Time in force values.

    API Docs: https://binance-docs.github.io/apidocs/#enums

    Attributes:
        value: Time in force value
    """

    value: Literal["GTC", "IOC", "FOK"]  # Good Till Cancel, Immediate or Cancel, Fill or Kill


class BinanceOrder(BaseModel):
    """
    Order from Binance API.

    API Docs: https://binance-docs.github.io/apidocs/#account-information-user_data

    Attributes:
        symbol: Trading symbol
        order_id: Unique order identifier
        order_list_id: Order list ID (for OCO orders)
        client_order_id: Client order ID (user-defined)
        price: Order price
        orig_qty: Original quantity
        executed_qty: Executed quantity
        cummulative_quote_qty: Cumulative quote quantity
        status: Order status
        time_in_force: Time in force
        order_type: Order type
        side: Order side
        stop_price: Stop price (for stop orders)
        iceberg_qty: Iceberg quantity (for iceberg orders)
        time: Order creation time (milliseconds)
        update_time: Order update time (milliseconds)
        is_working: Is the order working?
        orig_quote_order_qty: Original quote order quantity
    """

    symbol: str = Field(alias="symbol")
    order_id: int = Field(alias="orderId")
    order_list_id: int = Field(alias="orderListId")
    client_order_id: str = Field(alias="clientOrderId")
    price: float = Field(alias="price")
    orig_qty: float = Field(alias="origQty")
    executed_qty: float = Field(alias="executedQty")
    cummulative_quote_qty: float = Field(alias="cummulativeQuoteQty")
    status: str = Field(alias="status")
    time_in_force: str = Field(alias="timeInForce")
    order_type: str = Field(alias="type")
    side: str = Field(alias="side")
    stop_price: float | None = Field(default=None, alias="stopPrice")
    iceberg_qty: float | None = Field(default=None, alias="icebergQty")
    time: int = Field(alias="time")  # milliseconds
    update_time: int = Field(alias="updateTime")  # milliseconds
    is_working: bool = Field(alias="isWorking")
    orig_quote_order_qty: float | None = Field(default=None, alias="origQuoteOrderQty")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate order status is valid."""
        valid_statuses = {
            "NEW",
            "PARTIALLY_FILLED",
            "FILLED",
            "CANCELED",
            "PENDING_CANCEL",
            "REJECTED",
            "EXPIRED",
        }
        v_upper = v.upper()
        if v_upper not in valid_statuses:
            raise ValueError(f"Invalid order status: {v}")
        return v_upper

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side is BUY or SELL."""
        v_upper = v.upper()
        if v_upper not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid order side: {v}")
        return v_upper


# =============================================================================
# WebSocket Models
# =============================================================================


class BinanceWebSocketTrade(BaseModel):
    """
    Trade stream message from Binance WebSocket.

    API Docs: https://binance-docs.github.io/apidocs/#trade-streams

    Attributes:
        event_type: Event type (e.g., "trade")
        event_time: Event time (milliseconds)
        symbol: Trading symbol
        trade_id: Trade ID
        price: Trade price
        quantity: Trade quantity
        buyer_order_id: Buyer order ID
        seller_order_id: Seller order ID
        trade_time: Trade time (milliseconds)
        is_buyer_maker: Was the buyer the maker?
    """

    event_type: str = Field(alias="e")
    event_time: int = Field(alias="E")
    symbol: str = Field(alias="s")
    trade_id: int = Field(alias="t")
    price: float = Field(alias="p")
    quantity: float = Field(alias="q")
    buyer_order_id: int = Field(alias="b")
    seller_order_id: int = Field(alias="a")
    trade_time: int = Field(alias="T")
    is_buyer_maker: bool = Field(alias="m")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type is trade."""
        if v != "trade":
            raise ValueError(f"Expected event_type 'trade', got '{v}'")
        return v


class BinanceUserDataStreamEvent(BaseModel):
    """
    User data stream event from Binance WebSocket.

    API Docs: https://binance-docs.github.io/apidocs/#payload-user-data-stream

    Attributes:
        event_type: Event type (e.g., "executionReport")
        event_time: Event time (milliseconds)
        symbol: Trading symbol
        order: Order details (if applicable)
    """

    event_type: str = Field(alias="event_type")
    event_time: int = Field(alias="event_time")
    symbol: str = Field(alias="symbol")
    # Order details would be included here for execution reports

    model_config = ConfigDict(populate_by_name=True)

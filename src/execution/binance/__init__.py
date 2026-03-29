"""
Binance Execution Module

This module provides integration with the Binance cryptocurrency exchange API.
It mirrors the structure of the TradeStation module but is adapted for Binance's
API patterns (API Key + HMAC signature vs OAuth, true WebSocket vs HTTP streaming).

Architecture:
- client.py: Main BinanceClient (async context manager)
- models.py: Pydantic models for API responses
- exceptions.py: Exception hierarchy (BinanceError, AuthError, APIError, OrderError)
- utils.py: Circuit breaker, weight-based rate limiting
- auth/: API key authentication and HMAC signature generation
- market_data/: REST API for quotes/history, WebSocket for streaming
- orders/: Order submission and status tracking

API Documentation: https://binance-docs.github.io/apidocs/

Usage:
    from src.execution.binance import BinanceClient

    async with BinanceClient() as client:
        quotes = await client.get_quotes("BTCUSDT")
        order = await client.place_order(...)

Environment Configuration:
- Requires .env.crypto file with Binance API credentials
- Uses CRYPTO_EXCHANGE_API_KEY and CRYPTO_EXCHANGE_API_SECRET
- Supports testnet (https://testnet.binance.vision) and production (https://api.binance.com)
"""

from src.execution.binance.exceptions import (
    APIError,
    AuthError,
    BinanceError,
    InsufficientFundsError,
    InvalidCredentialsError,
    NetworkError,
    OrderError,
    OrderNotFoundError,
    OrderRejectedError,
    PositionLimitError,
    RateLimitError,
    SignatureGenerationError,
    ValidationError,
)
from src.execution.binance.models import (
    BinanceAccount,
    BinanceBalance,
    BinanceKline,
    BinanceOrder,
    BinanceOrderBook,
    BinanceOrderSide,
    BinanceOrderStatus,
    BinanceOrderType,
    BinanceQuote,
    BinanceSymbolPrice,
    BinanceTimeInForce,
    BinanceTrade,
    BinanceWebSocketTrade,
)
from src.execution.binance.utils import CircuitBreaker, WeightBasedRateLimitTracker

__all__ = [
    # Exceptions
    "BinanceError",
    "AuthError",
    "InvalidCredentialsError",
    "SignatureGenerationError",
    "APIError",
    "RateLimitError",
    "NetworkError",
    "ValidationError",
    "OrderError",
    "OrderRejectedError",
    "PositionLimitError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    # Models
    "BinanceSymbolPrice",
    "BinanceQuote",
    "BinanceKline",
    "BinanceOrderBook",
    "BinanceTrade",
    "BinanceAccount",
    "BinanceBalance",
    "BinanceOrderType",
    "BinanceOrderSide",
    "BinanceOrderStatus",
    "BinanceTimeInForce",
    "BinanceOrder",
    "BinanceWebSocketTrade",
    # Utils
    "WeightBasedRateLimitTracker",
    "CircuitBreaker",
]

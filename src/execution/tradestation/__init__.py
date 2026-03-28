"""
TradeStation SDK Package

This package provides a Python SDK for interacting with the TradeStation API,
supporting both SIM (paper trading) and LIVE (production) environments.

Main Components:
- TradeStationClient: Async API client with OAuth 2.0 authentication
- Market Data: Historical quotes and real-time streaming
- Order Management: Order submission, modification, and cancellation
- Authentication: OAuth 2.0 flows (Authorization Code + Client Credentials)

Example:
    async with TradeStationClient(env="sim", config=config) as client:
        quotes = await client.get_quotes(["MNQH26"])
        print(quotes)
"""

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import (
    TradeStationError,
    AuthError,
    TokenExpiredError,
    InvalidCredentialsError,
    AuthRefreshFailedError,
    APIError,
    RateLimitError,
    NetworkError,
    ValidationError,
    OrderError,
    OrderRejectedError,
    PositionLimitError,
    InsufficientFundsError,
    OrderNotFoundError,
)

__all__ = [
    "TradeStationClient",
    "TradeStationError",
    "AuthError",
    "TokenExpiredError",
    "InvalidCredentialsError",
    "AuthRefreshFailedError",
    "APIError",
    "RateLimitError",
    "NetworkError",
    "ValidationError",
    "OrderError",
    "OrderRejectedError",
    "PositionLimitError",
    "InsufficientFundsError",
    "OrderNotFoundError",
]

__version__ = "0.1.0"

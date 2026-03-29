"""
KuCoin Execution Module

This module provides integration with the KuCoin cryptocurrency exchange API.
It is adapted from the Binance module for US users.

API Documentation: https://docs.kucoin.com/

Usage:
    from src.execution.kucoin import KuCoinClient

    async with KuCoinClient() as client:
        quotes = await client.get_quotes("BTC-USDT")
        order = await client.place_order(...)

Environment Configuration:
    Requires .env.kucoin file with KuCoin API credentials
"""

from src.execution.kucoin.exceptions import (
    APIError,
    AuthError,
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
from src.execution.kucoin.models import (
    KuCoinAccount,
    KuCoinKline,
    KuCoinOrder,
    KuCoinOrderBook,
    KuCoinQuote,
    KuCoinTrade,
    KuCoinWebSocketOrderUpdate,
    KuCoinWebSocketTrade,
)
from src.execution.kucoin.utils import CircuitBreaker, setup_logger

__all__ = [
    # Exceptions
    "KuCoinError",
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
    "KuCoinQuote",
    "KuCoinKline",
    "KuCoinOrderBook",
    "KuCoinTrade",
    "KuCoinAccount",
    "KuCoinOrder",
    "KuCoinWebSocketTrade",
    "KuCoinWebSocketOrderUpdate",
    # Utils
    "setup_logger",
    "CircuitBreaker",
]

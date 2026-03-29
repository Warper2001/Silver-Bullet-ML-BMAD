"""
Binance Market Data Module

This module provides market data access for Binance API.

API Documentation: https://binance-docs.github.io/apidocs/#market-data-endpoints
"""

from src.execution.binance.market_data.streaming import BinanceWebSocketClient, create_binance_websocket_client

__all__ = [
    "BinanceWebSocketClient",
    "create_binance_websocket_client",
]

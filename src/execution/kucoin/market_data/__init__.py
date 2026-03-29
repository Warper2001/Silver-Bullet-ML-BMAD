"""
KuCoin Market Data Module

This module provides market data access for KuCoin API.

API Documentation: https://docs.kucoin.com/
"""

from src.execution.kucoin.market_data.streaming import KuCoinWebSocketClient, create_kucoin_websocket_client

__all__ = [
    "KuCoinWebSocketClient",
    "create_kucoin_websocket_client",
]

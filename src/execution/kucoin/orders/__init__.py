"""
KuCoin Orders Module

This module provides order submission and management for KuCoin API.

API Documentation: https://docs.kucoin.com/#orders
"""

from src.execution.kucoin.orders.submission import KuCoinOrdersClient, create_kucoin_orders_client

__all__ = [
    "KuCoinOrdersClient",
    "create_kucoin_orders_client",
]

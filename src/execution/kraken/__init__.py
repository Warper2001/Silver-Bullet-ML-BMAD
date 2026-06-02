"""Kraken execution module — Futures and Spot clients."""

from src.execution.kraken.auth.api_key import KrakenFuturesAuth
from src.execution.kraken.client import KrakenFuturesClient
from src.execution.kraken.orders.submission import KrakenOrdersClient
from src.execution.kraken.spot.client import KrakenSpotClient
from src.execution.kraken.spot.models import SpotOrderResult

__all__ = [
    "KrakenFuturesClient",
    "KrakenFuturesAuth",
    "KrakenOrdersClient",
    "KrakenSpotClient",
    "SpotOrderResult",
]

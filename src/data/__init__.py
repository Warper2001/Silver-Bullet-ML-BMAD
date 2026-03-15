"""Data ingestion and Dollar Bar conversion module."""

from .auth import TradeStationAuth, TokenResponse
from .config import Settings, load_settings
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    TokenRefreshError,
)
from .websocket import TradeStationWebSocketClient, ConnectionState
from .models import MarketData, WebSocketMessage, DollarBar
from .transformation import DollarBarTransformer, BarBuilderState

__all__ = [
    "TradeStationAuth",
    "TokenResponse",
    "Settings",
    "load_settings",
    "AuthenticationError",
    "ConfigurationError",
    "TokenRefreshError",
    "TradeStationWebSocketClient",
    "ConnectionState",
    "MarketData",
    "WebSocketMessage",
    "DollarBar",
    "DollarBarTransformer",
    "BarBuilderState",
]

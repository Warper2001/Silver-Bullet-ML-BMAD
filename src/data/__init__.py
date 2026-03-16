"""Data ingestion and Dollar Bar conversion module."""

from .auth import TradeStationAuth, TokenResponse
from .config import Settings, load_settings
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    TokenRefreshError,
)
from .websocket import TradeStationWebSocketClient, ConnectionState
from .models import (
    MarketData,
    WebSocketMessage,
    DollarBar,
    ValidationResult,
    SwingPoint,
    MSSEvent,
    GapRange,
    FVGEvent,
    LiquiditySweepEvent,
    SilverBulletSetup,
    TimeWindow,
)
from .transformation import DollarBarTransformer, BarBuilderState
from .validation import DataValidator
from .gap_detection import GapDetector, GapStatistics
from .persistence import HDF5DataSink
from .orchestrator import DataPipelineOrchestrator

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
    "ValidationResult",
    "SwingPoint",
    "MSSEvent",
    "GapRange",
    "FVGEvent",
    "LiquiditySweepEvent",
    "SilverBulletSetup",
    "TimeWindow",
    "DollarBarTransformer",
    "BarBuilderState",
    "DataValidator",
    "GapDetector",
    "GapStatistics",
    "HDF5DataSink",
    "DataPipelineOrchestrator",
]

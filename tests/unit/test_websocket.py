"""Unit tests for TradeStation WebSocket client."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.data.models import MarketData, WebSocketMessage
from src.data.websocket import ConnectionState, TradeStationWebSocketClient
from src.data.auth import TradeStationAuth


class TestMarketData:
    """Test MarketData model validation."""

    def test_market_data_parsing(self) -> None:
        """Test valid market data parsing."""
        data = {
            "symbol": "MNQ",
            "timestamp": datetime.now(),
            "bid": 4523.25,
            "ask": 4523.50,
            "last": 4523.50,
            "volume": 1250,
        }

        market_data = MarketData(**data)

        assert market_data.symbol == "MNQ"
        assert market_data.bid == 4523.25
        assert market_data.ask == 4523.50
        assert market_data.last == 4523.50
        assert market_data.volume == 1250

    def test_required_fields_check_all_present(self) -> None:
        """Test has_required_fields returns True when all fields present."""
        data = {
            "symbol": "MNQ",
            "timestamp": datetime.now(),
            "bid": 4523.25,
            "ask": 4523.50,
            "last": 4523.50,
            "volume": 1250,
        }

        market_data = MarketData(**data)

        assert market_data.has_required_fields() is True

    def test_required_fields_check_missing_bid(self) -> None:
        """Test has_required_fields returns False when bid missing."""
        data = {
            "symbol": "MNQ",
            "timestamp": datetime.now(),
            "bid": None,
            "ask": 4523.50,
            "last": 4523.50,
            "volume": 1250,
        }

        market_data = MarketData(**data)

        assert market_data.has_required_fields() is False

    def test_required_fields_check_missing_ask(self) -> None:
        """Test has_required_fields returns False when ask missing."""
        data = {
            "symbol": "MNQ",
            "timestamp": datetime.now(),
            "bid": 4523.25,
            "ask": None,
            "last": 4523.50,
            "volume": 1250,
        }

        market_data = MarketData(**data)

        assert market_data.has_required_fields() is False

    def test_ask_greater_than_bid_validation(self) -> None:
        """Test ask must be greater than bid validation."""
        data = {
            "symbol": "MNQ",
            "timestamp": datetime.now(),
            "bid": 4523.50,
            "ask": 4523.25,  # ask < bid (invalid)
            "last": 4523.50,
            "volume": 1250,
        }

        with pytest.raises(ValidationError, match="ask must be greater than bid"):
            MarketData(**data)

    def test_ask_greater_than_bid_valid(self) -> None:
        """Test ask greater than bid is accepted."""
        data = {
            "symbol": "MNQ",
            "timestamp": datetime.now(),
            "bid": 4523.25,
            "ask": 4523.50,  # ask > bid (valid)
            "last": 4523.50,
            "volume": 1250,
        }

        market_data = MarketData(**data)

        assert market_data.ask == 4523.50


class TestWebSocketMessage:
    """Test WebSocketMessage model."""

    def test_websocket_message_parsing(self) -> None:
        """Test valid WebSocket message parsing."""
        message = WebSocketMessage(
            symbol="MNQ",
            timestamp="2026-03-15T14:30:00.123Z",
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=1250,
        )

        assert message.symbol == "MNQ"
        assert message.timestamp == "2026-03-15T14:30:00.123Z"
        assert message.bid == 4523.25
        assert message.ask == 4523.50
        assert message.last == 4523.50
        assert message.volume == 1250

    def test_websocket_message_to_market_data(self) -> None:
        """Test conversion to MarketData model."""
        message = WebSocketMessage(
            symbol="MNQ",
            timestamp="2026-03-15T14:30:00.123Z",
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=1250,
        )

        market_data = message.to_market_data()

        assert market_data.symbol == "MNQ"
        assert isinstance(market_data.timestamp, datetime)
        assert market_data.bid == 4523.25
        assert market_data.ask == 4523.50
        assert market_data.last == 4523.50
        assert market_data.volume == 1250


class TestConnectionState:
    """Test ConnectionState enum."""

    def test_connection_state_values(self) -> None:
        """Test all connection states are defined."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.ERROR.value == "error"


class TestTradeStationWebSocketClient:
    """Test TradeStationWebSocketClient."""

    @pytest.fixture
    def auth(self):
        """Create mock TradeStationAuth instance."""
        with patch("src.data.websocket.TradeStationAuth"):
            auth = MagicMock(spec=TradeStationAuth)
            auth.authenticate = AsyncMock(return_value="test_access_token")
            return auth

    @pytest.fixture
    def client(self, auth):
        """Create WebSocket client instance."""
        return TradeStationWebSocketClient(auth)

    def test_initialization(self, client: TradeStationWebSocketClient) -> None:
        """Test WebSocket client initializes correctly."""
        assert client._state == ConnectionState.DISCONNECTED
        assert client._websocket is None
        assert client._last_message_time is None
        assert client._message_count == 0

    def test_state_property(self, client: TradeStationWebSocketClient) -> None:
        """Test state property returns current state."""
        assert client.state == ConnectionState.DISCONNECTED

    def test_message_count_property(self, client: TradeStationWebSocketClient) -> None:
        """Test message_count property returns message count."""
        assert client.message_count == 0

    def test_staleness_threshold_configuration(self) -> None:
        """Test staleness threshold is 30 seconds."""
        assert TradeStationWebSocketClient.STALENESS_THRESHOLD == 30

    def test_heartbeat_interval_configuration(self) -> None:
        """Test heartbeat interval is 15 seconds."""
        assert TradeStationWebSocketClient.HEARTBEAT_INTERVAL == 15

    def test_retry_delays_configuration(self) -> None:
        """Test exponential backoff delays are correct."""
        client = TradeStationWebSocketClient(auth=None)
        assert client.RETRY_DELAYS == [1, 2, 4]  # 1s, 2s, 4s

    def test_max_retry_attempts_configuration(self) -> None:
        """Test max 3 retry attempts."""
        client = TradeStationWebSocketClient(auth=None)
        assert client.MAX_RETRY_ATTEMPTS == 3

    def test_max_queue_size_configuration(self) -> None:
        """Test max queue size is 10000."""
        client = TradeStationWebSocketClient(auth=None)
        assert client.MAX_QUEUE_SIZE == 10000
        assert client._data_queue.maxsize == 10000

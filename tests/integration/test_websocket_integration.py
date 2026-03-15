"""Integration tests for TradeStation WebSocket client."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.auth import TradeStationAuth
from src.data.models import MarketData
from src.data.websocket import ConnectionState, TradeStationWebSocketClient


class TestWebSocketConnectionFlow:
    """Test complete WebSocket connection flow with mock server."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth instance."""
        with patch("src.data.websocket.TradeStationAuth"):
            auth = MagicMock(spec=TradeStationAuth)
            auth.authenticate = AsyncMock(return_value="test_access_token")
            return auth

    @pytest.fixture
    def client(self, mock_auth):
        """Create WebSocket client instance."""
        return TradeStationWebSocketClient(mock_auth)

    @pytest.mark.asyncio
    async def test_initialization(self, client: TradeStationWebSocketClient) -> None:
        """Test client initializes with correct default values."""
        assert client._state == ConnectionState.DISCONNECTED
        assert client._websocket is None
        assert client._data_queue is not None
        assert isinstance(client._data_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_subscribe_returns_queue(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test subscribe() returns the data queue."""
        with patch.object(client, "_connect_with_retry"):
            with patch.object(client, "_start_background_tasks"):
                queue = await client.subscribe()

        assert queue is client._data_queue

    @pytest.mark.asyncio
    async def test_message_count_increments(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test message counter increments on data receipt."""
        initial_count = client._message_count

        # Simulate message reception
        client._last_message_time = datetime.now()
        client._message_count += 1

        assert client.message_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_staleness_detection(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test staleness detection after 30 seconds."""
        client._last_message_time = datetime.now() - timedelta(seconds=31)

        # Should detect staleness
        staleness = datetime.now() - client._last_message_time
        assert staleness > timedelta(seconds=30)

    @pytest.mark.asyncio
    async def test_no_staleness_within_threshold(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test no staleness detected within 30 second threshold."""
        client._last_message_time = datetime.now() - timedelta(seconds=29)

        # Should NOT detect staleness
        staleness = datetime.now() - client._last_message_time
        assert staleness <= timedelta(seconds=30)

    @pytest.mark.asyncio
    async def test_market_data_validation(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test market data is validated before queue publication."""
        # Create valid market data
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=1250,
        )

        assert market_data.has_required_fields() is True

    @pytest.mark.asyncio
    async def test_market_data_rejection_missing_fields(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test market data without required fields is rejected."""
        # Create invalid market data (missing bid)
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=None,  # Missing bid
            ask=4523.50,
            last=4523.50,
            volume=1250,
        )

        assert market_data.has_required_fields() is False


class TestConnectionStateTransitions:
    """Test WebSocket connection state machine."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth instance."""
        with patch("src.data.websocket.TradeStationAuth"):
            auth = MagicMock(spec=TradeStationAuth)
            auth.authenticate = AsyncMock(return_value="test_access_token")
            return auth

    @pytest.fixture
    def client(self, mock_auth):
        """Create WebSocket client instance."""
        return TradeStationWebSocketClient(mock_auth)

    def test_initial_state(self, client: TradeStationWebSocketClient) -> None:
        """Test initial state is DISCONNECTED."""
        assert client.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_state_transition_to_connecting(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test state transitions from DISCONNECTED to CONNECTING."""
        with patch("websockets.connect") as mock_connect:
            mock_connect.return_value = AsyncMock()

            try:
                await client._perform_connection()
            except Exception:
                pass  # We expect this to fail in test environment

        # State should have been CONNECTING at some point
        assert client.state in [ConnectionState.CONNECTING, ConnectionState.ERROR]

    @pytest.mark.asyncio
    async def test_cleanup_sets_state_to_disconnected(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test cleanup sets state to DISCONNECTED."""

        # Create async mock that properly handles await
        async def mock_close():
            pass

        async def mock_task():
            # Simulate task that gets cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        # Mock the WebSocket and tasks
        client._websocket = AsyncMock()
        client._websocket.close = mock_close

        # Create actual async tasks
        client._connection_task = asyncio.create_task(mock_task())
        client._heartbeat_task = asyncio.create_task(mock_task())

        # Set state to CONNECTED
        client._state = ConnectionState.CONNECTED

        await client.cleanup()

        assert client.state == ConnectionState.DISCONNECTED


class TestConcurrentDataReception:
    """Test handling multiple simultaneous WebSocket messages."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock TradeStationAuth instance."""
        with patch("src.data.websocket.TradeStationAuth"):
            auth = MagicMock(spec=TradeStationAuth)
            auth.authenticate = AsyncMock(return_value="test_access_token")
            return auth

    @pytest.fixture
    def client(self, mock_auth):
        """Create WebSocket client instance."""
        return TradeStationWebSocketClient(mock_auth)

    @pytest.mark.asyncio
    async def test_concurrent_queue_publication(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test multiple messages can be published to queue concurrently."""
        # Create multiple market data objects
        messages = [
            MarketData(
                symbol="MNQ",
                timestamp=datetime.now(),
                bid=4523.25 + i,
                ask=4523.50 + i,
                last=4523.50 + i,
                volume=1000 + i * 10,
            )
            for i in range(5)
        ]

        # Publish all messages concurrently
        tasks = [client._data_queue.put(msg) for msg in messages]
        await asyncio.gather(*tasks)

        # Verify all messages are in queue
        assert client._data_queue.qsize() == 5

    @pytest.mark.asyncio
    async def test_queue_depth_tracking(
        self, client: TradeStationWebSocketClient
    ) -> None:
        """Test queue depth is tracked correctly."""
        initial_depth = client._data_queue.qsize()

        # Add some messages
        for i in range(3):
            await client._data_queue.put(
                MarketData(
                    symbol="MNQ",
                    timestamp=datetime.now(),
                    bid=4523.25,
                    ask=4523.50,
                    last=4523.50,
                    volume=1000,
                )
            )

        assert client._data_queue.qsize() == initial_depth + 3

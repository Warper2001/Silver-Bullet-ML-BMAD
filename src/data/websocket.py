"""TradeStation WebSocket client for real-time market data."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import websockets
from pydantic import ValidationError

from .auth import TradeStationAuth
from .exceptions import AuthenticationError
from .models import MarketData, WebSocketMessage

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class TradeStationWebSocketClient:
    """TradeStation WebSocket client for MNQ futures market data.

    Handles:
    - WebSocket connection with OAuth 2.0 authentication
    - Real-time market data subscription (bid, ask, last, volume)
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong for connection health
    - Data validation and queue publication
    """

    WEBSOCKET_ENDPOINT = "wss://api.tradestation.com/v2/data/marketstream/subscribe"
    HEARTBEAT_INTERVAL = 15  # seconds
    STALENESS_THRESHOLD = 30  # seconds (no messages)
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
    MAX_QUEUE_SIZE = 10000  # Maximum messages in queue before backpressure

    def __init__(self, auth: TradeStationAuth) -> None:
        """Initialize WebSocket client.

        Args:
            auth: TradeStationAuth instance for access tokens
        """
        self.auth = auth
        self._state = ConnectionState.DISCONNECTED
        self._websocket: Optional[Any] = None
        self._data_queue: asyncio.Queue[MarketData] = asyncio.Queue(
            maxsize=self.MAX_QUEUE_SIZE
        )
        self._last_message_time: Optional[datetime] = None
        self._connection_task: Optional[asyncio.Task[None]] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._message_count = 0
        self._connection_start_time: Optional[datetime] = None

    async def connect(self) -> None:
        """Establish WebSocket connection with authentication.

        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection fails after all retries
        """
        await self._connect_with_retry()

    async def subscribe(self) -> asyncio.Queue[MarketData]:
        """Subscribe to MNQ futures market data.

        Returns:
            Async queue that will receive MarketData objects

        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection fails after all retries
        """
        await self.connect()
        await self._start_background_tasks()
        return self._data_queue

    async def _connect_with_retry(self) -> None:
        """Connect with exponential backoff retry.

        Raises:
            AuthenticationError: If authentication fails after all retries
            ConnectionError: If connection fails after all retries
        """
        last_error: Exception | None = None

        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                await self._perform_connection()
                return

            except (AuthenticationError, ConnectionError) as e:
                last_error = e
                logger.warning(
                    f"WebSocket connection attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS} "
                    f"failed: {str(e)}"
                )
                if attempt < len(self.RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise ConnectionError(
            f"WebSocket connection failed after {self.MAX_RETRY_ATTEMPTS} attempts"
        ) from last_error

    async def _perform_connection(self) -> None:
        """Perform actual WebSocket connection.

        Raises:
            AuthenticationError: If access token retrieval fails
            ConnectionError: If WebSocket connection fails
        """
        self._state = ConnectionState.CONNECTING

        try:
            # Get access token
            access_token = await self.auth.authenticate()

            # Establish WebSocket connection
            additional_headers = {"Authorization": f"Bearer {access_token}"}

            self._websocket = await websockets.connect(
                self.WEBSOCKET_ENDPOINT,
                additional_headers=additional_headers,
                ping_interval=self.HEARTBEAT_INTERVAL,
                ping_timeout=5,
            )

            # Send subscription message for MNQ futures data
            subscription_msg = {
                "symbol": "MNQ",
                "contract_type": "futures",
                "fields": ["bid", "ask", "last", "volume"],
                "interval": "tick",
            }
            await self._websocket.send(json.dumps(subscription_msg))
            logger.info(f"WebSocket subscription sent: {subscription_msg}")

            self._state = ConnectionState.CONNECTED
            self._connection_start_time = datetime.now()
            self._message_count = 0
            self._last_message_time = datetime.now()

            logger.info(
                f"WebSocket connection established (state: {self._state.value})"
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"WebSocket connection failed: {str(e)}") from e

    async def _start_background_tasks(self) -> None:
        """Start background tasks for message handling and heartbeat."""
        self._connection_task = asyncio.create_task(self._message_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("WebSocket background tasks started")

    async def _message_loop(self) -> None:
        """Receive and process WebSocket messages.

        This runs in a background task and:
        1. Receives messages from WebSocket
        2. Parses and validates market data
        3. Publishes valid data to queue
        4. Logs errors and publishes invalid data to error queue
        """
        while self._state == ConnectionState.CONNECTED:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self._websocket.recv(),  # type: ignore[union-attr]
                    timeout=1.0,
                )

                # Update last message time
                self._last_message_time = datetime.now()
                self._message_count += 1

                # Parse and validate message
                try:
                    ws_message = WebSocketMessage.model_validate_json(message)
                    market_data = ws_message.to_market_data()

                    # Validate required fields
                    if market_data.has_required_fields():
                        try:
                            await self._data_queue.put(market_data)
                            logger.debug(
                                f"Received market data: {market_data.symbol} "
                                f"@ {market_data.timestamp} "
                                f"(bid: {market_data.bid}, ask: {market_data.ask}, "
                                f"last: {market_data.last}, vol: {market_data.volume})"
                            )
                        except asyncio.QueueFull:
                            logger.warning(
                                f"Data queue full ({self._data_queue.qsize()}/{self.MAX_QUEUE_SIZE}), "
                                f"dropping market data message"
                            )
                    else:
                        logger.warning(
                            f"Market data missing required fields: {message}"
                        )

                except ValidationError as e:
                    logger.error(f"Invalid market data format: {e}")

            except asyncio.TimeoutError:
                # No message received (normal, check staleness)
                await self._check_staleness()

            except Exception as e:
                logger.error(f"Message processing error: {e}")
                self._state = ConnectionState.ERROR
                break

    async def _heartbeat_loop(self) -> None:
        """Monitor connection health and heartbeat.

        This runs in a background task and:
        1. Checks staleness (no messages for 30 seconds)
        2. Logs connection metrics
        3. Triggers reconnection if connection is stale
        """
        while self._state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._check_staleness()

                # Log connection metrics
                if self._connection_start_time:
                    uptime = datetime.now() - self._connection_start_time
                else:
                    uptime = timedelta(0)
                logger.info(
                    f"WebSocket connection stats: "
                    f"state={self._state.value}, "
                    f"uptime={uptime.total_seconds():.0f}s, "
                    f"messages={self._message_count}, "
                    f"queue_depth={self._data_queue.qsize()}"
                )

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def _check_staleness(self) -> None:
        """Check for data staleness.

        Logs warning if no messages received for 30 seconds.
        Triggers reconnection if stale.
        """
        if self._last_message_time is None:
            return

        staleness = datetime.now() - self._last_message_time

        if staleness > timedelta(seconds=self.STALENESS_THRESHOLD):
            logger.warning(
                f"Data staleness detected: {staleness.total_seconds():.0f}s "
                f"since last message"
            )

            # Trigger reconnection
            self._state = ConnectionState.RECONNECTING
            await self.reconnect()

    async def reconnect(self) -> None:
        """Reconnect to WebSocket with exponential backoff."""
        logger.info("Attempting WebSocket reconnection...")

        await self.cleanup()
        await self._connect_with_retry()
        await self._start_background_tasks()

    async def cleanup(self) -> None:
        """Clean up WebSocket connection and background tasks."""
        self._state = ConnectionState.DISCONNECTED

        # Cancel background tasks
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self._websocket:
            await self._websocket.close()

        logger.info("WebSocket connection cleaned up")

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def message_count(self) -> int:
        """Get total messages received since connection."""
        return self._message_count

"""TradeStation HTTP Streaming client for real-time market data.

This module implements HTTP streaming (chunked transfer encoding) for real-time
market data from TradeStation API v3, replacing the deprecated WebSocket approach.

TradeStation API v3 uses HTTP Streaming with chunked transfer encoding instead of
WebSocket connections. This implementation handles the chunked responses properly.

Key Features:
- HTTP Streaming with chunked transfer encoding
- Automatic reconnection with exponential backoff
- Variable-length JSON chunk handling
- Proper stream lifetime management
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import ValidationError

from .auth_v3 import TradeStationAuthV3
from .exceptions import AuthenticationError
from .models import MarketData

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """HTTP streaming connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class TradeStationHTTPStreamClient:
    """TradeStation HTTP Streaming client for market data.

    Uses HTTP/1.1 chunked transfer encoding for real-time market data
    instead of WebSocket connections. Handles proper JSON chunk parsing
    and stream lifetime management.

    Attributes:
        auth: TradeStationAuthV3 instance for authentication
        symbols: List of symbols to stream (e.g., ['MNQM26'])
    """

    # API endpoints
    STREAM_BASE_URL = "https://api.tradestation.com/v3/marketdata/stream"
    QUOTES_ENDPOINT = "{stream_base}/quotes/{symbol}"
    BARCHARTS_ENDPOINT = "{stream_base}/barcharts/{symbol}"

    # Connection parameters
    HEARTBEAT_INTERVAL = 15  # seconds
    STALENESS_THRESHOLD = 30  # seconds (no messages)
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
    MAX_QUEUE_SIZE = 10000  # Maximum messages in queue

    def __init__(
        self,
        auth: TradeStationAuthV3,
        symbols: list[str] = None,
    ) -> None:
        """Initialize HTTP streaming client.

        Args:
            auth: TradeStationAuthV3 instance for access tokens
            symbols: List of symbols to stream (default: ['MNQM26'])
        """
        self.auth = auth
        self.symbols = symbols or ["MNQM26"]
        self._state = ConnectionState.DISCONNECTED
        self._client: Optional[httpx.AsyncClient] = None
        self._data_queue: asyncio.Queue[MarketData] = asyncio.Queue(
            maxsize=self.MAX_QUEUE_SIZE
        )
        self._last_message_time: Optional[datetime] = None
        self._stream_task: Optional[asyncio.Task[None]] = None
        self._health_monitor_task: Optional[asyncio.Task[None]] = None
        self._message_count = 0
        self._connection_start_time: Optional[datetime] = None
        self._should_stop = False

    async def connect(self) -> None:
        """Establish HTTP streaming connection.

        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection fails after all retries
        """
        await self._connect_with_retry()

    async def subscribe(self) -> asyncio.Queue[MarketData]:
        """Subscribe to market data stream.

        Returns:
            Async queue that will receive MarketData objects

        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection fails after all retries
        """
        # Start background task immediately - it will handle connection
        self._should_stop = False
        self._state = ConnectionState.CONNECTING

        # Create HTTP client if needed
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5),
            )

        # Start the stream loop in background
        self._stream_task = asyncio.create_task(self._stream_loop())

        # Start health monitor task
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        # Wait a moment for connection to establish
        await asyncio.sleep(1)

        # Mark as connected
        self._state = ConnectionState.CONNECTED
        self._connection_start_time = datetime.now()
        self._last_message_time = datetime.now()

        logger.info("HTTP streaming subscription started")

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
                    f"HTTP stream connection attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS} "
                    f"failed: {str(e)}"
                )
                if attempt < len(self.RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise ConnectionError(
            f"HTTP stream connection failed after {self.MAX_RETRY_ATTEMPTS} attempts"
        ) from last_error

    async def _perform_connection(self) -> None:
        """Perform actual HTTP streaming connection.

        Raises:
            AuthenticationError: If access token retrieval fails
            ConnectionError: If HTTP connection fails
        """
        self._state = ConnectionState.CONNECTING

        try:
            # Get access token
            access_token = await self.auth.authenticate()

            # Create HTTP client with streaming enabled
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(max_keepalive_connections=5),
                )

            # Build streaming URL for quotes
            symbol = self.symbols[0] if self.symbols else "MNQM26"
            stream_url = f"{self.QUOTES_ENDPOINT.format(stream_base=self.STREAM_BASE_URL, symbol=symbol)}"

            logger.info(f"Starting HTTP stream from: {stream_url}")

            # Prepare headers with authentication
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.tradestation.streams.v3+json",
            }

            # Start streaming in background task
            self._should_stop = False
            self._connection_start_time = datetime.now()
            self._state = ConnectionState.CONNECTED
            self._last_message_time = datetime.now()

            logger.info(
                f"HTTP streaming connection established (state: {self._state.value})"
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"HTTP streaming connection failed: {str(e)}") from e

    async def _start_background_tasks(self) -> None:
        """Start background task for message handling."""
        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.info("HTTP streaming background task started")

    async def _stream_loop(self) -> None:
        """Receive and process HTTP stream messages.

        This runs in a background task and:
        1. Connects to HTTP streaming endpoint
        2. Receives chunked JSON data
        3. Parses and validates market data
        4. Publishes valid data to queue
        5. Automatically reconnects on connection failures or stale connections
        6. Monitors connection health and reconnects if stale
        """
        retry_count = 0
        logger.info("HTTP stream loop starting...")

        try:
            # Keep the connection alive regardless of initial state
            while not self._should_stop:
                try:
                    logger.debug("Attempting HTTP stream connection...")
                    # Get access token for this connection attempt
                    access_token = await self.auth.authenticate()

                    # Build streaming URL
                    symbol = self.symbols[0] if self.symbols else "MNQM26"
                    stream_url = f"{self.QUOTES_ENDPOINT.format(stream_base=self.STREAM_BASE_URL, symbol=symbol)}"

                    # Prepare headers
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.tradestation.streams.v3+json",
                    }

                    # Make streaming request
                    logger.debug(f"Connecting to {stream_url}...")
                    async with self._client.stream("GET", stream_url, headers=headers) as response:
                        if response.status_code != 200:
                            raise ConnectionError(f"HTTP stream error: {response.status_code}")

                        logger.info("HTTP stream receiving data...")
                        self._state = ConnectionState.CONNECTED  # Set connected state
                        retry_count = 0  # Reset retry count on successful connection
                        self._last_message_time = datetime.now()  # Reset stale timer

                        # Process chunked response with stale detection
                        buffer = ""
                        chunk_count = 0
                        last_health_check = datetime.now()

                        async for chunk in response.aiter_bytes():
                            if self._should_stop:
                                logger.info("Stream stopped by user request")
                                break

                            chunk_count += 1

                            # Decode chunk and add to buffer
                            try:
                                chunk_text = chunk.decode('utf-8')
                                buffer += chunk_text

                                # Process complete JSON objects (separated by newlines or braces)
                                buffer = await self._process_json_buffer(buffer)

                                if chunk_count % 100 == 0:
                                    logger.debug(f"Processed {chunk_count} chunks...")

                                # Periodic health check (every 30 seconds)
                                now = datetime.now()
                                if (now - last_health_check).total_seconds() >= 30:
                                    await self._check_connection_health()
                                    last_health_check = now

                            except UnicodeDecodeError as e:
                                logger.warning(f"Failed to decode chunk: {e}")
                                continue

                        # Stream ended normally - check if it's stale
                        logger.info("HTTP stream ended normally")
                        if await self._is_connection_stale():
                            logger.warning("Connection was stale, reconnecting...")
                            await asyncio.sleep(1)  # Brief pause before reconnect
                            continue  # Reconnect immediately

                        logger.info("HTTP stream ended, reconnecting in 5 seconds...")
                        await asyncio.sleep(5)  # Wait before reconnecting

                except Exception as e:
                    if not self._should_stop:
                        retry_count += 1
                        if retry_count <= self.MAX_RETRY_ATTEMPTS:
                            delay = self.RETRY_DELAYS[min(retry_count - 1, len(self.RETRY_DELAYS) - 1)]
                            logger.warning(
                                f"HTTP stream connection lost (attempt {retry_count}/{self.MAX_RETRY_ATTEMPTS}): {e}. "
                                f"Reconnecting in {delay} seconds..."
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"HTTP stream failed after {retry_count} attempts: {e}")
                            self._state = ConnectionState.ERROR
                            # Don't break - keep trying to reconnect indefinitely
                            retry_count = 0
                            logger.info("Resetting retry count and attempting reconnection...")
                            await asyncio.sleep(10)  # Longer pause before retry

        except asyncio.CancelledError:
            logger.info("HTTP stream loop cancelled")
            raise
        except Exception as e:
            logger.error(f"HTTP stream loop error: {e}", exc_info=True)
            self._state = ConnectionState.ERROR
        finally:
            logger.info("HTTP stream loop ended")

    async def _process_json_buffer(self, buffer: str) -> str:
        """Process JSON objects from buffer.

        Args:
            buffer: Current buffer content

        Returns:
            Remaining buffer content after processing complete JSON objects
        """
        lines = buffer.split('\n')
        processed_buffer = ""

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON
            try:
                data = json.loads(line)

                # Check for stream status messages
                if "StreamStatus" in data:
                    if data["StreamStatus"] == "GoAway":
                        logger.info("Server sent GoAway - reconnecting...")
                        await asyncio.sleep(1)
                        break
                    elif data["StreamStatus"] == "EndSnapshot":
                        logger.debug("End of initial snapshot")
                        continue

                # Check for error messages
                if "Error" in data:
                    logger.error(f"Stream error: {data['Error']}")
                    if data.get("Error") == "DualLogon":
                        logger.error("Dual login detected - stopping stream")
                        self._should_stop = True
                        break
                    continue

                # Process market data
                await self._process_market_data(data)

                # Update last message time
                self._last_message_time = datetime.now()
                self._message_count += 1

            except json.JSONDecodeError:
                # Not a complete JSON object, keep in buffer
                if i == len(lines) - 1:
                    processed_buffer = line
                continue

        return processed_buffer

    async def _health_monitor_loop(self) -> None:
        """Background task that monitors connection health and triggers reconnection if stale.

        This runs independently of the stream loop and provides an additional layer of
        protection against stale connections that the stream loop might not detect.

        Checks every 30 seconds if the connection is stale (no messages for STALENESS_THRESHOLD).
        If stale, logs a warning but doesn't force reconnection (stream loop handles that).
        """
        logger.info("Health monitor loop started")

        try:
            while not self._should_stop:
                await asyncio.sleep(30)  # Check every 30 seconds

                if self._should_stop:
                    break

                # Check if connection is stale
                if await self._is_connection_stale():
                    logger.warning(
                        "⚠️  Health monitor detected stale connection - "
                        "stream loop should auto-reconnect"
                    )

                    # Log detailed stats
                    stats = self.get_stats()
                    logger.warning(f"Connection stats: {stats}")

        except asyncio.CancelledError:
            logger.info("Health monitor loop cancelled")
        except Exception as e:
            logger.error(f"Health monitor loop error: {e}", exc_info=True)

        logger.info("Health monitor loop ended")

    async def _process_market_data(self, data: dict[str, Any]) -> None:
        """Process market data from stream.

        Args:
            data: Parsed JSON data from stream
        """
        try:
            # Convert stream data to MarketData model
            market_data = self._parse_stream_data(data)

            # Validate required fields
            if market_data.has_required_fields():
                try:
                    await self._data_queue.put(market_data)
                except asyncio.QueueFull:
                    logger.warning("Data queue full, dropping message")

        except ValidationError as e:
            logger.warning(f"Failed to validate market data: {e}")
        except Exception as e:
            logger.warning(f"Failed to process market data: {e}")

    def _parse_stream_data(self, data: dict[str, Any]) -> MarketData:
        """Parse stream data into MarketData object.

        Args:
            data: Raw stream data dictionary

        Returns:
            MarketData object
        """
        # Extract symbol from data
        symbol = data.get("Symbol") or (self.symbols[0] if self.symbols else "MNQM26")

        # Parse timestamp
        timestamp_str = data.get("TimeStamp") or data.get("LastTradeTime")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Extract price data
        last_price = self._parse_float(data.get("Last") or data.get("LastTradePrice"))
        bid = self._parse_float(data.get("Bid"))
        ask = self._parse_float(data.get("Ask"))
        bid_size = self._parse_int(data.get("BidSize") or data.get("BidLot"))
        ask_size = self._parse_int(data.get("AskSize") or data.get("AskLot"))
        volume = self._parse_int(data.get("Volume") or data.get("TotalVolume"))

        # Create MarketData object (allow None/0 for closed market data)
        return MarketData(
            symbol=symbol,
            timestamp=timestamp,
            last_price=last_price,  # Can be None or 0 for closed market
            bid=bid,  # Can be None or 0 for closed market
            ask=ask,  # Can be None or 0 for closed market
            volume=volume or 0,
        )

    def _parse_float(self, value: Any) -> Optional[float]:
        """Safely parse float value.

        Args:
            value: Value to parse

        Returns:
            Float value or None if parsing fails
        """
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """Safely parse integer value.

        Args:
            value: Value to parse

        Returns:
            Integer value or None if parsing fails
        """
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    async def disconnect(self) -> None:
        """Disconnect from HTTP stream."""
        logger.info("Disconnecting HTTP stream...")
        self._should_stop = True
        self._state = ConnectionState.DISCONNECTED

        # Cancel health monitor task
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("HTTP stream disconnected")

    async def _is_connection_stale(self) -> bool:
        """Check if connection is stale (no messages for STALENESS_THRESHOLD seconds).

        Returns:
            True if connection is stale, False otherwise
        """
        if self._last_message_time is None:
            return False

        stale_duration = (datetime.now() - self._last_message_time).total_seconds()
        is_stale = stale_duration > self.STALENESS_THRESHOLD

        if is_stale:
            logger.warning(
                f"Connection is stale: No messages for {stale_duration:.1f} seconds "
                f"(threshold: {self.STALENESS_THRESHOLD}s)"
            )

        return is_stale

    async def _check_connection_health(self) -> None:
        """Check connection health and log warnings if stale.

        This is called periodically during streaming to detect stale connections.
        """
        if self._last_message_time is None:
            return

        time_since_last_message = (datetime.now() - self._last_message_time).total_seconds()

        if time_since_last_message > self.STALENESS_THRESHOLD:
            logger.warning(
                f"⚠️  Connection health check: No messages for {time_since_last_message:.1f} seconds "
                f"(threshold: {self.STALENESS_THRESHOLD}s)"
            )
        else:
            logger.debug(
                f"Connection healthy: Last message {time_since_last_message:.1f} seconds ago"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics.

        Returns:
            Dictionary with connection statistics
        """
        time_since_last_message = None
        if self._last_message_time is not None:
            time_since_last_message = (datetime.now() - self._last_message_time).total_seconds()

        return {
            "state": self._state.value,
            "message_count": self._message_count,
            "connection_start_time": self._connection_start_time,
            "last_message_time": self._last_message_time,
            "time_since_last_message_seconds": time_since_last_message,
            "is_stale": time_since_last_message is not None and time_since_last_message > self.STALENESS_THRESHOLD,
            "symbols": self.symbols,
        }
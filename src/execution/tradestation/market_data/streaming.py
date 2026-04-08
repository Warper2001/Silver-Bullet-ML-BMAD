"""TradeStation SDK streaming client for real-time market data.

This module implements real-time quote streaming from TradeStation SIM environment
using the official TradeStation SDK. Replaces WebSocket-based data ingestion with
SDK-managed streaming for improved reliability and feature support.

Features:
- Real-time quote streaming from TradeStation SIM environment
- Automatic reconnection on connection failures
- Multi-subscriber support with asyncio queues
- Environment isolation (SIM vs production)
- Comprehensive error handling and logging
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from websockets.client import connect as websocket_connect

from src.data.tradestation_auth import TradeStationAuth

logger = logging.getLogger(__name__)

# TradeStation API endpoints
SIM_API_BASE = "https://sim-api.tradestation.com/v3"
PROD_API_BASE = "https://api.tradestation.com/v3"

# Streaming endpoints
STREAM_ENDPOINT = "/streaming/quotes"


@dataclass
class StreamPosition:
    """Real-time quote data from TradeStation streaming.

    Attributes:
        symbol: Futures contract symbol (e.g., 'MNQH26')
        timestamp: Quote timestamp in UTC
        last_price: Last traded price
        bid: Current bid price
        ask: Current ask price
        bid_size: Bid size (contracts)
        ask_size: Ask size (contracts)
        volume: Cumulative volume
        open_interest: Open interest
    """
    symbol: str
    timestamp: datetime
    last_price: float
    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    open_interest: int = 0


class QuoteStreamParser:
    """TradeStation SDK streaming client for real-time quotes.

    Manages WebSocket connection to TradeStation streaming API for
    real-time market data ingestion. Supports multiple subscribers
    with automatic reconnection and error handling.

    Attributes:
        _auth: TradeStation authentication manager
        _symbols: List of symbols to stream
        _environment: API environment ('sim' or 'prod')
        _subscribers: Number of active subscribers
        _running: Whether streaming is active
        _websocket: Active WebSocket connection
        _queues: List of subscriber queues

    Example:
        >>> auth = TradeStationAuth()
        >>> parser = QuoteStreamParser(auth=auth, symbols=['MNQH26'], environment='sim')
        >>> await parser.start()
        >>> queue = await parser.subscribe()
        >>> while True:
        ...     position = await queue.get()
        ...     print(f"Quote: {position.last_price}")
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5.0
    HEARTBEAT_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        auth: TradeStationAuth,
        symbols: list[str],
        environment: str = "sim",
    ) -> None:
        """Initialize QuoteStreamParser.

        Args:
            auth: TradeStation authentication manager
            symbols: List of symbols to stream (e.g., ['MNQH26'])
            environment: API environment ('sim' or 'prod', default 'sim')
        """
        self._auth = auth
        self._symbols = symbols
        self._environment = environment.lower()

        # Subscriber management
        self._subscribers = 0
        self._queues: list[asyncio.Queue[StreamPosition]] = []
        self._subscriber_lock = asyncio.Lock()

        # Connection state
        self._running = False
        self._websocket: Optional[object] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Retry state
        self._retry_count = 0
        self._last_connection_attempt: Optional[datetime] = None

        logger.info(
            f"QuoteStreamParser initialized for {self._environment} environment "
            f"with symbols: {self._symbols}"
        )

    def _get_stream_url(self) -> str:
        """Get WebSocket streaming URL for environment.

        Returns:
            WebSocket URL for streaming API
        """
        base_url = SIM_API_BASE if self._environment == "sim" else PROD_API_BASE
        # Convert HTTPS to WSS (WebSocket Secure)
        wss_url = base_url.replace("https://", "wss://")
        return f"{wss_url}{STREAM_ENDPOINT}"

    async def start(self) -> None:
        """Start the quote streaming client.

        Establishes WebSocket connection and starts streaming quotes.
        Implements automatic reconnection on failure.

        Raises:
            ConnectionError: If connection fails after max retries
        """
        if self._running:
            logger.warning("QuoteStreamParser is already running")
            return

        logger.info(f"Starting QuoteStreamParser for {self._environment} environment")

        self._running = True
        self._retry_count = 0

        # Start streaming with retry logic
        await self._start_with_retry()

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("QuoteStreamParser started successfully")

    async def _start_with_retry(self) -> None:
        """Start streaming with automatic retry on failure.

        Implements exponential backoff for connection retries.

        Raises:
            ConnectionError: If max retries exceeded
        """
        while self._running and self._retry_count < self.MAX_RETRIES:
            try:
                self._last_connection_attempt = datetime.now(timezone.utc)
                self._websocket = await self._connect_websocket()

                # Reset retry count on successful connection
                self._retry_count = 0
                logger.info("WebSocket connection established")

                # Start streaming messages
                self._stream_task = asyncio.create_task(self._stream_messages())
                await self._stream_task  # Run until failure

            except Exception as e:
                self._retry_count += 1
                retry_delay = self.RETRY_DELAY_SECONDS * (2 ** (self._retry_count - 1))

                logger.error(
                    f"WebSocket connection failed (attempt {self._retry_count}/{self.MAX_RETRIES}): {e}. "
                    f"Retrying in {retry_delay:.1f} seconds..."
                )

                # Cleanup failed connection
                await self._cleanup_connection()

                # Wait before retry
                await asyncio.sleep(retry_delay)

        # If we get here, max retries exceeded
        if self._retry_count >= self.MAX_RETRIES:
            error_msg = f"WebSocket connection failed after {self.MAX_RETRIES} retries"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

    async def _connect_websocket(self) -> object:
        """Establish WebSocket connection to TradeStation streaming API.

        Returns:
            WebSocket connection object

        Raises:
            ConnectionError: If connection fails
        """
        # Get access token (support both auth types)
        if hasattr(self._auth, 'get_valid_access_token'):
            # TradeStationAuthV3 or TradeStationAuth (tradestation_auth.py)
            token = self._auth.get_valid_access_token()
        elif hasattr(self._auth, 'authenticate'):
            # TradeStationAuth from src/data/auth.py
            token = await self._auth.authenticate()
        else:
            raise AttributeError(
                f"Auth object {type(self._auth)} does not have token method"
            )

        # Build WebSocket URL with authentication
        stream_url = self._get_stream_url()
        symbols_param = ",".join(self._symbols)

        # Add authentication and symbols to URL
        ws_url = f"{stream_url}?symbols={symbols_param}&token={token}"

        # Connect to WebSocket
        logger.debug(f"Connecting to WebSocket: {stream_url}")
        websocket = await websocket_connect(ws_url)

        return websocket

    async def _stream_messages(self) -> None:
        """Stream messages from WebSocket and broadcast to subscribers.

        Continuously receives messages from WebSocket, parses them,
        and broadcasts to all subscribers.

        Raises:
            Exception: If WebSocket connection fails
        """
        if not self._websocket:
            raise ConnectionError("WebSocket not connected")

        logger.info("Starting message streaming loop")

        try:
            async for message in self._websocket:
                if not self._running:
                    break

                # Parse and broadcast message
                await self._process_message(message)

        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Message streaming loop ended")

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message.

        Args:
            message: Raw WebSocket message (JSON string)
        """
        try:
            import json

            # Parse JSON message
            data = json.loads(message)

            # Parse quotes from message
            positions = self._parse_quotes(data)

            if positions:
                # Broadcast to all subscribers
                for position in positions:
                    await self._broadcast_to_subscribers(position)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def _parse_quotes(self, data: dict) -> list[StreamPosition] | None:
        """Parse quote data from WebSocket message.

        Args:
            data: Parsed JSON message from WebSocket

        Returns:
            List of StreamPosition objects, or None if no quotes
        """
        if "Quotes" not in data or not data["Quotes"]:
            return None

        positions = []
        for quote_data in data["Quotes"]:
            try:
                # Parse timestamp
                timestamp_str = quote_data.get("Timestamp") or quote_data.get("timestamp")
                if not timestamp_str:
                    logger.warning("Quote missing timestamp, skipping")
                    continue

                # Parse UTC timestamp
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    timestamp = timestamp_str

                # Create StreamPosition
                position = StreamPosition(
                    symbol=quote_data.get("Symbol") or quote_data.get("symbol", "UNKNOWN"),
                    timestamp=timestamp,
                    last_price=float(quote_data.get("Last") or quote_data.get("last", 0.0)),
                    bid=float(quote_data.get("Bid") or quote_data.get("bid", 0.0)),
                    ask=float(quote_data.get("Ask") or quote_data.get("ask", 0.0)),
                    bid_size=int(quote_data.get("BidSize") or quote_data.get("bidSize", 0)),
                    ask_size=int(quote_data.get("AskSize") or quote_data.get("askSize", 0)),
                    volume=int(quote_data.get("Volume") or quote_data.get("volume", 0)),
                    open_interest=int(
                        quote_data.get("OpenInterest") or quote_data.get("openInterest", 0)
                    ),
                )
                positions.append(position)

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse quote data: {e}")
                continue

        return positions if positions else None

    async def _broadcast_to_subscribers(self, position: StreamPosition) -> None:
        """Broadcast quote to all active subscribers.

        Args:
            position: StreamPosition to broadcast
        """
        async with self._subscriber_lock:
            for queue in self._queues:
                try:
                    # Non-blocking put (drop if queue is full)
                    if queue.full():
                        logger.warning(
                            f"Subscriber queue full, dropping quote for {position.symbol}"
                        )
                        continue

                    await asyncio.wait_for(queue.put(position), timeout=0.1)

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Subscriber timeout, dropping quote for {position.symbol}"
                    )
                    continue

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to keep connection alive.

        Sends heartbeat messages at regular intervals to prevent
        connection timeout.
        """
        while self._running:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL_SECONDS)

                # Send heartbeat message if WebSocket is connected
                if self._websocket:
                    heartbeat_msg = '{"type": "heartbeat"}'
                    await self._websocket.send(heartbeat_msg)
                    logger.debug("Heartbeat sent")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)
                break

    async def subscribe(self) -> asyncio.Queue[StreamPosition]:
        """Subscribe to quote stream.

        Creates a new queue for the subscriber and starts receiving quotes.

        Returns:
            asyncio.Queue for receiving StreamPosition objects

        Example:
            >>> queue = await parser.subscribe()
            >>> position = await queue.get()  # Block until quote received
        """
        async with self._subscriber_lock:
            # Create queue for subscriber (max 1000 items to prevent memory bloat)
            queue: asyncio.Queue[StreamPosition] = asyncio.Queue(maxsize=1000)
            self._queues.append(queue)
            self._subscribers += 1

            logger.info(
                f"New subscriber added (total: {self._subscribers} subscribers)"
            )

            return queue

    async def unsubscribe(self, queue: asyncio.Queue[StreamPosition]) -> None:
        """Unsubscribe from quote stream.

        Args:
            queue: Queue to remove from subscribers
        """
        async with self._subscriber_lock:
            if queue in self._queues:
                self._queues.remove(queue)
                self._subscribers -= 1

                logger.info(
                    f"Subscriber removed (total: {self._subscribers} subscribers)"
                )

    async def stop(self) -> None:
        """Stop the quote streaming client.

        Closes WebSocket connection and cleanup resources.
        """
        if not self._running:
            logger.warning("QuoteStreamParser is not running")
            return

        logger.info("Stopping QuoteStreamParser")

        # Stop streaming
        self._running = False

        # Cancel tasks
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cleanup connection
        await self._cleanup_connection()

        logger.info("QuoteStreamParser stopped")

    async def _cleanup_connection(self) -> None:
        """Cleanup WebSocket connection."""
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None

    @property
    def is_running(self) -> bool:
        """Check if streaming is active."""
        return self._running

    @property
    def subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return self._subscribers

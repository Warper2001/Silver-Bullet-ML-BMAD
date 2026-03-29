"""
BYDFI WebSocket Client

Provides real-time market data streaming from BYDFI WebSocket API.
Based on BYDFI API documentation: https://developers.bydfi.com/en/domainName

WebSocket URL: wss://stream.bydfi.com/v1/public/fapi

Features:
- Direct WebSocket connection (no token handshake)
- Real-time trade streaming
- Automatic reconnection
- Staleness detection
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from src.data.bydfi_config import load_bydfi_settings

logger = logging.getLogger(__name__)


class BYDFIWebSocketMessage:
    """
    BYDFI WebSocket message.

    Attributes:
        event: Message type (trade, kline, etc.)
        symbol: Trading symbol
        data: Message data
        timestamp: Message timestamp
    """

    def __init__(
        self,
        event: str,
        symbol: str,
        data: dict,
        timestamp: datetime,
    ):
        self.event = event
        self.symbol = symbol
        self.data = data
        self.timestamp = timestamp

    def __repr__(self):
        return f"BYDFIWebSocketMessage(event={self.event}, symbol={self.symbol}, timestamp={self.timestamp})"


class BYDFIWebSocketClient:
    """
    BYDFI WebSocket client for real-time market data.

    Unlike KuCoin, BYDFI uses direct WebSocket connection without token handshake.

    Example:
        >>> client = BYDFIWebSocketClient()
        >>> await client.connect()
        >>> await client.subscribe_trades("BTC-USDT")
        >>> async for message in client.messages():
        ...     print(f"Trade: {message.data}")
    """

    def __init__(self):
        """Initialize BYDFI WebSocket client."""
        settings = load_bydfi_settings()

        self.websocket_url = settings.websocket_url
        self.trading_symbol = settings.bydfi_trading_symbol

        # WebSocket connection
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._message_queue: asyncio.Queue[BYDFIWebSocketMessage] = asyncio.Queue()

        # Reconnection settings
        self._reconnect_delay = 5  # seconds
        self._max_reconnect_attempts = 10
        self._reconnect_attempts = 0

        # Staleness detection
        self._last_message_time: Optional[datetime] = None
        self._staleness_seconds = 30

        logger.info(f"BYDFI WebSocket client initialized: url={self.websocket_url}")

    async def connect(self):
        """
        Connect to BYDFI WebSocket.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info(f"Connecting to BYDFI WebSocket: {self.websocket_url}")

            self._ws = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=20,
            )

            self._connected = True
            self._reconnect_attempts = 0
            self._last_message_time = datetime.now(timezone.utc)

            logger.info("BYDFI WebSocket connected successfully")

            # Start message handler
            asyncio.create_task(self._message_handler())

        except Exception as e:
            logger.error(f"Failed to connect to BYDFI WebSocket: {e}")
            raise ConnectionError(f"WebSocket connection failed: {e}")

    async def disconnect(self):
        """Disconnect from BYDFI WebSocket."""
        self._connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("BYDFI WebSocket disconnected")

    async def _message_handler(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    self._last_message_time = datetime.now(timezone.utc)

                    # Parse message type
                    event = data.get("event", "unknown")
                    symbol = data.get("symbol", self.trading_symbol)

                    # Create message object
                    ws_message = BYDFIWebSocketMessage(
                        event=event,
                        symbol=symbol,
                        data=data,
                        timestamp=self._last_message_time,
                    )

                    # Put in queue
                    await self._message_queue.put(ws_message)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False

        except Exception as e:
            logger.error(f"Message handler error: {e}")
            self._connected = False

        # Attempt reconnection
        if self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            logger.info(f"Attempting to reconnect ({self._reconnect_attempts}/{self._max_reconnect_attempts})")
            await asyncio.sleep(self._reconnect_delay)
            await self.connect()

    async def subscribe_trades(self, symbol: str):
        """
        Subscribe to trade updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., BTC-USDT)
        """
        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket not connected")

        try:
            # BYDFI subscription format (adjust based on actual API)
            subscription_msg = {
                "action": "subscribe",
                "channel": f"trade:{symbol}",
            }

            await self._ws.send(json.dumps(subscription_msg))
            logger.info(f"Subscribed to trades for {symbol}")

        except Exception as e:
            logger.error(f"Error subscribing to trades: {e}")
            raise

    async def subscribe_kline(self, symbol: str, interval: str = "5m"):
        """
        Subscribe to kline updates for a symbol.

        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 1d)
        """
        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket not connected")

        try:
            # BYDFI subscription format (adjust based on actual API)
            subscription_msg = {
                "action": "subscribe",
                "channel": f"kline_{interval}:{symbol}",
            }

            await self._ws.send(json.dumps(subscription_msg))
            logger.info(f"Subscribed to {interval} klines for {symbol}")

        except Exception as e:
            logger.error(f"Error subscribing to klines: {e}")
            raise

    async def messages(self) -> asyncio.AsyncGenerator[BYDFIWebSocketMessage, None]:
        """
        Get messages from WebSocket.

        Yields:
            BYDFIWebSocketMessage: Next message from queue
        """
        while self._connected or not self._message_queue.empty():
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message

            except asyncio.TimeoutError:
                # Check for staleness
                if self._last_message_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_message_time).total_seconds()
                    if elapsed > self._staleness_seconds:
                        logger.warning(f"No messages for {elapsed}s, connection may be stale")

    def is_stale(self) -> bool:
        """
        Check if WebSocket connection is stale.

        Returns:
            bool: True if no messages received recently
        """
        if not self._last_message_time:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_message_time).total_seconds()
        return elapsed > self._staleness_seconds

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()


def create_bydfi_websocket_client() -> BYDFIWebSocketClient:
    """
    Factory function to create BYDFI WebSocket client.

    Returns:
        BYDFIWebSocketClient: Configured WebSocket client
    """
    return BYDFIWebSocketClient()

"""
Binance WebSocket Market Data Streaming

This module provides WebSocket streaming for real-time Binance market data.

Binance uses TRUE WebSocket (wss://) for real-time data streaming, unlike
TradeStation which uses HTTP chunked transfer with SSE format.

API Docs: https://binance-docs.github.io/apidocs/#websocket-market-streams

Architecture:
- WebSocket connection management
- Message parsing and validation
- Reconnection logic with exponential backoff
- State recovery after disconnect
- Staleness detection (30-second threshold)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

import websockets

from src.execution.binance.exceptions import NetworkError
from src.execution.binance.models import BinanceWebSocketTrade
from src.execution.binance.utils import setup_logger


class BinanceWebSocketClient:
    """
    Binance WebSocket client for real-time market data streaming.

    This client connects to Binance WebSocket streams and provides:
    - Real-time trade streaming
    - Automatic reconnection with exponential backoff
    - State recovery after disconnect
    - Staleness detection
    - Graceful shutdown

    API Docs: https://binance-docs.github.io/apidocs/#trade-streams

    Attributes:
        symbol: Trading symbol to stream
        base_url: WebSocket base URL
        reconnect_interval: Initial reconnection interval (seconds)
        max_reconnect_interval: Maximum reconnection interval (seconds)
        staleness_threshold: Staleness detection threshold (seconds)
        is_connected: Connection status
        last_message_time: Timestamp of last received message
        last_trade_id: Last received trade ID (for state recovery)

    Example:
        >>> client = BinanceWebSocketClient(symbol="BTCUSDT")
        >>> await client.connect()
        >>> async for trade in client.stream_trades():
        ...     print(f"Trade: {trade.price} @ {trade.quantity}")
        ...     # Break when done
        ...     break
        >>> await client.disconnect()
    """

    def __init__(
        self,
        symbol: str,
        base_url: str = "wss://stream.binance.com:9443/ws",
        reconnect_interval: float = 1.0,
        max_reconnect_interval: float = 60.0,
        staleness_threshold: float = 30.0,
    ) -> None:
        """
        Initialize Binance WebSocket client.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            base_url: WebSocket base URL
            reconnect_interval: Initial reconnection interval (seconds)
            max_reconnect_interval: Maximum reconnection interval (seconds)
            staleness_threshold: Staleness detection threshold (seconds)
        """
        self.symbol = symbol.upper()
        self.base_url = base_url
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.staleness_threshold = staleness_threshold

        # Connection state
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.is_connected = False
        self.is_running = False

        # State tracking
        self.last_message_time: datetime | None = None
        self.last_trade_id: int | None = None
        self.gaps_detected = 0

        # Logging
        self.logger = setup_logger(f"{__name__}.BinanceWebSocketClient")

    @property
    def stream_url(self) -> str:
        """
        Get WebSocket stream URL for the symbol.

        Returns:
            WebSocket stream URL

        Example:
            >>> client = BinanceWebSocketClient(symbol="BTCUSDT")
            >>> print(client.stream_url)
            'wss://stream.binance.com:9443/ws/btcusdt@trade'
        """
        return f"{self.base_url}/{self.symbol.lower()}@trade"

    async def connect(self) -> None:
        """
        Connect to Binance WebSocket stream.

        Raises:
            NetworkError: If connection fails
        """
        try:
            self.logger.info(f"Connecting to Binance WebSocket for {self.symbol}...")

            # Connect to WebSocket
            self.websocket = await websockets.connect(self.stream_url)

            self.is_connected = True
            self.is_running = True
            self.last_message_time = datetime.now(timezone.utc)

            self.logger.info(f"Connected to Binance WebSocket for {self.symbol}")

        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise NetworkError(f"WebSocket connection failed: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from Binance WebSocket stream.
        """
        self.is_running = False

        if self.websocket:
            try:
                await self.websocket.close()
                self.logger.info(f"Disconnected from Binance WebSocket for {self.symbol}")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

        self.is_connected = False
        self.websocket = None

    async def stream_trades(self):
        """
        Stream real-time trades from Binance WebSocket.

        Yields:
            BinanceWebSocketTrade objects

        Raises:
            NetworkError: If connection fails
        """
        if not self.is_connected or not self.websocket:
            raise NetworkError("WebSocket not connected")

        try:
            async for message in self.websocket:
                if not self.is_running:
                    break

                # Update last message time
                self.last_message_time = datetime.now(timezone.utc)

                # Parse message
                trade = self._parse_message(message)

                # Validate sequence (state recovery)
                if self.last_trade_id is not None:
                    if trade.trade_id < self.last_trade_id:
                        self.logger.warning(
                            f"Received out-of-order trade: {trade.trade_id} "
                            f"(expected > {self.last_trade_id})"
                        )
                    elif trade.trade_id > self.last_trade_id + 1:
                        # Gap detected
                        gap = trade.trade_id - self.last_trade_id - 1
                        self.gaps_detected += 1
                        self.logger.warning(
                            f"Trade gap detected: {gap} trades missing between "
                            f"{self.last_trade_id} and {trade.trade_id}"
                        )

                self.last_trade_id = trade.trade_id

                yield trade

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"WebSocket connection closed: {e}")
            self.is_connected = False
            raise NetworkError(f"WebSocket connection closed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error streaming trades: {e}")
            raise

    def _parse_message(self, message: str) -> BinanceWebSocketTrade:
        """
        Parse WebSocket message into BinanceWebSocketTrade.

        Args:
            message: JSON message string

        Returns:
            BinanceWebSocketTrade object

        Raises:
            ValueError: If message is invalid
        """
        import json

        try:
            data = json.loads(message)
            return BinanceWebSocketTrade(**data)

        except Exception as e:
            self.logger.error(f"Failed to parse WebSocket message: {e}")
            raise ValueError(f"Invalid WebSocket message: {message}") from e

    async def check_staleness(self) -> bool:
        """
        Check if data stream is stale.

        Data is considered stale if no messages received within
        staleness_threshold seconds.

        Returns:
            True if stream is stale, False otherwise
        """
        if self.last_message_time is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
        return elapsed > self.staleness_threshold

    async def reconnect_with_backoff(self) -> bool:
        """
        Reconnect to WebSocket with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise
        """
        current_interval = self.reconnect_interval

        while self.is_running:
            try:
                self.logger.info(f"Attempting to reconnect in {current_interval:.1f}s...")

                await asyncio.sleep(current_interval)

                # Close existing connection if any
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except Exception:
                        pass

                # Attempt reconnection
                await self.connect()

                self.logger.info("Reconnection successful")

                # Reset reconnection interval
                return True

            except Exception as e:
                self.logger.error(f"Reconnection attempt failed: {e}")

                # Exponential backoff
                current_interval = min(current_interval * 2, self.max_reconnect_interval)

        return False

    async def run_with_reconnection(self, callback):
        """
        Run WebSocket stream with automatic reconnection.

        Args:
            callback: Async function to call with each trade

        Example:
            >>> async def handle_trade(trade):
            ...     print(f"Trade: {trade.price}")
            >>>
            >>> client = BinanceWebSocketClient(symbol="BTCUSDT")
            >>> await client.run_with_reconnection(handle_trade)
        """
        self.is_running = True

        while self.is_running:
            try:
                # Connect if not connected
                if not self.is_connected:
                    await self.connect()

                # Stream trades
                async for trade in self.stream_trades():
                    await callback(trade)

            except NetworkError as e:
                self.logger.error(f"Connection error: {e}")

                # Attempt reconnection
                if not await self.reconnect_with_backoff():
                    break

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                break

        # Cleanup
        await self.disconnect()


async def create_binance_websocket_client(
    symbol: str,
    base_url: str = "wss://stream.binance.com:9443/ws",
) -> BinanceWebSocketClient:
    """
    Factory function to create and connect a BinanceWebSocketClient.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        base_url: WebSocket base URL

    Returns:
        Connected BinanceWebSocketClient

    Example:
        >>> client = await create_binance_websocket_client("BTCUSDT")
        >>> async for trade in client.stream_trades():
        ...     print(trade)
        ...     break
        >>> await client.disconnect()
    """
    client = BinanceWebSocketClient(symbol=symbol, base_url=base_url)
    await client.connect()
    return client

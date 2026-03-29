"""
KuCoin WebSocket Market Data Streaming

This module provides WebSocket streaming for real-time KuCoin market data.

KuCoin uses a token-based WebSocket system that requires:
1. REST API call to get connection token
2. WebSocket connection using the token
3. Periodic token refresh (tokens expire)

API Documentation: https://docs.kucoin.com/#websocket-market-data

Architecture:
- Token acquisition from REST API
- WebSocket connection management
- Message parsing and validation
- Automatic reconnection with token refresh
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable

import websockets

from src.execution.kucoin.client import KuCoinClient
from src.execution.kucoin.exceptions import NetworkError
from src.execution.kucoin.models import KuCoinWebSocketTrade
from src.execution.kucoin.utils import setup_logger


class KuCoinWebSocketClient:
    """
    KuCoin WebSocket client for real-time market data streaming.

    This client connects to KuCoin WebSocket streams and provides:
    - Real-time trade streaming
    - Token-based authentication
    - Automatic token refresh
    - Reconnection logic with token refresh
    - Staleness detection

    API Docs: https://docs.kucoin.com/#match-execution-data

    Attributes:
        symbol: Trading symbol to stream
        client: KuCoinClient instance
        token: WebSocket connection token
        reconnect_interval: Initial reconnection interval (seconds)
        max_reconnect_interval: Maximum reconnection interval (seconds)
        staleness_threshold: Staleness detection threshold (seconds)
        is_connected: Connection status
        is_running: Stream running status

    Example:
        >>> async with KuCoinClient() as client:
        ...     ws_client = KuCoinWebSocketClient("BTC-USDT", client)
        ...     await ws_client.connect()
        ...     async for trade in ws_client.stream_trades():
        ...         print(f"Trade: {trade.price} @ {trade.quantity}")
    """

    def __init__(
        self,
        symbol: str,
        client: KuCoinClient,
        reconnect_interval: float = 1.0,
        max_reconnect_interval: float = 60.0,
        staleness_threshold: float = 30.0,
    ) -> None:
        """
        Initialize KuCoin WebSocket client.

        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            client: KuCoinClient instance for token acquisition
            reconnect_interval: Initial reconnection interval (seconds)
            max_reconnect_interval: Maximum reconnection interval (seconds)
            staleness_threshold: Staleness detection threshold (seconds)
        """
        self.symbol = symbol.upper()
        self.client = client
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.staleness_threshold = staleness_threshold

        # Connection state
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.is_connected = False
        self.is_running = False

        # Token management
        self.token: str | None = None
        self.token_expiry: datetime | None = None

        # State tracking
        self.last_message_time: datetime | None = None
        self.last_trade_id: str | None = None
        self.gaps_detected = 0

        # Logging
        self.logger = setup_logger(f"{__name__}.KuCoinWebSocketClient")

    async def connect(self) -> None:
        """
        Connect to KuCoin WebSocket stream.

        This acquires a token and establishes WebSocket connection.

        Raises:
            NetworkError: If connection fails
        """
        try:
            # Get WebSocket token
            self.logger.info(f"Acquiring WebSocket token for {self.symbol}...")
            self.token = await self.client.get_websocket_token()

            # Construct WebSocket URL
            # For public channels, use: /endpoint?token=xxx&subscribe=xxx
            ws_url = f"{self.client.websocket_base_url}/endpoint?token={self.token}"

            # Subscribe to trade channel
            subscribe_message = {
                "id": str(int(datetime.now().timestamp() * 1000)),
                "type": "subscribe",
                "topic": f"/market/match:{self.symbol}",
                "privateChannel": False,
            }

            # Connect to WebSocket
            self.logger.info(f"Connecting to KuCoin WebSocket for {self.symbol}...")
            self.websocket = await websockets.connect(ws_url)

            # Send subscription message
            await self.websocket.send(json.dumps(subscribe_message))

            self.is_connected = True
            self.is_running = True
            self.last_message_time = datetime.now(timezone.utc)

            self.logger.info(f"Connected to KuCoin WebSocket for {self.symbol}")

        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise NetworkError(f"WebSocket connection failed: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from KuCoin WebSocket stream.
        """
        self.is_running = False

        if self.websocket:
            try:
                # Send unsubscribe message
                unsubscribe_message = {
                    "id": str(int(datetime.now().timestamp() * 1000)),
                    "type": "unsubscribe",
                    "topic": f"/market/match:{self.symbol}",
                    "privateChannel": False,
                }
                await self.websocket.send(json.dumps(unsubscribe_message))

                # Close connection
                await self.websocket.close()
                self.logger.info(f"Disconnected from KuCoin WebSocket for {self.symbol}")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

        self.is_connected = False
        self.websocket = None

    async def stream_trades(self):
        """
        Stream real-time trades from KuCoin WebSocket.

        Yields:
            KuCoinWebSocketTrade objects

        Raises:
            NetworkError: If connection fails
        """
        if not self.is_connected or not self.websocket:
            raise NetworkError("WebSocket not connected")

        try:
            async for message in self.websocket:
                if not self.is_running:
                    break

                # Parse message
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "message":
                    # Trade data
                    trade_data = data.get("data", {})
                    if trade_data.get("symbol") == self.symbol:
                        trade = KuCoinWebSocketTrade(**trade_data)

                        # Update last message time
                        self.last_message_time = datetime.now(timezone.utc)

                        yield trade

                elif msg_type == "error":
                    self.logger.error(f"WebSocket error: {data.get('data')}")
                    raise NetworkError(f"WebSocket error: {data.get('data')}")

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"WebSocket connection closed: {e}")
            self.is_connected = False
            raise NetworkError(f"WebSocket connection closed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error streaming trades: {e}")
            raise

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
        Reconnect to WebSocket with exponential backoff and token refresh.

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

                # Refresh token
                try:
                    self.token = await self.client.get_websocket_token()
                    self.logger.info("WebSocket token refreshed")
                except Exception as e:
                    self.logger.warning(f"Failed to refresh token: {e}")

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

    async def run_with_reconnection(self, callback: Callable) -> None:
        """
        Run WebSocket stream with automatic reconnection.

        Args:
            callback: Async function to call with each trade

        Example:
            >>> async def handle_trade(trade):
            ...     print(f"Trade: {trade.price}")
            >>>
            >>> client = KuCoinWebSocketClient("BTC-USDT", kucoin_client)
            >>> await client.connect()
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


async def create_kucoin_websocket_client(
    symbol: str,
    client: KuCoinClient,
) -> KuCoinWebSocketClient:
    """
    Factory function to create and connect a KuCoinWebSocketClient.

    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        client: KuCoinClient instance

    Returns:
        Connected KuCoinWebSocketClient

    Example:
        >>> async with KuCoinClient() as client:
        ...     ws_client = await create_kucoin_websocket_client("BTC-USDT", client)
        ...     async for trade in ws_client.stream_trades():
        ...         print(trade)
        ...         break
        ...     await ws_client.disconnect()
    """
    ws_client = KuCoinWebSocketClient(symbol=symbol, client=client)
    await ws_client.connect()
    return ws_client

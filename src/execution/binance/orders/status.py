"""
Binance Order Status Streaming

This module provides WebSocket streaming for order status updates.

Binance uses user data streams to push real-time order updates via WebSocket.
This requires a listen key acquired via signed REST API request.

API Documentation: https://binance-docs.github.io/apidocs/#user-data-streams

Architecture:
- Listen key acquisition and management
- WebSocket connection for user data stream
- Execution report parsing
- Keepalive mechanism (every 30 minutes)
- Position tracker updates on fills
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable

import websockets

from src.execution.binance.client import BinanceClient
from src.execution.binance.exceptions import NetworkError
from src.execution.binance.models import BinanceOrder
from src.execution.binance.utils import setup_logger


class EventType(str, Enum):
    """User data stream event types."""

    EXECUTION_REPORT = "executionReport"
    ACCOUNT_UPDATE = "accountUpdate"
    LISTEN_KEY_EXPIRED = "expired"


class ExecutionType(str, Enum):
    """Execution types for orders."""

    NEW = "NEW"
    CANCELED = "CANCELED"
    REPLACED = "REPLACED"
    REJECTED = "REJECTED"
    TRADE = "TRADE"
    EXPIRED = "EXPIRED"


class OrderStatus(str, Enum):
    """Order status values."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class BinanceOrderStatusStream:
    """
    Binance user data stream for order status updates.

    This client connects to Binance user data stream and provides:
    - Real-time order execution reports
    - Account position updates
    - Listen key management
    - Automatic reconnection
    - Position tracker updates on fills

    API Docs: https://binance-docs.github.io/apidocs/#user-data-streams

    Attributes:
        binance_client: BinanceClient instance
        listen_key: User data stream listen key
        base_url: WebSocket base URL
        is_connected: Connection status
        is_running: Stream running status
        on_order_update: Callback for order updates
        on_account_update: Callback for account updates

    Example:
        >>> async with BinanceClient() as client:
        ...     stream = BinanceOrderStatusStream(client)
        ...     await stream.connect()
        ...
        ...     async for update in stream.stream_updates():
        ...         if update["type"] == "execution":
        ...             print(f"Order update: {update}")
        ...         elif update["type"] == "account":
        ...             print(f"Account update: {update}")
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        on_order_update: Callable | None = None,
        on_account_update: Callable | None = None,
        keepalive_interval: float = 1800.0,  # 30 minutes
    ) -> None:
        """
        Initialize Binance order status stream.

        Args:
            binance_client: BinanceClient instance
            on_order_update: Callback for order updates (receives execution report)
            on_account_update: Callback for account updates (receives account update)
            keepalive_interval: Keepalive ping interval in seconds (default: 30 min)
        """
        self.binance_client = binance_client
        self.on_order_update = on_order_update
        self.on_account_update = on_account_update
        self.keepalive_interval = keepalive_interval

        # Listen key (acquired on connect)
        self.listen_key: str | None = None

        # WebSocket connection
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.base_url = binance_client.websocket_base_url

        # Connection state
        self.is_connected = False
        self.is_running = False

        # Keepalive task
        self._keepalive_task: asyncio.Task | None = None

        # Logging
        self.logger = setup_logger(f"{__name__}.BinanceOrderStatusStream")

    async def connect(self) -> None:
        """
        Connect to Binance user data stream.

        This acquires a listen key and establishes WebSocket connection.

        Raises:
            NetworkError: If connection fails
        """
        try:
            # Acquire listen key
            self.logger.info("Acquiring listen key...")
            self.listen_key = await self.binance_client.auth.acquire_listen_key(
                http_client=self.binance_client.http_client,
                base_url=self.binance_client.base_url,
            )

            self.logger.info(f"Listen key acquired: {self.listen_key[:10]}...")

            # Construct WebSocket URL
            ws_url = f"{self.base_url}/{self.listen_key}"

            # Connect to WebSocket
            self.logger.info("Connecting to user data stream...")
            self.websocket = await websockets.connect(ws_url)

            self.is_connected = True
            self.is_running = True

            # Start keepalive task
            self._start_keepalive()

            self.logger.info("Connected to Binance user data stream")

        except Exception as e:
            self.logger.error(f"Failed to connect to user data stream: {e}")
            raise NetworkError(f"User data stream connection failed: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from Binance user data stream.

        This closes the WebSocket connection and invalidates the listen key.
        """
        self.is_running = False

        # Stop keepalive task
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

        # Invalidate listen key
        if self.listen_key and self.binance_client.http_client:
            try:
                await self.binance_client.auth.close_listen_key(
                    http_client=self.binance_client.http_client,
                    base_url=self.binance_client.base_url,
                    listen_key=self.listen_key,
                )
                self.logger.info("Listen key invalidated")
            except Exception as e:
                self.logger.error(f"Failed to close listen key: {e}")

        self.is_connected = False
        self.websocket = None

    async def stream_updates(self):
        """
        Stream order and account updates from user data stream.

        Yields:
            Dictionary with update data:
            {
                "type": "execution" | "account" | "expired",
                "data": <event data>
            }

        Raises:
            NetworkError: If connection fails
        """
        if not self.is_connected or not self.websocket:
            raise NetworkError("User data stream not connected")

        try:
            async for message in self.websocket:
                if not self.is_running:
                    break

                # Parse message
                update = self._parse_message(message)

                # Call callbacks
                if update["type"] == "execution" and self.on_order_update:
                    await self._call_callback(self.on_order_update, update["data"])
                elif update["type"] == "account" and self.on_account_update:
                    await self._call_callback(self.on_account_update, update["data"])

                yield update

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"WebSocket connection closed: {e}")
            self.is_connected = False
            raise NetworkError(f"User data stream connection closed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error streaming updates: {e}")
            raise

    def _parse_message(self, message: str) -> dict:
        """
        Parse WebSocket message from user data stream.

        Args:
            message: JSON message string

        Returns:
            Dictionary with update type and data

        Raises:
            ValueError: If message is invalid
        """
        try:
            data = json.loads(message)

            event_type = data.get("e")

            if event_type == EventType.EXECUTION_REPORT:
                return {
                    "type": "execution",
                    "data": self._parse_execution_report(data),
                }
            elif event_type == EventType.ACCOUNT_UPDATE:
                return {
                    "type": "account",
                    "data": self._parse_account_update(data),
                }
            elif event_type == EventType.LISTEN_KEY_EXPIRED:
                return {
                    "type": "expired",
                    "data": {"message": "Listen key expired, need to reconnect"},
                }
            else:
                self.logger.warning(f"Unknown event type: {event_type}")
                return {
                    "type": "unknown",
                    "data": data,
                }

        except Exception as e:
            self.logger.error(f"Failed to parse user data stream message: {e}")
            raise ValueError(f"Invalid message: {message}") from e

    def _parse_execution_report(self, data: dict) -> dict:
        """
        Parse execution report from user data stream.

        API Docs: https://binance-docs.github.io/apidocs/#payload-order-update

        Args:
            data: Raw execution report data

        Returns:
            Parsed execution report dictionary
        """
        return {
            "event_type": data.get("eventType"),
            "event_time": data.get("E"),
            "symbol": data.get("s"),
            "order_id": data.get("i"),
            "client_order_id": data.get("c"),
            "side": data.get("S"),
            "order_type": data.get("o"),
            "time_in_force": data.get("f"),
            "quantity": data.get("q"),
            "price": data.get("p"),
            "stop_price": data.get("P"),
            "execution_type": data.get("x"),
            "order_status": data.get("X"),
            "reject_reason": data.get("r"),
            "last_executed_quantity": data.get("l"),
            "cumulative_filled_quantity": data.get("z"),
            "last_executed_price": data.get("L"),
            "commission_amount": data.get("n"),
            "commission_asset": data.get("N"),
            "transaction_time": data.get("T"),
            "trade_id": data.get("t"),
            "is_order_working": data.get("w"),
            "is_buyer_maker": data.get("m"),
            "creation_time": data.get("O"),
            "fill_price": data.get("F"),  # Average fill price
            "total_quote_trade_quantity": data.get("Y"),  # Total quote asset traded
        }

    def _parse_account_update(self, data: dict) -> dict:
        """
        Parse account update from user data stream.

        API Docs: https://binance-docs.github.io/apidocs/#payload-balance-update

        Args:
            data: Raw account update data

        Returns:
            Parsed account update dictionary
        """
        balances = data.get("B", [])

        return {
            "event_type": data.get("eventType"),
            "event_time": data.get("E"),
            "last_account_update": data.get("u"),
            "balances": [
                {
                    "asset": balance.get("a"),
                    "free_balance": balance.get("f"),
                    "locked_balance": balance.get("l"),
                }
                for balance in balances
            ],
        }

    async def _call_callback(self, callback: Callable, data: dict) -> None:
        """
        Call user-provided callback with data.

        Args:
            callback: Async callback function
            data: Data to pass to callback
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            self.logger.error(f"Callback error: {e}")

    def _start_keepalive(self) -> None:
        """
        Start background task to send keepalive pings.

        Listen keys expire after 24 hours. Keepalive pings extend validity.
        This sends a keepalive ping every 30 minutes.
        """
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        """
        Keepalive loop - sends keepalive ping at regular intervals.

        Listen key must be kept alive via PUT request to prevent expiration.
        """
        while self.is_running and self.listen_key:
            try:
                await asyncio.sleep(self.keepalive_interval)

                if not self.is_running or not self.listen_key:
                    break

                # Send keepalive ping
                success = await self.binance_client.auth.keepalive_listen_key(
                    http_client=self.binance_client.http_client,
                    base_url=self.binance_client.base_url,
                    listen_key=self.listen_key,
                )

                if success:
                    self.logger.debug("Keepalive ping successful")
                else:
                    self.logger.warning("Keepalive ping failed")

            except Exception as e:
                self.logger.error(f"Keepalive ping error: {e}")

    async def run_with_reconnection(self, callback: Callable | None = None) -> None:
        """
        Run user data stream with automatic reconnection.

        Args:
            callback: Optional callback function for updates

        Example:
            >>> async def handle_update(update):
            ...     if update["type"] == "execution":
            ...         print(f"Order update: {update['data']}")
            >>>
            >>> stream = BinanceOrderStatusStream(client)
            >>> await stream.connect()
            >>> await stream.run_with_reconnection(handle_update)
        """
        self.is_running = True

        while self.is_running:
            try:
                # Connect if not connected
                if not self.is_connected:
                    await self.connect()

                # Stream updates
                async for update in self.stream_updates():
                    if callback:
                        await self._call_callback(callback, update)

            except NetworkError as e:
                self.logger.error(f"Connection error: {e}")

                # Attempt reconnection after delay
                await asyncio.sleep(5)

                # Reconnect
                try:
                    await self.disconnect()
                except Exception:
                    pass

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                break

        # Cleanup
        await self.disconnect()


async def create_binance_order_status_stream(
    binance_client: BinanceClient,
    on_order_update: Callable | None = None,
    on_account_update: Callable | None = None,
) -> BinanceOrderStatusStream:
    """
    Factory function to create and connect a BinanceOrderStatusStream.

    Args:
        binance_client: BinanceClient instance
        on_order_update: Callback for order updates
        on_account_update: Callback for account updates

    Returns:
        Connected BinanceOrderStatusStream

    Example:
        >>> async with BinanceClient() as client:
        ...     stream = await create_binance_order_status_stream(client)
        ...     async for update in stream.stream_updates():
        ...         print(update)
        ...         break
        ...     await stream.disconnect()
    """
    stream = BinanceOrderStatusStream(
        binance_client=binance_client,
        on_order_update=on_order_update,
        on_account_update=on_account_update,
    )
    await stream.connect()
    return stream

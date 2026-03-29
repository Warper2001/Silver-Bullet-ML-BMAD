"""
BYDFI Order Submission Client

Handles order submission and management for BYDFI spot trading.
Includes circuit breaker pattern for resilience.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.data.bydfi_config import load_bydfi_settings
from src.execution.bydfi.client import BYDFIClient
from src.execution.bydfi.models import BYDFIOrder

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    States:
        - CLOSED: Normal operation
        - OPEN: Failing, stop requests
        - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            reset_timeout_seconds: Seconds before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_failure(self):
        """Record a failure and potentially open circuit."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.error(
                f"Circuit breaker OPEN after {self._failure_count} failures"
            )

    def record_success(self):
        """Record a success and reset circuit if needed."""
        self._failure_count = 0
        self._state = "CLOSED"
        self._last_failure_time = None

    def allow_request(self) -> bool:
        """
        Check if request should be allowed.

        Returns:
            bool: True if request allowed, False if circuit is open
        """
        if self._state == "CLOSED":
            return True

        if self._state == "OPEN":
            # Check if reset timeout has passed
            if self._last_failure_time:
                elapsed = (
                    datetime.now(timezone.utc) - self._last_failure_time
                ).total_seconds()

                if elapsed >= self.reset_timeout_seconds:
                    self._state = "HALF_OPEN"
                    logger.info("Circuit breaker HALF_OPEN")
                    return True

            return False

        if self._state == "HALF_OPEN":
            return True

        return False


class BYDFIOrderRequest(BaseModel):
    """
    BYDFI order request.

    Attributes:
        symbol: Trading symbol
        side: Order side (buy/sell)
        order_type: Order type (market/limit)
        quantity: Order quantity
        price: Order price (required for limit orders)
        client_order_id: Client order ID
    """

    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    order_type: str = Field(..., description="Order type (market/limit)")
    quantity: str = Field(..., description="Order quantity")
    price: Optional[str] = Field(None, description="Order price")
    client_order_id: str = Field(default_factory=lambda: str(uuid4()))


class BYDFIOrdersClient:
    """
    BYDFI order submission and management client.

    Features:
        - Order CRUD operations
        - Circuit breaker pattern
        - Fill timeout mechanism
        - Input validation

    Example:
        >>> client = BYDFIClient()
        >>> orders_client = BYDFIOrdersClient(client)
        >>> order = await orders_client.place_order(
        ...     symbol="BTC-USDT",
        ...     side="buy",
        ...     order_type="market",
        ...     quantity="0.001"
        ... )
    """

    def __init__(
        self,
        bydfi_client: BYDFIClient,
        default_fill_timeout: float = 5.0,
    ):
        """
        Initialize BYDFI orders client.

        Args:
            bydfi_client: BYDFI REST API client
            default_fill_timeout: Default timeout for order fills (seconds)
        """
        self.bydfi_client = bydfi_client
        self.default_fill_timeout = default_fill_timeout

        # Initialize circuit breaker
        settings = load_bydfi_settings()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout_seconds=60,
        )

        logger.info("BYDFI orders client initialized")

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: str,
        price: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> BYDFIOrder:
        """
        Place an order on BYDFI.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            quantity: Order quantity
            price: Order price (required for limit orders)
            client_order_id: Client order ID

        Returns:
            BYDFIOrder: Placed order

        Raises:
            CircuitBreakerError: If circuit breaker is open
            ValueError: If validation fails
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerError("Circuit breaker is open, rejecting request")

        # Validate order
        if order_type == "limit" and not price:
            raise ValueError("Price required for limit orders")

        # Create order request
        order_request = BYDFIOrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id or str(uuid4()),
        )

        try:
            # Build request body
            body = {
                "symbol": order_request.symbol,
                "side": order_request.side,
                "type": order_request.order_type,
                "amount": order_request.quantity,
                "clientOid": order_request.client_order_id,
            }

            if order_request.price:
                body["price"] = order_request.price

            # Build headers for signed request
            headers = self.bydfi_client._build_headers(
                method="POST",
                endpoint="/v1/spot/order",
                body=json.dumps(body),
            )

            # Submit order
            response = await self.bydfi_client._client.post(
                "/v1/spot/order",
                json=body,
                headers=headers,
            )

            data = self.bydfi_client._handle_response(response)

            # Parse order response
            order = BYDFIOrder(
                order_id=str(data.get("orderId", "")),
                client_order_id=order_request.client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                price=price or "0",
                quantity=quantity,
                filled_quantity=str(data.get("filledQty", "0")),
                status=str(data.get("status", "pending")),
                timestamp=datetime.now(timezone.utc),
            )

            # Record success
            self.circuit_breaker.record_success()
            logger.info(f"Order placed: {order.order_id} ({side} {quantity} {symbol})")

            return order

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            logger.error(f"Order placement failed: {e}")
            raise

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            bool: True if cancellation successful
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerError("Circuit breaker is open, rejecting request")

        try:
            # Build request body
            body = {"symbol": symbol, "orderId": order_id}

            # Build headers
            headers = self.bydfi_client._build_headers(
                method="DELETE",
                endpoint="/v1/spot/order",
                body=json.dumps(body),
            )

            # Cancel order
            response = await self.bydfi_client._client.delete(
                "/v1/spot/order",
                json=body,
                headers=headers,
            )

            self.bydfi_client._handle_response(response)

            # Record success
            self.circuit_breaker.record_success()
            logger.info(f"Order cancelled: {order_id}")

            return True

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            logger.error(f"Order cancellation failed: {e}")
            raise

    async def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            int: Number of orders cancelled
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerError("Circuit breaker is open, rejecting request")

        try:
            # Build request body
            body = {"symbol": symbol}

            # Build headers
            headers = self.bydfi_client._build_headers(
                method="DELETE",
                endpoint="/v1/spot/orders",
                body=json.dumps(body),
            )

            # Cancel all orders
            response = await self.bydfi_client._client.delete(
                "/v1/spot/orders",
                json=body,
                headers=headers,
            )

            data = self.bydfi_client._handle_response(response)

            # Record success
            self.circuit_breaker.record_success()
            logger.info(f"All orders cancelled for {symbol}")

            return len(data) if isinstance(data, list) else 0

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            logger.error(f"Cancel all orders failed: {e}")
            raise

    async def get_open_orders(self, symbol: str) -> list[BYDFIOrder]:
        """
        Get all open orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            list[BYDFIOrder]: List of open orders
        """
        # Build query string
        query_params = {"symbol": symbol}
        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

        # Build headers
        headers = self.bydfi_client._build_headers(
            method="GET",
            endpoint="/v1/spot/orders",
            query_string=query_string,
        )

        # Get open orders
        response = await self.bydfi_client._client.get(
            "/v1/spot/orders",
            params=query_params,
            headers=headers,
        )

        data = self.bydfi_client._handle_response(response)

        # Parse orders
        orders = []
        for item in data if isinstance(data, list) else []:
            orders.append(
                BYDFIOrder(
                    order_id=str(item.get("orderId", "")),
                    client_order_id=str(item.get("clientOid", "")),
                    symbol=symbol,
                    side=str(item.get("side", "")),
                    order_type=str(item.get("type", "")),
                    price=str(item.get("price", "0")),
                    quantity=str(item.get("amount", "0")),
                    filled_quantity=str(item.get("filledQty", "0")),
                    status=str(item.get("status", "pending")),
                    timestamp=datetime.fromtimestamp(
                        int(item.get("createTime", 0)) / 1000,
                        tz=timezone.utc,
                    ),
                )
            )

        return orders


async def create_bydfi_orders_client(
    bydfi_client: BYDFIClient,
) -> BYDFIOrdersClient:
    """
    Factory function to create BYDFI orders client.

    Args:
        bydfi_client: BYDFI REST API client

    Returns:
        BYDFIOrdersClient: Configured orders client
    """
    return BYDFIOrdersClient(bydfi_client=bydfi_client)

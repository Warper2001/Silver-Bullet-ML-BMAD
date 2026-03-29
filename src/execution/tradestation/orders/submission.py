"""
TradeStation SDK - Order Submission Module

This module provides order CRUD operations for the TradeStation API.

Key Features:
- Place market, limit, stop, and stop-limit orders
- Modify existing orders (price, quantity)
- Cancel individual orders
- Cancel all orders for an account
- Input validation and error handling
- Integration with trade execution pipeline

Usage:
    async with TradeStationClient(env="sim", ...) as client:
        orders_client = OrdersClient(client)

        # Place a limit order
        order = await orders_client.place_order(
            symbol="MNQH26",
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=15000.0
        )

        # Modify the order
        modified = await orders_client.modify_order(
            order_id=order.order_id,
            price=15001.0
        )

        # Cancel the order
        await orders_client.cancel_order(order.order_id)

Integration with Trade Execution Pipeline:
    The OrdersClient integrates with trade_execution_pipeline.py to route
    signals through risk validation to SIM order execution. All orders are
    submitted to the SIM environment (env="sim") for paper trading.
"""

import logging
from typing import Literal

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import APIError, NetworkError, ValidationError
from src.execution.tradestation.models import NewOrderRequest, TradeStationOrder
from src.execution.tradestation.utils import setup_logger


class OrdersClient:
    """
    Client for order management operations with TradeStation API.

    Provides methods to place, modify, and cancel orders with proper
    validation and error handling.

    Attributes:
        client: TradeStationClient instance for API communication

    Example:
        async with TradeStationClient(env="sim", ...) as client:
            orders_client = OrdersClient(client)

            # Place a market order
            order = await orders_client.place_order(
                symbol="MNQH26",
                side="Buy",
                order_type="Market",
                quantity=1
            )

            # Place a limit order
            order = await orders_client.place_order(
                symbol="MNQH26",
                side="Buy",
                order_type="Limit",
                quantity=1,
                price=15000.0,
                time_in_force="Day"
            )

            # Modify order price
            await orders_client.modify_order(
                order_id=order.order_id,
                price=15001.0
            )

            # Cancel order
            await orders_client.cancel_order(order.order_id)

            # Cancel all orders
            await orders_client.cancel_all_orders()
    """

    # Valid order types
    VALID_ORDER_TYPES = ["Market", "Limit", "Stop", "StopLimit", "MarketOnClose"]

    # Valid order sides
    VALID_SIDES = ["Buy", "Sell"]

    # Valid time in force values
    VALID_TIF = ["Day", "GTC", "IOC", "FOK"]

    def __init__(self, client: TradeStationClient) -> None:
        """
        Initialize OrdersClient.

        Args:
            client: Authenticated TradeStationClient instance
        """
        self.client = client
        self.logger = setup_logger(f"{__name__}.OrdersClient")

    async def place_order(
        self,
        symbol: str,
        side: Literal["Buy", "Sell"],
        order_type: Literal["Market", "Limit", "Stop", "StopLimit", "MarketOnClose"],
        quantity: int,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: Literal["Day", "GTC", "IOC", "FOK"] = "Day",
    ) -> TradeStationOrder:
        """
        Place a new order.

        Args:
            symbol: Trading symbol (e.g., "MNQH26")
            side: Order side ("Buy" or "Sell")
            order_type: Order type ("Market", "Limit", "Stop", "StopLimit", "MarketOnClose")
            quantity: Number of contracts (must be > 0)
            price: Limit price (required for Limit and StopLimit orders)
            stop_price: Stop price (required for Stop and StopLimit orders)
            time_in_force: Order duration ("Day", "GTC", "IOC", "FOK")

        Returns:
            TradeStationOrder object with order details

        Raises:
            ValidationError: If parameters are invalid
            APIError: On API errors
            NetworkError: On network errors

        Example:
            # Market order
            order = await orders_client.place_order(
                symbol="MNQH26",
                side="Buy",
                order_type="Market",
                quantity=1
            )

            # Limit order
            order = await orders_client.place_order(
                symbol="MNQH26",
                side="Buy",
                order_type="Limit",
                quantity=1,
                price=15000.0
            )

            # Stop limit order
            order = await orders_client.place_order(
                symbol="MNQH26",
                side="Sell",
                order_type="StopLimit",
                quantity=1,
                stop_price=14995.0,
                price=14994.0
            )
        """
        self.logger.info(f"Placing {side} {order_type} order for {quantity} {symbol}")

        # Validate parameters
        self._validate_order_parameters(symbol, side, order_type, quantity, price, stop_price, time_in_force)

        # Build order request
        order_request = NewOrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

        try:
            # Convert to API format (camelCase)
            payload = {
                "Symbol": order_request.symbol,
                "Side": order_request.side,
                "OrderType": order_request.order_type,
                "Quantity": order_request.quantity,
                "TimeInForce": order_request.time_in_force,
            }

            if order_request.price is not None:
                payload["Price"] = order_request.price

            if order_request.stop_price is not None:
                payload["StopPrice"] = order_request.stop_price

            # Submit order
            response = await self.client._request(
                "POST",
                "/order",
                json=payload,
            )

            # Parse response
            order_data = response.get("Order", {})
            order = TradeStationOrder(**order_data)

            self.logger.info(f"Order placed successfully: {order.order_id} ({order.status})")
            return order

        except (ValidationError, APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error placing order: {e}")
            raise APIError(f"Unexpected error placing order: {e}")

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> TradeStationOrder:
        """
        Modify an existing order.

        Note: Not all fields can be modified. Check TradeStation API documentation
        for specific limitations. Generally, you can modify price and quantity.

        Args:
            order_id: Order ID to modify
            quantity: New quantity (None = no change)
            price: New limit price (None = no change)
            stop_price: New stop price (None = no change)

        Returns:
            Updated TradeStationOrder object

        Raises:
            ValidationError: If order_id is invalid or no changes specified
            APIError: On API errors (e.g., order already filled)
            NetworkError: On network errors

        Example:
            # Modify price
            order = await orders_client.modify_order(
                order_id="12345",
                price=15001.0
            )

            # Modify quantity
            order = await orders_client.modify_order(
                order_id="12345",
                quantity=2
            )

            # Modify both
            order = await orders_client.modify_order(
                order_id="12345",
                quantity=2,
                price=15001.0
            )
        """
        if not order_id:
            raise ValidationError("Order ID cannot be empty")

        if quantity is None and price is None and stop_price is None:
            raise ValidationError("At least one parameter (quantity, price, or stop_price) must be specified")

        self.logger.info(f"Modifying order {order_id}")

        try:
            # Build modification payload
            payload = {"OrderID": order_id}

            if quantity is not None:
                if quantity <= 0:
                    raise ValidationError("Quantity must be positive")
                payload["Quantity"] = quantity

            if price is not None:
                if price <= 0:
                    raise ValidationError("Price must be positive")
                payload["Price"] = price

            if stop_price is not None:
                if stop_price <= 0:
                    raise ValidationError("Stop price must be positive")
                payload["StopPrice"] = stop_price

            # Submit modification
            response = await self.client._request(
                "PUT",
                f"/order/{order_id}",
                json=payload,
            )

            # Parse response
            order_data = response.get("Order", {})
            order = TradeStationOrder(**order_data)

            self.logger.info(f"Order modified successfully: {order_id} ({order.status})")
            return order

        except (ValidationError, APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error modifying order: {e}")
            raise APIError(f"Unexpected error modifying order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation was successful

        Raises:
            ValidationError: If order_id is invalid
            APIError: On API errors (e.g., order already filled)
            NetworkError: On network errors

        Example:
            success = await orders_client.cancel_order("12345")
            if success:
                print("Order cancelled successfully")
        """
        if not order_id:
            raise ValidationError("Order ID cannot be empty")

        self.logger.info(f"Cancelling order {order_id}")

        try:
            # Submit cancellation
            response = await self.client._request(
                "DELETE",
                f"/order/{order_id}",
            )

            # Check response
            cancelled = response.get("Cancelled", False)

            if cancelled:
                self.logger.info(f"Order cancelled successfully: {order_id}")
            else:
                self.logger.warning(f"Order cancellation may have failed: {order_id}")

            return cancelled

        except (ValidationError, APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error cancelling order: {e}")
            raise APIError(f"Unexpected error cancelling order: {e}")

    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders for the account.

        This is a convenience method that cancels all working orders.
        Use with caution in production.

        Returns:
            Number of orders cancelled

        Raises:
            APIError: On API errors
            NetworkError: On network errors

        Example:
            count = await orders_client.cancel_all_orders()
            print(f"Cancelled {count} orders")
        """
        self.logger.warning("Cancelling all orders for account")

        try:
            # Submit cancellation
            response = await self.client._request(
                "DELETE",
                "/orders",
            )

            # Get cancelled count
            cancelled_count = response.get("CancelledCount", 0)

            self.logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count

        except (APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error cancelling all orders: {e}")
            raise APIError(f"Unexpected error cancelling all orders: {e}")

    async def get_order_status(self, order_id: str) -> TradeStationOrder:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to query

        Returns:
            TradeStationOrder with current status

        Raises:
            ValidationError: If order_id is invalid
            APIError: On API errors
            NetworkError: On network errors

        Example:
            order = await orders_client.get_order_status("12345")
            print(f"Order status: {order.status}")
            print(f"Filled: {order.filled_quantity}/{order.quantity}")
        """
        if not order_id:
            raise ValidationError("Order ID cannot be empty")

        self.logger.debug(f"Fetching order status: {order_id}")

        try:
            # Query order
            response = await self.client._request(
                "GET",
                f"/order/{order_id}",
            )

            # Parse response
            order_data = response.get("Order", {})
            order = TradeStationOrder(**order_data)

            return order

        except (ValidationError, APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching order status: {e}")
            raise APIError(f"Unexpected error fetching order status: {e}")

    def _validate_order_parameters(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: float | None,
        stop_price: float | None,
        time_in_force: str,
    ) -> None:
        """
        Validate order parameters.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Limit price
            stop_price: Stop price
            time_in_force: Time in force

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate symbol
        if not symbol or len(symbol) < 2:
            raise ValidationError(f"Invalid symbol: {symbol}")

        # Validate side
        if side not in self.VALID_SIDES:
            raise ValidationError(
                f"Invalid side: {side}. Must be one of: {', '.join(self.VALID_SIDES)}"
            )

        # Validate order type
        if order_type not in self.VALID_ORDER_TYPES:
            raise ValidationError(
                f"Invalid order_type: {order_type}. Must be one of: {', '.join(self.VALID_ORDER_TYPES)}"
            )

        # Validate quantity
        if quantity <= 0:
            raise ValidationError(f"Quantity must be positive: {quantity}")

        # Validate time in force
        if time_in_force not in self.VALID_TIF:
            raise ValidationError(
                f"Invalid time_in_force: {time_in_force}. Must be one of: {', '.join(self.VALID_TIF)}"
            )

        # Validate price for limit orders
        if order_type in ("Limit", "StopLimit") and price is None:
            raise ValidationError(f"Price is required for {order_type} orders")

        if price is not None and price <= 0:
            raise ValidationError(f"Price must be positive: {price}")

        # Validate stop price for stop orders
        if order_type in ("Stop", "StopLimit") and stop_price is None:
            raise ValidationError(f"Stop price is required for {order_type} orders")

        if stop_price is not None and stop_price <= 0:
            raise ValidationError(f"Stop price must be positive: {stop_price}")

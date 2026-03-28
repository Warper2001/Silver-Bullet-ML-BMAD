"""
TradeStation SDK - Order Management Package

This package provides order management functionality for the TradeStation API.

Components:
- OrdersClient: Order CRUD operations (place, modify, cancel)
- OrderStatusStream: Real-time order status streaming

Usage:
    from src.execution.tradestation.orders import OrdersClient, OrderStatusStream

    async with TradeStationClient(env="sim", ...) as client:
        orders_client = OrdersClient(client)

        # Place an order
        order = await orders_client.place_order(
            symbol="MNQH26",
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=15000.0
        )

        # Stream order status
        stream = OrderStatusStream(client)
        async for status in stream.stream_order_status([order.order_id]):
            print(f"Order {status.order_id}: {status.status}")
"""

from src.execution.tradestation.orders.submission import OrdersClient
from src.execution.tradestation.orders.status import OrderStatusStream

__all__ = [
    "OrdersClient",
    "OrderStatusStream",
]

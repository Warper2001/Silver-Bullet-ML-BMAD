"""Integration tests for TradeStation order execution lifecycle.

These tests verify the complete order lifecycle with the actual TradeStation SIM API.

⚠️  WARNING: These tests place REAL orders in the SIM environment.
    - No real money is at risk (SIM environment)
    - Orders will be cancelled after testing
    - Ensure account has sufficient buying power

Prerequisites:
- TRADESTATION_SIM_CLIENT_ID environment variable
- TRADESTATION_SIM_CLIENT_SECRET environment variable
- SIM account with sufficient buying power
- Market must be open for order fills (or use limit orders away from market)

Safety Features:
- Small position sizes (1 contract)
- Limit orders placed away from market
- Automatic cleanup after tests
- All orders cancelled on test completion
"""

import asyncio
import time
from datetime import datetime, timezone

import pytest

from src.execution.tradestation.orders.submission import OrdersClient
from src.execution.tradestation.orders.status import OrderStatusStream
from src.execution.tradestation.models import TradeStationOrder


@pytest.mark.integration
class TestOrderPlacementIntegration:
    """Integration tests for order placement."""

    @pytest.mark.asyncio
    async def test_place_limit_order_buy(self, tradestation_client, test_symbol):
        """Test placing a buy limit order."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote to determine price
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place limit order well below market (won't fill immediately)
        limit_price = round(current_price * 0.95, 2)  # 5% below market

        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=limit_price,
            time_in_force="Day",
        )

        # Verify order
        assert order is not None
        assert order.order_id is not None
        assert order.symbol == test_symbol
        assert order.side == "Buy"
        assert order.order_type == "Limit"
        assert order.quantity == 1
        assert order.price == limit_price
        assert order.status in ["Working", "Submitted", "Pending"]

        # Cleanup: cancel the order
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Buy limit order placed and cancelled")
        print(f"   Order ID: {order.order_id}")
        print(f"   Limit price: {limit_price} (market: {current_price})")

    @pytest.mark.asyncio
    async def test_place_limit_order_sell(self, tradestation_client, test_symbol):
        """Test placing a sell limit order."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place limit order well above market (won't fill immediately)
        limit_price = round(current_price * 1.05, 2)  # 5% above market

        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Sell",
            order_type="Limit",
            quantity=1,
            price=limit_price,
            time_in_force="Day",
        )

        # Verify
        assert order.side == "Sell"
        assert order.price == limit_price

        # Cleanup
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Sell limit order placed and cancelled")
        print(f"   Order ID: {order.order_id}")

    @pytest.mark.asyncio
    async def test_place_market_order_buy(self, tradestation_client, test_symbol):
        """Test placing a buy market order (will fill if market is open).

        WARNING: This order will fill immediately if market is open!
        """
        orders_client = OrdersClient(tradestation_client)

        # Place small market order
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Market",
            quantity=1,
            time_in_force="IOC",  # Immediate or Cancel (safer)
        )

        # Verify order placed
        assert order is not None
        assert order.order_id is not None
        assert order.order_type == "Market"

        # Wait a moment for potential fill
        await asyncio.sleep(1.0)

        # Check order status
        status = await orders_client.get_order_status(order.order_id)

        print(f"✅ Market buy order placed")
        print(f"   Order ID: {order.order_id}")
        print(f"   Final status: {status.status}")
        print(f"   Filled: {status.filled_quantity}/{status.quantity}")

        # If filled, we now have a position - note this for cleanup
        # (In real testing, you'd track positions and flatten them)

    @pytest.mark.asyncio
    async def test_place_stop_limit_order(self, tradestation_client, test_symbol):
        """Test placing a stop-limit order."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place stop-limit order (stop below current, limit even lower)
        stop_price = round(current_price * 0.98, 2)  # 2% below market
        limit_price = round(current_price * 0.97, 2)  # 3% below market

        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Sell",  # Sell stop (stop loss)
            order_type="StopLimit",
            quantity=1,
            stop_price=stop_price,
            price=limit_price,
            time_in_force="Day",
        )

        # Verify
        assert order.order_type == "StopLimit"
        assert order.stop_price == stop_price
        assert order.price == limit_price

        # Cleanup
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Stop-limit order placed and cancelled")
        print(f"   Order ID: {order.order_id}")


@pytest.mark.integration
class TestOrderModificationIntegration:
    """Integration tests for order modification."""

    @pytest.mark.asyncio
    async def test_modify_order_price(self, tradestation_client, test_symbol):
        """Test modifying order price."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place initial limit order
        initial_price = round(current_price * 0.95, 2)
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=initial_price,
        )

        # Modify price
        new_price = round(initial_price * 0.99, 2)  # Slightly higher
        modified_order = await orders_client.modify_order(
            order_id=order.order_id,
            price=new_price,
        )

        # Verify modification
        assert modified_order.order_id == order.order_id
        assert modified_order.price == new_price

        # Cleanup
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Order price modified")
        print(f"   Order ID: {order.order_id}")
        print(f"   Old price: {initial_price}, New price: {new_price}")

    @pytest.mark.asyncio
    async def test_modify_order_quantity(self, tradestation_client, test_symbol):
        """Test modifying order quantity."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place initial order
        initial_price = round(current_price * 0.95, 2)
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=initial_price,
        )

        # Modify quantity
        modified_order = await orders_client.modify_order(
            order_id=order.order_id,
            quantity=2,
        )

        # Verify
        assert modified_order.quantity == 2

        # Cleanup
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Order quantity modified")
        print(f"   Order ID: {order.order_id}")


@pytest.mark.integration
class TestOrderCancellationIntegration:
    """Integration tests for order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_single_order(self, tradestation_client, test_symbol):
        """Test cancelling a single order."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place order
        limit_price = round(current_price * 0.95, 2)
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=limit_price,
        )

        # Cancel order
        cancelled = await orders_client.cancel_order(order.order_id)

        # Verify cancellation
        assert cancelled is True

        # Check order status
        status = await orders_client.get_order_status(order.order_id)
        assert status.status in ["Cancelled", "Canceled", "Rejected"]

        print(f"✅ Order cancelled successfully")
        print(f"   Order ID: {order.order_id}")
        print(f"   Final status: {status.status}")

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, tradestation_client, test_symbol):
        """Test cancelling all open orders."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place multiple orders
        limit_price = round(current_price * 0.95, 2)
        orders = []
        for i in range(3):
            order = await orders_client.place_order(
                symbol=test_symbol,
                side="Buy",
                order_type="Limit",
                quantity=1,
                price=limit_price - (i * 0.25),  # Different prices
            )
            orders.append(order)

        # Cancel all orders
        cancelled_count = await orders_client.cancel_all_orders()

        # Verify
        assert cancelled_count >= 3  # At least our 3 orders

        print(f"✅ Cancelled {cancelled_count} orders")


@pytest.mark.integration
class TestOrderStatusStreamingIntegration:
    """Integration tests for order status streaming."""

    @pytest.mark.asyncio
    async def test_stream_order_status(self, tradestation_client, test_symbol):
        """Test streaming order status updates."""
        orders_client = OrdersClient(tradestation_client)
        status_stream = OrderStatusStream(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Place order
        limit_price = round(current_price * 0.95, 2)
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=limit_price,
        )

        # Stream status updates
        status_updates = []
        stream_duration = 3.0

        start_time = time.time()
        try:
            async for status in status_stream.stream_order_status([order.order_id]):
                status_updates.append(status)

                # Stop after duration
                if time.time() - start_time >= stream_duration:
                    status_stream.stop_streaming()
                    break

                if len(status_updates) >= 10:
                    status_stream.stop_streaming()
                    break
        except Exception as e:
            print(f"⚠️  Streaming error: {e}")

        # Cleanup
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Order status streaming test completed")
        print(f"   Status updates received: {len(status_updates)}")


@pytest.mark.integration
class TestOrderLatency:
    """Performance tests for order operations."""

    @pytest.mark.asyncio
    async def test_order_placement_latency(self, tradestation_client, test_symbol):
        """Test order placement latency."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last
        limit_price = round(current_price * 0.95, 2)

        # Measure placement time
        start_time = time.time()
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=limit_price,
        )
        placement_time = time.time() - start_time

        # Should be < 200ms per NFR
        assert placement_time < 0.2, f"Order placement too slow: {placement_time*1000:.1f}ms"

        # Cleanup
        await orders_client.cancel_order(order.order_id)

        print(f"✅ Order placement latency: {placement_time*1000:.1f}ms")
        print(f"   Target: < 200ms (NFR requirement)")


@pytest.mark.integration
class TestOrderLifecycleComplete:
    """Complete order lifecycle tests."""

    @pytest.mark.asyncio
    async def test_complete_order_lifecycle(self, tradestation_client, test_symbol):
        """Test complete order lifecycle: place → modify → cancel."""
        orders_client = OrdersClient(tradestation_client)

        # Get current quote
        from src.execution.tradestation.market_data.quotes import QuotesClient
        quotes_client = QuotesClient(tradestation_client)
        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) == 0 or quotes[0].last is None:
            pytest.skip("Market data unavailable")

        current_price = quotes[0].last

        # Step 1: Place order
        initial_price = round(current_price * 0.95, 2)
        order = await orders_client.place_order(
            symbol=test_symbol,
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=initial_price,
        )
        assert order.status in ["Working", "Submitted", "Pending"]

        # Step 2: Modify order
        new_price = round(initial_price * 0.99, 2)
        modified = await orders_client.modify_order(
            order_id=order.order_id,
            price=new_price,
        )
        assert modified.price == new_price

        # Step 3: Check status
        status = await orders_client.get_order_status(order.order_id)
        assert status.order_id == order.order_id

        # Step 4: Cancel order
        cancelled = await orders_client.cancel_order(order.order_id)
        assert cancelled is True

        # Step 5: Verify cancellation
        final_status = await orders_client.get_order_status(order.order_id)
        assert final_status.status in ["Cancelled", "Canceled"]

        print(f"✅ Complete order lifecycle test passed")
        print(f"   Order ID: {order.order_id}")
        print(f"   Status flow: Placed → Modified → Cancelled")


# Test Cleanup
@pytest.fixture(scope="function", autouse=True)
async def cleanup_orders(tradestation_client):
    """Cleanup any remaining orders after each test."""
    yield

    # After test, cancel any remaining orders
    try:
        orders_client = OrdersClient(tradestation_client)
        cancelled_count = await orders_client.cancel_all_orders()
        if cancelled_count > 0:
            print(f"\n🧹 Cleaned up {cancelled_count} remaining orders")
    except Exception as e:
        print(f"\n⚠️  Cleanup failed: {e}")

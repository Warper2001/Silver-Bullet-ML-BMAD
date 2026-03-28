"""Unit tests for OrdersClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import ValidationError, APIError
from src.execution.tradestation.orders.submission import OrdersClient
from src.execution.tradestation.models import TradeStationOrder


class TestOrdersClient:
    """Test suite for OrdersClient class."""

    @pytest.fixture
    def mock_client(self) -> TradeStationClient:
        """Create a mock TradeStationClient."""
        client = MagicMock(spec=TradeStationClient)
        client.api_base_url = "https://sim-api.tradestation.com/v3"
        return client

    @pytest.fixture
    def orders_client(self, mock_client: MagicMock) -> OrdersClient:
        """Create an OrdersClient with mock TradeStationClient."""
        return OrdersClient(mock_client)

    @pytest.fixture
    def sample_order_response(self) -> dict:
        """Sample order response from API."""
        return {
            "Order": {
                "OrderID": "order123",
                "Symbol": "MNQH26",
                "OrderType": "Limit",
                "Side": "Buy",
                "Quantity": 1,
                "Price": 15000.0,
                "Status": "Working",
                "FilledQuantity": 0,
                "AvgFillPrice": None,
                "TimeStamp": "2026-03-28T12:00:00Z",
            }
        }

    def test_initialization(self, orders_client: OrdersClient) -> None:
        """Test OrdersClient initialization."""
        assert orders_client.client is not None
        assert orders_client.logger is not None
        assert "Market" in orders_client.VALID_ORDER_TYPES
        assert "Buy" in orders_client.VALID_SIDES

    def test_validate_order_parameters_valid(self, orders_client: OrdersClient) -> None:
        """Test validation of valid order parameters."""
        # Should not raise
        orders_client._validate_order_parameters(
            symbol="MNQH26",
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=15000.0,
            stop_price=None,
            time_in_force="Day",
        )

    def test_validate_order_parameters_invalid_symbol(self, orders_client: OrdersClient) -> None:
        """Test validation of invalid symbol."""
        with pytest.raises(ValidationError, match="Invalid symbol"):
            orders_client._validate_order_parameters(
                symbol="",
                side="Buy",
                order_type="Market",
                quantity=1,
                price=None,
                stop_price=None,
                time_in_force="Day",
            )

    def test_validate_order_parameters_invalid_side(self, orders_client: OrdersClient) -> None:
        """Test validation of invalid side."""
        with pytest.raises(ValidationError, match="Invalid side"):
            orders_client._validate_order_parameters(
                symbol="MNQH26",
                side="Invalid",
                order_type="Market",
                quantity=1,
                price=None,
                stop_price=None,
                time_in_force="Day",
            )

    def test_validate_order_parameters_invalid_quantity(self, orders_client: OrdersClient) -> None:
        """Test validation of invalid quantity."""
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            orders_client._validate_order_parameters(
                symbol="MNQH26",
                side="Buy",
                order_type="Market",
                quantity=0,
                price=None,
                stop_price=None,
                time_in_force="Day",
            )

    def test_validate_order_parameters_limit_without_price(self, orders_client: OrdersClient) -> None:
        """Test validation of limit order without price."""
        with pytest.raises(ValidationError, match="Price is required"):
            orders_client._validate_order_parameters(
                symbol="MNQH26",
                side="Buy",
                order_type="Limit",
                quantity=1,
                price=None,
                stop_price=None,
                time_in_force="Day",
            )

    def test_validate_order_parameters_stop_without_stop_price(self, orders_client: OrdersClient) -> None:
        """Test validation of stop order without stop price."""
        with pytest.raises(ValidationError, match="Stop price is required"):
            orders_client._validate_order_parameters(
                symbol="MNQH26",
                side="Buy",
                order_type="Stop",
                quantity=1,
                price=None,
                stop_price=None,
                time_in_force="Day",
            )

    @pytest.mark.asyncio
    async def test_place_order_market(
        self, orders_client: OrdersClient, sample_order_response: dict
    ) -> None:
        """Test placing a market order."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value=sample_order_response)

        # Place market order
        order = await orders_client.place_order(
            symbol="MNQH26",
            side="Buy",
            order_type="Market",
            quantity=1,
        )

        # Verify
        assert order.order_id == "order123"
        assert order.symbol == "MNQH26"
        assert order.side == "Buy"
        assert order.quantity == 1
        assert order.status == "Working"

    @pytest.mark.asyncio
    async def test_place_order_limit(
        self, orders_client: OrdersClient, sample_order_response: dict
    ) -> None:
        """Test placing a limit order."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value=sample_order_response)

        # Place limit order
        order = await orders_client.place_order(
            symbol="MNQH26",
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=15000.0,
        )

        # Verify
        assert order.order_id == "order123"
        assert order.order_type == "Limit"

    @pytest.mark.asyncio
    async def test_place_order_stop_limit(
        self, orders_client: OrdersClient, sample_order_response: dict
    ) -> None:
        """Test placing a stop-limit order."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value=sample_order_response)

        # Place stop-limit order
        order = await orders_client.place_order(
            symbol="MNQH26",
            side="Sell",
            order_type="StopLimit",
            quantity=1,
            stop_price=14995.0,
            price=14994.0,
        )

        # Verify
        assert order.order_id == "order123"

    @pytest.mark.asyncio
    async def test_modify_order(
        self, orders_client: OrdersClient, sample_order_response: dict
    ) -> None:
        """Test modifying an order."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value=sample_order_response)

        # Modify order
        order = await orders_client.modify_order(
            order_id="order123",
            price=15001.0,
        )

        # Verify
        assert order.order_id == "order123"

    @pytest.mark.asyncio
    async def test_modify_order_no_changes(self, orders_client: OrdersClient) -> None:
        """Test modifying an order with no changes specified."""
        with pytest.raises(ValidationError, match="At least one parameter"):
            await orders_client.modify_order(order_id="order123")

    @pytest.mark.asyncio
    async def test_cancel_order(
        self, orders_client: OrdersClient
    ) -> None:
        """Test cancelling an order."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value={"Cancelled": True})

        # Cancel order
        cancelled = await orders_client.cancel_order("order123")

        # Verify
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_order_empty_id(self, orders_client: OrdersClient) -> None:
        """Test cancelling with empty order ID."""
        with pytest.raises(ValidationError, match="Order ID cannot be empty"):
            await orders_client.cancel_order("")

    @pytest.mark.asyncio
    async def test_cancel_all_orders(
        self, orders_client: OrdersClient
    ) -> None:
        """Test cancelling all orders."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value={"CancelledCount": 5})

        # Cancel all orders
        count = await orders_client.cancel_all_orders()

        # Verify
        assert count == 5

    @pytest.mark.asyncio
    async def test_get_order_status(
        self, orders_client: OrdersClient, sample_order_response: dict
    ) -> None:
        """Test getting order status."""
        # Mock the _request method
        orders_client.client._request = AsyncMock(return_value=sample_order_response)

        # Get order status
        order = await orders_client.get_order_status("order123")

        # Verify
        assert order.order_id == "order123"
        assert order.status == "Working"

    @pytest.mark.asyncio
    async def test_get_order_status_empty_id(self, orders_client: OrdersClient) -> None:
        """Test getting status with empty order ID."""
        with pytest.raises(ValidationError, match="Order ID cannot be empty"):
            await orders_client.get_order_status("")

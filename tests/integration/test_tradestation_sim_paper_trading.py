"""End-to-end integration tests for TradeStation SIM paper trading.

Tests the complete flow: data ingestion → ML inference → order execution → P&L tracking.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.models import SilverBulletSetup
from src.execution.trade_execution_pipeline import TradingSignal
from src.execution.position_tracker_pnl import PositionWithPnL, PositionTrackerWithPnL
from src.execution.tradestation.market_data.streaming import StreamPosition
from src.execution.tradestation.orders.submission import OrderResult, SIMOrderSubmitter


@pytest.fixture
def mock_auth():
    """Create mock TradeStation authentication."""
    auth = MagicMock()
    auth.get_valid_access_token = MagicMock(return_value="test_token")
    return auth


@pytest.fixture
def sample_stream_position():
    """Create sample stream position for testing."""
    return StreamPosition(
        symbol="MNQH26",
        timestamp=datetime.now(timezone.utc),
        last_price=11850.0,
        bid=11849.75,
        ask=11850.25,
        volume=1000,
    )


@pytest.fixture
def sample_trading_signal():
    """Create sample trading signal for testing."""
    return TradingSignal(
        signal_id="SIG-001",
        symbol="MNQH26",
        direction="bullish",
        confidence_score=0.75,
        timestamp=datetime.now(timezone.utc),
        entry_price=11850.0,
        patterns=["MSS", "FVG", "LIQUIDITY_SWEEP"],
        prediction={"probability": 0.75},
    )


# Default quantity for orders (used in tests)
DEFAULT_ORDER_QUANTITY = 5


class TestPositionWithPnL:
    """Tests for PositionWithPnL dataclass."""

    def test_position_creation(self):
        """Test creating a position with P&L tracking."""
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        assert position.order_id == "ORDER-001"
        assert position.symbol == "MNQH26"
        assert position.entry_price == 11850.0
        assert position.quantity == 5
        assert position.filled_quantity == 5
        assert position.current_price == 11850.0
        assert position.unrealized_pnl == 0.0

    def test_mark_to_market_long_profit(self):
        """Test mark-to-market calculation for profitable long position."""
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        # Price goes up
        position.update_mark_to_market(11860.0)

        # Calculate expected P&L: (11860 - 11850) * 5 * 0.5 = $25
        assert position.unrealized_pnl == 25.0
        assert position.current_price == 11860.0

    def test_mark_to_market_long_loss(self):
        """Test mark-to-market calculation for losing long position."""
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        # Price goes down
        position.update_mark_to_market(11840.0)

        # Calculate expected P&L: (11840 - 11850) * 5 * 0.5 = -$25
        assert position.unrealized_pnl == -25.0

    def test_mark_to_market_short_profit(self):
        """Test mark-to-market calculation for profitable short position."""
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bearish",
            timestamp=datetime.now(timezone.utc),
        )

        # Price goes down (profit for short)
        position.update_mark_to_market(11840.0)

        # Calculate expected P&L: -(11840 - 11850) * 5 * 0.5 = $25
        assert position.unrealized_pnl == 25.0

    def test_close_position(self):
        """Test closing a position and calculating realized P&L."""
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        # Update to new price
        position.update_mark_to_market(11860.0)

        # Close position
        realized_pnl = position.close_position(11860.0)

        assert position.status == "CLOSED"
        assert realized_pnl == 25.0


class TestPositionTrackerWithPnL:
    """Tests for PositionTrackerWithPnL class."""

    def test_add_position(self):
        """Test adding a position to tracking."""
        tracker = PositionTrackerWithPnL()
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        tracker.add_position(position)

        assert tracker.get_position("ORDER-001") == position
        assert tracker.open_position_count == 1

    def test_update_from_quote(self):
        """Test updating positions from quote stream."""
        tracker = PositionTrackerWithPnL()

        # Add two positions for the same symbol
        position1 = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        position2 = PositionWithPnL(
            order_id="ORDER-002",
            signal_id="SIG-002",
            symbol="MNQH26",
            entry_price=11840.0,
            quantity=3,
            direction="bearish",
            timestamp=datetime.now(timezone.utc),
        )

        tracker.add_position(position1)
        tracker.add_position(position2)

        # Update from quote
        quote = StreamPosition(
            symbol="MNQH26",
            timestamp=datetime.now(timezone.utc),
            last_price=11855.0,
            bid=11854.75,
            ask=11855.25,
            volume=1000,
        )
        tracker.update_from_quote(quote)

        # Check positions updated
        assert position1.current_price == 11855.0
        assert position2.current_price == 11855.0

    def test_close_position(self):
        """Test closing a position."""
        tracker = PositionTrackerWithPnL()
        position = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )

        tracker.add_position(position)

        # Update to new price
        position.update_mark_to_market(11860.0)

        # Close position
        realized_pnl = tracker.close_position("ORDER-001", 11860.0)

        assert realized_pnl == 25.0
        assert tracker.total_realized_pnl == 25.0
        assert tracker.open_position_count == 0

    def test_total_pnl_calculation(self):
        """Test total P&L calculation across multiple positions."""
        tracker = PositionTrackerWithPnL()

        # Add profitable position
        position1 = PositionWithPnL(
            order_id="ORDER-001",
            signal_id="SIG-001",
            symbol="MNQH26",
            entry_price=11850.0,
            quantity=5,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )
        position1.update_mark_to_market(11860.0)
        tracker.add_position(position1)

        # Add losing position
        position2 = PositionWithPnL(
            order_id="ORDER-002",
            signal_id="SIG-002",
            symbol="MNQH26",
            entry_price=11840.0,
            quantity=3,
            direction="bullish",
            timestamp=datetime.now(timezone.utc),
        )
        position2.update_mark_to_market(11835.0)
        tracker.add_position(position2)

        # Total unrealized should be $25 + (-$7.5) = $17.5
        assert tracker.total_unrealized_pnl == 17.5


class TestSIMOrderSubmitter:
    """Tests for SIMOrderSubmitter class."""

    @pytest.mark.asyncio
    async def test_order_submitter_initialization(self, mock_auth):
        """Test initializing SIM order submitter."""
        submitter = SIMOrderSubmitter(auth=mock_auth)

        assert submitter._auth == mock_auth
        assert submitter._http_client is None

    @pytest.mark.asyncio
    async def test_submit_order_success(self, mock_auth, sample_trading_signal):
        """Test successful order submission."""
        submitter = SIMOrderSubmitter(auth=mock_auth)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"OrderID": "ORDER-123"}

        with patch.object(submitter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await submitter.submit_order(sample_trading_signal)

            assert result.success is True
            assert result.order_id == "ORDER-123"
            assert result.status == "FILLED"
            assert result.fill_price == sample_trading_signal.entry_price

    @pytest.mark.asyncio
    async def test_submit_order_authentication_failure(self, mock_auth, sample_trading_signal):
        """Test order submission with authentication failure."""
        submitter = SIMOrderSubmitter(auth=mock_auth)

        # Mock HTTP client with 401 response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")

        with patch.object(submitter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await submitter.submit_order(sample_trading_signal)

            assert result.success is False
            assert result.error_message == "Authentication failed"

    @pytest.mark.asyncio
    async def test_submit_order_rate_limit(self, mock_auth, sample_trading_signal):
        """Test order submission with rate limit."""
        submitter = SIMOrderSubmitter(auth=mock_auth)

        # Mock HTTP client with 429 response
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(submitter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await submitter.submit_order(sample_trading_signal)

            assert result.success is False
            assert result.error_message == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_build_order_payload(self, mock_auth, sample_trading_signal):
        """Test building order payload."""
        submitter = SIMOrderSubmitter(auth=mock_auth)

        payload = submitter._build_order_payload(sample_trading_signal)

        assert payload["Symbol"] == "MNQH26"
        assert payload["Side"] == "Buy"
        assert payload["OrderType"] == "Market"
        assert payload["TimeInForce"] == "DAY"


class TestEndToEndIntegration:
    """End-to-end integration tests for SIM paper trading."""

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(
        self,
        mock_auth,
        sample_stream_position,
        sample_trading_signal,
    ):
        """Test complete workflow: signal → order → position → P&L update."""
        # Step 1: Initialize components
        order_submitter = SIMOrderSubmitter(auth=mock_auth)
        position_tracker = PositionTrackerWithPnL()

        # Step 2: Submit order (mock success)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"OrderID": "ORDER-123"}

        with patch.object(order_submitter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            order_result = await order_submitter.submit_order(sample_trading_signal)

            assert order_result.success is True

        # Step 3: Create position
        position = PositionWithPnL(
            order_id=order_result.order_id,
            signal_id=sample_trading_signal.signal_id,
            symbol=sample_trading_signal.symbol,
            entry_price=sample_trading_signal.entry_price,
            quantity=DEFAULT_ORDER_QUANTITY,  # Use default quantity
            direction=sample_trading_signal.direction,
            timestamp=sample_trading_signal.timestamp,
        )
        position_tracker.add_position(position)

        # Step 4: Update position from live quote
        position_tracker.update_from_quote(sample_stream_position)

        # Step 5: Verify P&L tracking
        assert position_tracker.open_position_count == 1
        updated_position = position_tracker.get_position(order_result.order_id)
        assert updated_position.current_price == sample_stream_position.last_price

        # Step 6: Close position
        realized_pnl = position_tracker.close_position(
            order_result.order_id,
            sample_stream_position.last_price,
        )

        assert position_tracker.open_position_count == 0
        assert position_tracker.total_realized_pnl == realized_pnl

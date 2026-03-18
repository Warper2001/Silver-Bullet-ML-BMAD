"""Unit tests for PartialFillHandler and PositionTracker.

Tests partial fill detection, unfilled quantity calculation,
limit price recalculation, order resubmission, position tracking updates,
cumulative fill time tracking, and 5-minute timeout handling.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock
import pytest
import tempfile
import csv

from src.execution.partial_fill_handler import (
    PartialFillHandler,
)
from src.execution.position_tracker import (
    PositionTracker,
    Position,
)


class TestPositionTracker:
    """Test PositionTracker functionality for partial fill tracking."""

    @pytest.fixture
    def tracker(self):
        """Create position tracker instance."""
        return PositionTracker()

    def test_add_position(self, tracker):
        """Verify adding a new position."""
        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.0,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=datetime.now(timezone.utc)
        )

        tracker.add_position(position)
        retrieved = tracker.get_position("ORDER-123")

        assert retrieved is not None
        assert retrieved.order_id == "ORDER-123"
        assert retrieved.quantity == 5
        assert retrieved.filled_quantity == 5  # Initially assumes full fill

    def test_update_fill_with_partial_fill(self, tracker):
        """Verify updating position with partial fill."""
        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.0,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=datetime.now(timezone.utc)
        )

        tracker.add_position(position)

        # Update with partial fill
        tracker.update_fill(
            order_id="ORDER-123",
            filled_quantity=2,
            new_order_id=None
        )

        retrieved = tracker.get_position("ORDER-123")
        assert retrieved.filled_quantity == 2
        assert retrieved.unfilled_quantity == 3

    def test_update_fill_with_new_order_id(self, tracker):
        """Verify updating position with resubmitted order."""
        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.0,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=datetime.now(timezone.utc)
        )

        tracker.add_position(position)

        # Update with partial fill and new order
        tracker.update_fill(
            order_id="ORDER-123",
            filled_quantity=2,
            new_order_id="ORDER-789"
        )

        retrieved = tracker.get_position("ORDER-123")
        assert retrieved.filled_quantity == 2
        assert retrieved.unfilled_quantity == 3
        assert retrieved.new_order_id == "ORDER-789"

    def test_update_status(self, tracker):
        """Verify updating position status."""
        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.0,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=datetime.now(timezone.utc)
        )

        tracker.add_position(position)
        tracker.update_status("ORDER-123", "PARTIALLY_FILLED")

        retrieved = tracker.get_position("ORDER-123")
        assert retrieved.status == "PARTIALLY_FILLED"

    def test_get_nonexistent_position(self, tracker):
        """Verify getting nonexistent position returns None."""
        result = tracker.get_position("ORDER-NONEXISTENT")
        assert result is None


class TestPartialFillHandlerInit:
    """Test PartialFillHandler initialization."""

    def test_init_with_required_parameters(self):
        """Verify PartialFillHandler initializes with required parameters."""
        mock_api = Mock()
        mock_tracker = Mock()
        audit_path = tempfile.mktemp(suffix=".csv")

        handler = PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker,
            audit_trail_path=audit_path
        )

        assert handler._api_client == mock_api
        assert handler._position_tracker == mock_tracker
        assert handler._audit_trail_path == audit_path
        assert handler._timeout_seconds == 300  # 5 minutes

    def test_init_with_custom_timeout(self):
        """Verify PartialFillHandler initializes with custom timeout."""
        mock_api = Mock()
        mock_tracker = Mock()

        handler = PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker,
            timeout_seconds=600  # 10 minutes
        )

        assert handler._timeout_seconds == 600


class TestUnfilledQuantityCalculation:
    """Test unfilled quantity calculation."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        return PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_calculate_unfilled_quantity(self, handler):
        """Verify unfilled quantity calculation."""
        # 5 target, 2 filled
        result = handler.calculate_unfilled_quantity(5, 2)
        assert result == 3

        # 3 target, 1 filled
        result = handler.calculate_unfilled_quantity(3, 1)
        assert result == 2

    def test_calculate_unfilled_quantity_zero_when_full(self, handler):
        """Verify unfilled quantity is zero when fully filled."""
        result = handler.calculate_unfilled_quantity(5, 5)
        assert result == 0


class TestLimitPriceRecalculation:
    """Test limit price recalculation."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        return PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_recalculate_limit_price_for_bullish(self, handler):
        """Verify limit price recalculation for bullish signals."""
        limit_price = handler.recalculate_limit_price(
            current_price=11800.00,
            direction="bullish"
        )

        # 11800 + (2 × $0.25) = 11800.50
        assert limit_price == 11800.50

    def test_recalculate_limit_price_for_bearish(self, handler):
        """Verify limit price recalculation for bearish signals."""
        limit_price = handler.recalculate_limit_price(
            current_price=11800.00,
            direction="bearish"
        )

        # 11800 - (2 × $0.25) = 11799.50
        assert limit_price == 11799.50


class TestOrderResubmission:
    """Test order resubmission logic."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "ORDER-789"
        }
        return PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_resubmit_unfilled_quantity(self, handler):
        """Verify resubmission of unfilled quantity."""
        new_order_id = handler.resubmit_unfilled_quantity(
            original_order_id="ORDER-123",
            unfilled_quantity=3,
            direction="bullish",
            limit_price=11800.50
        )

        assert new_order_id == "ORDER-789"
        # Verify API was called
        handler._api_client.submit_order.assert_called_once()

        # Verify payload had unfilled quantity
        call_args = handler._api_client.submit_order.call_args
        payload = call_args[0][0]
        assert payload["quantity"] == 3
        assert payload["orderType"] == "LIMIT"
        assert payload["limitPrice"] == 11800.50


class TestCumulativeFillTimeTracking:
    """Test cumulative fill time tracking."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        return PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_calculate_elapsed_time(self, handler):
        """Verify elapsed time calculation."""
        initial_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        elapsed = handler._calculate_elapsed_time(initial_time)

        assert elapsed == pytest.approx(30, abs=2)

    def test_calculate_elapsed_time_none(self, handler):
        """Verify elapsed time handles None."""
        elapsed = handler._calculate_elapsed_time(None)
        assert elapsed == 0.0


class TestTimeoutHandling:
    """Test 5-minute timeout handling."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        mock_api.cancel_order.return_value = {"success": True}
        return PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_handle_timeout_cancels_order(self, handler):
        """Verify timeout handling cancels order."""
        handler._handle_timeout(
            original_order_id="ORDER-123",
            unfilled_quantity=3
        )

        # Verify order cancelled
        handler._api_client.cancel_order.assert_called_once_with("ORDER-123")

    def test_handle_timeout_updates_status(self, handler):
        """Verify timeout handling updates position status."""
        handler._handle_timeout(
            original_order_id="ORDER-123",
            unfilled_quantity=3
        )

        # Verify status updated
        handler._position_tracker.update_status.assert_called_once_with(
            "ORDER-123", "UNFILLED_TIMEOUT"
        )


class TestMainHandleMethod:
    """Test main handle_partial_fill() method."""

    @pytest.fixture
    def handler(self):
        """Create handler instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()

        # Mock position with initial submission time
        thirty_seconds_ago = datetime.now(timezone.utc) - timedelta(
            seconds=30
        )
        mock_position = {
            "order_id": "ORDER-123",
            "initial_submission_time": thirty_seconds_ago,
            "quantity": 5,
            "direction": "bullish"
        }
        mock_tracker.get_position.return_value = mock_position

        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "ORDER-789"
        }

        return PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_handle_partial_fill_success(self, handler):
        """Verify successful partial fill handling."""
        result = handler.handle_partial_fill(
            original_order_id="ORDER-123",
            filled_quantity=2,
            target_quantity=5,
            current_price=11800.00,
            direction="bullish"
        )

        assert result.original_order_id == "ORDER-123"
        assert result.filled_quantity == 2
        assert result.unfilled_quantity == 3
        assert result.new_order_id == "ORDER-789"
        assert result.status == "PARTIALLY_FILLED"

    def test_handle_partial_fill_fully_filled(self, handler):
        """Verify handling when order becomes fully filled."""
        result = handler.handle_partial_fill(
            original_order_id="ORDER-123",
            filled_quantity=5,
            target_quantity=5,
            current_price=11800.00,
            direction="bullish"
        )

        assert result.unfilled_quantity == 0
        assert result.new_order_id is None
        assert result.status == "FULLY_FILLED"

    def test_handle_partial_fill_timeout(self, handler):
        """Verify handling when timeout is reached."""
        # Mock position that's timed out
        old_time = datetime.now(timezone.utc) - timedelta(seconds=400)
        mock_tracker = handler._position_tracker
        mock_tracker.get_position.return_value = {
            "order_id": "ORDER-123",
            "initial_submission_time": old_time,
            "quantity": 5,
            "direction": "bullish"
        }

        result = handler.handle_partial_fill(
            original_order_id="ORDER-123",
            filled_quantity=2,
            target_quantity=5,
            current_price=11800.00,
            direction="bullish"
        )

        assert result.status == "UNFILLED_TIMEOUT"
        assert result.new_order_id is None


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging for partial fills."""

    @pytest.fixture
    def handler(self):
        """Create handler instance with temp audit file."""
        mock_api = Mock()
        mock_tracker = Mock()
        audit_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        audit_file.close()

        handler = PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker,
            audit_trail_path=audit_file.name
        )
        yield handler

        # Cleanup
        Path(audit_file.name).unlink(missing_ok=True)

    def test_log_partial_fill_event(self, handler):
        """Verify partial fill event logging."""
        handler.log_partial_fill_event(
            original_order_id="ORDER-123",
            filled_quantity=2,
            unfilled_quantity=3,
            new_order_id="ORDER-789",
            cumulative_time=30.5
        )

        # Verify file exists and has entry
        with open(handler._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + at least one entry
        assert len(rows) >= 2

        # Find partial fill entry
        partial_fill_rows = [r for r in rows if "PARTIAL_FILL" in r]
        assert len(partial_fill_rows) >= 1


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_handling_completes_under_100ms(self):
        """Verify partial fill handling completes in < 100ms."""
        import time

        mock_api = Mock()
        mock_tracker = Mock()
        thirty_seconds_ago = datetime.now(timezone.utc) - timedelta(
            seconds=30
        )
        mock_tracker.get_position.return_value = {
            "order_id": "ORDER-123",
            "initial_submission_time": thirty_seconds_ago,
            "quantity": 5,
            "direction": "bullish"
        }
        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "ORDER-789"
        }

        handler = PartialFillHandler(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

        start_time = time.perf_counter()
        result = handler.handle_partial_fill(
            original_order_id="ORDER-123",
            filled_quantity=2,
            target_quantity=5,
            current_price=11800.00,
            direction="bullish"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.status == "PARTIALLY_FILLED"
        assert elapsed_ms < 100.0, (
            "Handling took {:.2f}ms, exceeds 100ms limit".format(
                elapsed_ms
            )
        )

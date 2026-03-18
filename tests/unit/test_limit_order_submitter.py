"""Unit tests for LimitOrderSubmitter.

Tests limit order payload construction with limit price,
API submission with retry logic, order confirmation handling,
position tracking updates, and CSV audit trail logging.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest
import tempfile
import csv

from src.execution.limit_order_submitter import (
    OrderSubmissionResult,
    OrderSubmissionError,
    LimitOrderSubmitter,
)


@dataclass
class MockPositionSizeResult:
    """Mock position size result for testing."""
    position_size: int
    entry_price: float
    stop_loss: float
    dollar_risk: float
    stop_distance: float
    calculation_time_ms: float
    valid: bool
    validation_reason: str | None


@dataclass
class MockOrderTypeDecision:
    """Mock order type decision for testing."""
    order_type: str
    limit_price: float | None
    rationale: str
    selection_time_ms: float


@dataclass
class MockSilverBulletSetup:
    """Mock Silver Bullet setup for testing."""
    signal_id: str
    direction: str
    timestamp: datetime


class TestLimitOrderSubmitterInit:
    """Test LimitOrderSubmitter initialization and configuration."""

    def test_init_with_required_parameters(self):
        """Verify LimitOrderSubmitter initializes with required parameters."""
        mock_api = Mock()
        mock_tracker = Mock()
        audit_path = tempfile.mktemp(suffix=".csv")

        submitter = LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker,
            audit_trail_path=audit_path
        )

        assert submitter._api_client == mock_api
        assert submitter._position_tracker == mock_tracker
        assert submitter._audit_trail_path == audit_path
        assert submitter._max_retries == 3

    def test_init_with_custom_max_retries(self):
        """Verify LimitOrderSubmitter initializes with custom max retries."""
        mock_api = Mock()
        mock_tracker = Mock()

        submitter = LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker,
            max_retries=5
        )

        assert submitter._max_retries == 5

    def test_init_raises_error_for_none_api_client(self):
        """Verify ValueError raised for None API client."""
        mock_tracker = Mock()

        with pytest.raises(ValueError, match="API client cannot be None"):
            LimitOrderSubmitter(
                api_client=None,
                position_tracker=mock_tracker
            )

    def test_init_raises_error_for_none_position_tracker(self):
        """Verify ValueError raised for None position tracker."""
        mock_api = Mock()

        with pytest.raises(ValueError, match="Position tracker cannot be None"):
            LimitOrderSubmitter(
                api_client=mock_api,
                position_tracker=None
            )


class TestOrderPayloadConstruction:
    """Test limit order payload construction."""

    @pytest.fixture
    def submitter(self):
        """Create submitter instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        return LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_construct_payload_for_bullish_signal(self, submitter):
        """Verify payload construction for bullish (BUY) signal with limit price."""
        payload = submitter.construct_order_payload(
            position_size=5,
            direction="bullish",
            limit_price=11800.50
        )

        assert payload["symbol"] == "MNQ"
        assert payload["quantity"] == 5
        assert payload["side"] == "BUY"
        assert payload["orderType"] == "LIMIT"
        assert payload["limitPrice"] == 11800.50
        assert payload["timeInForce"] == "DAY"

    def test_construct_payload_for_bearish_signal(self, submitter):
        """Verify payload construction for bearish (SELL) signal with limit price."""
        payload = submitter.construct_order_payload(
            position_size=3,
            direction="bearish",
            limit_price=11799.50
        )

        assert payload["symbol"] == "MNQ"
        assert payload["quantity"] == 3
        assert payload["side"] == "SELL"
        assert payload["orderType"] == "LIMIT"
        assert payload["limitPrice"] == 11799.50
        assert payload["timeInForce"] == "DAY"

    def test_construct_payload_raises_error_for_invalid_position(self, submitter):
        """Verify ValueError raised for invalid position size."""
        with pytest.raises(ValueError, match="Position size must be positive"):
            submitter.construct_order_payload(
                position_size=0, direction="bullish", limit_price=11800.50
            )

        with pytest.raises(ValueError, match="Position size must be positive"):
            submitter.construct_order_payload(
                position_size=-1, direction="bullish", limit_price=11800.50
            )

    def test_construct_payload_raises_error_for_invalid_limit_price(self, submitter):
        """Verify ValueError raised for invalid limit price."""
        with pytest.raises(ValueError, match="Limit price must be positive"):
            submitter.construct_order_payload(
                position_size=5, direction="bullish", limit_price=0.0
            )

        with pytest.raises(ValueError, match="Limit price must be positive"):
            submitter.construct_order_payload(
                position_size=5, direction="bullish", limit_price=-100.0
            )

    def test_construct_payload_raises_error_for_invalid_direction(self, submitter):
        """Verify ValueError raised for invalid direction."""
        with pytest.raises(ValueError, match="Invalid direction"):
            submitter.construct_order_payload(
                position_size=5, direction="invalid", limit_price=11800.50
            )


class TestOrderSubmissionWithRetry:
    """Test order submission with retry logic."""

    @pytest.fixture
    def submitter(self):
        """Create submitter instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        return LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

    def test_submit_order_succeeds_on_first_attempt(self, submitter):
        """Verify order submission succeeds on first attempt."""
        payload = {"symbol": "MNQ", "quantity": 5, "orderType": "LIMIT"}
        submitter._api_client.submit_order.return_value = {
            "success": True,
            "order_id": "ORDER-123",
            "limitPrice": 11800.50
        }

        response = submitter.submit_order_with_retry(payload)

        assert response["success"] is True
        assert response["order_id"] == "ORDER-123"
        assert response["limitPrice"] == 11800.50
        assert submitter._api_client.submit_order.call_count == 1

    def test_submit_order_retries_on_failure(self, submitter):
        """Verify order submission retries on transient failure."""
        payload = {"symbol": "MNQ", "quantity": 5, "orderType": "LIMIT"}

        # First call fails, second succeeds
        submitter._api_client.submit_order.side_effect = [
            Exception("API timeout"),
            {"success": True, "order_id": "ORDER-456", "limitPrice": 11801.00}
        ]

        response = submitter.submit_order_with_retry(payload)

        assert response["success"] is True
        assert response["order_id"] == "ORDER-456"
        assert submitter._api_client.submit_order.call_count == 2

    def test_submit_order_fails_after_max_retries(self, submitter):
        """Verify OrderSubmissionError raised after max retries."""
        payload = {"symbol": "MNQ", "quantity": 5, "orderType": "LIMIT"}
        submitter._api_client.submit_order.side_effect = Exception("API error")

        with pytest.raises(OrderSubmissionError, match="Failed to submit order after 3 attempts"):
            submitter.submit_order_with_retry(payload)

        assert submitter._api_client.submit_order.call_count == 3

    def test_submit_order_no_retry_for_rejection(self, submitter):
        """Verify order rejection doesn't trigger retry."""
        payload = {"symbol": "MNQ", "quantity": 5, "orderType": "LIMIT"}
        submitter._api_client.submit_order.return_value = {
            "success": False,
            "error": "Insufficient margin"
        }

        with pytest.raises(OrderSubmissionError, match="Order rejected by API"):
            submitter.submit_order_with_retry(payload)

        # Should not retry rejected orders
        assert submitter._api_client.submit_order.call_count == 1


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def submitter(self):
        """Create submitter instance with temp audit file."""
        mock_api = Mock()
        mock_tracker = Mock()
        audit_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        audit_file.close()

        submitter = LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker,
            audit_trail_path=audit_file.name
        )
        yield submitter

        # Cleanup
        Path(audit_file.name).unlink(missing_ok=True)

    def test_log_to_audit_trail_creates_file_with_header(self, submitter):
        """Verify audit trail file created with header."""
        submitter.log_to_audit_trail(
            signal_id="SIG-123",
            order_id="ORDER-456",
            quantity=5,
            limit_price=11800.50,
            order_type="LIMIT"
        )

        # Verify file exists and has header + data
        with open(submitter._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0] == ["timestamp", "signal_id", "order_id", "quantity", "execution_price", "order_type"]
        assert rows[1][1] == "SIG-123"
        assert rows[1][2] == "ORDER-456"
        assert rows[1][4] == "11800.50"  # limit_price logged to execution_price column

    def test_log_to_audit_trail_appends_multiple_entries(self, submitter):
        """Verify multiple log entries append correctly."""
        submitter.log_to_audit_trail("SIG-1", "ORDER-1", 5, 11800.50, "LIMIT")
        submitter.log_to_audit_trail("SIG-2", "ORDER-2", 3, 11799.75, "LIMIT")

        with open(submitter._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 2 entries
        assert len(rows) == 3
        assert rows[1][1] == "SIG-1"
        assert rows[2][1] == "SIG-2"


class TestMainSubmitMethod:
    """Test main submit_limit_order() method."""

    @pytest.fixture
    def submitter(self):
        """Create submitter instance for testing."""
        mock_api = Mock()
        mock_tracker = Mock()
        audit_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        audit_file.close()

        submitter = LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker,
            audit_trail_path=audit_file.name
        )
        yield submitter

        Path(audit_file.name).unlink(missing_ok=True)

    @pytest.fixture
    def mock_signal(self):
        """Create mock signal."""
        return MockSilverBulletSetup(
            signal_id="SIG-123",
            direction="bullish",
            timestamp=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def mock_position_result(self):
        """Create mock position size result."""
        return MockPositionSizeResult(
            position_size=5,
            entry_price=11800.0,
            stop_loss=11740.0,
            dollar_risk=200.0,
            stop_distance=60.0,
            calculation_time_ms=0.5,
            valid=True,
            validation_reason=None
        )

    @pytest.fixture
    def mock_order_decision(self):
        """Create mock order type decision."""
        return MockOrderTypeDecision(
            order_type="LIMIT",
            limit_price=11800.50,
            rationale="Position size >= 3: using limit order",
            selection_time_ms=0.1
        )

    def test_submit_limit_order_success(self, submitter, mock_signal, mock_position_result, mock_order_decision):
        """Verify successful limit order submission."""
        submitter._api_client.submit_order.return_value = {
            "success": True,
            "order_id": "ORDER-789",
            "limitPrice": 11800.50
        }

        result = submitter.submit_limit_order(
            signal=mock_signal,
            position_size_result=mock_position_result,
            order_type_decision=mock_order_decision
        )

        assert result.success is True
        assert result.order_id == "ORDER-789"
        assert result.execution_price == 11800.50
        assert result.submitted_quantity == 5
        assert result.error_message is None

        # Verify position tracking updated
        submitter._position_tracker.add_position.assert_called_once()

        # Verify audit trail logged
        assert Path(submitter._audit_trail_path).exists()

    def test_submit_limit_order_raises_error_for_non_limit_order(self, submitter, mock_signal, mock_position_result):
        """Verify ValueError raised for non-LIMIT order type."""
        market_decision = MockOrderTypeDecision(
            order_type="MARKET",
            limit_price=None,
            rationale="Small position",
            selection_time_ms=0.1
        )

        with pytest.raises(ValueError, match="Expected LIMIT order type"):
            submitter.submit_limit_order(
                signal=mock_signal,
                position_size_result=mock_position_result,
                order_type_decision=market_decision
            )

    def test_submit_limit_order_raises_error_for_missing_limit_price(self, submitter, mock_signal, mock_position_result):
        """Verify ValueError raised for missing limit price."""
        limit_decision_no_price = MockOrderTypeDecision(
            order_type="LIMIT",
            limit_price=None,
            rationale="Large position",
            selection_time_ms=0.1
        )

        with pytest.raises(ValueError, match="Limit price must be provided"):
            submitter.submit_limit_order(
                signal=mock_signal,
                position_size_result=mock_position_result,
                order_type_decision=limit_decision_no_price
            )


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_submission_completes_under_200ms(self):
        """Verify order submission completes in < 200ms."""
        import time

        mock_api = Mock()
        mock_tracker = Mock()
        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "ORDER-123",
            "limitPrice": 11800.50
        }

        submitter = LimitOrderSubmitter(
            api_client=mock_api,
            position_tracker=mock_tracker
        )

        signal = MockSilverBulletSetup(
            signal_id="SIG-123",
            direction="bullish",
            timestamp=datetime.now(timezone.utc)
        )
        position_result = MockPositionSizeResult(
            position_size=5,
            entry_price=11800.0,
            stop_loss=11740.0,
            dollar_risk=200.0,
            stop_distance=60.0,
            calculation_time_ms=0.5,
            valid=True,
            validation_reason=None
        )
        order_decision = MockOrderTypeDecision(
            order_type="LIMIT",
            limit_price=11800.50,
            rationale="Large position",
            selection_time_ms=0.1
        )

        start_time = time.perf_counter()
        result = submitter.submit_limit_order(
            signal=signal,
            position_size_result=position_result,
            order_type_decision=order_decision
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.success is True
        assert elapsed_ms < 200.0, f"Submission took {elapsed_ms:.2f}ms, exceeds 200ms limit"

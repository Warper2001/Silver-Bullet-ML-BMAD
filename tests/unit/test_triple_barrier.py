"""Unit tests for Triple Barrier Exit Strategy.

Tests triple barrier calculation, barrier monitoring, exit execution,
profit/loss calculation, and CSV audit trail logging.
"""

import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
import pytest

from src.execution.triple_barrier_calculator import (
    TripleBarrierCalculator,
)
from src.execution.triple_barrier_monitor import (
    TripleBarrierMonitor,
    BarrierCheckResult,
)
from src.execution.triple_barrier_exit_executor import (
    TripleBarrierExitExecutor,
)
from src.execution.position_tracker import (
    PositionTracker,
    Position,
)


class TestTripleBarrierCalculator:
    """Test triple barrier calculation."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return TripleBarrierCalculator()

    def test_calculate_barriers_for_bullish(self, calculator):
        """Verify barrier calculation for bullish direction."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        barriers = calculator.calculate_barriers(
            entry_price=11800.00,
            direction="bullish",
            entry_time=entry_time
        )

        # Upper barrier = entry + $15
        assert barriers.upper_barrier_price == 11815.00
        # Lower barrier = entry - $7.50
        assert barriers.lower_barrier_price == 11792.50
        # Time barrier = 19:00 UTC same day
        assert barriers.time_barrier_utc.hour == 19
        assert barriers.time_barrier_utc.day == 17
        assert barriers.entry_price == 11800.00
        assert barriers.direction == "bullish"

    def test_calculate_barriers_for_bearish(self, calculator):
        """Verify barrier calculation for bearish direction."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        barriers = calculator.calculate_barriers(
            entry_price=11800.00,
            direction="bearish",
            entry_time=entry_time
        )

        # Upper barrier = entry - $15 (profit when price drops)
        assert barriers.upper_barrier_price == 11785.00
        # Lower barrier = entry + $7.50 (stop when price rises)
        assert barriers.lower_barrier_price == 11807.50
        assert barriers.direction == "bearish"

    def test_time_barrier_same_day(self, calculator):
        """Verify time barrier is same day when entry before 19:00 UTC."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        barriers = calculator.calculate_barriers(
            entry_price=11800.00,
            direction="bullish",
            entry_time=entry_time
        )

        assert barriers.time_barrier_utc.hour == 19
        assert barriers.time_barrier_utc.day == 17
        assert barriers.time_barrier_utc.month == 3

    def test_time_barrier_next_day(self, calculator):
        """Verify time barrier is next day when entry after 19:00 UTC."""
        entry_time = datetime(2026, 3, 17, 20, 0, 0, tzinfo=timezone.utc)

        barriers = calculator.calculate_barriers(
            entry_price=11800.00,
            direction="bullish",
            entry_time=entry_time
        )

        # Should be next day 19:00 UTC
        assert barriers.time_barrier_utc.hour == 19
        assert barriers.time_barrier_utc.day == 18
        assert barriers.time_barrier_utc.month == 3

    def test_invalid_direction_raises_error(self, calculator):
        """Verify invalid direction raises ValueError."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Invalid direction"):
            calculator.calculate_barriers(
                entry_price=11800.00,
                direction="invalid",
                entry_time=entry_time
            )


class TestTripleBarrierMonitor:
    """Test barrier monitoring."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return TripleBarrierCalculator()

    @pytest.fixture
    def tracker(self):
        """Create position tracker."""
        return PositionTracker()

    @pytest.fixture
    def monitor(self, tracker):
        """Create monitor instance."""
        return TripleBarrierMonitor(tracker)

    def test_upper_barrier_hit_bullish(self, monitor, tracker):
        """Verify upper barrier detection for bullish position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        # Add position with barriers
        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        # Check with price at upper barrier
        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11815.50,  # Above upper barrier
            current_time=current_time
        )

        assert result.barrier_hit == "UPPER"
        assert result.should_exit is True
        assert "Upper barrier hit" in result.exit_reason

    def test_upper_barrier_hit_bearish(self, monitor, tracker):
        """Verify upper barrier detection for bearish position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        # Add bearish position
        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bearish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11785.00
        position.lower_barrier_price = 11807.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        # Check with price at upper barrier (price dropped)
        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11784.50,  # Below upper barrier
            current_time=current_time
        )

        assert result.barrier_hit == "UPPER"
        assert result.should_exit is True

    def test_lower_barrier_hit_bullish(self, monitor, tracker):
        """Verify lower barrier detection for bullish position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11792.00,  # Below lower barrier
            current_time=current_time
        )

        assert result.barrier_hit == "LOWER"
        assert result.should_exit is True

    def test_lower_barrier_hit_bearish(self, monitor, tracker):
        """Verify lower barrier detection for bearish position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bearish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11785.00
        position.lower_barrier_price = 11807.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11808.00,  # Above lower barrier
            current_time=current_time
        )

        assert result.barrier_hit == "LOWER"
        assert result.should_exit is True

    def test_time_barrier_hit(self, monitor, tracker):
        """Verify time barrier detection."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        # Check after time barrier
        current_time = datetime(2026, 3, 17, 19, 30, 0, tzinfo=timezone.utc)
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11805.00,  # Between barriers
            current_time=current_time
        )

        assert result.barrier_hit == "TIME"
        assert result.should_exit is True

    def test_no_barrier_hit(self, monitor, tracker):
        """Verify no barrier hit when price between barriers."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11805.00,  # Between barriers
            current_time=current_time
        )

        assert result.barrier_hit is None
        assert result.should_exit is False

    def test_position_not_found(self, monitor, tracker):
        """Verify handling when position not found."""
        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)

        result = monitor.check_barriers(
            order_id="ORDER-NONEXISTENT",
            current_price=11805.00,
            current_time=current_time
        )

        assert result.should_exit is False
        assert "Position not found" in result.exit_reason


class TestTripleBarrierExitExecutor:
    """Test exit execution."""

    @pytest.fixture
    def api_client(self):
        """Create mock API client."""
        mock_api = Mock()
        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "EXIT-456",
            "filled_price": 11815.50
        }
        return mock_api

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return TripleBarrierCalculator()

    @pytest.fixture
    def monitor(self):
        """Create mock monitor."""
        mock_monitor = Mock()
        return mock_monitor

    @pytest.fixture
    def tracker(self):
        """Create position tracker."""
        return PositionTracker()

    @pytest.fixture
    def executor(self, api_client, calculator, monitor, tracker):
        """Create executor instance."""
        return TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )

    def test_execute_exit_on_upper_barrier(self, executor, tracker, monitor):
        """Verify exit execution when upper barrier hit."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        # Mock monitor to return upper barrier hit
        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="UPPER",
            hit_time=datetime.now(timezone.utc),
            current_price=11815.50,
            exit_price=11815.50,
            exit_reason="Upper barrier hit",
            should_exit=True
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11815.50,
            current_time=current_time
        )

        assert result.success is True
        assert result.exit_order_id == "EXIT-456"
        assert result.exit_barrier == "UPPER"
        assert result.exit_quantity == 5

    def test_execute_exit_on_lower_barrier(self, executor, tracker, monitor):
        """Verify exit execution when lower barrier hit."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="LOWER",
            hit_time=datetime.now(timezone.utc),
            current_price=11792.00,
            exit_price=11792.00,
            exit_reason="Lower barrier hit",
            should_exit=True
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11792.00,
            current_time=current_time
        )

        assert result.exit_barrier == "LOWER"

    def test_execute_exit_on_time_barrier(self, executor, tracker, monitor):
        """Verify exit execution when time barrier hit."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="TIME",
            hit_time=datetime.now(timezone.utc),
            current_price=11805.00,
            exit_price=11805.00,
            exit_reason="Time barrier hit",
            should_exit=True
        )

        current_time = datetime(2026, 3, 17, 19, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11805.00,
            current_time=current_time
        )

        assert result.exit_barrier == "TIME"

    def test_profit_loss_calculation_bullish(self, executor, tracker, monitor):
        """Verify profit/loss calculation for bullish position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="UPPER",
            hit_time=datetime.now(timezone.utc),
            current_price=11815.50,
            exit_price=11815.50,
            exit_reason="Upper barrier hit",
            should_exit=True
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11815.50,
            current_time=current_time
        )

        # Profit = (11815.50 - 11800.00) * 5 contracts * $20/point
        # = 15.50 * 5 * 20 = $1550.00
        expected_profit = 15.50 * 5 * 20
        assert result.profit_loss == expected_profit

    def test_profit_loss_calculation_bearish(self, executor, tracker, monitor):
        """Verify profit/loss calculation for bearish position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bearish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11785.00
        position.lower_barrier_price = 11807.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        # Mock API to return bearish exit price
        executor._api_client.submit_order.return_value = {
            "success": True,
            "order_id": "EXIT-456",
            "filled_price": 11784.00  # Price dropped, profit for short
        }

        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="UPPER",
            hit_time=datetime.now(timezone.utc),
            current_price=11784.00,
            exit_price=11784.00,
            exit_reason="Upper barrier hit",
            should_exit=True
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11784.00,
            current_time=current_time
        )

        # Profit = (11800.00 - 11784.00) * 5 * 20 = $1600.00
        expected_profit = 16.00 * 5 * 20
        assert result.profit_loss == expected_profit

    def test_no_barrier_hit_no_exit(self, executor, tracker, monitor):
        """Verify no exit when no barrier hit."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit=None,
            hit_time=datetime.now(timezone.utc),
            current_price=11805.00,
            exit_price=11805.00,
            exit_reason="No barrier hit",
            should_exit=False
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11805.00,
            current_time=current_time
        )

        assert result.success is True  # No error
        assert result.exit_order_id is None  # No exit order
        assert "No barrier hit" in result.exit_reason

    def test_position_not_found(self, executor, monitor):
        """Verify handling when position not found."""
        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit=None,
            hit_time=datetime.now(timezone.utc),
            current_price=11805.00,
            exit_price=11805.00,
            exit_reason="Position not found",
            should_exit=False
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        result = executor.execute_exit(
            order_id="ORDER-NONEXISTENT",
            current_price=11805.00,
            current_time=current_time
        )

        assert result.success is False
        assert "Position not found" in result.exit_reason


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging for exits."""

    @pytest.fixture
    def api_client(self):
        """Create mock API client."""
        mock_api = Mock()
        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "EXIT-456",
            "filled_price": 11815.50
        }
        return mock_api

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return TripleBarrierCalculator()

    @pytest.fixture
    def monitor(self):
        """Create mock monitor."""
        mock_monitor = Mock()
        mock_monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="UPPER",
            hit_time=datetime.now(timezone.utc),
            current_price=11815.50,
            exit_price=11815.50,
            exit_reason="Upper barrier hit",
            should_exit=True
        )
        return mock_monitor

    @pytest.fixture
    def tracker(self):
        """Create position tracker."""
        return PositionTracker()

    @pytest.fixture
    def executor(self, api_client, calculator, monitor, tracker):
        """Create executor with temp audit file."""
        audit_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        )
        audit_file.close()

        exec_instance = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor,
            audit_trail_path=audit_file.name
        )

        yield exec_instance

        # Cleanup
        Path(audit_file.name).unlink(missing_ok=True)

    def test_log_exit_event(self, executor, tracker):
        """Verify exit event logging to CSV."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        executor.execute_exit(
            order_id="ORDER-123",
            current_price=11815.50,
            current_time=current_time
        )

        # Verify file exists and has entry
        with open(executor._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + at least one entry
        assert len(rows) >= 2

        # Find exit entry
        exit_rows = [r for r in rows if "EXIT" in r]
        assert len(exit_rows) >= 1


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_barrier_check_completes_under_10ms(self):
        """Verify barrier check completes in < 10ms."""
        import time

        tracker = PositionTracker()
        monitor = TripleBarrierMonitor(tracker)

        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)

        start_time = time.perf_counter()
        result = monitor.check_barriers(
            order_id="ORDER-123",
            current_price=11815.50,
            current_time=current_time
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.barrier_hit == "UPPER"
        assert elapsed_ms < 10.0, (
            "Barrier check took {:.2f}ms, exceeds 10ms limit".format(
                elapsed_ms
            )
        )

    def test_exit_execution_completes_under_200ms(self):
        """Verify exit execution completes in < 200ms."""
        import time

        mock_api = Mock()
        mock_api.submit_order.return_value = {
            "success": True,
            "order_id": "EXIT-456",
            "filled_price": 11815.50
        }

        calculator = TripleBarrierCalculator()
        monitor = Mock()
        monitor.check_barriers.return_value = BarrierCheckResult(
            barrier_hit="UPPER",
            hit_time=datetime.now(timezone.utc),
            current_price=11815.50,
            exit_price=11815.50,
            exit_reason="Upper barrier hit",
            should_exit=True
        )
        tracker = PositionTracker()

        executor = TripleBarrierExitExecutor(
            api_client=mock_api,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )

        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        position = Position(
            order_id="ORDER-123",
            signal_id="SIG-456",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            order_type="LIMIT",
            timestamp=entry_time
        )
        position.upper_barrier_price = 11815.00
        position.lower_barrier_price = 11792.50
        position.time_barrier_utc = datetime(
            2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc
        )

        tracker.add_position(position)

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)

        start_time = time.perf_counter()
        result = executor.execute_exit(
            order_id="ORDER-123",
            current_price=11815.50,
            current_time=current_time
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.success is True
        assert elapsed_ms < 200.0, (
            "Exit execution took {:.2f}ms, exceeds 200ms limit".format(
                elapsed_ms
            )
        )

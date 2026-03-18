"""Unit tests for Position Monitoring Service.

Tests position entry handling, barrier calculation, price update monitoring,
exit execution, position status queries, and CSV audit trail logging.
"""

import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
import pytest

from src.execution.position_monitoring_service import (
    PositionMonitoringService,
)
from src.execution.triple_barrier_calculator import TripleBarrierCalculator
from src.execution.triple_barrier_monitor import TripleBarrierMonitor
from src.execution.triple_barrier_exit_executor import TripleBarrierExitExecutor
from src.execution.position_tracker import PositionTracker


class TestPositionMonitoringServiceInit:
    """Test PositionMonitoringService initialization."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return TripleBarrierCalculator()

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        tracker = PositionTracker()
        return TripleBarrierMonitor(tracker)

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        api_client = Mock()
        tracker = PositionTracker()
        calculator = TripleBarrierCalculator()
        monitor = Mock()
        return TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )

    @pytest.fixture
    def tracker(self):
        """Create position tracker."""
        return PositionTracker()

    @pytest.fixture
    def api_client(self):
        """Create mock API client."""
        return Mock()

    def test_init_with_required_components(
        self, calculator, monitor, executor, tracker, api_client
    ):
        """Verify service initializes with required components."""
        service = PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client
        )

        assert service._calculator == calculator
        assert service._monitor == monitor
        assert service._executor == executor
        assert service._position_tracker == tracker
        assert service._api_client == api_client


class TestOnPositionEntered:
    """Test barrier calculation and storage on position entry."""

    @pytest.fixture
    def service(self):
        """Create monitoring service."""
        calculator = TripleBarrierCalculator()
        monitor = TripleBarrierMonitor(PositionTracker())
        api_client = Mock()
        tracker = PositionTracker()
        executor = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )
        return PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client
        )

    def test_calculate_and_store_barriers_for_bullish(self, service):
        """Verify barrier calculation for bullish position entry."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        position = service._position_tracker.get_position("ORDER-123")

        assert position is not None
        assert position.upper_barrier_price == 11815.00
        assert position.lower_barrier_price == 11792.50
        assert position.time_barrier_utc.hour == 19

    def test_calculate_and_store_barriers_for_bearish(self, service):
        """Verify barrier calculation for bearish position entry."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bearish",
            entry_time=entry_time
        )

        position = service._position_tracker.get_position("ORDER-123")

        assert position.upper_barrier_price == 11785.00
        assert position.lower_barrier_price == 11807.50


class TestOnPriceUpdate:
    """Test barrier checking and exit execution on price updates."""

    @pytest.fixture
    def service(self):
        """Create monitoring service with mocked executor."""
        calculator = TripleBarrierCalculator()
        tracker = PositionTracker()
        monitor = TripleBarrierMonitor(tracker)

        # Mock API and executor
        api_client = Mock()
        api_client.submit_order.return_value = {
            "success": True,
            "order_id": "EXIT-456",
            "filled_price": 11815.50
        }

        executor = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )

        return PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client
        )

    def test_check_and_exit_single_position(self, service):
        """Verify single position checked and exited."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        # Add position with barriers
        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        # Update price at upper barrier
        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        exits = service.on_price_update(
            current_price=11815.50,
            current_time=current_time
        )

        assert len(exits) == 1
        assert exits[0].order_id == "ORDER-123"
        assert exits[0].exit_barrier == "UPPER"

    def test_check_multiple_positions(self, service):
        """Verify multiple positions checked and exited."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        # Add two positions with prices far apart
        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        service.on_position_entered(
            order_id="ORDER-456",
            entry_price=11900.00,
            quantity=3,
            direction="bullish",
            entry_time=entry_time
        )

        # Check with price that will hit barriers for both
        # ORDER-123 (bullish, entry 11800): upper=11815, lower=11792.50
        # ORDER-456 (bullish, entry 11900): upper=11915, lower=11892.50
        # Price 11850 hits upper for ORDER-123 and lower for ORDER-456
        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        exits = service.on_price_update(
            current_price=11850.00,
            current_time=current_time
        )

        # Both positions should exit
        assert len(exits) == 2
        assert exits[0].order_id == "ORDER-123"
        assert exits[0].exit_barrier == "UPPER"
        assert exits[1].order_id == "ORDER-456"
        assert exits[1].exit_barrier == "LOWER"

    def test_no_exit_when_no_barrier_hit(self, service):
        """Verify no exit when price between barriers."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        exits = service.on_price_update(
            current_price=11805.00,  # Between barriers
            current_time=current_time
        )

        assert len(exits) == 0


class TestGetPositionStatus:
    """Test position status queries."""

    @pytest.fixture
    def service(self):
        """Create monitoring service."""
        calculator = TripleBarrierCalculator()
        monitor = TripleBarrierMonitor(PositionTracker())
        api_client = Mock()
        tracker = PositionTracker()
        executor = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )
        return PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client
        )

    def test_get_position_status_active(self, service):
        """Verify status calculation for active position."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        status = service.get_position_status(
            order_id="ORDER-123",
            current_price=11805.00,
            current_time=current_time
        )

        assert status.order_id == "ORDER-123"
        assert status.entry_price == 11800.00
        assert status.current_price == 11805.00
        assert status.upper_barrier == 11815.00
        assert status.lower_barrier == 11792.50
        assert status.distance_to_upper == 10.00
        assert status.distance_to_lower == 12.50
        assert status.status == "ACTIVE"

    def test_get_position_status_not_found(self, service):
        """Verify handling when position not found."""
        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)

        status = service.get_position_status(
            order_id="ORDER-NONEXISTENT",
            current_price=11805.00,
            current_time=current_time
        )

        assert status is None


class TestGetAllPositionsStatus:
    """Test all positions status query."""

    @pytest.fixture
    def service(self):
        """Create monitoring service."""
        calculator = TripleBarrierCalculator()
        monitor = TripleBarrierMonitor(PositionTracker())
        api_client = Mock()
        tracker = PositionTracker()
        executor = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )
        return PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client
        )

    def test_get_all_positions_status(self, service):
        """Verify status query for multiple positions."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        service.on_position_entered(
            order_id="ORDER-456",
            entry_price=11850.00,
            quantity=3,
            direction="bearish",
            entry_time=entry_time
        )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
        statuses = service.get_all_positions_status(
            current_price=11805.00,
            current_time=current_time
        )

        assert len(statuses) == 2
        assert statuses[0].order_id == "ORDER-123"
        assert statuses[1].order_id == "ORDER-456"


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging for monitoring events."""

    @pytest.fixture
    def service(self):
        """Create monitoring service with temp audit file."""
        calculator = TripleBarrierCalculator()
        monitor = TripleBarrierMonitor(PositionTracker())
        api_client = Mock()
        tracker = PositionTracker()
        executor = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )

        audit_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        )
        audit_file.close()

        service_instance = PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client,
            audit_trail_path=audit_file.name
        )

        yield service_instance

        # Cleanup
        Path(audit_file.name).unlink(missing_ok=True)

    def test_log_position_entry_event(self, service):
        """Verify position entry event logging."""
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        service.on_position_entered(
            order_id="ORDER-123",
            entry_price=11800.00,
            quantity=5,
            direction="bullish",
            entry_time=entry_time
        )

        # Verify file exists and has entry
        with open(service._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + at least one entry
        assert len(rows) >= 2

        # Find entry event
        entry_rows = [r for r in rows if "ENTER" in r]
        assert len(entry_rows) >= 1


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_monitoring_completes_under_50ms(self):
        """Verify monitoring completes in < 50ms."""
        import time

        calculator = TripleBarrierCalculator()
        monitor = TripleBarrierMonitor(PositionTracker())
        api_client = Mock()
        tracker = PositionTracker()
        executor = TripleBarrierExitExecutor(
            api_client=api_client,
            position_tracker=tracker,
            calculator=calculator,
            monitor=monitor
        )

        service = PositionMonitoringService(
            calculator=calculator,
            monitor=monitor,
            executor=executor,
            position_tracker=tracker,
            api_client=api_client
        )

        # Add 10 positions
        entry_time = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        for i in range(10):
            service.on_position_entered(
                order_id="ORDER-{}".format(i),
                entry_price=11800.00 + i,
                quantity=5,
                direction="bullish",
                entry_time=entry_time
            )

        current_time = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)

        start_time = time.perf_counter()
        service.on_price_update(
            current_price=11805.00,
            current_time=current_time
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 50.0, (
            "Monitoring took {:.2f}ms, exceeds 50ms limit".format(
                elapsed_ms
            )
        )

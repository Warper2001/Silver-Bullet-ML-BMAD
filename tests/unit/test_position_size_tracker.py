"""Unit tests for Maximum Open Position Size Limit.

Tests position size tracking, limit enforcement, capacity calculation,
CSV logging, and integration with position tracker.
"""

import tempfile
from pathlib import Path
import pytest

from src.risk.position_size_tracker import PositionSizeTracker


class TestPositionSizeTrackerInit:
    """Test PositionSizeTracker initialization."""

    def test_init_with_valid_parameters(self):
        """Verify tracker initializes with valid parameters."""
        tracker = PositionSizeTracker(max_position_size=20)

        assert tracker._max_position_size == 20
        assert tracker._current_position_size == 0
        assert tracker._position_counts == {}

    def test_init_with_invalid_size(self):
        """Verify tracker raises error with non-positive limit."""
        with pytest.raises(ValueError):
            PositionSizeTracker(max_position_size=0)

        with pytest.raises(ValueError):
            PositionSizeTracker(max_position_size=-5)


class TestAddPosition:
    """Test adding positions."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_add_position_increases_size(self, tracker):
        """Verify adding position increases size."""
        tracker.add_position("ORDER-1", 5)

        assert tracker.get_current_position_size() == 5
        assert tracker.get_position_count() == 1

    def test_add_multiple_positions_accumulate(self, tracker):
        """Verify multiple positions accumulate correctly."""
        tracker.add_position("ORDER-1", 5)
        tracker.add_position("ORDER-2", 3)
        tracker.add_position("ORDER-3", 7)

        assert tracker.get_current_position_size() == 15

    def test_add_same_order_id_replaces(self, tracker):
        """Verify adding same order ID replaces position."""
        tracker.add_position("ORDER-1", 5)
        tracker.add_position("ORDER-1", 8)

        # Should replace, not accumulate
        assert tracker.get_current_position_size() == 8


class TestRemovePosition:
    """Test removing positions."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker with positions."""
        tracker = PositionSizeTracker(max_position_size=20)
        tracker.add_position("ORDER-1", 5)
        tracker.add_position("ORDER-2", 3)
        return tracker

    def test_remove_position_decreases_size(self, tracker):
        """Verify removing position decreases size."""
        tracker.remove_position("ORDER-1")

        assert tracker.get_current_position_size() == 3
        assert tracker.get_position_count() == 1

    def test_remove_last_position(self, tracker):
        """Verify removing last position resets to zero."""
        tracker.remove_position("ORDER-1")
        tracker.remove_position("ORDER-2")

        assert tracker.get_current_position_size() == 0
        assert tracker.get_position_count() == 0

    def test_remove_nonexistent_position(self, tracker):
        """Verify removing non-existent position is graceful."""
        initial_size = tracker.get_current_position_size()

        # Should not raise error
        tracker.remove_position("NONEXISTENT")

        # Size should be unchanged
        assert tracker.get_current_position_size() == initial_size


class TestCanOpenPosition:
    """Test checking if position can be opened."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_can_open_within_limit(self, tracker):
        """Verify can open when within limit."""
        tracker.add_position("ORDER-1", 10)

        assert tracker.can_open_position(5) is True
        assert tracker.can_open_position(10) is True

    def test_cannot_open_exceeds_limit(self, tracker):
        """Verify cannot open when would exceed limit."""
        tracker.add_position("ORDER-1", 15)

        assert tracker.can_open_position(6) is False
        assert tracker.can_open_position(10) is False

    def test_can_open_at_exact_capacity(self, tracker):
        """Verify can open at exact capacity."""
        tracker.add_position("ORDER-1", 18)

        # 2 contracts remaining
        assert tracker.can_open_position(2) is True
        assert tracker.can_open_position(3) is False


class TestGetAvailableCapacity:
    """Test available capacity calculation."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_capacity_at_start(self, tracker):
        """Verify full capacity available at start."""
        assert tracker.get_available_capacity() == 20

    def test_capacity_decreases_with_positions(self, tracker):
        """Verify capacity decreases as positions added."""
        tracker.add_position("ORDER-1", 5)
        tracker.add_position("ORDER-2", 3)

        assert tracker.get_available_capacity() == 12

    def test_capacity_zero_at_limit(self, tracker):
        """Verify capacity zero when at limit."""
        tracker.add_position("ORDER-1", 20)

        assert tracker.get_available_capacity() == 0

    def test_capacity_increases_with_removals(self, tracker):
        """Verify capacity increases when positions removed."""
        tracker.add_position("ORDER-1", 15)
        tracker.remove_position("ORDER-1")

        assert tracker.get_available_capacity() == 20


class TestIsAtCapacity:
    """Test capacity detection."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_not_at_capacity_initially(self, tracker):
        """Verify not at capacity initially."""
        assert tracker.is_at_capacity() is False

    def test_not_at_capacity_below_limit(self, tracker):
        """Verify not at capacity below limit."""
        tracker.add_position("ORDER-1", 15)

        assert tracker.is_at_capacity() is False

    def test_at_capacity_at_limit(self, tracker):
        """Verify at capacity when at limit."""
        tracker.add_position("ORDER-1", 20)

        assert tracker.is_at_capacity() is True


class TestGetPositionSummary:
    """Test position summary generation."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_summary_at_start(self, tracker):
        """Verify summary at start."""
        summary = tracker.get_position_summary()

        assert summary['max_position_size'] == 20
        assert summary['current_position_size'] == 0
        assert summary['available_capacity'] == 20
        assert summary['is_at_capacity'] is False

    def test_summary_with_positions(self, tracker):
        """Verify summary with positions."""
        tracker.add_position("ORDER-1", 5)
        tracker.add_position("ORDER-2", 3)

        summary = tracker.get_position_summary()

        assert summary['current_position_size'] == 8
        assert summary['available_capacity'] == 12
        assert summary['position_count'] == 2


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "position_size_tracking.csv")

        return PositionSizeTracker(
            max_position_size=20,
            audit_trail_path=audit_path
        )

    def test_csv_file_created(self, tracker):
        """Verify CSV file created on first position."""
        tracker.add_position("ORDER-1", 5)

        assert Path(tracker._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, tracker):
        """Verify CSV has all required columns."""
        tracker.add_position("ORDER-1", 5)

        import csv
        with open(tracker._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "order_id",
            "quantity",
            "current_position_size",
            "available_capacity",
            "is_at_capacity"
        ]

        assert headers == expected_headers


class TestLimitEnforcement:
    """Test limit enforcement."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_cannot_add_position_exceeding_limit(self, tracker):
        """Verify cannot add position that exceeds limit."""
        tracker.add_position("ORDER-1", 15)

        # Trying to add 10 contracts would exceed limit (25 > 20)
        assert tracker.can_open_position(10) is False

    def test_multiple_small_positions_within_limit(self, tracker):
        """Verify multiple small positions allowed within limit."""
        tracker.add_position("ORDER-1", 5)
        tracker.add_position("ORDER-2", 5)
        tracker.add_position("ORDER-3", 5)
        tracker.add_position("ORDER-4", 5)

        assert tracker.get_current_position_size() == 20
        assert tracker.is_at_capacity() is True

    def test_capacity_tracks_removals(self, tracker):
        """Verify capacity updates correctly after removals."""
        tracker.add_position("ORDER-1", 20)
        assert tracker.is_at_capacity() is True

        tracker.remove_position("ORDER-1")
        assert tracker.is_at_capacity() is False
        assert tracker.get_available_capacity() == 20


class TestIntegration:
    """Test integration with position tracker."""

    @pytest.fixture
    def tracker(self):
        """Create position size tracker."""
        return PositionSizeTracker(max_position_size=20)

    def test_signal_rejected_when_at_capacity(self, tracker):
        """Verify signal rejected when at capacity."""
        # Fill to capacity
        tracker.add_position("ORDER-1", 20)

        # Check signal would be rejected
        assert tracker.can_open_position(5) is False

    def test_position_size_updates_after_exit(self, tracker):
        """Verify position size updates after position exit."""
        tracker.add_position("ORDER-1", 10)
        tracker.add_position("ORDER-2", 5)

        assert tracker.get_current_position_size() == 15

        # Simulate position exit
        tracker.remove_position("ORDER-1")

        assert tracker.get_current_position_size() == 5
        assert tracker.get_available_capacity() == 15

"""Unit tests for News Event Filter.

Tests news event blackout period management, trading status checking,
CSV logging, and integration with trade execution pipeline.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.risk.news_event_filter import NewsEventFilter


class TestNewsEventFilterInit:
    """Test NewsEventFilter initialization."""

    def test_init_with_default_windows(self):
        """Verify filter initializes with default windows."""
        filter_obj = NewsEventFilter()

        assert filter_obj._pre_event_window_minutes == 30
        assert filter_obj._post_event_window_minutes == 30
        assert filter_obj._blackout_periods == []

    def test_init_with_custom_windows(self):
        """Verify filter initializes with custom windows."""
        filter_obj = NewsEventFilter(
            pre_event_window_minutes=15,
            post_event_window_minutes=45
        )

        assert filter_obj._pre_event_window_minutes == 15
        assert filter_obj._post_event_window_minutes == 45

    def test_init_with_audit_trail(self):
        """Verify filter initializes with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "news_filter.csv")

        filter_obj = NewsEventFilter(audit_trail_path=audit_path)

        assert filter_obj._audit_trail_path == audit_path


class TestAddBlackoutPeriod:
    """Test adding blackout periods."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter."""
        return NewsEventFilter(
            pre_event_window_minutes=30,
            post_event_window_minutes=30
        )

    def test_add_blackout_creates_window(self, filter_obj):
        """Verify adding blackout creates correct window."""
        event_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=event_time,
            event_duration_minutes=0
        )

        assert len(filter_obj._blackout_periods) == 1

        blackout = filter_obj._blackout_periods[0]
        assert blackout["event_name"] == "NFP"
        # 30 min before + 0 min event + 30 min after = 1 hour window
        start_expected = datetime(2026, 3, 7, 13, 0, 0, tzinfo=timezone.utc)
        end_expected = datetime(2026, 3, 7, 14, 0, 0, tzinfo=timezone.utc)
        assert blackout["start_time"] == start_expected
        assert blackout["end_time"] == end_expected

    def test_add_blackout_with_duration(self, filter_obj):
        """Verify adding blackout with event duration."""
        event_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        filter_obj.add_blackout_period(
            event_name="FOMC",
            event_time=event_time,
            event_duration_minutes=15
        )

        blackout = filter_obj._blackout_periods[0]
        # 30 min before + 15 min event + 30 min after = 75 min window
        start_expected = datetime(2026, 3, 7, 13, 0, 0, tzinfo=timezone.utc)
        end_expected = datetime(2026, 3, 7, 14, 15, 0, tzinfo=timezone.utc)
        assert blackout["start_time"] == start_expected
        assert blackout["end_time"] == end_expected

    def test_add_multiple_blackouts(self, filter_obj):
        """Verify adding multiple blackouts."""
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )
        filter_obj.add_blackout_period(
            event_name="CPI",
            event_time=datetime(
                2026, 3, 8, 12, 30, 0, tzinfo=timezone.utc
            )
        )

        assert len(filter_obj._blackout_periods) == 2


class TestIsTradingAllowed:
    """Test trading status checking."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter with blackout."""
        filter_obj = NewsEventFilter(
            pre_event_window_minutes=30,
            post_event_window_minutes=30
        )
        # Add NFP blackout from 13:00 to 14:00 UTC
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )
        return filter_obj

    def test_trading_allowed_when_no_blackout(self, filter_obj):
        """Verify trading allowed when not in blackout."""
        current_time = datetime(
            2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is True

    def test_trading_blocked_during_blackout(self, filter_obj):
        """Verify trading blocked during blackout."""
        # During blackout (13:30 UTC)
        current_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is False

    def test_trading_blocked_before_event(self, filter_obj):
        """Verify trading blocked in pre-event window."""
        # 20 minutes before event (in pre-event window)
        current_time = datetime(
            2026, 3, 7, 13, 10, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is False

    def test_trading_blocked_after_event(self, filter_obj):
        """Verify trading blocked in post-event window."""
        # 10 minutes after event (in post-event window)
        current_time = datetime(
            2026, 3, 7, 13, 40, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is False

    def test_trading_allowed_after_blackout_expires(self, filter_obj):
        """Verify trading allowed after blackout expires."""
        # After blackout ends (14:00 UTC)
        current_time = datetime(
            2026, 3, 7, 14, 1, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is True

    def test_trading_allowed_at_exact_boundary(self, filter_obj):
        """Verify trading allowed at exact boundary time."""
        # At exact boundary (14:00 UTC - blackout end)
        current_time = datetime(
            2026, 3, 7, 14, 0, 0, tzinfo=timezone.utc
        )

        # Should be allowed at exact boundary
        assert filter_obj.is_trading_allowed(current_time) is True


class TestGetBlackoutStatus:
    """Test blackout status retrieval."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter with blackout."""
        filter_obj = NewsEventFilter(
            pre_event_window_minutes=30,
            post_event_window_minutes=30
        )
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )
        return filter_obj

    def test_status_when_not_in_blackout(self, filter_obj):
        """Verify status when not in blackout."""
        current_time = datetime(
            2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc
        )

        status = filter_obj.get_blackout_status(current_time)

        assert status["is_blackout"] is False
        assert status["event_name"] is None
        assert status["blackout_start"] is None
        assert status["blackout_end"] is None
        assert status["minutes_remaining"] is None

    def test_status_when_in_blackout(self, filter_obj):
        """Verify status when in blackout."""
        current_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        status = filter_obj.get_blackout_status(current_time)

        assert status["is_blackout"] is True
        assert status["event_name"] == "NFP"
        start_expected = datetime(2026, 3, 7, 13, 0, 0, tzinfo=timezone.utc)
        end_expected = datetime(2026, 3, 7, 14, 0, 0, tzinfo=timezone.utc)
        assert status["blackout_start"] == start_expected
        assert status["blackout_end"] == end_expected
        assert status["minutes_remaining"] == 30

    def test_status_minutes_remaining_updates(self, filter_obj):
        """Verify minutes remaining updates correctly."""
        # 15 minutes into blackout
        current_time = datetime(
            2026, 3, 7, 13, 15, 0, tzinfo=timezone.utc
        )

        status = filter_obj.get_blackout_status(current_time)

        assert status["minutes_remaining"] == 45


class TestGetUpcomingEvents:
    """Test upcoming events list."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter with multiple events."""
        filter_obj = NewsEventFilter()
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )
        filter_obj.add_blackout_period(
            event_name="CPI",
            event_time=datetime(
                2026, 3, 8, 12, 30, 0, tzinfo=timezone.utc
            )
        )
        filter_obj.add_blackout_period(
            event_name="FOMC",
            event_time=datetime(
                2026, 3, 10, 18, 0, 0, tzinfo=timezone.utc
            )
        )
        return filter_obj

    def test_get_upcoming_events_returns_list(self, filter_obj):
        """Verify upcoming events returned."""
        current_time = datetime(
            2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc
        )

        events = filter_obj.get_upcoming_events(current_time, hours_ahead=48)

        # Within 48 hours: NFP and CPI only (FOMC is on March 10)
        assert len(events) == 2
        assert events[0]["event_name"] == "NFP"
        assert events[1]["event_name"] == "CPI"

    def test_get_upcoming_events_filters_by_time(self, filter_obj):
        """Verify upcoming events filtered by time window."""
        current_time = datetime(
            2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc
        )

        events = filter_obj.get_upcoming_events(current_time, hours_ahead=24)

        # Only NFP within 24 hours (CPI at 12:30 is exactly at boundary)
        assert len(events) == 1
        assert events[0]["event_name"] == "NFP"

    def test_get_upcoming_events_excludes_past(self, filter_obj):
        """Verify past events excluded from list."""
        # After NFP already occurred
        current_time = datetime(
            2026, 3, 7, 15, 0, 0, tzinfo=timezone.utc
        )

        events = filter_obj.get_upcoming_events(current_time, hours_ahead=48)

        # Should only include CPI (FOMC is beyond 48 hours)
        assert len(events) == 1
        assert events[0]["event_name"] == "CPI"


class TestRemoveExpiredBlackouts:
    """Test removal of expired blackout periods."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter with past and future events."""
        filter_obj = NewsEventFilter()
        # Add past event
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 1, 13, 30, 0, tzinfo=timezone.utc
            )
        )
        # Add future event
        filter_obj.add_blackout_period(
            event_name="CPI",
            event_time=datetime(
                2026, 3, 8, 12, 30, 0, tzinfo=timezone.utc
            )
        )
        return filter_obj

    def test_remove_expired_deletes_past_blackouts(self, filter_obj):
        """Verify expired blackouts removed."""
        current_time = datetime(
            2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc
        )

        filter_obj.remove_expired_blackouts(current_time)

        assert len(filter_obj._blackout_periods) == 1
        assert filter_obj._blackout_periods[0]["event_name"] == "CPI"

    def test_remove_expired_keeps_future_blackouts(self, filter_obj):
        """Verify future blackouts kept."""
        current_time = datetime(
            2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc
        )

        filter_obj.remove_expired_blackouts(current_time)

        # NFP blackout ended on March 1, so only CPI should remain
        assert len(filter_obj._blackout_periods) == 1
        assert filter_obj._blackout_periods[0]["event_name"] == "CPI"


class TestCalculateBlackoutWindow:
    """Test blackout window calculation."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter."""
        return NewsEventFilter(
            pre_event_window_minutes=30,
            post_event_window_minutes=30
        )

    def test_window_calculation_no_duration(self, filter_obj):
        """Verify window calculation without event duration."""
        event_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        start, end = filter_obj._calculate_blackout_window(
            event_time,
            event_duration_minutes=0
        )

        start_expected = datetime(2026, 3, 7, 13, 0, 0, tzinfo=timezone.utc)
        end_expected = datetime(2026, 3, 7, 14, 0, 0, tzinfo=timezone.utc)
        assert start == start_expected
        assert end == end_expected

    def test_window_calculation_with_duration(self, filter_obj):
        """Verify window calculation with event duration."""
        event_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        start, end = filter_obj._calculate_blackout_window(
            event_time,
            event_duration_minutes=15
        )

        start_expected = datetime(2026, 3, 7, 13, 0, 0, tzinfo=timezone.utc)
        end_expected = datetime(2026, 3, 7, 14, 15, 0, tzinfo=timezone.utc)
        assert start == start_expected
        assert end == end_expected

    def test_window_calculation_custom_windows(self, filter_obj):
        """Verify window calculation with custom windows."""
        event_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        # Create filter with different windows
        custom_filter = NewsEventFilter(
            pre_event_window_minutes=15,
            post_event_window_minutes=45
        )

        start, end = custom_filter._calculate_blackout_window(
            event_time,
            event_duration_minutes=0
        )

        start_expected = datetime(2026, 3, 7, 13, 15, 0, tzinfo=timezone.utc)
        end_expected = datetime(2026, 3, 7, 14, 15, 0, tzinfo=timezone.utc)
        assert start == start_expected
        assert end == end_expected


class TestOverlappingBlackouts:
    """Test overlapping blackout periods."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter."""
        return NewsEventFilter()

    def test_multiple_overlapping_blackouts(self, filter_obj):
        """Verify handling of overlapping blackouts."""
        # Add two events that overlap
        filter_obj.add_blackout_period(
            event_name="Event1",
            event_time=datetime(
                2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc
            )
        )
        filter_obj.add_blackout_period(
            event_name="Event2",
            event_time=datetime(
                2026, 3, 7, 10, 30, 0, tzinfo=timezone.utc
            )
        )

        # Check during overlap (10:30 UTC)
        current_time = datetime(
            2026, 3, 7, 10, 30, 0, tzinfo=timezone.utc
        )

        # Should be blocked (at least one blackout active)
        assert filter_obj.is_trading_allowed(current_time) is False


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "news_filter.csv")

        return NewsEventFilter(audit_trail_path=audit_path)

    def test_csv_file_created(self, filter_obj):
        """Verify CSV file created on first event."""
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )

        assert Path(filter_obj._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, filter_obj):
        """Verify CSV has all required columns."""
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )

        import csv
        with open(filter_obj._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "event_name",
            "is_blackout",
            "blackout_start",
            "blackout_end",
            "minutes_remaining"
        ]

        assert headers == expected_headers


class TestIntegration:
    """Test integration with trade pipeline."""

    @pytest.fixture
    def filter_obj(self):
        """Create news event filter with blackout."""
        filter_obj = NewsEventFilter()
        filter_obj.add_blackout_period(
            event_name="NFP",
            event_time=datetime(
                2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
            )
        )
        return filter_obj

    def test_signal_rejected_during_blackout(self, filter_obj):
        """Verify signal rejected during blackout."""
        # During blackout
        current_time = datetime(
            2026, 3, 7, 13, 30, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is False

    def test_signal_allowed_after_blackout(self, filter_obj):
        """Verify signal allowed after blackout."""
        # After blackout
        current_time = datetime(
            2026, 3, 7, 14, 30, 0, tzinfo=timezone.utc
        )

        assert filter_obj.is_trading_allowed(current_time) is True

    def test_signal_rejected_with_event_name(self, filter_obj):
        """Verify signal rejection includes event name."""
        # During blackout
        current_time = datetime(
            2026, 3, 7, 13, 45, 0, tzinfo=timezone.utc
        )

        status = filter_obj.get_blackout_status(current_time)

        assert status["is_blackout"] is True
        assert status["event_name"] == "NFP"
        assert status["minutes_remaining"] == 15

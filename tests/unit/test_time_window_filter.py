"""Unit tests for Time Window Restrictions.

Tests trading window definitions, time window filtering, signal blocking,
weekend filtering, and CSV audit trail logging.
"""

import csv
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from src.execution.time_window_filter import (
    TimeWindowFilter,
    TimeWindowResult,
    TradingWindows,
)


class TestTradingWindows:
    """Test trading window definitions."""

    @pytest.fixture
    def windows(self):
        """Create trading windows instance."""
        return TradingWindows()

    def test_morning_session_allowed(self, windows):
        """Verify morning session is allowed."""
        # 10:00 CT = 15:00 UTC (within 9:00-11:30 CT window)
        timestamp = datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is True

    def test_afternoon_session_allowed(self, windows):
        """Verify afternoon session is allowed."""
        # 14:00 CT = 19:00 UTC (within 12:30-15:30 CT window)
        timestamp = datetime(2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is True

    def test_market_open_blocked(self, windows):
        """Verify market open period is blocked."""
        # 8:45 CT = 13:45 UTC (within 8:30-9:00 CT window)
        timestamp = datetime(2026, 3, 17, 13, 45, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False

    def test_lunch_blocked(self, windows):
        """Verify lunch period is blocked."""
        # 12:00 CT = 17:00 UTC (within 11:30-12:30 CT window)
        timestamp = datetime(2026, 3, 17, 17, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False

    def test_market_close_blocked(self, windows):
        """Verify market close period is blocked."""
        # 15:45 CT = 20:45 UTC (within 15:30-16:00 CT window)
        timestamp = datetime(2026, 3, 17, 20, 45, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False

    def test_pre_market_blocked(self, windows):
        """Verify pre-market is blocked."""
        # 7:00 CT = 12:00 UTC (before 8:00 CT)
        timestamp = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False

    def test_after_hours_blocked(self, windows):
        """Verify after-hours is blocked."""
        # 17:00 CT = 22:00 UTC (after 16:00 CT)
        timestamp = datetime(2026, 3, 17, 22, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False

    def test_saturday_blocked(self, windows):
        """Verify Saturday is blocked."""
        # Saturday 10:00 CT = 15:00 UTC
        timestamp = datetime(2026, 3, 21, 15, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False

    def test_sunday_blocked(self, windows):
        """Verify Sunday is blocked."""
        # Sunday 10:00 CT = 15:00 UTC
        timestamp = datetime(2026, 3, 22, 15, 0, 0, tzinfo=timezone.utc)
        result = windows.is_trading_allowed(timestamp)

        assert result is False


class TestTimeWindowFilter:
    """Test time window filtering."""

    @pytest.fixture
    def filter(self):
        """Create time window filter."""
        return TimeWindowFilter()

    def test_allowed_time_returns_allowed_result(self, filter):
        """Verify allowed time returns allowed result."""
        # 10:00 CT = 15:00 UTC
        timestamp = datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is True
        assert result.window_name == "MORNING_SESSION"

    def test_blocked_time_returns_blocked_result(self, filter):
        """Verify blocked time returns blocked result."""
        # 8:45 CT = 13:45 UTC (market open)
        timestamp = datetime(2026, 3, 17, 13, 45, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is False
        assert result.window_name == "MARKET_OPEN"
        assert "Market open" in result.reason

    def test_time_until_open_calculation(self, filter):
        """Verify time until next open window."""
        # 8:00 CT = 13:00 UTC (pre-market, 1 hour until morning session)
        timestamp = datetime(2026, 3, 17, 13, 0, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is False
        assert result.time_until_open is not None
        # Should be ~1 hour until 14:00 UTC
        assert 3500 <= result.time_until_open.total_seconds() <= 3700

    def test_time_until_close_calculation(self, filter):
        """Verify time until window close."""
        # 11:00 CT = 16:00 UTC (morning session, 30 min until lunch)
        timestamp = datetime(2026, 3, 17, 16, 0, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is True
        assert result.time_until_close is not None
        # Should be ~30 min until 16:30 UTC
        assert 1700 <= result.time_until_close.total_seconds() <= 1900

    def test_weekend_blocked(self, filter):
        """Verify weekend blocking."""
        # Saturday
        timestamp = datetime(2026, 3, 22, 15, 0, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is False
        assert "weekend" in result.reason.lower()

    def test_exact_window_start_allowed(self, filter):
        """Verify exact window start time is allowed."""
        # 9:00 CT = 14:00 UTC (morning session start)
        timestamp = datetime(2026, 3, 17, 14, 0, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is True

    def test_exact_window_end_allowed(self, filter):
        """Verify exact window end time is allowed."""
        # 11:30 CT = 16:30 UTC (morning session end)
        timestamp = datetime(2026, 3, 17, 16, 30, 0, tzinfo=timezone.utc)

        result = filter.check_time_window(timestamp)

        assert result.allowed is True


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging for time window events."""

    @pytest.fixture
    def filter(self):
        """Create filter with temp audit file."""
        audit_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        )
        audit_file.close()

        filter_instance = TimeWindowFilter(
            audit_trail_path=audit_file.name
        )

        yield filter_instance

        # Cleanup
        Path(audit_file.name).unlink(missing_ok=True)

    def test_log_filter_event(self, filter):
        """Verify time window filter event logging."""
        timestamp = datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)

        filter.check_time_window(timestamp)

        # Verify file exists and has entry
        with open(filter._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + at least one entry
        assert len(rows) >= 2

        # Find filter event
        filter_rows = [r for r in rows if "FILTER" in r]
        assert len(filter_rows) >= 1


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_time_check_completes_under_1ms(self):
        """Verify time window check completes in < 1ms."""
        import time

        filter = TimeWindowFilter()

        timestamp = datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)

        start_time = time.perf_counter()
        result = filter.check_time_window(timestamp)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.allowed is True
        assert elapsed_ms < 1.0, (
            "Time check took {:.2f}ms, exceeds 1ms limit".format(
                elapsed_ms
            )
        )

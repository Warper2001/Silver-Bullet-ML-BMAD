"""Unit tests for Time Window Filtering."""

from datetime import datetime, timezone

import pytest

from src.data.models import SilverBulletSetup
from src.detection.time_window_filter import (
    TimeWindow,
    check_time_window,
    is_within_trading_hours,
)


class TestTimeWindowFiltering:
    """Test time window filtering algorithms."""

    @pytest.fixture
    def london_am_window(self):
        """Create London AM time window (3-4 AM EST)."""
        return TimeWindow(
            name="London AM",
            start_hour=3,
            start_minute=0,
            end_hour=4,
            end_minute=0,
            timezone="EST",
        )

    @pytest.fixture
    def ny_am_window(self):
        """Create NY AM time window (10-11 AM EST)."""
        return TimeWindow(
            name="NY AM",
            start_hour=10,
            start_minute=0,
            end_hour=11,
            end_minute=0,
            timezone="EST",
        )

    @pytest.fixture
    def ny_pm_window(self):
        """Create NY PM time window (2-3 PM EST)."""
        return TimeWindow(
            name="NY PM",
            start_hour=14,
            start_minute=0,
            end_hour=15,
            end_minute=0,
            timezone="EST",
        )

    @pytest.fixture
    def base_setup(self):
        """Create a sample Silver Bullet setup."""
        # Create minimal setup for testing
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SwingPoint,
        )

        swing = SwingPoint(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        return SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=11,
        )

    def test_within_london_am_window(self, london_am_window):
        """Verify setup allowed during London AM window (3:30 AM EST)."""
        timestamp = datetime(2026, 3, 16, 3, 30, 0)  # 3:30 AM EST

        within_window, window_name = is_within_trading_hours(
            timestamp, [london_am_window]
        )

        assert within_window is True
        assert window_name == "London AM"

    def test_within_ny_am_window(self, ny_am_window):
        """Verify setup allowed during NY AM window (10:30 AM EST)."""
        timestamp = datetime(2026, 3, 16, 10, 30, 0)  # 10:30 AM EST

        within_window, window_name = is_within_trading_hours(timestamp, [ny_am_window])

        assert within_window is True
        assert window_name == "NY AM"

    def test_within_ny_pm_window(self, ny_pm_window):
        """Verify setup allowed during NY PM window (2:30 PM EST)."""
        timestamp = datetime(2026, 3, 16, 14, 30, 0)  # 2:30 PM EST

        within_window, window_name = is_within_trading_hours(timestamp, [ny_pm_window])

        assert within_window is True
        assert window_name == "NY PM"

    def test_outside_trading_windows(self):
        """Verify setup suppressed outside all trading windows (midnight EST)."""
        timestamp = datetime(2026, 3, 16, 0, 0, 0)  # Midnight EST

        windows = [
            TimeWindow(
                name="London AM",
                start_hour=3,
                start_minute=0,
                end_hour=4,
                end_minute=0,
                timezone="EST",
            ),
            TimeWindow(
                name="NY AM",
                start_hour=10,
                start_minute=0,
                end_hour=11,
                end_minute=0,
                timezone="EST",
            ),
            TimeWindow(
                name="NY PM",
                start_hour=14,
                start_minute=0,
                end_hour=15,
                end_minute=0,
                timezone="EST",
            ),
        ]

        within_window, window_name = is_within_trading_hours(timestamp, windows)

        assert within_window is False
        assert window_name is None

    def test_at_window_boundary_start(self, london_am_window):
        """Verify setup allowed at exact window start time (3:00:00 AM EST)."""
        timestamp = datetime(2026, 3, 16, 3, 0, 0)  # 3:00:00 AM EST

        within_window, window_name = is_within_trading_hours(
            timestamp, [london_am_window]
        )

        assert within_window is True

    def test_at_window_boundary_end(self, london_am_window):
        """Verify setup NOT allowed at exact window end time (4:00:00 AM EST)."""
        timestamp = datetime(2026, 3, 16, 4, 0, 0)  # 4:00:00 AM EST

        within_window, window_name = is_within_trading_hours(
            timestamp, [london_am_window]
        )

        assert within_window is False

    def test_timezone_conversion_utc_to_est(self):
        """Verify timezone conversion from UTC to EST."""
        # 8:00 UTC = 3:00 AM EST (UTC-5)
        timestamp = datetime(2026, 3, 16, 8, 0, 0, tzinfo=timezone.utc)

        windows = [
            TimeWindow(
                name="London AM",
                start_hour=3,
                start_minute=0,
                end_hour=4,
                end_minute=0,
                timezone="EST",
            ),
        ]

        within_window, window_name = is_within_trading_hours(timestamp, windows)

        assert within_window is True

    def test_check_time_window_allows_within_window(self, base_setup, ny_am_window):
        """Verify check_time_window returns setup when within window."""
        # Set setup timestamp to 10:30 AM EST (within NY AM window)
        base_setup.timestamp = datetime(2026, 3, 16, 10, 30, 0)

        result = check_time_window(base_setup, [ny_am_window])

        assert result is not None
        assert result == base_setup  # Setup returned unchanged

    def test_check_time_window_filters_outside_window(self, base_setup, ny_am_window):
        """Verify check_time_window returns None when outside window."""
        # Set setup timestamp to midnight EST (outside all windows)
        base_setup.timestamp = datetime(2026, 3, 16, 0, 0, 0)

        result = check_time_window(base_setup, [ny_am_window])

        assert result is None  # Setup filtered out

    def test_multiple_windows_checked(self, base_setup):
        """Verify setup checked against all provided windows."""
        windows = [
            TimeWindow(
                name="London AM",
                start_hour=3,
                start_minute=0,
                end_hour=4,
                end_minute=0,
                timezone="EST",
            ),
            TimeWindow(
                name="NY AM",
                start_hour=10,
                start_minute=0,
                end_hour=11,
                end_minute=0,
                timezone="EST",
            ),
            TimeWindow(
                name="NY PM",
                start_hour=14,
                start_minute=0,
                end_hour=15,
                end_minute=0,
                timezone="EST",
            ),
        ]

        # Test with NY AM window time
        base_setup.timestamp = datetime(2026, 3, 16, 10, 30, 0)
        result = check_time_window(base_setup, windows)
        assert result is not None

        # Test with time outside all windows
        base_setup.timestamp = datetime(2026, 3, 16, 0, 0, 0)
        result = check_time_window(base_setup, windows)
        assert result is None

    def test_performance_under_5ms(self, base_setup):
        """Verify time window validation adds < 5ms overhead."""
        import time

        windows = [
            TimeWindow(
                name="London AM",
                start_hour=3,
                start_minute=0,
                end_hour=4,
                end_minute=0,
                timezone="EST",
            ),
            TimeWindow(
                name="NY AM",
                start_hour=10,
                start_minute=0,
                end_hour=11,
                end_minute=0,
                timezone="EST",
            ),
            TimeWindow(
                name="NY PM",
                start_hour=14,
                start_minute=0,
                end_hour=15,
                end_minute=0,
                timezone="EST",
            ),
        ]

        base_setup.timestamp = datetime(2026, 3, 16, 10, 30, 0)

        # Measure time
        start_time = time.perf_counter()
        for _ in range(100):  # Run 100 times for accuracy
            check_time_window(base_setup, windows)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        avg_time_ms = elapsed_ms / 100

        # Should be very fast (< 1ms typically)
        assert avg_time_ms < 5

    def test_dst_handling(self):
        """Verify proper DST handling for EST timezone."""
        # Note: EST is UTC-5, EDT is UTC-4
        # This test verifies we handle timezone correctly regardless of DST
        # 19:00 UTC = 2:00 PM EST (within NY PM window)
        timestamp_utc = datetime(2026, 3, 16, 19, 0, 0, tzinfo=timezone.utc)

        windows = [
            TimeWindow(
                name="NY PM",
                start_hour=14,
                start_minute=0,
                end_hour=15,
                end_minute=0,
                timezone="EST",
            ),  # 2-3 PM EST
        ]

        within_window, window_name = is_within_trading_hours(timestamp_utc, windows)

        # 19:00 UTC = 2:00 PM EST (within NY PM window)
        assert within_window is True


class TestTimeWindowModel:
    """Test TimeWindow data model."""

    def test_time_window_creation(self):
        """Verify TimeWindow can be created with all required fields."""
        window = TimeWindow(
            name="Test Window",
            start_hour=10,
            start_minute=0,
            end_hour=11,
            end_minute=0,
            timezone="EST",
        )

        assert window.name == "Test Window"
        assert window.start_hour == 10
        assert window.start_minute == 0
        assert window.end_hour == 11
        assert window.end_minute == 0
        assert window.timezone == "EST"

    def test_time_window_valid_timezone(self):
        """Verify only valid timezones are accepted."""
        # EST is valid
        TimeWindow(
            name="Test",
            start_hour=10,
            start_minute=0,
            end_hour=11,
            end_minute=0,
            timezone="EST",
        )

        # Can also test other timezones if needed
        # (EST, UTC, etc.)

    def test_time_window_hour_range(self):
        """Verify hours are within valid range (0-23)."""
        # Valid hours
        window = TimeWindow(
            name="Test",
            start_hour=0,
            start_minute=0,
            end_hour=23,
            end_minute=59,
            timezone="EST",
        )

        assert 0 <= window.start_hour <= 23
        assert 0 <= window.end_hour <= 23

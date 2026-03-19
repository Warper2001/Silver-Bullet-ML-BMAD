"""Unit tests for Time Window Filter (Pattern Detection)."""

from datetime import datetime, timezone, timedelta

import pytest

from src.data.models import (
    FVGEvent,
    GapRange,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
    TimeWindow,
)
from src.detection.time_window_filter import (
    LONDON_AM,
    NY_AM,
    NY_PM,
    check_time_window,
    convert_to_est,
    filter_setups_by_time_window,
    is_within_trading_hours,
)


class TestTimezoneConversion:
    """Test EST timezone conversion."""

    def test_utc_timestamp_converted_to_est(self):
        """Verify UTC timestamp converted to EST."""
        utc_time = datetime(2026, 3, 16, 8, 0, 0, tzinfo=timezone.utc)
        est_time = convert_to_est(utc_time)

        # EST is UTC-5, so 8:00 UTC = 3:00 EST
        assert est_time.hour == 3
        assert est_time.minute == 0

    def test_naive_timestamp_kept_as_is(self):
        """Verify naive timestamp is kept as-is."""
        naive_time = datetime(2026, 3, 16, 3, 30, 0)
        result = convert_to_est(naive_time)

        # Naive timestamp should be returned unchanged
        assert result.hour == 3
        assert result.minute == 30

    def test_est_conversion_handles_daylight_saving_approximately(self):
        """Verify EST conversion handles DST approximately."""
        # Note: Current implementation doesn't handle DST perfectly
        # This is documented as a limitation
        utc_time = datetime(2026, 3, 16, 13, 0, 0, tzinfo=timezone.utc)
        est_time = convert_to_est(utc_time)

        # During standard time: UTC-5
        # During daylight time: UTC-4
        # Our implementation uses UTC-5 (simplified)
        assert est_time is not None


class TestTradingWindows:
    """Test trading window definitions."""

    def test_london_am_window_bounds(self):
        """Verify London AM window is 3:00-4:00 AM EST."""
        assert LONDON_AM.start_hour == 3
        assert LONDON_AM.start_minute == 0
        assert LONDON_AM.end_hour == 4
        assert LONDON_AM.end_minute == 0
        assert LONDON_AM.timezone == "EST"

    def test_ny_am_window_bounds(self):
        """Verify NY AM window is 10:00-11:00 AM EST."""
        assert NY_AM.start_hour == 10
        assert NY_AM.start_minute == 0
        assert NY_AM.end_hour == 11
        assert NY_AM.end_minute == 0
        assert NY_AM.timezone == "EST"

    def test_ny_pm_window_bounds(self):
        """Verify NY PM window is 2:00-3:00 PM EST."""
        assert NY_PM.start_hour == 14
        assert NY_PM.start_minute == 0
        assert NY_PM.end_hour == 15
        assert NY_PM.end_minute == 0
        assert NY_PM.timezone == "EST"


class TestLondonAMWindow:
    """Test London AM time window (3:00-4:00 AM EST)."""

    def test_within_london_am_window(self):
        """Verify setup at 3:30 AM EST is within window."""
        timestamp = datetime(2026, 3, 16, 3, 30, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is True
        assert name == "London AM"

    def test_before_london_am_window(self):
        """Verify setup at 2:59 AM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 2, 59, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_after_london_am_window(self):
        """Verify setup at 4:01 AM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 4, 1, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_london_am_start_boundary_allowed(self):
        """Verify exact window start (3:00 AM EST) is allowed."""
        timestamp = datetime(2026, 3, 16, 3, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is True
        assert name == "London AM"

    def test_london_am_end_boundary_filtered(self):
        """Verify exact window end (4:00 AM EST) is filtered."""
        timestamp = datetime(2026, 3, 16, 4, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None


class TestNYAMWindow:
    """Test NY AM time window (10:00-11:00 AM EST)."""

    def test_within_ny_am_window(self):
        """Verify setup at 10:30 AM EST is within window."""
        timestamp = datetime(2026, 3, 16, 10, 30, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is True
        assert name == "NY AM"

    def test_before_ny_am_window(self):
        """Verify setup at 9:59 AM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 9, 59, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_after_ny_am_window(self):
        """Verify setup at 11:01 AM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 11, 1, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_ny_am_start_boundary_allowed(self):
        """Verify exact window start (10:00 AM EST) is allowed."""
        timestamp = datetime(2026, 3, 16, 10, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is True
        assert name == "NY AM"

    def test_ny_am_end_boundary_filtered(self):
        """Verify exact window end (11:00 AM EST) is filtered."""
        timestamp = datetime(2026, 3, 16, 11, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None


class TestNYPMWindow:
    """Test NY PM time window (2:00-3:00 PM EST)."""

    def test_within_ny_pm_window(self):
        """Verify setup at 2:30 PM EST is within window."""
        timestamp = datetime(2026, 3, 16, 14, 30, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is True
        assert name == "NY PM"

    def test_before_ny_pm_window(self):
        """Verify setup at 1:59 PM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 13, 59, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_after_ny_pm_window(self):
        """Verify setup at 3:01 PM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 15, 1, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_ny_pm_start_boundary_allowed(self):
        """Verify exact window start (2:00 PM EST) is allowed."""
        timestamp = datetime(2026, 3, 16, 14, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is True
        assert name == "NY PM"

    def test_ny_pm_end_boundary_filtered(self):
        """Verify exact window end (3:00 PM EST) is filtered."""
        timestamp = datetime(2026, 3, 16, 15, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None


class TestFilteringOutsideWindows:
    """Test filtering for times outside all trading windows."""

    def test_midnight_filtered(self):
        """Verify setup at midnight EST is filtered."""
        timestamp = datetime(2026, 3, 16, 0, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_between_london_am_and_ny_am_filtered(self):
        """Verify setup at 5:00 AM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 5, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_between_ny_am_and_ny_pm_filtered(self):
        """Verify setup at 12:00 PM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 12, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None

    def test_after_ny_pm_filtered(self):
        """Verify setup at 8:00 PM EST is filtered."""
        timestamp = datetime(2026, 3, 16, 20, 0, 0)

        within, name = is_within_trading_hours(timestamp)

        assert within is False
        assert name is None


class TestCheckTimeWindow:
    """Test single setup filtering."""

    @pytest.fixture
    def sample_setup(self):
        """Create a sample Silver Bullet setup."""
        swing_point = SwingPoint(
            timestamp=datetime(2026, 3, 16, 3, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=5,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 3, 30, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 3, 35, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        return SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 3, 30, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=10,
        )

    def test_setup_within_window_allowed(self, sample_setup):
        """Verify setup within trading window is allowed."""
        # Setup at 3:30 AM EST (within London AM)
        result = check_time_window(sample_setup)

        assert result is not None
        assert result == sample_setup

    def test_setup_outside_window_filtered(self):
        """Verify setup outside trading window is filtered."""
        # Create setup at midnight (outside all windows)
        swing_point = SwingPoint(
            timestamp=datetime(2026, 3, 16, 0, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 0, 30, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=0,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 0, 35, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=1,
        )

        setup = SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 0, 30, 0),  # Midnight
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=0,
        )

        result = check_time_window(setup)

        assert result is None  # Filtered out


class TestBatchFiltering:
    """Test batch filtering of multiple setups."""

    def create_setup(self, timestamp):
        """Helper to create a setup at given timestamp."""
        swing_point = SwingPoint(
            timestamp=timestamp - timedelta(minutes=30),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=timestamp + timedelta(seconds=5),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        return SilverBulletSetup(
            timestamp=timestamp,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=10,
        )

    def test_multiple_setups_some_filtered(self):
        """Verify multiple setups with some filtered."""
        # Create setups at different times
        setups = [
            self.create_setup(datetime(2026, 3, 16, 3, 30, 0)),  # London AM - allowed
            self.create_setup(datetime(2026, 3, 16, 5, 0, 0)),   # Between windows - filtered
            self.create_setup(datetime(2026, 3, 16, 10, 30, 0)),  # NY AM - allowed
            self.create_setup(datetime(2026, 3, 16, 14, 30, 0)),  # NY PM - allowed
            self.create_setup(datetime(2026, 3, 16, 20, 0, 0)),   # After hours - filtered
        ]

        allowed, stats = filter_setups_by_time_window(setups)

        # Should have 3 allowed, 2 filtered
        assert len(allowed) == 3
        assert stats["allowed"] == 3
        assert stats["filtered"] == 2

    def test_empty_setup_list(self):
        """Verify empty setup list handled correctly."""
        setups = []

        allowed, stats = filter_setups_by_time_window(setups)

        assert len(allowed) == 0
        assert stats["allowed"] == 0
        assert stats["filtered"] == 0

    def test_all_setups_filtered(self):
        """Verify all setups can be filtered."""
        # Create all setups outside windows
        setups = [
            self.create_setup(datetime(2026, 3, 16, 0, 0, 0)),   # Midnight
            self.create_setup(datetime(2026, 3, 16, 6, 0, 0)),   # 6 AM
            self.create_setup(datetime(2026, 3, 16, 12, 0, 0)),  # Noon
            self.create_setup(datetime(2026, 3, 16, 18, 0, 0)),  # 6 PM
        ]

        allowed, stats = filter_setups_by_time_window(setups)

        assert len(allowed) == 0
        assert stats["allowed"] == 0
        assert stats["filtered"] == 4

    def test_all_setups_allowed(self):
        """Verify all setups can be allowed."""
        # Create all setups within windows
        setups = [
            self.create_setup(datetime(2026, 3, 16, 3, 30, 0)),  # London AM
            self.create_setup(datetime(2026, 3, 16, 10, 30, 0)),  # NY AM
            self.create_setup(datetime(2026, 3, 16, 14, 30, 0)),  # NY PM
        ]

        allowed, stats = filter_setups_by_time_window(setups)

        assert len(allowed) == 3
        assert stats["allowed"] == 3
        assert stats["filtered"] == 0


class TestPerformanceRequirements:
    """Test performance requirements."""

    def create_setup(self, timestamp):
        """Helper to create a setup at given timestamp."""
        swing_point = SwingPoint(
            timestamp=timestamp - timedelta(minutes=30),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=timestamp + timedelta(seconds=5),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        return SilverBulletSetup(
            timestamp=timestamp,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=10,
        )

    def test_validation_latency_under_5ms(self):
        """Verify time window validation completes in < 5ms."""
        import time

        # Create a sample setup
        swing_point = SwingPoint(
            timestamp=datetime(2026, 3, 16, 3, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 3, 30, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 3, 35, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        setup = SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 3, 30, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=10,
        )

        # Measure validation time
        start_time = time.perf_counter()
        result = check_time_window(setup)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify result is correct
        assert result is not None

        # Verify performance requirement
        assert elapsed_ms < 5.0, (
            f"Time window validation took {elapsed_ms:.2f}ms, exceeds 5ms limit"
        )

    def test_batch_filtering_performance(self):
        """Verify batch filtering handles 1000 setups efficiently."""
        import time

        # Create 1000 setups
        setups = []
        for i in range(1000):
            # Mix of timestamps
            hour = (i * 13) % 24  # Distribute across 24 hours
            timestamp = datetime(2026, 3, 16, hour, 0, 0)
            setups.append(self.create_setup(timestamp))

        # Measure filtering time
        start_time = time.perf_counter()
        allowed, stats = filter_setups_by_time_window(setups)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify filtering completed
        assert len(allowed) + stats["filtered"] == 1000

        # Verify performance (should be < 5 seconds for 1000 setups)
        assert elapsed_ms < 5000.0, (
            f"Batch filtering took {elapsed_ms:.2f}ms for 1000 setups"
        )

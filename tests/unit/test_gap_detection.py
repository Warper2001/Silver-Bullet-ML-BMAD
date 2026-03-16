"""Unit tests for gap detection and forward-fill."""

from datetime import datetime, timedelta

import pytest

from src.data.gap_detection import GapDetector, GapStatistics
from src.data.models import DollarBar


class TestGapStatistics:
    """Test GapStatistics data class."""

    def test_initialization(self) -> None:
        """Test GapStatistics initializes with default values."""
        stats = GapStatistics()

        assert stats.short_gap_count == 0
        assert stats.short_gap_duration_total == 0.0
        assert stats.extended_gap_count == 0
        assert stats.extended_gap_duration_total == 0.0
        assert stats.extended_gap_log == []

    def test_total_gap_count(self) -> None:
        """Test total_gap_count property."""
        stats = GapStatistics(
            short_gap_count=3,
            extended_gap_count=2,
        )

        assert stats.total_gap_count == 5

    def test_total_gap_duration(self) -> None:
        """Test total_gap_duration property."""
        stats = GapStatistics(
            short_gap_duration_total=120.0,
            extended_gap_duration_total=600.0,
        )

        assert stats.total_gap_duration == 720.0

    def test_average_gap_duration(self) -> None:
        """Test average_gap_duration property."""
        stats = GapStatistics(
            short_gap_count=2,
            extended_gap_count=1,
            short_gap_duration_total=180.0,
            extended_gap_duration_total=600.0,
        )

        assert stats.average_gap_duration == 260.0  # 780 / 3

    def test_average_gap_duration_no_gaps(self) -> None:
        """Test average_gap_duration when no gaps occurred."""
        stats = GapStatistics()

        assert stats.average_gap_duration == 0.0


class TestGapDetectorInitialization:
    """Test GapDetector class initialization."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {
            "validated": asyncio.Queue(),
            "gap_filled": asyncio.Queue(),
        }

    def test_initialization_with_defaults(self, queues) -> None:
        """Test GapDetector initializes with default thresholds."""
        detector = GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

        assert detector.staleness_threshold_seconds == 30
        assert detector.forward_fill_limit_seconds == 300

    def test_initialization_with_custom_thresholds(self, queues) -> None:
        """Test GapDetector initializes with custom thresholds."""
        detector = GapDetector(
            queues["validated"],
            queues["gap_filled"],
            staleness_threshold_seconds=60,
            forward_fill_limit_seconds=600,
        )

        assert detector.staleness_threshold_seconds == 60
        assert detector.forward_fill_limit_seconds == 600

    def test_initial_statistics(self, queues) -> None:
        """Test GapDetector initializes with zero statistics."""
        detector = GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

        assert detector.short_gap_count == 0
        assert detector.extended_gap_count == 0
        assert detector.total_gap_count == 0


class TestStalenessDetection:
    """Test staleness detection logic."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {
            "validated": asyncio.Queue(),
            "gap_filled": asyncio.Queue(),
        }

    @pytest.fixture
    def detector(self, queues):
        """Create GapDetector instance."""
        return GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

    @pytest.mark.asyncio
    async def test_staleness_detected_after_timeout(self, detector, queues) -> None:
        """Test staleness detected after timeout expires."""
        # Put a bar to establish baseline
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )
        queues["validated"].put_nowait(bar)

        # Wait for staleness timeout (30 seconds)
        # Note: We'll test this with a much shorter timeout in actual implementation
        # This test will be updated once we implement the logic

    @pytest.mark.asyncio
    async def test_no_staleness_before_timeout(self, detector, queues) -> None:
        """Test no staleness detected before timeout expires."""
        # This test will verify that bars received before timeout don't trigger gap detection
        pass


class TestGapHandling:
    """Test gap handling logic."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {
            "validated": asyncio.Queue(),
            "gap_filled": asyncio.Queue(),
        }

    @pytest.fixture
    def detector(self, queues):
        """Create GapDetector instance."""
        return GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

    @pytest.mark.asyncio
    async def test_short_gap_triggers_forward_fill(self, detector) -> None:
        """Test short gap (< 5 min) triggers forward-fill."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=120)  # 2 minutes

        last_bar = DollarBar(
            timestamp=gap_start,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        filled_bars = await detector._forward_fill_bars(
            gap_start,
            gap_end,
            last_bar,
        )

        # Should create 24 bars (120 seconds / 5 seconds per bar)
        assert len(filled_bars) == 24
        for bar in filled_bars:
            assert bar.is_forward_filled is True
            assert bar.volume == 0
            assert bar.notional_value == 0
            assert bar.open == last_bar.close
            assert bar.high == last_bar.close
            assert bar.low == last_bar.close
            assert bar.close == last_bar.close

    @pytest.mark.asyncio
    async def test_extended_gap_does_not_forward_fill(self, detector) -> None:
        """Test extended gap (≥ 5 min) does NOT forward-fill."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=600)  # 10 minutes

        # _handle_gap checks gap duration and doesn't call _forward_fill_bars for extended gaps
        initial_count = detector.extended_gap_count
        await detector._handle_gap(gap_start, gap_end)

        # Should increment extended gap count
        assert detector.extended_gap_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_gap_statistics_tracked_correctly(self, detector) -> None:
        """Test gap statistics tracked correctly."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=120)

        await detector._handle_gap(gap_start, gap_end)

        # Short gap should be tracked
        assert detector.short_gap_count == 1
        assert detector.extended_gap_count == 0

    @pytest.mark.asyncio
    async def test_gap_timestamps_logged(self, detector) -> None:
        """Test gap start/end timestamps logged correctly."""
        gap_start = datetime(2026, 3, 15, 10, 0, 0)
        gap_end = datetime(2026, 3, 15, 10, 2, 30)

        await detector._handle_gap(gap_start, gap_end)

        # Verify gap duration logged (150 seconds)
        assert detector.short_gap_duration_total == 150.0


class TestForwardFillLogic:
    """Test forward-fill logic."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {
            "validated": asyncio.Queue(),
            "gap_filled": asyncio.Queue(),
        }

    @pytest.fixture
    def detector(self, queues):
        """Create GapDetector instance."""
        return GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

    @pytest.mark.asyncio
    async def test_forward_fill_uses_last_known_close_price(self, detector) -> None:
        """Test forward-filled bars use last-known close price."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=30)  # 6 bars

        last_bar = DollarBar(
            timestamp=gap_start,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        filled_bars = await detector._forward_fill_bars(
            gap_start,
            gap_end,
            last_bar,
        )

        assert len(filled_bars) == 6
        for bar in filled_bars:
            assert bar.close == 4523.75

    @pytest.mark.asyncio
    async def test_forward_fill_zero_volume_notional(self, detector) -> None:
        """Test forward-filled bars have volume=0 and notional_value=0."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=30)

        last_bar = DollarBar(
            timestamp=gap_start,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        filled_bars = await detector._forward_fill_bars(
            gap_start,
            gap_end,
            last_bar,
        )

        for bar in filled_bars:
            assert bar.volume == 0
            assert bar.notional_value == 0

    @pytest.mark.asyncio
    async def test_forward_fill_marked_correctly(self, detector) -> None:
        """Test forward-filled bars have is_forward_filled=True."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=30)

        last_bar = DollarBar(
            timestamp=gap_start,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        filled_bars = await detector._forward_fill_bars(
            gap_start,
            gap_end,
            last_bar,
        )

        for bar in filled_bars:
            assert bar.is_forward_filled is True

    @pytest.mark.asyncio
    async def test_forward_fill_5_second_intervals(self, detector) -> None:
        """Test forward-fill uses 5-second intervals."""
        gap_start = datetime(2026, 3, 15, 10, 0, 0)
        gap_end = datetime(2026, 3, 15, 10, 0, 30)  # 30 seconds

        last_bar = DollarBar(
            timestamp=gap_start,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        filled_bars = await detector._forward_fill_bars(
            gap_start,
            gap_end,
            last_bar,
        )

        # Should create 6 bars at 5-second intervals
        assert len(filled_bars) == 6

        # Verify timestamps
        expected_timestamps = [
            datetime(2026, 3, 15, 10, 0, 5),
            datetime(2026, 3, 15, 10, 0, 10),
            datetime(2026, 3, 15, 10, 0, 15),
            datetime(2026, 3, 15, 10, 0, 20),
            datetime(2026, 3, 15, 10, 0, 25),
            datetime(2026, 3, 15, 10, 0, 30),
        ]

        actual_timestamps = [bar.timestamp for bar in filled_bars]
        assert actual_timestamps == expected_timestamps

    @pytest.mark.asyncio
    async def test_forward_fill_correct_bar_count(self, detector) -> None:
        """Test forward-fill creates correct number of bars for gap duration."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=125)  # 25 bars

        last_bar = DollarBar(
            timestamp=gap_start,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        filled_bars = await detector._forward_fill_bars(
            gap_start,
            gap_end,
            last_bar,
        )

        # 125 seconds / 5 seconds = 25 bars
        assert len(filled_bars) == 25


class TestGapStatisticsTracking:
    """Test gap statistics tracking."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {
            "validated": asyncio.Queue(),
            "gap_filled": asyncio.Queue(),
        }

    @pytest.fixture
    def detector(self, queues):
        """Create GapDetector instance."""
        return GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

    @pytest.mark.asyncio
    async def test_short_gap_count_increments(self, detector) -> None:
        """Test short gap count increments correctly."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=120)

        await detector._handle_gap(gap_start, gap_end)

        assert detector.short_gap_count == 1

    @pytest.mark.asyncio
    async def test_extended_gap_count_increments(self, detector) -> None:
        """Test extended gap count increments correctly."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=600)  # 10 minutes

        await detector._handle_gap(gap_start, gap_end)

        assert detector.extended_gap_count == 1

    @pytest.mark.asyncio
    async def test_total_gap_duration_calculated(self, detector) -> None:
        """Test total gap duration calculated correctly."""
        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=180)

        await detector._handle_gap(gap_start, gap_end)

        assert detector.total_gap_duration == 180.0

    @pytest.mark.asyncio
    async def test_average_gap_duration_calculated(self, detector) -> None:
        """Test average gap duration calculated correctly."""
        # First gap: 120 seconds
        gap1_start = datetime.now()
        gap1_end = gap1_start + timedelta(seconds=120)
        await detector._handle_gap(gap1_start, gap1_end)

        # Second gap: 180 seconds
        gap2_start = datetime.now()
        gap2_end = gap2_start + timedelta(seconds=180)
        await detector._handle_gap(gap2_start, gap2_end)

        # Average: (120 + 180) / 2 = 150
        assert detector.average_gap_duration == 150.0

    @pytest.mark.asyncio
    async def test_statistics_logging_no_exception(self, detector) -> None:
        """Test statistics logging doesn't raise exceptions."""
        # Should not raise exception even if no gaps yet
        detector._log_gap_statistics()

        assert detector.total_gap_count == 0

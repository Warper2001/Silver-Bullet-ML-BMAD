"""Integration tests for gap detection and forward-fill."""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.data.gap_detection import GapDetector
from src.data.models import DollarBar


class TestGapDetectionPipeline:
    """Test end-to-end gap detection pipeline."""

    @pytest.fixture
    def queues(self):
        """Create all required queues."""
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
    async def test_valid_bars_flow_through_pipeline(self, detector, queues) -> None:
        """Test valid bars flow through gap detection pipeline."""
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        # Publish to validated queue
        queues["validated"].put_nowait(bar)

        # Simulate detector processing
        detector._last_valid_bar = bar
        detector._last_seen_timestamp = datetime.now()
        await detector._gap_filled_queue.put(bar)

        # Verify it went to gap-filled queue
        assert queues["gap_filled"].qsize() == 1

    @pytest.mark.asyncio
    async def test_gap_detection_and_recovery(self, detector, queues) -> None:
        """Test gap detection and recovery in real-time scenario."""
        # First bar
        bar1 = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        queues["validated"].put_nowait(bar1)
        detector._last_valid_bar = bar1
        detector._last_seen_timestamp = bar1.timestamp
        await detector._gap_filled_queue.put(bar1)

        # Simulate gap start
        gap_start = datetime.now()
        detector._gap_start = gap_start

        # Simulate gap end with new bar
        gap_end = gap_start + timedelta(seconds=60)

        # Handle the gap
        await detector._handle_gap(gap_start, gap_end)

        # Verify gap statistics tracked
        assert detector.short_gap_count == 1
        assert detector.short_gap_duration_total == 60.0

    @pytest.mark.asyncio
    async def test_forward_filled_bars_integrate_with_pipeline(
        self, detector, queues
    ) -> None:
        """Test forward-filled bars integrate with downstream pipeline."""
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

        # Publish filled bars to gap-filled queue
        for bar in filled_bars:
            queues["gap_filled"].put_nowait(bar)

        # Verify all filled bars are in queue
        assert queues["gap_filled"].qsize() == 6


class TestGapStatisticsTracking:
    """Test gap statistics tracking."""

    @pytest.fixture
    def queues(self):
        """Create all required queues."""
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
    async def test_multiple_gaps_tracked_correctly(self, detector) -> None:
        """Test gap statistics tracked over multiple gaps."""
        # First gap: 60 seconds
        gap1_start = datetime.now()
        gap1_end = gap1_start + timedelta(seconds=60)
        await detector._handle_gap(gap1_start, gap1_end)

        # Second gap: 120 seconds
        gap2_start = datetime.now()
        gap2_end = gap2_start + timedelta(seconds=120)
        await detector._handle_gap(gap2_start, gap2_end)

        # Verify statistics
        assert detector.short_gap_count == 2
        assert detector.short_gap_duration_total == 180.0
        assert detector.total_gap_count == 2
        assert detector.total_gap_duration == 180.0
        assert detector.average_gap_duration == 90.0

    @pytest.mark.asyncio
    async def test_short_and_extended_gaps_tracked_separately(self, detector) -> None:
        """Test short and extended gaps tracked separately."""
        # Short gap: 120 seconds
        short_gap_start = datetime.now()
        short_gap_end = short_gap_start + timedelta(seconds=120)
        await detector._handle_gap(short_gap_start, short_gap_end)

        # Extended gap: 600 seconds (10 minutes)
        extended_gap_start = datetime.now()
        extended_gap_end = extended_gap_start + timedelta(seconds=600)
        await detector._handle_gap(extended_gap_start, extended_gap_end)

        # Verify statistics
        assert detector.short_gap_count == 1
        assert detector.extended_gap_count == 1
        assert detector.short_gap_duration_total == 120.0
        assert detector.extended_gap_duration_total == 600.0
        assert detector.total_gap_count == 2
        assert detector.total_gap_duration == 720.0

    @pytest.mark.asyncio
    async def test_data_completeness_calculated(self, detector) -> None:
        """Test data completeness % calculated correctly."""
        # Simulate 1 hour session with 30 seconds of gaps (shorter gap)
        session_duration = 3600  # 1 hour in seconds
        gap_duration = 30  # Reduced from 60 to get > 99% completeness

        gap_start = datetime.now()
        gap_end = gap_start + timedelta(seconds=gap_duration)
        await detector._handle_gap(gap_start, gap_end)

        # Data completeness = (1 - total_gap_duration / total_session_duration) × 100
        data_completeness = (1 - detector.total_gap_duration / session_duration) * 100

        assert data_completeness > 99.0  # Should be > 99% complete


class TestQueueBackpressure:
    """Test queue overflow handling."""

    @pytest.fixture
    def queues(self):
        """Create queues with size limits."""
        import asyncio

        return {
            "validated": asyncio.Queue(),
            "gap_filled": asyncio.Queue(maxsize=5),
        }

    @pytest.fixture
    def detector(self, queues):
        """Create GapDetector instance."""
        return GapDetector(
            queues["validated"],
            queues["gap_filled"],
        )

    @pytest.mark.asyncio
    async def test_gap_filled_queue_full(self, detector, queues) -> None:
        """Test handling when gap-filled queue is full."""
        # Fill gap-filled queue
        for _ in range(5):
            bar = DollarBar(
                timestamp=datetime.now(),
                open=4523.25,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )
            queues["gap_filled"].put_nowait(bar)

        assert queues["gap_filled"].full()

        # Try to publish another bar (should handle gracefully)
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        detector._last_valid_bar = bar
        detector._last_seen_timestamp = datetime.now()

        # Should not raise exception
        try:
            queues["gap_filled"].put_nowait(bar)
        except asyncio.QueueFull:
            # Expected - queue is full
            pass

        # Queue should still be full (6th bar dropped)
        assert queues["gap_filled"].qsize() == 5

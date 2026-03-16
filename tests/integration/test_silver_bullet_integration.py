"""Integration tests for Silver Bullet Setup Detector."""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.data.models import (
    FVGEvent,
    GapRange,
    LiquiditySweepEvent,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.detection.silver_bullet_detector import SilverBulletDetector


class TestSilverBulletSetupIntegration:
    """Test Silver Bullet setup detection with real event sequences."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for testing."""
        return asyncio.Queue()

    @pytest.fixture
    def output_queue(self):
        """Create output queue for testing."""
        return asyncio.Queue()

    @pytest.fixture
    def detector(self, input_queue, output_queue):
        """Create SilverBulletDetector instance for testing."""
        return SilverBulletDetector(
            input_queue=input_queue,
            output_queue=output_queue,
            max_bar_distance=10,
        )

    @pytest.mark.asyncio
    async def test_silver_bullet_detection_from_events(
        self, detector, input_queue, output_queue
    ):
        """Test end-to-end Silver Bullet detection from event queues."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create swing point
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        # Create MSS event
        mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_low,
            volume_ratio=1.8,
            bar_index=10,
        )

        # Create FVG event
        fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=15),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        # Feed events to detector (in practice would come from queues)
        detector.add_mss_event(mss)
        detector.add_fvg_event(fvg)

        # Trigger detection
        await detector._detect_setups()

        # Check output queue
        assert not output_queue.empty()
        setup = await output_queue.get()
        assert isinstance(setup, SilverBulletSetup)
        assert setup.direction == "bullish"
        assert setup.confluence_count == 2

    @pytest.mark.asyncio
    async def test_silver_bullet_with_3_pattern_confluence(
        self, detector, input_queue, output_queue
    ):
        """Test Silver Bullet with MSS + FVG + Sweep (highest priority)."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create events
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_low,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=15),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        sweep = LiquiditySweepEvent(
            timestamp=base_time + timedelta(seconds=20),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )

        # Add events
        detector.add_mss_event(mss)
        detector.add_fvg_event(fvg)
        detector.add_sweep_event(sweep)

        # Trigger detection
        await detector._detect_setups()

        # Check output
        setup = await output_queue.get()
        assert setup.confluence_count == 3
        assert setup.priority == "high"
        assert setup.liquidity_sweep_event == sweep

    @pytest.mark.asyncio
    async def test_multiple_setups_detected(self, detector, output_queue):
        """Test detector can identify multiple setups."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create bullish setup
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        bull_mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_low,
            volume_ratio=1.8,
            bar_index=10,
        )

        bull_fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=15),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        # Create bearish setup
        swing_high = SwingPoint(
            timestamp=base_time,
            price=11900.0,
            swing_type="swing_high",
            bar_index=0,
            confirmed=True,
        )

        bear_mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=30),
            direction="bearish",
            breakout_price=11890.0,
            swing_point=swing_high,
            volume_ratio=1.8,
            bar_index=20,
        )

        bear_fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=35),
            direction="bearish",
            gap_range=GapRange(top=11910.0, bottom=11880.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=21,
        )

        # Add events
        detector.add_mss_event(bull_mss)
        detector.add_fvg_event(bull_fvg)
        detector.add_mss_event(bear_mss)
        detector.add_fvg_event(bear_fvg)

        # Trigger detection
        await detector._detect_setups()

        # Should have 2 setups
        setups = []
        while not output_queue.empty():
            setups.append(await output_queue.get())

        assert len(setups) == 2

    @pytest.mark.asyncio
    async def test_event_history_management(self, detector):
        """Test detector maintains event history (max 50 each)."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add 100 MSS events
        for i in range(100):
            swing = SwingPoint(
                timestamp=base_time + timedelta(seconds=i),
                price=11800.0 + i,
                swing_type="swing_low",
                bar_index=i,
                confirmed=True,
            )

            mss = MSSEvent(
                timestamp=base_time + timedelta(seconds=i * 10),
                direction="bullish",
                breakout_price=11810.0 + i,
                swing_point=swing,
                volume_ratio=1.8,
                bar_index=i,
            )
            detector.add_mss_event(mss)

        # Should only keep last 50
        assert detector.mss_events_count == 50

    @pytest.mark.asyncio
    async def test_detection_latency_under_100ms(self, detector):
        """Test detection latency meets performance requirement (< 100ms)."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create events
        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=15),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        detector.add_mss_event(mss)
        detector.add_fvg_event(fvg)

        # Measure detection time
        import time

        start_time = time.perf_counter()
        await detector._detect_setups()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Should be very fast (< 10ms typically)
        assert latency_ms < 100

    @pytest.mark.asyncio
    async def test_setup_contains_all_required_fields(self, detector, output_queue):
        """Test Silver Bullet setups contain all required fields."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=15),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        detector.add_mss_event(mss)
        detector.add_fvg_event(fvg)
        await detector._detect_setups()

        setup = await output_queue.get()

        # Verify all required fields
        assert hasattr(setup, "timestamp")
        assert hasattr(setup, "direction")
        assert setup.direction in ["bullish", "bearish"]
        assert hasattr(setup, "mss_event")
        assert hasattr(setup, "fvg_event")
        assert hasattr(setup, "entry_zone_top")
        assert setup.entry_zone_top > 0
        assert hasattr(setup, "entry_zone_bottom")
        assert setup.entry_zone_bottom > 0
        assert hasattr(setup, "invalidation_point")
        assert setup.invalidation_point > 0
        assert hasattr(setup, "confluence_count")
        assert 2 <= setup.confluence_count <= 3
        assert hasattr(setup, "priority")
        assert setup.priority in ["low", "medium", "high"]
        assert hasattr(setup, "bar_index")
        assert setup.bar_index >= 0
        assert hasattr(setup, "confidence")
        assert 0 <= setup.confidence <= 5

    @pytest.mark.asyncio
    async def test_no_setup_when_directions_mismatch(self, detector):
        """Test no setup when MSS and FVG have opposite directions."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        # Bullish MSS
        mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        # Bearish FVG (mismatch!)
        fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=15),
            direction="bearish",
            gap_range=GapRange(top=11900.0, bottom=11870.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        detector.add_mss_event(mss)
        detector.add_fvg_event(fvg)
        await detector._detect_setups()

        # No setup should be created
        assert detector.output_queue.empty()

    @pytest.mark.asyncio
    async def test_no_setup_when_events_too_far_apart(self, detector):
        """Test no setup when MSS and FVG more than 10 bars apart."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time + timedelta(seconds=10),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=0,  # Bar 0
        )

        fvg = FVGEvent(
            timestamp=base_time + timedelta(seconds=100),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=20,  # Bar 20 (20 bars apart!)
        )

        detector.add_mss_event(mss)
        detector.add_fvg_event(fvg)
        await detector._detect_setups()

        # No setup should be created
        assert detector.output_queue.empty()

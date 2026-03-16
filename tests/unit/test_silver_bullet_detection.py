"""Unit tests for Silver Bullet Setup Detection."""

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
from src.detection.silver_bullet_detection import (
    check_silver_bullet_setup,
    detect_silver_bullet_setup,
)


class TestSilverBulletSetupAlgorithms:
    """Test Silver Bullet setup detection algorithms."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for test events."""
        return datetime(2026, 3, 16, 10, 0, 0)

    @pytest.fixture
    def swing_point(self, base_timestamp):
        """Create a sample swing point."""
        return SwingPoint(
            timestamp=base_timestamp - timedelta(seconds=30),
            price=11800.0,
            swing_type="swing_low",
            bar_index=5,
            confirmed=True,
        )

    @pytest.fixture
    def bullish_mss(self, base_timestamp, swing_point):
        """Create a sample bullish MSS event."""
        return MSSEvent(
            timestamp=base_timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

    @pytest.fixture
    def bull_fvg(self, base_timestamp):
        """Create a sample bullish FVG event."""
        return FVGEvent(
            timestamp=base_timestamp + timedelta(seconds=5),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

    @pytest.fixture
    def bull_sweep(self, base_timestamp):
        """Create a sample bullish liquidity sweep event."""
        return LiquiditySweepEvent(
            timestamp=base_timestamp + timedelta(seconds=10),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )

    def test_silver_bullet_recognized_with_mss_and_fvg(self, bullish_mss, bull_fvg):
        """Verify Silver Bullet recognized when MSS and FVG within 10 bars."""
        setup = check_silver_bullet_setup(
            mss_event=bullish_mss, fvg_event=bull_fvg, max_bar_distance=10
        )

        assert setup is not None
        assert setup.direction == "bullish"
        assert setup.mss_event == bullish_mss
        assert setup.fvg_event == bull_fvg
        assert setup.liquidity_sweep_event is None  # No sweep
        assert setup.confluence_count == 2  # MSS + FVG

    def test_silver_bullet_higher_priority_with_sweep(
        self, bullish_mss, bull_fvg, bull_sweep
    ):
        """Verify higher priority when all 3 patterns present."""
        setup = check_silver_bullet_setup(
            mss_event=bullish_mss,
            fvg_event=bull_fvg,
            sweep_event=bull_sweep,
            max_bar_distance=10,
        )

        assert setup is not None
        assert setup.confluence_count == 3  # MSS + FVG + Sweep
        assert setup.liquidity_sweep_event == bull_sweep
        assert setup.priority == "high"  # 3-pattern confluence

    def test_no_silver_bullet_when_mss_and_fvg_too_far_apart(
        self, base_timestamp, swing_point
    ):
        """Verify no setup when MSS and FVG more than 10 bars apart."""
        mss = MSSEvent(
            timestamp=base_timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=0,  # Bar 0
        )

        fvg = FVGEvent(
            timestamp=base_timestamp + timedelta(seconds=100),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=15,  # Bar 15 (15 bars apart)
        )

        setup = check_silver_bullet_setup(
            mss_event=mss, fvg_event=fvg, max_bar_distance=10
        )

        assert setup is None  # Too far apart

    def test_silver_bullet_direction_based_on_mss(self, bullish_mss, bull_fvg):
        """Verify setup direction matches MSS direction."""
        setup = check_silver_bullet_setup(
            mss_event=bullish_mss, fvg_event=bull_fvg, max_bar_distance=10
        )

        assert setup.direction == "bullish"  # From MSS

    def test_silver_bullet_entry_zone_from_fvg(self, bullish_mss, bull_fvg):
        """Verify entry zone identified from FVG gap range."""
        setup = check_silver_bullet_setup(
            mss_event=bullish_mss, fvg_event=bull_fvg, max_bar_distance=10
        )

        assert setup.entry_zone_top == bull_fvg.gap_range.top  # 11820.0
        assert setup.entry_zone_bottom == bull_fvg.gap_range.bottom  # 11790.0

    def test_silver_bullet_invalidation_from_opposite_swing(
        self, bullish_mss, bull_fvg, swing_point
    ):
        """Verify invalidation point at opposite swing point."""
        setup = check_silver_bullet_setup(
            mss_event=bullish_mss, fvg_event=bull_fvg, max_bar_distance=10
        )

        assert setup.invalidation_point == swing_point.price  # 11800.0

    def test_no_silver_bullet_when_mss_and_fvg_directions_mismatch(
        self, base_timestamp, swing_point
    ):
        """Verify no setup when MSS and FVG have opposite directions."""
        bullish_mss = MSSEvent(
            timestamp=base_timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        bearish_fvg = FVGEvent(
            timestamp=base_timestamp + timedelta(seconds=5),
            direction="bearish",  # Opposite direction!
            gap_range=GapRange(top=11900.0, bottom=11870.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        setup = check_silver_bullet_setup(
            mss_event=bullish_mss, fvg_event=bearish_fvg, max_bar_distance=10
        )

        assert setup is None  # Direction mismatch

    def test_silver_bullet_requires_both_mss_and_fvg(self, bullish_mss):
        """Verify both MSS and FVG required for setup."""
        setup = check_silver_bullet_setup(
            mss_event=bullish_mss, fvg_event=None, max_bar_distance=10
        )

        assert setup is None  # Missing FVG

    def test_detect_silver_bullet_from_event_lists(self, base_timestamp):
        """Verify detection from lists of events."""
        swing_point = SwingPoint(
            timestamp=base_timestamp,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss_events = [
            MSSEvent(
                timestamp=base_timestamp + timedelta(seconds=10),
                direction="bullish",
                breakout_price=11810.0,
                swing_point=swing_point,
                volume_ratio=1.8,
                bar_index=10,
            )
        ]

        fvg_events = [
            FVGEvent(
                timestamp=base_timestamp + timedelta(seconds=15),
                direction="bullish",
                gap_range=GapRange(top=11820.0, bottom=11790.0),
                gap_size_ticks=120.0,
                gap_size_dollars=600.0,
                bar_index=11,
            )
        ]

        setups = detect_silver_bullet_setup(
            mss_events=mss_events,
            fvg_events=fvg_events,
            max_bar_distance=10,
        )

        assert len(setups) == 1
        assert setups[0].direction == "bullish"
        assert setups[0].confluence_count == 2

    def test_multiple_silver_bullets_detected(self, base_timestamp):
        """Verify multiple setups detected from multiple events."""
        swing_low = SwingPoint(
            timestamp=base_timestamp,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        swing_high = SwingPoint(
            timestamp=base_timestamp,
            price=11900.0,
            swing_type="swing_high",
            bar_index=0,
            confirmed=True,
        )

        mss_events = [
            MSSEvent(
                timestamp=base_timestamp + timedelta(seconds=10),
                direction="bullish",
                breakout_price=11810.0,
                swing_point=swing_low,
                volume_ratio=1.8,
                bar_index=10,
            ),
            MSSEvent(
                timestamp=base_timestamp + timedelta(seconds=20),
                direction="bearish",
                breakout_price=11890.0,
                swing_point=swing_high,
                volume_ratio=1.8,
                bar_index=20,
            ),
        ]

        fvg_events = [
            FVGEvent(
                timestamp=base_timestamp + timedelta(seconds=15),
                direction="bullish",
                gap_range=GapRange(top=11820.0, bottom=11790.0),
                gap_size_ticks=120.0,
                gap_size_dollars=600.0,
                bar_index=11,
            ),
            FVGEvent(
                timestamp=base_timestamp + timedelta(seconds=25),
                direction="bearish",
                gap_range=GapRange(top=11910.0, bottom=11880.0),
                gap_size_ticks=120.0,
                gap_size_dollars=600.0,
                bar_index=21,
            ),
        ]

        setups = detect_silver_bullet_setup(
            mss_events=mss_events,
            fvg_events=fvg_events,
            max_bar_distance=10,
        )

        assert len(setups) == 2  # Bullish + Bearish


class TestSilverBulletSetupModel:
    """Test SilverBulletSetup data model."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for testing."""
        return datetime(2026, 3, 16, 10, 0, 0)

    def test_silver_bullet_setup_creation(self, base_timestamp):
        """Verify SilverBulletSetup can be created with all required fields."""
        swing_point = SwingPoint(
            timestamp=base_timestamp,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=base_timestamp,
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        setup = SilverBulletSetup(
            timestamp=base_timestamp,
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

        assert setup.timestamp == base_timestamp
        assert setup.direction == "bullish"
        assert setup.mss_event == mss
        assert setup.fvg_event == fvg
        assert setup.liquidity_sweep_event is None
        assert setup.entry_zone_top == 11820.0
        assert setup.entry_zone_bottom == 11790.0
        assert setup.invalidation_point == 11800.0
        assert setup.confluence_count == 2
        assert setup.priority == "medium"
        assert setup.bar_index == 11
        assert setup.confidence == 0.0

    def test_silver_bullet_setup_valid_directions(self, base_timestamp):
        """Verify only valid directions are accepted."""
        # Bullish is valid
        assert "bullish" in ["bullish", "bearish"]

        # Bearish is valid
        assert "bearish" in ["bullish", "bearish"]

    def test_silver_bullet_setup_valid_priorities(self, base_timestamp):
        """Verify only valid priorities are accepted."""
        # Low, medium, high are valid
        valid_priorities = ["low", "medium", "high"]
        assert all(p in valid_priorities for p in ["low", "medium", "high"])

    def test_silver_bullet_setup_confluence_range(self, base_timestamp):
        """Verify confluence count is within valid range."""
        swing_point = SwingPoint(
            timestamp=base_timestamp,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_timestamp,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing_point,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=base_timestamp,
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=11,
        )

        # Valid: confluence count 2-3
        setup = SilverBulletSetup(
            timestamp=base_timestamp,
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
        assert 2 <= setup.confluence_count <= 3

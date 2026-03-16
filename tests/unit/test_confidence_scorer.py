"""Unit tests for Confidence Score Calculation."""

from datetime import datetime

import pytest

from src.data.models import (
    FVGEvent,
    GapRange,
    LiquiditySweepEvent,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.detection.confidence_scorer import (
    calculate_confidence_score,
    score_setup,
)


class TestConfidenceScoreCalculation:
    """Test confidence score calculation algorithms."""

    @pytest.fixture
    def base_setup(self):
        """Create a base Silver Bullet setup for testing.

        Uses neutral values that won't trigger automatic score increases:
        - volume_ratio=1.3 (between 1.2x and 1.5x thresholds)
        - gap_size_ticks=15.0 (between 10 and 20 tick thresholds)
        """
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
            volume_ratio=1.3,  # Between thresholds (1.2x - 1.5x)
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=15.0,  # Between thresholds (10-20 ticks)
            gap_size_dollars=75.0,
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

    def test_base_score_2_pattern_confluence(self, base_setup):
        """Verify base score of 1 for MSS + FVG confluence."""
        score = calculate_confidence_score(base_setup)

        assert score == 1

    def test_score_3_for_3_pattern_confluence(self, base_setup):
        """Verify score of 3 for MSS + FVG + liquidity sweep."""
        # Add liquidity sweep
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        score = calculate_confidence_score(base_setup)

        assert score == 3

    def test_score_4_high_volume_ratio(self, base_setup):
        """Verify score of 4 when volume ratio > 1.5x with 3-pattern."""
        # Add liquidity sweep for 3-pattern confluence
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        # Set volume ratio to 2.0 (above 1.5x threshold)
        base_setup.mss_event.volume_ratio = 2.0

        score = calculate_confidence_score(base_setup)

        assert score == 4  # Base 3 + 1 for high volume

    def test_score_5_large_fvg(self, base_setup):
        """Verify score of 5 when FVG size > 20 ticks with 3-pattern confluence."""
        # Add liquidity sweep for 3-pattern confluence
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        # Set volume ratio to 2.0 (above 1.5x threshold for high score)
        base_setup.mss_event.volume_ratio = 2.0

        # Set FVG size to 25 ticks (above 20 tick threshold)
        base_setup.fvg_event.gap_size_ticks = 25.0

        score = calculate_confidence_score(base_setup)

        assert score == 5  # Base 3 + 1 for high volume + 1 for large FVG

    def test_score_2_weak_patterns_low_volume(self, base_setup):
        """Verify score of 2 when patterns weak (low volume) with 3-pattern."""
        # Add liquidity sweep for 3-pattern confluence
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        # Set volume ratio to 1.1 (below 1.2x threshold)
        base_setup.mss_event.volume_ratio = 1.1

        score = calculate_confidence_score(base_setup)

        assert score == 2  # Base 3 - 1 for low volume

    def test_score_2_weak_patterns_small_fvg(self, base_setup):
        """Verify score of 2 when patterns weak (small FVG) with 3-pattern."""
        # Add liquidity sweep for 3-pattern confluence
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        # Set FVG size to 8 ticks (below 10 tick threshold)
        base_setup.fvg_event.gap_size_ticks = 8.0

        score = calculate_confidence_score(base_setup)

        assert score == 2  # Base 3 - 1 for small FVG

    def test_score_5_maximum_confluence(self, base_setup):
        """Verify maximum score of 5 with all factors combined."""
        # Add liquidity sweep
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        # High volume ratio
        base_setup.mss_event.volume_ratio = 2.0

        # Large FVG
        base_setup.fvg_event.gap_size_ticks = 25.0

        score = calculate_confidence_score(base_setup)

        assert score == 5  # Max score

    def test_score_capped_at_5(self, base_setup):
        """Verify score is capped at 5 (maximum)."""
        # Add sweep
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        # Very high volume ratio
        base_setup.mss_event.volume_ratio = 5.0

        # Very large FVG
        base_setup.fvg_event.gap_size_ticks = 100.0

        score = calculate_confidence_score(base_setup)

        assert score == 5  # Capped at maximum

    def test_score_1_minimum_2_pattern(self, base_setup):
        """Verify minimum score of 1 for basic 2-pattern setup."""
        # Ensure minimum conditions
        base_setup.mss_event.volume_ratio = 1.3  # Above 1.2x
        base_setup.fvg_event.gap_size_ticks = 15.0  # Above 10 ticks
        base_setup.confluence_count = 2  # Only 2 patterns

        score = calculate_confidence_score(base_setup)

        assert score == 1  # Base score

    def test_score_prioritization(self, base_setup):
        """Verify score increases with confluence factors."""
        # Start with base setup
        base_setup.confluence_count = 2
        base_setup.mss_event.volume_ratio = 1.3
        base_setup.fvg_event.gap_size_ticks = 15.0

        score = calculate_confidence_score(base_setup)
        assert score == 1

        # Add liquidity sweep
        base_setup.confluence_count = 3
        base_setup.liquidity_sweep_event = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )

        score = calculate_confidence_score(base_setup)
        assert score >= 3  # Should be 3 or higher

        # Increase volume ratio
        base_setup.mss_event.volume_ratio = 2.0

        score = calculate_confidence_score(base_setup)
        assert score >= 4  # Should be 4 or 5


class TestSetupScoring:
    """Test setup scoring integration."""

    @pytest.fixture
    def base_setup(self):
        """Create a base Silver Bullet setup for testing.

        Uses neutral values that won't trigger automatic score increases:
        - volume_ratio=1.3 (between 1.2x and 1.5x thresholds)
        - gap_size_ticks=15.0 (between 10 and 20 tick thresholds)
        """
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
            volume_ratio=1.3,  # Between thresholds (1.2x - 1.5x)
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=15.0,  # Between thresholds (10-20 ticks)
            gap_size_dollars=75.0,
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

    def test_score_setup_updates_confidence(self, base_setup):
        """Verify score_setup updates setup confidence field."""
        assert base_setup.confidence == 0.0  # Initially 0

        scored_setup = score_setup(base_setup)

        assert scored_setup.confidence > 0
        assert scored_setup == base_setup  # Same object modified

    def test_score_setup_returns_setup(self, base_setup):
        """Verify score_setup returns the setup object."""
        result = score_setup(base_setup)

        assert result is not None
        assert isinstance(result, SilverBulletSetup)
        assert result.confidence > 0

    def test_score_2_pattern_medium_priority_setup(self, base_setup):
        """Verify 2-pattern setup gets appropriate score."""
        base_setup.confluence_count = 2
        base_setup.mss_event.volume_ratio = 1.3
        base_setup.fvg_event.gap_size_ticks = 15.0

        scored_setup = score_setup(base_setup)

        assert 1 <= scored_setup.confidence <= 3

    def test_score_3_pattern_high_priority_setup(self, base_setup):
        """Verify 3-pattern setup gets higher score."""
        sweep = LiquiditySweepEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=12,
        )
        base_setup.liquidity_sweep_event = sweep
        base_setup.confluence_count = 3

        scored_setup = score_setup(base_setup)

        assert scored_setup.confidence >= 3  # 3-pattern base score

    def test_performance_under_10ms(self, base_setup):
        """Verify confidence calculation adds < 10ms overhead."""
        import time

        # Measure time
        start_time = time.perf_counter()
        for _ in range(100):
            score_setup(base_setup)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        avg_time_ms = elapsed_ms / 100

        # Should be very fast (< 1ms typically)
        assert avg_time_ms < 10

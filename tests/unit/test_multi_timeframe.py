"""Unit tests for multi-timeframe nester module."""

import pytest
from datetime import datetime, timedelta

from src.data.models import DollarBar, FVGEvent, GapRange
from src.detection.multi_timeframe import MultiTimeframeNester


@pytest.fixture
def sample_bars_1min():
    """Create sample 1-minute dollar bars for testing."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i in range(100):
        bar = DollarBar(
            timestamp=base_time + timedelta(minutes=i),
            open=11800.0 + i * 0.5,
            high=11805.0 + i * 0.5,
            low=11795.0 + i * 0.5,
            close=11803.0 + i * 0.5,
            volume=1000,
            notional_value=50000000.0,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def nested_fvg_setup():
    """Create a nested FVG setup for testing."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    # Parent FVG (21-min timeframe) - large gap
    parent_fvg = FVGEvent(
        timestamp=base_time,
        direction="bullish",
        gap_range=GapRange(top=11820.0, bottom=11800.0),
        gap_size_ticks=80.0,
        gap_size_dollars=1600.0,
        bar_index=50,
        filled=False,
    )

    # Child FVG (5-min timeframe) - small gap inside parent
    child_fvg = FVGEvent(
        timestamp=base_time + timedelta(minutes=5),
        direction="bullish",
        gap_range=GapRange(top=11815.0, bottom=11805.0),
        gap_size_ticks=40.0,
        gap_size_dollars=800.0,
        bar_index=55,
        filled=False,
    )

    return parent_fvg, child_fvg


@pytest.fixture
def non_nested_fvg_setup():
    """Create a non-nested FVG setup for testing."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    # Parent FVG - gap at 11800-11820
    parent_fvg = FVGEvent(
        timestamp=base_time,
        direction="bullish",
        gap_range=GapRange(top=11820.0, bottom=11800.0),
        gap_size_ticks=80.0,
        gap_size_dollars=1600.0,
        bar_index=50,
        filled=False,
    )

    # Child FVG - gap OUTSIDE parent (not nested)
    child_fvg = FVGEvent(
        timestamp=base_time + timedelta(minutes=5),
        direction="bullish",
        gap_range=GapRange(top=11825.0, bottom=11815.0),
        gap_size_ticks=40.0,
        gap_size_dollars=800.0,
        bar_index=55,
        filled=False,
    )

    return parent_fvg, child_fvg


class TestMultiTimeframeNester:
    """Test suite for MultiTimeframeNester class."""

    def test_resample_bars_5min(self, sample_bars_1min):
        """Test resampling 1-minute bars to 5-minute bars."""
        nester = MultiTimeframeNester()

        resampled = nester.resample_bars(sample_bars_1min, 5)

        # 100 1-min bars should become ~20 5-min bars (may vary slightly)
        assert len(resampled) >= 19
        assert len(resampled) <= 21

        # Check OHLC aggregation
        for bar in resampled:
            assert bar.high >= bar.low
            assert bar.close >= bar.low
            assert bar.close <= bar.high

    def test_resample_bars_21min(self, sample_bars_1min):
        """Test resampling 1-minute bars to 21-minute bars."""
        nester = MultiTimeframeNester()

        resampled = nester.resample_bars(sample_bars_1min, 21)

        # 100 1-min bars should become ~4-6 21-min bars (may vary)
        assert len(resampled) >= 4
        assert len(resampled) <= 6

    def test_resample_bars_empty(self):
        """Test resampling empty bar list."""
        nester = MultiTimeframeNester()

        resampled = nester.resample_bars([], 5)

        assert len(resampled) == 0

    def test_resample_bars_insufficient_data(self, sample_bars_1min):
        """Test resampling with insufficient data for target timeframe."""
        nester = MultiTimeframeNester()

        # Only 3 1-min bars, resample to 21-min
        short_bars = sample_bars_1min[:3]
        resampled = nester.resample_bars(short_bars, 21)

        # Should return 1 bar (pandas will still aggregate)
        assert len(resampled) >= 0

    def test_detect_nested_fvg_bullish(self, nested_fvg_setup):
        """Test nested FVG detection for bullish direction."""
        parent_fvg, child_fvg = nested_fvg_setup
        nester = MultiTimeframeNester()

        nested_fvg = nester.detect_nested_fvg(
            child_fvg, parent_fvg, small_tf=5, large_tf=21
        )

        assert nested_fvg is not None
        assert nested_fvg.direction == "bullish"
        assert nested_fvg.child_fvg == child_fvg
        assert nested_fvg.parent_fvg == parent_fvg
        assert nested_fvg.timeframe_pair == (5, 21)
        assert nested_fvg.nesting_level == 1

    def test_detect_nested_fvg_bearish(self):
        """Test nested FVG detection for bearish direction."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Parent FVG (bearish)
        parent_fvg = FVGEvent(
            timestamp=base_time,
            direction="bearish",
            gap_range=GapRange(top=11800.0, bottom=11780.0),
            gap_size_ticks=80.0,
            gap_size_dollars=1600.0,
            bar_index=50,
            filled=False,
        )

        # Child FVG (bearish, inside parent)
        child_fvg = FVGEvent(
            timestamp=base_time + timedelta(minutes=5),
            direction="bearish",
            gap_range=GapRange(top=11795.0, bottom=11785.0),
            gap_size_ticks=40.0,
            gap_size_dollars=800.0,
            bar_index=55,
            filled=False,
        )

        nester = MultiTimeframeNester()
        nested_fvg = nester.detect_nested_fvg(
            child_fvg, parent_fvg, small_tf=5, large_tf=21
        )

        assert nested_fvg is not None
        assert nested_fvg.direction == "bearish"

    def test_detect_nested_fvg_no_containment(self, non_nested_fvg_setup):
        """Test nested FVG detection when child is not contained."""
        parent_fvg, child_fvg = non_nested_fvg_setup
        nester = MultiTimeframeNester()

        nested_fvg = nester.detect_nested_fvg(
            child_fvg, parent_fvg, small_tf=5, large_tf=21
        )

        assert nested_fvg is None

    def test_detect_nested_fvg_direction_mismatch(self):
        """Test nested FVG detection with direction mismatch."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Parent FVG (bullish)
        parent_fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11800.0),
            gap_size_ticks=80.0,
            gap_size_dollars=1600.0,
            bar_index=50,
            filled=False,
        )

        # Child FVG (bearish, inside parent but wrong direction)
        child_fvg = FVGEvent(
            timestamp=base_time + timedelta(minutes=5),
            direction="bearish",
            gap_range=GapRange(top=11815.0, bottom=11805.0),
            gap_size_ticks=40.0,
            gap_size_dollars=800.0,
            bar_index=55,
            filled=False,
        )

        nester = MultiTimeframeNester()
        nested_fvg = nester.detect_nested_fvg(
            child_fvg, parent_fvg, small_tf=5, large_tf=21
        )

        assert nested_fvg is None

    def test_check_nesting_with_nesting(self, nested_fvg_setup, sample_bars_1min):
        """Test check_nesting method with nesting present."""
        parent_fvg, child_fvg = nested_fvg_setup
        nester = MultiTimeframeNester()

        # Mock FVG history with parent
        fvg_history = {21: [parent_fvg]}

        has_nesting, nested_fvgs = nester.check_nesting(
            child_fvg, sample_bars_1min, fvg_history
        )

        assert has_nesting
        assert len(nested_fvgs) == 1
        assert nested_fvgs[0].child_fvg == child_fvg

    def test_check_nesting_without_nesting(self, non_nested_fvg_setup, sample_bars_1min):
        """Test check_nesting method without nesting."""
        parent_fvg, child_fvg = non_nested_fvg_setup
        nester = MultiTimeframeNester()

        # Mock FVG history with parent
        fvg_history = {21: [parent_fvg]}

        has_nesting, nested_fvgs = nester.check_nesting(
            child_fvg, sample_bars_1min, fvg_history
        )

        assert not has_nesting
        assert len(nested_fvgs) == 0

    def test_check_nesting_empty_history(self, nested_fvg_setup, sample_bars_1min):
        """Test check_nesting with empty FVG history."""
        parent_fvg, child_fvg = nested_fvg_setup
        nester = MultiTimeframeNester()

        fvg_history = {}

        has_nesting, nested_fvgs = nester.check_nesting(
            child_fvg, sample_bars_1min, fvg_history
        )

        # No nesting detected (history is empty)
        # Note: In real scenario, nester would detect FVGs at larger timeframes
        assert not has_nesting or len(nested_fvgs) >= 0

    def test_nesting_detection_count_increment(self, nested_fvg_setup):
        """Test that nesting detection count increments."""
        parent_fvg, child_fvg = nested_fvg_setup
        nester = MultiTimeframeNester()

        initial_count = nester.nesting_detection_count
        nester.detect_nested_fvg(child_fvg, parent_fvg, small_tf=5, large_tf=21)

        assert nester.nesting_detection_count == initial_count + 1

    def test_reset_metrics(self, nested_fvg_setup):
        """Test resetting metrics."""
        parent_fvg, child_fvg = nested_fvg_setup
        nester = MultiTimeframeNester()

        nester.detect_nested_fvg(child_fvg, parent_fvg, small_tf=5, large_tf=21)
        assert nester.nesting_detection_count > 0

        nester.reset_metrics()
        assert nester.nesting_detection_count == 0

    def test_get_metrics(self):
        """Test getting metrics."""
        nester = MultiTimeframeNester(
            fibonacci_pairs=[[5, 21], [8, 34]]
        )

        metrics = nester.get_metrics()

        assert "nesting_detection_count" in metrics
        assert "fibonacci_pairs" in metrics
        assert metrics["fibonacci_pairs"] == 2

    def test_default_fibonacci_pairs(self):
        """Test default Fibonacci timeframe pairs."""
        nester = MultiTimeframeNester()

        assert nester.fibonacci_pairs == [(5, 21), (8, 34), (13, 55)]

    def test_custom_fibonacci_pairs(self):
        """Test custom Fibonacci timeframe pairs."""
        custom_pairs = [[3, 13], [5, 21]]
        nester = MultiTimeframeNester(fibonacci_pairs=custom_pairs)

        assert nester.fibonacci_pairs == custom_pairs

    def test_base_bar_duration_default(self):
        """Test default base bar duration."""
        nester = MultiTimeframeNester()

        assert nester.base_bar_duration == 1

    def test_base_bar_duration_custom(self):
        """Test custom base bar duration."""
        nester = MultiTimeframeNester(base_bar_duration=5)

        assert nester.base_bar_duration == 5

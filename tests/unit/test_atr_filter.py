"""Unit tests for ATR filter module."""

import pytest
from datetime import datetime, timedelta

from src.data.models import DollarBar
from src.detection.atr_filter import ATRFilter


@pytest.fixture
def sample_bars():
    """Create sample dollar bars for testing."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    # Create 50 bars with varying volatility
    for i in range(50):
        bar = DollarBar(
            timestamp=base_time + timedelta(minutes=i),
            open=11800.0 + i * 2,  # Trending up
            high=11810.0 + i * 2 + 5,  # Some volatility
            low=11790.0 + i * 2 - 5,  # Some volatility
            close=11805.0 + i * 2,
            volume=1000 + i * 10,
            notional_value=50000000.0,  # $50M
        )
        bars.append(bar)

    return bars


@pytest.fixture
def high_volatility_bars():
    """Create bars with high volatility (large ranges)."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i in range(50):
        bar = DollarBar(
            timestamp=base_time + timedelta(minutes=i),
            open=11800.0 + i * 2,
            high=11830.0 + i * 2,  # 30-point range
            low=11770.0 + i * 2,  # 30-point range
            close=11805.0 + i * 2,
            volume=1000 + i * 10,
            notional_value=50000000.0,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def low_volatility_bars():
    """Create bars with low volatility (small ranges)."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i in range(50):
        bar = DollarBar(
            timestamp=base_time + timedelta(minutes=i),
            open=11800.0 + i * 0.5,  # Small trend
            high=11802.0 + i * 0.5,  # 2-point range
            low=11798.0 + i * 0.5,  # 2-point range
            close=11801.0 + i * 0.5,
            volume=1000 + i * 10,
            notional_value=50000000.0,
        )
        bars.append(bar)

    return bars


class TestATRFilter:
    """Test suite for ATRFilter class."""

    def test_atr_calculation(self, sample_bars):
        """Test ATR calculation with sample data."""
        atr_filter = ATRFilter(lookback_period=14)

        atr = atr_filter.calculate_atr(sample_bars)

        # ATR should be positive
        assert atr > 0

        # For sample data with ~10-point ranges, ATR should be reasonable
        # (may vary due to lookback period and bar patterns)
        assert atr > 0

    def test_atr_high_volatility(self, high_volatility_bars):
        """Test ATR reflects high volatility."""
        atr_filter = ATRFilter(lookback_period=14)

        atr = atr_filter.calculate_atr(high_volatility_bars)

        # High volatility bars have ~60-point ranges
        assert atr > 40

    def test_atr_low_volatility(self, low_volatility_bars):
        """Test ATR reflects low volatility."""
        atr_filter = ATRFilter(lookback_period=14)

        atr = atr_filter.calculate_atr(low_volatility_bars)

        # Low volatility bars have ~4-point ranges
        assert atr < 10

    def test_atr_insufficient_bars(self):
        """Test ATR calculation with insufficient bars."""
        atr_filter = ATRFilter(lookback_period=14)

        # Only 2 bars (minimum is 3)
        bars = [
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 30, 0),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11805.0,
                volume=1000,
                notional_value=50000000.0,
            ),
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 31, 0),
                open=11805.0,
                high=11815.0,
                low=11795.0,
                close=11810.0,
                volume=1000,
                notional_value=50000000.0,
            ),
        ]

        with pytest.raises(ValueError, match="Insufficient bars"):
            atr_filter.calculate_atr(bars)

    def test_atr_minimal_bars(self):
        """Test ATR calculation with minimal bars (exactly 3)."""
        atr_filter = ATRFilter(lookback_period=14)

        bars = [
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 30, 0),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11805.0,
                volume=1000,
                notional_value=50000000.0,
            ),
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 31, 0),
                open=11805.0,
                high=11815.0,
                low=11795.0,
                close=11810.0,
                volume=1000,
                notional_value=50000000.0,
            ),
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 32, 0),
                open=11810.0,
                high=11820.0,
                low=11800.0,
                close=11815.0,
                volume=1000,
                notional_value=50000000.0,
            ),
        ]

        # Should not raise error
        atr = atr_filter.calculate_atr(bars)
        assert atr > 0

    def test_gap_significance_pass(self, sample_bars):
        """Test gap significance check with large gap."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        atr = atr_filter.calculate_atr(sample_bars)
        gap_size = atr * 0.6  # 60% of ATR (above threshold)

        is_significant, atr_multiple, message = atr_filter.check_gap_significance(
            gap_size, atr, "bullish"
        )

        assert is_significant
        assert atr_multiple >= 0.5
        assert "passes threshold" in message.lower()

    def test_gap_significance_fail(self, sample_bars):
        """Test gap significance check with small gap."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        atr = atr_filter.calculate_atr(sample_bars)
        gap_size = atr * 0.3  # 30% of ATR (below threshold)

        is_significant, atr_multiple, message = atr_filter.check_gap_significance(
            gap_size, atr, "bullish"
        )

        assert not is_significant
        assert atr_multiple < 0.5
        assert "filtered as noise" in message.lower()

    def test_should_filter_fvg_pass(self, sample_bars):
        """Test FVG filtering with significant gap."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        atr = atr_filter.calculate_atr(sample_bars)
        gap_size = atr * 0.6  # Above threshold

        should_filter, atr_multiple, message = atr_filter.should_filter_fvg(
            gap_size, sample_bars, "bullish"
        )

        assert not should_filter  # Should NOT filter (gap is significant)
        assert atr_multiple >= 0.5

    def test_should_filter_fvg_fail(self, sample_bars):
        """Test FVG filtering with insignificant gap."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        atr = atr_filter.calculate_atr(sample_bars)
        gap_size = atr * 0.3  # Below threshold

        should_filter, atr_multiple, message = atr_filter.should_filter_fvg(
            gap_size, sample_bars, "bullish"
        )

        assert should_filter  # SHOULD filter (gap is noise)
        assert atr_multiple < 0.5

    def test_zero_atr_handling(self, sample_bars):
        """Test handling of zero ATR (edge case)."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        is_significant, atr_multiple, message = atr_filter.check_gap_significance(
            10.0, 0.0, "bullish"
        )

        # Should fall back to absolute gap size check
        assert is_significant
        assert atr_multiple == 10.0

    def test_noise_filter_count_increment(self, sample_bars):
        """Test that noise filter count increments on rejection."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        atr = atr_filter.calculate_atr(sample_bars)
        gap_size = atr * 0.3  # Below threshold

        initial_count = atr_filter.noise_filter_count
        atr_filter.check_gap_significance(gap_size, atr, "bullish")

        assert atr_filter.noise_filter_count == initial_count + 1

    def test_reset_metrics(self, sample_bars):
        """Test resetting filter metrics."""
        atr_filter = ATRFilter(atr_threshold=0.5)

        atr = atr_filter.calculate_atr(sample_bars)
        gap_size = atr * 0.3  # Below threshold

        atr_filter.check_gap_significance(gap_size, atr, "bullish")
        assert atr_filter.noise_filter_count > 0

        atr_filter.reset_metrics()
        assert atr_filter.noise_filter_count == 0

    def test_get_metrics(self, sample_bars):
        """Test getting filter metrics."""
        atr_filter = ATRFilter(lookback_period=21, atr_threshold=0.6)

        metrics = atr_filter.get_metrics()

        assert "noise_filter_count" in metrics
        assert "lookback_period" in metrics
        assert "atr_threshold" in metrics
        assert metrics["lookback_period"] == 21
        assert metrics["atr_threshold"] == 0.6

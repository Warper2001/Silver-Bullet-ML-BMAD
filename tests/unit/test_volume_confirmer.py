"""Unit tests for volume confirmer module."""

import pytest
from datetime import datetime, timedelta

from src.data.models import DollarBar
from src.detection.volume_confirmer import VolumeConfirmer


@pytest.fixture
def bullish_volume_bars():
    """Create bars with strong bullish volume (up bars dominate)."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i in range(30):
        # 80% up bars with high volume
        if i % 5 != 0:  # 4 out of 5 bars are up
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11799.0 + i,
                close=11804.0 + i,  # Close > Open (up bar)
                volume=2000,  # High volume
                notional_value=50000000.0,
            )
        else:  # 1 out of 5 bars is down
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11802.0 + i,
                low=11798.0 + i,
                close=11799.0 + i,  # Close < Open (down bar)
                volume=500,  # Low volume
                notional_value=50000000.0,
            )
        bars.append(bar)

    return bars


@pytest.fixture
def bearish_volume_bars():
    """Create bars with strong bearish volume (down bars dominate)."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i in range(30):
        # 80% down bars with high volume
        if i % 5 != 0:  # 4 out of 5 bars are down
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11802.0 + i,
                low=11798.0 + i,
                close=11799.0 + i,  # Close < Open (down bar)
                volume=2000,  # High volume
                notional_value=50000000.0,
            )
        else:  # 1 out of 5 bars is up
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11799.0 + i,
                close=11804.0 + i,  # Close > Open (up bar)
                volume=500,  # Low volume
                notional_value=50000000.0,
            )
        bars.append(bar)

    return bars


@pytest.fixture
def mixed_volume_bars():
    """Create bars with balanced volume (no clear direction)."""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i in range(30):
        # Alternating up/down with equal volume
        if i % 2 == 0:
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11799.0 + i,
                close=11804.0 + i,  # Up bar
                volume=1000,
                notional_value=50000000.0,
            )
        else:
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11802.0 + i,
                low=11798.0 + i,
                close=11799.0 + i,  # Down bar
                volume=1000,
                notional_value=50000000.0,
            )
        bars.append(bar)

    return bars


class TestVolumeConfirmer:
    """Test suite for VolumeConfirmer class."""

    def test_calculate_volume_ratios_bullish(self, bullish_volume_bars):
        """Test volume ratio calculation with bullish dominance."""
        confirmer = VolumeConfirmer(lookback_period=20)

        ratios = confirmer.calculate_volume_ratios(bullish_volume_bars)

        assert ratios["up_volume"] > ratios["down_volume"]
        assert ratios["up_volume_ratio"] > 2.0  # Strong bullish signal
        assert ratios["down_volume_ratio"] < 1.0

    def test_calculate_volume_ratios_bearish(self, bearish_volume_bars):
        """Test volume ratio calculation with bearish dominance."""
        confirmer = VolumeConfirmer(lookback_period=20)

        ratios = confirmer.calculate_volume_ratios(bearish_volume_bars)

        assert ratios["down_volume"] > ratios["up_volume"]
        assert ratios["down_volume_ratio"] > 2.0  # Strong bearish signal
        assert ratios["up_volume_ratio"] < 1.0

    def test_calculate_volume_ratios_mixed(self, mixed_volume_bars):
        """Test volume ratio calculation with balanced volume."""
        confirmer = VolumeConfirmer(lookback_period=20)

        ratios = confirmer.calculate_volume_ratios(mixed_volume_bars)

        # Should be roughly balanced
        assert 0.5 < ratios["up_volume_ratio"] < 2.0
        assert 0.5 < ratios["down_volume_ratio"] < 2.0

    def test_calculate_volume_ratios_insufficient_bars(self):
        """Test volume ratio calculation with insufficient bars."""
        confirmer = VolumeConfirmer(lookback_period=20)

        # Only 1 bar
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
        ]

        with pytest.raises(ValueError, match="Insufficient bars"):
            confirmer.calculate_volume_ratios(bars)

    def test_check_volume_confirmation_bullish_pass(self, bullish_volume_bars):
        """Test volume confirmation for bullish FVG with strong volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        ratios = confirmer.calculate_volume_ratios(bullish_volume_bars)
        is_confirmed, volume_ratio, message = confirmer.check_volume_confirmation(
            "bullish", ratios
        )

        assert is_confirmed
        assert volume_ratio >= 1.5
        assert "passes threshold" in message.lower()

    def test_check_volume_confirmation_bullish_fail(self, mixed_volume_bars):
        """Test volume confirmation for bullish FVG with weak volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        ratios = confirmer.calculate_volume_ratios(mixed_volume_bars)
        is_confirmed, volume_ratio, message = confirmer.check_volume_confirmation(
            "bullish", ratios
        )

        assert not is_confirmed
        assert volume_ratio < 1.5
        assert "filtered" in message.lower()

    def test_check_volume_confirmation_bearish_pass(self, bearish_volume_bars):
        """Test volume confirmation for bearish FVG with strong volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        ratios = confirmer.calculate_volume_ratios(bearish_volume_bars)
        is_confirmed, volume_ratio, message = confirmer.check_volume_confirmation(
            "bearish", ratios
        )

        assert is_confirmed
        assert volume_ratio >= 1.5
        assert "passes threshold" in message.lower()

    def test_check_volume_confirmation_bearish_fail(self, mixed_volume_bars):
        """Test volume confirmation for bearish FVG with weak volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        ratios = confirmer.calculate_volume_ratios(mixed_volume_bars)
        is_confirmed, volume_ratio, message = confirmer.check_volume_confirmation(
            "bearish", ratios
        )

        assert not is_confirmed
        assert volume_ratio < 1.5
        assert "filtered" in message.lower()

    def test_should_filter_fvg_bullish_pass(self, bullish_volume_bars):
        """Test FVG filtering for bullish direction with strong volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        should_filter, volume_ratio, message = confirmer.should_filter_fvg(
            "bullish", bullish_volume_bars
        )

        assert not should_filter  # Should NOT filter (volume confirms)
        assert volume_ratio >= 1.5

    def test_should_filter_fvg_bullish_fail(self, mixed_volume_bars):
        """Test FVG filtering for bullish direction with weak volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        should_filter, volume_ratio, message = confirmer.should_filter_fvg(
            "bullish", mixed_volume_bars
        )

        assert should_filter  # SHOULD filter (volume doesn't confirm)
        assert volume_ratio < 1.5

    def test_should_filter_fvg_bearish_pass(self, bearish_volume_bars):
        """Test FVG filtering for bearish direction with strong volume."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        should_filter, volume_ratio, message = confirmer.should_filter_fvg(
            "bearish", bearish_volume_bars
        )

        assert not should_filter  # Should NOT filter (volume confirms)
        assert volume_ratio >= 1.5

    def test_zero_volume_handling(self):
        """Test handling of zero volume (edge case)."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        # Create bars with zero volume (closed market)
        bars = [
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 30, 0),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0,
                notional_value=0.0,
            ),
            DollarBar(
                timestamp=datetime(2024, 1, 1, 9, 31, 0),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0,
                notional_value=0.0,
            ),
        ]

        # Should handle gracefully (all bars are flat with zero volume)
        ratios = confirmer.calculate_volume_ratios(bars)
        assert ratios["total_volume"] == 0
        assert ratios["up_volume"] == 0
        assert ratios["down_volume"] == 0
        assert ratios["flat_bar_count"] == 2

    def test_volume_filter_count_increment(self, mixed_volume_bars):
        """Test that volume filter count increments on rejection."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        ratios = confirmer.calculate_volume_ratios(mixed_volume_bars)
        initial_count = confirmer.volume_filter_count

        confirmer.check_volume_confirmation("bullish", ratios)

        assert confirmer.volume_filter_count == initial_count + 1

    def test_reset_metrics(self, mixed_volume_bars):
        """Test resetting filter metrics."""
        confirmer = VolumeConfirmer(volume_ratio_threshold=1.5)

        ratios = confirmer.calculate_volume_ratios(mixed_volume_bars)
        confirmer.check_volume_confirmation("bullish", ratios)

        assert confirmer.volume_filter_count > 0

        confirmer.reset_metrics()
        assert confirmer.volume_filter_count == 0

    def test_get_metrics(self):
        """Test getting filter metrics."""
        confirmer = VolumeConfirmer(
            lookback_period=25, volume_ratio_threshold=2.0
        )

        metrics = confirmer.get_metrics()

        assert "volume_filter_count" in metrics
        assert "lookback_period" in metrics
        assert "volume_ratio_threshold" in metrics
        assert metrics["lookback_period"] == 25
        assert metrics["volume_ratio_threshold"] == 2.0

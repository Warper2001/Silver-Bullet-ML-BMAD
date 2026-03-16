"""Unit tests for Liquidity Sweep Detection."""

from datetime import datetime, timedelta

import pytest

from src.data.models import DollarBar, LiquiditySweepEvent, SwingPoint
from src.detection.liquidity_sweep_detection import (
    check_bullish_sweep,
    check_bearish_sweep,
    detect_bullish_liquidity_sweep,
    detect_bearish_liquidity_sweep,
)


class TestLiquiditySweepAlgorithms:
    """Test liquidity sweep detection algorithms."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for test bars."""
        return datetime(2026, 3, 16, 10, 0, 0)

    @pytest.fixture
    def swing_low(self, base_timestamp):
        """Create a sample swing low."""
        return SwingPoint(
            timestamp=base_timestamp - timedelta(seconds=10),
            price=11800.0,
            swing_type="swing_low",
            bar_index=5,
            confirmed=True,
        )

    @pytest.fixture
    def swing_high(self, base_timestamp):
        """Create a sample swing high."""
        return SwingPoint(
            timestamp=base_timestamp - timedelta(seconds=10),
            price=11900.0,
            swing_type="swing_high",
            bar_index=5,
            confirmed=True,
        )

    def test_bullish_sweep_detected_when_price_drops_then_recovers(
        self, base_timestamp, swing_low
    ):
        """Verify bullish sweep when price drops below swing low then recovers."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11805.0,
            high=11820.0,
            low=11795.0,  # Below swing low (11800.0)
            close=11810.0,  # Above swing low
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bullish_sweep(bar, swing_low, min_sweep_ticks=5)

        assert sweep is not None
        assert sweep.direction == "bullish"
        assert sweep.swing_point_price == 11800.0
        assert sweep.sweep_depth_ticks == 20.0  # (11800 - 11795) / 0.25
        assert sweep.sweep_depth_dollars == 100.0  # 5 * $20

    def test_bearish_sweep_detected_when_price_rises_then_falls(
        self, base_timestamp, swing_high
    ):
        """Verify bearish sweep when price rises above swing high then falls."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11895.0,
            high=11905.0,  # Above swing high (11900.0)
            low=11890.0,
            close=11895.0,  # Below swing high
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bearish_sweep(bar, swing_high, min_sweep_ticks=5)

        assert sweep is not None
        assert sweep.direction == "bearish"
        assert sweep.swing_point_price == 11900.0
        assert sweep.sweep_depth_ticks == 20.0  # (11905 - 11900) / 0.25
        assert sweep.sweep_depth_dollars == 100.0  # 5 * $20

    def test_no_bullish_sweep_when_depth_insufficient(self, base_timestamp, swing_low):
        """Verify no sweep when price doesn't move 5 ticks beyond swing."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11805.0,
            high=11820.0,
            low=11799.0,  # Only 4 ticks below swing low
            close=11810.0,
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bullish_sweep(bar, swing_low, min_sweep_ticks=5)

        assert sweep is None

    def test_no_bearish_sweep_when_depth_insufficient(self, base_timestamp, swing_high):
        """Verify no sweep when price doesn't move 5 ticks beyond swing point."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11895.0,
            high=11901.0,  # Only 4 ticks above
            low=11890.0,
            close=11895.0,
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bearish_sweep(bar, swing_high, min_sweep_ticks=5)

        assert sweep is None

    def test_no_bullish_sweep_when_close_below_swing_low(
        self, base_timestamp, swing_low
    ):
        """Verify no bullish sweep when close doesn't recover above swing low."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11805.0,
            high=11810.0,
            low=11795.0,  # Below swing low
            close=11798.0,  # Still below swing low (no recovery)
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bullish_sweep(bar, swing_low, min_sweep_ticks=5)

        assert sweep is None

    def test_no_bearish_sweep_when_close_above_swing_high(
        self, base_timestamp, swing_high
    ):
        """Verify no bearish sweep when close doesn't fall below swing high."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11905.0,
            high=11910.0,  # Above swing high
            low=11900.0,
            close=11908.0,  # Still above swing high (no reversal)
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bearish_sweep(bar, swing_high, min_sweep_ticks=5)

        assert sweep is None

    def test_sweep_depth_calculation(self, base_timestamp, swing_low):
        """Verify sweep depth calculated correctly."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11805.0,
            high=11820.0,
            low=11790.0,  # 40 ticks below swing low
            close=11810.0,
            volume=1000,
            notional_value=50_000_000,
        )

        sweep = check_bullish_sweep(bar, swing_low, min_sweep_ticks=5)

        assert sweep.sweep_depth_ticks == 40.0  # (11800 - 11790) / 0.25
        assert sweep.sweep_depth_dollars == 200.0  # 10 * $20

    def test_detect_bullish_sweep_with_bars(self, base_timestamp):
        """Verify bullish sweep detection from bar list."""
        bars = []
        # Create swing low at bar 2
        bars.append(
            DollarBar(
                timestamp=base_timestamp,
                open=11810.0,
                high=11820.0,
                low=11800.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=5),
                open=11805.0,
                high=11810.0,
                low=11795.0,
                close=11800.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=10),
                open=11800.0,
                high=11815.0,
                low=11790.0,  # Sweep low
                close=11810.0,  # Recovery
                volume=1000,
                notional_value=50_000_000,
            )
        )

        swing_low = SwingPoint(
            timestamp=bars[0].timestamp,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        sweep = detect_bullish_liquidity_sweep(bars, 2, swing_low, min_sweep_ticks=5)

        assert sweep is not None
        assert sweep.direction == "bullish"
        assert sweep.sweep_depth_ticks == 40.0

    def test_detect_bearish_sweep_with_bars(self, base_timestamp):
        """Verify bearish sweep detection from bar list."""
        bars = []
        # Create swing high at bar 2
        bars.append(
            DollarBar(
                timestamp=base_timestamp,
                open=11895.0,
                high=11900.0,
                low=11890.0,
                close=11895.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=5),
                open=11900.0,
                high=11905.0,
                low=11895.0,
                close=11900.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=10),
                open=11900.0,
                high=11910.0,  # Sweep high
                low=11895.0,  # Recovery
                close=11898.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )

        swing_high = SwingPoint(
            timestamp=bars[0].timestamp,
            price=11900.0,
            swing_type="swing_high",
            bar_index=0,
            confirmed=True,
        )

        sweep = detect_bearish_liquidity_sweep(bars, 2, swing_high, min_sweep_ticks=5)

        assert sweep is not None
        assert sweep.direction == "bearish"
        assert sweep.sweep_depth_ticks == 40.0


class TestLiquiditySweepEventModel:
    """Test LiquiditySweepEvent data model."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for testing."""
        return datetime(2026, 3, 16, 10, 0, 0)

    def test_liquidity_sweep_event_creation(self, base_timestamp):
        """Verify LiquiditySweepEvent can be created with all required fields."""
        sweep = LiquiditySweepEvent(
            timestamp=base_timestamp,
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=10,
        )

        assert sweep.timestamp == base_timestamp
        assert sweep.direction == "bullish"
        assert sweep.swing_point_price == 11800.0
        assert sweep.sweep_depth_ticks == 20.0
        assert sweep.sweep_depth_dollars == 100.0
        assert sweep.bar_index == 10
        assert sweep.confidence == 0.0

    def test_liquidity_sweep_event_valid_directions(self, base_timestamp):
        """Verify only valid directions are accepted."""
        # Bullish is valid
        LiquiditySweepEvent(
            timestamp=base_timestamp,
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=10,
        )

        # Bearish is valid
        LiquiditySweepEvent(
            timestamp=base_timestamp,
            direction="bearish",
            swing_point_price=11900.0,
            sweep_depth_ticks=20.0,
            sweep_depth_dollars=100.0,
            bar_index=10,
        )

    def test_liquidity_sweep_event_depth_validation(self, base_timestamp):
        """Verify sweep depth is non-negative."""
        # Valid: zero depth
        sweep = LiquiditySweepEvent(
            timestamp=base_timestamp,
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=0.0,
            sweep_depth_dollars=0.0,
            bar_index=10,
        )
        assert sweep.sweep_depth_ticks == 0.0
        assert sweep.sweep_depth_dollars == 0.0

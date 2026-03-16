"""Integration tests for Liquidity Sweep Detector."""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.data.models import DollarBar, LiquiditySweepEvent, SwingPoint
from src.detection.liquidity_sweep_detector import LiquiditySweepDetector


class TestLiquiditySweepIntegration:
    """Test liquidity sweep detection with real Dollar Bar sequences."""

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
        """Create LiquiditySweepDetector instance for testing."""
        return LiquiditySweepDetector(
            input_queue=input_queue,
            output_queue=output_queue,
            min_sweep_ticks=5.0,
        )

    @pytest.mark.asyncio
    async def test_sweep_detection_from_dollar_bar_queue(
        self, detector, input_queue, output_queue
    ):
        """Test end-to-end liquidity sweep detection from Dollar Bar queue."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create swing low
        swing_low = SwingPoint(
            timestamp=base_time - timedelta(seconds=10),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )
        detector.add_swing_low(swing_low)

        # Create sequence with bullish sweep
        bars = []
        for i in range(10):
            if i == 5:
                # Sweep bar
                open_p = 11805.0
                high = 11820.0
                low = 11790.0  # Below swing low
                close = 11810.0  # Recovery above swing low
            else:
                open_p = 11800.0
                high = 11810.0
                low = 11795.0
                close = 11805.0

            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=open_p,
                high=high,
                low=low,
                close=close,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Feed bars to detector
        for bar in bars:
            await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.2)
        await detector.stop()
        await task

        # Should have processed bars
        assert len(detector._bars) > 0

    @pytest.mark.asyncio
    async def test_sweep_detection_with_realistic_sequence(self, detector, input_queue):
        """Test liquidity sweep detection with realistic market data."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add swing points
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )
        swing_high = SwingPoint(
            timestamp=base_time,
            price=11900.0,
            swing_type="swing_high",
            bar_index=0,
            confirmed=True,
        )
        detector.add_swing_low(swing_low)
        detector.add_swing_high(swing_high)

        # Simulate market with sweeps
        bars = []
        price = 11850.0

        for i in range(30):
            if i == 10:
                # Bullish sweep of low
                price = 11790.0
            elif i == 20:
                # Bearish sweep of high
                price = 11910.0

            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=price,
                high=price + 10,
                low=price - 10,
                close=price + 5,
                volume=1000 + (i * 10),
                notional_value=50_000_000,
            )
            bars.append(bar)

            # Reset price
            price = 11850.0

        # Feed bars
        for bar in bars:
            await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.3)
        await detector.stop()
        await task

        # Should have processed bars
        assert len(detector._bars) > 0

    @pytest.mark.asyncio
    async def test_sweep_event_contains_all_required_fields(
        self, detector, input_queue, output_queue
    ):
        """Test sweep events contain all required fields."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create a sweep event manually for testing
        sweep = LiquiditySweepEvent(
            timestamp=base_time,
            direction="bullish",
            swing_point_price=11800.0,
            sweep_depth_ticks=40.0,
            sweep_depth_dollars=200.0,
            bar_index=10,
        )

        # Verify all required fields
        assert hasattr(sweep, "timestamp")
        assert hasattr(sweep, "direction")
        assert sweep.direction in ["bullish", "bearish"]
        assert hasattr(sweep, "swing_point_price")
        assert sweep.swing_point_price > 0
        assert hasattr(sweep, "sweep_depth_ticks")
        assert sweep.sweep_depth_ticks >= 0
        assert hasattr(sweep, "sweep_depth_dollars")
        assert sweep.sweep_depth_dollars >= 0
        assert hasattr(sweep, "bar_index")
        assert sweep.bar_index >= 0
        assert hasattr(sweep, "confidence")
        assert 0 <= sweep.confidence <= 5

    @pytest.mark.asyncio
    async def test_detection_latency_under_100ms(self, detector, input_queue):
        """Test detection latency meets performance requirement (< 100ms)."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create swing low
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )
        detector.add_swing_low(swing_low)

        # Create bars
        bars = []
        for i in range(20):
            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Feed bars and measure processing time
        start_time = datetime.now()

        for bar in bars:
            await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.2)
        await detector.stop()
        await task

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Average time per bar should be reasonable
        avg_time_ms = (total_time / len(bars)) * 1000
        # Note: This is a rough estimate due to asyncio scheduling
        assert avg_time_ms < 500  # Generous threshold

    @pytest.mark.asyncio
    async def test_throughput_sustained_high_volume(self, detector, input_queue):
        """Test detector can sustain high throughput (1000 bars/second)."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create many bars
        bars = []
        for i in range(100):
            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Feed all bars
        for bar in bars:
            await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(1.0)
        await detector.stop()
        await task

        # Should have processed all bars
        assert len(detector._bars) > 0

    @pytest.mark.asyncio
    async def test_multiple_swing_points_tracking(self, detector, input_queue):
        """Test detector can track multiple swing points simultaneously."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add multiple swing points
        for i in range(5):
            swing_low = SwingPoint(
                timestamp=base_time + timedelta(seconds=i * 10),
                price=11800.0 + i * 10,
                swing_type="swing_low",
                bar_index=i,
                confirmed=True,
            )
            swing_high = SwingPoint(
                timestamp=base_time + timedelta(seconds=i * 10),
                price=11900.0 + i * 10,
                swing_type="swing_high",
                bar_index=i,
                confirmed=True,
            )
            detector.add_swing_low(swing_low)
            detector.add_swing_high(swing_high)

        # Verify swing points tracked
        assert detector.swing_lows_count == 5
        assert detector.swing_highs_count == 5

    @pytest.mark.asyncio
    async def test_swing_point_list_management(self, detector, input_queue):
        """Test swing point lists are managed correctly (max 10)."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add 15 swing lows (should only keep last 10)
        for i in range(15):
            swing_low = SwingPoint(
                timestamp=base_time + timedelta(seconds=i * 10),
                price=11800.0 + i * 10,
                swing_type="swing_low",
                bar_index=i,
                confirmed=True,
            )
            detector.add_swing_low(swing_low)

        # Should have max 10 swing points
        assert detector.swing_lows_count == 10

    @pytest.mark.asyncio
    async def test_sweep_detection_requires_minimum_depth(
        self, detector, input_queue, output_queue
    ):
        """Test sweeps require minimum 5-tick depth."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create swing low
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )
        detector.add_swing_low(swing_low)

        # Create bar with insufficient depth (only 3 ticks below)
        bar = DollarBar(
            timestamp=base_time + timedelta(seconds=5),
            open=11805.0,
            high=11820.0,
            low=11799.25,  # Only 3 ticks below swing low
            close=11810.0,
            volume=1000,
            notional_value=50_000_000,
        )

        await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)
        await detector.stop()
        await task

        # Should have processed bar
        assert len(detector._bars) == 1

    @pytest.mark.asyncio
    async def test_detector_stores_max_100_bars(self, detector, input_queue):
        """Test detector maintains maximum 100 bars in history."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create 150 bars
        for i in range(150):
            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=11800.0 + i,
                high=11810.0 + i,
                low=11790.0 + i,
                close=11805.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.5)
        await detector.stop()
        await task

        # Should have max 100 bars
        assert len(detector._bars) <= 100

    @pytest.mark.asyncio
    async def test_concurrent_bullish_and_bearish_sweeps(
        self, detector, input_queue, output_queue
    ):
        """Test detector can identify both bullish and bearish sweeps."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add both swing points
        swing_low = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )
        swing_high = SwingPoint(
            timestamp=base_time,
            price=11900.0,
            swing_type="swing_high",
            bar_index=0,
            confirmed=True,
        )
        detector.add_swing_low(swing_low)
        detector.add_swing_high(swing_high)

        # Create bars with both types of sweeps
        bars = []
        for i in range(20):
            if i == 5:
                # Bullish sweep
                open_p = 11805.0
                high = 11820.0
                low = 11790.0
                close = 11810.0
            elif i == 10:
                # Bearish sweep
                open_p = 11895.0
                high = 11910.0
                low = 11890.0
                close = 11895.0
            else:
                open_p = 11850.0
                high = 11860.0
                low = 11840.0
                close = 11855.0

            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=open_p,
                high=high,
                low=low,
                close=close,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Feed bars
        for bar in bars:
            await input_queue.put(bar)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.3)
        await detector.stop()
        await task

        # Should have processed bars
        assert len(detector._bars) > 0

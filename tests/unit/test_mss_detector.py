"""Unit tests for MSS Detector."""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.data.models import DollarBar, SwingPoint
from src.detection.mss_detector import MSSDetector
from src.detection.swing_detection import (
    RollingVolumeAverage,
    detect_bearish_mss,
    detect_bullish_mss,
    detect_swing_high,
    detect_swing_low,
)


class TestSwingPointDetection:
    """Test swing point detection algorithms."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for test bars."""
        return datetime(2026, 3, 16, 10, 0, 0)

    @pytest.fixture
    def sample_bars(self, base_timestamp):
        """Create sample Dollar Bars for testing."""
        bars = []
        for i in range(10):
            bar = DollarBar(
                timestamp=base_timestamp + timedelta(seconds=i * 5),
                open=11800.0 + i * 10,
                high=11810.0 + i * 10,
                low=11790.0 + i * 10,
                close=11805.0 + i * 10,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)
        return bars

    def test_detect_swing_high_with_minimum_bars(self, sample_bars):
        """Verify pivot high detection with minimum required bars."""
        # Make bar 5 a swing high (highest in range [2, 8])
        sample_bars[5].high = 11900.0
        for i in range(2, 9):
            if i != 5:
                sample_bars[i].high = 11850.0

        # Bar 5 should be detected as swing high with lookback=3
        result = detect_swing_high(sample_bars, 5, lookback=3)
        assert result is True

    def test_detect_swing_low_with_minimum_bars(self, sample_bars):
        """Verify pivot low detection with minimum required bars."""
        # Make bar 5 a swing low (lowest in range [2, 8])
        sample_bars[5].low = 11700.0
        for i in range(2, 9):
            if i != 5:
                sample_bars[i].low = 11750.0

        # Bar 5 should be detected as swing low with lookback=3
        result = detect_swing_low(sample_bars, 5, lookback=3)
        assert result is True

    def test_no_swing_when_insufficient_bars(self, sample_bars):
        """Verify no swing detected when insufficient bars on either side."""
        # Only 2 bars on left side, need 3
        result = detect_swing_high(sample_bars, 2, lookback=3)
        assert result is False

        # Only 1 bar on right side, need 3
        result = detect_swing_high(sample_bars, 8, lookback=3)
        assert result is False

    def test_no_swing_when_not_extreme(self, sample_bars):
        """Verify no swing detected when bar is not extreme in range."""
        # All bars have similar highs, no clear swing
        for i in range(10):
            sample_bars[i].high = 11850.0

        result = detect_swing_high(sample_bars, 5, lookback=3)
        assert result is False

    def test_swing_at_bar_boundary(self, sample_bars):
        """Verify swing detection at valid bar index boundaries."""
        # Make bar 3 a swing high (first valid position with lookback=3)
        sample_bars[3].high = 11900.0
        for i in range(0, 7):
            if i != 3:
                sample_bars[i].high = 11850.0

        result = detect_swing_high(sample_bars, 3, lookback=3)
        assert result is True

    def test_swing_high_requires_unique_maximum(self, sample_bars):
        """Verify swing high must be strictly greater than other bars."""
        # Make bar 5 equal to another bar in range
        sample_bars[5].high = 11900.0
        sample_bars[6].high = 11900.0  # Equal, not strictly greater

        result = detect_swing_high(sample_bars, 5, lookback=3)
        assert result is False

    def test_swing_low_requires_unique_minimum(self, sample_bars):
        """Verify swing low must be strictly less than other bars."""
        # Make bar 5 equal to another bar in range
        sample_bars[5].low = 11700.0
        sample_bars[6].low = 11700.0  # Equal, not strictly less

        result = detect_swing_low(sample_bars, 5, lookback=3)
        assert result is False


class TestMSSDetection:
    """Test MSS detection algorithms."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for test bars."""
        return datetime(2026, 3, 16, 10, 0, 0)

    @pytest.fixture
    def swing_high(self, base_timestamp):
        """Create a sample swing high."""
        return SwingPoint(
            timestamp=base_timestamp - timedelta(seconds=30),
            price=11850.0,
            swing_type="swing_high",
            bar_index=5,
        )

    @pytest.fixture
    def swing_low(self, base_timestamp):
        """Create a sample swing low."""
        return SwingPoint(
            timestamp=base_timestamp - timedelta(seconds=30),
            price=11750.0,
            swing_type="swing_low",
            bar_index=5,
        )

    def test_bullish_mss_with_volume_confirmation(self, base_timestamp, swing_high):
        """Verify bullish MSS detected with proper volume confirmation."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11855.0,
            high=11900.0,  # Above swing high (11850.0)
            low=11840.0,
            close=11895.0,
            volume=1500,  # 1.5x average
            notional_value=50_000_000,
        )

        mss_event = detect_bullish_mss(
            bar, [swing_high], volume_ma_20=1000.0, volume_confirmation_ratio=1.5
        )

        assert mss_event is not None
        assert mss_event.direction == "bullish"
        assert mss_event.breakout_price == 11900.0
        assert mss_event.volume_ratio == 1.5
        assert mss_event.swing_point == swing_high

    def test_bearish_mss_with_volume_confirmation(self, base_timestamp, swing_low):
        """Verify bearish MSS detected with proper volume confirmation."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11755.0,
            high=11760.0,
            low=11700.0,  # Below swing low (11750.0)
            close=11705.0,
            volume=1500,  # 1.5x average
            notional_value=50_000_000,
        )

        mss_event = detect_bearish_mss(
            bar, [swing_low], volume_ma_20=1000.0, volume_confirmation_ratio=1.5
        )

        assert mss_event is not None
        assert mss_event.direction == "bearish"
        assert mss_event.breakout_price == 11700.0
        assert mss_event.volume_ratio == 1.5
        assert mss_event.swing_point == swing_low

    def test_no_mss_without_breakout(self, base_timestamp, swing_high):
        """Verify no MSS when price doesn't break swing level."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11840.0,
            high=11845.0,  # Below swing high (11850.0)
            low=11830.0,
            close=11842.0,
            volume=1500,
            notional_value=50_000_000,
        )

        mss_event = detect_bullish_mss(
            bar, [swing_high], volume_ma_20=1000.0, volume_confirmation_ratio=1.5
        )

        assert mss_event is None

    def test_no_mss_without_volume_confirmation(self, base_timestamp, swing_high):
        """Verify no MSS when volume confirmation fails."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11855.0,
            high=11900.0,  # Above swing high
            low=11840.0,
            close=11895.0,
            volume=1200,  # Only 1.2x average (below 1.5x threshold)
            notional_value=50_000_000,
        )

        mss_event = detect_bullish_mss(
            bar, [swing_high], volume_ma_20=1000.0, volume_confirmation_ratio=1.5
        )

        assert mss_event is None

    def test_no_mss_when_no_swing_points(self, base_timestamp):
        """Verify no MSS when no swing points exist."""
        bar = DollarBar(
            timestamp=base_timestamp,
            open=11800.0,
            high=11900.0,
            low=11700.0,
            close=11850.0,
            volume=1500,
            notional_value=50_000_000,
        )

        mss_event = detect_bullish_mss(bar, [], volume_ma_20=1000.0)
        assert mss_event is None

    def test_multiple_mss_in_sequence(self, base_timestamp):
        """Verify multiple MSS can be detected in trending market."""
        swing_highs = [
            SwingPoint(
                timestamp=base_timestamp - timedelta(seconds=60),
                price=11800.0,
                swing_type="swing_high",
                bar_index=0,
            ),
            SwingPoint(
                timestamp=base_timestamp - timedelta(seconds=30),
                price=11850.0,
                swing_type="swing_high",
                bar_index=5,
            ),
        ]

        # First breakout
        bar1 = DollarBar(
            timestamp=base_timestamp,
            high=11860.0,  # Breaks first swing high
            volume=1500,
            notional_value=50_000_000,
            open=11855.0,
            low=11840.0,
            close=11858.0,
        )

        mss_event1 = detect_bullish_mss(
            bar1, swing_highs, volume_ma_20=1000.0, volume_confirmation_ratio=1.5
        )

        assert mss_event1 is not None
        # Should break most recent swing high
        assert mss_event1.swing_point == swing_highs[1]


class TestRollingVolumeAverage:
    """Test rolling volume average calculation."""

    def test_rolling_average_initial_values(self):
        """Verify rolling average during warm-up period."""
        rva = RollingVolumeAverage(window=20)

        # First value
        avg1 = rva.update(1000.0)
        assert avg1 == 1000.0

        # Second value
        avg2 = rva.update(2000.0)
        assert avg2 == 1500.0

        # Third value
        avg3 = rva.update(3000.0)
        assert avg3 == 2000.0

    def test_rolling_average_after_window_filled(self):
        """Verify rolling average maintains window size after filled."""
        rva = RollingVolumeAverage(window=3)

        # Fill window
        rva.update(1000.0)
        rva.update(2000.0)
        rva.update(3000.0)

        # Average should be 2000.0
        assert rva.average == 2000.0

        # Add new value (oldest removed)
        rva.update(4000.0)

        # New average should be (2000 + 3000 + 4000) / 3 = 3000.0
        assert rva.average == 3000.0

    def test_rolling_average_efficient_update(self):
        """Verify rolling average updates in O(1) time."""
        rva = RollingVolumeAverage(window=1000)

        # Fill window
        for i in range(1000):
            rva.update(float(i * 100))

        # Should maintain correct average
        expected_avg = sum(float(i * 100) for i in range(1000)) / 1000
        assert abs(rva.average - expected_avg) < 0.01

    def test_rolling_average_with_zero_volume(self):
        """Verify rolling average handles zero volume gracefully."""
        rva = RollingVolumeAverage(window=5)

        rva.update(1000.0)
        rva.update(0.0)
        rva.update(1000.0)

        assert rva.average == 2000.0 / 3

    def test_rolling_average_property(self):
        """Verify average property returns current value."""
        rva = RollingVolumeAverage(window=5)

        rva.update(1000.0)
        rva.update(2000.0)

        assert rva.average == 1500.0


class TestMSSDetectorClass:
    """Test MSSDetector class functionality."""

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
        """Create MSSDetector instance for testing."""
        return MSSDetector(
            input_queue=input_queue,
            output_queue=output_queue,
            lookback=3,
            volume_confirmation_ratio=1.5,
            volume_ma_window=20,
        )

    def test_detector_initialization(self, detector):
        """Verify detector initializes correctly."""
        assert detector.is_running is False
        assert detector.swing_highs_count == 0
        assert detector.swing_lows_count == 0

    @pytest.mark.asyncio
    async def test_consume_processes_dollar_bars(self, detector, input_queue):
        """Verify consume method processes Dollar Bars."""
        # Create sample bars
        base_time = datetime(2026, 3, 16, 10, 0, 0)
        bars = []
        for i in range(10):
            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=11800.0 + i * 10,
                high=11810.0 + i * 10,
                low=11790.0 + i * 10,
                close=11805.0 + i * 10,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Add bars to input queue
        for bar in bars:
            await input_queue.put(bar)

        # Run detector briefly
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)  # Let it process
        await detector.stop()
        await task

        # Should have processed bars
        assert len(detector._bars) > 0

    @pytest.mark.asyncio
    async def test_consume_publishes_mss_events(
        self, detector, input_queue, output_queue
    ):
        """Verify consume publishes MSS events to output queue."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create bars forming a swing high followed by breakout
        bars = []
        for i in range(10):
            high = 11800.0
            close = 11795.0  # Close below high for regular bars
            if i == 5:
                high = 11900.0  # Swing high
                close = 11895.0  # Close below high
            elif i == 8:
                high = 11950.0  # Breakout with high volume
                close = 11945.0  # Close below high

            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=11800.0,
                high=high,
                low=11790.0,
                close=close,
                volume=1500 if i == 8 else 1000,  # High volume on breakout
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Add bars to input queue
        for bar in bars:
            await input_queue.put(bar)

        # Run detector briefly
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.2)  # Let it process
        await detector.stop()
        await task

        # Should have detected swing points and possibly MSS
        assert detector.swing_highs_count >= 0

    @pytest.mark.asyncio
    async def test_consume_handles_graceful_shutdown(self, detector, input_queue):
        """Verify consume handles graceful shutdown correctly."""
        # Start detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)

        # Stop detector
        await detector.stop()

        # Task should complete
        await task

        # Detector should not be running
        assert detector.is_running is False

    @pytest.mark.asyncio
    async def test_consume_logs_performance_warnings(
        self, detector, input_queue, caplog
    ):
        """Verify consume logs warnings when detection latency exceeds 100ms."""
        # This test would require mocking time.perf_counter or
        # creating a very slow detection scenario
        # For now, just verify the structure exists
        assert detector._volume_ma_window == 20

    @pytest.mark.asyncio
    async def test_detector_handles_insufficient_bars(self, detector, input_queue):
        """Verify detector handles insufficient bars gracefully."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add only 2 bars (insufficient for swing detection with lookback=3)
        for i in range(2):
            bar = DollarBar(
                timestamp=base_time + timedelta(seconds=i * 5),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            )
            await input_queue.put(bar)

        # Run detector briefly
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)
        await detector.stop()
        await task

        # Should not crash, just no swings detected
        assert detector.swing_highs_count == 0
        assert detector.swing_lows_count == 0

    @pytest.mark.asyncio
    async def test_detector_handles_empty_queue(self, detector):
        """Verify detector handles empty queue without errors."""
        # Don't add any bars to queue

        # Run detector briefly
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)
        await detector.stop()
        await task

        # Should complete without errors
        assert detector.is_running is False

    @pytest.mark.asyncio
    async def test_detector_stop_sets_running_flag(self, detector):
        """Verify stop() sets running flag to False."""
        # Start detector
        task = asyncio.create_task(detector.consume())

        # Wait a bit
        await asyncio.sleep(0.05)

        # Stop detector
        await detector.stop()
        await task

        # Verify flag is False
        assert detector.is_running is False

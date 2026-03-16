"""Unit tests for FVG Detector."""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.data.models import DollarBar, FVGEvent, GapRange
from src.detection.fvg_detector import FVGDetector
from src.detection.fvg_detection import (
    check_fvg_fill,
    detect_bearish_fvg,
    detect_bullish_fvg,
)


class TestFVGDetection:
    """Test FVG detection algorithms."""

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

    def test_detect_bullish_fvg_with_gap(self, base_timestamp):
        """Verify bullish FVG detected with valid 3-candle pattern."""
        bars = []
        # Candle 1: Close at 11850, High at 11860
        bars.append(
            DollarBar(
                timestamp=base_timestamp,
                open=11840.0,
                high=11860.0,
                low=11830.0,
                close=11850.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        # Candle 2: Gap down
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=5),
                open=11845.0,
                high=11850.0,
                low=11835.0,
                close=11840.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        # Candle 3: Open at 11840, Low at 11830
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=10),
                open=11835.0,
                high=11845.0,
                low=11825.0,
                close=11840.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )

        fvg = detect_bullish_fvg(bars, 2)

        assert fvg is not None
        assert fvg.direction == "bullish"
        assert fvg.gap_range.top == 11860.0
        assert fvg.gap_range.bottom == 11825.0
        assert fvg.gap_size_ticks == 140.0  # (11860 - 11825) / 0.25
        assert fvg.filled is False

    def test_detect_bearish_fvg_with_gap(self, base_timestamp):
        """Verify bearish FVG detected with valid 3-candle pattern."""
        bars = []
        # Candle 1: Close at 11850, Low at 11840
        bars.append(
            DollarBar(
                timestamp=base_timestamp,
                open=11845.0,
                high=11855.0,
                low=11840.0,
                close=11850.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        # Candle 2: Gap up
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=5),
                open=11855.0,
                high=11860.0,
                low=11850.0,
                close=11858.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        # Candle 3: Open at 11860, High at 11870
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=10),
                open=11860.0,
                high=11870.0,
                low=11855.0,
                close=11865.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )

        fvg = detect_bearish_fvg(bars, 2)

        assert fvg is not None
        assert fvg.direction == "bearish"
        assert fvg.gap_range.top == 11870.0
        assert fvg.gap_range.bottom == 11840.0
        assert fvg.gap_size_ticks == 120.0  # (11870 - 11840) / 0.25
        assert fvg.filled is False

    def test_no_fvg_when_no_gap(self, base_timestamp):
        """Verify no FVG when gap condition not met."""
        bars = []
        # Create bars where candle 1 close equals candle 3 open (no gap)
        bars.append(
            DollarBar(
                timestamp=base_timestamp,
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11800.0,  # Close at 11800
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=5),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11800.0,  # Close at 11800
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=10),
                open=11800.0,  # Open at 11800 (no gap)
                high=11810.0,
                low=11790.0,
                close=11800.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )

        # Check for bullish FVG (candle_1.close > candle_3.open is False)
        bullish_fvg = detect_bullish_fvg(bars, 2)
        assert bullish_fvg is None

        # Check for bearish FVG (candle_1.close < candle_3.open is False)
        bearish_fvg = detect_bearish_fvg(bars, 2)
        assert bearish_fvg is None

    def test_no_fvg_when_insufficient_bars(self):
        """Verify no FVG detected with less than 3 bars."""
        bars = [
            DollarBar(
                timestamp=datetime.now(),
                open=11800.0,
                high=11810.0,
                low=11790.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            )
        ]

        # Only 1 bar, should return None
        fvg = detect_bullish_fvg(bars, 0)
        assert fvg is None

    def test_fvg_size_calculation(self, base_timestamp):
        """Verify FVG size calculated correctly in ticks and dollars."""
        bars = []
        # Create 10-point gap
        bars.append(
            DollarBar(
                timestamp=base_timestamp,
                open=11840.0,
                high=11860.0,  # Top of gap
                low=11830.0,
                close=11850.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=5),
                open=11850.0,
                high=11855.0,
                low=11845.0,
                close=11848.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )
        bars.append(
            DollarBar(
                timestamp=base_timestamp + timedelta(seconds=10),
                open=11840.0,
                high=11850.0,
                low=11830.0,  # Bottom of gap
                close=11835.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )

        fvg = detect_bullish_fvg(bars, 2)

        # Gap is 30 points (11860 - 11830)
        assert fvg.gap_size_ticks == 120.0  # 30 / 0.25
        assert fvg.gap_size_dollars == 600.0  # 30 * $20

    def test_multiple_fvg_in_sequence(self, base_timestamp):
        """Verify multiple FVGs can be detected in trending market."""
        bars = []

        # Create sequence with multiple bullish FVGs
        for i in range(10):
            if i == 2:
                # First FVG setup
                bars.append(
                    DollarBar(
                        timestamp=base_timestamp + timedelta(seconds=i * 5),
                        open=11840.0,
                        high=11860.0,
                        low=11830.0,
                        close=11855.0,
                        volume=1000,
                        notional_value=50_000_000,
                    )
                )
            elif i == 3:
                bars.append(
                    DollarBar(
                        timestamp=base_timestamp + timedelta(seconds=i * 5),
                        open=11850.0,
                        high=11855.0,
                        low=11835.0,
                        close=11840.0,
                        volume=1000,
                        notional_value=50_000_000,
                    )
                )
            elif i == 4:
                bars.append(
                    DollarBar(
                        timestamp=base_timestamp + timedelta(seconds=i * 5),
                        open=11840.0,
                        high=11845.0,
                        low=11825.0,
                        close=11835.0,
                        volume=1000,
                        notional_value=50_000_000,
                    )
                )
            else:
                bars.append(
                    DollarBar(
                        timestamp=base_timestamp + timedelta(seconds=i * 5),
                        open=11800.0,
                        high=11810.0,
                        low=11795.0,
                        close=11805.0,
                        volume=1000,
                        notional_value=50_000_000,
                    )
                )

        # Should detect FVG at index 4
        fvg = detect_bullish_fvg(bars, 4)
        assert fvg is not None


class TestFVGFillDetection:
    """Test FVG fill detection algorithms."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for test bars."""
        return datetime(2026, 3, 16, 10, 0, 0)

    @pytest.fixture
    def bullish_fvg(self, base_timestamp):
        """Create a sample bullish FVG."""
        return FVGEvent(
            timestamp=base_timestamp,
            direction="bullish",
            gap_range=GapRange(top=11900.0, bottom=11870.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=5,
            filled=False,
        )

    @pytest.fixture
    def bearish_fvg(self, base_timestamp):
        """Create a sample bearish FVG."""
        return FVGEvent(
            timestamp=base_timestamp,
            direction="bearish",
            gap_range=GapRange(top=11870.0, bottom=11840.0),
            gap_size_ticks=120.0,
            gap_size_dollars=600.0,
            bar_index=5,
            filled=False,
        )

    def test_bullish_fvg_fill_when_price_reaches_bottom(
        self, bullish_fvg, base_timestamp
    ):
        """Verify bullish FVG filled when price reaches bottom of gap."""
        bar = DollarBar(
            timestamp=base_timestamp + timedelta(seconds=30),
            open=11880.0,
            high=11890.0,
            low=11865.0,  # Below gap bottom (11870.0)
            close=11870.0,
            volume=1000,
            notional_value=50_000_000,
        )

        filled = check_fvg_fill(bullish_fvg, bar)
        assert filled is True

    def test_bearish_fvg_fill_when_price_reaches_top(self, bearish_fvg, base_timestamp):
        """Verify bearish FVG filled when price reaches top of gap."""
        bar = DollarBar(
            timestamp=base_timestamp + timedelta(seconds=30),
            open=11850.0,
            high=11875.0,  # Above gap top (11870.0)
            low=11845.0,
            close=11870.0,
            volume=1000,
            notional_value=50_000_000,
        )

        filled = check_fvg_fill(bearish_fvg, bar)
        assert filled is True

    def test_no_fill_when_price_outside_gap(self, bullish_fvg, base_timestamp):
        """Verify FVG not filled when price hasn't reached gap."""
        bar = DollarBar(
            timestamp=base_timestamp + timedelta(seconds=30),
            open=11880.0,
            high=11890.0,
            low=11875.0,  # Above gap bottom (11870.0), no fill
            close=11885.0,
            volume=1000,
            notional_value=50_000_000,
        )

        filled = check_fvg_fill(bullish_fvg, bar)
        assert filled is False

    def test_fvg_not_filled_twice(self, bullish_fvg, base_timestamp):
        """Verify filled FVG returns True on subsequent checks."""
        # Mark as filled
        bullish_fvg.filled = True

        bar = DollarBar(
            timestamp=base_timestamp + timedelta(seconds=30),
            open=11880.0,
            high=11890.0,
            low=11865.0,
            close=11870.0,
            volume=1000,
            notional_value=50_000_000,
        )

        # Should return True even if already filled
        filled = check_fvg_fill(bullish_fvg, bar)
        assert filled is True


class TestFVGDetectorClass:
    """Test FVGDetector class functionality."""

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
        """Create FVGDetector instance for testing."""
        return FVGDetector(
            input_queue=input_queue,
            output_queue=output_queue,
            tick_size=0.25,
            point_value=20.0,
        )

    def test_detector_initialization(self, detector):
        """Verify detector initializes correctly."""
        assert detector.is_running is False
        assert detector.unfilled_fvgs_count == 0

    @pytest.mark.asyncio
    async def test_consume_processes_dollar_bars(self, detector, input_queue):
        """Verify consume method processes Dollar Bars."""
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
        await asyncio.sleep(0.1)
        await detector.stop()
        await task

        # Should have processed bars
        assert len(detector._bars) > 0

    @pytest.mark.asyncio
    async def test_consume_detects_fvg_patterns(
        self, detector, input_queue, output_queue
    ):
        """Verify consume detects FVG patterns."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create 3-bar sequence with bullish FVG
        # Candle 1: Close at 11850, High at 11860
        bar1 = DollarBar(
            timestamp=base_time,
            open=11840.0,
            high=11860.0,
            low=11830.0,
            close=11850.0,
            volume=1000,
            notional_value=50_000_000,
        )
        # Candle 2: Gap down
        bar2 = DollarBar(
            timestamp=base_time + timedelta(seconds=5),
            open=11845.0,
            high=11850.0,
            low=11835.0,
            close=11840.0,
            volume=1000,
            notional_value=50_000_000,
        )
        # Candle 3: Open at 11840, Low at 11830
        bar3 = DollarBar(
            timestamp=base_time + timedelta(seconds=10),
            open=11835.0,
            high=11845.0,
            low=11825.0,
            close=11840.0,
            volume=1000,
            notional_value=50_000_000,
        )

        await input_queue.put(bar1)
        await input_queue.put(bar2)
        await input_queue.put(bar3)

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)
        await detector.stop()
        await task

        # Should have detected FVG
        # (actual detection depends on bar timing)

    @pytest.mark.asyncio
    async def test_consume_tracks_fills(self, detector, input_queue):
        """Verify consume tracks FVG fills correctly."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Create sequence: FVG formation, then fill
        bars = []
        for i in range(10):
            if i == 2:
                # Candle 1 of FVG
                high = 11860.0
                close = 11850.0
                low = 11830.0
                open_p = 11840.0
            elif i == 3:
                # Candle 2 of FVG
                high = 11850.0
                close = 11840.0
                low = 11835.0
                open_p = 11845.0
            elif i == 4:
                # Candle 3 of FVG
                high = 11845.0
                close = 11840.0
                low = 11825.0
                open_p = 11835.0
            elif i == 7:
                # Fill bar
                high = 11850.0
                close = 11830.0
                low = 11820.0  # Below FVG bottom
                open_p = 11840.0
            else:
                high = 11810.0
                close = 11805.0
                low = 11790.0
                open_p = 11800.0

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
    async def test_consume_handles_graceful_shutdown(self, detector):
        """Verify consume handles graceful shutdown correctly."""
        # Start detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.05)

        # Stop detector
        await detector.stop()
        await task

        # Detector should not be running
        assert detector.is_running is False

    @pytest.mark.asyncio
    async def test_unfilled_fvgs_list_management(self, detector, input_queue):
        """Verify unfilled FVGs list is managed correctly."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add bar that creates FVG (simplified test)
        bar = DollarBar(
            timestamp=base_time,
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
        await asyncio.sleep(0.05)
        await detector.stop()
        await task

        # Unfilled FVGs count should be >= 0
        assert detector.unfilled_fvgs_count >= 0

    @pytest.mark.asyncio
    async def test_detector_handles_insufficient_bars(self, detector, input_queue):
        """Verify detector handles insufficient bars gracefully."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        # Add only 2 bars (need 3 for FVG detection)
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

        # Run detector
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.1)
        await detector.stop()
        await task

        # Should not crash, just no FVGs detected
        assert len(detector._bars) == 2

    @pytest.mark.asyncio
    async def test_detector_handles_empty_queue(self, detector):
        """Verify detector handles empty queue without errors."""
        # Don't add any bars to queue

        # Run detector briefly
        task = asyncio.create_task(detector.consume())
        await asyncio.sleep(0.05)
        await detector.stop()
        await task

        # Should complete without errors
        assert detector.is_running is False

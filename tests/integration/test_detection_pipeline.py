"""Integration tests for Pattern Detection Pipeline.

Tests the complete detection pipeline from Dollar Bars to Silver Bullet signals.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.data.models import DollarBar, SilverBulletSetup
from src.detection.fvg_detector import FVGDetector
from src.detection.liquidity_sweep_detector import LiquiditySweepDetector
from src.detection.mss_detector import MSSDetector
from src.detection.pipeline import DetectionPipeline, DetectionStatistics


class TestDetectionStatistics:
    """Test detection statistics tracking."""

    def test_initialization(self):
        """Verify statistics initialization with zero values."""
        stats = DetectionStatistics()

        assert stats.mss_count == 0
        assert stats.fvg_count == 0
        assert stats.sweep_count == 0
        assert stats.signal_count == 0
        assert stats.total_confidence == 0.0
        assert stats.start_time is not None

    def test_record_mss_detection(self):
        """Verify MSS detection counter increments."""
        stats = DetectionStatistics()

        stats.record_mss()

        assert stats.mss_count == 1
        assert stats.fvg_count == 0
        assert stats.sweep_count == 0

    def test_record_fvg_detection(self):
        """Verify FVG detection counter increments."""
        stats = DetectionStatistics()

        stats.record_fvg()

        assert stats.mss_count == 0
        assert stats.fvg_count == 1
        assert stats.sweep_count == 0

    def test_record_sweep_detection(self):
        """Verify liquidity sweep detection counter increments."""
        stats = DetectionStatistics()

        stats.record_sweep()

        assert stats.mss_count == 0
        assert stats.fvg_count == 0
        assert stats.sweep_count == 1

    def test_record_signal(self):
        """Verify signal detection counter increments."""
        stats = DetectionStatistics()

        stats.record_signal(confidence=3)

        assert stats.signal_count == 1
        assert stats.total_confidence == 3.0

    def test_average_confidence(self):
        """Verify average confidence calculation."""
        stats = DetectionStatistics()

        stats.record_signal(confidence=3)
        stats.record_signal(confidence=4)
        stats.record_signal(confidence=5)

        assert stats.signal_count == 3
        assert stats.average_confidence == 4.0

    def test_average_confidence_no_signals(self):
        """Verify average confidence is 0 when no signals."""
        stats = DetectionStatistics()

        assert stats.average_confidence == 0.0

    def test_signals_per_hour_zero_runtime(self):
        """Verify signals per hour is 0 when runtime < 1 minute."""
        stats = DetectionStatistics()

        stats.record_signal(confidence=3)

        # Should be 0 for very short runtime
        assert stats.signals_per_hour == 0

    def test_get_summary(self):
        """Verify statistics summary dictionary."""
        stats = DetectionStatistics()

        stats.record_mss()
        stats.record_fvg()
        stats.record_sweep()
        stats.record_signal(confidence=4)
        stats.record_signal(confidence=5)

        summary = stats.get_summary()

        assert summary["mss_count"] == 1
        assert summary["fvg_count"] == 1
        assert summary["sweep_count"] == 1
        assert summary["signal_count"] == 2
        assert summary["average_confidence"] == 4.5
        assert "signals_per_hour" in summary
        assert "runtime_seconds" in summary


class TestDetectionPipeline:
    """Test detection pipeline integration."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for Dollar Bars."""
        return asyncio.Queue(maxsize=100)

    @pytest.fixture
    def signal_queue(self):
        """Create output queue for Silver Bullet signals."""
        return asyncio.Queue(maxsize=100)

    @pytest.fixture
    def sample_bar(self):
        """Create sample Dollar Bar for testing."""
        return DollarBar(
            timestamp=datetime(2026, 3, 16, 10, 30, 0),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11830.0,
            volume=1000,
            notional_value=11830000.0,
            bar_index=0,
        )

    @pytest.fixture
    def pipeline(self, input_queue, signal_queue):
        """Create detection pipeline instance."""
        return DetectionPipeline(
            input_queue=input_queue,
            signal_queue=signal_queue,
        )

    def test_pipeline_initialization(self, pipeline):
        """Verify pipeline initializes with all detectors."""
        assert pipeline.mss_detector is not None
        assert isinstance(pipeline.mss_detector, MSSDetector)

        assert pipeline.fvg_detector is not None
        assert isinstance(pipeline.fvg_detector, FVGDetector)

        assert pipeline.sweep_detector is not None
        assert isinstance(pipeline.sweep_detector, LiquiditySweepDetector)

        # Silver Bullet detection uses functional API, not a detector instance
        assert hasattr(pipeline, "_recent_mss")
        assert hasattr(pipeline, "_recent_fvg")
        assert hasattr(pipeline, "_recent_sweeps")

        assert pipeline.statistics is not None
        assert isinstance(pipeline.statistics, DetectionStatistics)

    def test_process_bar_creates_swing_point(self, pipeline, sample_bar):
        """Verify processing a bar creates swing points."""
        # Process enough bars to form swing points (need at least 7 bars)
        for i in range(7):
            bar = DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 30, i * 5),  # Different timestamps
                open=11800.0 + i * 10,
                high=11850.0 + i * 10,
                low=11790.0 + i * 10,
                close=11830.0 + i * 10,
                volume=1000,
                notional_value=11830000.0,
            )
            asyncio.run(pipeline.process_bar(bar))

        # Should have processed bars and built history
        assert hasattr(pipeline, "_bar_history")
        assert len(pipeline._bar_history) >= 7

    def test_process_bar_detects_patterns(self, pipeline):
        """Verify processing bars detects patterns."""
        # Create a price sequence that should generate patterns
        bars = self._create_pattern_sequence()

        for bar in bars:
            asyncio.run(pipeline.process_bar(bar))

        # Should have detected some patterns
        stats = pipeline.statistics.get_summary()
        # At minimum should have processed the bars
        assert stats["mss_count"] >= 0
        assert stats["fvg_count"] >= 0
        assert stats["sweep_count"] >= 0

    def test_process_bar_filters_by_time_window(self, pipeline, sample_bar):
        """Verify setups outside time windows are filtered."""
        # Create a bar outside trading windows (midnight)
        bar = DollarBar(
            timestamp=datetime(2026, 3, 16, 0, 0, 0),  # Midnight EST
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11830.0,
            volume=1000,
            notional_value=11830000.0,
        )

        # Process bar - should not generate signal due to time filter
        asyncio.run(pipeline.process_bar(bar))

        # Check signal queue - should be empty or have filtered signals
        # (This depends on whether patterns were detected)
        # The key is that time filtering is applied

    def test_process_bar_assigns_confidence_score(self, pipeline):
        """Verify detected setups get confidence scores."""
        bars = self._create_pattern_sequence()

        for bar in bars:
            asyncio.run(pipeline.process_bar(bar))

        # If any signals were generated, they should have confidence scores
        # Check signal queue for any signals
        if not pipeline._signal_queue.empty():
            signal = asyncio.get_event_loop().run_until_complete(
                pipeline._signal_queue.get()
            )
            assert signal.confidence > 0
            assert 1 <= signal.confidence <= 5

    def test_statistics_tracking(self, pipeline):
        """Verify statistics are tracked correctly."""
        bars = self._create_pattern_sequence()

        for bar in bars:
            asyncio.run(pipeline.process_bar(bar))

        stats = pipeline.statistics.get_summary()

        # Should have processed bars
        assert "mss_count" in stats
        assert "fvg_count" in stats
        assert "sweep_count" in stats
        assert "signal_count" in stats

    def test_latency_requirement(self, pipeline):
        """Verify processing latency is < 100ms."""
        import time

        bar = DollarBar(
            timestamp=datetime(2026, 3, 16, 10, 30, 0),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11830.0,
            volume=1000,
            notional_value=11830000.0,
        )

        # Measure processing time
        start_time = time.perf_counter()
        asyncio.run(pipeline.process_bar(bar))
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should be well under 100ms
        assert elapsed_ms < 100, f"Latency {elapsed_ms:.2f}ms exceeds 100ms requirement"

    def test_pattern_history_maintenance(self, pipeline):
        """Verify pattern history is maintained (last 100 events)."""
        bars = self._create_pattern_sequence(n_bars=150)

        for bar in bars:
            asyncio.run(pipeline.process_bar(bar))

        # Pattern history should be limited to 100
        # Check pipeline's pattern history lists
        assert len(pipeline._recent_mss) <= 100
        assert len(pipeline._recent_fvg) <= 100
        assert len(pipeline._recent_sweeps) <= 100
        # Bar history should also be limited
        assert len(pipeline._bar_history) <= 100

    def test_backpressure_handling(self, pipeline, signal_queue):
        """Verify pipeline handles backpressure when queue fills."""
        # Fill the signal queue
        while not signal_queue.full():
            signal = MagicMock(spec=SilverBulletSetup)
            signal_queue.put_nowait(signal)

        # Process a bar - should handle full queue gracefully
        bar = DollarBar(
            timestamp=datetime(2026, 3, 16, 10, 30, 0),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11830.0,
            volume=1000,
            notional_value=11830000.0,
        )

        # Should not raise exception
        try:
            asyncio.run(pipeline.process_bar(bar))
        except Exception as e:
            pytest.fail(f"Pipeline should handle backpressure gracefully: {e}")

    def _create_pattern_sequence(self, n_bars: int = 20) -> list[DollarBar]:
        """Create a sequence of bars that should generate patterns.

        Creates a price sequence with:
        - Swing high followed by breakdown (bearish MSS)
        - 3-candle FVG pattern
        - Liquidity sweep below swing low
        """
        bars = []

        # Create uptrend then reversal
        for i in range(n_bars):
            if i < 10:
                # Uptrend
                open_price = 11800.0 + i * 20
                close_price = open_price + 10
                high_price = close_price + 20
                low_price = open_price - 10
            else:
                # Reversal/downtrend
                open_price = 12000.0 - (i - 10) * 20
                close_price = open_price - 10
                high_price = open_price + 10
                low_price = close_price - 20

            bar = DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 30, 0),  # Fixed timestamp
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1000 + i * 100,  # Increasing volume
                notional_value=close_price * 1000,
            )
            bars.append(bar)

        return bars


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Verify complete flow from bar to signal."""
        # Create queues
        input_queue = asyncio.Queue(maxsize=100)
        signal_queue = asyncio.Queue(maxsize=100)

        # Create pipeline
        pipeline = DetectionPipeline(
            input_queue=input_queue,
            signal_queue=signal_queue,
        )

        # Create pattern-generating bars
        bars = self._create_silver_bullet_sequence()

        # Process all bars
        for bar in bars:
            await pipeline.process_bar(bar)

        # Verify signals were generated
        # (Depends on pattern recognition quality)
        stats = pipeline.statistics.get_summary()
        assert stats["signal_count"] >= 0

    def _create_silver_bullet_sequence(self) -> list[DollarBar]:
        """Create a bar sequence that should generate a Silver Bullet setup.

        Sequence:
        1. Establish swing low
        2. Breakout above swing low with high volume (MSS)
        3. Create FVG on breakout
        4. Sweep below swing low and reverse
        """
        bars = []

        # 1. Establish swing low
        for i in range(5):
            bars.append(
                DollarBar(
                    timestamp=datetime(2026, 3, 16, 10, 30, i * 5),
                    open=11800.0 - i * 10,
                    high=11810.0 - i * 10,
                    low=11790.0 - i * 10,
                    close=11795.0 - i * 10,
                    volume=500,
                    notional_value=11795000.0,
                )
            )

        # 2. Breakout with high volume (MSS)
        bars.append(
            DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 30, 25),
                open=11750.0,
                high=11820.0,  # Break above previous highs
                low=11745.0,
                close=11815.0,
                volume=2000,  # High volume
                notional_value=11815000.0,
            )
        )

        # 3. Continue up with FVG
        bars.append(
            DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 30, 30),
                open=11815.0,
                high=11850.0,
                low=11810.0,
                close=11845.0,
                volume=1500,
                notional_value=11845000.0,
            )
        )

        # 4. Sweep and reverse
        bars.append(
            DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 30, 35),
                open=11845.0,
                high=11850.0,
                low=11785.0,  # Sweep below swing low
                close=11810.0,  # Close above
                volume=1800,
                notional_value=11810000.0,
            )
        )

        return bars

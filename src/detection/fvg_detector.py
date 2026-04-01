"""Fair Value Gap (FVG) Detector.

This module implements the FVGDetector class which consumes Dollar Bars
from the gap_filled_queue, identifies 3-candle FVG patterns, tracks
unfilled FVGs, checks for fills on each bar, and publishes FVG events
to the fvg_event_queue.
"""

import asyncio
import logging
import time

from src.data.models import DollarBar, FVGEvent
from src.detection.fvg_detection import (
    check_fvg_fill,
    detect_bearish_fvg,
    detect_bullish_fvg,
)

logger = logging.getLogger(__name__)


class FVGDetector:
    """Detects Fair Value Gaps (FVG) from Dollar Bars.

    Consumes Dollar Bars from gap_filled_queue, identifies 3-candle
    FVG patterns, tracks unfilled FVGs, checks for fills on each bar,
    and publishes FVG events to fvg_event_queue.

    Performance Requirement:
        - Detection latency < 100ms per Dollar Bar

    Attributes:
        _input_queue: Queue consuming Dollar Bars from gap detector
        _output_queue: Queue publishing FVG events for downstream consumption
        _tick_size: Tick size for futures contract (0.25 for MNQ)
        _point_value: Dollar value per point ($20 for MNQ)
        _bars: History of recent Dollar Bars (max 100)
        _unfilled_fvgs: List of unfilled FVG events being tracked
        _running: Flag indicating if detector is running
    """

    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size
    DEFAULT_POINT_VALUE = 20.0  # $20 per point for MNQ
    MAX_BAR_HISTORY = 100  # Keep last 100 bars for FVG detection

    def __init__(
        self,
        input_queue: asyncio.Queue[DollarBar],
        output_queue: asyncio.Queue[FVGEvent],
        tick_size: float = DEFAULT_TICK_SIZE,
        point_value: float = DEFAULT_POINT_VALUE,
    ) -> None:
        """Initialize FVG detector.

        Args:
            input_queue: Queue consuming Dollar Bars from gap detector
            output_queue: Queue publishing FVG events for downstream consumption
            tick_size: Tick size for futures contract (default: 0.25 for MNQ)
            point_value: Dollar value per point (default: $20 for MNQ)
        """
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._tick_size = tick_size
        self._point_value = point_value

        # State
        self._bars: list[DollarBar] = []
        self._unfilled_fvgs: list[FVGEvent] = []
        self._running = False

    async def consume(self) -> None:
        """Consume Dollar Bars from input queue and detect FVG.

        This is the main async task that runs continuously.
        For each Dollar Bar:
            1. Add bar to history
            2. Check for new FVG patterns (3-candle setup)
            3. Check all unfilled FVGs for fills
            4. Publish new FVG events to output queue
            5. Update and republish filled FVG events
        """
        self._running = True
        logger.info("FVG detector started")

        while self._running:
            try:
                # Consume Dollar Bar with timeout (allows graceful shutdown)
                bar = await asyncio.wait_for(self._input_queue.get(), timeout=1.0)

                # Start timing for performance measurement
                start_time = time.perf_counter()

                # Process bar
                await self._process_bar(bar)

                # Measure detection latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                if latency_ms >= 100:
                    logger.warning(
                        f"FVG detection latency exceeded 100ms: {latency_ms:.2f}ms"
                    )

            except asyncio.TimeoutError:
                # No data available, continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"FVG detection error: {e}", exc_info=True)

        self._running = False
        logger.info("FVG detector stopped")

    async def _process_bar(self, bar: DollarBar) -> None:
        """Process a single Dollar Bar for FVG detection.

        Args:
            bar: Dollar Bar to process
        """
        # Add to bar history (keep max 100 bars)
        self._bars.append(bar)
        if len(self._bars) > self.MAX_BAR_HISTORY:
            self._bars.pop(0)

        # Get current bar index
        bar_index = len(self._bars) - 1

        # Check for new FVG patterns (need at least 3 bars)
        if bar_index >= 2:
            # Check for bullish FVG
            bullish_fvg = detect_bullish_fvg(self._bars, bar_index)
            if bullish_fvg:
                bullish_fvg.bar_index = bar_index
                self._unfilled_fvgs.append(bullish_fvg)
                await self._output_queue.put(bullish_fvg)
                logger.info(
                    f"Bullish FVG detected: {bullish_fvg.gap_range.top:.2f} - "
                    f"{bullish_fvg.gap_range.bottom:.2f} "
                    f"({bullish_fvg.gap_size_ticks:.1f} ticks)"
                )

            # Check for bearish FVG
            bearish_fvg = detect_bearish_fvg(self._bars, bar_index)
            if bearish_fvg:
                bearish_fvg.bar_index = bar_index
                self._unfilled_fvgs.append(bearish_fvg)
                await self._output_queue.put(bearish_fvg)
                logger.info(
                    f"Bearish FVG detected: {bearish_fvg.gap_range.top:.2f} - "
                    f"{bearish_fvg.gap_range.bottom:.2f} "
                    f"({bearish_fvg.gap_size_ticks:.1f} ticks)"
                )

        # Check unfilled FVGs for fills
        await self._check_fvg_fills(bar, bar_index)

    async def _check_fvg_fills(self, bar: DollarBar, bar_index: int) -> None:
        """Check all unfilled FVGs for fill conditions.

        Args:
            bar: Current Dollar Bar
            bar_index: Current bar index
        """
        filled_fvgs = []

        for fvg in self._unfilled_fvgs:
            if check_fvg_fill(fvg, bar):
                # Mark as filled
                fvg.filled = True
                fvg.fill_time = bar.timestamp
                fvg.fill_bar_index = bar_index
                filled_fvgs.append(fvg)

                logger.info(
                    f"FVG filled: {fvg.direction} gap at "
                    f"{fvg.gap_range.top:.2f} - {fvg.gap_range.bottom:.2f} "
                    f"filled at bar {bar_index}"
                )

        # Remove filled FVGs from unfilled list
        for fvg in filled_fvgs:
            self._unfilled_fvgs.remove(fvg)
            # Publish updated FVG event (now marked as filled)
            await self._output_queue.put(fvg)

    async def stop(self) -> None:
        """Stop FVG detection gracefully."""
        self._running = False
        logger.info("FVG detector stop requested")

    @property
    def is_running(self) -> bool:
        """Check if detector is running.

        Returns:
            True if detector is running
        """
        return self._running

    @property
    def unfilled_fvgs_count(self) -> int:
        """Get number of unfilled FVGs being tracked.

        Returns:
            Count of unfilled FVGs
        """
        return len(self._unfilled_fvgs)


# ============================================================
# TRIPLE CONFLUENCE SCALPER FVG DETECTOR
# ============================================================

from src.detection.models import TripleConfluenceFVGEvent


class SimpleFVGDetector:
    """Simple synchronous FVG detector for Triple Confluence Scalper strategy.

    This is a simplified version of the FVG detector that takes a list of
    bars and returns detected FVG events synchronously. Designed for use
    by the Triple Confluence Scalper strategy.

    Attributes:
        _min_gap_size: Minimum gap size in ticks
        _tick_size: Tick size for futures contract (0.25 for MNQ)
    """

    DEFAULT_MIN_GAP_SIZE = 4  # Minimum gap size in ticks
    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size

    def __init__(
        self,
        min_gap_size: int = DEFAULT_MIN_GAP_SIZE,
        tick_size: float = DEFAULT_TICK_SIZE,
    ) -> None:
        """Initialize Simple FVG detector.

        Args:
            min_gap_size: Minimum gap size in ticks (default: 4)
            tick_size: Tick size for futures contract (default: 0.25 for MNQ)
        """
        self._min_gap_size = min_gap_size
        self._tick_size = tick_size

    def detect_fvg(self, bars: list[DollarBar]) -> list[TripleConfluenceFVGEvent]:
        """Detect FVG patterns from a list of Dollar Bars.

        Args:
            bars: List of Dollar Bars to analyze

        Returns:
            List of TripleConfluenceFVGEvent objects (may be empty)
        """
        if len(bars) < 3:
            return []

        detected_fvgs: list[TripleConfluenceFVGEvent] = []

        # Scan through all 3-candle combinations
        for i in range(len(bars) - 2):
            bar1 = bars[i]  # First candle
            bar2 = bars[i + 1]  # Second candle (middle)
            bar3 = bars[i + 2]  # Third candle

            # Check for bullish FVG: Bar 1 low > Bar 3 high
            if bar1.low > bar3.high:
                gap_size = bar1.low - bar3.high
                gap_size_ticks = gap_size / self._tick_size

                # Filter by minimum gap size
                if gap_size_ticks >= self._min_gap_size:
                    fvg = TripleConfluenceFVGEvent(
                        timestamp=bar3.timestamp,
                        fvg_type="bullish",
                        gap_size_ticks=gap_size_ticks,
                        gap_edge_high=bar1.low,  # Top of gap is bar 1 low
                        gap_edge_low=bar3.high,  # Bottom of gap is bar 3 high
                        third_candle_close=bar3.close,
                    )
                    detected_fvgs.append(fvg)
                    logger.debug(
                        f"Bullish FVG detected at {bar3.timestamp}: "
                        f"gap {bar1.low:.2f} - {bar3.high:.2f} "
                        f"({gap_size_ticks:.1f} ticks)"
                    )

            # Check for bearish FVG: Bar 1 high < Bar 3 low
            if bar1.high < bar3.low:
                gap_size = bar3.low - bar1.high
                gap_size_ticks = gap_size / self._tick_size

                # Filter by minimum gap size
                if gap_size_ticks >= self._min_gap_size:
                    fvg = TripleConfluenceFVGEvent(
                        timestamp=bar3.timestamp,
                        fvg_type="bearish",
                        gap_size_ticks=gap_size_ticks,
                        gap_edge_high=bar3.low,  # Top of gap is bar 3 low
                        gap_edge_low=bar1.high,  # Bottom of gap is bar 1 high
                        third_candle_close=bar3.close,
                    )
                    detected_fvgs.append(fvg)
                    logger.debug(
                        f"Bearish FVG detected at {bar3.timestamp}: "
                        f"gap {bar3.low:.2f} - {bar1.high:.2f} "
                        f"({gap_size_ticks:.1f} ticks)"
                    )

        return detected_fvgs

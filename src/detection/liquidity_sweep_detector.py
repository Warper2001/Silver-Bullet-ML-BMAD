"""Liquidity Sweep Detector.

This module implements the LiquiditySweepDetector class which consumes Dollar Bars
from the fvg_event_queue, identifies liquidity sweeps at swing points, and publishes
sweep events to the liquidity_sweep_queue.
"""

import asyncio
import logging
import time

from src.data.models import DollarBar, LiquiditySweepEvent, SwingPoint
from src.detection.liquidity_sweep_detection import (
    check_bearish_sweep,
    check_bullish_sweep,
)

logger = logging.getLogger(__name__)


class LiquiditySweepDetector:
    """Detects Liquidity Sweeps from Dollar Bars and swing points.

    Consumes Dollar Bars from FVG detector output queue, identifies
    liquidity sweeps at swing points, and publishes sweep events.

    Performance Requirement:
        - Detection latency < 100ms per Dollar Bar

    Attributes:
        _input_queue: Queue consuming Dollar Bars from FVG detector
        _output_queue: Queue publishing sweep events for downstream consumption
        _min_sweep_ticks: Minimum ticks beyond swing point for valid sweep
        _bars: History of recent Dollar Bars (max 100)
        _swing_highs: List of detected swing highs
        _swing_lows: List of detected swing lows
        _running: Flag indicating if detector is running
    """

    DEFAULT_MIN_SWEEP_TICKS = 5.0  # Minimum 5-tick sweep for validation
    MAX_BAR_HISTORY = 100  # Keep last 100 bars for sweep detection

    def __init__(
        self,
        input_queue: asyncio.Queue[DollarBar],
        output_queue: asyncio.Queue[LiquiditySweepEvent],
        min_sweep_ticks: float = DEFAULT_MIN_SWEEP_TICKS,
    ) -> None:
        """Initialize Liquidity Sweep detector.

        Args:
            input_queue: Queue consuming Dollar Bars from FVG detector
            output_queue: Queue publishing sweep events for downstream
            min_sweep_ticks: Minimum ticks beyond swing point (default: 5)
        """
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._min_sweep_ticks = min_sweep_ticks

        # State
        self._bars: list[DollarBar] = []
        self._swing_highs: list[SwingPoint] = []
        self._swing_lows: list[SwingPoint] = []
        self._running = False

    async def consume(self) -> None:
        """Consume Dollar Bars from input queue and detect liquidity sweeps.

        This is the main async task that runs continuously.
        For each Dollar Bar:
            1. Add bar to history
            2. Detect swing points (simplified - actual swing detection in MSS detector)
            3. Check for liquidity sweeps at swing points
            4. Publish sweep events to output queue
        """
        self._running = True
        logger.info("Liquidity Sweep detector started")

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
                        f"Sweep detection latency exceeded 100ms: "
                        f"{latency_ms:.2f}ms"
                    )

            except asyncio.TimeoutError:
                # No data available, continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Liquidity Sweep detection error: {e}", exc_info=True)

        self._running = False
        logger.info("Liquidity Sweep detector stopped")

    async def _process_bar(self, bar: DollarBar) -> None:
        """Process a single Dollar Bar for sweep detection.

        Args:
            bar: Dollar Bar to process
        """
        # Add to bar history (keep max 100 bars)
        self._bars.append(bar)
        if len(self._bars) > self.MAX_BAR_HISTORY:
            self._bars.pop(0)

        # Get current bar index
        bar_index = len(self._bars) - 1

        # TODO: Integrate with MSSDetector for actual swing point detection
        # For now, we'll check sweeps against swing points if they exist
        # In production, swing points would come from MSS detector

        # Check for sweeps at swing lows (bullish sweeps)
        for swing_low in self._swing_lows:
            sweep = check_bullish_sweep(bar, swing_low, self._min_sweep_ticks)
            if sweep:
                sweep.bar_index = bar_index
                await self._output_queue.put(sweep)
                logger.info(
                    f"Bullish liquidity sweep detected at {swing_low.price:.2f}, "
                    f"depth: {sweep.sweep_depth_ticks:.1f} ticks"
                )

        # Check for sweeps at swing highs (bearish sweeps)
        for swing_high in self._swing_highs:
            sweep = check_bearish_sweep(bar, swing_high, self._min_sweep_ticks)
            if sweep:
                sweep.bar_index = bar_index
                await self._output_queue.put(sweep)
                logger.info(
                    f"Bearish liquidity sweep detected at {swing_high.price:.2f}, "
                    f"depth: {sweep.sweep_depth_ticks:.1f} ticks"
                )

    def add_swing_high(self, swing_high: SwingPoint) -> None:
        """Add a swing high for sweep detection.

        Args:
            swing_high: Swing high point to track
        """
        self._swing_highs.append(swing_high)
        # Keep only recent swing points (last 10)
        if len(self._swing_highs) > 10:
            self._swing_highs.pop(0)

    def add_swing_low(self, swing_low: SwingPoint) -> None:
        """Add a swing low for sweep detection.

        Args:
            swing_low: Swing low point to track
        """
        self._swing_lows.append(swing_low)
        # Keep only recent swing points (last 10)
        if len(self._swing_lows) > 10:
            self._swing_lows.pop(0)

    async def stop(self) -> None:
        """Stop liquidity sweep detection gracefully."""
        self._running = False
        logger.info("Liquidity Sweep detector stop requested")

    @property
    def is_running(self) -> bool:
        """Check if detector is running.

        Returns:
            True if detector is running
        """
        return self._running

    @property
    def swing_highs_count(self) -> int:
        """Get number of swing highs being tracked.

        Returns:
            Count of swing highs
        """
        return len(self._swing_highs)

    @property
    def swing_lows_count(self) -> int:
        """Get number of swing lows being tracked.

        Returns:
            Count of swing lows
        """
        return len(self._swing_lows)

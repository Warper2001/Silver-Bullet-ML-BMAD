"""Market Structure Shift (MSS) Detector.

This module implements the MSSDetector class which consumes Dollar Bars
from the gap_filled_queue, identifies swing points, detects breakouts/
breakdowns with volume confirmation, and publishes MSS events to the
mss_event_queue.
"""

import asyncio
import logging
import time

from src.data.models import DollarBar, MSSEvent, SwingPoint
from src.detection.swing_detection import (
    RollingVolumeAverage,
    detect_bearish_mss,
    detect_bullish_mss,
    detect_swing_high,
    detect_swing_low,
)

logger = logging.getLogger(__name__)


class MSSDetector:
    """Detects Market Structure Shifts (MSS) from Dollar Bars.

    Consumes Dollar Bars from gap_filled_queue, identifies swing points,
    detects breakouts/breakdowns with volume confirmation, and publishes
    MSS events to mss_event_queue.

    Performance Requirement:
        - Detection latency < 100ms per Dollar Bar

    Attributes:
        _input_queue: Queue consuming Dollar Bars from gap detector
        _output_queue: Queue publishing MSS events for downstream consumption
        _lookback: Minimum bars on each side for swing point confirmation
        _volume_confirmation_ratio: Minimum volume ratio for breakout confirmation
        _volume_ma_window: Window size for volume moving average
        _bars: History of recent Dollar Bars (max 100)
        _swing_highs: List of confirmed swing highs
        _swing_lows: List of confirmed swing lows
        _volume_ma: Rolling volume average calculator
        _running: Flag indicating if detector is running
    """

    DEFAULT_LOOKBACK = 3  # Bars on each side for swing detection
    DEFAULT_VOLUME_CONFIRMATION_RATIO = 1.5  # 1.5x average volume
    DEFAULT_VOLUME_MA_WINDOW = 20  # 20-bar moving average
    MAX_BAR_HISTORY = 100  # Keep last 100 bars for swing detection

    def __init__(
        self,
        input_queue: asyncio.Queue[DollarBar],
        output_queue: asyncio.Queue[MSSEvent],
        lookback: int = DEFAULT_LOOKBACK,
        volume_confirmation_ratio: float = DEFAULT_VOLUME_CONFIRMATION_RATIO,
        volume_ma_window: int = DEFAULT_VOLUME_MA_WINDOW,
    ) -> None:
        """Initialize MSS detector.

        Args:
            input_queue: Queue consuming Dollar Bars from gap detector
            output_queue: Queue publishing MSS events for downstream consumption
            lookback: Minimum bars on each side for swing point confirmation
            volume_confirmation_ratio: Minimum volume ratio for breakout confirmation
            volume_ma_window: Window size for volume moving average
        """
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._lookback = lookback
        self._volume_confirmation_ratio = volume_confirmation_ratio
        self._volume_ma_window = volume_ma_window

        # State
        self._bars: list[DollarBar] = []
        self._swing_highs: list[SwingPoint] = []
        self._swing_lows: list[SwingPoint] = []
        self._volume_ma = RollingVolumeAverage(window=volume_ma_window)
        self._running = False

    async def consume(self) -> None:
        """Consume Dollar Bars from input queue and detect MSS.

        This is the main async task that runs continuously.
        For each Dollar Bar:
            1. Update rolling volume average
            2. Store bar in history (max 100 bars)
            3. Check for new swing points
            4. Check for bullish/bearish MSS
            5. Publish MSS events to output queue
        """
        self._running = True
        logger.info("MSS detector started")

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
                        f"MSS detection latency exceeded 100ms: {latency_ms:.2f}ms"
                    )

            except asyncio.TimeoutError:
                # No data available, continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MSS detection error: {e}", exc_info=True)

        self._running = False
        logger.info("MSS detector stopped")

    async def _process_bar(self, bar: DollarBar) -> None:
        """Process a single Dollar Bar for MSS detection.

        Args:
            bar: Dollar Bar to process
        """
        # Update volume moving average
        self._volume_ma.update(bar.volume)

        # Add to bar history (keep max 100 bars)
        self._bars.append(bar)
        if len(self._bars) > self.MAX_BAR_HISTORY:
            self._bars.pop(0)

        # Get current bar index
        bar_index = len(self._bars) - 1

        # Check for new swing points (only if we have enough bars)
        if len(self._bars) >= 2 * self._lookback + 1:
            await self._detect_swing_points(bar_index)

        # Check for bullish MSS
        bullish_mss = detect_bullish_mss(
            bar,
            self._swing_highs,
            self._volume_ma.average,
            self._volume_confirmation_ratio,
        )
        if bullish_mss:
            # Set bar_index
            bullish_mss.bar_index = bar_index
            await self._output_queue.put(bullish_mss)
            logger.info(
                f"Bullish MSS detected at {bullish_mss.breakout_price:.2f} "
                f"(volume ratio: {bullish_mss.volume_ratio:.2f})"
            )

        # Check for bearish MSS
        bearish_mss = detect_bearish_mss(
            bar,
            self._swing_lows,
            self._volume_ma.average,
            self._volume_confirmation_ratio,
        )
        if bearish_mss:
            # Set bar_index
            bearish_mss.bar_index = bar_index
            await self._output_queue.put(bearish_mss)
            logger.info(
                f"Bearish MSS detected at {bearish_mss.breakout_price:.2f} "
                f"(volume ratio: {bearish_mss.volume_ratio:.2f})"
            )

    async def _detect_swing_points(self, bar_index: int) -> None:
        """Detect new swing points at the given bar index.

        Args:
            bar_index: Index of current bar in self._bars
        """
        # Check for swing high
        if detect_swing_high(self._bars, bar_index, self._lookback):
            swing_high = SwingPoint(
                timestamp=self._bars[bar_index].timestamp,
                price=self._bars[bar_index].high,
                swing_type="swing_high",
                bar_index=bar_index,
            )
            self._swing_highs.append(swing_high)
            logger.info(f"Swing high detected at {swing_high.price:.2f}")

        # Check for swing low
        if detect_swing_low(self._bars, bar_index, self._lookback):
            swing_low = SwingPoint(
                timestamp=self._bars[bar_index].timestamp,
                price=self._bars[bar_index].low,
                swing_type="swing_low",
                bar_index=bar_index,
            )
            self._swing_lows.append(swing_low)
            logger.info(f"Swing low detected at {swing_low.price:.2f}")

    async def stop(self) -> None:
        """Stop MSS detection gracefully."""
        self._running = False
        logger.info("MSS detector stop requested")

    @property
    def is_running(self) -> bool:
        """Check if detector is running.

        Returns:
            True if detector is running
        """
        return self._running

    @property
    def swing_highs_count(self) -> int:
        """Get number of detected swing highs.

        Returns:
            Count of swing highs
        """
        return len(self._swing_highs)

    @property
    def swing_lows_count(self) -> int:
        """Get number of detected swing lows.

        Returns:
            Count of swing lows
        """
        return len(self._swing_lows)

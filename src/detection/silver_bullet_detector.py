"""Silver Bullet Setup Detector.

This module implements the SilverBulletDetector class which consumes MSS, FVG,
and liquidity sweep events, identifies Silver Bullet setups (pattern confluence),
and publishes setup events for downstream consumption.
"""

import asyncio
import logging
import time

from src.data.models import FVGEvent, LiquiditySweepEvent, MSSEvent, SilverBulletSetup
from src.detection.silver_bullet_detection import detect_silver_bullet_setup

logger = logging.getLogger(__name__)


class SilverBulletDetector:
    """Detects Silver Bullet setups from MSS, FVG, and liquidity sweep events.

    Consumes pattern events from upstream detectors, identifies Silver Bullet
    setups when patterns align in confluence, and publishes setup events.

    Performance Requirement:
        - Detection latency < 100ms from pattern receipt to setup publication

    Attributes:
        _input_queue_mss: Queue consuming MSS events
        _input_queue_fvg: Queue consuming FVG events
        _input_queue_sweep: Queue consuming liquidity sweep events
        _output_queue: Queue publishing Silver Bullet setup events
        _max_bar_distance: Maximum bars between MSS and FVG (default: 10)
        _mss_events: History of recent MSS events (max 50)
        _fvg_events: History of recent FVG events (max 50)
        _sweep_events: History of recent sweep events (max 50)
        _running: Flag indicating if detector is running
    """

    DEFAULT_MAX_BAR_DISTANCE = 10  # Maximum 10 bars between MSS and FVG
    MAX_EVENT_HISTORY = 50  # Keep last 50 events of each type

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue[SilverBulletSetup],
        max_bar_distance: int = DEFAULT_MAX_BAR_DISTANCE,
    ) -> None:
        """Initialize Silver Bullet detector.

        Args:
            input_queue: Queue consuming pattern events (not used in current impl)
            output_queue: Queue publishing Silver Bullet setup events
            max_bar_distance: Maximum bars between MSS and FVG (default: 10)
        """
        self._input_queue = input_queue  # Reserved for future use
        self._output_queue = output_queue
        self._max_bar_distance = max_bar_distance

        # State
        self._mss_events: list[MSSEvent] = []
        self._fvg_events: list[FVGEvent] = []
        self._sweep_events: list[LiquiditySweepEvent] = []
        self._running = False

    async def consume(self) -> None:
        """Consume pattern events and detect Silver Bullet setups.

        This is the main async task that runs continuously.
        For each pattern event:
            1. Add event to history
            2. Check for Silver Bullet setups
            3. Publish setup events to output queue

        Note: Current implementation uses add_*_event() methods instead of
        consuming from queues. This will be updated in future iterations.
        """
        self._running = True
        logger.info("Silver Bullet detector started")

        while self._running:
            try:
                # Wait for events (with timeout to allow graceful shutdown)
                # In future, will consume from actual queues
                await asyncio.sleep(0.1)

                # Detect setups from current events
                await self._detect_setups()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Silver Bullet detection error: {e}", exc_info=True)

        self._running = False
        logger.info("Silver Bullet detector stopped")

    async def _detect_setups(self) -> None:
        """Detect Silver Bullet setups from current event history.

        This method searches for combinations of MSS, FVG, and sweep events
        that form valid Silver Bullet setups.
        """
        # Start timing for performance measurement
        start_time = time.perf_counter()

        # Detect setups
        setups = detect_silver_bullet_setup(
            mss_events=self._mss_events,
            fvg_events=self._fvg_events,
            sweep_events=self._sweep_events,
            max_bar_distance=self._max_bar_distance,
        )

        # Publish setups to output queue
        for setup in setups:
            await self._output_queue.put(setup)

        # Measure detection latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        if latency_ms >= 100:
            logger.warning(
                f"Silver Bullet detection latency exceeded 100ms: {latency_ms:.2f}ms"
            )

    def add_mss_event(self, event: MSSEvent) -> None:
        """Add an MSS event for setup detection.

        Args:
            event: MSS event to add
        """
        self._mss_events.append(event)
        # Keep only recent events (last 50)
        if len(self._mss_events) > self.MAX_EVENT_HISTORY:
            self._mss_events.pop(0)

    def add_fvg_event(self, event: FVGEvent) -> None:
        """Add an FVG event for setup detection.

        Args:
            event: FVG event to add
        """
        self._fvg_events.append(event)
        # Keep only recent events (last 50)
        if len(self._fvg_events) > self.MAX_EVENT_HISTORY:
            self._fvg_events.pop(0)

    def add_sweep_event(self, event: LiquiditySweepEvent) -> None:
        """Add a liquidity sweep event for setup detection.

        Args:
            event: Liquidity sweep event to add
        """
        self._sweep_events.append(event)
        # Keep only recent events (last 50)
        if len(self._sweep_events) > self.MAX_EVENT_HISTORY:
            self._sweep_events.pop(0)

    async def stop(self) -> None:
        """Stop Silver Bullet detection gracefully."""
        self._running = False
        logger.info("Silver Bullet detector stop requested")

    @property
    def is_running(self) -> bool:
        """Check if detector is running.

        Returns:
            True if detector is running
        """
        return self._running

    @property
    def mss_events_count(self) -> int:
        """Get number of MSS events being tracked.

        Returns:
            Count of MSS events
        """
        return len(self._mss_events)

    @property
    def fvg_events_count(self) -> int:
        """Get number of FVG events being tracked.

        Returns:
            Count of FVG events
        """
        return len(self._fvg_events)

    @property
    def sweep_events_count(self) -> int:
        """Get number of sweep events being tracked.

        Returns:
            Count of sweep events
        """
        return len(self._sweep_events)

    @property
    def output_queue(self) -> asyncio.Queue:
        """Get output queue for testing purposes.

        Returns:
            Output queue
        """
        return self._output_queue

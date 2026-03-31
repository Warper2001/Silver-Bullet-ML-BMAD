"""Pattern Detection Pipeline for Silver Bullet signals.

This module integrates all pattern detection components into the asyncio data pipeline.
Consumes Dollar Bars from the gap-filled queue and publishes Silver Bullet signals
to the signal queue after applying time window filtering and confidence scoring.
"""

import asyncio
import logging
from datetime import datetime

from src.data.models import DollarBar, SilverBulletSetup, SwingPoint
from src.detection.fvg_detector import FVGDetector
from src.detection.liquidity_sweep_detector import LiquiditySweepDetector
from src.detection.mss_detector import MSSDetector
from src.detection.time_window_filter import DEFAULT_TRADING_WINDOWS, check_time_window

# Import functional APIs to avoid circular imports
from src.detection.confidence_scorer import score_setup
from src.detection.silver_bullet_detection import detect_silver_bullet_setup

logger = logging.getLogger(__name__)


class DetectionStatistics:
    """Tracks detection statistics for monitoring and logging.

    Attributes:
        mss_count: Number of MSS events detected
        fvg_count: Number of FVG events detected
        sweep_count: Number of liquidity sweep events detected
        signal_count: Number of Silver Bullet signals generated
        total_confidence: Sum of all confidence scores (for calculating average)
        start_time: Timestamp when statistics tracking started
    """

    def __init__(self) -> None:
        """Initialize detection statistics with zero values."""
        self.mss_count: int = 0
        self.fvg_count: int = 0
        self.sweep_count: int = 0
        self.signal_count: int = 0
        self.total_confidence: float = 0.0
        self.start_time: datetime = datetime.now()

    def record_mss(self) -> None:
        """Record an MSS detection event."""
        self.mss_count += 1

    def record_fvg(self) -> None:
        """Record an FVG detection event."""
        self.fvg_count += 1

    def record_sweep(self) -> None:
        """Record a liquidity sweep detection event."""
        self.sweep_count += 1

    def record_signal(self, confidence: int) -> None:
        """Record a Silver Bullet signal with its confidence score.

        Args:
            confidence: Confidence score (1-5) for the signal
        """
        self.signal_count += 1
        self.total_confidence += confidence

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score across all signals.

        Returns:
            Average confidence score, or 0.0 if no signals recorded
        """
        if self.signal_count == 0:
            return 0.0
        return self.total_confidence / self.signal_count

    @property
    def runtime_seconds(self) -> float:
        """Get statistics tracking runtime in seconds.

        Returns:
            Runtime in seconds since start_time
        """
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def signals_per_hour(self) -> float:
        """Calculate signal rate per hour.

        Returns:
            Number of signals per hour, or 0.0 if runtime < 1 minute
        """
        runtime_hours = self.runtime_seconds / 3600
        if runtime_hours < 1 / 60:  # Less than 1 minute
            return 0.0
        return self.signal_count / runtime_hours

    def get_summary(self) -> dict[str, int | float]:
        """Get statistics summary as a dictionary.

        Returns:
            Dictionary containing all statistics metrics
        """
        return {
            "mss_count": self.mss_count,
            "fvg_count": self.fvg_count,
            "sweep_count": self.sweep_count,
            "signal_count": self.signal_count,
            "average_confidence": self.average_confidence,
            "signals_per_hour": self.signals_per_hour,
            "runtime_seconds": self.runtime_seconds,
        }


class DetectionPipeline:
    """Integrates pattern detection into the data pipeline.

    Consumes Dollar Bars from the gap-filled queue and publishes Silver Bullet
    signals to the signal queue after running pattern detection, time window
    filtering, and confidence scoring.

    Pipeline Flow:
    1. Consume Dollar Bar from input queue
    2. Detect MSS, FVG, and liquidity sweeps
    3. Combine patterns into Silver Bullet setups
    4. Apply time window filtering
    5. Assign confidence scores
    6. Publish signals to output queue

    Attributes:
        mss_detector: MSS pattern detector
        fvg_detector: FVG pattern detector
        sweep_detector: Liquidity sweep detector
        silver_bullet_detector: Silver Bullet setup recognizer
        statistics: Detection statistics tracker
    """

    MAX_PATTERN_HISTORY = 100  # Keep last 100 pattern events

    def __init__(
        self,
        input_queue: asyncio.Queue[DollarBar],
        signal_queue: asyncio.Queue[SilverBulletSetup],
        time_windows: list | None = None,
    ) -> None:
        """Initialize the detection pipeline.

        Args:
            input_queue: Queue consuming Dollar Bars from gap detector
            signal_queue: Queue publishing Silver Bullet signals
            time_windows: Trading time windows (uses default if None)
        """
        self._input_queue = input_queue
        self._signal_queue = signal_queue
        self._time_windows = time_windows or DEFAULT_TRADING_WINDOWS

        # Pattern event queues for combining patterns
        self._mss_queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_PATTERN_HISTORY)
        self._fvg_queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_PATTERN_HISTORY)
        self._sweep_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.MAX_PATTERN_HISTORY
        )

        # Initialize pattern detectors (with queues)
        self.mss_detector = MSSDetector(
            input_queue=input_queue,
            output_queue=self._mss_queue,
        )
        self.fvg_detector = FVGDetector(
            input_queue=input_queue,
            output_queue=self._fvg_queue,
        )
        self.sweep_detector = LiquiditySweepDetector(
            input_queue=input_queue,
            output_queue=self._sweep_queue,
        )

        # We'll use the functional API for Silver Bullet detection
        # instead of the class-based detector

        # Pattern history for confluence detection
        self._recent_mss: list = []
        self._recent_fvg: list = []
        self._recent_sweeps: list = []

        # Statistics tracking
        self.statistics = DetectionStatistics()

        # Bar index tracking for detection functions
        self._current_bar_index = 0

        logger.info(
            f"DetectionPipeline initialized with {self.MAX_PATTERN_HISTORY} "
            f"pattern history limit"
        )

    async def process_bar(self, bar: DollarBar) -> None:
        """Process a single Dollar Bar through the detection pipeline.

        Args:
            bar: Dollar Bar to process

        Pipeline Steps:
        1. Detect patterns (MSS, FVG, liquidity sweep)
        2. Combine into Silver Bullet setups
        3. Apply time window filtering
        4. Assign confidence scores
        5. Publish signals to queue

        Latency:
            Total processing time should be < 100ms per bar
        """
        start_time = datetime.now()

        try:
            # Step 1: Detect individual patterns
            mss_events, fvg_events, sweep_events = await self._detect_patterns(bar)

            # Step 2: Combine patterns into Silver Bullet setups
            setups = await self._combine_patterns(mss_events, fvg_events, sweep_events)

            # Step 3-5: Filter, score, and publish signals
            for setup in setups:
                await self._process_setup(setup)

            # Log processing time
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > 100:
                logger.warning(
                    f"Detection pipeline latency exceeded 100ms: {elapsed_ms:.2f}ms"
                )

        except Exception as e:
            logger.error(f"Error processing bar at {bar.timestamp}: {e}", exc_info=True)

    async def _detect_patterns(self, bar: DollarBar) -> tuple[list, list, list]:
        """Run all pattern detectors on a Dollar Bar.

        Args:
            bar: Dollar Bar to analyze

        Returns:
            Tuple of (mss_events, fvg_events, sweep_events)
        """
        from src.detection.swing_detection import (
            RollingVolumeAverage,
            detect_bearish_mss,
            detect_bullish_mss,
            detect_swing_high,
            detect_swing_low,
        )
        from src.detection.fvg_detection import (
            detect_bearish_fvg,
            detect_bullish_fvg,
        )
        from src.detection.liquidity_sweep_detection import (
            detect_bearish_liquidity_sweep,
            detect_bullish_liquidity_sweep,
        )

        mss_events = []
        fvg_events = []
        sweep_events = []

        # Detect MSS (Market Structure Shift)
        # Need to maintain bar history for swing detection
        if not hasattr(self, "_bar_history"):
            self._bar_history = []
        self._bar_history.append(bar)

        # Keep only last 100 bars
        if len(self._bar_history) > self.MAX_PATTERN_HISTORY:
            self._bar_history = self._bar_history[-self.MAX_PATTERN_HISTORY:]

        # Detect swing points
        swing_highs, swing_lows = [], []
        if len(self._bar_history) >= 7:  # Need at least 7 bars for swing detection
            for i, bar_to_check in enumerate(self._bar_history):
                if i >= 3 and i <= len(self._bar_history) - 4:
                    is_swing_high = detect_swing_high(self._bar_history, i)
                    is_swing_low = detect_swing_low(self._bar_history, i)
                    if is_swing_high:
                        # Create SwingPoint object
                        swing_highs.append(SwingPoint(
                            timestamp=bar_to_check.timestamp,
                            price=bar_to_check.high,
                            swing_type="swing_high",
                            bar_index=i
                        ))
                    if is_swing_low:
                        # Create SwingPoint object
                        swing_lows.append(SwingPoint(
                            timestamp=bar_to_check.timestamp,
                            price=bar_to_check.low,
                            swing_type="swing_low",
                            bar_index=i
                        ))

        # Detect MSS
        volume_ma = RollingVolumeAverage(window=20)
        for historical_bar in self._bar_history[:-1]:
            volume_ma.update(historical_bar.volume)

        bullish_mss = detect_bullish_mss(
            self._bar_history[-1],  # Current bar (most recent)
            swing_highs,  # List of SwingPoint objects
            volume_ma.average,  # Rolling average as float
            volume_confirmation_ratio=1.5,
        )
        bearish_mss = detect_bearish_mss(
            self._bar_history[-1],  # Current bar (most recent)
            swing_lows,  # List of SwingPoint objects
            volume_ma.average,  # Rolling average as float
            volume_confirmation_ratio=1.5,
        )

        if bullish_mss:
            mss_events.append(bullish_mss)
            self.statistics.record_mss()
            logger.debug(f"Bullish MSS detected at {bar.timestamp}")

        if bearish_mss:
            mss_events.append(bearish_mss)
            self.statistics.record_mss()
            logger.debug(f"Bearish MSS detected at {bar.timestamp}")

        # Detect FVG
        if len(self._bar_history) >= 3:
            bullish_fvg = detect_bullish_fvg(
                self._bar_history, len(self._bar_history) - 1
            )
            bearish_fvg = detect_bearish_fvg(
                self._bar_history, len(self._bar_history) - 1
            )

            if bullish_fvg:
                fvg_events.append(bullish_fvg)
                self.statistics.record_fvg()
                logger.debug(f"Bullish FVG detected at {bar.timestamp}")

            if bearish_fvg:
                fvg_events.append(bearish_fvg)
                self.statistics.record_fvg()
                logger.debug(f"Bearish FVG detected at {bar.timestamp}")

        # Detect liquidity sweeps (need swing points)
        if swing_highs or swing_lows:
            # Pass the most recent swing point, not the entire list
            bullish_sweep = detect_bullish_liquidity_sweep(
                self._bar_history,
                len(self._bar_history) - 1,
                swing_lows[-1] if swing_lows else None,  # Most recent swing low
                min_sweep_ticks=5,
            )
            bearish_sweep = detect_bearish_liquidity_sweep(
                self._bar_history,
                len(self._bar_history) - 1,
                swing_highs[-1] if swing_highs else None,  # Most recent swing high
                min_sweep_ticks=5,
            )

            if bullish_sweep:
                sweep_events.append(bullish_sweep)
                self.statistics.record_sweep()
                logger.debug(f"Bullish liquidity sweep detected at {bar.timestamp}")

            if bearish_sweep:
                sweep_events.append(bearish_sweep)
                self.statistics.record_sweep()
                logger.debug(f"Bearish liquidity sweep detected at {bar.timestamp}")

        return mss_events, fvg_events, sweep_events

    async def _combine_patterns(
        self,
        mss_events: list,
        fvg_events: list,
        sweep_events: list,
    ) -> list[SilverBulletSetup]:
        """Combine detected patterns into Silver Bullet setups.

        Args:
            mss_events: List of MSS events
            fvg_events: List of FVG events
            sweep_events: List of liquidity sweep events

        Returns:
            List of Silver Bullet setups detected

        Confluence Detection:
            - Checks if MSS, FVG, and sweep occurred within 10 bars
            - Prioritizes 3-pattern confluence (MSS + FVG + sweep)
            - Falls back to 2-pattern confluence (MSS + FVG)
        """
        setups = []

        # Add new events to history
        self._recent_mss.extend(mss_events)
        self._recent_fvg.extend(fvg_events)
        self._recent_sweeps.extend(sweep_events)

        # Keep only last 100 events
        self._recent_mss = self._recent_mss[-self.MAX_PATTERN_HISTORY:]
        self._recent_fvg = self._recent_fvg[-self.MAX_PATTERN_HISTORY:]
        self._recent_sweeps = self._recent_sweeps[-self.MAX_PATTERN_HISTORY:]

        # Detect Silver Bullet setups using confluence
        detected_setups = []

        # Try to combine recent patterns into setups
        for mss in self._recent_mss:
            for fvg in self._recent_fvg:
                # Check if MSS and FVG are in confluence (within 10 bars)
                if abs(mss.bar_index - fvg.bar_index) <= 10:
                    # Check for direction match
                    if mss.direction == fvg.direction:
                        # Create a setup (may also include sweep)
                        setups = detect_silver_bullet_setup(
                            mss_events=[mss],
                            fvg_events=[fvg],
                            sweep_events=self._recent_sweeps,
                        )
                        detected_setups.extend(setups)

        setups.extend(detected_setups)

        return setups

    async def _process_setup(self, setup: SilverBulletSetup) -> None:
        """Process a Silver Bullet setup through filtering and scoring.

        Args:
            setup: Silver Bullet setup to process

        Processing Steps:
        1. Apply time window filtering
        2. Assign confidence score
        3. Publish to signal queue (if passes filters)
        """
        # Step 1: Time window filtering
        filtered_setup = check_time_window(setup, self._time_windows)

        if filtered_setup is None:
            # Setup filtered out by time window
            logger.debug(f"Setup filtered by time window at {setup.timestamp}")
            return

        # Step 2: Assign confidence score
        scored_setup = score_setup(filtered_setup)

        # Step 3: Publish to signal queue
        try:
            self._signal_queue.put_nowait(scored_setup)
            self.statistics.record_signal(scored_setup.confidence)

            logger.info(
                f"Silver Bullet signal published: {scored_setup.direction} "
                f"(confidence: {scored_setup.confidence}, "
                f"confluence: {scored_setup.confluence_count}, "
                f"priority: {scored_setup.priority})"
            )
        except asyncio.QueueFull:
            logger.warning(f"Signal queue full, dropping signal at {setup.timestamp}")

    async def consume(self) -> None:
        """Consume Dollar Bars from input queue and process them.

        This is the main entry point for the detection pipeline.
        Should be run as an async task.
        """
        logger.info("Detection pipeline consumer started")

        while True:
            try:
                # Get next bar from queue (blocks if empty)
                bar = await self._input_queue.get()

                # Process the bar
                await self.process_bar(bar)

                # Mark task as done
                self._input_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Detection pipeline consumer cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in detection pipeline consumer: {e}", exc_info=True
                )

    def log_statistics(self) -> None:
        """Log current detection statistics.

        Should be called periodically (e.g., every 100 signals or hourly).
        """
        stats = self.statistics.get_summary()

        logger.info(
            f"Detection Statistics: "
            f"MSS={stats['mss_count']} "
            f"FVG={stats['fvg_count']} "
            f"Sweeps={stats['sweep_count']} "
            f"Signals={stats['signal_count']} "
            f"Avg Confidence={stats['average_confidence']:.2f} "
            f"Signals/Hour={stats['signals_per_hour']:.1f} "
            f"Runtime={stats['runtime_seconds']:.0f}s"
        )

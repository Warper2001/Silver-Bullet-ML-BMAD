"""Gap detection and forward-fill for Dollar Bar data stream."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import DollarBar

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_STALENESS_THRESHOLD_SECONDS = 30  # 30 seconds
DEFAULT_FORWARD_FILL_LIMIT_SECONDS = 300  # 5 minutes


@dataclass
class GapStatistics:
    """Statistics tracking data gaps."""

    short_gap_count: int = 0  # Gaps < 5 minutes
    short_gap_duration_total: float = 0.0  # Total seconds
    extended_gap_count: int = 0  # Gaps ≥ 5 minutes
    extended_gap_duration_total: float = 0.0  # Total seconds
    extended_gap_log: list[dict] = field(default_factory=list)  # List of extended gaps

    @property
    def total_gap_count(self) -> int:
        """Get total gap count (short + extended)."""
        return self.short_gap_count + self.extended_gap_count

    @property
    def total_gap_duration(self) -> float:
        """Get total gap duration in seconds."""
        return self.short_gap_duration_total + self.extended_gap_duration_total

    @property
    def average_gap_duration(self) -> float:
        """Get average gap duration in seconds."""
        if self.total_gap_count == 0:
            return 0.0
        return self.total_gap_duration / self.total_gap_count


class GapDetector:
    """Detect data gaps and forward-fill Dollar Bars for short interruptions.

    Handles:
    - Staleness detection (30-second timeout)
    - Gap detection and classification (short < 5 min vs. extended ≥ 5 min)
    - Forward-fill for short gaps using last-known close price
    - Gap statistics tracking and logging
    - Async consumption from validated DollarBar queue
    """

    def __init__(
        self,
        validated_queue: asyncio.Queue[DollarBar],
        gap_filled_queue: asyncio.Queue[DollarBar],
        staleness_threshold_seconds: int = DEFAULT_STALENESS_THRESHOLD_SECONDS,
        forward_fill_limit_seconds: int = DEFAULT_FORWARD_FILL_LIMIT_SECONDS,
    ) -> None:
        """Initialize gap detector.

        Args:
            validated_queue: Queue receiving validated DollarBar from Story 1.5
            gap_filled_queue: Queue publishing gap-filled DollarBar for Story 1.7
            staleness_threshold_seconds: Seconds without data before gap detection (default: 30)
            forward_fill_limit_seconds: Maximum gap duration to forward-fill (default: 300 = 5 min)
        """
        self._validated_queue = validated_queue
        self._gap_filled_queue = gap_filled_queue
        self.staleness_threshold_seconds = staleness_threshold_seconds
        self.forward_fill_limit_seconds = forward_fill_limit_seconds

        # Gap statistics
        self._stats = GapStatistics()

        # State tracking
        self._last_seen_timestamp: datetime | None = None
        self._last_valid_bar: DollarBar | None = None
        self._gap_start: datetime | None = None
        self._last_log_time = datetime.now()

    async def consume(self) -> None:
        """Consume DollarBar stream and detect/fill gaps.

        This runs in a background task and:
        1. Receives DollarBar from validated queue
        2. Detects staleness (30-second timeout)
        3. Forward-fills short gaps (< 5 min)
        4. Logs extended gaps (≥ 5 min)
        5. Tracks gap statistics
        """
        logger.info("GapDetector started")

        while True:
            try:
                # Receive DollarBar with timeout for staleness detection
                bar = await asyncio.wait_for(
                    self._validated_queue.get(),
                    timeout=self.staleness_threshold_seconds,
                )

                # Check if we were in a gap state
                if self._gap_start is not None:
                    # Gap has ended - handle it
                    gap_end = bar.timestamp
                    await self._handle_gap(self._gap_start, gap_end)
                    self._gap_start = None

                # Update state
                self._last_seen_timestamp = datetime.now()
                self._last_valid_bar = bar

                # Forward bar to gap-filled queue
                try:
                    self._gap_filled_queue.put_nowait(bar)
                except asyncio.QueueFull:
                    logger.error("Gap-filled queue full, dropping bar")

                # Log statistics periodically
                self._log_gap_statistics_periodically()

            except asyncio.TimeoutError:
                # Staleness detected - gap has started
                if self._gap_start is None and self._last_seen_timestamp is not None:
                    self._gap_start = self._last_seen_timestamp
                    logger.warning(f"Data gap started at {self._gap_start}")

            except Exception as e:
                logger.error(f"Gap detection error: {e}")
                # Continue processing - don't let one error stop the pipeline

    async def _handle_gap(self, gap_start: datetime, gap_end: datetime) -> None:
        """Handle detected gap by forward-filling or logging.

        Args:
            gap_start: Timestamp when gap started
            gap_end: Timestamp when gap ended (data resumed)
        """
        gap_duration = (gap_end - gap_start).total_seconds()

        logger.info(
            f"Data gap detected: start={gap_start}, end={gap_end}, "
            f"duration={gap_duration:.1f}s"
        )

        if gap_duration < self.forward_fill_limit_seconds:
            # Short gap - forward-fill
            logger.info(
                f"Short gap (< {self.forward_fill_limit_seconds}s) - forward-filling"
            )
            self._stats.short_gap_count += 1
            self._stats.short_gap_duration_total += gap_duration

            # Forward-fill bars
            if self._last_valid_bar is not None:
                filled_bars = await self._forward_fill_bars(
                    gap_start,
                    gap_end,
                    self._last_valid_bar,
                )

                # Publish filled bars to gap-filled queue
                for filled_bar in filled_bars:
                    try:
                        self._gap_filled_queue.put_nowait(filled_bar)
                    except asyncio.QueueFull:
                        logger.error(
                            "Gap-filled queue full, dropping forward-filled bar"
                        )

        else:
            # Extended gap - log but don't forward-fill
            logger.warning(
                f"Extended gap (≥ {self.forward_fill_limit_seconds}s) - "
                f"logging only, no forward-fill"
            )
            self._stats.extended_gap_count += 1
            self._stats.extended_gap_duration_total += gap_duration

            # Log to extended gap log
            self._stats.extended_gap_log.append(
                {
                    "start": gap_start,
                    "end": gap_end,
                    "duration_seconds": gap_duration,
                }
            )

    async def _forward_fill_bars(
        self,
        gap_start: datetime,
        gap_end: datetime,
        last_bar: DollarBar,
    ) -> list[DollarBar]:
        """Create forward-filled DollarBars for gap period.

        Forward-fill strategy:
        - Use 5-second intervals (consistent with Story 1.4 low-volume timeout)
        - Use last_bar.close as open, high, low, close (flat price)
        - Set volume=0, notional_value=0 (synthetic bars)
        - Set is_forward_filled=True

        Args:
            gap_start: Timestamp when gap started
            gap_end: Timestamp when gap ended
            last_bar: Last valid DollarBar before gap (used for price)

        Returns:
            List of forward-filled DollarBar objects
        """
        gap_duration = (gap_end - gap_start).total_seconds()
        num_bars = int(gap_duration / 5)  # 5-second intervals

        filled_bars = []
        for i in range(num_bars):
            bar_timestamp = gap_start + timedelta(seconds=(i + 1) * 5)

            bar = DollarBar(
                timestamp=bar_timestamp,
                open=last_bar.close,
                high=last_bar.close,
                low=last_bar.close,
                close=last_bar.close,
                volume=0,
                notional_value=0,
                is_forward_filled=True,
            )
            filled_bars.append(bar)

        logger.info(f"Created {len(filled_bars)} forward-filled bars")
        return filled_bars

    def _log_gap_statistics_periodically(self) -> None:
        """Log gap statistics periodically (every 60 seconds)."""
        now = datetime.now()
        if (now - self._last_log_time).total_seconds() >= 60:
            logger.info(
                f"Gap statistics: "
                f"short_gaps={self._stats.short_gap_count} "
                f"short_duration={self._stats.short_gap_duration_total:.1f}s "
                f"extended_gaps={self._stats.extended_gap_count} "
                f"extended_duration={self._stats.extended_gap_duration_total:.1f}s "
                f"total_gaps={self._stats.total_gap_count} "
                f"total_duration={self._stats.total_gap_duration:.1f}s "
                f"avg_duration={self._stats.average_gap_duration:.1f}s "
                f"gap_filled_queue_depth={self._gap_filled_queue.qsize()}"
            )
            self._last_log_time = now

    def _log_gap_statistics(self) -> None:
        """Log gap statistics (public method for testing)."""
        self._log_gap_statistics_periodically()

    @property
    def short_gap_count(self) -> int:
        """Get short gap count."""
        return self._stats.short_gap_count

    @property
    def short_gap_duration_total(self) -> float:
        """Get total short gap duration in seconds."""
        return self._stats.short_gap_duration_total

    @property
    def extended_gap_count(self) -> int:
        """Get extended gap count."""
        return self._stats.extended_gap_count

    @property
    def extended_gap_duration_total(self) -> float:
        """Get total extended gap duration in seconds."""
        return self._stats.extended_gap_duration_total

    @property
    def total_gap_count(self) -> int:
        """Get total gap count."""
        return self._stats.total_gap_count

    @property
    def total_gap_duration(self) -> float:
        """Get total gap duration in seconds."""
        return self._stats.total_gap_duration

    @property
    def average_gap_duration(self) -> float:
        """Get average gap duration in seconds."""
        return self._stats.average_gap_duration

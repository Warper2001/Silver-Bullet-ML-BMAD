"""Dollar Bar transformation from tick data."""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from .models import DollarBar, MarketData

logger = logging.getLogger(__name__)

# MNQ futures contract specifications
MNQ_MULTIPLIER = 0.5  # $0.5 per point per contract
DOLLAR_THRESHOLD = 50_000_000  # $50M notional value per bar
BAR_TIMEOUT_SECONDS = 5  # Maximum bar duration for low-volume periods


class BarBuilderState(Enum):
    """Bar builder state machine."""

    IDLE = "idle"
    ACCUMULATING = "accumulating"
    COMPLETED = "completed"


class DollarBarTransformer:
    """Transform MarketData stream into Dollar Bars.

    Handles:
    - Accumulation of trades into $50M notional value bars
    - 5-second timeout for low-volume periods
    - OHLCV calculation from tick data
    - Async consumption from MarketData queue
    - Publication to DollarBar queue
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[MarketData],
        output_queue: asyncio.Queue[DollarBar],
    ) -> None:
        """Initialize DollarBar transformer.

        Args:
            input_queue: Queue receiving MarketData from WebSocket
            output_queue: Queue publishing DollarBar for validation
        """
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._state = BarBuilderState.IDLE

        # Bar accumulation state
        self._bar_open: Optional[float] = None
        self._bar_high: Optional[float] = None
        self._bar_low: Optional[float] = None
        self._bar_close: Optional[float] = None
        self._bar_volume: int = 0
        self._bar_notional: float = 0.0
        self._bar_start_time: Optional[datetime] = None

        # Metrics
        self._bars_created = 0
        self._total_trades_processed = 0
        self._transformation_start_time: Optional[datetime] = None

    async def consume(self) -> None:
        """Consume MarketData stream and transform to Dollar Bars.

        This runs in a background task and:
        1. Receives MarketData from input queue
        2. Accumulates trades into current bar
        3. Publishes DollarBar when threshold or timeout reached
        4. Logs transformation metrics
        """
        self._transformation_start_time = datetime.now()
        logger.info("DollarBarTransformer started")

        while True:
            try:
                # Receive market data with timeout
                market_data = await asyncio.wait_for(
                    self._input_queue.get(),
                    timeout=BAR_TIMEOUT_SECONDS,
                )

                self._total_trades_processed += 1
                await self._process_market_data(market_data)

                # Check for timeout on current bar
                await self._check_bar_timeout()

            except asyncio.TimeoutError:
                # No market data received - check if we need to flush bar
                await self._check_bar_timeout()

            except Exception as e:
                logger.error(f"Transformation error: {e}")
                # Continue processing - don't let one error stop the pipeline

    async def _process_market_data(self, market_data: MarketData) -> None:
        """Process incoming market data and update bar accumulation.

        Args:
            market_data: Market data from WebSocket
        """
        # Use last price for Dollar Bars (actual trades, not quotes)
        price = market_data.last
        if price is None:
            logger.warning("Market data missing last price, skipping")
            return

        volume = market_data.volume
        notional = price * volume * MNQ_MULTIPLIER

        # Initialize new bar if idle
        if self._state == BarBuilderState.IDLE:
            self._initialize_bar(price)

        # Update bar OHLCV
        self._update_bar(price, volume, notional)

        # Check if threshold reached
        if self._bar_notional >= DOLLAR_THRESHOLD:
            await self._complete_bar(reason="threshold")

    def _initialize_bar(self, price: float) -> None:
        """Initialize a new Dollar Bar.

        Args:
            price: First trade price in bar
        """
        self._state = BarBuilderState.ACCUMULATING
        self._bar_start_time = datetime.now()

        self._bar_open = price
        self._bar_high = price
        self._bar_low = price
        self._bar_close = price
        self._bar_volume = 0
        self._bar_notional = 0.0

        logger.debug(f"Initialized new bar at {price}")

    def _update_bar(self, price: float, volume: int, notional: float) -> None:
        """Update bar with new trade data.

        Args:
            price: Trade price
            volume: Trade volume
            notional: Notional value (price × volume × multiplier)
        """
        self._bar_high = max(self._bar_high, price)  # type: ignore[assignment]
        self._bar_low = min(self._bar_low, price)  # type: ignore[assignment]
        self._bar_close = price
        self._bar_volume += volume
        self._bar_notional += notional

        logger.debug(
            f"Updated bar: price={price}, notional={self._bar_notional:.2f}/{DOLLAR_THRESHOLD}"
        )

    async def _complete_bar(self, reason: str) -> None:
        """Complete current bar and publish to output queue.

        Args:
            reason: Why bar completed (threshold or timeout)
        """
        if self._state != BarBuilderState.ACCUMULATING:
            return

        # Create DollarBar
        bar = DollarBar(
            timestamp=datetime.now(),
            open=self._bar_open,  # type: ignore[assignment]
            high=self._bar_high,  # type: ignore[assignment]
            low=self._bar_low,  # type: ignore[assignment]
            close=self._bar_close,  # type: ignore[assignment]
            volume=self._bar_volume,
            notional_value=self._bar_notional,
        )

        # Publish to output queue
        try:
            await self._output_queue.put(bar)
            self._bars_created += 1

            logger.info(
                f"DollarBar created (#{self._bars_created}): "
                f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} "
                f"V={bar.volume} notional=${bar.notional_value:.2f} "
                f"reason={reason}"
            )
        except asyncio.QueueFull:
            logger.error("DollarBar output queue full, dropping bar")

        # Reset for next bar
        self._state = BarBuilderState.IDLE

    async def _check_bar_timeout(self) -> None:
        """Check if current bar has exceeded timeout.

        Completes bar if 5 seconds elapsed since bar start.
        """
        if self._state != BarBuilderState.ACCUMULATING:
            return

        if self._bar_start_time is None:
            return

        elapsed = datetime.now() - self._bar_start_time
        if elapsed >= timedelta(seconds=BAR_TIMEOUT_SECONDS):
            await self._complete_bar(reason="timeout")

    @property
    def state(self) -> BarBuilderState:
        """Get current bar builder state."""
        return self._state

    @property
    def bars_created(self) -> int:
        """Get total bars created since start."""
        return self._bars_created

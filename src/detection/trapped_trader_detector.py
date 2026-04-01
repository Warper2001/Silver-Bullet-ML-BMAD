"""Trapped Trader Detector for Wolf Pack 3-Edge strategy.

This module implements trapped trader detection, which identifies when
traders are likely trapped on the wrong side of a move (behavioral edge).
"""

import logging
from collections import deque

import numpy as np

from src.data.models import DollarBar
from src.detection.models import TrappedTraderEvent, WolfPackSweepEvent

logger = logging.getLogger(__name__)


class TrappedTraderDetector:
    """Detects trapped traders for Wolf Pack 3-Edge strategy.

    Identifies patterns where traders were likely trapped on the wrong
    side of a move, indicated by sweeps followed by strong rejections.

    Attributes:
        _volume_window: Window for calculating average volume
        _severity_threshold: Minimum severity ratio for valid trap
        _bars: History of recent Dollar Bars
    """

    DEFAULT_VOLUME_WINDOW = 20  # Bars for average volume calculation
    DEFAULT_SEVERITY_THRESHOLD = 1.5  # Trap volume must be 1.5x average
    MAX_BAR_HISTORY = 100

    def __init__(
        self,
        volume_window: int = DEFAULT_VOLUME_WINDOW,
        severity_threshold: float = DEFAULT_SEVERITY_THRESHOLD,
    ) -> None:
        """Initialize Trapped Trader detector.

        Args:
            volume_window: Window for calculating average volume
            severity_threshold: Minimum severity ratio for valid trap
        """
        self._volume_window = volume_window
        self._severity_threshold = severity_threshold

        self._bars: deque[DollarBar] = deque(maxlen=self.MAX_BAR_HISTORY)

    def process_bars(
        self, bars: list[DollarBar], sweep_event: WolfPackSweepEvent | None = None
    ) -> list[TrappedTraderEvent]:
        """Process new bars and detect trapped traders.

        Args:
            bars: List of new Dollar Bars to process
            sweep_event: Optional sweep event to check for trapped traders

        Returns:
            List of detected trap events
        """
        traps = []

        for bar in bars:
            if len(self._bars) == 0 or bar.timestamp > self._bars[-1].timestamp:
                self._bars.append(bar)

        # If a sweep event is provided, check for trapped traders
        if sweep_event:
            trap = self._check_trap_from_sweep(sweep_event)
            if trap:
                traps.append(trap)

        return traps

    def _check_trap_from_sweep(
        self, sweep: WolfPackSweepEvent
    ) -> TrappedTraderEvent | None:
        """Check for trapped traders based on a sweep event.

        Trapped Longs:
        - Bullish sweep of swing high (buyers trapped at top)
        - Strong rejection with high volume

        Trapped Shorts:
        - Bearish sweep of swing low (sellers trapped at bottom)
        - Strong rejection with high volume

        Args:
            sweep: Sweep event to check for trapped traders

        Returns:
            TrappedTraderEvent if trap detected, None otherwise
        """
        bars_list = list(self._bars)
        if len(bars_list) < self._volume_window:
            return None

        # Find bars around the sweep timestamp
        sweep_bar = None
        for bar in bars_list:
            if bar.timestamp >= sweep.timestamp:
                sweep_bar = bar
                break

        if not sweep_bar:
            return None

        # Calculate average volume over the window
        avg_volume = np.mean([bar.volume for bar in bars_list[-self._volume_window:]])

        # Determine trap type based on sweep direction
        if sweep.sweep_direction == "bearish":
            # Bearish sweep of swing high -> trapped longs
            # (Buyers bought the breakout, now trapped)
            trap_type = "trapped_long"
            entry_price = sweep.swing_level  # Where they likely entered
            rejection_price = sweep.reversal_price  # Where they're trapped
        else:  # bullish sweep
            # Bullish sweep of swing low -> trapped shorts
            # (Sellers sold the breakdown, now trapped)
            trap_type = "trapped_short"
            entry_price = sweep.swing_level  # Where they likely entered
            rejection_price = sweep.reversal_price  # Where they're trapped

        # Calculate severity (volume ratio)
        severity = sweep.reversal_volume / avg_volume if avg_volume > 0 else 1.0

        # Check if severity meets threshold
        if severity < self._severity_threshold:
            return None

        # Create trap event
        event = TrappedTraderEvent(
            timestamp=sweep.timestamp,
            trap_type=trap_type,
            severity=severity,
            entry_price=entry_price,
            rejection_price=rejection_price,
            volume_at_trap=sweep.reversal_volume,
        )

        logger.info(
            f"Trapped {trap_type} detected: severity={severity:.2f}x, "
            f"entry={entry_price:.2f}, rejection={rejection_price:.2f}"
        )

        return event

    def reset(self) -> None:
        """Reset detector state."""
        self._bars.clear()

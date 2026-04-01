"""Level Sweep Detector for Triple Confluence Scalper strategy.

This module detects when price sweeps beyond daily high/low levels
and reverses, indicating a potential liquidity sweep (stop hunt).
"""

import logging
from collections import deque

from src.data.models import DollarBar
from src.detection.models import LevelSweepEvent

logger = logging.getLogger(__name__)


class LevelSweepDetector:
    """Detects level sweeps from Dollar Bars.

    Tracks rolling daily high and low over a lookback period,
    identifies when price trades beyond established levels,
    and confirms sweeps when price reverses through level.

    Attributes:
        _lookback_period: Number of bars to track for daily high/low
        _bars: Deque of recent Dollar Bars for analysis
        _tick_size: Tick size for futures contract (0.25 for MNQ)
    """

    DEFAULT_LOOKBACK = 20  # Bars to track for daily high/low
    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size

    def __init__(
        self,
        lookback_period: int = DEFAULT_LOOKBACK,
        tick_size: float = DEFAULT_TICK_SIZE,
    ) -> None:
        """Initialize Level Sweep detector.

        Args:
            lookback_period: Number of bars to track for daily high/low
            tick_size: Tick size for futures contract (default: 0.25 for MNQ)
        """
        self._lookback_period = lookback_period
        self._tick_size = tick_size
        self._bars: deque[DollarBar] = deque(maxlen=lookback_period)

    def detect_sweep(self, bars: list[DollarBar]) -> LevelSweepEvent | None:
        """Detect level sweep from a list of Dollar Bars.

        Args:
            bars: List of Dollar Bars to analyze

        Returns:
            LevelSweepEvent if sweep detected, None otherwise
        """
        if len(bars) < 5:
            return None

        # Update internal bar history with latest bars
        for bar in bars[-self._lookback_period:]:
            if len(self._bars) == 0 or bar.timestamp > self._bars[-1].timestamp:
                self._bars.append(bar)

        # Need at least lookback period to detect sweep
        if len(self._bars) < self._lookback_period:
            return None

        # Calculate daily high and low from lookback period
        daily_high = max(bar.high for bar in list(self._bars)[:self._lookback_period])
        daily_low = min(bar.low for bar in list(self._bars)[:self._lookback_period])

        # Get recent bars (last 5 bars) for sweep confirmation
        recent_bars = list(self._bars)[-5:]
        if len(recent_bars) < 3:
            return None

        # Check for bullish sweep of daily high
        bullish_sweep = self._check_bullish_sweep(recent_bars, daily_high)
        if bullish_sweep:
            return bullish_sweep

        # Check for bearish sweep of daily low
        bearish_sweep = self._check_bearish_sweep(recent_bars, daily_low)
        if bearish_sweep:
            return bearish_sweep

        return None

    def _check_bullish_sweep(
        self,
        bars: list[DollarBar],
        daily_high: float,
    ) -> LevelSweepEvent | None:
        """Check for bullish sweep of daily high.

        A bullish sweep occurs when:
        1. Price trades above the daily high
        2. Price then reverses and closes back below the daily high

        Args:
            bars: Recent bars to analyze
            daily_high: Daily high level to check

        Returns:
            LevelSweepEvent if bullish sweep detected, None otherwise
        """
        if len(bars) < 3:
            return None

        # Find the highest high in the sequence
        max_high = max(bar.high for bar in bars)
        max_high_bar = next(bar for bar in bars if bar.high == max_high)

        # Check if we swept above daily high
        if max_high <= daily_high:
            return None

        # Find the bar that first broke above daily high
        sweep_bar = None
        for bar in bars:
            if bar.high > daily_high:
                sweep_bar = bar
                break

        if not sweep_bar:
            return None

        # Check for reversal: current close below daily high
        current_bar = bars[-1]
        if current_bar.close >= daily_high:
            return None

        # Calculate sweep extent in ticks
        sweep_extent_ticks = (max_high - daily_high) / self._tick_size

        # Create sweep event
        event = LevelSweepEvent(
            timestamp=current_bar.timestamp,
            level_type="daily_high",
            level_price=daily_high,
            sweep_extreme=max_high,
            reversal_price=current_bar.close,
            sweep_direction="bullish",
            sweep_extent_ticks=sweep_extent_ticks,
            volume_at_sweep=sweep_bar.volume,
        )

        logger.info(
            f"Bullish level sweep detected: high {daily_high:.2f} swept to "
            f"{max_high:.2f} ({sweep_extent_ticks:.1f} ticks), reversed to {current_bar.close:.2f}"
        )

        return event

    def _check_bearish_sweep(
        self,
        bars: list[DollarBar],
        daily_low: float,
    ) -> LevelSweepEvent | None:
        """Check for bearish sweep of daily low.

        A bearish sweep occurs when:
        1. Price trades below the daily low
        2. Price then reverses and closes back above the daily low

        Args:
            bars: Recent bars to analyze
            daily_low: Daily low level to check

        Returns:
            LevelSweepEvent if bearish sweep detected, None otherwise
        """
        if len(bars) < 3:
            return None

        # Find the lowest low in the sequence
        min_low = min(bar.low for bar in bars)
        min_low_bar = next(bar for bar in bars if bar.low == min_low)

        # Check if we swept below daily low
        if min_low >= daily_low:
            return None

        # Find the bar that first broke below daily low
        sweep_bar = None
        for bar in bars:
            if bar.low < daily_low:
                sweep_bar = bar
                break

        if not sweep_bar:
            return None

        # Check for reversal: current close above daily low
        current_bar = bars[-1]
        if current_bar.close <= daily_low:
            return None

        # Calculate sweep extent in ticks
        sweep_extent_ticks = (daily_low - min_low) / self._tick_size

        # Create sweep event
        event = LevelSweepEvent(
            timestamp=current_bar.timestamp,
            level_type="daily_low",
            level_price=daily_low,
            sweep_extreme=min_low,
            reversal_price=current_bar.close,
            sweep_direction="bearish",
            sweep_extent_ticks=sweep_extent_ticks,
            volume_at_sweep=sweep_bar.volume,
        )

        logger.info(
            f"Bearish level sweep detected: low {daily_low:.2f} swept to "
            f"{min_low:.2f} ({sweep_extent_ticks:.1f} ticks), reversed to {current_bar.close:.2f}"
        )

        return event

    def reset(self) -> None:
        """Reset detector state."""
        self._bars.clear()

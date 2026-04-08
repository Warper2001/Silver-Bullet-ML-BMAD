"""Wolf Pack Liquidity Sweep Detector.

This module implements liquidity sweep detection for the Wolf Pack 3-Edge strategy.
It detects when price sweeps beyond swing points and reverses, indicating a
microstructure edge.
"""

import logging
from collections import deque
from datetime import datetime

import numpy as np
from scipy import signal

from src.data.models import DollarBar
from src.detection.models import SwingPoint, WolfPackSweepEvent

logger = logging.getLogger(__name__)


class WolfPackLiquiditySweepDetector:
    """Detects liquidity sweeps for Wolf Pack 3-Edge strategy.

    Identifies swing points using scipy.signal.argrelextrema, detects
    sweeps beyond these levels, and confirms reversals.

    Attributes:
        _tick_size: Tick size for futures contract (0.25 for MNQ)
        _swing_window: Window size for swing point detection (default: 5 bars)
        _min_sweep_ticks: Minimum ticks beyond swing for valid sweep
        _bars: History of recent Dollar Bars
        _swing_highs: List of detected swing highs
        _swing_lows: List of detected swing lows
    """

    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size
    DEFAULT_SWING_WINDOW = 5  # Bars to check for swing points
    DEFAULT_MIN_SWEEP_TICKS = 5.0  # Minimum 5-tick sweep
    MAX_BAR_HISTORY = 100

    def __init__(
        self,
        tick_size: float = DEFAULT_TICK_SIZE,
        swing_window: int = DEFAULT_SWING_WINDOW,
        min_sweep_ticks: float = DEFAULT_MIN_SWEEP_TICKS,
    ) -> None:
        """Initialize Wolf Pack Liquidity Sweep detector.

        Args:
            tick_size: Tick size for futures contract (default: 0.25 for MNQ)
            swing_window: Window size for swing point detection
            min_sweep_ticks: Minimum ticks beyond swing for valid sweep
        """
        self._tick_size = tick_size
        self._swing_window = swing_window
        self._min_sweep_ticks = min_sweep_ticks

        self._bars: deque[DollarBar] = deque(maxlen=self.MAX_BAR_HISTORY)
        self._swing_highs: list[SwingPoint] = []
        self._swing_lows: list[SwingPoint] = []

    def process_bars(self, bars: list[DollarBar]) -> list[WolfPackSweepEvent]:
        """Process new bars and detect liquidity sweeps.

        Args:
            bars: List of new Dollar Bars to process

        Returns:
            List of detected sweep events
        """
        sweeps = []

        for bar in bars:
            if len(self._bars) == 0 or bar.timestamp > self._bars[-1].timestamp:
                self._bars.append(bar)

        # Need minimum bars for swing detection
        if len(self._bars) < self._swing_window * 2:
            return sweeps

        # Detect swing points
        self._detect_swing_points()

        # Check for sweeps at swing lows (bullish sweeps)
        for swing_low in self._swing_lows:
            sweep = self._check_bullish_sweep(swing_low)
            if sweep:
                sweeps.append(sweep)

        # Check for sweeps at swing highs (bearish sweeps)
        for swing_high in self._swing_highs:
            sweep = self._check_bearish_sweep(swing_high)
            if sweep:
                sweeps.append(sweep)

        return sweeps

    def _detect_swing_points(self) -> None:
        """Detect swing points using scipy.signal.argrelextrema."""
        if len(self._bars) < self._swing_window * 2:
            return

        bars_list = list(self._bars)
        highs = [bar.high for bar in bars_list]
        lows = [bar.low for bar in bars_list]

        # Find swing highs
        high_indices = signal.argrelextrema(
            np.array(highs), comparator=np.greater, order=self._swing_window
        )[0]

        for idx in high_indices:
            if idx >= len(bars_list):
                continue
            swing_point = SwingPoint(
                timestamp=bars_list[idx].timestamp,
                price=highs[idx],
                swing_type="high",
                bar_index=idx,
            )
            # Add if not already tracked
            if not any(
                sp.timestamp == swing_point.timestamp and sp.price == swing_point.price
                for sp in self._swing_highs
            ):
                self._swing_highs.append(swing_point)

        # Find swing lows
        low_indices = signal.argrelextrema(
            np.array(lows), comparator=np.less, order=self._swing_window
        )[0]

        for idx in low_indices:
            if idx >= len(bars_list):
                continue
            swing_point = SwingPoint(
                timestamp=bars_list[idx].timestamp,
                price=lows[idx],
                swing_type="low",
                bar_index=idx,
            )
            # Add if not already tracked
            if not any(
                sp.timestamp == swing_point.timestamp and sp.price == swing_point.price
                for sp in self._swing_lows
            ):
                self._swing_lows.append(swing_point)

        # Keep only recent swing points (last 10 of each type)
        if len(self._swing_highs) > 10:
            self._swing_highs = self._swing_highs[-10:]
        if len(self._swing_lows) > 10:
            self._swing_lows = self._swing_lows[-10:]

    def _check_bullish_sweep(
        self, swing_low: SwingPoint
    ) -> WolfPackSweepEvent | None:
        """Check for bullish sweep (price sweeps below swing low and reverses).

        A bullish sweep occurs when:
        1. Price trades below the swing low
        2. Price then reverses and closes back above the swing low

        Args:
            swing_low: Swing low to check for sweep

        Returns:
            WolfPackSweepEvent if bullish sweep detected, None otherwise
        """
        bars_list = list(self._bars)
        if len(bars_list) < 3:
            return None

        # Get recent bars (after swing low)
        recent_bars = [
            bar for bar in bars_list[swing_low.bar_index + 1 :]
            if bar.timestamp > swing_low.timestamp
        ]

        if len(recent_bars) < 3:
            return None

        # Check if price swept below swing low
        min_low = min(bar.low for bar in recent_bars)
        if min_low >= swing_low.price:
            return None

        # Calculate sweep extent in ticks
        sweep_extent_ticks = (swing_low.price - min_low) / self._tick_size

        # Check if sweep exceeds minimum
        if sweep_extent_ticks < self._min_sweep_ticks:
            return None

        # Check for reversal: current close above swing low
        current_bar = recent_bars[-1]
        if current_bar.close <= swing_low.price:
            return None

        # Create sweep event
        event = WolfPackSweepEvent(
            timestamp=current_bar.timestamp,
            swing_level=swing_low.price,
            swing_type="low",
            sweep_extreme=min_low,
            reversal_price=current_bar.close,
            sweep_direction="bullish",
            sweep_extent_ticks=sweep_extent_ticks,
            reversal_volume=float(current_bar.volume),
        )

        logger.info(
            f"Bullish liquidity sweep detected: swing_low {swing_low.price:.2f} "
            f"swept to {min_low:.2f} ({sweep_extent_ticks:.1f} ticks), "
            f"reversed to {current_bar.close:.2f}"
        )

        return event

    def _check_bearish_sweep(
        self, swing_high: SwingPoint
    ) -> WolfPackSweepEvent | None:
        """Check for bearish sweep (price sweeps above swing high and reverses).

        A bearish sweep occurs when:
        1. Price trades above the swing high
        2. Price then reverses and closes back below the swing high

        Args:
            swing_high: Swing high to check for sweep

        Returns:
            WolfPackSweepEvent if bearish sweep detected, None otherwise
        """
        bars_list = list(self._bars)
        if len(bars_list) < 3:
            return None

        # Get recent bars (after swing high)
        recent_bars = [
            bar for bar in bars_list[swing_high.bar_index + 1 :]
            if bar.timestamp > swing_high.timestamp
        ]

        if len(recent_bars) < 3:
            return None

        # Check if price swept above swing high
        max_high = max(bar.high for bar in recent_bars)
        if max_high <= swing_high.price:
            return None

        # Calculate sweep extent in ticks
        sweep_extent_ticks = (max_high - swing_high.price) / self._tick_size

        # Check if sweep exceeds minimum
        if sweep_extent_ticks < self._min_sweep_ticks:
            return None

        # Check for reversal: current close below swing high
        current_bar = recent_bars[-1]
        if current_bar.close >= swing_high.price:
            return None

        # Create sweep event
        event = WolfPackSweepEvent(
            timestamp=current_bar.timestamp,
            swing_level=swing_high.price,
            swing_type="high",
            sweep_extreme=max_high,
            reversal_price=current_bar.close,
            sweep_direction="bearish",
            sweep_extent_ticks=sweep_extent_ticks,
            reversal_volume=float(current_bar.volume),
        )

        logger.info(
            f"Bearish liquidity sweep detected: swing_high {swing_high.price:.2f} "
            f"swept to {max_high:.2f} ({sweep_extent_ticks:.1f} ticks), "
            f"reversed to {current_bar.close:.2f}"
        )

        return event

    def reset(self) -> None:
        """Reset detector state."""
        self._bars.clear()
        self._swing_highs.clear()
        self._swing_lows.clear()

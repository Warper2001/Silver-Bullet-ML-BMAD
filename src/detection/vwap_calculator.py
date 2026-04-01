"""VWAP Calculator for Triple Confluence Scalper strategy.

This module calculates the Volume-Weighted Average Price (VWAP)
for determining market bias and session-based price levels.
"""

import logging
from datetime import datetime, time

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class VWAPCalculator:
    """Calculates Volume-Weighted Average Price (VWAP) from Dollar Bars.

    VWAP = Cumulative(typical_price × volume) / Cumulative(volume)
    Where typical_price = (high + low + close) / 3

    Attributes:
        _session_start: Trading session start time (default 09:30:00 ET)
        _cumulative_tp_v: Cumulative (typical_price × volume)
        _cumulative_volume: Cumulative volume
        _session_date: Current session date for reset detection
        _tick_size: Tick size for futures contract (0.25 for MNQ)
    """

    DEFAULT_SESSION_START = "09:30:00"
    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size

    def __init__(
        self,
        session_start: str = DEFAULT_SESSION_START,
        tick_size: float = DEFAULT_TICK_SIZE,
    ) -> None:
        """Initialize VWAP calculator.

        Args:
            session_start: Trading session start time (default "09:30:00")
            tick_size: Tick size for futures contract (default: 0.25 for MNQ)
        """
        self._session_start = session_start
        self._tick_size = tick_size
        self._cumulative_tp_v = 0.0
        self._cumulative_volume = 0
        self._session_date: datetime | None = None

    def calculate_vwap(self, bars: list[DollarBar]) -> float:
        """Calculate VWAP from a list of Dollar Bars.

        Args:
            bars: List of Dollar Bars to calculate VWAP from

        Returns:
            VWAP value as a float
        """
        if not bars:
            return 0.0

        # Check if we need to reset (new session detected)
        if self._session_date is None:
            # First run - set session date from first bar
            self._session_date = bars[0].timestamp.date()
        elif bars[0].timestamp.date() != self._session_date:
            # New day detected - reset
            self.reset_session()
            self._session_date = bars[0].timestamp.date()

        # Process bars and update cumulative values
        for bar in bars:
            typical_price = (bar.high + bar.low + bar.close) / 3.0
            self._cumulative_tp_v += typical_price * bar.volume
            self._cumulative_volume += bar.volume

        # Calculate VWAP
        if self._cumulative_volume == 0:
            return 0.0

        vwap = self._cumulative_tp_v / self._cumulative_volume
        return vwap

    def reset_session(self) -> None:
        """Reset VWAP calculator for new trading session."""
        self._cumulative_tp_v = 0.0
        self._cumulative_volume = 0
        self._session_date = None
        logger.debug("VWAP calculator reset for new session")

    def get_bias(self, current_price: float, vwap: float) -> str:
        """Determine market bias based on price vs VWAP.

        Args:
            current_price: Current price to compare
            vwap: VWAP value to compare against

        Returns:
            "bullish" if price above VWAP
            "bearish" if price below VWAP
            "neutral" if within 2 ticks of VWAP
        """
        if vwap == 0:
            return "neutral"

        diff = current_price - vwap
        ticks_away = abs(diff) / self._tick_size

        if ticks_away <= 2:
            return "neutral"
        elif current_price > vwap:
            return "bullish"
        else:
            return "bearish"

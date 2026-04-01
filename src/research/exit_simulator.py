"""Exit Simulator for realistic trade exit simulation.

This module simulates trade exits by checking if stop loss or take profit
is hit during the hold period, rather than using fixed-bar exits.
"""

import logging
from datetime import datetime

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class ExitSimulator:
    """Simulates realistic trade exits with SL/TP execution.

    Instead of holding for a fixed number of bars, this simulator:
    1. Checks each bar during the hold period
    2. Exits immediately if SL or TP is hit
    3. Falls back to time-based exit if SL/TP not hit

    This provides more realistic performance metrics.
    """

    def __init__(
        self,
        max_hold_bars: int = 10,
        sl_buffer_ticks: int = 1,
        tp_buffer_ticks: int = 1,
    ) -> None:
        """Initialize exit simulator.

        Args:
            max_hold_bars: Maximum bars to hold (default 10 = 50 minutes for 5-min bars)
            sl_buffer_ticks: Buffer ticks for SL execution (slippage)
            tp_buffer_ticks: Buffer ticks for TP execution (slippage)
        """
        self.max_hold_bars = max_hold_bars
        self.sl_buffer_ticks = sl_buffer_ticks
        self.tp_buffer_ticks = tp_buffer_ticks
        self.tick_size = 0.25  # MNQ tick size

    def simulate_exit(
        self,
        entry_bar: DollarBar,
        bars: list[DollarBar],
        entry_index: int,
        direction: str,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[DollarBar, float, str, int]:
        """Simulate trade exit with SL/TP execution.

        Args:
            entry_bar: The bar where trade was entered
            bars: All bars (full dataset)
            entry_index: Index of entry_bar in bars
            direction: "long" or "short"
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Tuple of (exit_bar, exit_price, exit_reason, bars_held)
            exit_reason: "stop_loss", "take_profit", or "max_time"
        """
        # Calculate buffer-adjusted levels
        sl_buffer = self.sl_buffer_ticks * self.tick_size
        tp_buffer = self.tp_buffer_ticks * self.tick_size

        if direction == "long":
            # For long: TP above, SL below
            sl_trigger = stop_loss - sl_buffer  # More conservative (worse fill)
            tp_trigger = take_profit + tp_buffer  # More conservative (worse fill)
        else:  # short
            # For short: TP below, SL above
            sl_trigger = stop_loss + sl_buffer  # More conservative (worse fill)
            tp_trigger = take_profit - tp_buffer  # More conservative (worse fill)

        # Check each bar during hold period
        for i in range(entry_index + 1, len(bars)):
            bars_held = i - entry_index

            # Max hold time reached
            if bars_held >= self.max_hold_bars:
                return bars[i], bars[i].close, "max_time", bars_held

            bar = bars[i]

            if direction == "long":
                # Check if SL hit (price went down)
                if bar.low <= sl_trigger:
                    # Execute at SL price (or low if slippage exceeded)
                    exit_price = min(stop_loss, bar.low + sl_buffer)
                    logger.debug(
                        f"Long SL hit at bar {i}: "
                        f"low={bar.low:.2f} <= sl_trigger={sl_trigger:.2f}, "
                        f"exit_price={exit_price:.2f}"
                    )
                    return bar, exit_price, "stop_loss", bars_held

                # Check if TP hit (price went up)
                if bar.high >= tp_trigger:
                    # Execute at TP price (or high if slippage)
                    exit_price = max(take_profit, bar.high - tp_buffer)
                    logger.debug(
                        f"Long TP hit at bar {i}: "
                        f"high={bar.high:.2f} >= tp_trigger={tp_trigger:.2f}, "
                        f"exit_price={exit_price:.2f}"
                    )
                    return bar, exit_price, "take_profit", bars_held

            else:  # short
                # Check if SL hit (price went up)
                if bar.high >= sl_trigger:
                    # Execute at SL price (or high if slippage)
                    exit_price = max(stop_loss, bar.high - sl_buffer)
                    logger.debug(
                        f"Short SL hit at bar {i}: "
                        f"high={bar.high:.2f} >= sl_trigger={sl_trigger:.2f}, "
                        f"exit_price={exit_price:.2f}"
                    )
                    return bar, exit_price, "stop_loss", bars_held

                # Check if TP hit (price went down)
                if bar.low <= tp_trigger:
                    # Execute at TP price (or low if slippage)
                    exit_price = min(take_profit, bar.low + tp_buffer)
                    logger.debug(
                        f"Short TP hit at bar {i}: "
                        f"low={bar.low:.2f} <= tp_trigger={tp_trigger:.2f}, "
                        f"exit_price={exit_price:.2f}"
                    )
                    return bar, exit_price, "take_profit", bars_held

        # If we run out of bars, exit at last bar
        last_bar = bars[-1]
        bars_held = len(bars) - 1 - entry_index
        return last_bar, last_bar.close, "end_of_data", bars_held

    def get_exit_stats(self, exit_reasons: list[str]) -> dict[str, int]:
        """Get statistics on exit reasons.

        Args:
            exit_reasons: List of exit reasons

        Returns:
            Dictionary with counts of each exit reason
        """
        stats = {
            "stop_loss": 0,
            "take_profit": 0,
            "max_time": 0,
            "end_of_data": 0,
        }

        for reason in exit_reasons:
            if reason in stats:
                stats[reason] += 1

        return stats

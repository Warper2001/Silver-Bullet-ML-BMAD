"""Multi-timeframe Fibonacci nesting for FVG quality enhancement.

This module implements nested FVG detection using Fibonacci timeframe ratios
(5/21, 8/34, 13/55) to identify the highest-probability trading setups.
"""

import logging
from datetime import timedelta
from typing import Literal

import pandas as pd
from pandas import Timedelta

from src.data.models import DollarBar, FVGEvent, NestedFVGEvent

logger = logging.getLogger(__name__)


class MultiTimeframeNester:
    """Detect nested FVGs across Fibonacci timeframes.

    Fibonacci Timeframe Ratios:
        - 5/21 ≈ 1:4.2 (small/medium)
        - 8/34 ≈ 1:4.25 (small/medium)
        - 13/55 ≈ 1:4.23 (medium/large)

    Nesting Logic:
        A nested FVG occurs when a smaller timeframe FVG is completely
        contained within a larger timeframe FVG in the same direction.

        For bullish FVGs:
            - small_fvg.gap_range.bottom >= large_fvg.gap_range.bottom
            - small_fvg.gap_range.top <= large_fvg.gap_range.top

        For bearish FVGs:
            - Same containment logic (gap range fully within parent)

    Args:
        fibonacci_pairs: List of (small_tf, large_tf) tuples in minutes
        base_bar_duration: Duration of base bars in minutes (default: 1)

    Attributes:
        nesting_detection_count: Number of nested FVGs detected
    """

    def __init__(
        self,
        fibonacci_pairs: list[tuple[int, int]] | None = None,
        base_bar_duration: int = 1,
    ):
        """Initialize multi-timeframe nester with Fibonacci ratios."""
        self.fibonacci_pairs = fibonacci_pairs or [(5, 21), (8, 34), (13, 55)]
        self.base_bar_duration = base_bar_duration
        self.nesting_detection_count = 0

        # Caching for resampled timeframes (performance optimization)
        self._cached_bars_length = 0  # Track how many bars were cached
        self._cached_timeframes: dict[int, list[DollarBar]] = {}  # {timeframe: resampled_bars}

    def resample_bars(
        self, bars: list[DollarBar], timeframe_minutes: int
    ) -> list[DollarBar]:
        """Resample dollar bars to a larger timeframe using OHLC aggregation.

        Args:
            bars: List of base timeframe Dollar Bars
            timeframe_minutes: Target timeframe in minutes

        Returns:
            List of resampled Dollar Bars at target timeframe

        Raises:
            ValueError: If fewer than 3 bars after resampling
        """
        if not bars:
            return []

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "notional_value": bar.notional_value,
            }
            for bar in bars
        ])

        # Set timestamp as index
        df = df.set_index("timestamp")

        # Resample to target timeframe with 'origin' parameter for clean alignment
        # origin='epoch' ensures bars align to XX:00, XX:05, XX:10, etc.
        resampled = df.resample(
            f"{timeframe_minutes}min",
            origin="epoch",
            label="right",  # Use right label (timestamp is end of bar)
            closed="right",  # Right-closed intervals
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "notional_value": "sum",
        })

        # Drop any rows with NaN values (insufficient data for resampling)
        resampled = resampled.dropna()

        # Convert back to DollarBar objects
        resampled_bars = [
            DollarBar(
                timestamp=timestamp.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                notional_value=float(row["notional_value"]),
                is_forward_filled=False,  # Resampled bars are not forward-filled
            )
            for timestamp, row in resampled.iterrows()
        ]

        return resampled_bars

    def detect_nested_fvg(
        self,
        base_fvg: FVGEvent,
        parent_fvg: FVGEvent,
        small_tf: int,
        large_tf: int,
    ) -> NestedFVGEvent | None:
        """Check if base FVG is nested within parent FVG.

        Args:
            base_fvg: Smaller timeframe FVG (e.g., 5-minute)
            parent_fvg: Larger timeframe FVG (e.g., 21-minute)
            small_tf: Small timeframe in minutes
            large_tf: Large timeframe in minutes

        Returns:
            NestedFVGEvent if nesting detected, None otherwise
        """
        # Check direction alignment
        if base_fvg.direction != parent_fvg.direction:
            return None

        # Check containment: base FVG must be fully within parent FVG
        is_contained = (
            base_fvg.gap_range.bottom >= parent_fvg.gap_range.bottom
            and base_fvg.gap_range.top <= parent_fvg.gap_range.top
        )

        if not is_contained:
            return None

        # Nested FVG detected!
        parent_fvg_id = f"{parent_fvg.timestamp.isoformat()}_{parent_fvg.bar_index}"

        logger.info(
            f"Nested {base_fvg.direction.upper()} FVG detected: "
            f"{small_tf}-min inside {large_tf}-min timeframe. "
            f"Parent gap: [{parent_fvg.gap_range.bottom:.2f}, {parent_fvg.gap_range.top:.2f}], "
            f"Child gap: [{base_fvg.gap_range.bottom:.2f}, {base_fvg.gap_range.top:.2f}]"
        )

        self.nesting_detection_count += 1

        return NestedFVGEvent(
            timestamp=base_fvg.timestamp,
            direction=base_fvg.direction,
            child_fvg=base_fvg,
            parent_fvg=parent_fvg,
            nesting_level=1,  # Single-level nesting
            timeframe_pair=(small_tf, large_tf),
            parent_fvg_id=parent_fvg_id,
            bar_index=base_fvg.bar_index,
            confidence=0.0,  # Calculated later by pipeline
        )

    def _update_cache_if_needed(self, bars: list[DollarBar]) -> None:
        """Update cached resampled timeframes if new bars were added.

        This is a performance optimization - we only resample when the bar list
        has grown, avoiding redundant resampling operations on every check.

        Args:
            bars: Current list of all dollar bars
        """
        current_bars_length = len(bars)

        # Only update if we have more bars than last time
        if current_bars_length > self._cached_bars_length:
            # Clear cache and rebuild with new bars
            self._cached_timeframes = {}
            for _, large_tf in self.fibonacci_pairs:
                try:
                    resampled = self.resample_bars(bars, large_tf)
                    self._cached_timeframes[large_tf] = resampled
                    logger.debug(f"Cached {len(resampled)} bars at {large_tf}-min timeframe")
                except Exception as e:
                    logger.error(f"Failed to cache {large_tf}-min bars: {e}")
                    self._cached_timeframes[large_tf] = []

            self._cached_bars_length = current_bars_length

    def find_nesting_across_timeframes(
        self,
        base_fvg: FVGEvent,
        bars: list[DollarBar],
        fvg_history: dict[int, list[FVGEvent]],
    ) -> list[NestedFVGEvent]:
        """Search for nested FVGs across all Fibonacci timeframe pairs.

        Args:
            base_fvg: Base timeframe FVG to check for nesting
            bars: All available dollar bars for resampling
            fvg_history: Dictionary mapping timeframe -> list of FVGs at that timeframe

        Returns:
            List of NestedFVGEvents (empty if no nesting found)
        """
        nested_fvgs = []

        # Update cache if new bars were added (performance optimization)
        self._update_cache_if_needed(bars)

        for small_tf, large_tf in self.fibonacci_pairs:
            # Use cached resampled timeframes instead of resampling every time
            large_tf_bars = self._cached_timeframes.get(large_tf, [])

            if not large_tf_bars:
                logger.warning(f"No cached bars available for {large_tf}-min timeframe")
                continue

            # Detect FVGs at larger timeframe if not already in history
            if large_tf not in fvg_history:
                from src.detection.fvg_detection import (
                    detect_bullish_fvg,
                    detect_bearish_fvg,
                )

                large_tf_fvgs = []
                for i in range(len(large_tf_bars)):
                    if base_fvg.direction == "bullish":
                        fvg = detect_bullish_fvg(large_tf_bars, i)
                    else:
                        fvg = detect_bearish_fvg(large_tf_bars, i)

                    if fvg:
                        large_tf_fvgs.append(fvg)

                fvg_history[large_tf] = large_tf_fvgs

            # Check for nesting with each parent FVG
            for parent_fvg in fvg_history[large_tf]:
                nested_fvg = self.detect_nested_fvg(
                    base_fvg, parent_fvg, small_tf, large_tf
                )
                if nested_fvg:
                    nested_fvgs.append(nested_fvg)

        return nested_fvgs

    def check_nesting(
        self,
        base_fvg: FVGEvent,
        bars: list[DollarBar],
        fvg_history: dict[int, list[FVGEvent]] | None = None,
    ) -> tuple[bool, list[NestedFVGEvent]]:
        """Check if base FVG has any nesting across Fibonacci timeframes.

        This is the main entry point for multi-timeframe nesting detection.

        Args:
            base_fvg: Base timeframe FVG to check
            bars: All available dollar bars for resampling
            fvg_history: Optional cache of FVGs at different timeframes

        Returns:
            Tuple of (has_nesting, nested_fvgs)
                - has_nesting: True if at least one nesting found
                - nested_fvgs: List of NestedFVGEvents (may be empty)
        """
        if fvg_history is None:
            fvg_history = {}

        # Ensure base timeframe is in history
        base_tf = self.base_bar_duration
        if base_tf not in fvg_history:
            fvg_history[base_tf] = [base_fvg]

        nested_fvgs = self.find_nesting_across_timeframes(
            base_fvg, bars, fvg_history
        )

        has_nesting = len(nested_fvgs) > 0

        if has_nesting:
            logger.info(
                f"Found {len(nested_fvgs)} nested FVG(s) for base {base_fvg.direction.upper()} FVG "
                f"at bar index {base_fvg.bar_index}"
            )

        return has_nesting, nested_fvgs

    def reset_metrics(self) -> None:
        """Reset metrics (e.g., for backtesting or new trading session)."""
        self.nesting_detection_count = 0

    def get_metrics(self) -> dict[str, int]:
        """Get current metrics.

        Returns:
            Dictionary with nesting statistics
        """
        return {
            "nesting_detection_count": self.nesting_detection_count,
            "fibonacci_pairs": len(self.fibonacci_pairs),
        }

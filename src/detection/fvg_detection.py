"""Fair Value Gap (FVG) detection algorithms.

This module implements the core algorithms for identifying Fair Value Gaps
(3-candle patterns), calculating gap sizes, and detecting when gaps are filled.
"""

import logging

from src.data.models import DollarBar, FVGEvent, GapRange

logger = logging.getLogger(__name__)

# MNQ contract specifications
MNQ_TICK_SIZE = 0.25  # 0.25 points per tick
MNQ_POINT_VALUE = 20.0  # $20 per point


def detect_bullish_fvg(bars: list[DollarBar], current_index: int) -> FVGEvent | None:
    """Detect bullish FVG: price gap where candle 1 close > candle 3 open.

    A bullish FVG indicates aggressive buying pressure leaving a gap
    that price often revisits (fills) before continuing higher.

    Args:
        bars: List of Dollar Bars (OHLCV data)
        current_index: Index of most recent bar (candle 3)

    Returns:
        FVGEvent if bullish FVG detected, None otherwise

    Visual Example:
        Candle 1: [====|=====]  Close at 11850, High at 11860
        Candle 2:    [===|===]   Gap down (price dropped)
        Candle 3:       [===|===] Open at 11840, Low at 11830

        Bullish FVG: Gap between 11860 (candle 1 high) and 11830 (candle 3 low)
    """
    # Need at least 3 bars
    if current_index < 2:
        return None

    candle_1 = bars[current_index - 2]
    _ = bars[current_index - 1]  # candle_2 (unused, part of 3-candle pattern)
    candle_3 = bars[current_index]

    # Check for bullish FVG condition: candle 1 close > candle 3 open
    if candle_1.close <= candle_3.open:
        return None

    # Calculate gap range
    top = candle_1.high
    bottom = candle_3.low

    # Verify actual gap exists
    if top <= bottom:
        return None

    # Calculate gap size
    gap_size_points = top - bottom
    gap_size_ticks = gap_size_points / MNQ_TICK_SIZE
    gap_size_dollars = gap_size_points * MNQ_POINT_VALUE

    # Bullish FVG detected!
    return FVGEvent(
        timestamp=candle_3.timestamp,
        direction="bullish",
        gap_range=GapRange(top=top, bottom=bottom),
        gap_size_ticks=gap_size_ticks,
        gap_size_dollars=gap_size_dollars,
        bar_index=current_index,
        filled=False,
    )


def detect_bearish_fvg(bars: list[DollarBar], current_index: int) -> FVGEvent | None:
    """Detect bearish FVG: price gap where candle 1 close < candle 3 open.

    A bearish FVG indicates aggressive selling pressure leaving a gap
    that price often revisits (fills) before continuing lower.

    Args:
        bars: List of Dollar Bars (OHLCV data)
        current_index: Index of most recent bar (candle 3)

    Returns:
        FVGEvent if bearish FVG detected, None otherwise

    Visual Example:
        Candle 1:       [===|===] Close at 11850, Low at 11840
        Candle 2:    [===|===]   Gap up (price rose)
        Candle 3: [====|=====]  Open at 11860, High at 11870

        Bearish FVG: Gap between 11870 (candle 3 high) and 11840 (candle 1 low)
    """
    # Need at least 3 bars
    if current_index < 2:
        return None

    candle_1 = bars[current_index - 2]
    _ = bars[current_index - 1]  # candle_2 (unused, part of 3-candle pattern)
    candle_3 = bars[current_index]

    # Check for bearish FVG condition: candle 1 close < candle 3 open
    if candle_1.close >= candle_3.open:
        return None

    # Calculate gap range (reversed for bearish)
    top = candle_3.high
    bottom = candle_1.low

    # Verify actual gap exists
    if top <= bottom:
        return None

    # Calculate gap size
    gap_size_points = top - bottom
    gap_size_ticks = gap_size_points / MNQ_TICK_SIZE
    gap_size_dollars = gap_size_points * MNQ_POINT_VALUE

    # Bearish FVG detected!
    return FVGEvent(
        timestamp=candle_3.timestamp,
        direction="bearish",
        gap_range=GapRange(top=top, bottom=bottom),
        gap_size_ticks=gap_size_ticks,
        gap_size_dollars=gap_size_dollars,
        bar_index=current_index,
        filled=False,
    )


def check_fvg_fill(fvg: FVGEvent, bar: DollarBar) -> bool:
    """Check if a Dollar Bar has filled an FVG.

    An FVG is considered filled when price trades through the gap range.

    Args:
        fvg: FVG event to check
        bar: Current Dollar Bar

    Returns:
        True if FVG is filled by this bar

    Logic for Bullish FVG:
        - Bullish FVG is filled when bar.low <= fvg.gap_range.bottom
        - Price has traded down through the entire gap

    Logic for Bearish FVG:
        - Bearish FVG is filled when bar.high >= fvg.gap_range.top
        - Price has traded up through the entire gap
    """
    if fvg.filled:
        return True  # Already filled

    if fvg.direction == "bullish":
        # Bullish FVG filled when price trades down to bottom of gap
        if bar.low <= fvg.gap_range.bottom:
            return True
    else:  # bearish
        # Bearish FVG filled when price trades up to top of gap
        if bar.high >= fvg.gap_range.top:
            return True

    return False

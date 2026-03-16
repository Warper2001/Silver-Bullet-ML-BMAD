"""Liquidity sweep detection algorithms.

This module implements the core algorithms for identifying liquidity sweeps
(price briefly trading beyond swing points then reversing), calculating sweep
depth, and validating minimum sweep requirements.
"""

import logging

from src.data.models import DollarBar, LiquiditySweepEvent, SwingPoint

logger = logging.getLogger(__name__)

# MNQ contract specifications
MNQ_TICK_SIZE = 0.25  # 0.25 points per tick
MNQ_POINT_VALUE = 20.0  # $20 per point


def check_bullish_sweep(
    bar: DollarBar,
    swing_low: SwingPoint,
    min_sweep_ticks: float = 5.0,
) -> LiquiditySweepEvent | None:
    """Check if a Dollar Bar shows a bullish liquidity sweep.

    A bullish liquidity sweep occurs when price trades below a swing low
    (stopping out bears) then closes back above it, indicating a reversal.

    Args:
        bar: Current Dollar Bar to check
        swing_low: Swing low point to check against
        min_sweep_ticks: Minimum ticks below swing low for valid sweep (default: 5)

    Returns:
        LiquiditySweepEvent if bullish sweep detected, None otherwise

    Visual Example:
        Swing Low:    ===== 11800.0 (support)
        Sweep Bar:   [===|=====]  Low 11790, Close 11810
                              ↓
                    Price dropped 10 ticks (2.5 points) below support,
                    then closed above it = Bullish Sweep
    """
    # Price must trade below swing low
    if bar.low >= swing_low.price:
        return None

    # Calculate sweep depth
    sweep_depth_points = swing_low.price - bar.low
    sweep_depth_ticks = sweep_depth_points / MNQ_TICK_SIZE

    # Validate minimum sweep depth
    if sweep_depth_ticks < min_sweep_ticks:
        return None

    # Price must close above swing low (reversal confirmation)
    if bar.close <= swing_low.price:
        return None

    # Bullish liquidity sweep detected!
    sweep_depth_dollars = sweep_depth_points * MNQ_POINT_VALUE

    return LiquiditySweepEvent(
        timestamp=bar.timestamp,
        direction="bullish",
        swing_point_price=swing_low.price,
        sweep_depth_ticks=sweep_depth_ticks,
        sweep_depth_dollars=sweep_depth_dollars,
        bar_index=0,  # Set by detector
    )


def check_bearish_sweep(
    bar: DollarBar,
    swing_high: SwingPoint,
    min_sweep_ticks: float = 5.0,
) -> LiquiditySweepEvent | None:
    """Check if a Dollar Bar shows a bearish liquidity sweep.

    A bearish liquidity sweep occurs when price trades above a swing high
    (stopping out bulls) then closes back below it, indicating a reversal.

    Args:
        bar: Current Dollar Bar to check
        swing_high: Swing high point to check against
        min_sweep_ticks: Minimum ticks above swing high for valid sweep (default: 5)

    Returns:
        LiquiditySweepEvent if bearish sweep detected, None otherwise

    Visual Example:
        Swing High:   ===== 11900.0 (resistance)
        Sweep Bar:   [=====|===]  High 11910, Close 11890
                              ↓
                    Price rose 10 ticks (2.5 points) above resistance,
                    then closed below it = Bearish Sweep
    """
    # Price must trade above swing high
    if bar.high <= swing_high.price:
        return None

    # Calculate sweep depth
    sweep_depth_points = bar.high - swing_high.price
    sweep_depth_ticks = sweep_depth_points / MNQ_TICK_SIZE

    # Validate minimum sweep depth
    if sweep_depth_ticks < min_sweep_ticks:
        return None

    # Price must close below swing high (reversal confirmation)
    if bar.close >= swing_high.price:
        return None

    # Bearish liquidity sweep detected!
    sweep_depth_dollars = sweep_depth_points * MNQ_POINT_VALUE

    return LiquiditySweepEvent(
        timestamp=bar.timestamp,
        direction="bearish",
        swing_point_price=swing_high.price,
        sweep_depth_ticks=sweep_depth_ticks,
        sweep_depth_dollars=sweep_depth_dollars,
        bar_index=0,  # Set by detector
    )


def detect_bullish_liquidity_sweep(
    bars: list[DollarBar],
    current_index: int,
    swing_low: SwingPoint,
    min_sweep_ticks: float = 5.0,
) -> LiquiditySweepEvent | None:
    """Detect bullish liquidity sweep from bar list.

    This is a convenience wrapper around check_bullish_sweep that
    extracts the current bar from the list and validates the index.

    Args:
        bars: List of Dollar Bars
        current_index: Index of current bar to check
        swing_low: Swing low point to check against
        min_sweep_ticks: Minimum ticks below swing low for valid sweep

    Returns:
        LiquiditySweepEvent if bullish sweep detected, None otherwise
    """
    if current_index < 0 or current_index >= len(bars):
        return None

    bar = bars[current_index]
    sweep = check_bullish_sweep(bar, swing_low, min_sweep_ticks)

    if sweep:
        sweep.bar_index = current_index

    return sweep


def detect_bearish_liquidity_sweep(
    bars: list[DollarBar],
    current_index: int,
    swing_high: SwingPoint,
    min_sweep_ticks: float = 5.0,
) -> LiquiditySweepEvent | None:
    """Detect bearish liquidity sweep from bar list.

    This is a convenience wrapper around check_bearish_sweep that
    extracts the current bar from the list and validates the index.

    Args:
        bars: List of Dollar Bars
        current_index: Index of current bar to check
        swing_high: Swing high point to check against
        min_sweep_ticks: Minimum ticks above swing high for valid sweep

    Returns:
        LiquiditySweepEvent if bearish sweep detected, None otherwise
    """
    if current_index < 0 or current_index >= len(bars):
        return None

    bar = bars[current_index]
    sweep = check_bearish_sweep(bar, swing_high, min_sweep_ticks)

    if sweep:
        sweep.bar_index = current_index

    return sweep

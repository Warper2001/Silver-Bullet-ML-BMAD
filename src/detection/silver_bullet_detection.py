"""Silver Bullet setup detection algorithms.

This module implements the core algorithms for identifying Silver Bullet setups
by combining MSS, FVG, and liquidity sweep patterns in confluence.
"""

import logging

from src.data.models import (
    FVGEvent,
    LiquiditySweepEvent,
    MSSEvent,
    SilverBulletSetup,
)

logger = logging.getLogger(__name__)


def check_silver_bullet_setup(
    mss_event: MSSEvent,
    fvg_event: FVGEvent,
    sweep_event: LiquiditySweepEvent | None = None,
    max_bar_distance: int = 10,
) -> SilverBulletSetup | None:
    """Check if MSS and FVG (and optionally sweep) form a Silver Bullet setup.

    A Silver Bullet setup is a high-probability trading opportunity where
    multiple ICT patterns align in confluence within a short time window.

    Args:
        mss_event: Market Structure Shift event
        fvg_event: Fair Value Gap event
        sweep_event: Optional liquidity sweep event (3-pattern confluence)
        max_bar_distance: Maximum bars between MSS and FVG (default: 10)

    Returns:
        SilverBulletSetup if valid setup detected, None otherwise

    Recognition Rules:
        - MSS and FVG must have matching directions (both bullish or both bearish)
        - MSS must precede or coincide with FVG (mss.bar_index <= fvg.bar_index)
        - MSS and FVG must occur within max_bar_distance bars of each other
        - When sweep present: sweep must precede MSS (sweep.bar_index < mss.bar_index)
        - FVG midpoint must be in the correct premium/discount zone relative to MSS range
        - Setup direction is determined by MSS direction
        - Entry zone is the FVG gap range
        - Invalidation point is the opposite swing point from MSS
        - Priority: "high" if sweep present (3-pattern), "medium" otherwise

    Visual Example (Bullish Silver Bullet):
        Sweep: Price briefly swept below swing low (bar 5)
        MSS: Price breaks above swing high (11850) with volume (bar 10)
        FVG: Gap forms at 11810-11830 in discount zone (bar 11)
        → Silver Bullet setup with entry zone 11810-11830, invalidation at 11800
    """
    # Both MSS and FVG are required
    if mss_event is None or fvg_event is None:
        return None

    # Check direction matching
    if mss_event.direction != fvg_event.direction:
        logger.debug(
            f"MSS and FVG directions mismatch: MSS={mss_event.direction}, "
            f"FVG={fvg_event.direction}"
        )
        return None

    # MSS must precede or coincide with FVG (FVG is created by the MSS displacement leg)
    if mss_event.bar_index > fvg_event.bar_index:
        logger.debug(
            f"MSS occurs after FVG: MSS bar={mss_event.bar_index}, "
            f"FVG bar={fvg_event.bar_index} — invalid sequence"
        )
        return None

    # Check bar distance (forward distance: FVG must be within max_bar_distance of MSS)
    bar_distance = fvg_event.bar_index - mss_event.bar_index
    if bar_distance > max_bar_distance:
        logger.debug(
            f"MSS and FVG too far apart: {bar_distance} bars (max: {max_bar_distance})"
        )
        return None

    # When sweep present: enforce sweep → MSS → FVG ordering
    if sweep_event is not None:
        if sweep_event.direction != mss_event.direction:
            logger.debug(
                f"Sweep direction mismatch: MSS={mss_event.direction}, "
                f"Sweep={sweep_event.direction}"
            )
            return None
        # Sweep must precede MSS (liquidity is swept, then structure shifts)
        if sweep_event.bar_index >= mss_event.bar_index:
            logger.debug(
                f"Sweep must precede MSS: sweep bar={sweep_event.bar_index}, "
                f"MSS bar={mss_event.bar_index}"
            )
            return None

    # Premium/discount zone validation:
    # The MSS swing range spans from the prior swing point (broken) to the breakout level.
    # Bullish FVGs must be in discount (midpoint below 50% equilibrium of swing range).
    # Bearish FVGs must be in premium (midpoint above 50% equilibrium of swing range).
    fvg_midpoint = (fvg_event.gap_range.top + fvg_event.gap_range.bottom) / 2
    swing_price = mss_event.swing_point.price
    breakout_price = mss_event.breakout_price
    equilibrium = (swing_price + breakout_price) / 2

    if mss_event.direction == "bullish":
        if fvg_midpoint >= equilibrium:
            logger.debug(
                f"Bullish FVG not in discount zone: midpoint={fvg_midpoint:.2f}, "
                f"equilibrium={equilibrium:.2f}"
            )
            return None
    else:  # bearish
        if fvg_midpoint <= equilibrium:
            logger.debug(
                f"Bearish FVG not in premium zone: midpoint={fvg_midpoint:.2f}, "
                f"equilibrium={equilibrium:.2f}"
            )
            return None

    # Determine confluence count and priority
    confluence_count = 3 if sweep_event else 2
    priority = "high" if sweep_event else "medium"

    # Entry zone from FVG
    entry_zone_top = fvg_event.gap_range.top
    entry_zone_bottom = fvg_event.gap_range.bottom

    # Invalidation point from MSS swing point
    invalidation_point = mss_event.swing_point.price

    return SilverBulletSetup(
        timestamp=mss_event.timestamp,
        direction=mss_event.direction,
        mss_event=mss_event,
        fvg_event=fvg_event,
        liquidity_sweep_event=sweep_event,
        entry_zone_top=entry_zone_top,
        entry_zone_bottom=entry_zone_bottom,
        invalidation_point=invalidation_point,
        confluence_count=confluence_count,
        priority=priority,
        bar_index=mss_event.bar_index,
    )


def detect_silver_bullet_setup(
    mss_events: list[MSSEvent],
    fvg_events: list[FVGEvent],
    sweep_events: list[LiquiditySweepEvent] | None = None,
    max_bar_distance: int = 10,
) -> list[SilverBulletSetup]:
    """Detect Silver Bullet setups from lists of events.

    This function searches for combinations of MSS and FVG events that form
    valid Silver Bullet setups, optionally including liquidity sweeps.

    Args:
        mss_events: List of MSS events
        fvg_events: List of FVG events
        sweep_events: Optional list of liquidity sweep events
        max_bar_distance: Maximum bars between MSS and FVG (default: 10)

    Returns:
        List of SilverBulletSetup objects (may be empty)

    Algorithm:
        1. For each MSS event, search for matching FVG events
        2. Enforce sequence: sweep (if any) < MSS ≤ FVG in bar_index
        3. If sweep events provided, find the most recent sweep preceding the MSS
        4. Return all valid setups (may have multiple from same events)
    """
    if sweep_events is None:
        sweep_events = []

    setups = []

    for mss in mss_events:
        for fvg in fvg_events:
            # Find the most recent sweep that precedes this MSS (if any)
            matching_sweep = None
            for sweep in sweep_events:
                if (
                    sweep.direction == mss.direction
                    and sweep.bar_index < mss.bar_index
                    and abs(sweep.bar_index - mss.bar_index) <= max_bar_distance
                ):
                    if matching_sweep is None or sweep.bar_index > matching_sweep.bar_index:
                        matching_sweep = sweep

            setup = check_silver_bullet_setup(
                mss_event=mss,
                fvg_event=fvg,
                sweep_event=matching_sweep,
                max_bar_distance=max_bar_distance,
            )

            if setup:
                setups.append(setup)
                logger.debug(
                    f"Silver Bullet setup detected: {setup.direction} "
                    f"with {setup.confluence_count}-pattern confluence, "
                    f"priority={setup.priority}"
                )

    return setups

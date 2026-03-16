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
        - MSS and FVG must occur within max_bar_distance bars of each other
        - Setup direction is determined by MSS direction
        - Entry zone is the FVG gap range
        - Invalidation point is the opposite swing point from MSS
        - Priority: "high" if sweep present (3-pattern), "medium" otherwise

    Visual Example (Bullish Silver Bullet):
        MSS: Price breaks above swing high (11850) with volume
        FVG: Gap forms at 11860-11880 (within 10 bars)
        Sweep: Price briefly swept below swing low then recovered
        → Silver Bullet setup with entry zone 11860-11880, invalidation at 11800
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

    # Check bar distance
    bar_distance = abs(mss_event.bar_index - fvg_event.bar_index)
    if bar_distance > max_bar_distance:
        logger.debug(
            f"MSS and FVG too far apart: {bar_distance} bars (max: {max_bar_distance})"
        )
        return None

    # Check sweep direction if present
    if sweep_event is not None:
        if sweep_event.direction != mss_event.direction:
            logger.debug(
                f"Sweep direction mismatch: MSS={mss_event.direction}, "
                f"Sweep={sweep_event.direction}"
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

    # Silver Bullet setup detected!
    return SilverBulletSetup(
        timestamp=mss_event.timestamp,  # Use MSS timestamp as setup time
        direction=mss_event.direction,  # Direction from MSS
        mss_event=mss_event,
        fvg_event=fvg_event,
        liquidity_sweep_event=sweep_event,
        entry_zone_top=entry_zone_top,
        entry_zone_bottom=entry_zone_bottom,
        invalidation_point=invalidation_point,
        confluence_count=confluence_count,
        priority=priority,
        bar_index=mss_event.bar_index,  # Use MSS bar index
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
        2. Check if they form a valid setup (direction, distance)
        3. If sweep events provided, look for matching sweeps
        4. Return all valid setups (may have multiple from same events)
    """
    if sweep_events is None:
        sweep_events = []

    setups = []

    # Search for MSS + FVG combinations
    for mss in mss_events:
        for fvg in fvg_events:
            # Check for valid setup
            setup = check_silver_bullet_setup(
                mss_event=mss,
                fvg_event=fvg,
                sweep_event=None,
                max_bar_distance=max_bar_distance,
            )

            if setup is None:
                continue

            # Look for matching sweep
            for sweep in sweep_events:
                # Check if sweep matches setup
                if (
                    sweep.direction == mss.direction
                    and abs(sweep.bar_index - mss.bar_index) <= max_bar_distance
                ):
                    # Create enhanced setup with sweep
                    setup = check_silver_bullet_setup(
                        mss_event=mss,
                        fvg_event=fvg,
                        sweep_event=sweep,
                        max_bar_distance=max_bar_distance,
                    )
                    break  # Use first matching sweep

            if setup:
                setups.append(setup)
                logger.info(
                    f"Silver Bullet setup detected: {setup.direction} "
                    f"with {setup.confluence_count}-pattern confluence, "
                    f"priority={setup.priority}"
                )

    return setups

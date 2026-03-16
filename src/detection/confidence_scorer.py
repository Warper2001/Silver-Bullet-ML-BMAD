"""Confidence score calculation for Silver Bullet setups.

This module implements confidence scoring based on confluence factors:
- Pattern confluence (2 vs 3 patterns)
- Volume ratio on MSS breakout
- FVG size (gap magnitude)
"""

import logging

from src.data.models import SilverBulletSetup

logger = logging.getLogger(__name__)

# Confidence scoring thresholds
VOLUME_RATIO_HIGH_THRESHOLD = 1.5  # Above this increases score
VOLUME_RATIO_LOW_THRESHOLD = 1.2  # Below this decreases score
FVG_SIZE_LARGE_THRESHOLD_TICKS = 20  # Above this increases score
FVG_SIZE_SMALL_THRESHOLD_TICKS = 10  # Below this decreases score

# Base scores
BASE_SCORE_2_PATTERNS = 1  # MSS + FVG only
BASE_SCORE_3_PATTERNS = 3  # MSS + FVG + Sweep


def calculate_confidence_score(setup: SilverBulletSetup) -> int:
    """Calculate confidence score for a Silver Bullet setup.

    Confidence scores range from 1-5 based on confluence factors:
    - Base score: 1 for MSS + FVG (2 patterns)
    - Base score: 3 for MSS + FVG + Sweep (3 patterns)
    - +1 for high volume ratio (> 1.5x average)
    - +1 for large FVG size (> 20 ticks)
    - -1 for weak patterns (volume < 1.2x or FVG < 10 ticks)
    - Maximum score: 5

    Args:
        setup: Silver Bullet setup to score

    Returns:
        Confidence score (1-5)

    Scoring Matrix:
        +-----------------------------------+
        | Patterns | Volume | FVG    | Score |
        +-----------------------------------+
        | 2        | < 1.2x | < 10   | 1     |
        | 2        | 1.2-1.5x| < 10   | 1     |
        | 2        | > 1.5x | < 10   | 1     |
        | 2        | < 1.2x | 10-20  | 1     |
        | 2        | 1.2-1.5x| 10-20  | 1     |
        | 2        | > 1.5x | 10-20  | 2     |
        | 2        | any    | > 20   | 2     |
        +-----------------------------------+
        | 3        | < 1.2x | < 10   | 2     |
        | 3        | 1.2-1.5x| < 10   | 3     |
        | 3        | > 1.5x | < 10   | 4     |
        | 3        | < 1.2x | 10-20  | 2     |
        | 3        | 1.2-1.5x| 10-20  | 3     |
        | 3        | > 1.5x | 10-20  | 4     |
        | 3        | any    | > 20   | 5     |
        +-----------------------------------+

    Examples:
        >>> setup.confluence_count = 2
        >>> setup.mss_event.volume_ratio = 1.3
        >>> setup.fvg_event.gap_size_ticks = 15
        >>> calculate_confidence_score(setup)
        1

        >>> setup.confluence_count = 3
        >>> setup.mss_event.volume_ratio = 2.0
        >>> setup.fvg_event.gap_size_ticks = 25
        >>> calculate_confidence_score(setup)
        5
    """
    # Start with base score from pattern confluence
    if setup.confluence_count == 3:
        score = BASE_SCORE_3_PATTERNS
    else:  # confluence_count == 2
        score = BASE_SCORE_2_PATTERNS

    # Adjust based on volume ratio
    volume_ratio = setup.mss_event.volume_ratio

    if volume_ratio > VOLUME_RATIO_HIGH_THRESHOLD:
        score += 1
        logger.debug(f"High volume ratio ({volume_ratio:.2f}) increased score")
    elif volume_ratio < VOLUME_RATIO_LOW_THRESHOLD and score > 1:
        # Decrease score for weak volume (but not below 1)
        score -= 1
        logger.debug(f"Low volume ratio ({volume_ratio:.2f}) decreased score")

    # Adjust based on FVG size
    fvg_size_ticks = setup.fvg_event.gap_size_ticks

    if fvg_size_ticks > FVG_SIZE_LARGE_THRESHOLD_TICKS:
        score += 1
        logger.debug(f"Large FVG ({fvg_size_ticks:.0f} ticks) increased score")
    elif fvg_size_ticks < FVG_SIZE_SMALL_THRESHOLD_TICKS and score > 1:
        # Decrease score for small FVG (but not below 1)
        score -= 1
        logger.debug(f"Small FVG ({fvg_size_ticks:.0f} ticks) decreased score")

    # Cap score at 5 (maximum)
    score = min(score, 5)

    # Ensure score is at least 1
    score = max(score, 1)

    logger.debug(
        f"Calculated confidence score: {score} (confluence: {setup.confluence_count}, "
        f"volume_ratio: {volume_ratio:.2f}, fvg_ticks: {fvg_size_ticks:.0f})"
    )

    return score


def score_setup(setup: SilverBulletSetup) -> SilverBulletSetup:
    """Calculate and assign confidence score to a Silver Bullet setup.

    This function calculates the confidence score based on confluence factors
    and assigns it to the setup's confidence field.

    Args:
        setup: Silver Bullet setup to score

    Returns:
        The same setup object with confidence field updated

    Example:
        >>> setup = SilverBulletSetup(...)
        >>> setup.confidence
        0.0
        >>> scored_setup = score_setup(setup)
        >>> scored_setup.confidence
        3
        >>> scored_setup is setup
        True
    """
    # Calculate confidence score
    confidence = calculate_confidence_score(setup)

    # Assign to setup
    setup.confidence = confidence

    logger.info(
        f"Assigned confidence score {confidence} to {setup.direction} setup "
        f"(confluence: {setup.confluence_count}, priority: {setup.priority})"
    )

    return setup

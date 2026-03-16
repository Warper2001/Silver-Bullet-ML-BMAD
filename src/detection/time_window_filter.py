"""Time window filtering for Silver Bullet signals.

This module implements time window validation to ensure Silver Bullet signals
are only generated during high-probability trading periods.
"""

import logging
from datetime import datetime, timezone, timedelta

from src.data.models import SilverBulletSetup, TimeWindow

logger = logging.getLogger(__name__)

# Trading time windows (EST)
LONDON_AM = TimeWindow(
    name="London AM",
    start_hour=3,
    start_minute=0,
    end_hour=4,
    end_minute=0,
    timezone="EST",
)
NY_AM = TimeWindow(
    name="NY AM",
    start_hour=10,
    start_minute=0,
    end_hour=11,
    end_minute=0,
    timezone="EST",
)
NY_PM = TimeWindow(
    name="NY PM",
    start_hour=14,
    start_minute=0,
    end_hour=15,
    end_minute=0,
    timezone="EST",
)

DEFAULT_TRADING_WINDOWS = [LONDON_AM, NY_AM, NY_PM]

# Timezone offsets (in hours)
# EST is UTC-5, but we handle this by converting timestamps
UTC_OFFSET_EST = -5  # EST is UTC-5 (standard time)
UTC_OFFSET_EDT = -4  # EDT is UTC-4 (daylight saving time)


def convert_to_est(timestamp: datetime) -> datetime:
    """Convert a timestamp to EST timezone.

    Args:
        timestamp: Input timestamp (naive or UTC-aware)

    Returns:
        Timestamp in EST timezone (naive, for easy comparison)

    Note:
        This function handles both naive and UTC-aware timestamps.
        For production, you should use pytz or zoneinfo for proper DST handling.
    """
    # If timestamp is naive, assume it's already in the desired timezone
    if timestamp.tzinfo is None:
        return timestamp

    # If timestamp is UTC-aware, convert to EST
    if timestamp.tzinfo == timezone.utc:
        # Approximate EST conversion (UTC-5)
        # Note: This doesn't handle DST perfectly - use pytz in production
        return timestamp + timedelta(hours=UTC_OFFSET_EST)

    # For other timezones, return as-is (naive comparison)
    # In production, you'd convert to EST properly
    return timestamp.replace(tzinfo=None)


def is_within_trading_hours(
    timestamp: datetime,
    windows: list[TimeWindow] | None = None,
) -> tuple[bool, str | None]:
    """Check if a timestamp falls within trading time windows.

    Args:
        timestamp: Timestamp to check
        windows: List of trading windows (uses default if None)

    Returns:
        Tuple of (within_window, window_name)
        - within_window: True if timestamp is within any window
        - window_name: Name of the matching window, or None if not within any window

    Example:
        >>> timestamp = datetime(2026, 3, 16, 3, 30, 0)  # 3:30 AM EST
        >>> within, name = is_within_trading_hours(timestamp)
        >>> within
        True
        >>> name
        'London AM'
    """
    if windows is None:
        windows = DEFAULT_TRADING_WINDOWS

    # Convert to EST for comparison
    est_time = convert_to_est(timestamp)

    # Check each window
    for window in windows:
        # Create datetime objects for window start and end
        window_start = est_time.replace(
            hour=window.start_hour, minute=window.start_minute, second=0, microsecond=0
        )
        window_end = est_time.replace(
            hour=window.end_hour, minute=window.end_minute, second=0, microsecond=0
        )

        # Check if timestamp is within this window
        # Note: Window is [start, end) - includes start, excludes end
        if window_start <= est_time < window_end:
            return True, window.name

    # Not within any window
    return False, None


def check_time_window(
    setup: SilverBulletSetup,
    windows: list[TimeWindow] | None = None,
) -> SilverBulletSetup | None:
    """Check if a Silver Bullet setup occurs within trading time windows.

    Args:
        setup: Silver Bullet setup to check
        windows: List of trading windows (uses default if None)

    Returns:
        Original setup if within trading windows, None if filtered out

    Filtering Logic:
        - If setup.timestamp is within any trading window → return setup
        - If setup.timestamp is outside all windows → return None (filtered)

    Example:
        >>> setup = SilverBulletSetup(...)
        >>> setup.timestamp = datetime(2026, 3, 16, 3, 30, 0)  # 3:30 AM EST
        >>> result = check_time_window(setup)
        >>> result is not None
        True  # Within London AM window

        >>> setup.timestamp = datetime(2026, 3, 16, 0, 0, 0)  # Midnight EST
        >>> result = check_time_window(setup)
        >>> result is None
        True  # Filtered out
    """
    within_window, window_name = is_within_trading_hours(setup.timestamp, windows)

    if within_window:
        logger.debug(
            f"Setup within {window_name} window - signal allowed at {setup.timestamp}"
        )
        return setup
    else:
        logger.debug(
            f"Setup outside trading windows - signal filtered at {setup.timestamp}"
        )
        return None


def filter_setups_by_time_window(
    setups: list[SilverBulletSetup],
    windows: list[TimeWindow] | None = None,
) -> tuple[list[SilverBulletSetup], dict[str, int]]:
    """Filter a list of Silver Bullet setups by time windows.

    Args:
        setups: List of Silver Bullet setups to filter
        windows: List of trading windows (uses default if None)

    Returns:
        Tuple of (filtered_setups, stats)
        - filtered_setups: List of setups within trading windows
        - stats: Dictionary with 'allowed' and 'filtered' counts

    Statistics:
        The stats dictionary tracks:
        - 'allowed': Number of setups within trading windows
        - 'filtered': Number of setups outside trading windows

    Example:
        >>> setups = [setup1, setup2, setup3]
        >>> allowed, stats = filter_setups_by_time_window(setups)
        >>> stats['allowed']
        2
        >>> stats['filtered']
        1
    """
    if windows is None:
        windows = DEFAULT_TRADING_WINDOWS

    allowed_setups = []
    filtered_count = 0

    for setup in setups:
        result = check_time_window(setup, windows)
        if result is not None:
            allowed_setups.append(result)
        else:
            filtered_count += 1

    stats = {
        "allowed": len(allowed_setups),
        "filtered": filtered_count,
    }

    logger.info(
        f"Time window filtering: {stats['allowed']} setups allowed, "
        f"{stats['filtered']} setups filtered"
    )

    return allowed_setups, stats

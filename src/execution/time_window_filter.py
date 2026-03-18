"""Time window filtering for trade execution restrictions.

This module enforces time-based trading restrictions to prevent trades during
high-risk periods including market open, lunch, market close, and after-hours.

Features:
- Trading window definitions (morning/afternoon sessions)
- Blocked period detection (open, lunch, close, pre/post market)
- Weekend blocking
- Time until open/close calculations
- CSV audit trail logging
- Performance: <1ms time check
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TimeWindowResult:
    """Result of time window check.

    Attributes:
        allowed: Whether timestamp is within allowed window
        reason: Human-readable explanation of decision
        window_name: Name of current/nearest window
        time_until_open: Time until next allowed window (if blocked)
        time_until_close: Time until window closes (if allowed)
    """

    allowed: bool
    reason: str
    window_name: str
    time_until_open: Optional[timedelta]
    time_until_close: Optional[timedelta]


class TradingWindows:
    """Standard trading windows for MNQ.

    Time Zones:
        CT (Central Time): Local market time
        UTC: CT + 6 hours (daylight savings in March)

    Windows:
        - PRE_MARKET: Before 8:00 CT (13:00 UTC) - NOT ALLOWED
        - MARKET_OPEN: 8:30-9:00 CT (13:30-14:00 UTC) - NOT ALLOWED
        - MORNING_SESSION: 9:00-11:30 CT (14:00-16:30 UTC) - ALLOWED
        - LUNCH: 11:30-12:30 CT (16:30-17:30 UTC) - NOT ALLOWED
        - AFTERNOON_SESSION: 12:30-15:30 CT (17:30-20:30 UTC) - ALLOWED
        - MARKET_CLOSE: 15:30-16:00 CT (20:30-21:00 UTC) - NOT ALLOWED
        - AFTER_HOURS: After 16:00 CT (21:00 UTC) - NOT ALLOWED

    Example:
        >>> windows = TradingWindows()
        >>> windows.is_trading_allowed(
        ...     datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)
        ... )
        True  # Within afternoon session
        >>> windows.is_trading_allowed(
        ...     datetime(2026, 3, 17, 13, 45, 0, tzinfo=timezone.utc)
        ... )
        False  # During market open volatility
    """

    # All times in UTC (CT + 6 hours during daylight savings)
    PRE_MARKET_END = "13:00"      # 8:00 CT
    MARKET_OPEN_START = "13:30"   # 8:30 CT
    MARKET_OPEN_END = "14:00"     # 9:00 CT
    MORNING_START = "14:00"       # 9:00 CT
    MORNING_END = "16:30"         # 11:30 CT
    LUNCH_START = "16:30"         # 11:30 CT
    LUNCH_END = "17:30"           # 12:30 CT
    AFTERNOON_START = "17:30"     # 12:30 CT
    AFTERNOON_END = "20:30"       # 15:30 CT
    MARKET_CLOSE_START = "20:30"  # 15:30 CT
    MARKET_CLOSE_END = "21:00"    # 16:00 CT

    def _parse_time(self, time_str: str) -> int:
        """Parse HH:MM string to seconds since midnight.

        Args:
            time_str: Time string in HH:MM format

        Returns:
            Seconds since midnight
        """
        hours, minutes = map(int, time_str.split(":"))
        return hours * 3600 + minutes * 60

    def _get_seconds(self, timestamp: datetime) -> int:
        """Get seconds since midnight for timestamp.

        Args:
            timestamp: Datetime to convert

        Returns:
            Seconds since midnight
        """
        return timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second

    def is_trading_allowed(self, timestamp: datetime) -> bool:
        """Check if timestamp is within allowed trading window.

        Args:
            timestamp: Timestamp to check (timezone-aware, UTC)

        Returns:
            True if within allowed window, False otherwise

        Example:
            >>> windows = TradingWindows()
            >>> windows.is_trading_allowed(
            ...     datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)
            ... )
            True
        """
        # Check weekend first
        if timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Get time of day in seconds
        time_seconds = self._get_seconds(timestamp)

        # Define allowed windows (inclusive boundaries)
        morning_start = self._parse_time(self.MORNING_START)
        morning_end = self._parse_time(self.MORNING_END)
        afternoon_start = self._parse_time(self.AFTERNOON_START)
        afternoon_end = self._parse_time(self.AFTERNOON_END)

        # Check if within morning session [14:00, 16:30]
        if morning_start <= time_seconds <= morning_end:
            return True

        # Check if within afternoon session [17:30, 20:30]
        if afternoon_start <= time_seconds <= afternoon_end:
            return True

        return False


class TimeWindowFilter:
    """Filters signals based on time window restrictions.

    Attributes:
        _windows: TradingWindows instance for time checks
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> filter = TimeWindowFilter()
        >>> result = filter.check_time_window(
        ...     timestamp=datetime(2026, 3, 17, 15, 0, 0, tzinfo=timezone.utc)
        ... )
        >>> if result.allowed:
        ...     # Execute trade
        ... else:
        ...     # Skip trade, log reason
    """

    def __init__(self, audit_trail_path: Optional[str] = None) -> None:
        """Initialize time window filter.

        Args:
            audit_trail_path: Path to CSV audit trail file (optional)
        """
        self._windows = TradingWindows()
        self._audit_trail_path = audit_trail_path

        logger.info("TimeWindowFilter initialized")

    def check_time_window(
        self,
        timestamp: datetime
    ) -> TimeWindowResult:
        """Check if timestamp is within allowed trading window.

        Args:
            timestamp: Timestamp to check (timezone-aware, UTC)

        Returns:
            TimeWindowResult with allow/deny decision and reason

        Example:
            >>> result = filter.check_time_window(
            ...     datetime(2026, 3, 17, 13, 45, 0, tzinfo=timezone.utc)
            ... )
            >>> result.allowed
            False
            >>> result.reason
            'Market open volatility period (8:30-9:00 CT)'
        """
        # Check weekend first
        if timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            result = TimeWindowResult(
                allowed=False,
                reason="Weekend trading not allowed",
                window_name="WEEKEND",
                time_until_open=self._calculate_time_until_next_open(timestamp),
                time_until_close=None
            )
            self._log_filter_event(timestamp, result)
            return result

        # Get time of day in seconds
        time_seconds = self._windows._get_seconds(timestamp)

        # Define window boundaries
        pre_market_end = self._windows._parse_time(self._windows.PRE_MARKET_END)
        market_open_start = self._windows._parse_time(
            self._windows.MARKET_OPEN_START
        )
        market_open_end = self._windows._parse_time(self._windows.MARKET_OPEN_END)
        morning_start = self._windows._parse_time(self._windows.MORNING_START)
        morning_end = self._windows._parse_time(self._windows.MORNING_END)
        lunch_start = self._windows._parse_time(self._windows.LUNCH_START)
        lunch_end = self._windows._parse_time(self._windows.LUNCH_END)
        afternoon_start = self._windows._parse_time(self._windows.AFTERNOON_START)
        afternoon_end = self._windows._parse_time(self._windows.AFTERNOON_END)
        market_close_start = self._windows._parse_time(
            self._windows.MARKET_CLOSE_START
        )
        market_close_end = self._windows._parse_time(self._windows.MARKET_CLOSE_END)

        # Pre-market: before 13:00 UTC
        if time_seconds < pre_market_end:
            result = TimeWindowResult(
                allowed=False,
                reason="Pre-market trading not allowed (before 8:00 CT)",
                window_name="PRE_MARKET",
                time_until_open=self._calculate_time_until_next_open(timestamp),
                time_until_close=None
            )
            self._log_filter_event(timestamp, result)
            return result

        # Market open: [13:30, 14:00)
        if market_open_start <= time_seconds < market_open_end:
            result = TimeWindowResult(
                allowed=False,
                reason="Market open volatility period (8:30-9:00 CT)",
                window_name="MARKET_OPEN",
                time_until_open=self._calculate_time_until_next_open(timestamp),
                time_until_close=None
            )
            self._log_filter_event(timestamp, result)
            return result

        # Morning session: [14:00, 16:30]
        if morning_start <= time_seconds <= morning_end:
            time_until_close = self._calculate_time_until_close(
                timestamp,
                morning_end
            )
            result = TimeWindowResult(
                allowed=True,
                reason="Within allowed trading window",
                window_name="MORNING_SESSION",
                time_until_open=None,
                time_until_close=time_until_close
            )
            self._log_filter_event(timestamp, result)
            return result

        # Lunch: [16:30, 17:30)
        if lunch_start <= time_seconds < lunch_end:
            result = TimeWindowResult(
                allowed=False,
                reason="Lunch period low liquidity (11:30-12:30 CT)",
                window_name="LUNCH",
                time_until_open=self._calculate_time_until_next_open(timestamp),
                time_until_close=None
            )
            self._log_filter_event(timestamp, result)
            return result

        # Afternoon session: [17:30, 20:30]
        if afternoon_start <= time_seconds <= afternoon_end:
            time_until_close = self._calculate_time_until_close(
                timestamp,
                afternoon_end
            )
            result = TimeWindowResult(
                allowed=True,
                reason="Within allowed trading window",
                window_name="AFTERNOON_SESSION",
                time_until_open=None,
                time_until_close=time_until_close
            )
            self._log_filter_event(timestamp, result)
            return result

        # Market close: [20:30, 21:00)
        if market_close_start <= time_seconds < market_close_end:
            result = TimeWindowResult(
                allowed=False,
                reason="Market close volatility period (15:30-16:00 CT)",
                window_name="MARKET_CLOSE",
                time_until_open=self._calculate_time_until_next_open(timestamp),
                time_until_close=None
            )
            self._log_filter_event(timestamp, result)
            return result

        # After-hours: 21:00 onwards
        result = TimeWindowResult(
            allowed=False,
            reason="After-hours trading not allowed (after 16:00 CT)",
            window_name="AFTER_HOURS",
            time_until_open=self._calculate_time_until_next_open(timestamp),
            time_until_close=None
        )
        self._log_filter_event(timestamp, result)
        return result

    def _calculate_time_until_close(
        self,
        current_time: datetime,
        window_end_seconds: int
    ) -> timedelta:
        """Calculate time until window closes.

        Args:
            current_time: Current timestamp
            window_end_seconds: Window end time in seconds since midnight

        Returns:
            Timedelta until window close
        """
        current_seconds = self._windows._get_seconds(current_time)
        seconds_until_close = window_end_seconds - current_seconds
        return timedelta(seconds=seconds_until_close)

    def _calculate_time_until_next_open(
        self,
        current_time: datetime
    ) -> timedelta:
        """Calculate time until next allowed trading window opens.

        Args:
            current_time: Current timestamp

        Returns:
            Timedelta until next open window
        """
        # Get current time in seconds
        current_seconds = self._windows._get_seconds(current_time)

        # Define next open windows
        morning_start = self._windows._parse_time(self._windows.MORNING_START)
        afternoon_start = self._windows._parse_time(self._windows.AFTERNOON_START)

        # If before morning session, time until morning
        if current_seconds < morning_start:
            seconds_until_open = morning_start - current_seconds
            return timedelta(seconds=seconds_until_open)

        # If between morning and afternoon, time until afternoon
        if current_seconds < afternoon_start:
            seconds_until_open = afternoon_start - current_seconds
            return timedelta(seconds=seconds_until_open)

        # If after afternoon, time until next morning (next day)
        seconds_until_midnight = 86400 - current_seconds
        seconds_until_open = seconds_until_midnight + morning_start
        return timedelta(seconds=seconds_until_open)

    def _log_filter_event(
        self,
        timestamp: datetime,
        result: TimeWindowResult
    ) -> None:
        """Log time window filter event to CSV audit trail.

        Args:
            timestamp: Timestamp being checked
            result: Filter result
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate current timestamp
        log_timestamp = datetime.now(timezone.utc).isoformat()

        # Check if file exists and has content
        file_exists = audit_path.exists() and audit_path.stat().st_size > 0

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "allowed",
                    "reason",
                    "window_name"
                ])

            # Write log entry
            writer.writerow([
                log_timestamp,
                "FILTER",
                "ALLOWED" if result.allowed else "BLOCKED",
                result.reason,
                result.window_name
            ])

        logger.debug("Time window filter audit trail updated: {}".format(
            audit_path
        ))

    def should_execute_trade(
        self,
        signal_timestamp: datetime
    ) -> bool:
        """Check if trade should be executed based on time window.

        Args:
            signal_timestamp: Signal generation timestamp

        Returns:
            True if within allowed window, False otherwise

        Example:
            >>> if not time_filter.should_execute_trade(signal.timestamp):
            ...     logger.info(f"Trade blocked: {time_filter.get_reason()}")
            ...     return
            >>> # Continue with order submission
        """
        result = self.check_time_window(signal_timestamp)
        return result.allowed

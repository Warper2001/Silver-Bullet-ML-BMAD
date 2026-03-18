"""News event filter for high-impact economic releases.

This module implements a news event filter that prevents trade execution
during high-impact economic news releases. These events cause extreme
volatility and unpredictable price movements.

Features:
- Blackout period management around news events
- Configurable pre/post-event windows
- Trading status checking
- Upcoming events list
- CSV audit trail logging
- Integration with trade execution pipeline
"""

import csv
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NewsEventFilter:
    """Filter trade signals during high-impact news events.

    Attributes:
        _blackout_periods: List of blackout period dictionaries
        _pre_event_window_minutes: Minutes before event to pause
        _post_event_window_minutes: Minutes after event to pause
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> filter_obj = NewsEventFilter(
        ...     pre_event_window_minutes=30,
        ...     post_event_window_minutes=30
        ... )
        >>> filter_obj.add_blackout_period(
        ...     event_name="NFP",
        ...     event_time=datetime(2026, 3, 7, 13, 30, 0),
        ...     event_duration_minutes=0
        ... )
        >>> filter_obj.is_trading_allowed(datetime(2026, 3, 7, 13, 30, 0))
        False
    """

    def __init__(
        self,
        pre_event_window_minutes: int = 30,
        post_event_window_minutes: int = 30,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize news event filter.

        Args:
            pre_event_window_minutes: Minutes before event to pause (default 30)
            post_event_window_minutes: Minutes after event to pause (default 30)
            audit_trail_path: Path to CSV audit trail file (optional)

        Example:
            >>> filter_obj = NewsEventFilter(
            ...     pre_event_window_minutes=30,
            ...     post_event_window_minutes=30
            ... )
        """
        self._blackout_periods = []
        self._pre_event_window_minutes = pre_event_window_minutes
        self._post_event_window_minutes = post_event_window_minutes
        self._audit_trail_path = audit_trail_path

        logger.info(
            "NewsEventFilter initialized: pre={} min, post={} min".format(
                pre_event_window_minutes,
                post_event_window_minutes
            )
        )

    def add_blackout_period(
        self,
        event_name: str,
        event_time: datetime,
        event_duration_minutes: int = 0
    ) -> None:
        """Add a news event blackout period.

        Args:
            event_name: Name of the news event (e.g., "NFP", "FOMC")
            event_time: When the news announcement occurs (UTC)
            event_duration_minutes: Duration of announcement (default 0)

        Example:
            >>> filter_obj.add_blackout_period(
            ...     event_name="NFP",
            ...     event_time=datetime(2026, 3, 7, 13, 30, 0),
            ...     event_duration_minutes=0
            ... )
        """
        # Calculate blackout window
        start_time, end_time = self._calculate_blackout_window(
            event_time,
            event_duration_minutes
        )

        # Add to blackout periods
        blackout = {
            "event_name": event_name,
            "start_time": start_time,
            "end_time": end_time,
            "event_time": event_time
        }

        self._blackout_periods.append(blackout)

        logger.info(
            "Added blackout period: {} from {} to {}".format(
                event_name,
                start_time.isoformat(),
                end_time.isoformat()
            )
        )

        # Log to audit trail
        self._log_audit_event(
            "BLACKOUT_ADD",
            event_name,
            False,
            start_time,
            end_time
        )

    def is_trading_allowed(self, current_time: datetime) -> bool:
        """Check if trading is allowed at current time.

        Args:
            current_time: Current time to check (UTC)

        Returns:
            True if trading allowed, False if during blackout

        Example:
            >>> if not filter_obj.is_trading_allowed(datetime.now(timezone.utc)):
            ...     logger.warning("Trading blocked: News event in progress")
        """
        for blackout in self._blackout_periods:
            if self._is_time_in_blackout(
                current_time,
                blackout["start_time"],
                blackout["end_time"]
            ):
                # Log blackout check
                self._log_audit_event(
                    "BLACKOUT_CHECK",
                    blackout["event_name"],
                    True,
                    blackout["start_time"],
                    blackout["end_time"]
                )
                return False

        # No active blackout
        return True

    def get_blackout_status(self, current_time: datetime) -> dict:
        """Get current blackout status.

        Args:
            current_time: Current time to check (UTC)

        Returns:
            Dictionary with status information:
            - is_blackout: Whether currently in blackout
            - event_name: Name of event causing blackout (if any)
            - blackout_start: When blackout period started
            - blackout_end: When blackout period ends
            - minutes_remaining: Minutes until blackout ends

        Example:
            >>> status = filter_obj.get_blackout_status(datetime.now(timezone.utc))
            >>> if status['is_blackout']:
            ...     print("Blocked by: {}".format(status['event_name']))
        """
        for blackout in self._blackout_periods:
            if self._is_time_in_blackout(
                current_time,
                blackout["start_time"],
                blackout["end_time"]
            ):
                # Calculate minutes remaining
                remaining = blackout["end_time"] - current_time
                minutes_remaining = int(remaining.total_seconds() / 60)

                return {
                    "is_blackout": True,
                    "event_name": blackout["event_name"],
                    "blackout_start": blackout["start_time"],
                    "blackout_end": blackout["end_time"],
                    "minutes_remaining": minutes_remaining
                }

        # No active blackout
        return {
            "is_blackout": False,
            "event_name": None,
            "blackout_start": None,
            "blackout_end": None,
            "minutes_remaining": None
        }

    def get_upcoming_events(
        self,
        current_time: datetime,
        hours_ahead: int = 24
    ) -> list:
        """Get list of upcoming news events.

        Args:
            current_time: Current time (UTC)
            hours_ahead: How many hours ahead to look (default 24)

        Returns:
            List of upcoming events with name and time

        Example:
            >>> events = filter_obj.get_upcoming_events(datetime.now(timezone.utc))
            >>> for event in events:
            ...     print("{} at {}".format(event['name'], event['time']))
        """
        cutoff_time = current_time + timedelta(hours=hours_ahead)

        upcoming = []
        for blackout in self._blackout_periods:
            # Include events within time window and haven't ended yet
            if (blackout["event_time"] <= cutoff_time and
                    blackout["end_time"] > current_time):
                upcoming.append({
                    "event_name": blackout["event_name"],
                    "time": blackout["event_time"],
                    "start_time": blackout["start_time"],
                    "end_time": blackout["end_time"]
                })

        # Sort by event time
        upcoming.sort(key=lambda x: x["time"])

        return upcoming

    def remove_expired_blackouts(self, current_time: datetime) -> None:
        """Remove blackout periods that have already passed.

        Args:
            current_time: Current time (UTC)

        Example:
            >>> filter_obj.remove_expired_blackouts(datetime.now(timezone.utc))
        """
        original_count = len(self._blackout_periods)

        # Keep only blackouts that haven't ended
        self._blackout_periods = [
            b for b in self._blackout_periods
            if b["end_time"] > current_time
        ]

        removed_count = original_count - len(self._blackout_periods)

        if removed_count > 0:
            logger.info(
                "Removed {} expired blackout period(s)".format(removed_count)
            )

    def _is_time_in_blackout(
        self,
        current_time: datetime,
        blackout_start: datetime,
        blackout_end: datetime
    ) -> bool:
        """Check if time falls within blackout period.

        Args:
            current_time: Time to check
            blackout_start: Blackout period start
            blackout_end: Blackout period end

        Returns:
            True if in blackout, False otherwise
        """
        return blackout_start <= current_time < blackout_end

    def _calculate_blackout_window(
        self,
        event_time: datetime,
        event_duration_minutes: int
    ) -> tuple[datetime, datetime]:
        """Calculate blackout window around event.

        Args:
            event_time: When event occurs
            event_duration_minutes: Duration of event

        Returns:
            Tuple of (blackout_start, blackout_end)

        Example:
            >>> start, end = filter_obj._calculate_blackout_window(
            ...     datetime(2026, 3, 7, 13, 30, 0),
            ...     0
            ... )
            >>> # 30 min before + event + 30 min after
            >>> start == datetime(2026, 3, 7, 13, 0, 0)
            >>> end == datetime(2026, 3, 7, 14, 0, 0)
        """
        # Calculate start (pre-event window)
        start_time = event_time - timedelta(
            minutes=self._pre_event_window_minutes
        )

        # Calculate end (event duration + post-event window)
        end_time = event_time + timedelta(
            minutes=event_duration_minutes + self._post_event_window_minutes
        )

        return start_time, end_time

    def _log_audit_event(
        self,
        event_type: str,
        event_name: str,
        is_blackout: bool,
        blackout_start: Optional[datetime],
        blackout_end: Optional[datetime]
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (BLACKOUT_ADD, BLACKOUT_CHECK, etc.)
            event_name: Name of news event
            is_blackout: Whether trading is blocked
            blackout_start: Blackout period start
            blackout_end: Blackout period end
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Calculate minutes remaining
        minutes_remaining = None
        if is_blackout and blackout_end:
            remaining = blackout_end - datetime.now(timezone.utc)
            if remaining.total_seconds() > 0:
                minutes_remaining = int(remaining.total_seconds() / 60)

        # Check if file exists
        file_exists = (
            audit_path.exists() and
            audit_path.stat().st_size > 0
        )

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "event_name",
                    "is_blackout",
                    "blackout_start",
                    "blackout_end",
                    "minutes_remaining"
                ])

            # Format times
            start_str = blackout_start.isoformat() if blackout_start else ""
            end_str = blackout_end.isoformat() if blackout_end else ""

            # Write event
            writer.writerow([
                timestamp,
                event_type,
                event_name,
                is_blackout,
                start_str,
                end_str,
                minutes_remaining or ""
            ])

        logger.debug("News event audit logged: {}".format(event_type))

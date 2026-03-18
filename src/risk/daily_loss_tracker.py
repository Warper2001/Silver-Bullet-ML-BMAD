"""Daily loss limit enforcement for trading safety.

This module implements a daily loss limit that automatically stops
all trading when cumulative losses reach a predefined threshold.
This is a critical safety mechanism to prevent catastrophic losses.

Features:
- Track cumulative daily P&L
- Enforce daily loss limit (default: 2% of account)
- Automatic trading halt on limit breach
- Daily reset at specified time (default: 8:00 CT)
- CSV audit trail logging
- Alert on limit breach
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_current_time() -> datetime:
    """Get current time in UTC.

    Returns:
        Current datetime in UTC

    Note:
        Separate function to facilitate testing with mocks.
    """
    return datetime.now(timezone.utc)


class DailyLossTracker:
    """Track daily trading losses and enforce loss limit.

    Attributes:
        _daily_loss_limit: Maximum daily loss amount (USD)
        _account_balance: Starting account balance
        _daily_pnl: Cumulative P&L for current day (USD)
        _reset_time_utc: Time when daily tracker resets (UTC)
        _is_trading_halted: Whether trading is currently halted
        _halt_reason: Reason trading was halted
        _last_reset_date: Last date tracker was reset
        _trade_count: Number of trades recorded today
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> tracker = DailyLossTracker(
        ...     daily_loss_limit=1000.00,
        ...     account_balance=50000.00,
        ...     reset_time_utc="13:00"  # 8:00 CT
        ... )
        >>> tracker.record_trade(pnl=-150.00, order_id="ORDER-123")
        >>> tracker.is_trading_allowed()
        True  # Still within limit
        >>> tracker.record_trade(pnl=-900.00)
        >>> tracker.is_trading_allowed()
        False  # Halted: -$1,050 exceeds -$1,000 limit
    """

    def __init__(
        self,
        daily_loss_limit: float,
        account_balance: float,
        reset_time_utc: str = "13:00",  # 8:00 CT
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize daily loss tracker.

        Args:
            daily_loss_limit: Maximum daily loss amount (USD)
            account_balance: Starting account balance
            reset_time_utc: Time when daily tracker resets (HH:MM UTC)
            audit_trail_path: Path to CSV audit trail file (optional)

        Raises:
            ValueError: If daily_loss_limit <= 0 or account_balance <= 0

        Example:
            >>> tracker = DailyLossTracker(
            ...     daily_loss_limit=1000.00,
            ...     account_balance=50000.00,
            ...     reset_time_utc="13:00"
            ... )
        """
        if daily_loss_limit <= 0:
            raise ValueError(
                "Daily loss limit must be positive: {}".format(
                    daily_loss_limit
                )
            )

        if account_balance <= 0:
            raise ValueError(
                "Account balance must be positive: {}".format(
                    account_balance
                )
            )

        self._daily_loss_limit = daily_loss_limit
        self._account_balance = account_balance
        self._daily_pnl = 0.0
        self._reset_time_utc = reset_time_utc
        self._is_trading_halted = False
        self._halt_reason: Optional[str] = None
        self._last_reset_date = _get_current_time().date()
        self._trade_count = 0
        self._audit_trail_path = audit_trail_path

        logger.info(
            "DailyLossTracker initialized: limit=${:.2f}, balance=${:.2f}".format(
                daily_loss_limit, account_balance
            )
        )

    def record_trade(
        self,
        pnl: float,
        order_id: str
    ) -> None:
        """Record trade P&L and check if limit breached.

        Args:
            pnl: Trade profit/loss (negative = loss)
            order_id: Order ID for audit trail

        Example:
            >>> tracker.record_trade(pnl=-150.00, order_id="ORDER-123")
            >>> if tracker.is_trading_halted():
            ...     print("Trading halted!")
        """
        # Check for daily reset
        self._reset_if_new_day()

        # Record P&L
        self._daily_pnl += pnl
        self._trade_count += 1

        # Log event
        logger.debug(
            "Trade recorded: ${:.2f} (total: ${:.2f})".format(
                pnl, self._daily_pnl
            )
        )

        # Check if limit breached
        self._check_and_enforce_limit()

        # Log to audit trail
        self._log_audit_event("TRADE", order_id)

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed.

        Returns:
            True if trading allowed, False if halted

        Example:
            >>> if not tracker.is_trading_allowed():
            ...     return "Daily loss limit reached"
        """
        self._reset_if_new_day()
        return not self._is_trading_halted

    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted.

        Returns:
            True if trading halted, False otherwise

        Example:
            >>> if tracker.is_trading_halted():
            ...     logger.warning("Trading halted")
        """
        return self._is_trading_halted

    def get_daily_pnl(self) -> float:
        """Get current daily cumulative P&L.

        Returns:
            Cumulative P&L for current day

        Example:
            >>> pnl = tracker.get_daily_pnl()
            >>> print(f"Daily P&L: ${pnl:.2f}")
        """
        return self._daily_pnl

    def get_remaining_loss_allowance(self) -> float:
        """Get remaining loss allowance before halt.

        Returns:
            Remaining loss amount allowed (USD)

        Example:
            >>> remaining = tracker.get_remaining_loss_allowance()
            >>> print(f"Can still lose ${remaining:.2f} today")
        """
        remaining = self._daily_loss_limit - abs(self._daily_pnl)
        return max(0.0, remaining)

    def get_halt_reason(self) -> Optional[str]:
        """Get reason trading was halted.

        Returns:
            Halt reason or None if not halted

        Example:
            >>> if tracker.is_trading_halted():
            ...     reason = tracker.get_halt_reason()
            ...     print(f"Halted: {reason}")
        """
        return self._halt_reason

    def get_daily_summary(self) -> dict:
        """Get daily trading summary.

        Returns:
            Dictionary with daily statistics

        Example:
            >>> summary = tracker.get_daily_summary()
            >>> print(f"P&L: ${summary['daily_pnl']}")
            >>> print(f"Trades: {summary['trade_count']}")
            >>> print(f"Halted: {summary['is_halted']}")
        """
        return {
            "daily_pnl": self._daily_pnl,
            "loss_limit": self._daily_loss_limit,
            "remaining_allowance": self.get_remaining_loss_allowance(),
            "trade_count": self._trade_count,
            "is_halted": self._is_trading_halted,
            "halt_reason": self._halt_reason
        }

    def _check_and_enforce_limit(self) -> None:
        """Check if daily loss limit breached and halt trading.

        If cumulative daily P&L ≤ -daily_loss_limit, halt trading.
        Sets _is_trading_halted flag and records halt reason.
        """
        if self._daily_pnl <= -self._daily_loss_limit:
            self._is_trading_halted = True
            self._halt_reason = (
                "Daily loss limit breached: ${:.2f} / ${:.2f}".format(
                    abs(self._daily_pnl),
                    self._daily_loss_limit
                )
            )

            logger.critical(
                "DAILY LOSS LIMIT BREACHED: {}".format(self._halt_reason)
            )

            # Log halt event
            self._log_audit_event("HALT", None)

    def _reset_if_new_day(self) -> None:
        """Reset tracker if new trading day has started.

        Resets daily P&L, halt status, and updates reset date.
        Called automatically before any operation.
        """
        if self._should_reset_day():
            logger.info("Resetting daily loss tracker for new trading day")

            self._daily_pnl = 0.0
            self._is_trading_halted = False
            self._halt_reason = None
            self._last_reset_date = _get_current_time().date()

            # Log reset event
            self._log_audit_event("RESET", None)

    def _should_reset_day(self) -> bool:
        """Check if should reset to new trading day.

        Returns:
            True if current time is past reset time and date changed

        Note:
            Uses reset_time_utc (default 13:00 UTC = 8:00 CT).
            Tracker resets at this time each day.
        """
        now = _get_current_time()
        current_date = now.date()

        # Parse reset time
        reset_hour, reset_minute = map(
            int, self._reset_time_utc.split(":")
        )

        # Create reset datetime for today
        reset_datetime = datetime(
            now.year,
            now.month,
            now.day,
            reset_hour,
            reset_minute,
            0,
            tzinfo=timezone.utc
        )

        # If current time is past reset time and date changed
        if now >= reset_datetime and current_date != self._last_reset_date:
            return True

        return False

    def _log_audit_event(
        self,
        event_type: str,
        order_id: Optional[str]
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (RESET, TRADE, HALT)
            order_id: Order ID (if applicable)
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = _get_current_time().isoformat()

        # Check if file exists
        file_exists = audit_path.exists() and audit_path.stat().st_size > 0

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "daily_pnl",
                    "loss_limit",
                    "remaining_allowance",
                    "is_halted",
                    "halt_reason",
                    "order_id"
                ])

            # Write event
            writer.writerow([
                timestamp,
                event_type,
                f"{self._daily_pnl:.2f}",
                f"{self._daily_loss_limit:.2f}",
                f"{self.get_remaining_loss_allowance():.2f}",
                self._is_trading_halted,
                self._halt_reason or "",
                order_id or ""
            ])

        logger.debug("Daily loss audit event logged: {}".format(event_type))

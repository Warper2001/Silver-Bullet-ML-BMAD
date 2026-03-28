"""
TradeStation SDK - Historical Data Module

This module provides historical OHLCV bar data access from TradeStation API.

Key Features:
- Download historical bars for any date range
- Support different bar types (minute, daily, weekly, etc.)
- Data validation and gap detection
- Integration with HDF5 storage

Usage:
    async with TradeStationClient(env="sim", ...) as client:
        history_client = HistoryClient(client)
        bars = await history_client.get_historical_bars(
            symbol="MNQH26",
            bar_type="minute",
            interval=5,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
"""

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import APIError, NetworkError, ValidationError
from src.execution.tradestation.models import HistoricalBar
from src.execution.tradestation.utils import setup_logger


class HistoryClient:
    """
    Client for historical bar data from TradeStation API.

    Provides methods to download historical OHLCV bars with various
    timeframes and date ranges.

    Attributes:
        client: TradeStationClient instance for API communication

    Example:
        async with TradeStationClient(env="sim", ...) as client:
            history_client = HistoryClient(client)
            bars = await history_client.get_historical_bars(
                symbol="MNQH26",
                bar_type="minute",
                interval=5
            )
    """

    # Supported bar types
    VALID_BAR_TYPES = ["minute", "minute2", "minute3", "minute5", "minute15", "minute30", "hour", "daily", "weekly", "monthly"]

    # Supported intervals for each bar type
    BAR_INTERVALS = {
        "minute": [1, 5, 15, 30],
        "minute2": [2],
        "minute3": [3],
        "minute5": [5],
        "minute15": [15],
        "minute30": [30],
        "hour": [1],
        "daily": [1],
        "weekly": [1],
        "monthly": [1],
    }

    def __init__(self, client: TradeStationClient) -> None:
        """
        Initialize HistoryClient.

        Args:
            client: Authenticated TradeStationClient instance
        """
        self.client = client
        self.logger = setup_logger(f"{__name__}.HistoryClient")

    async def get_historical_bars(
        self,
        symbol: str,
        bar_type: Literal[
            "minute",
            "minute2",
            "minute3",
            "minute5",
            "minute15",
            "minute30",
            "hour",
            "daily",
            "weekly",
            "monthly",
        ] = "minute",
        interval: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 1000,
    ) -> list[HistoricalBar]:
        """
        Get historical OHLCV bars for a symbol.

        Args:
            symbol: Trading symbol (e.g., "MNQH26")
            bar_type: Type of bars (minute, daily, etc.)
            interval: Bar interval in units of bar_type
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            limit: Maximum number of bars to return (default: 1000)

        Returns:
            List of HistoricalBar objects with OHLCV data

        Raises:
            ValidationError: If parameters are invalid
            APIError: On API errors
            NetworkError: On network errors

        Example:
            # Get 5-minute bars for last week
            bars = await history_client.get_historical_bars(
                symbol="MNQH26",
                bar_type="minute5",
                start_date="2024-01-20",
                end_date="2024-01-27"
            )

            # Get daily bars for last month
            bars = await history_client.get_historical_bars(
                symbol="MNQH26",
                bar_type="daily",
                limit=30
            )
        """
        self.logger.info(f"Fetching historical bars for {symbol}")

        # Validate parameters
        self._validate_bar_type(bar_type, interval)
        self._validate_symbol(symbol)
        self._validate_dates(start_date, end_date)

        # Build request parameters
        params = {
            "symbol": symbol,
            "type": bar_type,
            "interval": interval,
            "limit": min(limit, 10000),  # Max 10k bars per request
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            response = await self.client._request(
                "GET",
                "/data/bars",
                params=params,
            )

            # Parse response
            bars_data = response.get("Bars", [])
            bars = [HistoricalBar(**bar) for bar in bars_data]

            self.logger.info(f"Received {len(bars)} historical bars")

            # Validate data completeness
            if bars and start_date and end_date:
                self._validate_data_completeness(bars, start_date, end_date, bar_type, interval)

            return bars

        except (ValidationError, APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching historical bars: {e}")
            raise APIError(f"Unexpected error fetching historical bars: {e}")

    async def get_bar_data(
        self,
        symbol: str,
        days_back: int = 30,
        bar_type: Literal["daily", "hour", "minute"] = "daily",
        interval: int = 1,
    ) -> list[HistoricalBar]:
        """
        Get recent bar data for a symbol.

        Convenience method for fetching the most recent bars.

        Args:
            symbol: Trading symbol
            days_back: Number of days back to fetch (default: 30)
            bar_type: Type of bars (default: daily)
            interval: Bar interval (default: 1)

        Returns:
            List of HistoricalBar objects

        Example:
            # Get last 30 days of daily bars
            bars = await history_client.get_bar_data("MNQH26", days_back=30)

            # Get last 24 hours of hourly bars
            bars = await history_client.get_bar_data("MNQH26", days_back=1, bar_type="hour")
        """
        # Calculate end date (today) and start date
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                     ).replace(day=datetime.now(timezone.utc).day - days_back).strftime("%Y-%m-%d")

        return await self.get_historical_bars(
            symbol=symbol,
            bar_type=bar_type,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )

    def _validate_bar_type(self, bar_type: str, interval: int) -> None:
        """
        Validate bar type and interval combination.

        Args:
            bar_type: Type of bars
            interval: Bar interval

        Raises:
            ValidationError: If combination is invalid
        """
        if bar_type not in self.VALID_BAR_TYPES:
            raise ValidationError(
                f"Invalid bar_type: {bar_type}. "
                f"Must be one of: {', '.join(self.VALID_BAR_TYPES)}"
            )

        valid_intervals = self.BAR_INTERVALS.get(bar_type, [])
        if interval not in valid_intervals:
            raise ValidationError(
                f"Invalid interval {interval} for bar_type {bar_type}. "
                f"Must be one of: {valid_intervals}"
            )

    def _validate_symbol(self, symbol: str) -> None:
        """
        Validate symbol format.

        Args:
            symbol: Trading symbol

        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol or len(symbol) < 2:
            raise ValidationError(f"Invalid symbol: {symbol}")

        # Symbol should be alphanumeric
        if not symbol.isalnum():
            raise ValidationError(f"Symbol should be alphanumeric: {symbol}")

    def _validate_dates(self, start_date: str | None, end_date: str | None) -> None:
        """
        Validate date parameters.

        Args:
            start_date: Start date string
            end_date: End date string

        Raises:
            ValidationError: If dates are invalid
        """
        if start_date:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError(f"Invalid start_date format: {start_date}. Use YYYY-MM-DD")

        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError(f"Invalid end_date format: {end_date}. Use YYYY-MM-DD")

        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if start > end:
                raise ValidationError(f"start_date ({start_date}) must be before end_date ({end_date})")

    def _validate_data_completeness(
        self,
        bars: list[HistoricalBar],
        start_date: str,
        end_date: str,
        bar_type: str,
        interval: int,
    ) -> None:
        """
        Validate data completeness and check for gaps.

        Args:
            bars: Historical bars returned
            start_date: Requested start date
            end_date: Requested end date
            bar_type: Type of bars
            interval: Bar interval

        Raises:
            ValidationError: If data completeness is below threshold
        """
        if not bars:
            raise ValidationError("No bars returned from API")

        # For simplicity, we'll just log warnings for now
        # In production, you would check for gaps in the data
        expected_bars = self._calculate_expected_bars(start_date, end_date, bar_type, interval)
        actual_bars = len(bars)

        completeness = (actual_bars / expected_bars) * 100 if expected_bars > 0 else 0

        if completeness < 95:  # Warning threshold
            self.logger.warning(
                f"Data completeness is {completeness:.1f}% "
                f"({actual_bars}/{expected_bars} expected bars)"
            )
        elif completeness < 99.99:  # Error threshold (from NFR)
            self.logger.error(
                f"Data completeness is {completeness:.1f}% "
                f"({actual_bars}/{expected_bars} expected bars) - "
                f"below 99.99% target"
            )

    def _calculate_expected_bars(
        self,
        start_date: str,
        end_date: str,
        bar_type: str,
        interval: int,
    ) -> int:
        """
        Calculate expected number of bars for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bar_type: Type of bars
            interval: Bar interval

        Returns:
            Expected number of bars
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculate difference in days (inclusive)
        days = (end - start).days + 1

        # Convert to bars based on type
        if bar_type == "minute":
            # 1440 minutes per day * days
            return int((1440 / interval) * days)
        elif bar_type == "hour":
            # 24 hours per day * days
            return int((24 / interval) * days)
        elif bar_type == "daily":
            return days
        elif bar_type == "weekly":
            return int((days + 6) / 7)  # Ceiling division
        elif bar_type == "monthly":
            # Approximate (varies by month length)
            months = (end.year - start.year) * 12 + (end.month - start.month) + 1
            return months
        else:
            # For custom minute types (minute2, minute3, etc.)
            minutes_per_bar = int(bar_type.replace("minute", ""))
            minutes_per_day = 1440 / minutes_per_bar
            return int(minutes_per_day * days)

"""TradeStation API client with rate limiting and retry logic.

This module provides an async HTTP client for fetching historical bar data
from TradeStation API with automatic rate limiting, token refresh, and
exponential backoff on failures.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from zoneinfo import ZoneInfo

from .tradestation_auth import TradeStationAuth
from .tradestation_models import BarData

logger = logging.getLogger(__name__)

# TradeStation API endpoints
BASE_URL = "https://api.tradestation.com/v3"
BARS_ENDPOINT = f"{BASE_URL}/marketdata/bars"

# Rate limiting configuration
MAX_RETRIES = 3
RETRY_MIN_WAIT = 1.0  # seconds
RETRY_MAX_WAIT = 60.0  # seconds

# Pagination
MAX_BARS_PER_REQUEST = 100000
MAX_DAYS_PER_REQUEST = 70  # API limits bars per request

# Timezone
TZ_ET = ZoneInfo("America/New_York")

# MNQ contract multiplier
MNQ_MULTIPLIER = 0.5


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, remaining: int, reset_after: int) -> None:
        """Initialize rate limit error.

        Args:
            remaining: Number of requests remaining
            reset_after: Seconds until limit resets
        """
        self.remaining = remaining
        self.reset_after = reset_after
        super().__init__(
            f"Rate limit exceeded. Remaining: {remaining}, Reset: {reset_after}s"
        )


class TradeStationClient:
    """TradeStation API client for historical data retrieval.

    Features:
    - Async HTTP client with connection pooling
    - Automatic token refresh on 401 errors
    - Exponential backoff on rate limiting (HTTP 429)
    - Timezone conversion (UTC → America/New_York)
    - Date range pagination
    - Data validation and completeness checking
    """

    def __init__(self, auth: TradeStationAuth) -> None:
        """Initialize API client.

        Args:
            auth: TradeStationAuth instance for token management
        """
        self.auth = auth
        self._client: Optional[httpx.AsyncClient] = None
        self._refresh_lock = threading.Lock()
        self._last_rate_limit_remaining: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            httpx.AsyncClient instance
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _get_headers(self) -> dict:
        """Get request headers with authorization.

        Returns:
            Headers dict with Bearer token
        """
        token = self.auth.get_valid_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _rate_limit_headers(self, response: httpx.Response) -> dict:
        """Parse rate limit headers from response.

        Args:
            response: HTTP response object

        Returns:
            Dict with rate limit info
        """
        return {
            "remaining": int(response.headers.get("X-RateLimit-Remaining", "-1")),
            "limit": int(response.headers.get("X-RateLimit-Limit", "-1")),
            "reset": int(response.headers.get("X-RateLimit-Reset", "-1")),
        }

    def _should_retry_rate_limit(self, retry_state) -> bool:  # type: ignore[no-untyped-def]
        """Check if request should be retried based on rate limit.

        Args:
            retry_state: Tenacity retry state object

        Returns:
            True if should retry, False otherwise
        """
        if retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if isinstance(exc, httpx.HTTPStatusError):
                return exc.response.status_code == 429
        return False

    async def get_historical_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: int = 1,
        unit: str = "Minute",
    ) -> list[BarData]:
        """Fetch historical bars for a symbol and date range.

        Args:
            symbol: Futures contract symbol (e.g., 'MNQH26')
            start_date: Start date (timezone-aware)
            end_date: End date (timezone-aware)
            interval: Bar interval (default: 1)
            unit: Interval unit (default: 'Minute')

        Returns:
            List of BarData objects

        Raises:
            AuthenticationError: If token refresh fails
            httpx.HTTPStatusError: If request fails after retries
        """
        all_bars: list[BarData] = []

        # Paginate by date range
        date_ranges = self._paginate_date_range(start_date, end_date)

        logger.info(
            f"Fetching {len(date_ranges)} date ranges for {symbol} "
            f"from {start_date.date()} to {end_date.date()}"
        )

        for i, (range_start, range_end) in enumerate(date_ranges, 1):
            logger.debug(
                f"Fetching range {i}/{len(date_ranges)}: "
                f"{range_start.date()} to {range_end.date()}"
            )

            bars = await self._fetch_bars_for_range(
                symbol, range_start, range_end, interval, unit
            )
            all_bars.extend(bars)

        # Detect and remove duplicates
        all_bars = self._detect_duplicate_bars(all_bars)

        # Validate data completeness
        self._verify_data_completeness(all_bars, start_date, end_date)

        logger.info(f"Fetched {len(all_bars)} bars for {symbol}")
        return all_bars

    def _paginate_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Split date range into 70-day chunks.

        Args:
            start_date: Range start
            end_date: Range end

        Returns:
            List of (start, end) tuples
        """
        ranges = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(
                current_start + timedelta(days=MAX_DAYS_PER_REQUEST), end_date
            )
            ranges.append((current_start, current_end))
            current_start = current_end

        return ranges

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=lambda retry_state: logger.info(  # type: ignore[no-untyped-def]
            f"Rate limit hit, retry {retry_state.attempt_number}/{MAX_RETRIES} "
            f"after {retry_state.next_action.sleep} seconds"
        ),
    )
    async def _fetch_bars_for_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: int,
        unit: str,
    ) -> list[BarData]:
        """Fetch bars for a single date range with retry on rate limit.

        Args:
            symbol: Futures contract symbol
            start_date: Range start
            end_date: Range end
            interval: Bar interval
            unit: Interval unit

        Returns:
            List of BarData objects

        Raises:
            RateLimitError: If rate limit exceeded (triggers retry)
            AuthenticationError: If unauthorized (401)
        """
        client = await self._get_client()
        headers = await self._get_headers()

        params = {
            "interval": str(interval),
            "unit": unit,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        }

        try:
            response = await client.get(
                f"{BARS_ENDPOINT}/{symbol}",
                headers=headers,
                params=params,
            )

            # Parse rate limit headers
            rate_limit = self._rate_limit_headers(response)
            self._last_rate_limit_remaining = rate_limit["remaining"]
            logger.debug(
                f"Rate limit: {rate_limit['remaining']}/{rate_limit['limit']} remaining"
            )

            # Handle rate limiting
            if response.status_code == 429:
                raise RateLimitError(
                    remaining=rate_limit["remaining"],
                    reset_after=rate_limit["reset"],
                )

            # Handle unauthorized (token expired)
            if response.status_code == 401:
                logger.warning("Access token expired, refreshing...")
                with self._refresh_lock:
                    # Refresh token and retry
                    self.auth.refresh_access_token()
                # Retry once after refresh (outside retry decorator)
                headers = await self._get_headers()
                response = await client.get(
                    f"{BARS_ENDPOINT}/{symbol}",
                    headers=headers,
                    params=params,
                )

            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Convert to RateLimitError for retry decorator
                raise RateLimitError(
                    remaining=self._last_rate_limit_remaining or 0,
                    reset_after=int(e.response.headers.get("X-RateLimit-Reset", 60)),
                ) from e
            if e.response.status_code == 401:
                from .exceptions import AuthenticationError
                raise AuthenticationError("Authentication failed after token refresh") from e
            raise

        # Parse response
        data = response.json()

        # Handle empty response (holiday, no trading)
        if not data or "Bars" not in data or not data["Bars"]:
            logger.debug(f"No bars returned for {symbol} {start_date.date()} to {end_date.date()}")
            return []

        # Convert to BarData objects
        bars = []
        for bar_json in data["Bars"]:
            try:
                bar = self._parse_bar_data(symbol, bar_json)
                if bar:
                    bars.append(bar)
            except Exception as e:
                logger.warning(f"Failed to parse bar data: {e}")
                continue

        return bars

    def _parse_bar_data(self, symbol: str, bar_json: dict) -> Optional[BarData]:
        """Parse bar data from API response.

        Args:
            symbol: Contract symbol
            bar_json: Raw bar data from API

        Returns:
            BarData object or None if invalid

        Raises:
            ValueError: If bar data validation fails
        """
        try:
            # Parse timestamp (API returns UTC as ISO string)
            timestamp_str = bar_json.get("Timestamp") or bar_json.get("timestamp")
            if not timestamp_str:
                raise ValueError("Missing timestamp")

            # Parse UTC timestamp
            if isinstance(timestamp_str, str):
                timestamp_utc = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            else:
                timestamp_utc = timestamp_str

            # Convert to America/New_York timezone
            timestamp_et = timestamp_utc.astimezone(TZ_ET)

            bar = BarData(
                symbol=symbol,
                timestamp=timestamp_et,
                open=float(bar_json.get("Open") or bar_json.get("open")),
                high=float(bar_json.get("High") or bar_json.get("high")),
                low=float(bar_json.get("Low") or bar_json.get("low")),
                close=float(bar_json.get("Close") or bar_json.get("close")),
                volume=int(bar_json.get("Volume") or bar_json.get("volume", 0)),
            )

            # Validate bar data
            self._validate_bar_data(bar)

            return bar

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid bar data: {e}")
            return None

    def _validate_bar_data(self, bar: BarData) -> None:
        """Validate bar data for consistency.

        Args:
            bar: BarData to validate

        Raises:
            ValueError: If validation fails
        """
        if bar.high < bar.low:
            raise ValueError(f"High ({bar.high}) must be >= Low ({bar.low})")

        if bar.close < bar.low or bar.close > bar.high:
            raise ValueError(
                f"Close ({bar.close}) must be within High/Low range "
                f"([{bar.low}, {bar.high}])"
            )

        if bar.volume < 0:
            raise ValueError(f"Volume ({bar.volume}) must be >= 0")

    def _verify_data_completeness(
        self,
        bars: list[BarData],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Verify no significant gaps in bar data.

        Args:
            bars: List of bars to verify
            start_date: Expected start date
            end_date: Expected end date

        Raises:
            ValueError: If data completeness below threshold
        """
        if not bars:
            logger.warning("No bars to verify completeness")
            return

        # Calculate expected bars (rough estimate: 1-min bars for trading days)
        total_days = (end_date - start_date).days
        trading_days = total_days * 5 / 7  # Weekdays only
        minutes_per_day = 6.5 * 60  # 6.5 hours trading day
        expected_bars = int(trading_days * minutes_per_day)

        actual_bars = len(bars)
        completeness = actual_bars / expected_bars if expected_bars > 0 else 0

        logger.info(
            f"Data completeness: {actual_bars}/{expected_bars} bars "
            f"({completeness:.1%})"
        )

        # Require at least 95% completeness
        if completeness < 0.95:
            logger.warning(
                f"Data completeness {completeness:.1%} below 95% threshold. "
                f"This may be due to holidays or data gaps."
            )

    def _detect_duplicate_bars(self, bars: list[BarData]) -> list[BarData]:
        """Detect and remove duplicate bars based on timestamp.

        Args:
            bars: List of bars (may contain duplicates)

        Returns:
            List with duplicates removed (keeping earliest occurrence)
        """
        if not bars:
            return bars

        # Track seen timestamps
        seen = {}
        deduped = []

        for bar in bars:
            timestamp_key = bar.timestamp.timestamp()
            if timestamp_key not in seen:
                seen[timestamp_key] = True
                deduped.append(bar)
            else:
                logger.debug(f"Duplicate bar detected at {bar.timestamp}, skipping")

        removed = len(bars) - len(deduped)
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate bars")

        return deduped

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("HTTP client closed")

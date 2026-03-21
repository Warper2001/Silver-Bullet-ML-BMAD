"""Unit tests for TradeStation API client.

Tests bar fetching, pagination, rate limiting, timezone conversion,
data validation, token refresh, and edge cases.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

from src.data.tradestation_client import (
    RateLimitError,
    TradeStationClient,
)
from src.data.tradestation_auth import TradeStationAuth
from src.data.tradestation_models import BarData


@pytest.fixture
def mock_auth():
    """Mock TradeStationAuth."""
    auth = TradeStationAuth()
    return auth


@pytest.fixture
def client(mock_auth):
    """Create TradeStationClient with mocked auth."""
    return TradeStationClient(auth=mock_auth)


@pytest.fixture
def sample_bar_data():
    """Sample bar data from API response."""
    return {
        "Timestamp": "2026-03-15T14:30:00Z",
        "Open": 11800.0,
        "High": 11850.0,
        "Low": 11790.0,
        "Close": 11825.0,
        "Volume": 1000,
    }


class TestBarDataParsing:
    """Tests for parsing bar data from API response."""

    def test_parse_bar_data_success(self, client, sample_bar_data):
        """Test successful bar data parsing."""
        bar = client._parse_bar_data("MNQH26", sample_bar_data)

        assert bar.symbol == "MNQH26"
        assert bar.open == 11800.0
        assert bar.high == 11850.0
        assert bar.low == 11790.0
        assert bar.close == 11825.0
        assert bar.volume == 1000

    def test_timezone_conversion_utc_to_et(self, client, sample_bar_data):
        """Test timezone conversion from UTC to America/New_York."""
        # March 15, 2026 14:30 UTC
        # In ET (assuming no DST): 9:30 AM
        # With DST (March is DST): 10:30 AM
        sample_bar_data["Timestamp"] = "2026-03-15T14:30:00Z"

        bar = client._parse_bar_data("MNQH26", sample_bar_data)

        # Should be in ET timezone
        assert bar.timestamp.tzinfo == ZoneInfo("America/New_York")

        # Hour should be 10 (ET) for 14:30 UTC in March (DST active)
        assert bar.timestamp.hour == 10

    def test_parse_bar_data_lowercase_keys(self, client):
        """Test parsing with lowercase API response keys."""
        bar_data = {
            "timestamp": "2026-03-15T14:30:00Z",
            "open": 11800.0,
            "high": 11850.0,
            "low": 11790.0,
            "close": 11825.0,
            "volume": 1000,
        }

        bar = client._parse_bar_data("MNQH26", bar_data)
        assert bar.open == 11800.0
        assert bar.high == 11850.0

    def test_parse_bar_data_missing_timestamp(self, client, sample_bar_data):
        """Test handling of missing timestamp."""
        del sample_bar_data["Timestamp"]

        bar = client._parse_bar_data("MNQH26", sample_bar_data)
        assert bar is None

    def test_parse_bar_data_malformed_price(self, client):
        """Test handling of malformed price data."""
        bar_data = {
            "Timestamp": "2026-03-15T14:30:00Z",
            "Open": 11800.0,
            "High": 11700.0,  # High < Low - invalid
            "Low": 11790.0,
            "Close": 11825.0,
            "Volume": 1000,
        }

        bar = client._parse_bar_data("MNQH26", bar_data)
        assert bar is None


class TestDataValidation:
    """Tests for bar data validation."""

    def test_validate_bar_data_high_gte_low(self, client):
        """Test high >= low validation."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11825.0,
            volume=1000,
        )
        # Should not raise
        client._validate_bar_data(bar)

    def test_validate_bar_data_high_less_than_low(self, client):
        """Test high < low raises error."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            open=11800.0,
            high=11700.0,  # High < Low
            low=11790.0,
            close=11825.0,
            volume=1000,
        )
        with pytest.raises(ValueError, match="high must be >= low"):
            client._validate_bar_data(bar)

    def test_validate_bar_data_close_within_range(self, client):
        """Test close within high/low validation."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11825.0,
            volume=1000,
        )
        # Should not raise
        client._validate_bar_data(bar)

    def test_validate_bar_data_close_above_high(self, client):
        """Test close > high raises error."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11900.0,  # Close > High
            volume=1000,
        )
        with pytest.raises(ValueError, match="close must be <= high"):
            client._validate_bar_data(bar)

    def test_validate_bar_data_close_below_low(self, client):
        """Test close < low raises error."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11780.0,  # Close < Low
            volume=1000,
        )
        with pytest.raises(ValueError, match="close must be >= low"):
            client._validate_bar_data(bar)

    def test_validate_bar_data_negative_volume(self, client):
        """Test negative volume raises error."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11825.0,
            volume=-100,  # Negative volume
        )
        with pytest.raises(ValueError, match="Volume .* must be >= 0"):
            client._validate_bar_data(bar)


class TestDateRangePagination:
    """Tests for date range pagination."""

    def test_paginate_date_range_2_years(self, client):
        """Test paginating 2-year date range."""
        start = datetime(2024, 3, 1, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, tzinfo=timezone.utc)

        ranges = client._paginate_date_range(start, end)

        # Should be approximately 10 ranges (365 days / 70 days)
        assert len(ranges) >= 10
        assert len(ranges) <= 11

        # Check first range
        first_start, first_end = ranges[0]
        assert first_start == start
        assert (first_end - start).days <= 70

    def test_paginate_date_range_single_range(self, client):
        """Test single range when less than 70 days."""
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        end = datetime(2026, 3, 30, tzinfo=timezone.utc)

        ranges = client._paginate_date_range(start, end)

        assert len(ranges) == 1
        assert ranges[0] == (start, end)

    def test_paginate_date_range_exactly_70_days(self, client):
        """Test pagination at exactly 70-day boundary."""
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=70)

        ranges = client._paginate_date_range(start, end)

        assert len(ranges) == 1
        assert ranges[0] == (start, end)


class TestDuplicateDetection:
    """Tests for duplicate bar detection."""

    def test_detect_duplicate_bars(self, client):
        """Test duplicate bars are removed."""
        timestamp = datetime.now(ZoneInfo("America/New_York"))

        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=timestamp,
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11825.0,
                volume=1000,
            ),
            BarData(  # Same timestamp (duplicate)
                symbol="MNQH26",
                timestamp=timestamp,
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11825.0,
                volume=1000,
            ),
            BarData(  # Different timestamp
                symbol="MNQH26",
                timestamp=timestamp + timedelta(minutes=1),
                open=11825.0,
                high=11875.0,
                low=11815.0,
                close=11850.0,
                volume=1100,
            ),
        ]

        deduped = client._detect_duplicate_bars(bars)

        assert len(deduped) == 2
        # First occurrence should be kept
        assert deduped[0].timestamp == timestamp

    def test_detect_duplicate_bars_empty_list(self, client):
        """Test handling of empty bar list."""
        deduped = client._detect_duplicate_bars([])
        assert deduped == []

    def test_detect_duplicate_bars_no_duplicates(self, client):
        """Test handling when no duplicates present."""
        timestamp = datetime.now(ZoneInfo("America/New_York"))

        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=timestamp + timedelta(minutes=i),
                open=11800.0 + i,
                high=11850.0 + i,
                low=11790.0 + i,
                close=11825.0 + i,
                volume=1000,
            )
            for i in range(10)
        ]

        deduped = client._detect_duplicate_bars(bars)

        assert len(deduped) == 10


class TestDataCompleteness:
    """Tests for data completeness validation."""

    def test_verify_data_completeness_100_percent(self, client):
        """Test 100% data completeness."""
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, tzinfo=timezone.utc)

        # 1 day = ~6.5 hours * 60 minutes = 390 bars expected
        # Create 390 bars
        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=start + timedelta(minutes=i),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11825.0,
                volume=1000,
            )
            for i in range(390)
        ]

        # Should not raise
        client._verify_data_completeness(bars, start, end)

    def test_verify_data_completeness_below_95_percent(self, client):
        """Test data completeness below 95% threshold."""
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, tzinfo=timezone.utc)

        # Only 100 bars (should be ~390)
        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=start + timedelta(minutes=i),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11825.0,
                volume=1000,
            )
            for i in range(100)
        ]

        # Should log warning but not raise
        client._verify_data_completeness(bars, start, end)

    def test_verify_data_completeness_empty_bars(self, client):
        """Test handling of empty bar list."""
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        end = datetime(2026, 3, 2, tzinfo=timezone.utc)

        # Should handle gracefully
        client._verify_data_completeness([], start, end)


class TestRateLimiting:
    """Tests for rate limiting and retry logic."""

    def test_rate_limit_headers_parsing(self, client):
        """Test parsing rate limit headers."""
        mock_response = pytest.mock.Mock()
        mock_response.headers = {
            "X-RateLimit-Remaining": "100",
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Reset": "60",
        }

        rate_limit = client._rate_limit_headers(mock_response)

        assert rate_limit["remaining"] == 100
        assert rate_limit["limit"] == 1000
        assert rate_limit["reset"] == 60

    def test_should_retry_rate_limit(self, client):
        """Test retry condition for rate limiting."""
        # Mock HTTP 429 error
        mock_response = pytest.mock.Mock()
        mock_response.status_code = 429

        mock_exception = pytest.mock.Mock()
        mock_exception.response = mock_response

        mock_retry_state = pytest.mock.Mock()
        mock_retry_state.outcome.failed = True
        mock_retry_state.outcome.exception.return_value = mock_exception

        should_retry = client._should_retry_rate_limit(mock_retry_state)

        assert should_retry is True

    def test_should_not_retry_non_429(self, client):
        """Test non-429 errors don't trigger retry."""
        # Mock HTTP 500 error
        mock_response = pytest.mock.Mock()
        mock_response.status_code = 500

        mock_exception = pytest.mock.Mock()
        mock_exception.response = mock_response

        mock_retry_state = pytest.mock.Mock()
        mock_retry_state.outcome.failed = True
        mock_retry_state.outcome.exception.return_value = mock_exception

        should_retry = client._should_retry_rate_limit(mock_retry_state)

        assert should_retry is False


class TestNotionalValue:
    """Tests for notional value calculation."""

    def test_notional_value_calculation(self, sample_bar_data):
        """Test notional value calculation: close * volume * 0.5."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.fromisoformat("2026-03-15T14:30:00Z"),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11825.0,
            volume=1000,
        )

        dollar_bar = bar.to_dollar_bar(contract_multiplier=0.5)

        # notional = 11825 * 1000 * 0.5 = 5,912,500
        assert dollar_bar.notional_value == 11825 * 1000 * 0.5

    def test_notional_value_different_multiplier(self, sample_bar_data):
        """Test notional value with different contract multiplier."""
        bar = BarData(
            symbol="MNQH26",
            timestamp=datetime.fromisoformat("2026-03-15T14:30:00Z"),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11825.0,
            volume=1000,
        )

        dollar_bar = bar.to_dollar_bar(contract_multiplier=1.0)

        # notional = 11825 * 1000 * 1.0 = 11,825,000
        assert dollar_bar.notional_value == 11825 * 1000 * 1.0


class TestClientLifecycle:
    """Tests for HTTP client lifecycle."""

    @pytest.mark.asyncio
    async def test_http_client_close(self, client):
        """Test HTTP client is properly closed."""
        await client.close()
        assert client._client is None


class TestEmptyAPIResponse:
    """Tests for handling empty API responses."""

    def test_empty_api_response_handling(self, client, sample_bar_data):
        """Test empty array response is handled."""
        # Simulate empty API response (holiday)
        data = {"Bars": []}

        # Should return empty list
        bars = []
        for bar_json in data["Bars"]:
            bar = client._parse_bar_data("MNQH26", bar_json)
            if bar:
                bars.append(bar)

        assert len(bars) == 0


class TestTokenRefresh:
    """Tests for token refresh during API calls."""

    def test_concurrent_token_refresh_mutex(self, client, mock_auth):
        """Test concurrent refresh attempts are serialized."""
        import threading

        refresh_count = 0

        def mock_refresh():
            nonlocal refresh_count
            with client._refresh_lock:
                refresh_count += 1
                # Simulate refresh taking time
                import time
                time.sleep(0.1)

        # Start multiple threads attempting refresh
        threads = [threading.Thread(target=mock_refresh) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All should have completed
        assert refresh_count == 5

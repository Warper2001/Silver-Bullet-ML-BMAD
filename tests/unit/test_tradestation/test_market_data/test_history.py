"""Unit tests for HistoryClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from datetime import datetime, timezone

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import ValidationError, APIError
from src.execution.tradestation.market_data.history import HistoryClient
from src.execution.tradestation.models import HistoricalBar


class TestHistoryClient:
    """Test suite for HistoryClient class."""

    @pytest.fixture
    def mock_client(self) -> TradeStationClient:
        """Create a mock TradeStationClient."""
        client = MagicMock(spec=TradeStationClient)
        client.api_base_url = "https://sim-api.tradestation.com/v3"
        return client

    @pytest.fixture
    def history_client(self, mock_client: MagicMock) -> HistoryClient:
        """Create a HistoryClient with mock TradeStationClient."""
        return HistoryClient(mock_client)

    @pytest.fixture
    def sample_bars_response(self) -> dict:
        """Sample historical bars response from API."""
        return {
            "Bars": [
                {
                    "Symbol": "MNQH26",
                    "TimeStamp": "2026-03-28T09:30:00Z",
                    "Open": 15000.00,
                    "High": 15010.00,
                    "Low": 14995.00,
                    "Close": 15005.00,
                    "Volume": 1000,
                    "BarType": "minute",
                },
                {
                    "Symbol": "MNQH26",
                    "TimeStamp": "2026-03-28T09:31:00Z",
                    "Open": 15005.00,
                    "High": 15015.00,
                    "Low": 15000.00,
                    "Close": 15012.00,
                    "Volume": 800,
                    "BarType": "minute",
                },
            ]
        }

    def test_initialization(self, history_client: HistoryClient) -> None:
        """Test HistoryClient initialization."""
        assert history_client.client is not None
        assert history_client.logger is not None
        assert "minute" in history_client.VALID_BAR_TYPES

    def test_validate_bar_type_valid(self, history_client: HistoryClient) -> None:
        """Test validation of valid bar type and interval."""
        # Should not raise
        history_client._validate_bar_type("minute", 1)
        history_client._validate_bar_type("minute5", 5)
        history_client._validate_bar_type("daily", 1)

    def test_validate_bar_type_invalid_type(self, history_client: HistoryClient) -> None:
        """Test validation of invalid bar type."""
        with pytest.raises(ValidationError, match="Invalid bar_type"):
            history_client._validate_bar_type("invalid_type", 1)

    def test_validate_bar_type_invalid_interval(self, history_client: HistoryClient) -> None:
        """Test validation of invalid interval for bar type."""
        with pytest.raises(ValidationError, match="Invalid interval"):
            history_client._validate_bar_type("minute", 99)

    def test_validate_symbol_valid(self, history_client: HistoryClient) -> None:
        """Test validation of valid symbol."""
        # Should not raise
        history_client._validate_symbol("MNQH26")
        history_client._validate_symbol("ESM26")

    def test_validate_symbol_invalid(self, history_client: HistoryClient) -> None:
        """Test validation of invalid symbol."""
        with pytest.raises(ValidationError, match="Invalid symbol"):
            history_client._validate_symbol("")

        with pytest.raises(ValidationError, match="Symbol should be alphanumeric"):
            history_client._validate_symbol("MNQ@26")

    def test_validate_dates_valid(self, history_client: HistoryClient) -> None:
        """Test validation of valid dates."""
        # Should not raise
        history_client._validate_dates("2024-01-01", "2024-01-31")
        history_client._validate_dates(None, None)
        history_client._validate_dates("2024-01-01", None)

    def test_validate_dates_invalid_format(self, history_client: HistoryClient) -> None:
        """Test validation of invalid date format."""
        with pytest.raises(ValidationError, match="Invalid start_date format"):
            history_client._validate_dates("01-01-2024", "2024-01-31")

        with pytest.raises(ValidationError, match="Invalid end_date format"):
            history_client._validate_dates("2024-01-01", "31-01-2024")

    def test_validate_dates_start_after_end(self, history_client: HistoryClient) -> None:
        """Test validation when start_date is after end_date."""
        with pytest.raises(ValidationError, match="start_date.*must be before end_date"):
            history_client._validate_dates("2024-12-31", "2024-01-01")

    @pytest.mark.asyncio
    async def test_get_historical_bars_success(
        self, history_client: HistoryClient, sample_bars_response: dict
    ) -> None:
        """Test successful historical bars retrieval."""
        # Mock the _request method
        history_client.client._request = AsyncMock(return_value=sample_bars_response)

        # Call get_historical_bars
        bars = await history_client.get_historical_bars(
            symbol="MNQH26",
            bar_type="minute",
            interval=1,
            start_date="2026-03-28",
            end_date="2026-03-28",
        )

        # Verify
        assert len(bars) == 2
        assert bars[0].symbol == "MNQH26"
        assert bars[0].open == 15000.00
        assert bars[0].high == 15010.00

    @pytest.mark.asyncio
    async def test_get_historical_bars_empty_response(self, history_client: HistoryClient) -> None:
        """Test get_historical_bars with empty response."""
        # Mock empty response
        history_client.client._request = AsyncMock(return_value={"Bars": []})

        # Should raise ValidationError
        with pytest.raises(ValidationError, match="No bars returned"):
            await history_client.get_historical_bars(symbol="MNQH26")

    @pytest.mark.asyncio
    async def test_get_bar_data(self, history_client: HistoryClient) -> None:
        """Test get_bar_data convenience method."""
        # Mock the _request method
        history_client.client._request = AsyncMock(return_value={"Bars": []})

        # Call get_bar_data
        bars = await history_client.get_bar_data(
            symbol="MNQH26",
            days_back=30,
            bar_type="daily",
        )

        # Verify it was called
        history_client.client._request.assert_called_once()

    def test_calculate_expected_bars_minute(self, history_client: HistoryClient) -> None:
        """Test expected bars calculation for minute data."""
        bars = history_client._calculate_expected_bars(
            start_date="2026-03-28",
            end_date="2026-03-28",
            bar_type="minute",
            interval=1,
        )

        # Should be 1440 minutes in a day
        assert bars == 1440

    def test_calculate_expected_bars_daily(self, history_client: HistoryClient) -> None:
        """Test expected bars calculation for daily data."""
        bars = history_client._calculate_expected_bars(
            start_date="2026-03-01",
            end_date="2026-03-31",
            bar_type="daily",
            interval=1,
        )

        # Should be 31 days in March
        assert bars == 31

    def test_calculate_expected_bars_hourly(self, history_client: HistoryClient) -> None:
        """Test expected bars calculation for hourly data."""
        bars = history_client._calculate_expected_bars(
            start_date="2026-03-28",
            end_date="2026-03-28",
            bar_type="hour",
            interval=1,
        )

        # Should be 24 hours in a day
        assert bars == 24

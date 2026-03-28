"""Unit tests for QuotesClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import ValidationError, APIError
from src.execution.tradestation.market_data.quotes import QuotesClient
from src.execution.tradestation.models import TradeStationQuote


class TestQuotesClient:
    """Test suite for QuotesClient class."""

    @pytest.fixture
    def mock_client(self) -> TradeStationClient:
        """Create a mock TradeStationClient."""
        client = MagicMock(spec=TradeStationClient)
        client.api_base_url = "https://sim-api.tradestation.com/v3"
        return client

    @pytest.fixture
    def quotes_client(self, mock_client: MagicMock) -> QuotesClient:
        """Create a QuotesClient with mock TradeStationClient."""
        return QuotesClient(mock_client)

    @pytest.fixture
    def sample_quotes_response(self) -> dict:
        """Sample quotes response from API."""
        return {
            "Quotes": [
                {
                    "Symbol": "MNQH26",
                    "Bid": 15000.00,
                    "Ask": 15000.25,
                    "Last": 15000.125,
                    "BidSize": 100,
                    "AskSize": 50,
                    "LastSize": 1,
                    "TimeStamp": "2026-03-28T12:00:00Z",
                    "Volume": 1000000,
                    "Open": 14950.00,
                    "High": 15050.00,
                    "Low": 14900.00,
                    "Close": 14975.00,
                },
                {
                    "Symbol": "MNQM26",
                    "Bid": 14900.00,
                    "Ask": 14900.25,
                    "Last": 14900.125,
                    "TimeStamp": "2026-03-28T12:00:00Z",
                },
            ]
        }

    def test_initialization(self, quotes_client: QuotesClient) -> None:
        """Test QuotesClient initialization."""
        assert quotes_client.client is not None
        assert quotes_client.logger is not None

    @pytest.mark.asyncio
    async def test_get_quotes_success(
        self, quotes_client: QuotesClient, sample_quotes_response: dict
    ) -> None:
        """Test successful quotes retrieval."""
        # Mock the _request method
        quotes_client.client._request = AsyncMock(return_value=sample_quotes_response)

        # Call get_quotes
        quotes = await quotes_client.get_quotes(["MNQH26", "MNQM26"])

        # Verify
        assert len(quotes) == 2
        assert quotes[0].symbol == "MNQH26"
        assert quotes[0].bid == 15000.00
        assert quotes[1].symbol == "MNQM26"

    @pytest.mark.asyncio
    async def test_get_quotes_empty_symbols(self, quotes_client: QuotesClient) -> None:
        """Test get_quotes with empty symbols list."""
        with pytest.raises(ValidationError, match="Symbols list cannot be empty"):
            await quotes_client.get_quotes([])

    @pytest.mark.asyncio
    async def test_get_quotes_too_many_symbols(self, quotes_client: QuotesClient) -> None:
        """Test get_quotes with too many symbols (> 100)."""
        symbols = [f"SYMBOL{i}" for i in range(101)]

        with pytest.raises(ValidationError, match="Cannot request more than 100 symbols"):
            await quotes_client.get_quotes(symbols)

    @pytest.mark.asyncio
    async def test_get_quotes_api_error(
        self, quotes_client: QuotesClient
    ) -> None:
        """Test get_quotes with API error."""
        # Mock API error
        quotes_client.client._request = AsyncMock(side_effect=APIError("API Error"))

        # Should raise the API error
        with pytest.raises(APIError):
            await quotes_client.get_quotes(["MNQH26"])

    @pytest.mark.asyncio
    async def test_get_quote_snapshot_success(
        self, quotes_client: QuotesClient, sample_quotes_response: dict
    ) -> None:
        """Test get_quote_snapshot for single symbol."""
        # Mock the _request method
        quotes_client.client._request = AsyncMock(return_value=sample_quotes_response)

        # Call get_quote_snapshot
        quote = await quotes_client.get_quote_snapshot("MNQH26")

        # Verify
        assert quote.symbol == "MNQH26"
        assert quote.bid == 15000.00
        assert quote.ask == 15000.25

    @pytest.mark.asyncio
    async def test_get_quote_snapshot_no_data(self, quotes_client: QuotesClient) -> None:
        """Test get_quote_snapshot when no data returned."""
        # Mock empty response
        quotes_client.client._request = AsyncMock(return_value={"Quotes": []})

        # Should raise ValidationError
        with pytest.raises(ValidationError, match="No quote data returned"):
            await quotes_client.get_quote_snapshot("UNKNOWN_SYMBOL")

    @pytest.mark.asyncio
    async def test_subscribe_quotes_placeholder(self, quotes_client: QuotesClient) -> None:
        """Test subscribe_quotes raises NotImplementedError."""
        async def dummy_callback(quote: TradeStationQuote) -> None:
            pass

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Use QuoteStreamParser"):
            await quotes_client.subscribe_quotes(["MNQH26"], dummy_callback)

"""Unit tests for TradeStation SDK streaming client.

Tests real-time quote streaming from TradeStation SIM environment.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.tradestation_auth import TradeStationAuth
from src.execution.tradestation.market_data.streaming import (
    QuoteStreamParser,
    StreamPosition,
)


@pytest.fixture
def mock_auth():
    """Create mock TradeStation authentication."""
    auth = MagicMock(spec=TradeStationAuth)
    auth.get_valid_access_token.return_value = "test_token"
    return auth


@pytest.fixture
def mock_quote_response():
    """Create mock quote response from TradeStation API."""
    return {
        "Quotes": [
            {
                "Symbol": "MNQH26",
                "Timestamp": "2026-04-03T14:30:00Z",
                "Last": 11850.0,
                "Bid": 11849.75,
                "Ask": 11850.25,
                "BidSize": 100,
                "AskSize": 100,
                "Volume": 1250,
                "OpenInterest": 50000,
            }
        ]
    }


class TestStreamPosition:
    """Tests for StreamPosition dataclass."""

    def test_stream_position_creation(self):
        """Test creating a StreamPosition."""
        position = StreamPosition(
            symbol="MNQH26",
            timestamp=datetime.now(timezone.utc),
            last_price=11850.0,
            bid=11849.75,
            ask=11850.25,
            volume=1000,
        )

        assert position.symbol == "MNQH26"
        assert position.last_price == 11850.0
        assert position.bid == 11849.75
        assert position.ask == 11850.25
        assert position.volume == 1000


class TestQuoteStreamParser:
    """Tests for QuoteStreamParser class."""

    @pytest.mark.asyncio
    async def test_quote_stream_parser_initialization(self, mock_auth):
        """Test initializing QuoteStreamParser."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        assert parser._auth == mock_auth
        assert parser._symbols == ["MNQH26"]
        assert parser._environment == "sim"
        assert parser._running is False

    @pytest.mark.asyncio
    async def test_start_streaming(self, mock_auth):
        """Test starting the quote stream."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Mock websocket connection
        with patch.object(parser, "_connect_websocket") as mock_connect:
            mock_connect.return_value = MagicMock()

            await parser.start()

            assert parser._running is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_streaming(self, mock_auth):
        """Test stopping the quote stream."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Start and then stop
        with patch.object(parser, "_connect_websocket"):
            await parser.start()
            assert parser._running is True

            await parser.stop()
            assert parser._running is False

    @pytest.mark.asyncio
    async def test_subscribe_to_quotes(self, mock_auth):
        """Test subscribing to quote stream."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Subscribe to quotes
        queue = await parser.subscribe()

        assert isinstance(queue, asyncio.Queue)
        assert parser._subscribers == 1

    @pytest.mark.asyncio
    async def test_parse_quote_message(self, mock_auth, mock_quote_response):
        """Test parsing quote message from TradeStation API."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Parse quote
        positions = parser._parse_quotes(mock_quote_response)

        assert len(positions) == 1
        assert positions[0].symbol == "MNQH26"
        assert positions[0].last_price == 11850.0
        assert positions[0].bid == 11849.75
        assert positions[0].ask == 11850.25

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self, mock_auth):
        """Test broadcasting quotes to all subscribers."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Create multiple subscribers
        queue1 = await parser.subscribe()
        queue2 = await parser.subscribe()

        # Create test position
        test_position = StreamPosition(
            symbol="MNQH26",
            timestamp=datetime.now(timezone.utc),
            last_price=11850.0,
            bid=11849.75,
            ask=11850.25,
            volume=1000,
        )

        # Broadcast to subscribers
        await parser._broadcast_to_subscribers(test_position)

        # Verify both subscribers received the quote
        received1 = await queue1.get()
        received2 = await queue2.get()

        assert received1.symbol == "MNQH26"
        assert received2.symbol == "MNQH26"

    @pytest.mark.asyncio
    async def test_sim_environment_url(self, mock_auth):
        """Test that SIM environment uses correct API URL."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Verify SIM URL is used
        assert "sim-api.tradestation.com" in parser._get_stream_url()

    @pytest.mark.asyncio
    async def test_connection_retry_on_failure(self, mock_auth):
        """Test connection retry on failure."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Mock connection that fails then succeeds
        call_count = 0

        async def mock_connect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return MagicMock()

        with patch.object(parser, "_connect_websocket", side_effect=mock_connect):
            # Should retry and eventually succeed
            await parser.start()

            assert call_count == 3  # Two failures, then success
            assert parser._running is True

    @pytest.mark.asyncio
    async def test_handle_invalid_quote_data(self, mock_auth):
        """Test handling invalid quote data gracefully."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Test with invalid data (missing required fields)
        invalid_response = {
            "Quotes": [
                {
                    "Symbol": "MNQH26",
                    # Missing timestamp and other required fields
                }
            ]
        }

        # Should return empty list or None, not crash
        positions = parser._parse_quotes(invalid_response)

        assert positions is None or len(positions) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_from_quotes(self, mock_auth):
        """Test unsubscribing from quote stream."""
        parser = QuoteStreamParser(
            auth=mock_auth,
            symbols=["MNQH26"],
            environment="sim",
        )

        # Subscribe then unsubscribe
        queue = await parser.subscribe()
        assert parser._subscribers == 1

        await parser.unsubscribe(queue)
        assert parser._subscribers == 0

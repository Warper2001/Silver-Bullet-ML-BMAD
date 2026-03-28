"""
TradeStation SDK - Real-Time Quotes Module

This module provides real-time quote data access from TradeStation API.

Key Features:
- Get current quotes for multiple symbols
- Subscribe to real-time quote updates
- Integration with TradeStationClient for authentication
- Error handling for invalid symbols and network issues

Usage:
    async with TradeStationClient(env="sim", ...) as client:
        quotes_client = QuotesClient(client)
        quotes = await quotes_client.get_quotes(["MNQH26", "MNQM26"])
"""

import logging
from typing import Any

from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import APIError, NetworkError, ValidationError
from src.execution.tradestation.models import TradeStationQuote
from src.execution.tradestation.utils import setup_logger


class QuotesClient:
    """
    Client for real-time quote data from TradeStation API.

    Provides methods to fetch current market data for multiple symbols.

    Attributes:
        client: TradeStationClient instance for API communication

    Example:
        async with TradeStationClient(env="sim", ...) as client:
            quotes_client = QuotesClient(client)
            quotes = await quotes_client.get_quotes(["MNQH26"])
            print(quotes[0].bid, quotes[0].ask)
    """

    def __init__(self, client: TradeStationClient) -> None:
        """
        Initialize QuotesClient.

        Args:
            client: Authenticated TradeStationClient instance
        """
        self.client = client
        self.logger = setup_logger(f"{__name__}.QuotesClient")

    async def get_quotes(
        self,
        symbols: list[str],
    ) -> list[TradeStationQuote]:
        """
        Get current quotes for multiple symbols.

        Args:
            symbols: List of trading symbols (e.g., ["MNQH26", "MNQM26"])

        Returns:
            List of TradeStationQuote objects with current market data

        Raises:
            ValidationError: If symbols list is empty or contains invalid symbols
            APIError: On API errors
            NetworkError: On network errors

        Example:
            quotes = await quotes_client.get_quotes(["MNQH26"])
            print(f"Bid: {quotes[0].bid}, Ask: {quotes[0].ask}")
        """
        if not symbols:
            raise ValidationError("Symbols list cannot be empty")

        if len(symbols) > 100:
            raise ValidationError("Cannot request more than 100 symbols at once")

        self.logger.info(f"Fetching quotes for {len(symbols)} symbols: {symbols}")

        params = {
            "symbols": ",".join(symbols),
        }

        try:
            response = await self.client._request(
                "GET",
                "/data/quote",
                params=params,
            )

            # Parse response
            quotes_data = response.get("Quotes", [])
            quotes = [TradeStationQuote(**quote) for quote in quotes_data]

            self.logger.info(f"Received {len(quotes)} quotes")
            return quotes

        except (ValidationError, APIError, NetworkError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching quotes: {e}")
            raise APIError(f"Unexpected error fetching quotes: {e}")

    async def get_quote_snapshot(
        self,
        symbol: str,
    ) -> TradeStationQuote:
        """
        Get current quote for a single symbol.

        Convenience method for fetching a single symbol's quote.

        Args:
            symbol: Trading symbol (e.g., "MNQH26")

        Returns:
            TradeStationQuote object with current market data

        Raises:
            ValidationError: If symbol is invalid
            APIError: On API errors
            NetworkError: On network errors

        Example:
            quote = await quotes_client.get_quote_snapshot("MNQH26")
            print(f"Bid: {quote.bid}, Ask: {quote.ask}, Last: {quote.last}")
        """
        self.logger.info(f"Fetching quote snapshot for {symbol}")

        quotes = await self.get_quotes([symbol])

        if not quotes:
            raise ValidationError(f"No quote data returned for symbol: {symbol}")

        return quotes[0]

    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: callable,  # type: ignore
    ) -> None:
        """
        Subscribe to real-time quote updates for symbols.

        This is a placeholder for WebSocket-like streaming functionality.
        In production, this would use the streaming endpoint.

        Args:
            symbols: List of symbols to subscribe to
            callback: Async callback function to receive quote updates

        Note:
            This is a placeholder. Real-time streaming should use
            QuoteStreamParser from the streaming module.

        Example:
            async def on_quote(quote: TradeStationQuote):
                print(f"Quote update: {quote.symbol} @ {quote.last}")

            await quotes_client.subscribe_quotes(["MNQH26"], on_quote)
        """
        self.logger.warning("subscribe_quotes is a placeholder - use QuoteStreamParser for real-time streaming")
        # TODO: Implement with QuoteStreamParser in streaming module
        raise NotImplementedError("Use QuoteStreamParser from streaming module for real-time quotes")

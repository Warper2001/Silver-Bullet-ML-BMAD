"""
BYDFI REST API Client

Provides comprehensive interface to BYDFI REST API for cryptocurrency spot trading.
Based on BYDFI API documentation: https://developers.bydfi.com/en/public

Features:
- Market data (quotes, klines, order book)
- Account information
- Order management
- Error handling and rate limiting
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from src.data.bydfi_config import load_bydfi_settings
from src.execution.bydfi.auth.signature import create_bydfi_signature_generator
from src.execution.bydfi.models import (
    BYDFIAccount,
    BYDFIErrorResponse,
    BYDFIKline,
    BYDFIOrderBook,
    BYDFIQuote,
)

logger = logging.getLogger(__name__)


class BYDFIClientError(Exception):
    """BYDFI API client error."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class BYDFIClient:
    """
    BYDFI REST API client.

    Handles all BYDFI API interactions including:
    - Market data retrieval
    - Account information
    - Order management
    - Error handling and rate limiting

    Example:
        >>> client = BYDFIClient()
        >>> async with client:
        ...     quote = await client.get_quotes("BTC-USDT")
        ...     print(f"Current price: {quote.price}")
    """

    def __init__(self):
        """Initialize BYDFI client."""
        settings = load_bydfi_settings()

        self.api_key = settings.bydfi_api_key
        self.api_secret = settings.bydfi_api_secret
        self.base_url = settings.base_url
        self.trading_symbol = settings.bydfi_trading_symbol

        # Initialize signature generator
        self.signature_generator = create_bydfi_signature_generator(
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

        # HTTP client (initialized in __aenter__)
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(f"BYDFI client initialized: base_url={self.base_url}")

    async def __aenter__(self):
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
            )

            # Test connectivity
            try:
                await self._test_connectivity()
                logger.info("BYDFI client connected successfully")
            except Exception as e:
                logger.error(f"BYDFI client connectivity test failed: {e}")
                raise

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("BYDFI client closed")

    async def _test_connectivity(self):
        """Test API connectivity."""
        try:
            response = await self._client.get("/v1/public/api_limit")
            response.raise_for_status()
            logger.debug("BYDFI API connectivity test passed")
        except Exception as e:
            logger.error(f"BYDFI API connectivity test failed: {e}")
            raise

    def _build_headers(
        self,
        method: str,
        endpoint: str,
        query_string: str = "",
        body: str = "",
    ) -> dict[str, str]:
        """
        Build signed headers for API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            query_string: Query parameters
            body: Request body

        Returns:
            dict: Complete headers with signature
        """
        return self.signature_generator.generate_headers(
            method=method,
            endpoint=endpoint,
            query_string=query_string,
            body=body,
        )

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """
        Handle API response and check for errors.

        Args:
            response: HTTP response

        Returns:
            dict: Response data

        Raises:
            BYDFIClientError: If API returns error
        """
        try:
            data = response.json()

            # BYDFI returns {code, message, data} format
            if data.get("code") != 200:
                error = BYDFIErrorResponse(
                    code=data.get("code", 0),
                    message=data.get("message", "Unknown error"),
                )
                raise BYDFIClientError(
                    f"BYDFI API error: {error.message}",
                    code=error.code,
                )

            return data.get("data", data)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise BYDFIClientError(f"HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Response handling error: {e}")
            raise

    # Market Data Methods

    async def get_quotes(self, symbol: str) -> BYDFIQuote:
        """
        Get current price quote for a symbol.

        Args:
            symbol: Trading symbol (e.g., BTC-USDT)

        Returns:
            BYDFIQuote: Current quote

        Raises:
            BYDFIClientError: If API request fails
        """
        if not self._client:
            raise BYDFIClientError("Client not initialized")

        try:
            # Build query string
            query_params = {"symbol": symbol}
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

            # Build headers (even public endpoints require auth on BYDFI)
            headers = self._build_headers(
                method="GET",
                endpoint="/v1/spot/ticker",
                query_string=query_string,
            )

            response = await self._client.get(
                "/v1/spot/ticker",
                params=query_params,
                headers=headers,
            )

            data = self._handle_response(response)

            # Parse quote (adjust based on actual BYDFI response format)
            return BYDFIQuote(
                symbol=symbol,
                price=str(data.get("price", "0")),
                volume=str(data.get("volume", "0")),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error getting quotes for {symbol}: {e}")
            raise

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 100,
    ) -> list[BYDFIKline]:
        """
        Get historical kline (candlestick) data.

        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of klines to retrieve

        Returns:
            list[BYDFIKline]: Historical klines

        Raises:
            BYDFIClientError: If API request fails
        """
        if not self._client:
            raise BYDFIClientError("Client not initialized")

        try:
            # Build query string
            query_params = {
                "symbol": symbol,
                "interval": interval,
                "limit": str(limit),
            }
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

            # Build headers
            headers = self._build_headers(
                method="GET",
                endpoint="/v1/spot/kline",
                query_string=query_string,
            )

            response = await self._client.get(
                "/v1/spot/kline",
                params=query_params,
                headers=headers,
            )

            data = self._handle_response(response)

            # Parse klines (adjust based on actual BYDFI response format)
            klines = []
            for item in data if isinstance(data, list) else []:
                klines.append(
                    BYDFIKline(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(
                            int(item.get("time", 0)) / 1000,
                            tz=timezone.utc,
                        ),
                        open=str(item.get("open", "0")),
                        high=str(item.get("high", "0")),
                        low=str(item.get("low", "0")),
                        close=str(item.get("close", "0")),
                        volume=str(item.get("vol", "0")),
                    )
                )

            return klines

        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            raise

    async def get_order_book_depth(
        self,
        symbol: str,
        limit: int = 20,
    ) -> BYDFIOrderBook:
        """
        Get order book depth for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of levels to retrieve

        Returns:
            BYDFIOrderBook: Order book depth

        Raises:
            BYDFIClientError: If API request fails
        """
        if not self._client:
            raise BYDFIClientError("Client not initialized")

        try:
            # Build query string
            query_params = {"symbol": symbol, "limit": str(limit)}
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

            # Build headers
            headers = self._build_headers(
                method="GET",
                endpoint="/v1/market/depth",
                query_string=query_string,
            )

            response = await self._client.get(
                "/v1/market/depth",
                params=query_params,
                headers=headers,
            )

            data = self._handle_response(response)

            # Parse order book (adjust based on actual BYDFI response format)
            bids = [
                (str(str(b[0])), str(str(b[1])))
                for b in data.get("bids", [])[:limit]
            ]
            asks = [
                (str(str(a[0])), str(str(a[1])))
                for a in data.get("asks", [])[:limit]
            ]

            return BYDFIOrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            raise

    async def get_account_info(self) -> list[BYDFIAccount]:
        """
        Get account information.

        Returns:
            list[BYDFIAccount]: List of account balances

        Raises:
            BYDFIClientError: If API request fails
        """
        if not self._client:
            raise BYDFIClientError("Client not initialized")

        try:
            # Build headers
            headers = self._build_headers(
                method="GET",
                endpoint="/v1/account/assets",
            )

            response = await self._client.get(
                "/v1/account/assets",
                headers=headers,
            )

            data = self._handle_response(response)

            # Parse accounts (adjust based on actual BYDFI response format)
            accounts = []
            for item in data if isinstance(data, list) else []:
                accounts.append(
                    BYDFIAccount(
                        currency=item.get("coin", ""),
                        balance=str(item.get("free", "0")),
                        frozen=str(item.get("frozen", "0")),
                        total=str(item.get("total", "0")),
                    )
                )

            return accounts

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise


def create_bydfi_client() -> BYDFIClient:
    """
    Factory function to create BYDFI client.

    Returns:
        BYDFIClient: Configured BYDFI client
    """
    return BYDFIClient()

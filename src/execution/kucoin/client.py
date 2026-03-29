"""
KuCoin REST API Client

This module provides the main KuCoinClient for interacting with the KuCoin REST API.
It handles authentication, rate limiting, error handling, and request signing.

API Documentation: https://docs.kucoin.com/

Architecture:
- Async context manager for connection lifecycle
- Rate limiting (weight-based)
- HMAC SHA256 signature with passphrase
- Comprehensive error handling with custom exceptions
"""

import logging
from datetime import datetime, timezone

import httpx

from src.data.kucoin_config import load_kucoin_settings
from src.execution.kucoin.auth.signature import SignatureGenerator
from src.execution.kucoin.exceptions import APIError, NetworkError, RateLimitError, ValidationError
from src.execution.kucoin.models import (
    KuCoinAccount,
    KuCoinKline,
    KuCoinOrder,
    KuCoinOrderBook,
    KuCoinQuote,
    KuCoinTrade,
)
from src.execution.kucoin.utils import setup_logger


class KuCoinClient:
    """
    Main KuCoin REST API client.

    This client provides methods for interacting with the KuCoin API:
    - Market data: quotes, historical klines, order book depth
    - Account: account information, balances
    - Orders: place, cancel, query orders
    - WebSocket token acquisition

    API Docs: https://docs.kucoin.com/

    Attributes:
        base_url: Base URL for KuCoin API (sandbox or production)
        http_client: httpx AsyncClient instance
        signature_generator: SignatureGenerator for request signing
        api_key: KuCoin API key
        passphrase: KuCoin API passphrase

    Example:
        >>> async with KuCoinClient() as client:
        ...     quotes = await client.get_quotes("BTC-USDT")
        ...     account = await client.get_account_info()
        ...     klines = await client.get_historical_klines("BTC-USDT", "5min", 1000)
    """

    def __init__(self) -> None:
        """
        Initialize KuCoin client.

        Raises:
            Exception: If configuration is invalid or environment validation fails
        """
        self.logger = setup_logger(f"{__name__}.KuCoinClient")

        # Load configuration
        try:
            self.settings = load_kucoin_settings()
        except Exception as e:
            raise Exception(f"Failed to load KuCoin configuration: {e}") from e

        # Set base URL based on environment
        self.base_url = self.settings.base_url
        self.websocket_base_url = self.settings.websocket_base_url

        # Initialize authentication
        self.signature_generator = SignatureGenerator(
            api_secret=self.settings.kucoin_api_secret,
            passphrase=self.settings.kucoin_api_passphrase,
        )
        self.api_key = self.settings.kucoin_api_key
        self.passphrase = self.settings.kucoin_api_passphrase

        # HTTP client (initialized in __aenter__)
        self.http_client: httpx.AsyncClient | None = None

        # WebSocket token (cached)
        self._websocket_token: str | None = None
        self._websocket_token_expiry: datetime | None = None

        self.logger.info(
            f"KuCoinClient initialized (environment: {self.settings.kucoin_environment}, "
            f"base_url: {self.base_url})"
        )

    async def __aenter__(self) -> "KuCoinClient":
        """
        Enter context manager and initialize HTTP client.

        Returns:
            KuCoinClient instance

        Raises:
            Exception: If HTTP client initialization or connectivity test fails
        """
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
            )

            # Test connectivity
            await self._test_connectivity()

            self.logger.info("KuCoinClient connected successfully")

            return self

        except Exception as e:
            self.logger.error(f"Failed to initialize KuCoinClient: {e}")
            if self.http_client:
                await self.http_client.aclose()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager and close HTTP client.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if self.http_client:
            await self.http_client.aclose()
            self.logger.info("KuCoinClient connection closed")

    async def _test_connectivity(self) -> None:
        """
        Test API connectivity on startup.

        Fails fast if misconfigured or network issues.

        Raises:
            Exception: If connectivity test fails
        """
        try:
            # Ping endpoint (public, no auth required)
            response = await self.http_client.get("/api/v1/timestamp")
            response.raise_for_status()

            self.logger.info("Connectivity test passed")

        except httpx.HTTPStatusError as e:
            raise Exception(f"Connectivity test failed: HTTP {e.response.status_code}") from e
        except Exception as e:
            raise Exception(f"Connectivity test failed: {e}") from e

    async def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Make authenticated API request with proper headers.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/api/v1/orders")
            signed: Whether request requires signature
            params: Query parameters
            **kwargs: Additional arguments for httpx request

        Returns:
            JSON response data

        Raises:
            RateLimitError: If rate limit exceeded
            APIError: If request fails
            NetworkError: If network error occurs
        """
        if not self.http_client:
            raise Exception("KuCoinClient not initialized (use async context manager)")

        try:
            # Prepare headers
            if signed:
                headers = self.signature_generator.get_headers(
                    api_key=self.api_key,
                    method=method,
                    endpoint=endpoint,
                    params=params,
                )
            else:
                headers = {
                    "Content-Type": "application/json",
                }

            # Make request
            if params:
                response = await self.http_client.request(
                    method,
                    endpoint,
                    headers=headers,
                    params=params,
                    **kwargs,
                )
            else:
                response = await self.http_client.request(
                    method,
                    endpoint,
                    headers=headers,
                    **kwargs,
                )

            # Handle errors
            if response.status_code == 429:
                raise RateLimitError(
                    f"Rate limit exceeded: {response.text}",
                    status_code=response.status_code,
                )

            response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            raise APIError(
                f"API request failed: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}") from e

    # =============================================================================
    # Market Data Methods
    # =============================================================================

    async def get_quotes(self, symbol: str) -> KuCoinQuote:
        """
        Get ticker price change statistics for a symbol.

        API Docs: https://docs.kucoin.com/#get-ticker

        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")

        Returns:
            KuCoinQuote with ticker statistics

        Raises:
            ValidationError: If symbol is invalid
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                f"/api/v1/market/orderbook/level1",
                params={"symbol": symbol.upper()},
            )
            return KuCoinQuote(**response["data"])

        except Exception as e:
            self.logger.error(f"Failed to get quotes for {symbol}: {e}")
            raise

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[KuCoinKline]:
        """
        Get historical kline (candlestick) data.

        API Docs: https://docs.kucoin.com/#get-klines

        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            interval: Kline interval (e.g., "1min", "5min", "1hour", "1day")
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)
            limit: Number of klines to return (max 1000)

        Returns:
            List of KuCoinKline objects

        Raises:
            ValidationError: If parameters are invalid
            APIError: If request fails
        """
        try:
            params = {
                "symbol": symbol.upper(),
                "type": self._convert_kline_interval(interval),
            }

            if start_time:
                params["startAt"] = str(start_time)
            if end_time:
                params["endAt"] = str(end_time)

            response = await self._request(
                "GET",
                "/api/v1/klines/query",
                params=params,
            )

            # Parse klines
            klines = []
            for kline_data in response["data"]:
                klines.append(
                    KuCoinKline(
                        symbol=symbol.upper(),
                        interval=interval,
                        open_time=int(kline_data[0]),
                        open=float(kline_data[1]),
                        high=float(kline_data[2]),
                        low=float(kline_data[3]),
                        close=float(kline_data[4]),
                        volume=float(kline_data[5]),
                        close_time=int(kline_data[6]),
                        quote_volume=float(kline_data[7]),
                    )
                )

            return klines

        except Exception as e:
            self.logger.error(f"Failed to get historical klines for {symbol}: {e}")
            raise

    async def get_order_book_depth(
        self,
        symbol: str,
        limit: int = 100,
    ) -> KuCoinOrderBook:
        """
        Get order book depth for a symbol (for liquidity filter).

        API Docs: https://docs.kucoin.com/#get-part-order-book-aggregated

        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            limit: Number of bids/asks to return (20 or 100)

        Returns:
            KuCoinOrderBook with bids and asks

        Raises:
            ValidationError: If parameters are invalid
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v3/market/book",
                params={
                    "symbol": symbol.upper(),
                    "limit": limit,
                },
            )
            return KuCoinOrderBook(symbol=symbol.upper(), **response["data"])

        except Exception as e:
            self.logger.error(f"Failed to get order book depth for {symbol}: {e}")
            raise

    # =============================================================================
    # Account Methods
    # =============================================================================

    async def get_account_info(self) -> list[KuCoinAccount]:
        """
        Get account information (signed endpoint).

        API Docs: https://docs.kucoin.com/#get-account-list-spot-margin-trade-hf

        Returns:
            List of KuCoinAccount with account information

        Raises:
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v1/accounts",
                signed=True,
            )
            return [KuCoinAccount(**acc) for acc in response["data"]]

        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise

    # =============================================================================
    # Order Methods
    # =============================================================================

    async def get_order_status(self, order_id: str) -> KuCoinOrder:
        """
        Get order status by order ID (signed endpoint).

        API Docs: https://docs.kucoin.com/#get-order-details-by-orderid

        Args:
            order_id: Order ID to query

        Returns:
            KuCoinOrder with order status

        Raises:
            OrderError: If order not found
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                f"/api/v1/orders/{order_id}",
                signed=True,
            )
            return KuCoinOrder(**response["data"])

        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            raise

    async def get_websocket_token(self) -> str:
        """
        Get WebSocket connection token for public channels.

        API Docs: https://docs.kucoin.com/#apply-connect-token

        Returns:
            WebSocket token for connection

        Raises:
            APIError: If request fails
        """
        # Check if token is still valid
        if self._websocket_token and self._websocket_token_expiry:
            if datetime.now(timezone.utc) < self._websocket_token_expiry:
                return self._websocket_token

        try:
            response = await self._request(
                "POST",
                "/api/v1/bullet-public",
                signed=False,
            )

            token = response["data"]["token"]
            servers = response["data"]["instanceServers"]

            # Token expires after 24 hours
            self._websocket_token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
            self._websocket_token = token

            self.logger.info("WebSocket token acquired successfully")
            return token

        except Exception as e:
            self.logger.error(f"Failed to get WebSocket token: {e}")
            raise

    def _convert_kline_interval(self, interval: str) -> str:
        """
        Convert interval string to KuCoin format.

        Args:
            interval: Interval string (e.g., "5m", "1h", "1d")

        Returns:
            KuCoin interval format

        Example:
            >>> client._convert_kline_interval("5m")
            '5min'
            >>> client._convert_kline_interval("1h")
            '1hour'
        """
        interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1hour",
            "4h": "4hour",
            "1d": "1day",
        }

        return interval_map.get(interval.lower(), "5min")

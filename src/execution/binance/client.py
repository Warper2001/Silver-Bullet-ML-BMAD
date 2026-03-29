"""
Binance REST API Client

This module provides the main BinanceClient for interacting with the Binance REST API.
It handles authentication, rate limiting, error handling, and request signing.

API Documentation: https://binance-docs.github.io/apidocs/

Architecture:
- Async context manager for connection lifecycle
- Weight-based rate limiting (1200 weight/minute)
- HMAC SHA256 signature for signed endpoints
- Comprehensive error handling with custom exceptions
"""

import logging
from datetime import datetime, timezone

import httpx

from src.data.crypto_config import load_crypto_settings
from src.execution.binance.auth import ApiKeyAuth
from src.execution.binance.exceptions import APIError, NetworkError, RateLimitError, ValidationError
from src.execution.binance.models import (
    BinanceAccount,
    BinanceKline,
    BinanceOrder,
    BinanceOrderBook,
    BinanceQuote,
)
from src.execution.binance.utils import WeightBasedRateLimitTracker, setup_logger


class BinanceClient:
    """
    Main Binance REST API client.

    This client provides methods for interacting with the Binance API:
    - Market data: quotes, historical klines, order book depth
    - Account: account information, balances
    - Orders: place, cancel, query orders
    - WebSocket listen key management

    API Docs: https://binance-docs.github.io/apidocs/

    Attributes:
        base_url: Base URL for Binance API (testnet or production)
        http_client: httpx AsyncClient instance
        auth: ApiKeyAuth instance for authentication
        rate_limiter: WeightBasedRateLimitTracker for rate limiting

    Example:
        >>> async with BinanceClient() as client:
        ...     quotes = await client.get_quotes("BTCUSDT")
        ...     account = await client.get_account_info()
        ...     klines = await client.get_historical_klines("BTCUSDT", "5m", 1000)
    """

    def __init__(self) -> None:
        """
        Initialize Binance client.

        Raises:
            Exception: If configuration is invalid or environment validation fails
        """
        self.logger = setup_logger(f"{__name__}.BinanceClient")

        # Load configuration
        try:
            self.settings = load_crypto_settings()
        except Exception as e:
            raise Exception(f"Failed to load crypto configuration: {e}") from e

        # Set base URL based on environment
        self.base_url = self.settings.base_url
        self.websocket_base_url = self.settings.websocket_base_url

        # Validate environment on startup
        self._validate_environment()

        # Initialize authentication
        self.auth = ApiKeyAuth(
            api_key=self.settings.crypto_exchange_api_key,
            api_secret=self.settings.crypto_exchange_api_secret,
        )

        # Initialize rate limiter (weight-based: 1200 weight/minute for production)
        self.rate_limiter = WeightBasedRateLimitTracker(
            weight_limit_per_minute=1200
        )

        # HTTP client (initialized in __aenter__)
        self.http_client: httpx.AsyncClient | None = None

        self.logger.info(
            f"BinanceClient initialized (environment: {self.settings.crypto_exchange_environment}, "
            f"base_url: {self.base_url})"
        )

    def _validate_environment(self) -> None:
        """
        Validate environment configuration on startup.

        This ensures the base URL matches the configured environment
        and fails fast if misconfigured.

        Raises:
            Exception: If environment validation fails
        """
        # Check environment is valid
        valid_environments = ["testnet", "production"]
        if self.settings.crypto_exchange_environment not in valid_environments:
            raise Exception(
                f"Invalid environment '{self.settings.crypto_exchange_environment}'. "
                f"Must be one of: {valid_environments}"
            )

        # Verify base URL matches environment
        expected_urls = {
            "testnet": "https://testnet.binance.vision",
            "production": "https://api.binance.com",
        }

        expected_url = expected_urls[self.settings.crypto_exchange_environment]
        if self.base_url != expected_url:
            raise Exception(
                f"Base URL mismatch: expected '{expected_url}' for "
                f"'{self.settings.crypto_exchange_environment}', got '{self.base_url}'"
            )

        self.logger.info(f"Environment validation passed: {self.settings.crypto_exchange_environment}")

    async def __aenter__(self) -> "BinanceClient":
        """
        Enter context manager and initialize HTTP client.

        Returns:
            BinanceClient instance

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

            self.logger.info("BinanceClient connected successfully")

            return self

        except Exception as e:
            self.logger.error(f"Failed to initialize BinanceClient: {e}")
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
            self.logger.info("BinanceClient connection closed")

    async def _test_connectivity(self) -> None:
        """
        Test API connectivity on startup.

        Fails fast if misconfigured or network issues.

        Raises:
            Exception: If connectivity test fails
        """
        try:
            # Ping endpoint (public, no auth required)
            response = await self.http_client.get("/api/v3/ping")
            response.raise_for_status()

            # Check server time (public, no auth required)
            response = await self.http_client.get("/api/v3/time")
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
        weight: int = 1,
        **kwargs,
    ) -> dict:
        """
        Make authenticated API request with rate limiting.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/api/v3/account")
            signed: Whether request requires signature
            weight: Request weight for rate limiting
            **kwargs: Additional arguments for httpx request

        Returns:
            JSON response data

        Raises:
            RateLimitError: If rate limit exceeded
            APIError: If request fails
            NetworkError: If network error occurs
        """
        if not self.http_client:
            raise Exception("BinanceClient not initialized (use async context manager)")

        # Acquire rate limit slot
        await self.rate_limiter.acquire_weight(weight)

        # Prepare headers
        headers = self.auth.get_headers()

        # Add signature for signed endpoints
        if signed:
            # Prepare query parameters
            params = kwargs.get("params", {})
            signature = self.auth.sign_request(params)
            params["signature"] = signature
            kwargs["params"] = params

        try:
            # Make request
            response = await self.http_client.request(method, endpoint, headers=headers, **kwargs)

            # Update rate limiter from response headers
            self.rate_limiter.update_from_headers(dict(response.headers))

            # Handle errors
            if response.status_code == 418 or response.status_code == 429:
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

    async def get_quotes(self, symbol: str) -> BinanceQuote:
        """
        Get 24hr ticker price change statistics for a symbol.

        API Docs: https://binance-docs.github.io/apidocs/#24hr-ticker-price-change-statistics

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            BinanceQuote with 24hr statistics

        Raises:
            ValidationError: If symbol is invalid
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v3/ticker/24hr",
                params={"symbol": symbol.upper()},
            )
            return BinanceQuote(**response)

        except Exception as e:
            self.logger.error(f"Failed to get quotes for {symbol}: {e}")
            raise

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
    ) -> list[BinanceKline]:
        """
        Get historical kline (candlestick) data.

        API Docs: https://binance-docs.github.io/apidocs/#kline-candlestick-data

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "15m", "1h", "1d")
            limit: Number of klines to return (max 1000, default 1000)

        Returns:
            List of BinanceKline objects

        Raises:
            ValidationError: If parameters are invalid
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v3/klines",
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "limit": limit,
                },
            )

            # Parse klines
            klines = []
            for kline_data in response:
                klines.append(
                    BinanceKline(
                        symbol=symbol.upper(),
                        interval=interval,
                        open_time=kline_data[0],
                        open=float(kline_data[1]),
                        high=float(kline_data[2]),
                        low=float(kline_data[3]),
                        close=float(kline_data[4]),
                        volume=float(kline_data[5]),
                        close_time=kline_data[6],
                        quote_volume=float(kline_data[7]),
                        trades=int(kline_data[8]),
                        taker_buy_base_volume=float(kline_data[9]),
                        taker_buy_quote_volume=float(kline_data[10]),
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
    ) -> BinanceOrderBook:
        """
        Get order book depth for a symbol (for liquidity filter).

        API Docs: https://binance-docs.github.io/apidocs/#order-book-depth

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            limit: Number of bids/asks to return (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            BinanceOrderBook with bids and asks

        Raises:
            ValidationError: If parameters are invalid
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v3/depth",
                params={
                    "symbol": symbol.upper(),
                    "limit": limit,
                },
            )
            return BinanceOrderBook(symbol=symbol.upper(), **response)

        except Exception as e:
            self.logger.error(f"Failed to get order book depth for {symbol}: {e}")
            raise

    # =============================================================================
    # Account Methods
    # =============================================================================

    async def get_account_info(self) -> BinanceAccount:
        """
        Get account information (signed endpoint).

        API Docs: https://binance-docs.github.io/apidocs/#account-information-user_data

        Returns:
            BinanceAccount with account information

        Raises:
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v3/account",
                signed=True,
                weight=10,  # Account endpoint has weight 10
            )
            return BinanceAccount(**response)

        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise

    # =============================================================================
    # Order Methods
    # =============================================================================

    async def get_order_status(self, symbol: str, order_id: int) -> BinanceOrder:
        """
        Get order status by order ID (signed endpoint).

        API Docs: https://binance-docs.github.io/apidocs/#query-order-user_data

        Args:
            symbol: Trading symbol
            order_id: Order ID

        Returns:
            BinanceOrder with order status

        Raises:
            OrderError: If order not found
            APIError: If request fails
        """
        try:
            response = await self._request(
                "GET",
                "/api/v3/order",
                signed=True,
                params={
                    "symbol": symbol.upper(),
                    "orderId": order_id,
                },
            )
            return BinanceOrder(**response)

        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            raise

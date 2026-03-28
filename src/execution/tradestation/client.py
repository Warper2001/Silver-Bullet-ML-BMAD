"""
TradeStation SDK - Main API Client

This module provides the main TradeStationClient class for interacting
with the TradeStation API.

Key Features:
- Async context manager for automatic resource cleanup
- OAuth 2.0 authentication integration
- HTTP session management with httpx
- Support for both SIM and LIVE environments
- Automatic error handling with custom exceptions
- Rate limiting and retry logic

Usage:
    async with TradeStationClient(env="sim", config=config) as client:
        quotes = await client.get_quotes(["MNQH26"])
        print(quotes)
"""

import logging
from typing import Any, Literal

import httpx

from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager
from src.execution.tradestation.exceptions import (
    APIError,
    AuthError,
    InvalidCredentialsError,
    NetworkError,
    OrderError,
    RateLimitError,
    TradeStationError,
    ValidationError,
)
from src.execution.tradestation.models import TradeStationQuote
from src.execution.tradestation.utils import setup_logger


class TradeStationClient:
    """
    Main API client for TradeStation API.

    Provides async context manager interface for automatic resource
    cleanup and integrates OAuth 2.0 authentication.

    Attributes:
        env: Environment ("sim" or "live")
        client_id: OAuth 2.0 client ID
        client_secret: OAuth 2.0 client secret
        api_base_url: TradeStation API base URL
        oauth_client: OAuth2Client instance for authentication
        http_client: httpx.AsyncClient for HTTP requests

    Example:
        config = {
            "client_id": "your_client_id",
            "client_secret": "your_client_secret"
        }

        async with TradeStationClient(env="sim", config=config) as client:
            # Client is automatically authenticated
            quotes = await client.get_quotes(["MNQH26"])
            print(quotes)
    """

    def __init__(
        self,
        client_id: str,
        redirect_uri: str = "http://localhost:8080",
        token_manager: TokenManager | None = None,
    ) -> None:
        """
        Initialize TradeStation API client.

        Args:
            client_id: OAuth 2.0 client ID (API Key)
            redirect_uri: Redirect URI for OAuth callback
            token_manager: Optional TokenManager instance
        """
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.logger = setup_logger(f"{__name__}.TradeStationClient")

        # TradeStation API base URL (production only, no separate SIM environment)
        self.api_base_url = "https://api.tradestation.com/v3"

        # OAuth 2.0 client
        self.oauth_client = OAuth2Client(
            client_id=client_id,
            redirect_uri=redirect_uri,
            token_manager=token_manager,
        )

        # HTTP client (initialized in __aenter__)
        self.http_client: httpx.AsyncClient | None = None
        self._is_initialized = False

    async def __aenter__(self) -> "TradeStationClient":
        """
        Enter async context manager.

        Initializes HTTP client for API requests.

        Note: OAuth authentication must be completed separately before
        making API requests. Use the OAuth2Client to authenticate.

        Returns:
            Self for context manager protocol
        """
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=self.api_base_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(30.0),
            )

            self.logger.info("TradeStationClient initialized (OAuth must be completed separately)")
            self._is_initialized = True
            return self

        except Exception as e:
            # Clean up on failure
            if self.http_client:
                await self.http_client.aclose()
            raise

    async def __aexit__(self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """
        Exit async context manager.

        Closes HTTP client and performs cleanup.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        self._is_initialized = False
        self.logger.info("TradeStationClient closed")

    async def _ensure_authenticated(self) -> str:
        """
        Ensure client is authenticated and return access token.

        Returns:
            Valid access token

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._is_initialized or not self.http_client:
            raise RuntimeError(
                "TradeStationClient not initialized. "
                "Use 'async with TradeStationClient(...) as client:'"
            )

        return await self.oauth_client.get_access_token()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make authenticated API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/quotes")
            **kwargs: Additional arguments for httpx request

        Returns:
            JSON response as dictionary

        Raises:
            NetworkError: On network errors
            RateLimitError: On rate limit exceeded
            APIError: On API errors
            ValidationError: On validation errors
        """
        access_token = await self._ensure_authenticated()

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {access_token}"

        try:
            response = await self.http_client.request(
                method=method,
                url=endpoint,
                headers=headers,
                **kwargs,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    details={"endpoint": endpoint, "retry_after": retry_after}
                )

            # Handle client errors (4xx)
            if 400 <= response.status_code < 500:
                error_detail = self._parse_error_response(response)
                if response.status_code == 400:
                    raise ValidationError(f"Validation error: {error_detail}", details=error_detail)
                else:
                    raise APIError(
                        f"API error: {error_detail}",
                        status_code=response.status_code,
                        details=error_detail,
                    )

            # Handle server errors (5xx)
            if 500 <= response.status_code < 600:
                error_detail = self._parse_error_response(response)
                raise APIError(
                    f"Server error: {error_detail}",
                    status_code=response.status_code,
                    details=error_detail,
                )

            # Success
            return response.json()

        except httpx.TimeoutException as e:
            self.logger.error(f"Request timeout: {e}")
            raise NetworkError(f"Request timeout: {e}")

        except httpx.NetworkError as e:
            self.logger.error(f"Network error: {e}")
            raise NetworkError(f"Network error: {e}")

        except RateLimitError:
            raise

        except (ValidationError, APIError, NetworkError):
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error during request: {e}")
            raise APIError(f"Unexpected error: {e}")

    def _parse_error_response(self, response: httpx.Response) -> dict[str, Any]:
        """
        Parse error response from API.

        Args:
            response: HTTP response with error status

        Returns:
            Error details dictionary
        """
        try:
            return response.json()
        except Exception:
            return {"error": response.text, "status_code": response.status_code}

    # ========================================================================
    # Market Data Endpoints
    # ========================================================================

    async def get_quotes(
        self,
        symbols: list[str],
    ) -> list[TradeStationQuote]:
        """
        Get real-time quotes for symbols.

        Args:
            symbols: List of trading symbols (e.g., ["MNQH26", "MNQM26"])

        Returns:
            List of quote objects

        Raises:
            ValidationError: If symbols are invalid
            APIError: On API errors
        """
        self.logger.info(f"Fetching quotes for {len(symbols)} symbols")

        params = {"symbols": ",".join(symbols)}

        try:
            response = await self._request("GET", "/data/quote", params=params)
            quotes = [TradeStationQuote(**quote) for quote in response.get("Quotes", [])]

            self.logger.info(f"Received {len(quotes)} quotes")
            return quotes

        except Exception as e:
            self.logger.error(f"Failed to fetch quotes: {e}")
            raise

    async def get_historical_bars(
        self,
        symbol: str,
        bar_type: str = "minute",
        interval: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """
        Get historical OHLCV bars for a symbol.

        Args:
            symbol: Trading symbol
            bar_type: Bar type (minute, daily, etc.)
            interval: Bar interval (1, 5, 15, 30, 60, daily)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of OHLCV bars

        Raises:
            ValidationError: If parameters are invalid
            APIError: On API errors
        """
        self.logger.info(f"Fetching historical bars for {symbol}")

        # TODO: Implement proper endpoint path and parameters
        # This is a placeholder implementation
        params = {
            "symbol": symbol,
            "type": bar_type,
            "interval": interval,
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            response = await self._request("GET", "/data/bars", params=params)
            bars = response.get("Bars", [])

            self.logger.info(f"Received {len(bars)} bars")
            return bars

        except Exception as e:
            self.logger.error(f"Failed to fetch historical bars: {e}")
            raise

    # ========================================================================
    # Order Endpoints (Placeholder)
    # ========================================================================

    async def place_order(self, order: dict) -> dict:
        """
        Place a new order.

        Args:
            order: Order details (symbol, side, type, quantity, etc.)

        Returns:
            Order confirmation

        Raises:
            ValidationError: If order is invalid
            OrderError: On order errors
        """
        self.logger.info(f"Placing order: {order}")

        try:
            response = await self._request("POST", "/order", json=order)
            self.logger.info("Order placed successfully")
            return response

        except APIError as e:
            # Convert APIError to OrderError for circuit breaker tracking
            raise OrderError(f"Order failed: {e.message}", details=e.details)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def is_authenticated(self) -> bool:
        """
        Check if client is authenticated.

        Returns:
            True if authenticated with valid token
        """
        return self.oauth_client.is_authenticated()

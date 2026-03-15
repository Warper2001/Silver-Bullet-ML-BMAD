"""TradeStation OAuth 2.0 authentication."""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
from pydantic import BaseModel, ValidationError

from .config import load_settings
from .exceptions import AuthenticationError, TokenRefreshError

logger = logging.getLogger(__name__)


class TokenResponse(BaseModel):
    """TradeStation OAuth 2.0 token response."""

    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str
    scope: str

    @property
    def expires_at(self) -> datetime:
        """Calculate token expiration datetime.

        Returns:
            Datetime when token expires
        """
        return datetime.now() + timedelta(seconds=self.expires_in)

    @property
    def token_hash(self) -> str:
        """Get SHA256 hash of access token for logging (safe, no exposure).

        Returns:
            First 16 characters of token hash
        """
        return hashlib.sha256(self.access_token.encode()).hexdigest()[:16]


class TradeStationAuth:
    """TradeStation OAuth 2.0 authentication manager.

    Handles:
    - Initial authentication with client credentials
    - Automatic token refresh every 10 minutes
    - Exponential backoff retry on failures
    - Secure in-memory token storage
    """

    TOKEN_ENDPOINT = "https://api.tradestation.com/v2/security/authorize"
    REFRESH_INTERVAL_SECONDS = 600  # 10 minutes
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

    def __init__(self) -> None:
        """Initialize authentication manager."""
        self.settings = load_settings()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self) -> str:
        """Authenticate with TradeStation API and return access token.

        Returns:
            Access token for API requests

        Raises:
            TokenRefreshError: If authentication fails after all retries
        """
        if self._is_token_valid():
            logger.debug("Using cached access token")
            return self._access_token  # type: ignore

        await self._perform_authentication()
        return self._access_token  # type: ignore

    async def start_refresh_task(self) -> None:
        """Start background token refresh task."""
        if self._refresh_task is not None and not self._refresh_task.done():
            logger.warning("Token refresh task already running")
            return

        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("Token refresh task started")

    async def stop_refresh_task(self) -> None:
        """Stop background token refresh task."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                logger.info("Token refresh task stopped")
            self._refresh_task = None

    async def _refresh_loop(self) -> None:
        """Background task to refresh token periodically."""
        while True:
            try:
                await asyncio.sleep(self.REFRESH_INTERVAL_SECONDS)
                await self._perform_authentication()
                logger.info("Token refreshed successfully")
            except asyncio.CancelledError:
                logger.info("Token refresh loop cancelled")
                break
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                # Continue loop - will retry on next interval

    async def _perform_authentication(self) -> None:
        """Perform authentication with exponential backoff retry.

        Raises:
            TokenRefreshError: If authentication fails after all retries
        """
        last_error: Exception | None = None

        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                token_response = await self._request_token()
                self._store_tokens(token_response)
                logger.info(
                    f"Authentication successful (token hash: {token_response.token_hash})"
                )
                return

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"Authentication attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS} "
                    f"failed: {e.response.status_code}"
                )
                if attempt < len(self.RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Authentication attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS} "
                    f"failed: {str(e)}"
                )
                if attempt < len(self.RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise TokenRefreshError(
            f"Authentication failed after {self.MAX_RETRY_ATTEMPTS} attempts",
            retry_count=self.MAX_RETRY_ATTEMPTS,
            original_error=last_error,
        )

    async def _request_token(self) -> TokenResponse:
        """Request token from TradeStation API.

        Returns:
            Token response with access and refresh tokens

        Raises:
            httpx.HTTPStatusError: If API returns error status
            httpx.RequestError: If network error occurs
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        data = {
            "grant_type": "client_credentials",
            "client_id": self.settings.tradestation_client_id,
            "client_secret": self.settings.tradestation_client_secret,
            "redirect_uri": self.settings.tradestation_redirect_uri,
        }

        response = await self._client.post(
            self.TOKEN_ENDPOINT,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        response.raise_for_status()

        token_data = response.json()
        try:
            return TokenResponse(**token_data)
        except ValidationError as e:
            raise AuthenticationError(f"Invalid token response: {e}") from e

    def _store_tokens(self, token_response: TokenResponse) -> None:
        """Store tokens in memory (never persist to disk).

        Args:
            token_response: Token response from API
        """
        self._access_token = token_response.access_token
        self._refresh_token = token_response.refresh_token
        self._token_expires_at = token_response.expires_at

    def _is_token_valid(self) -> bool:
        """Check if current token is valid and not expired.

        Returns:
            True if token exists and is valid, False otherwise
        """
        if self._access_token is None:
            return False

        if self._token_expires_at is None:
            return False

        # Add 5-minute buffer before expiration
        return datetime.now() < (self._token_expires_at - timedelta(minutes=5))

    async def cleanup(self) -> None:
        """Clean up resources (close HTTP client, cancel tasks)."""
        await self.stop_refresh_task()

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        # Clear tokens from memory
        self._access_token = None
        self._refresh_token = None
        self._token_expires_at = None

        logger.info("Authentication resources cleaned up")

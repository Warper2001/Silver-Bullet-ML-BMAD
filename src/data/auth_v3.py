"""TradeStation API v3 authentication using existing OAuth tokens.

This module implements a v3-only authentication class that loads existing
OAuth tokens from file (obtained via browser flow) and provides token
refresh functionality without requiring browser interaction.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
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
    refresh_token: str | None = None  # Optional - not returned in refresh response
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


class TradeStationAuthV3:
    """TradeStation API v3 authentication using existing OAuth tokens.

    This class loads existing v3 OAuth tokens from file and provides
    authentication without requiring browser interaction. It automatically
    refreshes tokens when they expire.

    Token endpoints:
    - Token refresh: https://signin.tradestation.com/oauth/token

    Example:
        Load token from file:
        auth = TradeStationAuthV3.from_file(".access_token")

        Or provide tokens directly:
        auth = TradeStationAuthV3(access_token="...", refresh_token="...")
    """

    TOKEN_ENDPOINT = "https://signin.tradestation.com/oauth/token"
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

    def __init__(
        self,
        access_token: str,
        refresh_token: str = "",
        token_expires_at: Optional[datetime] = None,
    ) -> None:
        """Initialize authentication with existing tokens.

        Args:
            access_token: Valid v3 access token
            refresh_token: Refresh token (empty string if not available)
            token_expires_at: Token expiration datetime (None if unknown)
        """
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expires_at = token_expires_at
        self._client: Optional[httpx.AsyncClient] = None
        self.settings = load_settings()
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._should_stop_refreshing = False

        logger.info(
            f"TradeStationAuthV3 initialized (token hash: {self._get_token_hash()})"
        )

    async def authenticate(self) -> str:
        """Get access token, refreshing if necessary.

        Returns:
            Valid access token for API requests

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check if current token is valid
        if self._is_token_valid():
            logger.debug("Using cached access token")
            return self._access_token

        # If we have a refresh token, use it
        if self._refresh_token:
            try:
                await self._refresh_token_flow()
                return self._access_token
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                raise AuthenticationError(
                    f"Token refresh failed: {e}", retry_count=self.MAX_RETRY_ATTEMPTS
                ) from e

        # No valid token and no refresh token
        raise AuthenticationError(
            "No valid access token and no refresh token available. "
            "Please run OAuth flow to obtain new tokens."
        )

    async def _refresh_token_flow(self) -> None:
        """Refresh access token using refresh token with exponential backoff.

        Raises:
            TokenRefreshError: If refresh fails after all retries
        """
        if not self._refresh_token:
            raise AuthenticationError("No refresh token available")

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        last_error: Optional[Exception] = None

        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                logger.debug(f"Token refresh attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}")

                data = {
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": self.settings.tradestation_client_id,
                    "client_secret": self.settings.tradestation_client_secret,
                }

                response = await self._client.post(
                    self.TOKEN_ENDPOINT,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                response.raise_for_status()

                token_data = response.json()
                token_response = TokenResponse(**token_data)

                # Store tokens
                self._access_token = token_response.access_token
                # Keep existing refresh token if not returned in response
                if token_response.refresh_token:
                    self._refresh_token = token_response.refresh_token
                self._token_expires_at = token_response.expires_at

                logger.info(
                    f"Token refreshed successfully (token hash: {token_response.token_hash}), "
                    f"expires at {self._token_expires_at}"
                )

                return  # Success, exit retry loop

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"Token refresh attempt {attempt + 1} failed: "
                    f"HTTP {e.response.status_code}"
                )

                # Don't retry on client errors (4xx) except 429 (rate limit)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise TokenRefreshError(
                        f"Token refresh failed with HTTP {e.response.status_code}: "
                        f"{e.response.text}",
                        retry_count=attempt + 1,
                        original_error=e,
                    )

                # Retry on server errors (5xx) and rate limit (429)
                if attempt < len(self.RETRY_DELAYS) - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

            except (ValidationError, ValueError) as e:
                last_error = e
                logger.error(f"Invalid token response: {e}")
                raise TokenRefreshError(
                    f"Invalid token response: {e}",
                    retry_count=attempt + 1,
                    original_error=e,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"Token refresh attempt {attempt + 1} failed: {e}")
                if attempt < len(self.RETRY_DELAYS) - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        assert last_error is not None
        raise TokenRefreshError(
            f"Token refresh failed after {self.MAX_RETRY_ATTEMPTS} attempts",
            retry_count=self.MAX_RETRY_ATTEMPTS,
            original_error=last_error,
        )

    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid token.

        Returns:
            True if token exists and is valid, False otherwise
        """
        return self._is_token_valid()

    def _is_token_valid(self) -> bool:
        """Check if current token is valid and not expired.

        Returns:
            True if token exists and is valid, False otherwise
        """
        if not self._access_token:
            return False

        if self._token_expires_at is None:
            # No expiration info, assume token is valid
            return True

        # Add 5-minute buffer before expiration
        return datetime.now() < (self._token_expires_at - timedelta(minutes=5))

    def _get_token_hash(self) -> str:
        """Get safe hash of current access token for logging.

        Returns:
            First 16 characters of token hash
        """
        if not self._access_token:
            return "None"
        return hashlib.sha256(self._access_token.encode()).hexdigest()[:16]

    @classmethod
    def from_file(cls, token_file: str = ".access_token") -> "TradeStationAuthV3":
        """Load authentication from token file.

        Args:
            token_file: Path to token file (default: ".access_token")

        Returns:
            TradeStationAuthV3 instance with loaded token

        Raises:
            AuthenticationError: If token file cannot be read
        """
        token_path = Path(token_file)

        if not token_path.exists():
            raise AuthenticationError(
                f"Token file not found: {token_file}. "
                "Please run OAuth flow to obtain tokens."
            )

        try:
            access_token = token_path.read_text().strip()
            logger.debug(f"Loaded access token from {token_file}")

            # Try to load refresh token from environment
            try:
                settings = load_settings()
                refresh_token = settings.tradestation_refresh_token
                logger.debug("Loaded refresh token from environment")
            except Exception:
                refresh_token = ""
                logger.debug("No refresh token in environment")

            return cls(
                access_token=access_token,
                refresh_token=refresh_token,
                token_expires_at=None,  # Will be determined on first use
            )

        except Exception as e:
            raise AuthenticationError(
                f"Failed to load token from {token_file}: {e}"
            ) from e

    async def cleanup(self) -> None:
        """Clean up resources (close HTTP client, stop background refresh)."""
        # Stop background refresh task
        if self._refresh_task is not None:
            self._should_stop_refreshing = True
            try:
                self._refresh_task.cancel()
                await asyncio.sleep(0.5)  # Give task time to cancel
            except Exception:
                pass
            self._refresh_task = None

        # Close HTTP client
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        logger.info("Authentication resources cleaned up")

    async def start_auto_refresh(self, interval_minutes: int = 10) -> None:
        """Start automatic token refresh every 10 minutes.

        This runs a background task that refreshes the access token at regular
        intervals to prevent authentication failures.

        Args:
            interval_minutes: Refresh interval in minutes (default: 10)
        """
        if self._refresh_task is not None:
            logger.warning("Auto-refresh task already running")
            return

        self._should_stop_refreshing = False
        self._refresh_task = asyncio.create_task(
            self._auto_refresh_loop(interval_minutes)
        )
        logger.info(f"Started auto-refresh task (every {interval_minutes} minutes)")

    async def _auto_refresh_loop(self, interval_minutes: int) -> None:
        """Background task loop for automatic token refresh.

        Args:
            interval_minutes: Refresh interval in minutes
        """
        interval_seconds = interval_minutes * 60

        while not self._should_stop_refreshing:
            try:
                await asyncio.sleep(interval_seconds)

                if self._should_stop_refreshing:
                    break

                logger.info("Running scheduled token refresh...")
                await self._refresh_token_flow()
                logger.info("✅ Scheduled token refresh completed")

                # Update token file
                try:
                    with open(".access_token", "w") as f:
                        f.write(self._access_token)
                    logger.debug("Updated .access_token file with refreshed token")
                except Exception as e:
                    logger.error(f"Failed to update token file: {e}")

            except asyncio.CancelledError:
                logger.info("Auto-refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Auto-refresh failed: {e}")
                # Continue the loop despite refresh failures

        logger.info("Auto-refresh loop stopped")

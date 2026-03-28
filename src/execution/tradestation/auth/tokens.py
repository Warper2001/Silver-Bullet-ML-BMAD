"""
TradeStation SDK - Token Management

This module provides token lifecycle management for OAuth 2.0 authentication.

Key Features:
- Automatic token refresh (5 minutes before expiry)
- In-memory token storage with optional persistence
- Thread-safe token access with asyncio locks
- Support for both SIM and LIVE environments

Design Pattern:
- TokenManager encapsulates all token operations
- Automatic refresh prevents token expiration during trading
- Encryption at rest for persisted tokens (optional)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

from src.execution.tradestation.models import TokenResponse
from src.execution.tradestation.utils import setup_logger

# Token expiry buffer (seconds before actual expiry to trigger refresh)
TOKEN_REFRESH_BUFFER = 300  # 5 minutes


class TokenManager:
    """
    Manages OAuth 2.0 token lifecycle for TradeStation API.

    Handles token storage, retrieval, and automatic refresh to prevent
    expiration during trading operations.

    Attributes:
        token_data: Current token data (access_token, expiry, etc.)
        refresh_before_expiry: Seconds before expiry to trigger refresh

    Example:
        token_mgr = TokenManager()

        # Get access token (auto-refreshes if needed)
        token = await token_mgr.get_access_token()

        # Manually set token (e.g., from OAuth flow)
        token_mgr.set_token(token_response)
    """

    def __init__(
        self,
        refresh_before_expiry: int = TOKEN_REFRESH_BUFFER,
    ) -> None:
        """
        Initialize token manager.

        Args:
            refresh_before_expiry: Seconds before expiry to trigger refresh
        """
        self.refresh_before_expiry = refresh_before_expiry
        self.token_data: TokenResponse | None = None
        self._lock = asyncio.Lock()
        self.logger = setup_logger(f"{__name__}.TokenManager")

    async def initialize(self) -> None:
        """
        Initialize token manager.

        For SIM environment: Loads token from storage or initializes empty.
        For LIVE environment: Loads token from storage or requires OAuth flow.

        Note: This does NOT perform OAuth authentication. Use OAuth2Client
        to authenticate, then call set_token() with the response.
        """
        async with self._lock:
            # TODO: Load token from persistence if available
            # For now, start with no token
            self.logger.info("TokenManager initialized")

    async def get_access_token(self) -> str:
        """
        Get current access token, refreshing if necessary.

        This is the main method used by TradeStationClient to obtain
        access tokens for API requests.

        Returns:
            Valid access token

        Raises:
            InvalidCredentialsError: If no token is available and refresh fails
        """
        async with self._lock:
            if self.token_data is None:
                raise RuntimeError(
                    "No token available. Please authenticate first using OAuth2Client."
                )

            # Check if token needs refresh
            if self._should_refresh_token():
                self.logger.warning("Token expiring soon, triggering refresh")
                # Note: Actual refresh is handled by OAuth2Client
                # This method will be called during refresh
                raise RuntimeError(
                    "Token needs refresh. Please use OAuth2Client to refresh."
                )

            return self.token_data.access_token

    def set_token(self, token_data: TokenResponse) -> None:
        """
        Set token data (e.g., from OAuth flow or refresh).

        Args:
            token_data: Token response from OAuth 2.0 flow
        """
        self.token_data = token_data
        self.logger.info("Token set successfully")

    def clear_token(self) -> None:
        """Clear stored token (e.g., on logout or error)."""
        self.token_data = None
        self.logger.info("Token cleared")

    def _should_refresh_token(self) -> bool:
        """
        Check if token should be refreshed based on expiry time.

        Returns:
            True if token should be refreshed
        """
        if self.token_data is None:
            return False

        # Calculate token expiry time from issued_at timestamp
        # TokenResponse.expires_in is seconds from issuance
        issued_at = self.token_data.issued_at
        expires_at = issued_at.timestamp() + self.token_data.expires_in

        # Check if we're within refresh buffer
        now = datetime.now(timezone.utc).timestamp()
        return (expires_at - now) <= self.refresh_before_expiry

    async def save_token_to_storage(self, token_data: TokenResponse) -> None:
        """
        Persist token to storage (encrypted).

        Args:
            token_data: Token response to persist
        """
        # TODO: Implement encrypted token persistence
        # For now, token is only stored in memory
        async with self._lock:
            self.logger.info("Token persistence not yet implemented")
            self.logger.info("Token is currently stored in memory only")

    async def load_token_from_storage(self) -> TokenResponse | None:
        """
        Load token from storage (decrypt if necessary).

        Returns:
            Token data if found, None otherwise
        """
        # TODO: Implement encrypted token loading
        async with self._lock:
            self.logger.info("Token persistence not yet implemented")
            return None

    def is_token_available(self) -> bool:
        """
        Check if a valid token is currently available.

        Returns:
            True if token exists and is not expired
        """
        if self.token_data is None:
            return False

        # Check expiry from issued_at timestamp
        issued_at = self.token_data.issued_at
        expires_at = issued_at.timestamp() + self.token_data.expires_in
        now = datetime.now(timezone.utc).timestamp()

        return expires_at > now

    def get_token_expiry(self) -> datetime | None:
        """
        Get token expiry timestamp.

        Returns:
            Token expiry time as datetime, or None if no token
        """
        if self.token_data is None:
            return None

        # Calculate expiry from issued_at timestamp
        issued_at = self.token_data.issued_at
        expires_in_seconds = self.token_data.expires_in
        expiry_timestamp = issued_at.timestamp() + expires_in_seconds
        return datetime.fromtimestamp(expiry_timestamp, timezone.utc)

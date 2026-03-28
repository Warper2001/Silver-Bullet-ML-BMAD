"""Unit tests for TokenManager."""

import asyncio
from datetime import datetime, timezone, timedelta

import pytest
from src.execution.tradestation.auth.tokens import TokenManager
from src.execution.tradestation.models import TokenResponse


class TestTokenManager:
    """Test suite for TokenManager class."""

    @pytest.fixture
    def token_manager(self) -> TokenManager:
        """Create a TokenManager instance for testing."""
        return TokenManager(env="sim")

    @pytest.fixture
    def sample_token_response(self) -> TokenResponse:
        """Create a sample TokenResponse for testing."""
        return TokenResponse(
            access_token="test_access_token",
            token_type="Bearer",
            expires_in=3600,  # 1 hour
            refresh_token="test_refresh_token",
            scope="read write",
        )

    @pytest.mark.asyncio
    async def test_initialize(self, token_manager: TokenManager) -> None:
        """Test token manager initialization."""
        await token_manager.initialize()
        assert token_manager.env == "sim"
        assert token_manager.token_data is None

    @pytest.mark.asyncio
    async def test_set_token(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test setting token data."""
        token_manager.set_token(sample_token_response)
        assert token_manager.token_data is not None
        assert token_manager.token_data.access_token == "test_access_token"

    @pytest.mark.asyncio
    async def test_clear_token(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test clearing token data."""
        token_manager.set_token(sample_token_response)
        assert token_manager.token_data is not None

        token_manager.clear_token()
        assert token_manager.token_data is None

    @pytest.mark.asyncio
    async def test_get_access_token(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test getting access token."""
        token_manager.set_token(sample_token_response)
        token = await token_manager.get_access_token()
        assert token == "test_access_token"

    @pytest.mark.asyncio
    async def test_get_access_token_no_token(self, token_manager: TokenManager) -> None:
        """Test getting access token when no token is set."""
        with pytest.raises(RuntimeError, match="No token available"):
            await token_manager.get_access_token()

    @pytest.mark.asyncio
    async def test_is_token_available(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test checking if token is available."""
        # No token set
        assert not token_manager.is_token_available()

        # Token set
        token_manager.set_token(sample_token_response)
        assert token_manager.is_token_available()

    def test_should_refresh_token_expired(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test token refresh detection for expired token."""
        # Set token with short expiry (less than refresh buffer)
        short_token = TokenResponse(
            access_token="test_token",
            token_type="Bearer",
            expires_in=200,  # Less than TOKEN_REFRESH_BUFFER (300)
            refresh_token="test_refresh",
        )
        token_manager.set_token(short_token)

        # Should indicate refresh needed
        assert token_manager._should_refresh_token()

    def test_should_refresh_token_valid(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test token refresh detection for valid token."""
        token_manager.set_token(sample_token_response)

        # Should NOT indicate refresh needed (1 hour expiry, buffer is 5 minutes)
        assert not token_manager._should_refresh_token()

    @pytest.mark.asyncio
    async def test_get_token_expiry(
        self, token_manager: TokenManager, sample_token_response: TokenResponse
    ) -> None:
        """Test getting token expiry timestamp."""
        token_manager.set_token(sample_token_response)
        expiry = token_manager.get_token_expiry()

        assert expiry is not None
        assert isinstance(expiry, datetime)

        # Expiry should be in the future
        assert expiry > datetime.now(timezone.utc)

    def test_get_token_expiry_no_token(self, token_manager: TokenManager) -> None:
        """Test getting token expiry when no token is set."""
        assert token_manager.get_token_expiry() is None

    @pytest.mark.asyncio
    async def test_token_manager_lock_safety(self, token_manager: TokenManager) -> None:
        """Test that token manager is thread-safe with asyncio lock."""
        # This test ensures the lock exists and can be used
        assert token_manager._lock is not None

        # Test concurrent access
        async def set_and_get():
            token = TokenResponse(
                access_token=f"token_{asyncio.current_task().get_name()}",
                token_type="Bearer",
                expires_in=3600,
            )
            token_manager.set_token(token)
            return await token_manager.get_access_token()

        # Run multiple concurrent tasks
        results = await asyncio.gather(
            set_and_get(),
            set_and_get(),
            set_and_get(),
        )

        # All should complete without deadlock
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

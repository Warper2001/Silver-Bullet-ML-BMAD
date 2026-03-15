"""Unit tests for TradeStation authentication."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.data.auth import TradeStationAuth, TokenResponse
from src.data.exceptions import AuthenticationError, TokenRefreshError


class TestTokenResponse:
    """Test TokenResponse model validation."""

    def test_token_response_parsing(self) -> None:
        """Test valid token response parsing."""
        token_data = {
            "access_token": "test_access_token_123",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "test_refresh_token_456",
            "scope": "read write",
        }

        response = TokenResponse(**token_data)

        assert response.access_token == "test_access_token_123"
        assert response.token_type == "Bearer"
        assert response.expires_in == 1800
        assert response.refresh_token == "test_refresh_token_456"
        assert response.scope == "read write"

    def test_expires_at_calculation(self) -> None:
        """Test token expiration datetime calculation."""
        token_data = {
            "access_token": "test_token",
            "token_type": "Bearer",
            "expires_in": 1800,  # 30 minutes
            "refresh_token": "test_refresh",
            "scope": "read",
        }

        response = TokenResponse(**token_data)
        expires_at = response.expires_at

        # Should expire approximately 30 minutes from now
        time_until_expiry = expires_at - datetime.now()
        assert timedelta(seconds=1790) <= time_until_expiry <= timedelta(seconds=1810)

    def test_token_hash_generation(self) -> None:
        """Test token hash generation doesn't expose actual token."""
        token_data = {
            "access_token": "sensitive_access_token_123",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "refresh_token",
            "scope": "read",
        }

        response = TokenResponse(**token_data)
        token_hash = response.token_hash

        # Hash should not contain the actual token
        assert "sensitive_access_token_123" not in token_hash
        assert len(token_hash) == 16  # First 16 chars of SHA256 hash

    def test_token_hash_is_deterministic(self) -> None:
        """Test token hash is deterministic for same token."""
        token_data = {
            "access_token": "same_token",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "refresh",
            "scope": "read",
        }

        response1 = TokenResponse(**token_data)
        response2 = TokenResponse(**token_data)

        assert response1.token_hash == response2.token_hash


class TestTradeStationAuth:
    """Test TradeStationAuth authentication logic."""

    @pytest.fixture
    def auth(self):
        """Create TradeStationAuth instance for testing."""
        with patch("src.data.auth.load_settings"):
            auth = TradeStationAuth()
            # Mock settings
            auth.settings.tradestation_client_id = "test_client_id"
            auth.settings.tradestation_client_secret = "test_client_secret"
            auth.settings.tradestation_redirect_uri = "http://localhost:8080/callback"
            return auth

    def test_initialization(self, auth: TradeStationAuth) -> None:
        """Test authentication manager initializes correctly."""
        assert auth._access_token is None
        assert auth._refresh_token is None
        assert auth._token_expires_at is None
        assert auth._client is None
        assert auth._refresh_task is None

    def test_is_token_valid_no_token(self, auth: TradeStationAuth) -> None:
        """Test token validity check when no token exists."""
        assert auth._is_token_valid() is False

    def test_is_token_valid_no_expiration(self, auth: TradeStationAuth) -> None:
        """Test token validity check when no expiration set."""
        auth._access_token = "test_token"
        assert auth._is_token_valid() is False

    def test_is_token_valid_expired(self, auth: TradeStationAuth) -> None:
        """Test token validity check for expired token."""
        auth._access_token = "test_token"
        auth._token_expires_at = datetime.now() - timedelta(minutes=10)
        assert auth._is_token_valid() is False

    def test_is_token_valid_with_buffer(self, auth: TradeStationAuth) -> None:
        """Test token validity check respects 5-minute buffer."""
        auth._access_token = "test_token"
        # Expires 4 minutes from now (within 5-minute buffer)
        auth._token_expires_at = datetime.now() + timedelta(minutes=4)
        assert auth._is_token_valid() is False

    def test_is_token_valid_active(self, auth: TradeStationAuth) -> None:
        """Test token validity check for active token."""
        auth._access_token = "test_token"
        # Expires 30 minutes from now (well outside buffer)
        auth._token_expires_at = datetime.now() + timedelta(minutes=30)
        assert auth._is_token_valid() is True

    @pytest.mark.asyncio
    async def test_store_tokens(self, auth: TradeStationAuth) -> None:
        """Test token storage in memory."""
        token_response = TokenResponse(
            access_token="test_access_token",
            token_type="Bearer",
            expires_in=1800,
            refresh_token="test_refresh_token",
            scope="read write",
        )

        auth._store_tokens(token_response)

        assert auth._access_token == "test_access_token"
        assert auth._refresh_token == "test_refresh_token"
        assert auth._token_expires_at is not None
        assert isinstance(auth._token_expires_at, datetime)

    @pytest.mark.asyncio
    async def test_cleanup(self, auth: TradeStationAuth) -> None:
        """Test resource cleanup clears tokens."""
        # Set up some state
        auth._access_token = "test_token"
        auth._refresh_token = "test_refresh"
        auth._token_expires_at = datetime.now() + timedelta(minutes=30)

        # Create mock client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        auth._client = mock_client

        # Run cleanup
        await auth.cleanup()

        # Verify tokens cleared
        assert auth._access_token is None
        assert auth._refresh_token is None
        assert auth._token_expires_at is None
        assert mock_client.aclose.called

    def test_retry_delays_configuration(self) -> None:
        """Test exponential backoff delays are correct."""
        auth = TradeStationAuth()
        assert auth.RETRY_DELAYS == [1, 2, 4]  # 1s, 2s, 4s
        assert auth.MAX_RETRY_ATTEMPTS == 3

    def test_refresh_interval_configuration(self) -> None:
        """Test token refresh interval is 10 minutes."""
        auth = TradeStationAuth()
        assert auth.REFRESH_INTERVAL_SECONDS == 600  # 10 minutes


class TestAuthenticationErrors:
    """Test custom exception types."""

    def test_authentication_error_creation(self) -> None:
        """Test AuthenticationError can be created with details."""
        error = AuthenticationError(
            message="Test error",
            retry_count=2,
            original_error=Exception("Original"),
        )

        assert str(error) == "Test error"
        assert error.retry_count == 2
        assert error.original_error is not None

    def test_token_refresh_error_is_authentication_error(self) -> None:
        """Test TokenRefreshError inherits from AuthenticationError."""
        error = TokenRefreshError(message="Test")

        assert isinstance(error, AuthenticationError)
        assert isinstance(error, Exception)

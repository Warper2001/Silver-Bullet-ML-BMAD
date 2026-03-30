"""Unit tests for TradeStation API v3 authentication."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import httpx

from src.data.auth_v3 import TradeStationAuthV3, TokenResponse
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
            "scope": "MarketData ReadAccount Trade",
        }

        response = TokenResponse(**token_data)

        assert response.access_token == "test_access_token_123"
        assert response.token_type == "Bearer"
        assert response.expires_in == 1800
        assert response.refresh_token == "test_refresh_token_456"
        assert response.scope == "MarketData ReadAccount Trade"

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


class TestTradeStationAuthV3:
    """Test TradeStationAuthV3 authentication class."""

    def test_initialization_with_tokens(self) -> None:
        """Test initialization with access and refresh tokens."""
        auth = TradeStationAuthV3(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
        )

        assert auth._access_token == "test_access_token"
        assert auth._refresh_token == "test_refresh_token"
        assert auth._token_expires_at is None

    def test_initialization_with_expiration(self) -> None:
        """Test initialization with token expiration datetime."""
        expires_at = datetime.now() + timedelta(minutes=30)
        auth = TradeStationAuthV3(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            token_expires_at=expires_at,
        )

        assert auth._token_expires_at == expires_at

    def test_is_authenticated_with_valid_token(self) -> None:
        """Test is_authenticated returns True for valid token."""
        expires_at = datetime.now() + timedelta(minutes=30)
        auth = TradeStationAuthV3(
            access_token="test_token",
            token_expires_at=expires_at,
        )

        assert auth.is_authenticated() is True

    def test_is_authenticated_with_expired_token(self) -> None:
        """Test is_authenticated returns False for expired token."""
        expires_at = datetime.now() - timedelta(minutes=10)
        auth = TradeStationAuthV3(
            access_token="test_token",
            token_expires_at=expires_at,
        )

        assert auth.is_authenticated() is False

    def test_is_authenticated_with_no_expiration_info(self) -> None:
        """Test is_authenticated returns True when no expiration info."""
        auth = TradeStationAuthV3(access_token="test_token")

        # No expiration info, assumes token is valid
        assert auth.is_authenticated() is True

    def test_is_authenticated_with_no_token(self) -> None:
        """Test is_authenticated returns False with no token."""
        auth = TradeStationAuthV3(access_token="")

        assert auth.is_authenticated() is False

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_token(self) -> None:
        """Test authenticate returns cached token if valid."""
        expires_at = datetime.now() + timedelta(minutes=30)
        auth = TradeStationAuthV3(
            access_token="cached_token",
            token_expires_at=expires_at,
        )

        token = await auth.authenticate()

        assert token == "cached_token"

    @pytest.mark.asyncio
    async def test_authenticate_with_expired_token_and_refresh(self) -> None:
        """Test authenticate refreshes expired token."""
        expires_at = datetime.now() - timedelta(minutes=10)
        auth = TradeStationAuthV3(
            access_token="expired_token",
            refresh_token="valid_refresh_token",
            token_expires_at=expires_at,
        )

        # Mock HTTP client
        mock_response = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "new_refresh_token",
            "scope": "MarketData ReadAccount Trade",
        }

        with patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = AsyncMock()
            mock_client.post.return_value = mock_response_obj

            token = await auth.authenticate()

            assert token == "new_access_token"
            assert auth._access_token == "new_access_token"
            assert auth._refresh_token == "new_refresh_token"

    @pytest.mark.asyncio
    async def test_authenticate_with_no_refresh_token(self) -> None:
        """Test authenticate raises error when no refresh token available."""
        auth = TradeStationAuthV3(
            access_token="",
            refresh_token="",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate()

        assert "No valid access token" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_token_refresh_with_retry(self) -> None:
        """Test token refresh with exponential backoff retry."""
        auth = TradeStationAuthV3(
            access_token="expired_token",
            refresh_token="refresh_token",
        )

        # Mock HTTP client to fail twice then succeed
        mock_success_response = {
            "access_token": "new_token",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "new_refresh",
            "scope": "MarketData",
        }

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            mock_resp = AsyncMock()
            if call_count[0] < 3:
                # First two calls fail
                mock_resp.status_code = 503
                mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Server error", request=AsyncMock(), response=mock_resp
                )
            else:
                # Third call succeeds
                mock_resp.status_code = 200
                mock_resp.json.return_value = mock_success_response
                mock_resp.raise_for_status = AsyncMock()
            return mock_resp

        with patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.post = mock_post

            await auth._refresh_token_flow()

            assert auth._access_token == "new_token"
            assert call_count[0] == 3  # Should retry 3 times

    @pytest.mark.asyncio
    async def test_from_file(self, tmp_path) -> None:
        """Test loading authentication from token file."""
        # Create temporary token file
        token_file = tmp_path / ".access_token"
        token_file.write_text("file_access_token")

        # Mock settings to include refresh token
        with patch("src.data.auth_v3.load_settings") as mock_load_settings:
            mock_settings = AsyncMock()
            mock_settings.tradestation_refresh_token = "file_refresh_token"
            mock_load_settings.return_value = mock_settings

            auth = TradeStationAuthV3.from_file(str(token_file))

            assert auth._access_token == "file_access_token"
            assert auth._refresh_token == "file_refresh_token"

    @pytest.mark.asyncio
    async def test_from_file_not_found(self) -> None:
        """Test from_file raises error when file doesn't exist."""
        with pytest.raises(AuthenticationError) as exc_info:
            TradeStationAuthV3.from_file("nonexistent_file.txt")

        assert "Token file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup closes HTTP client."""
        auth = TradeStationAuthV3(access_token="test_token")

        # Create mock client
        mock_client = AsyncMock()
        auth._client = mock_client

        await auth.cleanup()

        mock_client.aclose.assert_called_once()
        assert auth._client is None

    def test_get_token_hash(self) -> None:
        """Test token hash generation."""
        auth = TradeStationAuthV3(access_token="sensitive_token")

        token_hash = auth._get_token_hash()

        # Should not contain actual token
        assert "sensitive_token" not in token_hash
        assert len(token_hash) == 16

    def test_get_token_hash_no_token(self) -> None:
        """Test token hash returns 'None' when no token."""
        auth = TradeStationAuthV3(access_token="")

        token_hash = auth._get_token_hash()

        assert token_hash == "None"

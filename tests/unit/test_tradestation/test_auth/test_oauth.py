"""Unit tests for OAuth2Client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager
from src.execution.tradestation.exceptions import InvalidCredentialsError, AuthError
from src.execution.tradestation.models import TokenResponse


class TestOAuth2Client:
    """Test suite for OAuth2Client class."""

    @pytest.fixture
    def oauth_client_sim(self) -> OAuth2Client:
        """Create OAuth2Client for SIM environment."""
        return OAuth2Client(
            env="sim",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

    @pytest.fixture
    def oauth_client_live(self) -> OAuth2Client:
        """Create OAuth2Client for LIVE environment."""
        return OAuth2Client(
            env="live",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

    @pytest.fixture
    def sample_token_response(self) -> dict:
        """Sample token response from API."""
        return {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test_refresh_token",
            "scope": "read write",
        }

    def test_initialization_sim(self, oauth_client_sim: OAuth2Client) -> None:
        """Test OAuth2Client initialization for SIM environment."""
        assert oauth_client_sim.env == "sim"
        assert oauth_client_sim.client_id == "test_client_id"
        assert oauth_client_sim.client_secret == "test_client_secret"
        assert "sim-api.tradestation.com" in oauth_client_sim.api_base_url
        assert oauth_client_sim.token_manager is not None

    def test_initialization_live(self, oauth_client_live: OAuth2Client) -> None:
        """Test OAuth2Client initialization for LIVE environment."""
        assert oauth_client_live.env == "live"
        assert "api.tradestation.com" in oauth_client_live.api_base_url
        assert oauth_client_live.token_manager is not None

    @pytest.mark.asyncio
    async def test_client_credentials_flow_success(
        self, oauth_client_sim: OAuth2Client, sample_token_response: dict
    ) -> None:
        """Test successful Client Credentials flow for SIM environment."""
        with patch("httpx.AsyncClient") as mock_httpx:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_token_response
            mock_response.headers = {"content-type": "application/json"}

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_httpx.return_value = mock_client

            # Authenticate
            token_response = await oauth_client_sim.authenticate()

            # Verify token was set
            assert token_response.access_token == "test_access_token"
            assert oauth_client_sim.is_authenticated()

    @pytest.mark.asyncio
    async def test_client_credentials_flow_failure(
        self, oauth_client_sim: OAuth2Client
    ) -> None:
        """Test Client Credentials flow with invalid credentials."""
        with patch("httpx.AsyncClient") as mock_httpx:
            # Mock error response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "invalid_client"}
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = "Unauthorized"

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_httpx.return_value = mock_client

            # Should raise InvalidCredentialsError
            with pytest.raises(InvalidCredentialsError, match="SIM authentication failed"):
                await oauth_client_sim.authenticate()

    @pytest.mark.asyncio
    async def test_client_credentials_flow_network_error(
        self, oauth_client_sim: OAuth2Client
    ) -> None:
        """Test Client Credentials flow with network error."""
        with patch("httpx.AsyncClient") as mock_httpx:
            # Mock network error
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.NetworkError("Connection failed")
            mock_httpx.return_value = mock_client

            # Should raise AuthError
            with pytest.raises(AuthError, match="Network error"):
                await oauth_client_sim.authenticate()

    @pytest.mark.asyncio
    async def test_refresh_token_success(
        self, oauth_client_live: OAuth2Client, sample_token_response: dict
    ) -> None:
        """Test successful token refresh."""
        with patch("httpx.AsyncClient") as mock_httpx:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_token_response
            mock_response.headers = {"content-type": "application/json"}

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_httpx.return_value = mock_client

            # Refresh token
            token_response = await oauth_client_live.refresh_token("test_refresh_token")

            # Verify
            assert token_response.access_token == "test_access_token"
            assert oauth_client_live.is_authenticated()

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self, oauth_client_live: OAuth2Client) -> None:
        """Test token refresh with invalid refresh token."""
        with patch("httpx.AsyncClient") as mock_httpx:
            # Mock error response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "invalid_grant"}
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = "Unauthorized"

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_httpx.return_value = mock_client

            # Should raise InvalidCredentialsError
            with pytest.raises(InvalidCredentialsError, match="Failed to refresh token"):
                await oauth_client_live.refresh_token("invalid_refresh_token")

    def test_is_authenticated_no_token(self, oauth_client_sim: OAuth2Client) -> None:
        """Test is_authenticated when no token is set."""
        assert not oauth_client_sim.is_authenticated()

    def test_is_authenticated_with_token(
        self, oauth_client_sim: OAuth2Client, sample_token_response: dict
    ) -> None:
        """Test is_authenticated with valid token."""
        token_data = TokenResponse(**sample_token_response)
        oauth_client_sim.token_manager.set_token(token_data)
        assert oauth_client_sim.is_authenticated()

    @pytest.mark.asyncio
    async def test_get_access_token(
        self, oauth_client_sim: OAuth2Client, sample_token_response: dict
    ) -> None:
        """Test getting access token."""
        # Set token
        token_data = TokenResponse(**sample_token_response)
        oauth_client_sim.token_manager.set_token(token_data)

        # Get access token
        token = await oauth_client_sim.get_access_token()
        assert token == "test_access_token"

    @pytest.mark.asyncio
    async def test_get_access_token_no_token(self, oauth_client_sim: OAuth2Client) -> None:
        """Test getting access token when no token is set."""
        with pytest.raises(RuntimeError, match="No token available"):
            await oauth_client_sim.get_access_token()

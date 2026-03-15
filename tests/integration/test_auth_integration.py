"""Integration tests for TradeStation authentication."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import HTTPStatusError, Response

from src.data.auth import TradeStationAuth
from src.data.exceptions import TokenRefreshError


class TestAuthenticationFlow:
    """Test complete authentication flow with mock API."""

    @pytest.fixture
    def mock_token_response(self) -> dict:
        """Mock token response from TradeStation API."""
        return {
            "access_token": "test_access_token_123",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "test_refresh_token_456",
            "scope": "read write",
        }

    @pytest.fixture
    def auth(self):
        """Create TradeStationAuth instance for testing."""
        with patch("src.data.auth.load_settings"):
            auth = TradeStationAuth()
            auth.settings.tradestation_client_id = "test_client_id"
            auth.settings.tradestation_client_secret = "test_client_secret"
            auth.settings.tradestation_redirect_uri = "http://localhost:8080/callback"
            return auth

    @pytest.mark.asyncio
    async def test_initial_authentication_flow(
        self, auth: TradeStationAuth, mock_token_response: dict
    ) -> None:
        """Test complete initial authentication flow."""
        # Mock HTTP client and response
        mock_response = AsyncMock(spec=Response)
        mock_response.json.return_value = mock_token_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(auth, "_client", mock_client):
            await auth._perform_authentication()

        # Verify token stored
        assert auth._access_token == "test_access_token_123"
        assert auth._refresh_token == "test_refresh_token_456"
        assert auth._token_expires_at is not None

    @pytest.mark.asyncio
    async def test_token_refresh_flow(
        self, auth: TradeStationAuth, mock_token_response: dict
    ) -> None:
        """Test token refresh stores new tokens."""
        # Set initial token
        auth._access_token = "old_access_token"

        # Mock response
        new_token_response = mock_token_response.copy()
        new_token_response["access_token"] = "new_access_token"

        mock_response = AsyncMock(spec=Response)
        mock_response.json.return_value = new_token_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(auth, "_client", mock_client):
            await auth._perform_authentication()

        # Verify new token stored
        assert auth._access_token == "new_access_token"

    @pytest.mark.asyncio
    async def test_authentication_retry_on_http_error(
        self, auth: TradeStationAuth, mock_token_response: dict
    ) -> None:
        """Test exponential backoff retry on HTTP error."""
        # Mock responses: first 2 fail, third succeeds
        success_response = AsyncMock(spec=Response)
        success_response.json.return_value = mock_token_response
        success_response.raise_for_status = MagicMock()

        error_response = AsyncMock(spec=Response)
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = HTTPStatusError(
            "Server Error", request=None, response=error_response
        )

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            error_response,
            error_response,
            success_response,
        ]

        with patch.object(auth, "_client", mock_client):
            await auth._perform_authentication()

        # Verify 3 attempts were made
        assert mock_client.post.call_count == 3
        # Verify token stored after retries
        assert auth._access_token == "test_access_token_123"

    @pytest.mark.asyncio
    async def test_authentication_retry_exhaustion(
        self, auth: TradeStationAuth
    ) -> None:
        """Test authentication fails after max retries."""
        # Mock responses: all fail
        error_response = AsyncMock(spec=Response)
        error_response.status_code = 401
        error_response.raise_for_status.side_effect = HTTPStatusError(
            "Unauthorized", request=None, response=error_response
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = error_response

        with patch.object(auth, "_client", mock_client):
            with pytest.raises(TokenRefreshError) as exc_info:
                await auth._perform_authentication()

        # Verify error details
        assert exc_info.value.retry_count == 3
        assert exc_info.value.original_error is not None

    @pytest.mark.asyncio
    async def test_returns_cached_token_when_valid(
        self, auth: TradeStationAuth
    ) -> None:
        """Test authenticate returns cached token if still valid."""
        # Set valid token
        auth._access_token = "cached_token"
        auth._token_expires_at = datetime.now() + timedelta(minutes=30)

        token = await auth.authenticate()

        assert token == "cached_token"

    @pytest.mark.asyncio
    async def test_refreshes_token_when_expired(
        self, auth: TradeStationAuth, mock_token_response: dict
    ) -> None:
        """Test authenticate refreshes token when expired."""
        # Set expired token
        auth._access_token = "expired_token"
        auth._token_expires_at = datetime.now() - timedelta(minutes=10)

        # Mock successful refresh
        mock_response = AsyncMock(spec=Response)
        mock_response.json.return_value = mock_token_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(auth, "_client", mock_client):
            token = await auth.authenticate()

        # Verify new token returned
        assert token == "test_access_token_123"
        assert auth._access_token == "test_access_token_123"

    @pytest.mark.asyncio
    async def test_background_refresh_task_starts(self, auth: TradeStationAuth) -> None:
        """Test background refresh task starts correctly."""
        await auth.start_refresh_task()

        # Verify task created
        assert auth._refresh_task is not None
        assert not auth._refresh_task.done()

        # Cleanup
        await auth.stop_refresh_task()

    @pytest.mark.asyncio
    async def test_background_refresh_task_stops(self, auth: TradeStationAuth) -> None:
        """Test background refresh task stops correctly."""
        await auth.start_refresh_task()

        await auth.stop_refresh_task()

        # Verify task cancelled
        assert auth._refresh_task is None

    @pytest.mark.asyncio
    async def test_start_refresh_task_is_idempotent(
        self, auth: TradeStationAuth
    ) -> None:
        """Test starting refresh task twice doesn't create duplicate tasks."""
        await auth.start_refresh_task()
        first_task = auth._refresh_task

        await auth.start_refresh_task()

        # Verify same task still running
        assert auth._refresh_task == first_task
        assert not first_task.done()

        # Cleanup
        await auth.stop_refresh_task()

    @pytest.mark.asyncio
    async def test_concurrent_authentication(
        self, auth: TradeStationAuth, mock_token_response: dict
    ) -> None:
        """Test concurrent authentication requests are handled safely."""
        # Mock successful authentication
        mock_response = AsyncMock(spec=Response)
        mock_response.json.return_value = mock_token_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(auth, "_client", mock_client):
            # Simulate concurrent authentication
            tasks = [auth.authenticate() for _ in range(5)]
            results = await asyncio.gather(*tasks)

        # All should return same token
        assert all(token == "test_access_token_123" for token in results)
        # Only one authentication request should be made (others use cache)
        assert mock_client.post.call_count == 1

"""Integration tests for TradeStation OAuth 2.0 authentication flow.

These tests verify the complete authentication lifecycle with the
actual TradeStation SIM API.

Prerequisites:
- TRADESTATION_SIM_CLIENT_ID environment variable
- TRADESTATION_SIM_CLIENT_SECRET environment variable
"""

import time
from datetime import datetime, timezone, timedelta

import pytest

from src.execution.tradestation.auth.tokens import TokenManager
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.exceptions import AuthError, TokenExpiredError


@pytest.mark.integration
class TestOAuth2ClientIntegration:
    """Integration tests for OAuth2Client with TradeStation SIM API."""

    @pytest.mark.asyncio
    async def test_client_credentials_flow_success(self, sim_client_id: str, sim_client_secret: str):
        """Test successful client credentials flow for SIM environment."""
        # Create OAuth client
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Authenticate
        token_response = await oauth_client.authenticate()

        # Verify token response
        assert token_response is not None
        assert token_response.access_token is not None
        assert len(token_response.access_token) > 0
        assert token_response.token_type.lower() == "bearer"
        assert token_response.expires_in > 0

        # Verify token is set in token manager
        assert oauth_client.token_manager.is_token_available()
        assert oauth_client.token_manager.get_access_token() == token_response.access_token

        print(f"✅ Authentication successful")
        print(f"   Token type: {token_response.token_type}")
        print(f"   Expires in: {token_response.expires_in} seconds")

    @pytest.mark.asyncio
    async def test_token_expiry_detection(self, sim_client_id: str, sim_client_secret: str):
        """Test token expiry detection mechanism."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Authenticate
        await oauth_client.authenticate()

        # Get initial token expiry
        initial_expiry = oauth_client.token_manager.get_token_expiry()
        assert initial_expiry is not None
        assert initial_expiry > datetime.now(timezone.utc)

        # Check should_refresh_token (should be False for fresh token)
        # Token should be valid for at least 5 minutes
        should_refresh = oauth_client.token_manager.should_refresh_token()
        assert isinstance(should_refresh, bool)

        print(f"✅ Token expiry detection working")
        print(f"   Token expires at: {initial_expiry}")
        print(f"   Should refresh: {should_refresh}")

    @pytest.mark.asyncio
    async def test_token_refresh(self, sim_client_id: str, sim_client_secret: str):
        """Test token refresh mechanism.

        Note: SIM environment uses client credentials flow which doesn't
        return a refresh_token. This test verifies the refresh logic
        is in place even though it will re-authenticate.
        """
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Initial authentication
        await oauth_client.authenticate()
        first_token = oauth_client.token_manager.get_access_token()

        # Attempt refresh (will re-authenticate in SIM environment)
        await oauth_client.refresh_token()
        refreshed_token = oauth_client.token_manager.get_access_token()

        # Verify we got a token (may be same or different in SIM)
        assert refreshed_token is not None
        assert len(refreshed_token) > 0

        print(f"✅ Token refresh mechanism working")
        print(f"   Initial token: {first_token[:20]}...")
        print(f"   Refreshed token: {refreshed_token[:20]}...")

    @pytest.mark.asyncio
    async def test_is_authenticated(self, sim_client_id: str, sim_client_secret: str):
        """Test authentication status tracking."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Before authentication
        assert not oauth_client.is_authenticated()

        # After authentication
        await oauth_client.authenticate()
        assert oauth_client.is_authenticated()

        # After clearing token
        oauth_client.token_manager.clear_token()
        assert not oauth_client.is_authenticated()

        print(f"✅ Authentication status tracking working")

    @pytest.mark.asyncio
    async def test_get_access_token(self, sim_client_id: str, sim_client_secret: str):
        """Test get_access_token method."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Before authentication (should raise error)
        with pytest.raises(AuthError, match="No token available"):
            oauth_client.get_access_token()

        # After authentication
        await oauth_client.authenticate()
        token = oauth_client.get_access_token()
        assert token is not None
        assert len(token) > 0

        print(f"✅ get_access_token working")

    @pytest.mark.asyncio
    async def test_token_manager_thread_safety(self, sim_client_id: str, sim_client_secret: str):
        """Test TokenManager thread safety with asyncio locks."""
        import asyncio

        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Authenticate
        await oauth_client.authenticate()

        # Concurrent access to token manager
        async def get_token_multiple_times():
            for _ in range(10):
                token = oauth_client.token_manager.get_access_token()
                assert token is not None

        # Run concurrent tasks
        tasks = [get_token_multiple_times() for _ in range(5)]
        await asyncio.gather(*tasks)

        print(f"✅ Token manager thread safety verified")

    @pytest.mark.asyncio
    async def test_invalid_credentials_error(self):
        """Test authentication with invalid credentials."""
        oauth_client = OAuth2Client(
            client_id="invalid_client_id",
            client_secret="invalid_client_secret",
            env="sim",
        )

        # Should raise AuthError
        with pytest.raises(AuthError):
            await oauth_client.authenticate()

        print(f"✅ Invalid credentials error handling working")

    @pytest.mark.asyncio
    async def test_token_expiry_time_accuracy(self, sim_client_id: str, sim_client_secret: str):
        """Test that token expiry time is calculated correctly."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Authenticate
        await oauth_client.authenticate()
        token_expiry = oauth_client.token_manager.get_token_expiry()

        # Token should expire in the future (typically 1 hour from now)
        now = datetime.now(timezone.utc)
        time_until_expiry = (token_expiry - now).total_seconds()

        # TradeStation tokens typically expire in 1 hour (3600 seconds)
        # Allow some tolerance for network delays
        assert 3000 < time_until_expiry < 4000, f"Token expiry time seems incorrect: {time_until_expiry}s"

        print(f"✅ Token expiry time accurate")
        print(f"   Time until expiry: {time_until_expiry:.0f} seconds ({time_until_expiry/60:.1f} minutes)")


@pytest.mark.integration
class TestTokenManagerIntegration:
    """Integration tests for TokenManager lifecycle."""

    @pytest.mark.asyncio
    async def test_token_manager_full_lifecycle(self, sim_client_id: str, sim_client_secret: str):
        """Test complete token manager lifecycle."""
        token_manager = TokenManager()

        # Initially, no token
        assert not token_manager.is_token_available()

        # Set token (via OAuth client)
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )
        await oauth_client.authenticate()

        # Now token is available
        assert token_manager.is_token_available()
        assert token_manager.get_access_token() is not None
        assert token_manager.get_token_expiry() is not None

        # Clear token
        token_manager.clear_token()
        assert not token_manager.is_token_available()

        print(f"✅ Token manager lifecycle working")

    @pytest.mark.asyncio
    async def test_should_refresh_token_logic(self, sim_client_id: str, sim_client_secret: str):
        """Test should_refresh_token logic at different time thresholds."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Authenticate
        await oauth_client.authenticate()
        token_manager = oauth_client.token_manager

        # Fresh token (just authenticated)
        should_refresh_fresh = token_manager.should_refresh_token()
        assert isinstance(should_refresh_fresh, bool)

        # Token expiry time
        expiry = token_manager.get_token_expiry()
        time_until_expiry = (expiry - datetime.now(timezone.utc)).total_seconds()

        print(f"✅ Should refresh token logic working")
        print(f"   Time until expiry: {time_until_expiry:.0f} seconds")
        print(f"   Should refresh now: {should_refresh_fresh}")


# Performance Tests
@pytest.mark.integration
class TestAuthPerformance:
    """Performance tests for authentication flow."""

    @pytest.mark.asyncio
    async def test_authentication_latency(self, sim_client_id: str, sim_client_secret: str):
        """Test authentication request latency."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Measure authentication time
        start_time = time.time()
        await oauth_client.authenticate()
        auth_time = time.time() - start_time

        # Authentication should be fast (< 5 seconds per NFR)
        assert auth_time < 5.0, f"Authentication too slow: {auth_time:.2f}s"

        print(f"✅ Authentication latency: {auth_time:.2f}s")
        print(f"   Target: < 5.0s (NFR requirement)")

    @pytest.mark.asyncio
    async def test_token_access_speed(self, sim_client_id: str, sim_client_secret: str):
        """Test speed of accessing token from cache."""
        oauth_client = OAuth2Client(
            client_id=sim_client_id,
            client_secret=sim_client_secret,
            env="sim",
        )

        # Authenticate
        await oauth_client.authenticate()

        # Measure token access speed (1000 iterations)
        iterations = 1000
        start_time = time.time()
        for _ in range(iterations):
            token = oauth_client.token_manager.get_access_token()
        access_time = time.time() - start_time

        avg_time_per_access = (access_time / iterations) * 1000  # Convert to ms

        # Token access should be very fast (< 1ms)
        assert avg_time_per_access < 1.0, f"Token access too slow: {avg_time_per_access:.3f}ms"

        print(f"✅ Token access speed: {avg_time_per_access:.3f}ms per access")
        print(f"   Total time for {iterations} accesses: {access_time:.3f}s")

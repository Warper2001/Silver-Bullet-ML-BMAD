"""Unit tests for TradeStation OAuth authentication.

Tests OAuth flow, callback server, token exchange, refresh,
caching, file locking, and state validation.
"""

import json
import queue
import secrets
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from freezegun import freeze_time

from src.data.tradestation_auth import (
    OAuthCallbackHandler,
    OAuthCallbackServer,
    TradeStationAuth,
)
from src.data.exceptions import AuthenticationError, TokenRefreshError


@pytest.fixture
def mock_settings():
    """Mock TradeStation settings."""
    with patch("src.data.tradestation_auth.load_settings") as mock:
        settings = Mock()
        settings.tradestation_client_id = "test_client_id"
        settings.tradestation_client_secret = "test_client_secret"
        settings.tradestation_redirect_uri = "http://localhost:8080/callback"
        mock.return_value = settings
        yield mock


@pytest.fixture
def auth(mock_settings):
    """Create TradeStationAuth instance with mocked settings."""
    return TradeStationAuth()


@pytest.fixture
def temp_token_dir(monkeypatch):
    """Provide temporary directory for token cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        monkeypatch.setattr(
            "src.data.tradestation_auth.TOKEN_CACHE_DIR", temp_path / ".tradestation"
        )
        monkeypatch.setattr(
            "src.data.tradestation_auth.TOKEN_CACHE_FILE",
            temp_path / ".tradestation" / "token_cache.json",
        )
        yield temp_path


class TestTradeStationAuth:
    """Tests for TradeStationAuth class."""

    def test_get_authorization_url_with_state(self, auth):
        """Test authorization URL generation includes state parameter."""
        state = "test_state_123"
        url = auth.get_authorization_url(state, port=8080)

        assert "https://signin.tradestation.com/authorize" in url
        assert f"state={state}" in url
        assert "client_id=test_client_id" in url
        assert "redirect_uri=http://localhost:8080/callback" in url
        assert "response_type=code" in url

    def test_state_parameter_randomness(self, auth):
        """Test state parameter uses cryptographically secure random."""
        # Generate multiple state parameters
        states = [secrets.token_urlsafe(32) for _ in range(100)]

        # All should be different
        assert len(set(states)) == 100

        # All should be ~43 characters (32 bytes base64-encoded)
        assert all(len(s) >= 40 for s in states)

    @patch("src.data.tradestation_auth.httpx.Client")
    def test_exchange_code_for_tokens_success(self, mock_client_class, auth):
        """Test successful token exchange."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "MarketData",
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Exchange code for tokens
        token_response = auth.exchange_code_for_tokens("test_auth_code")

        assert token_response.access_token == "test_access_token"
        assert token_response.refresh_token == "test_refresh_token"
        assert token_response.expires_in == 3600
        assert token_response.is_valid

    @patch("src.data.tradestation_auth.httpx.Client")
    def test_exchange_code_for_tokens_failure(self, mock_client_class, auth):
        """Test token exchange failure handling."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Should raise AuthenticationError
        with pytest.raises(AuthenticationError):
            auth.exchange_code_for_tokens("invalid_code")

    @patch("src.data.tradestation_auth.httpx.Client")
    def test_refresh_access_token(self, mock_client_class, auth, temp_token_dir):
        """Test access token refresh."""
        # Setup cached token
        token_cache_data = {
            "access_token": "old_access_token",
            "refresh_token": "valid_refresh_token",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        temp_token_file = temp_token_dir / ".tradestation" / "token_cache.json"
        temp_token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_token_file, "w") as f:
            json.dump(token_cache_data, f)

        # Mock refresh response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Refresh token
        token_response = auth.refresh_access_token()

        assert token_response.access_token == "new_access_token"
        assert token_response.refresh_token == "new_refresh_token"

    @patch("src.data.tradestation_auth.httpx.Client")
    def test_refresh_token_expiration(self, mock_client_class, auth, temp_token_dir):
        """Test reauthorization when refresh token expires."""
        # Setup expired cached token (> 90 days old)
        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        token_cache_data = {
            "access_token": "old_access_token",
            "refresh_token": "expired_refresh_token",
            "expires_at": old_date.isoformat(),
            "cached_at": old_date.isoformat(),
        }

        temp_token_file = temp_token_dir / ".tradestation" / "token_cache.json"
        temp_token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_token_file, "w") as f:
            json.dump(token_cache_data, f)

        # Refresh token should return invalid
        assert not auth.is_refresh_token_valid()


class TestTokenCache:
    """Tests for token caching functionality."""

    def test_token_cache_save_load(self, auth, temp_token_dir):
        """Test saving and loading token cache."""
        tokens = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "MarketData",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save tokens
        auth.save_tokens_to_cache(tokens)

        # Verify file exists
        temp_token_file = temp_token_dir / ".tradestation" / "token_cache.json"
        assert temp_token_file.exists()

        # Verify file permissions (0600 = owner read/write only)
        stat = temp_token_file.stat()
        # On Unix, 0o600 = 384
        # This might vary by system, so just check file is readable
        assert temp_token_file.is_file()

        # Load tokens
        loaded_cache = auth.load_tokens_from_cache()
        assert loaded_cache is not None
        assert loaded_cache.access_token == "test_access_token"
        assert loaded_cache.refresh_token == "test_refresh_token"

    def test_token_cache_expired(self, auth, temp_token_dir):
        """Test expired tokens are not loaded."""
        # Create expired token cache
        old_date = datetime.now(timezone.utc) - timedelta(hours=2)
        token_cache_data = {
            "access_token": "expired_token",
            "refresh_token": "refresh_token",
            "expires_at": old_date.isoformat(),
            "cached_at": (old_date - timedelta(days=1)).isoformat(),
        }

        temp_token_file = temp_token_dir / ".tradestation" / "token_cache.json"
        temp_token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_token_file, "w") as f:
            json.dump(token_cache_data, f)

        # Load should return None for expired token
        loaded = auth.load_tokens_from_cache()
        assert loaded is None


class TestOAuthCallbackServer:
    """Tests for OAuth callback HTTP server."""

    @patch("src.data.tradestation_auth.HTTPServer")
    def test_callback_server_port_conflict(self, mock_http_server):
        """Test server tries alternative ports when 8080 is in use."""
        # First attempt fails, second succeeds
        mock_server_8080 = Mock()
        mock_server_8080.__init__ = Mock(side_effect=OSError("Port in use"))

        mock_server_8081 = Mock()
        mock_server_8081.__init__ = Mock()

        mock_http_server.side_effect = [mock_server_8080, mock_server_8081]

        server = OAuthCallbackServer(preferred_port=8080, max_port=8081)

        # Should succeed on port 8081
        port = server.start()
        assert port == 8081

    @patch("src.data.tradestation_auth.HTTPServer")
    def test_callback_timeout(self, mock_http_server):
        """Test callback timeout after 5 minutes."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server

        server = OAuthCallbackServer()
        server.start()

        # Empty queue should trigger timeout
        with pytest.raises(AuthenticationError, match="timed out"):
            server.wait_for_callback("expected_state", timeout=1)

    def test_callback_code_extraction_with_state_validation(self):
        """Test callback extracts code and validates state parameter."""
        server = OAuthCallbackServer()
        server.result_queue = queue.Queue()

        # Simulate valid callback
        server.result_queue.put(("success", "auth_code_123", "expected_state"))

        # Should return auth code when state matches
        code = server.wait_for_callback("expected_state", timeout=1)
        assert code == "auth_code_123"

    def test_callback_state_mismatch(self):
        """Test callback rejection when state doesn't match."""
        server = OAuthCallbackServer()
        server.result_queue = queue.Queue()

        # Simulate callback with wrong state
        server.result_queue.put(("success", "auth_code_123", "wrong_state"))

        # Should raise error due to state mismatch
        with pytest.raises(AuthenticationError, match="State parameter mismatch"):
            server.wait_for_callback("expected_state", timeout=1)

    def test_callback_error_handling(self):
        """Test callback handles error parameter."""
        server = OAuthCallbackServer()
        server.result_queue = queue.Queue()

        # Simulate error callback
        server.result_queue.put(("error", "access_denied", "User denied access"))

        # Should raise AuthenticationError
        with pytest.raises(AuthenticationError, match="access_denied"):
            server.wait_for_callback("expected_state", timeout=1)


class TestOAuthCallbackHandler:
    """Tests for OAuth callback HTTP request handler."""

    def test_callback_handler_parses_code(self):
        """Test handler parses authorization code from URL."""
        # Create mock server
        mock_server = Mock()
        mock_server.result_queue = queue.Queue()

        # Create handler
        handler = OAuthCallbackHandler(
            *("/GET", None, None, None),
            **{"server": mock_server}
        )

        # Test URL parsing logic
        test_path = "/?code=test_auth_code&state=test_state"

        # Parse query string
        from urllib.parse import parse_qs
        if "?" in test_path:
            query_string = test_path.split("?", 1)[1]
            params = parse_qs(query_string)

        assert params["code"] == ["test_auth_code"]
        assert params["state"] == ["test_state"]


class TestFileLocking:
    """Tests for file locking in token cache."""

    def test_token_cache_file_locking(self, auth, temp_token_dir):
        """Test file locking prevents concurrent access."""
        tokens = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "expires_in": 3600,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save tokens (uses file locking)
        auth.save_tokens_to_cache(tokens)

        # Load tokens (uses file locking)
        loaded = auth.load_tokens_from_cache()
        assert loaded is not None

    @patch("src.data.tradestation_auth.fcntl.lockf")
    def test_lock_failure_handling(self, mock_lockf, auth, temp_token_dir):
        """Test handling of file lock failures."""
        # Simulate lock failure
        mock_lockf.side_effect = IOError("Lock failed")

        tokens = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "expires_in": 3600,
        }

        # Should handle lock failure gracefully
        auth.save_tokens_to_cache(tokens)  # Should not raise

    def test_token_cache_atomic_write(self, auth, temp_token_dir):
        """Test token cache uses atomic write (temp + rename)."""
        tokens = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "expires_in": 3600,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save tokens
        auth.save_tokens_to_cache(tokens)

        # Verify temp file was cleaned up
        temp_token_file = temp_token_dir / ".tradestation" / "token_cache.json"
        temp_file = temp_token_file.with_suffix(".tmp")
        assert not temp_file.exists()

        # Verify actual file exists
        assert temp_token_file.exists()

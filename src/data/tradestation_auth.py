"""TradeStation OAuth 2.0 authentication for historical data downloader.

This module implements OAuth 2.0 authorization code flow with token caching,
refresh token management, and concurrent access protection via file locking.
"""

import fcntl
import json
import logging
import os
import queue
import secrets
import shutil
import signal
import socket
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode

import httpx

from .config import load_settings
from .exceptions import AuthenticationError, TokenRefreshError
from .tradestation_models import OAuthTokenResponse, TokenCache

logger = logging.getLogger(__name__)

# OAuth endpoints
AUTHORIZATION_ENDPOINT = "https://signin.tradestation.com/authorize"
TOKEN_ENDPOINT = "https://api.tradestation.com/v2/oauth/token"

# Token cache location
TOKEN_CACHE_DIR = Path.home() / ".tradestation"
TOKEN_CACHE_FILE = TOKEN_CACHE_DIR / "token_cache.json"

# OAuth callback timeout (seconds)
CALLBACK_TIMEOUT = 300


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback.

    Handles the callback from TradeStation after user authorization.
    """

    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        """Suppress default HTTP logging."""
        pass

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET request for OAuth callback."""
        # Parse query parameters from path
        if "?" in self.path:
            query_string = self.path.split("?", 1)[1]
            params = parse_qs(query_string)
        else:
            params = {}

        # Extract authorization code or error
        if "code" in params:
            auth_code = params["code"][0]
            state_param = params.get("state", [""])[0]

            # Put result in queue for main thread
            if hasattr(self.server, "result_queue"):
                self.server.result_queue.put(("success", auth_code, state_param))  # type: ignore[attr-defined]

            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            response_body = (
                "<html><body><h1>Authentication Successful!</h1>"
                "<p>You can close this window and return to the terminal.</p>"
                "</body></html>"
            )
            self.wfile.write(response_body.encode("utf-8"))
            logger.info("OAuth callback received authorization code")

        elif "error" in params:
            error = params["error"][0]
            error_description = params.get("error_description", [""])[0]

            # Put error in queue
            if hasattr(self.server, "result_queue"):
                self.server.result_queue.put(("error", error, error_description))  # type: ignore[attr-defined]

            # Send error response
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            response_body = (
                f"<html><body><h1>Authentication Failed</h1>"
                f"<p>Error: {error}</p>"
                f"<p>{error_description}</p>"
                f"</body></html>"
            )
            self.wfile.write(response_body.encode("utf-8"))
            logger.error(f"OAuth callback error: {error} - {error_description}")

        else:
            # No code or error parameter
            if hasattr(self.server, "result_queue"):
                self.server.result_queue.put(("error", "invalid_response", "No code or error parameter"))  # type: ignore[attr-defined]

            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            response_body = (
                "<html><body><h1>Invalid Callback</h1>"
                "<p>Expected authorization code or error parameter.</p>"
                "</body></html>"
            )
            self.wfile.write(response_body.encode("utf-8"))
            logger.error("OAuth callback: invalid response (no code or error)")


class OAuthCallbackServer:
    """HTTP server for handling OAuth callback.

    Starts a local HTTP server to receive the OAuth callback after
    user authorization in browser.
    """

    def __init__(self, preferred_port: int = 8080, max_port: int = 8090) -> None:
        """Initialize callback server.

        Args:
            preferred_port: Preferred port for server (default: 8080)
            max_port: Maximum port to try if preferred port unavailable
        """
        self.preferred_port = preferred_port
        self.max_port = max_port
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.result_queue: queue.Queue[tuple[str, str, str]] = queue.Queue()
        self.actual_port: Optional[int] = None

    def start(self) -> int:
        """Start HTTP server on available port.

        Returns:
            Port number server is listening on

        Raises:
            AuthenticationError: If no available port found
        """
        # Try preferred port first, then increment
        for port in range(self.preferred_port, self.max_port + 1):
            try:
                self.server = HTTPServer(("localhost", port), OAuthCallbackHandler)
                self.server.result_queue = self.result_queue  # type: ignore[attr-defined]
                self.actual_port = port

                # Start server in background thread
                self.server_thread = threading.Thread(
                    target=self.server.serve_forever,
                    daemon=True,
                )
                self.server_thread.start()

                logger.info(f"OAuth callback server started on port {port}")
                return port

            except OSError:
                # Port in use, try next
                logger.debug(f"Port {port} in use, trying next port")
                continue

        raise AuthenticationError(
            f"No available port between {self.preferred_port} and {self.max_port}"
        )

    def wait_for_callback(
        self, expected_state: str, timeout: int = CALLBACK_TIMEOUT
    ) -> str:
        """Wait for OAuth callback with authorization code.

        Args:
            expected_state: Expected state parameter for CSRF protection
            timeout: Seconds to wait before timeout (default: 300)

        Returns:
            Authorization code from callback

        Raises:
            AuthenticationError: If timeout, state mismatch, or error in callback
        """
        try:
            result_type, result1, result2 = self.result_queue.get(timeout=timeout)
        except queue.Empty:
            raise AuthenticationError(
                f"Authorization timed out after {timeout} seconds. "
                "Please restart the downloader."
            )

        if result_type == "error":
            raise AuthenticationError(f"Authorization failed: {result1} - {result2}")

        if result_type == "success":
            auth_code = result1
            received_state = result2

            # Validate state parameter for CSRF protection
            if received_state != expected_state:
                raise AuthenticationError(
                    f"State parameter mismatch (CSRF protection). "
                    f"Expected: {expected_state[:8]}..., Got: {received_state[:8]}..."
                )

            return auth_code

        raise AuthenticationError(f"Unexpected callback result: {result_type}")

    def stop(self) -> None:
        """Stop HTTP server and cleanup."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("OAuth callback server stopped")

        if self.server_thread:
            self.server_thread.join(timeout=2.0)
            self.server_thread = None

        self.server = None
        self.actual_port = None


class TradeStationAuth:
    """TradeStation OAuth 2.0 authentication manager.

    Handles OAuth flow, token refresh, caching, and concurrent access protection.
    """

    def __init__(self) -> None:
        """Initialize authentication manager."""
        self.settings = load_settings()
        self._token_cache: Optional[TokenCache] = None
        self._token_lock = threading.Lock()
        self._http_client: Optional[httpx.Client] = None

    def get_authorization_url(self, state: str, port: int) -> str:
        """Generate OAuth authorization URL.

        Args:
            state: CSRF protection state parameter
            port: Port for callback redirect URI

        Returns:
            Full authorization URL for browser
        """
        redirect_uri = f"http://localhost:{port}/callback"

        params = {
            "response_type": "code",
            "client_id": self.settings.tradestation_client_id,
            "redirect_uri": redirect_uri,
            "audience": "https://api.tradestation.com",
            "scope": "openid profile offline_access MarketData ReadAccount Trade",
            "state": state,
        }

        return f"{AUTHORIZATION_ENDPOINT}?{urlencode(params)}"

    def start_callback_server(self, port: int = 8080) -> OAuthCallbackServer:
        """Start OAuth callback HTTP server.

        Args:
            port: Preferred port for callback server

        Returns:
            OAuthCallbackServer instance
        """
        server = OAuthCallbackServer(preferred_port=port)
        actual_port = server.start()
        logger.info(f"Callback server listening on port {actual_port}")
        return server

    def wait_for_authorization_code(
        self,
        server: OAuthCallbackServer,
        expected_state: str,
    ) -> str:
        """Wait for authorization code from OAuth callback.

        Args:
            server: OAuth callback server instance
            expected_state: Expected state parameter for CSRF validation

        Returns:
            Authorization code

        Raises:
            AuthenticationError: If callback timeout or error
        """
        return server.wait_for_callback(expected_state)

    def exchange_code_for_tokens(self, code: str, port: int = 8080) -> OAuthTokenResponse:
        """Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from OAuth callback
            port: Port number used for callback redirect URI

        Returns:
            OAuth token response with access and refresh tokens

        Raises:
            AuthenticationError: If token exchange fails
        """
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.settings.tradestation_client_id,
            "client_secret": self.settings.tradestation_client_secret,
            "redirect_uri": f"http://localhost:{port}/callback",
        }

        try:
            response = self._http_client.post(
                TOKEN_ENDPOINT,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                f"Token exchange failed: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise AuthenticationError(f"Token exchange failed: {e}") from e

        token_data = response.json()
        return OAuthTokenResponse(**token_data)

    def refresh_access_token(self) -> OAuthTokenResponse:
        """Refresh access token using refresh token.

        Returns:
            New OAuth token response

        Raises:
            TokenRefreshError: If refresh fails
        """
        with self._token_lock:
            # Load cached token to get refresh token
            if self._token_cache is None:
                self._token_cache = self.load_tokens_from_cache()

            if self._token_cache is None:
                raise TokenRefreshError("No cached token available for refresh")

            if self._http_client is None:
                self._http_client = httpx.Client(timeout=30.0)

            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._token_cache.refresh_token,
                "client_id": self.settings.tradestation_client_id,
                "client_secret": self.settings.tradestation_client_secret,
            }

            try:
                response = self._http_client.post(
                    TOKEN_ENDPOINT,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    # Refresh token expired
                    raise TokenRefreshError(
                        "Refresh token expired. Reauthorization required."
                    ) from e
                raise TokenRefreshError(
                    f"Token refresh failed: HTTP {e.response.status_code}"
                ) from e
            except httpx.RequestError as e:
                raise TokenRefreshError(f"Token refresh failed: {e}") from e

            token_data = response.json()
            token_response = OAuthTokenResponse(**token_data)

            # Update cache with new token
            self._token_cache = TokenCache(
                access_token=token_response.access_token,
                refresh_token=token_response.refresh_token,
                expires_at=token_response.expires_at,
            )
            self.save_tokens_to_cache(token_response.to_dict())

            logger.info("Access token refreshed successfully")
            return token_response

    def get_valid_access_token(self) -> str:
        """Get valid access token, refreshing if necessary.

        Returns:
            Valid access token

        Raises:
            AuthenticationError: If no valid token available
        """
        with self._token_lock:
            # Check if we have a valid cached token
            if self._token_cache is None:
                self._token_cache = self.load_tokens_from_cache()

            if self._token_cache is not None and self._token_cache.is_valid:
                return self._token_cache.access_token

        # No valid token, try refresh
        try:
            token_response = self.refresh_access_token()
            return token_response.access_token
        except TokenRefreshError:
            raise AuthenticationError(
                "No valid token available. Reauthorization required."
            )

    def save_tokens_to_cache(self, tokens: dict) -> None:
        """Save tokens to disk cache with file locking.

        Args:
            tokens: Token dictionary to cache
        """
        # Ensure cache directory exists
        TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write to temp file first (atomic write)
        temp_file = TOKEN_CACHE_FILE.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                # Acquire exclusive lock
                fcntl.lockf(f.fileno(), fcntl.LOCK_EX)

                try:
                    json.dump(tokens, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    # Release lock
                    fcntl.lockf(f.fileno(), fcntl.LOCK_UN)

            # Atomic rename (POSIX guarantees atomicity)
            temp_file.replace(TOKEN_CACHE_FILE)

            # Set file permissions to owner read/write only (0600)
            TOKEN_CACHE_FILE.chmod(0o600)

            logger.debug(f"Tokens cached to {TOKEN_CACHE_FILE}")

        except (IOError, OSError) as e:
            logger.error(f"Failed to save token cache: {e}")

    def load_tokens_from_cache(self) -> Optional[TokenCache]:
        """Load tokens from disk cache with file locking.

        Returns:
            Cached token data if valid, None otherwise
        """
        if not TOKEN_CACHE_FILE.exists():
            return None

        try:
            with open(TOKEN_CACHE_FILE, "r") as f:
                # Acquire shared lock for reading
                fcntl.lockf(f.fileno(), fcntl.LOCK_SH)

                try:
                    data = json.load(f)
                finally:
                    # Release lock
                    fcntl.lockf(f.fileno(), fcntl.LOCK_UN)

            # Convert to TokenCache
            token_cache = TokenCache(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=datetime.fromisoformat(data["expires_at"]),
                cached_at=datetime.fromisoformat(data["cached_at"]) if "cached_at" in data else datetime.now(timezone.utc),  # noqa: E501
            )

            # Check if token is still valid
            if token_cache.is_valid:
                logger.debug("Loaded valid tokens from cache")
                return token_cache
            else:
                logger.debug("Cached token has expired")
                return None

        except (IOError, OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load token cache: {e}")
            return None

    def is_refresh_token_valid(self) -> bool:
        """Check if refresh token is valid (not expired).

        Returns:
            True if refresh token exists and is valid, False otherwise
        """
        if self._token_cache is None:
            self._token_cache = self.load_tokens_from_cache()

        if self._token_cache is None:
            return False

        # Refresh tokens typically expire in 30-90 days
        # Check if cached more than 90 days ago
        token_age = (datetime.now(timezone.utc) - self._token_cache.cached_at).days
        return token_age < 90

    def reauthorize_from_scratch(self) -> None:
        """Perform full OAuth flow from scratch.

        This should be called when refresh token is expired.

        Raises:
            AuthenticationError: If authorization fails
        """
        logger.info("Initiating full OAuth flow from scratch")

        # Generate cryptographically random state parameter
        state = secrets.token_urlsafe(32)

        # Start callback server
        server = self.start_callback_server()
        actual_port = server.actual_port or 8080

        try:
            # Get authorization URL with actual port
            auth_url = self.get_authorization_url(state, actual_port)

            # Display instructions to user
            print("\n" + "=" * 70)
            print("TRADESTATION AUTHORIZATION REQUIRED")
            print("=" * 70)
            print("\n1. Copy the URL below and open it in your browser:")
            print(f"\n   {auth_url}\n")
            print("2. Log in with your TradeStation credentials")
            print("3. Authorize the application")
            print("4. The browser will redirect to localhost")
            print("\nWaiting for authorization callback...")
            print("=" * 70 + "\n")

            # Wait for authorization code
            auth_code = self.wait_for_authorization_code(server, state)

            # Exchange code for tokens
            token_response = self.exchange_code_for_tokens(auth_code, actual_port)

            # Cache tokens
            self._token_cache = TokenCache(
                access_token=token_response.access_token,
                refresh_token=token_response.refresh_token,
                expires_at=token_response.expires_at,
            )
            self.save_tokens_to_cache(token_response.to_dict())

            print("\n✅ Authorization successful!\n")

        finally:
            server.stop()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None

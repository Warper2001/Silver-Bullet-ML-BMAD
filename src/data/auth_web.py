"""TradeStation OAuth 2.0 Authorization Code Flow authentication.

This module implements the browser-based Authorization Code Flow, which is
the default authentication method for TradeStation API keys.
"""

import asyncio
import hashlib
import logging
import socket
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

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
    refresh_token: str
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


class TradeStationAuthWeb:
    """TradeStation OAuth 2.0 Authorization Code Flow authentication.

    This implements the browser-based Authorization Code Flow, which is
    the default authentication method for TradeStation API keys.

    Flow:
    1. Start local HTTP server to handle callback
    2. Open browser for user to authenticate
    3. Receive authorization code via callback
    4. Exchange code for access token
    5. Automatically refresh token every 10 minutes
    """

    AUTH_ENDPOINT = "https://signin.tradestation.com/authorize"
    TOKEN_ENDPOINT = "https://signin.tradestation.com/oauth/token"
    REFRESH_INTERVAL_SECONDS = 600  # 10 minutes
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

    def __init__(self, port: int = 8080) -> None:
        """Initialize authentication manager.

        Args:
            port: Local port for callback server (default: 8080)
        """
        self.settings = load_settings()
        self.port = port
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._auth_code: Optional[str] = None

    async def get_access_token(self) -> str:
        """Get access token, performing full auth flow if needed.

        Returns:
            Access token for API requests

        Raises:
            AuthenticationError: If authentication fails
        """
        if self._is_token_valid():
            logger.debug("Using cached access token")
            return self._access_token  # type: ignore

        # If we have a refresh token, use it
        if self._refresh_token:
            try:
                await self._refresh_token_flow()
                return self._access_token  # type: ignore
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}, initiating full auth flow")

        # Perform full authorization code flow
        await self._perform_authorization_flow()
        return self._access_token  # type: ignore

    async def _perform_authorization_flow(self) -> None:
        """Perform full Authorization Code Flow with browser authentication.

        This method:
        1. Starts a local HTTP server to handle the callback
        2. Opens the browser for user authentication
        3. Waits for the callback with authorization code
        4. Exchanges the code for an access token

        Raises:
            AuthenticationError: If authentication fails
        """
        logger.info("Starting Authorization Code Flow...")

        # Start callback server in background
        server_task = asyncio.create_task(self._start_callback_server())

        # Give server time to start
        await asyncio.sleep(0.5)

        # Construct authorization URL
        auth_url = self._build_authorization_url()

        # Open browser for user authentication
        print("\n" + "="*70)
        print("TRADESTATION AUTHENTICATION REQUIRED")
        print("="*70)
        print("\n1. A browser window will open automatically")
        print("2. Log in with your TradeStation credentials")
        print("3. Authorize the application")
        print("4. You'll be redirected back to localhost")
        print("\n" + "="*70)
        print(f"\nOpening browser to:\n   {auth_url}\n")

        import webbrowser
        webbrowser.open(auth_url)

        # Wait for authorization code
        try:
            await asyncio.wait_for(server_task, timeout=300)  # 5 minute timeout
        except asyncio.TimeoutError:
            raise AuthenticationError(
                "Authentication timeout - no callback received within 5 minutes"
            )

        if not self._auth_code:
            raise AuthenticationError("No authorization code received")

        # Exchange authorization code for access token
        await self._exchange_code_for_token()

        logger.info("✅ Authorization Code Flow completed successfully")

    def _build_authorization_url(self) -> str:
        """Build the authorization URL for browser authentication.

        Returns:
            Full authorization URL
        """
        params = {
            "response_type": "code",
            "client_id": self.settings.tradestation_client_id,
            "redirect_uri": self.settings.tradestation_redirect_uri,
            "audience": "https://api.tradestation.com",
            "scope": "openid profile offline_access MarketData ReadAccount Trade",
        }

        return f"{self.AUTH_ENDPOINT}?{urlencode(params)}"

    async def _start_callback_server(self) -> None:
        """Start HTTP server to handle OAuth callback.

        This server listens for the callback from TradeStation and extracts
        the authorization code from the URL query parameters.

        The callback will be to the root path (e.g., http://localhost:8080?code=...),
        not to a /callback path.

        Raises:
            AuthenticationError: If callback URL is invalid
        """
        # Create socket to find available port if needed
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind(("localhost", self.port))
        except OSError:
            # Port in use, try to find available port
            sock.bind(("", 0))
            self.port = sock.getsockname()[1]
            logger.info(f"Port 8080 in use, using port {self.port}")

        sock.listen(1)

        logger.info(f"Callback server listening on port {self.port}")

        # Wait for callback
        conn, addr = sock.accept()
        data = conn.recv(4096).decode("utf-8")

        logger.debug(f"Received callback data: {data[:200]}...")

        # Parse authorization code from callback URL
        # The callback comes to root path with code in query string
        # Format: GET /?code=AUTH_CODE&state=STATE HTTP/1.1
        if "code=" in data:
            # Extract code from URL query string
            code_start = data.index("code=") + 5
            code_end = data.find("&", code_start)
            if code_end == -1:
                # Try to find the end of the query string (space before HTTP/)
                code_end = data.find(" ", code_start)
                if code_end == -1:
                    code_end = len(data)
            self._auth_code = data[code_start:code_end]

            logger.info(f"Authorization code received: {self._auth_code[:20]}...")

            # Send success response
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html; charset=utf-8\r\n"
                "\r\n"
                "<html><body><h1>Authentication Successful!</h1>"
                "<p>You can close this window and return to the terminal.</p>"
                "</body></html>"
            )
            conn.sendall(response.encode("utf-8"))
        else:
            # Send error response
            error_msg = "No authorization code received"
            if "error=" in data:
                # Extract error message
                error_start = data.index("error=") + 6
                error_end = data.find("&", error_start)
                if error_end == -1:
                    error_end = data.find(" ", error_start)
                if error_end == -1:
                    error_end = len(data)
                error_msg = data[error_start:error_end]

            logger.error(f"Authentication error: {error_msg}")

            response = (
                "HTTP/1.1 400 Bad Request\r\n"
                "Content-Type: text/html; charset=utf-8\r\n"
                "\r\n"
                f"<html><body><h1>Authentication Failed</h1>"
                f"<p>Error: {error_msg}</p>"
                f"<p>Please try again.</p>"
                f"</body></html>"
            )
            conn.sendall(response.encode("utf-8"))
            raise AuthenticationError(f"No authorization code in callback: {error_msg}")

        conn.close()
        sock.close()

    async def _exchange_code_for_token(self) -> None:
        """Exchange authorization code for access token.

        Raises:
            TokenRefreshError: If token exchange fails
        """
        if not self._auth_code:
            raise AuthenticationError("No authorization code to exchange")

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        data = {
            "grant_type": "authorization_code",
            "code": self._auth_code,
            "client_id": self.settings.tradestation_client_id,
            "client_secret": self.settings.tradestation_client_secret,
            "redirect_uri": self.settings.tradestation_redirect_uri,
        }

        response = await self._client.post(
            self.TOKEN_ENDPOINT,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        response.raise_for_status()

        token_data = response.json()
        try:
            token_response = TokenResponse(**token_data)
        except ValidationError as e:
            raise AuthenticationError(f"Invalid token response: {e}") from e

        # Store tokens
        self._access_token = token_response.access_token
        self._refresh_token = token_response.refresh_token
        self._token_expires_at = token_response.expires_at

        logger.info(
            f"Token received (token hash: {token_response.token_hash}), "
            f"expires at {self._token_expires_at}"
        )

    async def _refresh_token_flow(self) -> None:
        """Refresh access token using refresh token.

        Raises:
            TokenRefreshError: If refresh fails
        """
        if not self._refresh_token:
            raise AuthenticationError("No refresh token available")

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

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
        try:
            token_response = TokenResponse(**token_data)
        except ValidationError as e:
            raise AuthenticationError(f"Invalid token response: {e}") from e

        # Store tokens
        self._access_token = token_response.access_token
        self._refresh_token = token_response.refresh_token
        self._token_expires_at = token_response.expires_at

        logger.info(
            f"Token refreshed (token hash: {token_response.token_hash}), "
            f"expires at {self._token_expires_at}"
        )

    def _is_token_valid(self) -> bool:
        """Check if current token is valid and not expired.

        Returns:
            True if token exists and is valid, False otherwise
        """
        if self._access_token is None:
            return False

        if self._token_expires_at is None:
            return False

        # Add 5-minute buffer before expiration
        return datetime.now() < (self._token_expires_at - timedelta(minutes=5))

    async def cleanup(self) -> None:
        """Clean up resources (close HTTP client, cancel tasks)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        # Clear tokens from memory
        self._access_token = None
        self._refresh_token = None
        self._token_expires_at = None

        logger.info("Authentication resources cleaned up")

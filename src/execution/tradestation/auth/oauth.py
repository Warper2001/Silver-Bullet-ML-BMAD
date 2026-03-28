"""
TradeStation SDK - OAuth 2.0 Authentication

This module implements OAuth 2.0 Authorization Code Flow with PKCE for TradeStation API.

IMPORTANT: TradeStation ONLY supports Authorization Code Flow with PKCE.
Client Credentials flow is NOT supported.

PKCE (Proof Key for Code Exchange) Flow:
1. Generate code_verifier and code_challenge
2. Redirect user to authorization URL with code_challenge
3. User logs in and authorizes
4. Receive authorization code via callback
5. Exchange authorization code + code_verifier for tokens

TradeStation Endpoints:
- Authorization: https://signin.tradestation.com/authorize
- Token: https://signin.tradestation.com/oauth/token
- API: https://api.tradestation.com/v3

Usage:
    # Step 1: Create client and generate authorization URL
    client = OAuth2Client(client_id="...")
    auth_url = client.get_authorization_url(redirect_uri="http://localhost:8080")

    # Step 2: User visits auth_url and authorizes
    # Browser redirects to: redirect_uri?code=AUTH_CODE&state=STATE

    # Step 3: Exchange authorization code for tokens
    token_response = await client.exchange_code_for_token(
        authorization_code="AUTH_CODE",
        redirect_uri="http://localhost:8080"
    )
"""

import urllib.parse
from typing import Literal

import httpx

from src.execution.tradestation.auth.pkce import PKCEHelper
from src.execution.tradestation.auth.tokens import TokenManager
from src.execution.tradestation.exceptions import (
    AuthError,
    InvalidCredentialsError,
    TokenExpiredError,
)
from src.execution.tradestation.models import TokenResponse
from src.execution.tradestation.utils import setup_logger


class OAuth2Client:
    """
    OAuth 2.0 client for TradeStation API authentication using PKCE.

    TradeStation ONLY supports Authorization Code Flow with PKCE.
    Client Credentials flow is NOT supported.

    Attributes:
        client_id: OAuth 2.0 client ID (API Key)
        redirect_uri: Redirect URI for OAuth callback
        pkce_helper: PKCE helper for code_verifier/challenge
        token_manager: TokenManager instance for token lifecycle
        api_base_url: TradeStation API base URL
        authorization_url: TradeStation authorization endpoint
        token_url: TradeStation token endpoint
    """

    # TradeStation OAuth endpoints
    AUTHORIZATION_URL = "https://signin.tradestation.com/authorize"
    TOKEN_URL = "https://signin.tradestation.com/oauth/token"
    API_BASE_URL = "https://api.tradestation.com/v3"

    # Required scopes
    DEFAULT_SCOPES = ["openid", "profile", "offline_access", "MarketData", "ReadAccount", "Trade"]

    # Allowed redirect URIs (from TradeStation docs)
    ALLOWED_REDIRECT_URIS = [
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://localhost:31022",
    ]

    def __init__(
        self,
        client_id: str,
        redirect_uri: str = "http://localhost:8080",
        scopes: list[str] | None = None,
        token_manager: TokenManager | None = None,
    ) -> None:
        """
        Initialize OAuth 2.0 client with PKCE.

        Args:
            client_id: OAuth 2.0 client ID (API Key from TradeStation)
            redirect_uri: Redirect URI for OAuth callback (must be in allowed list)
            scopes: List of OAuth scopes (default: DEFAULT_SCOPES)
            token_manager: Optional TokenManager instance

        Raises:
            ValueError: If redirect_uri is not in allowed list
        """
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.scopes = scopes or self.DEFAULT_SCOPES
        self.token_manager = token_manager or TokenManager()
        self.pkce_helper = PKCEHelper()
        self.logger = setup_logger(f"{__name__}.OAuth2Client")

        # Validate redirect_uri
        if redirect_uri not in self.ALLOWED_REDIRECT_URIS:
            # Extract just the path part for comparison
            from urllib.parse import urlparse
            redirect_path = urlparse(redirect_uri).path
            if not any(redirect_path == urlparse(uri).path for uri in self.ALLOWED_REDIRECT_URIS):
                raise ValueError(
                    f"redirect_uri '{redirect_uri}' not in allowed list. "
                    f"Allowed: {self.ALLOWED_REDIRECT_URIS}"
                )

        # Set endpoints
        self.api_base_url = self.API_BASE_URL
        self.authorization_url = self.AUTHORIZATION_URL
        self.token_url = self.TOKEN_URL

    def get_authorization_url(
        self,
        state: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """
        Generate the authorization URL for the user to visit.

        Args:
            state: Optional state string for CSRF protection
            prompt: Optional prompt parameter ("login" forces login screen)

        Returns:
            Full authorization URL for user to visit

        Example:
            client = OAuth2Client(client_id="...")
            url = client.get_authorization_url()
            print(f"Visit: {url}")
            # User visits URL, authorizes, gets redirected back
        """
        # Generate state if not provided
        if not state:
            state = PKCEHelper.generate_state()

        # Build query parameters
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "audience": "https://api.tradestation.com",
            "scope": " ".join(self.scopes),
            "state": state,
            "code_challenge": self.pkce_helper.code_challenge,
            "code_challenge_method": self.pkce_helper.code_challenge_method,
        }

        # Add optional prompt parameter
        if prompt:
            params["prompt"] = prompt

        # Build URL
        query_string = urllib.parse.urlencode(params)
        auth_url = f"{self.authorization_url}?{query_string}"

        self.logger.info(f"Generated authorization URL for client: {self.client_id}")
        return auth_url

    async def exchange_code_for_token(
        self,
        authorization_code: str,
    ) -> TokenResponse:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Code received from OAuth callback

        Returns:
            Token response with access_token, refresh_token, and id_token

        Raises:
            InvalidCredentialsError: If code is invalid or expired
            AuthError: If token exchange fails

        Example:
            # After user authorizes and redirects back with code
            token_response = await client.exchange_code_for_token(
                authorization_code="AUTH_CODE_FROM_CALLBACK"
            )
            print(f"Access token: {token_response.access_token}")
        """
        params = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": self.pkce_helper.code_verifier,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        self.logger.info("Exchanging authorization code for tokens")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    token_data = response.json()
                    token_response = TokenResponse(**token_data)
                    self.token_manager.set_token(token_response)
                    self.logger.info("Token exchange successful")
                    return token_response
                else:
                    error_detail = response.json() if response.headers.get("content-type") == "application/json" else response.text
                    self.logger.error(f"Token exchange failed: {error_detail}")
                    raise InvalidCredentialsError(
                        "Failed to exchange authorization code",
                        details={"status_code": response.status_code, "error": error_detail},
                    )

        except httpx.HTTPError as e:
            self.logger.error(f"Network error during token exchange: {e}")
            raise AuthError(f"Network error during token exchange: {e}")

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from initial authentication

        Returns:
            New token response

        Raises:
            InvalidCredentialsError: If refresh token is invalid
            AuthError: If refresh fails
        """
        params = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": refresh_token,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        self.logger.info("Refreshing access token")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    token_data = response.json()
                    token_response = TokenResponse(**token_data)

                    # Log if refresh token rotation occurred
                    if token_response.refresh_token and token_response.refresh_token != refresh_token:
                        self.logger.info("Token refreshed with new refresh token (rotation enabled)")

                    self.token_manager.set_token(token_response)
                    self.logger.info("Token refreshed successfully")
                    return token_response
                else:
                    error_detail = response.json() if response.headers.get("content-type") == "application/json" else response.text
                    self.logger.error(f"Token refresh failed: {error_detail}")
                    raise InvalidCredentialsError(
                        "Failed to refresh token",
                        details={"status_code": response.status_code, "error": error_detail},
                    )

        except httpx.HTTPError as e:
            self.logger.error(f"Network error during token refresh: {e}")
            raise AuthError(f"Network error during token refresh: {e}")

    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        Revoke a refresh token for security.

        WARNING: This will revoke ALL valid refresh tokens for this API key,
        not just the one passed in.

        Args:
            refresh_token: Refresh token to revoke

        Returns:
            True if revocation was successful

        Raises:
            AuthError: If revocation fails

        Example:
            await client.revoke_refresh_token(refresh_token)
            # Use this when tokens are compromised or user logs out
        """
        revoke_url = "https://signin.tradestation.com/oauth/revoke"

        # Note: TradeStation docs show using JSON body for revoke endpoint
        # even though other endpoints use form encoding
        data = {
            "client_id": self.client_id,
            "token": refresh_token,
        }

        headers = {
            "Content-Type": "application/json",
        }

        self.logger.warning("Revoking refresh token (this will revoke ALL tokens for this API key)")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    revoke_url,
                    json=data,
                    headers=headers,
                )

                if response.status_code == 200:
                    self.logger.info("Refresh token revoked successfully")
                    # Clear token from manager as well
                    self.token_manager.clear_token()
                    return True
                else:
                    error_detail = response.json() if response.headers.get("content-type") == "application/json" else response.text
                    self.logger.error(f"Token revocation failed: {error_detail}")
                    raise AuthError(
                        "Failed to revoke refresh token",
                        details={"status_code": response.status_code, "error": error_detail},
                    )

        except httpx.HTTPError as e:
            self.logger.error(f"Network error during token revocation: {e}")
            raise AuthError(f"Network error during token revocation: {e}")

    async def get_access_token(self) -> str:
        """
        Get current access token, refreshing if necessary.

        This is a convenience method that delegates to TokenManager.

        Returns:
            Valid access token

        Raises:
            RuntimeError: If no token is available
        """
        return await self.token_manager.get_access_token()

    def is_authenticated(self) -> bool:
        """
        Check if client is authenticated with valid token.

        Returns:
            True if authenticated with valid token
        """
        return self.token_manager.is_token_available()

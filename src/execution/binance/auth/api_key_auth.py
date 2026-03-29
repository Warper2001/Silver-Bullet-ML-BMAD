"""
Binance API Key Authentication

This module provides API key authentication for Binance API requests.

Binance uses API Key + HMAC SHA256 signature authentication (simpler than OAuth).
- API key is sent in the X-MBX-APIKEY header
- Signature is sent as a query parameter for signed endpoints
- No token refresh, no callback server, no PKCE complexity

API Docs: https://binance-docs.github.io/apidocs/#endpoint-security-type

Security:
- API key identifies the account
- Secret key generates HMAC SHA256 signatures for trading operations
- Public endpoints do not require authentication
- Signed endpoints require both API key and signature
"""

import logging
from datetime import datetime, timezone

from httpx import AsyncClient

from src.execution.binance.auth.signature import SignatureGenerator
from src.execution.binance.exceptions import InvalidCredentialsError, SignatureGenerationError
from src.execution.binance.utils import setup_logger


class ApiKeyAuth:
    """
    Binance API key authentication manager.

    This class handles:
    - API key storage and validation
    - HMAC SHA256 signature generation
    - Request signing for signed endpoints
    - Listen key management for WebSocket user data streams

    API Docs: https://binance-docs.github.io/apidocs/#endpoint-security-type

    Attributes:
        api_key: Binance API key (64+ characters)
        signature_generator: HMAC SHA256 signature generator

    Example:
        >>> auth = ApiKeyAuth(api_key="...", api_secret="...")
        >>> headers = auth.get_headers()
        >>> signature = auth.sign_request({"symbol": "BTCUSDT", "timestamp": ...})
        >>> listen_key = await auth.acquire_listen_key(httpx.AsyncClient())
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        """
        Initialize API key authentication.

        Args:
            api_key: Binance API key (64+ characters)
            api_secret: Binance API secret key (64+ characters)

        Raises:
            InvalidCredentialsError: If credentials are invalid
        """
        if len(api_key) < 64:
            raise InvalidCredentialsError("API key must be at least 64 characters")

        if len(api_secret) < 64:
            raise InvalidCredentialsError("API secret must be at least 64 characters")

        self.api_key = api_key
        self.api_secret = api_secret
        self.signature_generator = SignatureGenerator(api_secret)
        self.logger = setup_logger(f"{__name__}.ApiKeyAuth")

    def get_headers(self) -> dict[str, str]:
        """
        Get HTTP headers for Binance API request.

        The X-MBX-APIKEY header is required for all authenticated requests.

        Returns:
            Dictionary with X-MBX-APIKEY header

        API Docs: https://binance-docs.github.io/apidocs/#endpoint-security-type
        """
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json",
        }

    def get_timestamp_ms(self) -> int:
        """
        Get current timestamp in milliseconds (required for signatures).

        Binance requires timestamp in milliseconds for all signed requests.

        Returns:
            Current timestamp in milliseconds (UTC)

        Example:
            >>> timestamp = auth.get_timestamp_ms()
            >>> print(timestamp)
            1234567890000
        """
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    def sign_request(self, query_params: dict[str, str | int]) -> str:
        """
        Sign request parameters with HMAC SHA256 signature.

        Add timestamp to query_params if not already present,
        then generate signature.

        Args:
            query_params: Query parameters for the request

        Returns:
            Hex-encoded HMAC SHA256 signature

        Raises:
            SignatureGenerationError: If signature generation fails

        Example:
            >>> params = {"symbol": "BTCUSDT"}
            >>> signature = auth.sign_request(params)
            >>> # Add signature to request
            >>> params["signature"] = signature
        """
        # Add timestamp if not present
        if "timestamp" not in query_params:
            query_params["timestamp"] = self.get_timestamp_ms()

        try:
            signature = self.signature_generator.generate_signature(query_params)
            return signature
        except Exception as e:
            self.logger.error(f"Failed to sign request: {e}")
            raise SignatureGenerationError(f"Signature generation failed: {e}") from e

    async def acquire_listen_key(self, http_client: AsyncClient, base_url: str) -> str:
        """
        Acquire listen key for WebSocket user data stream.

        The user data stream requires a listen key, which is acquired via
        a POST request to /api/v3/userDataStream with HMAC signature.

        API Docs: https://binance-docs.github.io/apidocs/#listen-key-spot

        Args:
            http_client: httpx AsyncClient instance
            base_url: Base URL for Binance API (e.g., "https://api.binance.com")

        Returns:
            Listen key for WebSocket user data stream

        Raises:
            InvalidCredentialsError: If authentication fails
            Exception: If request fails

        Note:
            - Listen key expires after 24 hours
            - Must send keepalive ping every 30 minutes
            - Use keepalive_listen_key() to extend validity
        """
        try:
            # Prepare signed request parameters
            query_params = {"timestamp": self.get_timestamp_ms()}
            signature = self.sign_request(query_params)
            query_params["signature"] = signature

            # Construct URL
            url = f"{base_url}/api/v3/userDataStream"

            # Send POST request with signature
            headers = self.get_headers()
            response = await http_client.post(
                url,
                headers=headers,
                params=query_params,
            )

            # Check response
            if response.status_code == 401 or response.status_code == 403:
                error_msg = response.json().get("msg", "Authentication failed")
                raise InvalidCredentialsError(f"Failed to acquire listen key: {error_msg}")

            response.raise_for_status()

            # Extract listen key from response
            data = response.json()
            listen_key = data.get("listeningKey") or data.get("listenKey")

            if not listen_key:
                raise Exception("Listen key not found in response")

            self.logger.info("Successfully acquired listen key for user data stream")

            return listen_key

        except InvalidCredentialsError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to acquire listen key: {e}")
            raise

    async def keepalive_listen_key(
        self, http_client: AsyncClient, base_url: str, listen_key: str
    ) -> bool:
        """
        Extend listen key validity (keepalive ping).

        Listen keys expire after 24 hours. Send keepalive ping every 30 minutes
        to extend validity. This is done via PUT request to /api/v3/userDataStream.

        API Docs: https://binance-docs.github.io/apidocs/#listen-key-spot

        Args:
            http_client: httpx AsyncClient instance
            base_url: Base URL for Binance API
            listen_key: Listen key to extend

        Returns:
            True if keepalive successful, False otherwise

        Raises:
            Exception: If request fails
        """
        try:
            # Prepare signed request parameters
            query_params = {"timestamp": self.get_timestamp_ms()}
            signature = self.sign_request(query_params)
            query_params["signature"] = signature

            # Construct URL
            url = f"{base_url}/api/v3/userDataStream?listenKey={listen_key}"

            # Send PUT request with signature
            headers = self.get_headers()
            response = await http_client.put(
                url,
                headers=headers,
                params=query_params,
            )

            response.raise_for_status()

            self.logger.debug("Successfully extended listen key validity")

            return True

        except Exception as e:
            self.logger.error(f"Failed to keepalive listen key: {e}")
            return False

    async def close_listen_key(
        self, http_client: AsyncClient, base_url: str, listen_key: str
    ) -> bool:
        """
        Close listen key (invalidate WebSocket user data stream).

        Use this to properly close the user data stream when shutting down.

        API Docs: https://binance-docs.github.io/apidocs/#listen-key-spot

        Args:
            http_client: httpx AsyncClient instance
            base_url: Base URL for Binance API
            listen_key: Listen key to close

        Returns:
            True if close successful, False otherwise
        """
        try:
            # Prepare signed request parameters
            query_params = {"timestamp": self.get_timestamp_ms()}
            signature = self.sign_request(query_params)
            query_params["signature"] = signature

            # Construct URL
            url = f"{base_url}/api/v3/userDataStream?listenKey={listen_key}"

            # Send DELETE request with signature
            headers = self.get_headers()
            response = await http_client.delete(
                url,
                headers=headers,
                params=query_params,
            )

            response.raise_for_status()

            self.logger.info("Successfully closed listen key")

            return True

        except Exception as e:
            self.logger.error(f"Failed to close listen key: {e}")
            return False


def create_api_key_auth(api_key: str, api_secret: str) -> ApiKeyAuth:
    """
    Factory function to create an ApiKeyAuth instance.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret key

    Returns:
        ApiKeyAuth instance

    Example:
        >>> from src.data.crypto_config import load_crypto_settings
        >>> settings = load_crypto_settings()
        >>> auth = create_api_key_auth(
        ...     settings.crypto_exchange_api_key,
        ...     settings.crypto_exchange_api_secret
        ... )
    """
    return ApiKeyAuth(api_key, api_secret)

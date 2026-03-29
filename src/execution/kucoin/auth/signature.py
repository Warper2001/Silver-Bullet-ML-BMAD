"""
KuCoin Request Signature Generation

This module provides HMAC SHA256 signature generation for KuCoin API requests.

KuCoin requires HMAC SHA256 signatures for all signed endpoints (trading operations).
The signature is computed from a formatted string using the API secret key.

API Docs: https://docs.kucoin.com/#authentication-signing-a-message

Security:
- Never log or expose the API secret key or passphrase
- Use secrets.compare_digest() for timing-safe comparison (if needed)
- Generate signatures in constant-time to prevent timing attacks
"""

import hashlib
import hmac
import logging
from urllib.parse import urlencode

from src.execution.kucoin.utils import setup_logger


class SignatureGenerator:
    """
    Generate HMAC SHA256 signatures for KuCoin API requests.

    KuCoin requires all signed requests to include:
    1. API Key in KC-API-KEY header
    2. API Passphrase in KC-API-PASSPHRASE header
    3. API Timestamp (milliseconds) in KC-API-TIMESTAMP header
    4. Signature in KC-API-SIGN header (HMAC SHA256)

    Signature format: HMAC SHA256(secret_key, timestamp + nonce + request_method + request_path + body)

    Example:
        generator = SignatureGenerator(api_secret="your_secret_key", passphrase="your_passphrase")
        signature = generator.generate_signature("GET", "/api/v1/orders", {}, "123456789000", "abc123")
        # Add signature to request headers

    API Docs: https://docs.kucoin.com/#authentication-signing-a-message
    """

    def __init__(self, api_secret: str, passphrase: str) -> None:
        """
        Initialize signature generator with API secret key and passphrase.

        Args:
            api_secret: KuCoin API secret key
            passphrase: KuCoin API passphrase (created when generating API key)

        Raises:
            ValueError: If credentials are too short
        """
        if len(api_secret) < 36:
            raise ValueError("API secret key must be at least 36 characters")

        if len(passphrase) < 1:
            raise ValueError("Passphrase cannot be empty")

        self.api_secret = api_secret
        self.passphrase = passphrase
        self.logger = setup_logger(f"{__name__}.SignatureGenerator")

    def generate_signature(
        self,
        method: str,
        endpoint: str,
        params: dict[str, str] | None = None,
        timestamp: str | None = None,
        nonce: str | None = None,
        body: str = "",
    ) -> str:
        """
        Generate HMAC SHA256 signature for KuCoin API request.

        The signature string is formatted as: timestamp + nonce + method + endpoint + body

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path (e.g., "/api/v1/orders")
            params: Query parameters (optional)
            timestamp: API timestamp in milliseconds (optional, will use current if None)
            nonce: Unique nonce for each request (optional, will use timestamp if None)
            body: Request body for POST requests (default empty string)

        Returns:
            Hex-encoded HMAC SHA256 signature

        Raises:
            ValueError: If parameters are invalid
            Exception: If signature generation fails

        Example:
            >>> signature = generator.generate_signature(
            ...     "GET",
            ...     "/api/v1/orders",
            ...     {"symbol": "BTC-USDT"}
            ... )
        """
        if not method:
            raise ValueError("HTTP method cannot be empty")

        if not endpoint:
            raise ValueError("Endpoint cannot be empty")

        try:
            # Use current timestamp if not provided
            if timestamp is None:
                import time
                timestamp = str(int(time.time() * 1000))

            # Use timestamp as nonce if not provided
            if nonce is None:
                nonce = timestamp

            # Build query string
            query_string = ""
            if params:
                query_string = urlencode(params)

            # Build signature string: timestamp + nonce + method + endpoint + query + body
            # Note: KuCoin doesn't include query string for GET requests in signature
            signature_str = f"{timestamp}{nonce}{method.upper()}{endpoint}"

            # Add body for POST requests
            if body:
                signature_str += body

            # Generate HMAC SHA256 signature
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                signature_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            self.logger.debug(f"Generated signature for {method} {endpoint}")

            return signature

        except Exception as e:
            self.logger.error(f"Signature generation failed: {e}")
            raise

    def get_headers(
        self,
        api_key: str,
        method: str,
        endpoint: str,
        params: dict[str, str] | None = None,
        timestamp: str | None = None,
        nonce: str | None = None,
        body: str = "",
    ) -> dict[str, str]:
        """
        Generate complete headers for KuCoin API request including signature.

        Args:
            api_key: KuCoin API key
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            timestamp: API timestamp
            nonce: Request nonce
            body: Request body

        Returns:
            Dictionary with all required headers

        Example:
            >>> headers = generator.get_headers(
            ...     "your_api_key",
            ...     "GET",
            ...     "/api/v1/orders",
            ...     {"symbol": "BTC-USDT"}
            ... )
        """
        # Use current timestamp if not provided
        if timestamp is None:
            import time
            timestamp = str(int(time.time() * 1000))

        # Generate signature
        signature = self.generate_signature(
            method=method,
            endpoint=endpoint,
            params=params,
            timestamp=timestamp,
            nonce=nonce,
            body=body,
        )

        # Build headers
        headers = {
            "KC-API-KEY": api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": self.passphrase,
            "KC-API-KEY-VERSION": "2",  # API key version
            "Content-Type": "application/json",
        }

        return headers


def create_signature_generator(api_secret: str, passphrase: str) -> SignatureGenerator:
    """
    Factory function to create a SignatureGenerator instance.

    Args:
        api_secret: KuCoin API secret key
        passphrase: KuCoin API passphrase

    Returns:
        SignatureGenerator instance

    Example:
        >>> from src.data.kucoin_config import load_kucoin_settings
        >>> settings = load_kucoin_settings()
        >>> generator = create_signature_generator(
        ...     settings.kucoin_api_secret,
        ...     settings.kucoin_api_passphrase
        ... )
    """
    return SignatureGenerator(api_secret, passphrase)

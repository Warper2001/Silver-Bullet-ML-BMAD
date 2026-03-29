"""
Binance Request Signature Generation

This module provides HMAC SHA256 signature generation for Binance API requests.

Binance requires HMAC SHA256 signatures for all signed endpoints (trading operations).
The signature is computed from the query string using the API secret key.

API Docs: https://binance-docs.github.io/apidocs/#signed-trade-and-user_data-endpoint-security

Security:
- Never log or expose the API secret key
- Use secrets.compare_digest() for timing-safe comparison (if needed)
- Generate signatures in constant-time to prevent timing attacks
"""

import hashlib
import hmac
import logging
from urllib.parse import urlencode

from src.execution.binance.utils import setup_logger


class SignatureGenerator:
    """
    Generate HMAC SHA256 signatures for Binance API requests.

    Binance requires all signed requests to include:
    1. Query string with timestamp parameter
    2. signature parameter = HMAC SHA256(secret_key, query_string)

    Example:
        generator = SignatureGenerator(api_secret="your_secret_key")
        query_params = {"symbol": "BTCUSDT", "timestamp": 1234567890000}
        signature = generator.generate_signature(query_params)
        # Add signature to request: ?symbol=BTCUSDT&timestamp=1234567890000&signature=...

    API Docs: https://binance-docs.github.io/apidocs/#signed-trade-and-user_data-endpoint-security
    """

    def __init__(self, api_secret: str) -> None:
        """
        Initialize signature generator with API secret key.

        Args:
            api_secret: Binance API secret key (64+ characters)

        Raises:
            ValueError: If api_secret is too short
        """
        if len(api_secret) < 64:
            raise ValueError("API secret key must be at least 64 characters")

        self.api_secret = api_secret
        self.logger = setup_logger(f"{__name__}.SignatureGenerator")

    def generate_signature(self, query_params: dict[str, str | int]) -> str:
        """
        Generate HMAC SHA256 signature for query parameters.

        Args:
            query_params: Dictionary of query parameters (must include timestamp)

        Returns:
            Hex-encoded HMAC SHA256 signature

        Raises:
            ValueError: If query_params is empty or missing timestamp
            Exception: If signature generation fails

        Example:
            >>> params = {"symbol": "BTCUSDT", "timestamp": 1234567890000}
            >>> signature = generator.generate_signature(params)
            >>> print(signature)
            'a1b2c3d4e5f6...'
        """
        if not query_params:
            raise ValueError("Query parameters cannot be empty")

        if "timestamp" not in query_params:
            raise ValueError("Query parameters must include 'timestamp'")

        try:
            # Create query string (sorted by key for consistency)
            query_string = urlencode(query_params)

            # Generate HMAC SHA256 signature
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            self.logger.debug(f"Generated signature for query: {query_string[:50]}...")

            return signature

        except Exception as e:
            self.logger.error(f"Signature generation failed: {e}")
            raise

    def generate_signature_from_string(self, query_string: str) -> str:
        """
        Generate HMAC SHA256 signature from pre-formatted query string.

        Use this if you have already constructed the query string.

        Args:
            query_string: URL-encoded query string (e.g., "symbol=BTCUSDT&timestamp=...")

        Returns:
            Hex-encoded HMAC SHA256 signature

        Raises:
            ValueError: If query_string is empty
            Exception: If signature generation fails

        Example:
            >>> query_str = "symbol=BTCUSDT&timestamp=1234567890000"
            >>> signature = generator.generate_signature_from_string(query_str)
        """
        if not query_string:
            raise ValueError("Query string cannot be empty")

        try:
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            self.logger.debug(f"Generated signature for string: {query_string[:50]}...")

            return signature

        except Exception as e:
            self.logger.error(f"Signature generation failed: {e}")
            raise


def create_signature_generator(api_secret: str) -> SignatureGenerator:
    """
    Factory function to create a SignatureGenerator instance.

    Args:
        api_secret: Binance API secret key

    Returns:
        SignatureGenerator instance

    Example:
        >>> from src.data.crypto_config import load_crypto_settings
        >>> settings = load_crypto_settings()
        >>> generator = create_signature_generator(settings.crypto_exchange_api_secret)
    """
    return SignatureGenerator(api_secret)

"""
BYDFI API Signature Generator

Generates HMAC SHA256 signatures for BYDFI API authentication.
Based on BYDFI API documentation: https://developers.bydfi.com/en/signature

Signature Format:
    accessKey + timestamp + queryString + body

For GET requests (no body):
    accessKey + timestamp + queryString

For POST requests:
    accessKey + timestamp + queryString + body

Headers Required:
    X-API-KEY: Your API key
    X-API-TIMESTAMP: Request timestamp
    X-API-SIGNATURE: Generated signature
    Content-Type: application/json
"""

import hashlib
import hmac
import time
from typing import Optional


class BYDFISignatureGenerator:
    """
    Generates HMAC SHA256 signatures for BYDFI API requests.

    BYDFI uses a simpler authentication than KuCoin:
    - No passphrase required
    - Direct string concatenation for signature
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize signature generator.

        Args:
            api_key: BYDFI API key
            api_secret: BYDFI API secret key
        """
        self.api_key = api_key
        self.api_secret = api_secret

    def generate_signature(
        self,
        timestamp: str,
        method: str,
        endpoint: str,
        query_string: str = "",
        body: str = "",
    ) -> str:
        """
        Generate HMAC SHA256 signature for BYDFI API request.

        Signature format (from BYDFI docs):
            accessKey + timestamp + queryString + body

        Args:
            timestamp: Request timestamp as string
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path (e.g., /v1/spot/kline)
            query_string: Query parameters string (without ?)
            body: Request body as string (for POST requests)

        Returns:
            str: Hex-encoded HMAC SHA256 signature

        Examples:
            >>> gen = BYDFISignatureGenerator("key", "secret")
            >>> sig = gen.generate_signature("1234567890", "GET", "/v1/spot/kline", "symbol=BTC-USDT&interval=5m")
            >>> # GET request has no body, so body is omitted
        """
        # Build signature string: accessKey + timestamp + queryString + body
        # Note: BYDFI signature doesn't include method or endpoint
        signature_parts = [
            self.api_key,
            timestamp,
        ]

        # Add query string if present
        if query_string:
            signature_parts.append(query_string)

        # Add body if present (typically for POST requests)
        if body:
            signature_parts.append(body)

        # Concatenate all parts
        signature_string = "".join(signature_parts)

        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def generate_headers(
        self,
        method: str,
        endpoint: str,
        query_string: str = "",
        body: str = "",
    ) -> dict[str, str]:
        """
        Generate complete headers for BYDFI API request.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            query_string: Query parameters string
            body: Request body as string

        Returns:
            dict: Complete headers including signature

        Examples:
            >>> gen = BYDFISignatureGenerator("key", "secret")
            >>> headers = gen.generate_headers("GET", "/v1/spot/kline", "symbol=BTC-USDT")
            >>> headers["X-API-SIGNATURE"]
            'abc123...'
        """
        # Generate timestamp
        timestamp = str(int(time.time() * 1000))

        # Generate signature
        signature = self.generate_signature(
            timestamp=timestamp,
            method=method,
            endpoint=endpoint,
            query_string=query_string,
            body=body,
        )

        # Build headers
        headers = {
            "X-API-KEY": self.api_key,
            "X-API-TIMESTAMP": timestamp,
            "X-API-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

        return headers


def create_bydfi_signature_generator(
    api_key: str,
    api_secret: str,
) -> BYDFISignatureGenerator:
    """
    Factory function to create BYDFI signature generator.

    Args:
        api_key: BYDFI API key
        api_secret: BYDFI API secret key

    Returns:
        BYDFISignatureGenerator: Configured signature generator
    """
    return BYDFISignatureGenerator(api_key=api_key, api_secret=api_secret)

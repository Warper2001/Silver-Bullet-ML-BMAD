"""Kraken Futures HMAC-SHA512 authentication.

Signature formula (non-obvious — see spec Design Notes):
    message = urlencode(post_data) + nonce + endpoint_path
    sha256  = SHA256(message)
    authent = base64( HMAC-SHA512(base64decode(secret), sha256) )
"""

import base64
import hashlib
import hmac
import os
import time
import urllib.parse

from src.execution.kraken.exceptions import KrakenAuthError


class KrakenFuturesAuth:
    """Loads Kraken Futures credentials from env and signs requests."""

    def __init__(self) -> None:
        api_key = os.environ.get("KRAKEN_FUTURES_API_KEY", "")
        api_secret = os.environ.get("KRAKEN_FUTURES_API_SECRET", "")
        if not api_key or not api_secret:
            raise KrakenAuthError(
                "KRAKEN_FUTURES_API_KEY and KRAKEN_FUTURES_API_SECRET must be set in .env"
            )
        self._api_key = api_key
        self._api_secret = api_secret

    def sign_request(self, endpoint: str, post_data: dict, nonce: str) -> str:
        """Return the Authent header value for a signed request.

        Args:
            endpoint: Path only, e.g. "/derivatives/api/v3/sendorder"
            post_data: Dict of POST params (empty dict for GET)
            nonce: Millisecond timestamp string
        """
        post_str = urllib.parse.urlencode(post_data)
        message = post_str + nonce + endpoint
        sha256_hash = hashlib.sha256(message.encode("utf-8")).digest()
        secret_bytes = base64.b64decode(self._api_secret)
        hmac_sig = hmac.new(secret_bytes, sha256_hash, hashlib.sha512).digest()
        return base64.b64encode(hmac_sig).decode("utf-8")

    def get_headers(self, endpoint: str, post_data: dict) -> dict:
        """Return auth headers for a request to the given endpoint."""
        nonce = str(int(time.time() * 1000))
        authent = self.sign_request(endpoint, post_data, nonce)
        return {
            "APIKey": self._api_key,
            "Authent": authent,
            "Nonce": nonce,
        }

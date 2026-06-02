"""Kraken Spot REST API HMAC-SHA512 authentication.

Algorithm (different from KrakenFuturesAuth — do not confuse):
    post_data["nonce"] = str(int(time.time() * 1000))  # nonce is IN the POST body
    sha256    = SHA256( (nonce_str + urlencode(post_data)).encode("utf-8") )
    message   = uri_path.encode("utf-8") + sha256       # bytes concat
    api_sign  = base64( HMAC-SHA512( base64_decode(secret), message ) )
    headers   = { "API-Key": key, "API-Sign": api_sign }  # no separate Nonce header
"""

import base64
import hashlib
import hmac
import os
import time
import urllib.parse

from src.execution.kraken.exceptions import KrakenAuthError


class KrakenSpotAuth:
    """Signs requests for the Kraken Spot REST API (api.kraken.com/0/private/*)."""

    def __init__(self) -> None:
        api_key    = os.environ.get("KRAKEN_SPOT_API_KEY", "")
        api_secret = os.environ.get("KRAKEN_SPOT_API_SECRET", "")
        if not api_key or not api_secret:
            raise KrakenAuthError(
                "KRAKEN_SPOT_API_KEY and KRAKEN_SPOT_API_SECRET must be set in environment"
            )
        self._api_key    = api_key
        self._api_secret = api_secret

    def get_headers(self, uri_path: str, post_data: dict) -> dict:
        """Return auth headers and inject nonce into post_data in-place.

        Args:
            uri_path:  Path only, e.g. "/0/private/AddOrder"
            post_data: Mutable dict of POST params (nonce is added here)

        Returns:
            {"API-Key": ..., "API-Sign": ...}
        """
        nonce = str(int(time.time() * 1000))
        post_data["nonce"] = nonce

        post_str     = urllib.parse.urlencode(post_data)
        sha256_input = (nonce + post_str).encode("utf-8")
        sha256_hash  = hashlib.sha256(sha256_input).digest()

        message  = uri_path.encode("utf-8") + sha256_hash
        mac      = hmac.new(base64.b64decode(self._api_secret), message, hashlib.sha512)
        api_sign = base64.b64encode(mac.digest()).decode("utf-8")

        return {
            "API-Key":  self._api_key,
            "API-Sign": api_sign,
        }

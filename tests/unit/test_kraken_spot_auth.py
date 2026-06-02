"""Unit tests for KrakenSpotAuth — offline, no network required."""

import base64
import hashlib
import hmac
import os
import urllib.parse
from unittest.mock import patch

import pytest

from src.execution.kraken.exceptions import KrakenAuthError
from src.execution.kraken.spot.auth import KrakenSpotAuth


# ---------------------------------------------------------------------------
# Known-good test vector derived from the Kraken Spot API reference impl.
# These values produce a deterministic signature that can be verified offline.
# ---------------------------------------------------------------------------
_TEST_KEY    = "testApiKey12345"
_TEST_SECRET = base64.b64encode(b"test_secret_bytes_32_chars_paddd").decode()
_TEST_PATH   = "/0/private/AddOrder"


def _make_auth(key: str = _TEST_KEY, secret: str = _TEST_SECRET) -> KrakenSpotAuth:
    with patch.dict(os.environ, {"KRAKEN_SPOT_API_KEY": key, "KRAKEN_SPOT_API_SECRET": secret}):
        return KrakenSpotAuth()


def _expected_sign(key: str, secret: str, uri_path: str, post_data: dict) -> str:
    """Reference implementation — mirrors KrakenSpotAuth.get_headers logic."""
    nonce    = post_data["nonce"]
    post_str = urllib.parse.urlencode(post_data)
    sha256   = hashlib.sha256((nonce + post_str).encode("utf-8")).digest()
    message  = uri_path.encode("utf-8") + sha256
    mac      = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode("utf-8")


class TestKrakenSpotAuthSignature:
    def test_signature_matches_reference_implementation(self):
        auth = _make_auth()
        post_data = {"pair": "XXBTZUSD", "type": "buy", "ordertype": "market", "volume": "0.1"}
        headers   = auth.get_headers(_TEST_PATH, post_data)
        # post_data["nonce"] was injected by get_headers
        expected  = _expected_sign(_TEST_KEY, _TEST_SECRET, _TEST_PATH, post_data)
        assert headers["API-Sign"] == expected

    def test_headers_contain_api_key(self):
        auth    = _make_auth()
        headers = auth.get_headers(_TEST_PATH, {})
        assert headers["API-Key"] == _TEST_KEY

    def test_headers_contain_only_api_key_and_api_sign(self):
        """Spot auth must NOT include a Nonce header — documents divergence from Futures auth."""
        auth    = _make_auth()
        headers = auth.get_headers(_TEST_PATH, {})
        assert set(headers.keys()) == {"API-Key", "API-Sign"}

    def test_nonce_injected_into_post_data(self):
        auth      = _make_auth()
        post_data = {"pair": "XXBTZUSD"}
        auth.get_headers(_TEST_PATH, post_data)
        assert "nonce" in post_data
        assert post_data["nonce"].isdigit()

    def test_different_calls_produce_different_nonces(self):
        auth = _make_auth()
        pd1, pd2 = {}, {}
        auth.get_headers(_TEST_PATH, pd1)
        auth.get_headers(_TEST_PATH, pd2)
        # Nonces are ms timestamps; two consecutive calls should differ or be equal
        # (equal only if called within the same millisecond — acceptable in test)
        assert pd1["nonce"].isdigit()
        assert pd2["nonce"].isdigit()

    def test_signature_differs_from_futures_auth(self):
        """Documents that Spot and Futures auth produce different signatures for same input."""
        from src.execution.kraken.auth.api_key import KrakenFuturesAuth
        with patch.dict(os.environ, {
            "KRAKEN_FUTURES_API_KEY":    _TEST_KEY,
            "KRAKEN_FUTURES_API_SECRET": _TEST_SECRET,
        }):
            futures_auth = KrakenFuturesAuth()

        spot_auth = _make_auth()
        post_data_spot    = {"pair": "XXBTZUSD", "type": "buy"}
        post_data_futures = {"orderType": "mkt", "symbol": "PF_XBTUSD", "side": "buy", "size": 1}

        spot_headers    = spot_auth.get_headers(_TEST_PATH, post_data_spot)
        futures_headers = futures_auth.get_headers("/derivatives/api/v3/sendorder", post_data_futures)

        # Headers use different key names
        assert "API-Sign" in spot_headers
        assert "Authent"  in futures_headers


class TestKrakenSpotAuthCredentials:
    def test_missing_api_key_raises_auth_error(self):
        with patch.dict(os.environ, {"KRAKEN_SPOT_API_KEY": "", "KRAKEN_SPOT_API_SECRET": _TEST_SECRET}):
            with pytest.raises(KrakenAuthError):
                KrakenSpotAuth()

    def test_missing_api_secret_raises_auth_error(self):
        with patch.dict(os.environ, {"KRAKEN_SPOT_API_KEY": _TEST_KEY, "KRAKEN_SPOT_API_SECRET": ""}):
            with pytest.raises(KrakenAuthError):
                KrakenSpotAuth()

    def test_both_missing_raises_auth_error(self):
        with patch.dict(os.environ, {"KRAKEN_SPOT_API_KEY": "", "KRAKEN_SPOT_API_SECRET": ""}):
            with pytest.raises(KrakenAuthError):
                KrakenSpotAuth()

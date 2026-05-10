"""Kraken Futures exception hierarchy."""


class KrakenError(Exception):
    """Base exception for all Kraken Futures errors."""


class KrakenAuthError(KrakenError):
    """Invalid credentials or malformed HMAC signature. Do not retry."""


class KrakenAPIError(KrakenError):
    """Non-auth API error (HTTP 4xx/5xx). Includes status code and response body."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Kraken API HTTP {status_code}: {body[:200]}")


class KrakenOrderError(KrakenError):
    """Order rejected by Kraken (invalid price, size, insufficient margin, etc.)."""

    def __init__(self, message: str, raw: dict | None = None) -> None:
        self.raw = raw or {}
        super().__init__(message)

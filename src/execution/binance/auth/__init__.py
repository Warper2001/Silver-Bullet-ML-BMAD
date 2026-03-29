"""
Binance Authentication Module

This module provides API key authentication for Binance API requests.
Binance uses API Key + HMAC SHA256 signature (simpler than OAuth).

API Documentation: https://binance-docs.github.io/apidocs/#endpoint-security-type
"""

from src.execution.binance.auth.api_key_auth import ApiKeyAuth, create_api_key_auth
from src.execution.binance.auth.signature import SignatureGenerator, create_signature_generator

__all__ = [
    "ApiKeyAuth",
    "create_api_key_auth",
    "SignatureGenerator",
    "create_signature_generator",
]

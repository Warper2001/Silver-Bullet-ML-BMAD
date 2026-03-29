"""
KuCoin Authentication Module

This module provides API key authentication for KuCoin API requests.

KuCoin uses API Key + Secret + Passphrase with HMAC SHA256 signature.

API Documentation: https://docs.kucoin.com/#authentication
"""

from src.execution.kucoin.auth.signature import SignatureGenerator, create_signature_generator

__all__ = [
    "SignatureGenerator",
    "create_signature_generator",
]

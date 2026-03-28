"""
PKCE (Proof Key for Code Exchange) Utilities for TradeStation OAuth 2.0

This module implements the PKCE extension to the Authorization Code flow,
which is required by TradeStation's API.

PKCE Flow:
1. Generate code_verifier (random string, 43-128 characters)
2. Create code_challenge = SHA256(code_verifier) then base64url-encode
3. Send code_challenge in authorization request
4. Exchange authorization code + code_verifier for tokens

TradeStation Requirements:
- code_verifier: 43-128 characters, random
- code_challenge_method: "S256" (SHA-256)
- Authorization URL: https://signin.tradestation.com/authorize
- Token URL: https://signin.tradestation.com/oauth/token
"""

import base64
import hashlib
import os
import secrets
import string
from typing import Literal


class PKCEHelper:
    """
    Helper class for PKCE (Proof Key for Code Exchange) operations.

    PKCE is used to secure public clients (native apps, SPAs) that cannot
    safely store a client_secret.

    Attributes:
        code_verifier: Random string used to generate code_challenge
        code_challenge: Hashed version of code_verifier sent to authorization server
        code_challenge_method: Always "S256" for SHA-256

    Example:
        pkce = PKCEHelper()
        print(f"Code Verifier: {pkce.code_verifier}")
        print(f"Code Challenge: {pkce.code_challenge}")

        # Use in authorization URL
        auth_url = f"https://signin.tradestation.com/authorize?code_challenge={pkce.code_challenge}&code_challenge_method=S256&..."
    """

    # Valid characters for code_verifier (RFC 7636)
    VERIFIER_CHARS = string.ascii_letters + string.digits + "-._~"

    def __init__(self) -> None:
        """Initialize PKCE helper and generate code_verifier and code_challenge."""
        self.code_verifier = self._generate_code_verifier()
        self.code_challenge = self._create_code_challenge(self.code_verifier)
        self.code_challenge_method: Literal["S256"] = "S256"

    def _generate_code_verifier(self, length: int = 128) -> str:
        """
        Generate a cryptographically random code_verifier.

        Args:
            length: Length of verifier (must be 43-128 characters per RFC 7636)

        Returns:
            Random string using valid characters

        Raises:
            ValueError: If length is not in valid range
        """
        if not 43 <= length <= 128:
            raise ValueError(f"code_verifier length must be 43-128, got {length}")

        # Use cryptographically secure random generator
        return "".join(secrets.choice(self.VERIFIER_CHARS) for _ in range(length))

    def _create_code_challenge(self, code_verifier: str) -> str:
        """
        Create code_challenge from code_verifier.

        Process:
        1. Hash code_verifier with SHA-256
        2. Base64 URL-safe encode (no padding)

        Args:
            code_verifier: The code_verifier string

        Returns:
            Base64 URL-safe encoded SHA-256 hash

        Raises:
            ValueError: If code_verifier is invalid
        """
        if not code_verifier:
            raise ValueError("code_verifier cannot be empty")

        if len(code_verifier) < 43 or len(code_verifier) > 128:
            raise ValueError(f"code_verifier must be 43-128 chars, got {len(code_verifier)}")

        # SHA-256 hash
        digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()

        # Base64 URL-safe encode (remove padding)
        challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

        return challenge

    @staticmethod
    def generate_state(length: int = 32) -> str:
        """
        Generate a state parameter for CSRF protection.

        Args:
            length: Length of state string

        Returns:
            Random URL-safe string
        """
        return secrets.token_urlsafe(length)[:length]

    @staticmethod
    def validate_code_verifier(code_verifier: str) -> bool:
        """
        Validate a code_verifier string.

        Args:
            code_verifier: String to validate

        Returns:
            True if valid, False otherwise
        """
        if not code_verifier:
            return False

        if not 43 <= len(code_verifier) <= 128:
            return False

        # Check all characters are valid
        if not all(c in PKCEHelper.VERIFIER_CHARS for c in code_verifier):
            return False

        return True


def generate_pkce_pair() -> tuple[str, str]:
    """
    Generate a PKCE code_verifier and code_challenge pair.

    Convenience function for quick PKCE pair generation.

    Returns:
        Tuple of (code_verifier, code_challenge)

    Example:
        verifier, challenge = generate_pkce_pair()
        print(f"Verifier: {verifier}")
        print(f"Challenge: {challenge}")
    """
    pkce = PKCEHelper()
    return pkce.code_verifier, pkce.code_challenge

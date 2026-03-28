"""
TradeStation SDK - Authentication Module

This module provides OAuth 2.0 authentication for the TradeStation API.

Components:
- TokenManager: Manages token lifecycle and automatic refresh
- OAuth2Client: Handles OAuth 2.0 flows (Authorization Code + Client Credentials)

Usage:
    # SIM environment (Client Credentials flow)
    token_mgr = TokenManager(env="sim")
    await token_mgr.initialize()

    # LIVE environment (Authorization Code flow)
    token_mgr = TokenManager(env="live")
    await token_mgr.initialize()  # Prompts user for authorization
"""

from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager

__all__ = ["TokenManager", "OAuth2Client"]

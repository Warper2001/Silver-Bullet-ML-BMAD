#!/usr/bin/env python3
"""Test OAuth URL generation for TradeStation."""

from urllib.parse import urlencode

AUTHORIZATION_ENDPOINT = "https://signin.tradestation.com/authorize"

# Your credentials
CLIENT_ID = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
REDIRECT_URI = "http://localhost:8080/callback"
STATE = "test_state_12345"

# Option 1: Original scopes (what the code uses)
params_original = {
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "audience": "https://api.tradestation.com",
    "scope": "openid profile offline_access MarketData ReadAccount Trade",
    "state": STATE,
}

# Option 2: Minimal scopes (for MarketData only)
params_minimal = {
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "scope": "openid offline_access MarketData",
    "state": STATE,
}

# Option 3: Without audience
params_no_audience = {
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "scope": "openid offline_access MarketData",
    "state": STATE,
}

print("=" * 70)
print("Option 1: Original (with audience & full scopes)")
print("=" * 70)
print(f"{AUTHORIZATION_ENDPOINT}?{urlencode(params_original)}")
print()

print("=" * 70)
print("Option 2: Minimal scopes (MarketData only)")
print("=" * 70)
print(f"{AUTHORIZATION_ENDPOINT}?{urlencode(params_minimal)}")
print()

print("=" * 70)
print("Option 3: Without audience parameter")
print("=" * 70)
print(f"{AUTHORIZATION_ENDPOINT}?{urlencode(params_no_audience)}")
print()

print("=" * 70)
print("IMPORTANT: Verify these settings in your TradeStation Developer Portal:")
print("=" * 70)
print(f"1. Client ID: {CLIENT_ID}")
print(f"2. Redirect URI: {REDIRECT_URI}")
print("   - This must match EXACTLY in your TradeStation app settings")
print("3. Make sure the app is enabled for the correct environment (SIM vs PROD)")
print("=" * 70)

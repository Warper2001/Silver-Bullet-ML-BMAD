#!/usr/bin/env python3
"""
Generate TradeStation Authorization URL

This script generates the authorization URL you need to visit.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.execution.tradestation.auth.oauth import OAuth2Client

# Your credentials
client_id = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
redirect_uri = "http://localhost:8080"

print("=" * 70)
print("TradeStation OAuth Authorization URL Generator")
print("=" * 70)
print()

# Create OAuth client
oauth_client = OAuth2Client(
    client_id=client_id,
    redirect_uri=redirect_uri,
)

# Generate authorization URL
auth_url = oauth_client.get_authorization_url()

print("✅ OAuth Client Initialized")
print(f"   Client ID: {client_id}")
print(f"   Redirect URI: {redirect_uri}")
print()
print("📋 Authorization URL:")
print("-" * 70)
print(auth_url)
print("-" * 70)
print()

print("📝 NEXT STEPS:")
print("1. Copy the URL above")
print("2. Paste it into your browser")
print("3. Log in with your TradeStation credentials")
print("4. Authorize the application")
print("5. You'll be redirected to: http://localhost:8080/?code=AUTH_CODE")
print()

print("⚠️  The callback server is NOT running yet.")
print("   Once you have the authorization code, run:")
print("   python exchange_auth_code.py <authorization_code>")
print()

print("=" * 70)

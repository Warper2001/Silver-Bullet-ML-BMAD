#!/usr/bin/env python3
"""Exchange authorization code for access token."""

import sys
import httpx
from urllib.parse import parse_qs

# Your credentials
CLIENT_ID = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
CLIENT_SECRET = "Ut3JTMUQoBcpIn7-8rUtB7tSm3Xi_GcrXl0QkpWhkgPIrueUtdRSho4gcHSeK7vc"
REDIRECT_URI = "http://localhost:8080"

# Get authorization code from command line
if len(sys.argv) > 1:
    AUTH_CODE = sys.argv[1]
else:
    print("❌ Error: No authorization code provided")
    print()
    print("Usage: .venv/bin/python exchange_token_simple.py <authorization_code>")
    sys.exit(1)

TOKEN_ENDPOINT = "https://signin.tradestation.com/oauth/token"

print("="*70)
print("🔄 Exchanging Authorization Code for Access Token")
print("="*70)
print()
print(f"Authorization Code: {AUTH_CODE[:30]}...")
print()

print("Exchanging code for tokens...")

# Exchange the authorization code for tokens
data = {
    "grant_type": "authorization_code",
    "code": AUTH_CODE,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI,
}

try:
    response = httpx.post(
        TOKEN_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0
    )
    response.raise_for_status()

    token_data = response.json()

    print("\n✅ SUCCESS! Tokens received:")
    print("="*70)
    print(f"Access Token: {token_data.get('access_token', '')[:50]}...")
    print(f"Refresh Token: {token_data.get('refresh_token', '')[:50]}...")
    print(f"Expires In: {token_data.get('expires_in', '')} seconds")
    print(f"Token Type: {token_data.get('token_type', '')}")
    print("="*70)
    print()

    # Save tokens to .env file
    with open(".env", "a") as f:
        f.write(f"\n# Updated tokens from OAuth flow\n")
        f.write(f"TRADESTATION_REFRESH_TOKEN={token_data.get('refresh_token', '')}\n")

    print("✅ Refresh token saved to .env file")
    print()

    # Also save access token for immediate use
    with open(".access_token", "w") as f:
        f.write(token_data.get('access_token', ''))

    print("✅ Access token saved to .access_token")
    print()
    print("🎉 Authentication complete! You can now run:")
    print()
    print("   .venv/bin/python start_paper_trading.py")

except httpx.HTTPStatusError as e:
    print(f"\n❌ ERROR: HTTP {e.response.status_code}")
    print(f"Response: {e.response.text}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

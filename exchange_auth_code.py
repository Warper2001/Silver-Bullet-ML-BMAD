#!/usr/bin/env python3
"""Exchange OAuth authorization code for access/refresh tokens."""

import httpx
from urllib.parse import parse_qs

# Your credentials
CLIENT_ID = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
CLIENT_SECRET = "Ut3JTMUQoBcpIn7-8rUtB7tSm3Xi_GcrXl0QkpWhkgPIrueUtdRSho4gcHSeK7vc"
REDIRECT_URI = "http://localhost:8080"

# Extract from your callback URL
CALLBACK_URL = "http://localhost:8080/?code=GTDA4JqL9jvdzfibay3p7VX4mwesJ1_V7KMfagcbLKMnK&state=ty8Fc4J3JRuDVbcTAqEwRZGPioT_UDzZVNZF-PE010E"

TOKEN_ENDPOINT = "https://signin.tradestation.com/oauth/token"

# Parse the callback URL to get the authorization code
parsed = parse_qs(CALLBACK_URL.split("?", 1)[1])
auth_code = parsed["code"][0]
state = parsed["state"][0]

print(f"Authorization Code: {auth_code}")
print(f"State: {state}")
print()
print("Exchanging code for tokens...")

# Exchange the authorization code for tokens
data = {
    "grant_type": "authorization_code",
    "code": auth_code,
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
    print("=" * 70)
    print(f"Access Token: {token_data.get('access_token', '')[:50]}...")
    print(f"Refresh Token: {token_data.get('refresh_token', '')[:50]}...")
    print(f"Expires In: {token_data.get('expires_in', '')} seconds")
    print(f"Token Type: {token_data.get('token_type', '')}")
    print("=" * 70)
    print("\nNow update your .env file with:")
    print(f"TRADESTATION_REFRESH_TOKEN={token_data.get('refresh_token', '')}")
    print("\nThen run the downloader in batch mode:")
    print(".venv/bin/python -m src.data.cli --batch-mode")

except httpx.HTTPStatusError as e:
    print(f"\n❌ ERROR: HTTP {e.response.status_code}")
    print(f"Response: {e.response.text}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")

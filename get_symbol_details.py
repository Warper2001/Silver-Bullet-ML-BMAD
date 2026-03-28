#!/usr/bin/env python3
"""Get symbol details from TradeStation."""

import httpx
import json

CLIENT_ID = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
CLIENT_SECRET = "Ut3JTMUQoBcpIn7-8rUtB7tSm3Xi_GcrXl0QkpWhkgPIrueUtdRSho4gcHSeK7vc"
REFRESH_TOKEN = "DPhH3vulejK39v79XyXcNwJorgMWtSza5R6__nolI0YpE"

# Get access token
TOKEN_ENDPOINT = "https://signin.tradestation.com/oauth/token"
data = {
    "grant_type": "refresh_token",
    "refresh_token": REFRESH_TOKEN,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}

response = httpx.post(TOKEN_ENDPOINT, data=data, timeout=30.0)
token_data = response.json()
access_token = token_data["access_token"]

headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json",
}

# Get symbol details
symbol = "MNQM26"
url = f"https://api.tradestation.com/v3/marketdata/symbols/{symbol}"

response = httpx.get(url, headers=headers, timeout=30.0)

print(f"Status: {response.status_code}")
data = response.json()
print(json.dumps(data, indent=2))

# Check if there are symbols returned
if "Symbols" in data and data["Symbols"]:
    print(f"\n✅ Found {len(data['Symbols'])} matching symbols")
    for s in data["Symbols"]:
        print(f"\n{json.dumps(s, indent=2)}")

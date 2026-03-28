#!/usr/bin/env python3
"""Test TradeStation symbol lookup."""

import httpx
from datetime import datetime, timezone

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

# Test various symbol formats
test_symbols = [
    "MNQM26",
    "mnqm26",
    "MNQ",
    "MNQH26",
    ".MNQM26",
    "MNQM26.CME",
]

print("Testing various symbol formats with symbol lookup:\n")

for symbol in test_symbols:
    # Try symbol lookup endpoint
    url = f"https://api.tradestation.com/v3/marketdata/symbols/{symbol}"
    response = httpx.get(url, headers=headers, timeout=30.0)

    print(f"Symbol: {symbol:15} -> Status: {response.status_code:3}  ", end="")
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and data:
            print(f"Found! Type: {data[0].get('SymbolType', 'N/A')}, Description: {data[0].get('Description', 'N/A')[:50]}")
        elif isinstance(data, dict):
            print(f"Found! Keys: {list(data.keys())[:5]}")
        else:
            print(f"Found! Data type: {type(data)}")
    elif response.status_code == 404:
        print("Not found")
    else:
        print(f"Error: {response.text[:100]}")

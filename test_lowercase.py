#!/usr/bin/env python3
"""Test TradeStation API with lowercase symbol."""

import httpx

# Get access token
import json
from datetime import datetime, timezone, timedelta

CLIENT_ID = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
CLIENT_SECRET = "Ut3JTMUQoBcpIn7-8rUtB7tSm3Xi_GcrXl0QkpWhkgPIrueUtdRSho4gcHSeK7vc"
REFRESH_TOKEN = "DPhH3vulejK39v79XyXcNwJorgMWtSza5R6__nolI0YpE"

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

# Test with lowercase symbol
symbol = "mnqm26"  # Lowercase!
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=7)

url = f"https://api.tradestation.com/v3/marketdata/bars/{symbol}"
params = {
    "interval": "1",
    "unit": "Minute",
    "startDate": start_date.strftime("%Y-%m-%d"),
    "endDate": end_date.strftime("%Y-%m-%d"),
}

headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json",
}

print(f"Testing with lowercase symbol: {symbol}")
print(f"URL: {url}")
print()

response = httpx.get(url, params=params, headers=headers, timeout=30.0)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    data = response.json()
    if "Bars" in data and data["Bars"]:
        print(f"\n✅ SUCCESS! Got {len(data['Bars'])} bars")
        print(f"First bar: {data['Bars'][0]}")
    else:
        print("\n⚠️  Got 200 but no bars in response")

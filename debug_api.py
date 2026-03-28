#!/usr/bin/env python3
"""Debug TradeStation API request."""

import httpx
import json
from datetime import datetime, timezone, timedelta

# Your credentials
CLIENT_ID = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
CLIENT_SECRET = "Ut3JTMUQoBcpIn7-8rUtB7tSm3Xi_GcrXl0QkpWhkgPIrueUtdRSho4gcHSeK7vc"

# Get a fresh access token
TOKEN_ENDPOINT = "https://signin.tradestation.com/oauth/token"

refresh_token = "DPhH3vulejK39v79XyXcNwJorgMWtSza5R6__nolI0YpE"

data = {
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}

print("Getting access token...")
response = httpx.post(TOKEN_ENDPOINT, data=data, timeout=30.0)
token_data = response.json()
access_token = token_data["access_token"]

print(f"Access token: {access_token[:50]}...")

# Test API request
symbol = "MNQM26"
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

print(f"\nTesting API request:")
print(f"URL: {url}")
print(f"Params: {params}")
print()

response = httpx.get(url, params=params, headers=headers, timeout=30.0)

print(f"Status Code: {response.status_code}")
print(f"Response Headers:")
for key, value in response.headers.items():
    print(f"  {key}: {value}")
print()
print(f"Response Body:")
print(response.text)

#!/usr/bin/env python3
"""Test TradeStation quotes endpoint."""

import httpx
import json
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

# Test different endpoints
symbol = "MNQM26"

print("Testing various TradeStation API endpoints:\n")

# 1. Quotes endpoint
url = f"https://api.tradestation.com/v3/marketdata/quotes/{symbol}"
response = httpx.get(url, headers=headers, timeout=30.0)
print(f"1. Quotes Endpoint (/quotes/{symbol}):")
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    print(f"   ✅ SUCCESS! Response: {response.text[:200]}")
else:
    print(f"   ❌ Error: {response.text[:100]}")
print()

# 2. Stream/bars endpoint
url = f"https://api.tradestation.com/v3/marketdata/bars/{symbol}"
params = {"interval": "1", "unit": "Minute", "startDate": "2026-03-15", "endDate": "2026-03-22"}
response = httpx.get(url, params=params, headers=headers, timeout=30.0)
print(f"2. Bars Endpoint (/bars/{symbol}):")
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    print(f"   ✅ SUCCESS! Response: {response.text[:200]}")
else:
    print(f"   ❌ Error: {response.text[:100]}")
print()

# 3. Try with a stock symbol (AAPL) to test if it's futures-specific
print("3. Testing with stock symbol (AAPL):")
url = "https://api.tradestation.com/v3/marketdata/bars/AAPL"
params = {"interval": "1", "unit": "Daily", "startDate": "2026-03-01", "endDate": "2026-03-22"}
response = httpx.get(url, params=params, headers=headers, timeout=30.0)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    print(f"   ✅ SUCCESS! Bars endpoint works for stocks")
else:
    print(f"   ❌ Error: {response.text[:100]}")

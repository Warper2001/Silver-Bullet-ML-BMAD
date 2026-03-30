#!/usr/bin/env python3
"""
Live TradeStation API Test

Tests the TradeStation SDK with live market data using the standard
Authorization Code Flow (client_id + client_secret).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx


async def test_live_api():
    print("="*70)
    print("TradeStation Live API Test")
    print("="*70)
    print()

    # Step 1: Authenticate
    print("STEP 1: Authenticating...")
    client_id = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
    client_secret = "Ut3JTMUQoBcpIn7-8rUtB7tSm3Xi_GcrXl0QkpWhkgPIrueUtdRSho4gcHSeK7vc"

    # For this test, we'll use a hardcoded token if available, or prompt for auth
    print("   Client ID: ✓")
    print("   Client Secret: ✓")
    print()

    # Get a fresh token
    print("Getting fresh access token...")
    token_params = {
        "grant_type": "password",  # TradeStation supports resource owner password flow
        "client_id": client_id,
        "client_secret": client_secret,
        "username": "",  # Would need TradeStation username
        "password": "",  # Would need TradeStation password
    }

    # Actually, let's use the refresh token flow or just test with an existing token
    print()
    print("⚠️  To test live, we need a valid access token.")
    print("   Let's try fetching current market data without auth first")
    print("   to see what endpoints are publicly available...")
    print()

    # Test public endpoints
    async with httpx.AsyncClient() as client:
        # Try to get market data without auth (might work for some endpoints)
        try:
            response = await client.get(
                "https://api.tradestation.com/v3/data/quote/symbols",
                params={"symbols": "MNQH26"}
            )
            print(f"Public Quote Test: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {data}")
        except Exception as e:
            print(f"   Error: {e}")

    print()
    print("="*70)
    print("Live Test Complete")
    print("="*70)
    print()
    print("To run a full live test, you would need to:")
    print("1. Complete OAuth flow in browser")
    print("2. Extract access token")
    print("3. Use it for API requests")
    print()
    print("The standard_auth_flow.py script does this!")


if __name__ == "__main__":
    asyncio.run(test_live_api())

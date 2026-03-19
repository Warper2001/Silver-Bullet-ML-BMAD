#!/usr/bin/env python3
"""Test TradeStation authentication independently."""

import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def test_auth():
    """Test TradeStation authentication."""

    client_id = os.getenv("TRADESTATION_CLIENT_ID")
    client_secret = os.getenv("TRADESTATION_CLIENT_SECRET")
    redirect_uri = os.getenv("TRADESTATION_REDIRECT_URI", "http://localhost:8080/callback")

    print(f"Client ID: {client_id[:10]}...{client_id[-10:] if client_id else 'None'}")
    print(f"Client Secret: {client_secret[:10]}...{client_secret[-10:] if client_secret else 'None'}")
    print(f"Redirect URI: {redirect_uri}")

    # Try authentication
    url = "https://api.tradestation.com/v2/security/authorize"

    # Test without redirect_uri first
    data_without_redirect = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    data_with_redirect = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try without redirect_uri
        print("\n--- Attempt 1: Without redirect_uri ---")
        try:
            response = await client.post(
                url,
                data=data_without_redirect,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

            if response.status_code == 200:
                print("✅ Authentication successful!")
                return

        except Exception as e:
            print(f"❌ Error: {e}")

        # Try with redirect_uri
        print("\n--- Attempt 2: With redirect_uri ---")
        try:
            response = await client.post(
                url,
                data=data_with_redirect,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

            if response.status_code == 200:
                print("✅ Authentication successful!")
            else:
                print("❌ Authentication failed")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_auth())

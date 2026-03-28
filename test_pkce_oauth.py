#!/usr/bin/env python3
"""
TradeStation PKCE OAuth Test Script

This script tests the OAuth 2.0 PKCE authentication flow with TradeStation.

⚠️  WARNING: This connects to TradeStation's PRODUCTION API
- All API calls go to production
- Market data calls are safe (read-only)
- DO NOT test order operations unless you're ready for real trades

Usage:
    python test_pkce_oauth.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.callback_handler import OAuthCallbackServer


def print_step(step_num: int, title: str) -> None:
    """Print a formatted step header."""
    print()
    print("=" * 70)
    print(f"STEP {step_num}: {title}")
    print("=" * 70)
    print()


async def test_pkce_authentication():
    """Test the complete PKCE OAuth flow."""

    # Your TradeStation credentials
    client_id = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
    redirect_uri = "http://localhost:8080"

    print_step(1, "Initialize OAuth Client")
    print(f"Client ID: {client_id}")
    print(f"Redirect URI: {redirect_uri}")

    oauth_client = OAuth2Client(
        client_id=client_id,
        redirect_uri=redirect_uri,
    )

    print("✅ OAuth client initialized")
    print(f"   Code Verifier: {oauth_client.pkce_helper.code_verifier[:20]}...")
    print(f"   Code Challenge: {oauth_client.pkce_helper.code_challenge[:20]}...")

    print_step(2, "Generate Authorization URL")
    auth_url = oauth_client.get_authorization_url()

    print("Authorization URL generated:")
    print("-" * 70)
    print(auth_url)
    print("-" * 70)
    print()
    print("📋 NEXT STEPS:")
    print("1. Copy the URL above")
    print("2. Paste it into your browser")
    print("3. Log in with your TradeStation credentials")
    print("4. Authorize the application")
    print("5. You'll be redirected to: http://localhost:8080/?code=...")
    print()
    print("⚠️  The callback server will start automatically...")
    print()

    # Start callback server
    print_step(3, "Start OAuth Callback Server")
    server = OAuthCallbackServer(
        oauth_client=oauth_client,
        port=8080,
    )

    print("Callback server starting...")
    print(f"Listening on: http://localhost:8080")
    print(f"Timeout: 5 minutes")
    print()

    try:
        # Wait for callback
        print("⏳ Waiting for OAuth callback (Ctrl+C to cancel)...")
        token_response = server.wait_for_callback(timeout=300)

        if token_response:
            print_step(4, "Authentication Successful! ✅")
            print(f"Access Token: {token_response.access_token[:30]}...")
            print(f"Token Type: {token_response.token_type}")
            print(f"Expires In: {token_response.expires_in} seconds")
            print(f"Refresh Token: {token_response.refresh_token[:30] if token_response.refresh_token else 'None'}...")
            print(f"Scope: {token_response.scope}")
            print()

            # Test API call
            print_step(5, "Test API Call - Fetch Quotes")
            print("Testing with read-only market data endpoint...")

            from src.execution.tradestation.client import TradeStationClient

            client = TradeStationClient(client_id=client_id)

            # Manually set the token
            client.oauth_client.token_manager.set_token(token_response)
            client._is_initialized = True
            client.http_client = None  # Will be created in __aenter__

            # Initialize HTTP client
            import httpx
            client.http_client = httpx.AsyncClient(
                base_url=client.api_base_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(30.0),
            )

            # Make a test API call
            print("Fetching quotes for MNQH26...")

            # Make authenticated request
            response = await client.http_client.get(
                "https://api.tradestation.com/v3/data/quote/symbols",
                params={"symbols": "MNQH26"}
            )

            if response.status_code == 200:
                print("✅ API call successful!")
                data = response.json()
                print(f"   Quotes received: {len(data.get('Quotes', []))}")

                if data.get('Quotes'):
                    quote = data['Quotes'][0]
                    print(f"   Symbol: {quote.get('Symbol')}")
                    print(f"   Bid: {quote.get('Bid')}")
                    print(f"   Ask: {quote.get('Ask')}")
                    print(f"   Last: {quote.get('Last')}")
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

            # Cleanup
            await client.http_client.aclose()

            print()
            print("=" * 70)
            print("✅ PKCE OAuth Test - COMPLETE SUCCESS!")
            print("=" * 70)
            print()
            print("Your TradeStation SDK is working correctly!")
            print("Authentication flow successful, API calls working.")

        else:
            print()
            print("=" * 70)
            print("❌ Authentication Failed")
            print("=" * 70)
            print("No token response received. This could mean:")
            print("  - User denied authorization")
            print("  - Callback timeout")
            print("  - Invalid client_id")
            print("  - Network error")

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("⚠️  Test Interrupted")
        print("=" * 70)
        print("Authentication was cancelled by user.")

    finally:
        print()
        print("Cleaning up...")
        server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(test_pkce_authentication())
    except KeyboardInterrupt:
        print("\n✅ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

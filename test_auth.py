#!/usr/bin/env python3
"""Test TradeStation Authorization Code Flow authentication."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.auth_web import TradeStationAuthWeb
from src.data.exceptions import AuthenticationError


async def test_auth():
    """Test TradeStation Authorization Code Flow authentication."""

    print("="*70)
    print("🔐 TRADESTATION AUTHENTICATION TEST")
    print("="*70)
    print("\nThis will test the Authorization Code Flow authentication.")
    print("A browser window will open for you to log in.\n")

    try:
        # Initialize authentication
        auth = TradeStationAuthWeb(port=8080)

        # Get access token (will open browser)
        print("Requesting access token...")
        access_token = await auth.get_access_token()

        print("\n" + "="*70)
        print("✅ AUTHENTICATION SUCCESSFUL!")
        print("="*70)
        print(f"\nAccess Token (first 20 chars): {access_token[:20]}...")
        print(f"Token Length: {len(access_token)} characters")

        if auth._token_expires_at:
            print(f"Expires At: {auth._token_expires_at}")

        if auth._refresh_token:
            print(f"Refresh Token: Available ({len(auth._refresh_token)} chars)")

        print("\n✅ Your TradeStation API credentials are working correctly!")
        print("\nYou can now:")
        print("  - Run: venv/bin/python collect_historical_data.py")
        print("  - Run: venv/bin/python collect_realtime_data.py")

        # Clean up
        await auth.cleanup()

        return True

    except AuthenticationError as e:
        print("\n" + "="*70)
        print("❌ AUTHENTICATION FAILED")
        print("="*70)
        print(f"\nError: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API credentials in .env file")
        print("  2. User cancelled authentication in browser")
        print("  3. Callback URL not configured correctly in TradeStation")
        print("  4. Network connectivity issues")
        print("\nSolutions:")
        print("  1. Verify credentials in .env file")
        print("  2. Check callback URL: http://localhost:8080/callback")
        print("  3. Ensure TradeStation account is active")
        print("  4. Try authentication again")

        return False

    except Exception as e:
        print("\n" + "="*70)
        print("❌ UNEXPECTED ERROR")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = asyncio.run(test_auth())
    sys.exit(0 if success else 1)

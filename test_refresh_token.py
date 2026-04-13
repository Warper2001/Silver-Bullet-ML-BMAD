#!/usr/bin/env python3
"""Test TradeStation token refresh."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.data.auth_v3 import TradeStationAuthV3


async def main():
    """Test token refresh."""
    print("="*70)
    print("TradeStation Token Refresh Test")
    print("="*70)
    print()

    # Load settings
    settings = load_settings()
    print(f"✅ Client ID: {settings.tradestation_client_id}")
    print(f"✅ Refresh Token: {settings.tradestation_refresh_token[:20]}...")
    print()

    # Load current access token
    try:
        with open(".access_token", "r") as f:
            current_token = f.read().strip()
        print(f"✅ Current Access Token: {current_token[:50]}...")
    except Exception as e:
        print(f"❌ Failed to load access token: {e}")
        return False

    # Create auth instance
    auth = TradeStationAuthV3(
        access_token=current_token,
        refresh_token=settings.tradestation_refresh_token,
    )

    print()
    print("🔄 Attempting token refresh...")
    print("-"*70)

    try:
        # Attempt refresh
        await auth._refresh_token_flow()

        print()
        print("✅ Token refresh successful!")
        print(f"   New Token: {auth._access_token[:50]}...")
        print(f"   Expires At: {auth._token_expires_at}")
        print()

        # Save new token
        with open(".access_token", "w") as f:
            f.write(auth._access_token)
        print("✅ Saved new access token to .access_token")

        # Cleanup
        await auth.cleanup()

        return True

    except Exception as e:
        print()
        print(f"❌ Token refresh failed: {e}")
        print()
        print("Possible causes:")
        print("  1. Refresh token is expired")
        print("  2. Refresh token is invalid")
        print("  3. Client credentials are incorrect")
        print()
        print("Solution: Re-authenticate using OAuth flow")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)

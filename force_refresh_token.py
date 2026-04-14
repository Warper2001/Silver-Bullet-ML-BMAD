#!/usr/bin/env python3
"""Force refresh TradeStation access token using refresh token."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.config import load_settings
from src.data.auth_v3 import TradeStationAuthV3


async def main():
    """Force token refresh and save to .access_token file."""

    # Load refresh token from settings
    settings = load_settings()
    refresh_token = settings.tradestation_refresh_token if hasattr(settings, 'tradestation_refresh_token') else ""

    if not refresh_token:
        print("❌ No refresh token found in settings!")
        print("   Please set TRADESTATION_REFRESH_TOKEN in .env file")
        return False

    print(f"🔄 Forcing token refresh...")

    # Initialize auth with dummy access token that's already expired
    # This forces authenticate() to use the refresh token
    from datetime import datetime, timedelta
    expired_time = datetime.now() - timedelta(hours=1)  # Expired 1 hour ago

    auth = TradeStationAuthV3(
        access_token="dummy",
        refresh_token=refresh_token,
        token_expires_at=expired_time  # Mark as already expired
    )

    # Force token refresh by calling authenticate
    try:
        # The first authenticate() call should use the refresh token
        new_token = await auth.authenticate()
        print(f"✅ Token refreshed successfully")
        print(f"   New token (first 30 chars): {new_token[:30]}...")

        # Save to file
        with open(".access_token", "w") as f:
            f.write(new_token)
        print(f"✅ Saved to .access_token")

        # Verify token works
        print(f"\n🧪 Testing new token...")

        # Try to use the token
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.tradestation.com/v3/marketdata/quotes/MNQM26",
                headers={"Authorization": f"Bearer {new_token}"}
            )

            if response.status_code == 200:
                print(f"✅ Token verified - API access working!")
            else:
                print(f"❌ Token failed with status {response.status_code}")
                return False

        return True

    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

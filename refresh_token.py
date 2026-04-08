#!/usr/bin/env python3
"""Manually refresh OAuth token immediately."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.auth_v3 import TradeStationAuthV3
from src.data.config import load_settings


async def main():
    """Refresh token manually."""
    try:
        # Load auth from file
        auth = TradeStationAuthV3.from_file(".access_token")
        print(f"✅ Auth loaded (token hash: {auth._get_token_hash()})")

        # Refresh token immediately
        print("🔄 Refreshing token now...")
        await auth._refresh_token_flow()

        # Save new token
        with open(".access_token", "w") as f:
            f.write(auth._access_token)

        print(f"✅ Token refreshed successfully (new hash: {auth._get_token_hash()})")
        print(f"✅ Saved to .access_token")

        # Cleanup
        await auth.cleanup()

    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Refresh TradeStation access token using refresh token.

This script uses the saved refresh token to obtain a fresh access token
without requiring the user to go through the OAuth flow again.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager


async def refresh_token():
    """Refresh access token using refresh token."""
    print("="*70)
    print("🔄 Refreshing TradeStation Access Token")
    print("="*70)
    print()

    # Load settings (which contains the refresh token)
    settings = load_settings()

    # Check if refresh token is available
    refresh_token = settings.tradestation_refresh_token
    if not refresh_token:
        print("❌ No refresh token found in .env file")
        print()
        print("Please ensure TRADESTATION_REFRESH_TOKEN is set in .env")
        return False

    print(f"✅ Refresh Token found: {refresh_token[:30]}...")
    print()

    # Create OAuth client
    token_manager = TokenManager()
    oauth_client = OAuth2Client(
        client_id=settings.tradestation_client_id,
        redirect_uri=settings.tradestation_redirect_uri,
        token_manager=token_manager,
    )

    try:
        print("📝 Refreshing access token...")
        token_response = await oauth_client.refresh_token(refresh_token)

        print(f"✅ New Access Token: {token_response.access_token[:30]}...")
        print(f"✅ New Refresh Token: {token_response.refresh_token[:30]}...")
        print(f"✅ Expires In: {token_response.expires_in} seconds")
        print()

        # Save new access token to file
        token_file = Path(".access_token")
        token_file.write_text(token_response.access_token)
        print(f"✅ New access token saved to {token_file}")
        print()

        # Update .env with new refresh token (if it rotated)
        if token_response.refresh_token != refresh_token:
            print("📝 Refresh token rotated! Updating .env file...")
            env_file = Path(".env")
            content = env_file.read_text()

            # Replace the refresh token line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("TRADESTATION_REFRESH_TOKEN="):
                    lines[i] = f"TRADESTATION_REFRESH_TOKEN={token_response.refresh_token}"
                    break

            env_file.write_text('\n'.join(lines))
            print(f"✅ New refresh token saved to .env")
            print()

        print("="*70)
        print("✅ TOKEN REFRESH SUCCESSFUL!")
        print("="*70)
        print()
        print("You can now restart the paper trading system:")
        print()
        print("  .venv/bin/python start_paper_trading.py")
        print()

        return True

    except Exception as e:
        print(f"❌ Error refreshing token: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(refresh_token())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nRefresh cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

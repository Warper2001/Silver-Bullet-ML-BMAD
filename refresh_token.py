#!/usr/bin/env python3
"""
Refresh TradeStation access token using refresh token.

This script uses the refresh token from .env to get a fresh access token
for the paper trading system.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.execution.tradestation.auth.tokens import TokenManager, TokenResponse
from src.execution.tradestation.auth.oauth import OAuth2Client


async def main():
    """Refresh access token using refresh token."""
    print("🔄 Refreshing TradeStation access token...")

    settings = load_settings()

    # Create token manager
    token_manager = TokenManager()

    # Create OAuth client
    oauth_client = OAuth2Client(
        client_id=settings.tradestation_client_id,
        redirect_uri=settings.tradestation_redirect_uri,
        token_manager=token_manager,
    )

    # Use refresh token to get new access token
    if settings.tradestation_refresh_token:
        print(f"✅ Refresh token found: {settings.tradestation_refresh_token[:20]}...")

        # Refresh using the refresh_token method
        try:
            token_response = await oauth_client.refresh_token(
                settings.tradestation_refresh_token
            )
            access_token = token_response.access_token
            print(f"✅ Access token refreshed: {access_token[:20]}...")

            # Set the new token
            token_manager.set_token(token_response)

            # Save to file for paper trading system
            token_file = Path(".access_token")
            token_file.write_text(access_token)
            print(f"✅ Access token saved to {token_file}")

            return True
        except Exception as e:
            print(f"❌ Failed to refresh token: {e}")
            print("\nPlease run standard_auth_flow.py to authenticate.")
            return False
    else:
        print("❌ No refresh token found in .env file")
        print("\nPlease run standard_auth_flow.py to authenticate.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

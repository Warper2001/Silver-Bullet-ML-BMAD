#!/usr/bin/env python3
"""
Exchange authorization code for access token using OAuth2Client.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager


async def exchange_code(auth_code: str):
    """Exchange authorization code for access token."""
    print("="*70)
    print("🔄 Exchanging Authorization Code for Access Token")
    print("="*70)
    print()

    settings = load_settings()

    # Create token manager and OAuth client
    token_manager = TokenManager()
    oauth_client = OAuth2Client(
        client_id=settings.tradestation_client_id,
        client_secret=settings.tradestation_client_secret,
        redirect_uri=settings.tradestation_redirect_uri,
        token_manager=token_manager,
    )

    print(f"Authorization Code: {auth_code[:30]}...")
    print()

    try:
        # Exchange code for token
        print("📝 Exchanging code for token...")
        token_response = await oauth_client.exchange_code_for_token(auth_code)

        print(f"✅ Access Token: {token_response.access_token[:30]}...")
        print(f"✅ Refresh Token: {token_response.refresh_token[:30]}...")
        print(f"✅ Expires In: {token_response.expires_in} seconds")
        print()

        # Token is automatically saved to token manager
        print("✅ Tokens saved to token manager")
        print()

        # Verify token works
        access_token = await oauth_client.get_access_token()
        print(f"✅ Token verification: {access_token[:20]}...")
        print()

        # Save access token to file for easy use
        token_file = Path(".access_token")
        token_file.write_text(access_token)
        print(f"✅ Access token saved to {token_file}")
        print()

        # Update .env with refresh token
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("TRADESTATION_REFRESH_TOKEN="):
                    lines[i] = f"TRADESTATION_REFRESH_TOKEN={token_response.refresh_token}"
                    break
            else:
                lines.append(f"TRADESTATION_REFRESH_TOKEN={token_response.refresh_token}")
            env_file.write_text('\n'.join(lines))
            print("✅ Refresh token saved to .env")
            print()

        print("="*70)
        print("✅ AUTHENTICATION SUCCESSFUL!")
        print("="*70)
        print()
        print("You can now restart the paper trading system:")
        print()
        print("  pkill -f 'start_paper_trading.py'")
        print("  .venv/bin/python start_paper_trading.py")
        print()

        return True

    except Exception as e:
        print(f"❌ Error exchanging code: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: No authorization code provided")
        print()
        print("Usage: .venv/bin/python exchange_auth_code.py <authorization_code>")
        sys.exit(1)

    auth_code = sys.argv[1]
    success = asyncio.run(exchange_code(auth_code))
    sys.exit(0 if success else 1)

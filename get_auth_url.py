#!/usr/bin/env python3
"""
Generate TradeStation OAuth authorization URL.

Run this script to get a fresh authorization URL, open it in your browser,
and then use exchange_token_simple.py with the authorization code from the callback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager


def main():
    """Generate OAuth authorization URL."""
    print("="*70)
    print("TradeStation OAuth Authorization URL Generator")
    print("="*70)
    print()

    # Load settings
    settings = load_settings()

    print(f"✅ Client ID: {settings.tradestation_client_id}")
    print(f"✅ Redirect URI: {settings.tradestation_redirect_uri}")
    print()

    # Create OAuth client
    token_manager = TokenManager()
    oauth_client = OAuth2Client(
        client_id=settings.tradestation_client_id,
        redirect_uri=settings.tradestation_redirect_uri,
        token_manager=token_manager,
    )

    # Generate authorization URL
    print("📋 Generating Authorization URL")
    print("-"*70)
    print()

    auth_url = oauth_client.get_authorization_url()

    print(f"Authorization URL:")
    print(f"  {auth_url}")
    print()

    print("="*70)
    print("📝 INSTRUCTIONS:")
    print("="*70)
    print()
    print("1. Copy the authorization URL above")
    print("2. Open it in your web browser")
    print("3. Log in to TradeStation")
    print("4. Authorize the application")
    print("5. You'll be redirected to: http://localhost:8080/?code=...")
    print("6. Copy the 'code' parameter from the URL")
    print("7. Run this command to exchange the code for tokens:")
    print()
    print("   .venv/bin/python exchange_token_simple.py <code_from_url>")
    print()
    print("="*70)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Non-interactive TradeStation Authentication

Uses environment variables for credentials instead of interactive input.
Run this after setting credentials in .env file.

Usage:
    export TRADESTATION_CLIENT_SECRET="your_secret_here"
    .venv/bin/python auth_non_interactive.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.auth.tokens import TokenManager


async def main():
    """Authenticate using environment variables."""
    print("="*70)
    print("TradeStation Non-Interactive Authentication")
    print("="*70)
    print()

    # Load settings
    settings = load_settings()

    # Check if client_secret is set
    client_secret = os.environ.get("TRADESTATION_CLIENT_SECRET")
    if not client_secret:
        print("❌ Error: TRADESTATION_CLIENT_SECRET not set")
        print()
        print("Set it first:")
        print("  export TRADESTATION_CLIENT_SECRET='your_secret_here'")
        print()
        print("Or use .env file:")
        print("  echo 'TRADESTATION_CLIENT_SECRET=your_secret' >> .env")
        return False

    print(f"✅ Client Secret found: {client_secret[:20]}...")
    print(f"✅ Client ID: {settings.tradestation_client_id}")
    print()

    # Create OAuth client
    token_manager = TokenManager()
    oauth_client = OAuth2Client(
        client_id=settings.tradestation_client_id,
        redirect_uri=settings.tradestation_redirect_uri,
        token_manager=token_manager,
    )

    # Generate authorization URL
    print("📋 Step 1: Generate Authorization URL")
    print("-"*70)

    auth_url = oauth_client.get_authorization_url()
    state = oauth_client._state
    code_verifier = oauth_client._pkce_manager.generate_code_verifier()

    print(f"Authorization URL:")
    print(f"  {auth_url}")
    print()
    print(f"State: {state}")
    print(f"Code Verifier: {code_verifier}")
    print()
    print("Open this URL in your browser to authorize:")
    print(f"  {auth_url}")
    print()
    print("="*70)
    print("⚠️  INSTRUCTIONS:")
    print("="*70)
    print()
    print("1. Copy the authorization URL above")
    print("2. Open it in your web browser")
    print("3. Log in to TradeStation")
    print("4. Authorize the application")
    print("5. You'll be redirected to: http://localhost:8080/?code=...")
    print("6. Copy the 'code' parameter from the URL")
    print("7. Run this command to complete authentication:")
    print()
    print("   .venv/bin/python exchange_auth_code.py <code_from_url>")
    print()
    print("="*70)
    print()

    # Save state for exchange script
    state_file = Path(".auth_state")
    with open(state_file, "w") as f:
        f.write(f"state={state}\n")
        f.write(f"code_verifier={code_verifier}\n")
        f.write(f"client_id={settings.tradestation_client_id}\n")
        f.write(f"redirect_uri={settings.tradestation_redirect_uri}\n")

    print(f"✅ Authentication state saved to {state_file}")
    print()
    print("Next step: Run exchange_auth_code.py with your authorization code")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nAuthentication cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

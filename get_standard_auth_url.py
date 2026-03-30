#!/usr/bin/env python3
"""
Generate TradeStation OAuth authorization URL (standard auth, no PKCE).
This version uses client_secret instead of PKCE code_verifier.
"""

import sys
from pathlib import Path
import urllib.parse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings


def main():
    """Generate OAuth authorization URL (standard auth)."""
    print("="*70)
    print("TradeStation OAuth Authorization URL (Standard Auth)")
    print("="*70)
    print()

    # Load settings
    settings = load_settings()

    CLIENT_ID = settings.tradestation_client_id
    REDIRECT_URI = settings.tradestation_redirect_uri

    print(f"✅ Client ID: {CLIENT_ID}")
    print(f"✅ Redirect URI: {REDIRECT_URI}")
    print()

    # Build authorization URL (standard auth, NO PKCE)
    print("📋 Generating Authorization URL (Standard Auth)")
    print("-"*70)
    print()

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "audience": "https://api.tradestation.com",
        "scope": "openid profile offline_access MarketData ReadAccount Trade",
    }

    auth_url = "https://signin.tradestation.com/authorize?" + urllib.parse.urlencode(params)

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
    print("   .venv/bin/python exchange_code_simple.py <code_from_url>")
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

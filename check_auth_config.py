#!/usr/bin/env python3
"""Check TradeStation OAuth configuration"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.execution.tradestation.auth.oauth import OAuth2Client
import httpx
import asyncio

async def check_config():
    print("="*70)
    print("TradeStation OAuth Configuration Check")
    print("="*70)
    print()
    
    client = OAuth2Client(
        client_id="8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK",
        redirect_uri="http://localhost:8080"
    )
    
    print("✅ OAuth Client Configuration:")
    print(f"   Client ID: {client.client_id}")
    print(f"   Redirect URI: {client.redirect_uri}")
    print(f"   Scopes: {client.scopes}")
    print(f"   Authorization URL: {client.authorization_url}")
    print(f"   Token URL: {client.token_url}")
    print()
    
    # Check if redirect URI is in allowed list
    print("✅ Redirect URI Validation:")
    if client.redirect_uri in client.ALLOWED_REDIRECT_URIS:
        print(f"   {client.redirect_uri} is in allowed list ✓")
    else:
        print(f"   {client.redirect_uri} is NOT in allowed list ✗")
        print(f"   Allowed URIs: {client.ALLOWED_REDIRECT_URIS}")
    print()
    
    # Show what a token request would look like
    print("📋 Sample Token Request Parameters:")
    print("   URL:", client.token_url)
    print("   Method: POST")
    print("   Headers: {'Content-Type': 'application/x-www-form-urlencoded'}")
    print("   Body Parameters:")
    print("     - grant_type: authorization_code")
    print("     - client_id:", client.client_id)
    print("     - code: [AUTHORIZATION_CODE]")
    print("     - redirect_uri:", client.redirect_uri)
    print("     - code_verifier:", client.pkce_helper.code_verifier[:20] + "...")
    print()
    
    print("="*70)
    print("⚠️  TROUBLESHOOTING STEPS:")
    print("="*70)
    print()
    print("If you're getting 'access_denied' or 'Unauthorized', check:")
    print()
    print("1. TradeStation App Configuration:")
    print("   - Log in to TradeStation Developer Portal")
    print("   - Check your app settings")
    print("   - Verify 'http://localhost:8080' is in Redirect URIs")
    print("   - Verify scopes are enabled: MarketData, ReadAccount, Trade")
    print()
    print("2. App Type:")
    print("   - Is this a 'Public' or 'Confidential' client?")
    print("   - Public clients should NOT use client_secret")
    print("   - Confidential clients require client_secret")
    print()
    print("3. Client ID:")
    print("   - Verify the client_id is correct")
    print("   - Make sure it's not from a test/sandbox environment")
    print()
    print("4. Authorization Code:")
    print("   - Codes expire quickly (use within 1-2 minutes)")
    print("   - Codes are single-use (can't reuse)")
    print("="*70)

asyncio.run(check_config())

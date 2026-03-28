#!/usr/bin/env python3
"""
Complete PKCE OAuth Flow - Single Client Instance

This script maintains the same OAuth2Client instance throughout the entire flow,
ensuring the code_verifier used for exchange matches the code_challenge used for authorization.
"""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.execution.tradestation.auth.oauth import OAuth2Client
import httpx

async def main():
    print("="*70)
    print("TradeStation PKCE OAuth - Complete Flow")
    print("="*70)
    print()

    # STEP 1: Create OAuth client and generate URL
    print("STEP 1: Initializing OAuth Client...")
    client = OAuth2Client(
        client_id="8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK",
        redirect_uri="http://localhost:8080"
    )

    print(f"✅ Client initialized")
    print(f"   Client ID: {client.client_id}")
    print(f"   Redirect URI: {client.redirect_uri}")
    print(f"   Code Verifier: {client.pkce_helper.code_verifier[:20]}...")
    print(f"   Code Challenge: {client.pkce_helper.code_challenge[:20]}...")
    print()

    # STEP 2: Generate authorization URL
    print("STEP 2: Generating Authorization URL...")
    auth_url = client.get_authorization_url()

    print("✅ Authorization URL generated:")
    print("-"*70)
    print(auth_url)
    print("-"*70)
    print()

    # STEP 3: Get callback URL from user
    print("STEP 3: Complete Authorization in Browser")
    print("   1. Copy the URL above")
    print("   2. Paste it into your browser")
    print("   3. Log in and authorize")
    print("   4. Copy the callback URL from your browser")
    print()

    callback_url = input("Paste callback URL here: ").strip()

    # STEP 4: Extract authorization code
    print()
    print("STEP 4: Processing Callback...")
    try:
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)

        if "error" in params:
            error = params["error"][0]
            error_desc = params.get("error_description", [""])[0]
            print(f"❌ Authorization failed: {error} - {error_desc}")
            return

        if "code" not in params:
            print("❌ No authorization code found in callback URL!")
            return

        code = params["code"][0]
        print(f"✅ Authorization code received: {code[:20]}...")
        print()

    except Exception as e:
        print(f"❌ Error parsing callback URL: {e}")
        return

    # STEP 5: Exchange code for token (using SAME client)
    print("STEP 5: Exchanging Authorization Code for Tokens...")
    print("   (Using same OAuth client with matching code_verifier)")

    try:
        token_response = await client.exchange_code_for_token(code)

        print()
        print("="*70)
        print("✅ AUTHENTICATION SUCCESSFUL!")
        print("="*70)
        print(f"Access Token: {token_response.access_token[:50]}...")
        print(f"Token Type: {token_response.token_type}")
        print(f"Expires In: {token_response.expires_in} seconds ({token_response.expires_in // 60} minutes)")
        print(f"Refresh Token: {token_response.refresh_token[:50] if token_response.refresh_token else 'None'}...")
        print(f"Scope: {token_response.scope}")
        print(f"Issued At: {token_response.issued_at}")
        print("="*70)
        print()

        # STEP 6: Test API call
        print("STEP 6: Testing TradeStation API...")
        headers = {"Authorization": f"Bearer {token_response.access_token}"}

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                "https://api.tradestation.com/v3/data/quote/symbols",
                headers=headers,
                params={"symbols": "MNQH26"}
            )

            if response.status_code == 200:
                data = response.json()
                print("✅ API call successful!")
                print(f"   Quotes received: {len(data.get('Quotes', []))}")

                if data.get('Quotes'):
                    quote = data['Quotes'][0]
                    print(f"   Symbol: {quote.get('Symbol')}")
                    print(f"   Bid: {quote.get('Bid')}")
                    print(f"   Ask: {quote.get('Ask')}")
                    print(f"   Last: {quote.get('Last')}")
                    print(f"   Volume: {quote.get('Volume')}")
                print()
                print("="*70)
                print("🎉 PKCE OAuth FLOW COMPLETE!")
                print("="*70)
                print()
                print("Your TradeStation SDK is now working correctly!")
                print("You can use the OAuth client to make authenticated API requests.")
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(f"   Response: {response.text[:300]}")

    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

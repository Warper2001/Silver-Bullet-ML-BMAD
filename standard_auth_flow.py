#!/usr/bin/env python3
"""
TradeStation Standard Authorization Code Flow (NOT PKCE)

This uses the standard Auth Code Flow with client_secret.
This is for confidential clients (server-side applications).

Requirements:
- Client ID (API Key)
- Client Secret
- Redirect URI configured in TradeStation app
"""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import getpass

sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx


async def main():
    print("="*70)
    print("TradeStation Standard Authorization Code Flow")
    print("="*70)
    print()
    print("⚠️  This is NOT PKCE - this requires a client_secret")
    print("    Use this for server-side/confidential applications")
    print()

    # Get credentials
    client_id = "8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK"
    client_secret = getpass.getpass("Enter Client Secret: ")
    redirect_uri = "http://localhost:8080"

    if not client_secret:
        print("❌ Client Secret is required for standard Auth Code Flow")
        return

    print(f"\n✅ Credentials received:")
    print(f"   Client ID: {client_id}")
    print(f"   Client Secret: {client_secret[:10]}...")
    print(f"   Redirect URI: {redirect_uri}")
    print()

    # Build authorization URL (no PKCE parameters)
    print("STEP 1: Generating Authorization URL...")
    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "audience": "https://api.tradestation.com",
        "scope": "openid profile offline_access MarketData ReadAccount Trade",
        "state": "standard_auth_flow_test"
    }

    from urllib.parse import urlencode
    auth_url = "https://signin.tradestation.com/authorize?" + urlencode(auth_params)

    print("✅ Authorization URL generated:")
    print("-"*70)
    print(auth_url)
    print("-"*70)
    print()

    # Get callback URL from user
    print("STEP 2: Complete Authorization in Browser")
    print("   1. Copy the URL above")
    print("   2. Paste it into your browser")
    print("   3. Log in and authorize")
    print("   4. Copy the callback URL from your browser")
    print()

    callback_url = input("Paste callback URL here: ").strip()

    # Extract authorization code
    print()
    print("STEP 3: Processing Callback...")
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

    # Exchange code for token (using client_secret, NOT code_verifier)
    print("STEP 4: Exchanging Authorization Code for Tokens...")
    print("   (Using client_secret, NOT PKCE)")

    token_params = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://signin.tradestation.com/oauth/token",
                data=token_params,
                headers=headers
            )

            if response.status_code == 200:
                token_data = response.json()

                print()
                print("="*70)
                print("✅ AUTHENTICATION SUCCESSFUL!")
                print("="*70)
                print(f"Access Token: {token_data.get('access_token', 'N/A')[:50]}...")
                print(f"Token Type: {token_data.get('token_type', 'N/A')}")
                print(f"Expires In: {token_data.get('expires_in', 'N/A')} seconds")
                print(f"Refresh Token: {token_data.get('refresh_token', 'N/A')[:50] if token_data.get('refresh_token') else 'None'}...")
                print(f"Scope: {token_data.get('scope', 'N/A')}")
                print("="*70)
                print()

                # Test API call
                print("STEP 5: Testing TradeStation API...")
                access_token = token_data.get('access_token')
                api_headers = {"Authorization": f"Bearer {access_token}"}

                async with httpx.AsyncClient() as api_client:
                    api_response = await api_client.get(
                        "https://api.tradestation.com/v3/data/quote/symbols",
                        headers=api_headers,
                        params={"symbols": "MNQH26"}
                    )

                    if api_response.status_code == 200:
                        data = api_response.json()
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
                        print("🎉 STANDARD AUTH CODE FLOW COMPLETE!")
                        print("="*70)
                    else:
                        print(f"❌ API call failed: {api_response.status_code}")
                        print(f"   Response: {api_response.text[:300]}")

            else:
                print()
                print(f"❌ Token exchange failed: {response.status_code}")
                print(f"   Response: {response.text[:500]}")

                if response.status_code == 401:
                    print()
                    print("⚠️  Possible causes:")
                    print("   - Incorrect client_secret")
                    print("   - Authorization code expired or already used")
                    print("   - Redirect URI doesn't match TradeStation app config")

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

#!/usr/bin/env python3
"""Simple PKCE OAuth test - manual flow"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.execution.tradestation.auth.oauth import OAuth2Client

async def main():
    print("="*70)
    print("TradeStation PKCE OAuth - Simplified Test")
    print("="*70)
    print()
    
    # Create OAuth client
    client = OAuth2Client(
        client_id="8mpwDPxyviXzglA6xCGs7X9PXnl1TyFK",
        redirect_uri="http://localhost:8080"
    )
    
    # Generate authorization URL
    auth_url = client.get_authorization_url()
    
    print("STEP 1: Visit this authorization URL:")
    print("-"*70)
    print(auth_url)
    print("-"*70)
    print()
    
    # Get authorization code from user
    print("STEP 2: After authorizing, paste the callback URL here:")
    callback_url = input("Callback URL: ").strip()
    
    # Extract authorization code
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(callback_url)
    params = parse_qs(parsed.query)
    
    if "code" not in params:
        print("❌ No authorization code found in URL!")
        return
    
    code = params["code"][0]
    print(f"\n✅ Authorization code received: {code[:20]}...")
    print()
    
    # Exchange code for token
    print("STEP 3: Exchanging authorization code for tokens...")
    try:
        token_response = await client.exchange_code_for_token(code)
        
        print("\n" + "="*70)
        print("✅ AUTHENTICATION SUCCESSFUL!")
        print("="*70)
        print(f"Access Token: {token_response.access_token[:50]}...")
        print(f"Token Type: {token_response.token_type}")
        print(f"Expires In: {token_response.expires_in} seconds")
        print(f"Refresh Token: {token_response.refresh_token[:50] if token_response.refresh_token else 'None'}...")
        print(f"Scope: {token_response.scope}")
        print(f"Issued At: {token_response.issued_at}")
        print("="*70)
        
        # Test API call
        print("\nSTEP 4: Testing API call...")
        import httpx
        
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
                print(f"Quotes received: {len(data.get('Quotes', []))}")
                
                if data.get('Quotes'):
                    quote = data['Quotes'][0]
                    print(f"  Symbol: {quote.get('Symbol')}")
                    print(f"  Bid: {quote.get('Bid')}")
                    print(f"  Ask: {quote.get('Ask')}")
                    print(f"  Last: {quote.get('Last')}")
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(f"Response: {response.text[:200]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

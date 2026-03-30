#!/usr/bin/env python3
"""
Test TradeStation Live API with authenticated access token
"""

import asyncio
import httpx
import sys
from pathlib import Path

# Read access token
with open(".access_token", "r") as f:
    ACCESS_TOKEN = f.read().strip()

async def test_live_api():
    print("="*70)
    print("TradeStation Live API Test - Authenticated")
    print("="*70)
    print()

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        # Test 1: Try multiple symbol formats
        print("📊 Test 1: Testing different symbol formats...")

        test_symbols = ["AAPL", "MNQH26", "MNQ M6", "NQH26", "/MNQH26"]

        for symbol in test_symbols:
            try:
                response = await client.get(
                    "https://api.tradestation.com/v3/data/quote/symbols",
                    headers=headers,
                    params={"symbols": symbol}
                )

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        quote = data[0]
                        if quote.get('Symbol') != 'symbols' and quote.get('Last', 0) != 0:
                            print(f"✅ SUCCESS with {symbol}!")
                            print(f"   Symbol: {quote.get('Symbol')}")
                            print(f"   Bid: {quote.get('Bid')}")
                            print(f"   Ask: {quote.get('Ask')}")
                            print(f"   Last: {quote.get('Last')}")
                            print(f"   Volume: {quote.get('Volume')}")
                            break
                        else:
                            print(f"❌ {symbol}: Invalid or no data")
                    else:
                        print(f"❌ {symbol}: Unexpected response format")
                else:
                    print(f"❌ {symbol}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")

        print()

        # Test 2: Test multiple futures contracts
        print("📈 Test 2: Testing futures contracts...")

        futures_symbols = ["ESZ4", "NQZ4", "/ESZ4", "/NQZ4"]  # Try E-mini S&P and Nasdaq

        for symbol in futures_symbols:
            try:
                response = await client.get(
                    "https://api.tradestation.com/v3/data/quote/symbols",
                    headers=headers,
                    params={"symbols": symbol}
                )

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        quote = data[0]
                        if quote.get('Last', 0) != 0:
                            print(f"✅ {symbol}: {quote.get('Last')} ({quote.get('Symbol')})")
                            break
                        else:
                            print(f"❌ {symbol}: No data")
                    else:
                        print(f"❌ {symbol}: Invalid response")
                else:
                    print(f"❌ {symbol}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")

    print()
    print("="*70)
    print("Live API Test Complete")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_live_api())
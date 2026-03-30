#!/usr/bin/env python3
"""
Test TradeStation Live API with correct endpoint format
"""

import asyncio
import httpx
from datetime import datetime, timedelta, timezone

# Read access token
with open(".access_token", "r") as f:
    ACCESS_TOKEN = f.read().strip()

async def test_live_api():
    print("="*70)
    print("TradeStation Live API Test - Correct Format")
    print("="*70)
    print()

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        # Test 1: Get Historical Bars (correct endpoint format)
        print("📊 Test 1: Fetching historical bars for MNQH26...")

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        params = {
            "interval": "1",
            "unit": "Minute",
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        }

        try:
            response = await client.get(
                "https://api.tradestation.com/v3/marketdata/bars/MNQH26",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("Bars"):
                    bars = data["Bars"]
                    print(f"✅ SUCCESS!")
                    print(f"   Bars received: {len(bars)}")
                    if bars:
                        latest = bars[0]
                        print(f"   Latest bar:")
                        print(f"     Timestamp: {latest.get('Timestamp')}")
                        print(f"     Open: {latest.get('Open')}")
                        print(f"     High: {latest.get('High')}")
                        print(f"     Low: {latest.get('Low')}")
                        print(f"     Close: {latest.get('Close')}")
                        print(f"     Volume: {latest.get('Volume')}")
                else:
                    print(f"❌ No bars in response")
                    print(f"   Response: {data}")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()

        # Test 2: Try getting current quotes
        print("💰 Test 2: Fetching current quotes for MNQH26...")

        try:
            response = await client.get(
                "https://api.tradestation.com/v3/data/quote/MNQH26",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")
                print(f"   Quote data: {data}")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()

        # Test 3: Try with simpler symbol
        print("🎯 Test 3: Testing with SPY stock...")

        try:
            response = await client.get(
                "https://api.tradestation.com/v3/data/quote/SPY",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")
                print(f"   Quote data: {data}")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print()
    print("="*70)
    print("Live API Test Complete")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_live_api())
#!/usr/bin/env python3
"""Test alternative endpoints for bar data."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3


async def test_alternatives():
    """Test alternative endpoints."""
    
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient() as client:
        print("Testing Alternative Endpoints:")
        print("=" * 70)
        
        # Test different endpoint variations
        endpoints = [
            "bars",
            "candles", 
            "ohlc",
            "pricehistory",
            "historical",
        ]
        
        base = "https://api.tradestation.com/v3/marketdata"
        
        for endpoint in endpoints:
            url = f"{base}/{endpoint}/MNQM26?interval=1min&maxRecords=5"
            
            try:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    print(f"✅ /{endpoint}/ - SUCCESS")
                    print(f"   Sample: {response.text[:100]}...")
                else:
                    print(f"❌ /{endpoint}/ - {response.status_code}: {response.reason_phrase}")
                    
            except Exception as e:
                print(f"❌ /{endpoint}/ - Error: {str(e)[:50]}")
        
        # Test API version variations
        print("\nTesting API Versions:")
        print("-" * 70)
        
        for version in ["v2", "v3", "v3.1"]:
            url = f"https://api.tradestation.com/{version}/marketdata/bars/MNQM26?barType=1&maxRecords=5"
            
            try:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    print(f"✅ {version}/marketdata/bars - SUCCESS")
                else:
                    print(f"❌ {version}/marketdata/bars - {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {version}/marketdata/bars - Error")
        
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_alternatives())

#!/usr/bin/env python3
"""Test barcharts endpoint specifically."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3


async def test_barcharts():
    """Test barcharts endpoint with different parameters."""
    
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient() as client:
        print("Testing Bar Charts Endpoints:")
        print("=" * 70)
        
        # Current time
        now = datetime.now()
        
        # Test different approaches
        tests = [
            # Test 1: Basic symbol with no time range
            ("Basic: no time range", 
             "https://api.tradestation.com/v3/marketdata/barcharts?symbol=MNQM26&interval=1min&maxRecords=10"),
            
            # Test 2: With recent time range
            ("Recent time range",
             f"https://api.tradestation.com/v3/marketdata/barcharts?symbol=MNQM26&interval=1min&startTime={(now - timedelta(hours=2)).isoformat()}&endTime={now.isoformat()}&maxRecords=10"),
            
            # Test 3: Try using quotes endpoint instead
            ("Quotes endpoint",
             "https://api.tradestation.com/v3/marketdata/quotes/MNQM26"),
        ]
        
        for name, url in tests:
            try:
                print(f"\n{name}:")
                print(f"  URL: {url}")
                response = await client.get(url, headers=headers)
                
                print(f"  Status: {response.status_code}: {response.reason_phrase}")
                
                if response.status_code == 200:
                    print(f"  ✅ SUCCESS - Sample: {response.text[:200]}...")
                else:
                    print(f"  ❌ Failed - Response: {response.text[:200]}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_barcharts())

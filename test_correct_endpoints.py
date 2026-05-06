#!/usr/bin/env python3
"""Test correct TradeStation API endpoints based on documentation."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3


async def test_correct_endpoints():
    """Test the correct endpoint format from documentation."""
    
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient() as client:
        print("Testing CORRECT TradeStation API Endpoints:")
        print("=" * 70)
        
        # Based on documentation examples
        tests = [
            # Test 1: Bar charts with symbol in path (like MSFT example)
            ("1. Bar charts (symbol in path)",
             "https://api.tradestation.com/v3/marketdata/barcharts/MNQ?interval=1&unit=Minute&barsback=5"),
            
            # Test 2: Try with MNQH26 (March contract)
            ("2. Bar charts (MNQH26)",
             "https://api.tradestation.com/v3/marketdata/barcharts/MNQH26?interval=1&unit=Minute&barsback=5"),
            
            # Test 3: Simulation API endpoint
            ("3. Simulation API",
             "https://sim-api.tradestation.com/v3/marketdata/barcharts/MNQ?interval=1&unit=Minute&barsback=5"),
            
            # Test 4: Streaming endpoint
            ("4. Streaming endpoint",
             "https://api.tradestation.com/v3/marketdata/stream/barcharts/MNQ?interval=1&unit=minute"),
        ]
        
        for name, url in tests:
            print(f"\n{name}:")
            print(f"  URL: {url}")
            
            try:
                if "stream" in url:
                    # For streaming, just check if endpoint exists
                    response = await client.get(url, headers=headers, timeout=5.0)
                    print(f"  Status: {response.status_code}: {response.reason_phrase}")
                    
                    if response.status_code == 200:
                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        print(f"  Content-Type: {content_type}")
                        
                        if 'streams' in content_type:
                            print(f"  ✅ Streaming endpoint confirmed!")
                    else:
                        print(f"  Response: {response.text[:200]}")
                else:
                    response = await client.get(url, headers=headers)
                    print(f"  Status: {response.status_code}: {response.reason_phrase}")
                    
                    if response.status_code == 200:
                        print(f"  ✅ SUCCESS - Sample: {response.text[:200]}...")
                    else:
                        print(f"  ❌ Response: {response.text[:200]}")
                        
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_correct_endpoints())

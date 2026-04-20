#!/usr/bin/env python3
"""Check TradeStation API endpoints for live vs simulation."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3


async def check_endpoints():
    """Test different TradeStation API endpoints."""
    
    # Load the new access token
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient() as client:
        print("Testing TradeStation API Endpoints:")
        print("=" * 70)
        
        # Test 1: Market Data Endpoint (for getting quotes/bars)
        print("1. Market Data Endpoints:")
        print("-" * 70)
        
        endpoints_to_test = [
            ("GET /v3/marketdata/quotes/{symbol}", "https://api.tradestation.com/v3/marketdata/quotes/MNQ/M26"),
            ("GET /v3/marketdata/barcharts", "https://api.tradestation.com/v3/marketdata/barcharts?symbol=MNQ/M26&interval=1min&maxRecords=1"),
            ("GET /v3/md/quotes/{symbol}", "https://api.tradestation.com/v3/md/quotes/MNQ/M26"),
        ]
        
        for name, url in endpoints_to_test:
            try:
                response = await client.get(url, headers=headers)
                print(f"  {name:40} - {response.status_code}: {response.reason_phrase}")
                
                if response.status_code == 200:
                    print(f"    ✅ SUCCESS - Response: {response.text[:100]}...")
                elif response.status_code == 404:
                    print(f"    ❌ Not Found - Symbol or endpoint wrong")
                elif response.status_code == 401:
                    print(f"    ⚠️  Unauthorized - Permission issue")
                    
            except Exception as e:
                print(f"  {name:40} - Error: {e}")
        
        print()
        print("2. Simulation Trading Endpoints:")
        print("-" * 70)
        
        sim_endpoints = [
            ("GET /v3/orderdeffee/simulated/accounts", "https://api.tradestation.com/v3/orderdeffee/simulated/accounts"),
            ("GET /v3/brokerage/simulated/accounts", "https://api.tradestation.com/v3/brokerage/simulated/accounts"),
        ]
        
        for name, url in sim_endpoints:
            try:
                response = await client.get(url, headers=headers)
                print(f"  {name:40} - {response.status_code}: {response.reason_phrase}")
                
                if response.status_code == 200:
                    print(f"    ✅ SUCCESS - Simulation accounts available")
                    
            except Exception as e:
                print(f"  {name:40} - Error: {e}")
        
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(check_endpoints())

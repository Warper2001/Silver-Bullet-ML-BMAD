#!/usr/bin/env python3
"""Test different MNQ symbol formats."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3


async def test_symbols():
    """Test different symbol formats."""
    
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient() as client:
        print("Testing MNQ Symbol Formats:")
        print("=" * 70)
        
        # Test different formats
        formats = [
            "MNQ/M26",
            "MNQM26", 
            ".MNQM26",
            "MNQ",
            ".MNQ",
            "$MNQ",
            "MNQH26",  # March 2026
            "MNQJ26",  # June 2026  
            "MNQU26",  # Sept 2026
            "MNQZ26",  # Dec 2026
        ]
        
        base_url = "https://api.tradestation.com/v3/marketdata/quotes/{}"
        
        for symbol in formats:
            url = base_url.format(symbol)
            
            try:
                response = await client.get(url, headers=headers)
                
                status = f"{response.status_code}: {response.reason_phrase}"
                if response.status_code == 200:
                    print(f"✅ {symbol:15} - {status}")
                    print(f"   Sample: {response.text[:100]}...")
                elif response.status_code == 404:
                    print(f"❌ {symbol:15} - {status}")
                elif response.status_code == 401:
                    print(f"⚠️  {symbol:15} - {status}")
                else:
                    print(f"❓ {symbol:15} - {status}")
                    
            except Exception as e:
                print(f"❌ {symbol:15} - Error: {str(e)[:50]}")
        
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_symbols())

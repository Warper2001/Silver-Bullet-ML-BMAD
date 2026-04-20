#!/usr/bin/env python3
"""Test TradeStation API connection with different symbol formats."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3
from src.data.config import load_settings


async def test_symbols():
    """Test different MNQ symbol formats."""
    settings = load_settings()
    
    # Load the new access token
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    async with AsyncClient() as client:
        # Test different symbol formats
        symbols = [
            'MNQ/M26',
            'MNQM26',
            'MNQ',
            '.MNQ',
            'MNQH26',  # Try March 2026
            'MNQU26',  # Try September 2026
        ]
        
        print("Testing different MNQ symbol formats:")
        print("=" * 60)
        
        for symbol in symbols:
            url = f"https://api.tradestation.com/v3/marketdata/barcharts"
            params = {
                "symbol": symbol,
                "interval": "1min",
                "maxRecords": 1
            }
            
            headers = {"Authorization": f"Bearer {token}"}
            
            try:
                response = await client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    print(f"✅ {symbol:15} - SUCCESS (200)")
                elif response.status_code == 404:
                    print(f"❌ {symbol:15} - Not Found (404)")
                elif response.status_code == 401:
                    print(f"⚠️  {symbol:15} - Unauthorized (401)")
                else:
                    print(f"❓ {symbol:15} - {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {symbol:15} - Error: {e}")
        
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_symbols())

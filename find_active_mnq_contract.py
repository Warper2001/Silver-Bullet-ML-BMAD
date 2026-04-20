#!/usr/bin/env python3
"""Find the active MNQ contract and test streaming."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient
from src.data.auth_v3 import TradeStationAuthV3


async def find_active_contract():
    """Find which MNQ contract is actively trading."""
    
    with open('.access_token', 'r') as f:
        access_token = f.read().strip()
    
    auth = TradeStationAuthV3(access_token=access_token)
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient() as client:
        print("Finding Active MNQ Contract:")
        print("=" * 70)
        
        # Test different MNQ contracts
        contracts = [
            "MNQH26",  # March 2026
            "MNQJ26",  # June 2026  
            "MNQM26",  # June 2026 (alternative)
            "MNQU26",  # September 2026
            "MNQZ26",  # December 2026
        ]
        
        active_contracts = []
        
        for contract in contracts:
            url = f"https://api.tradestation.com/v3/marketdata/barcharts/{contract}?interval=1&unit=Minute&barsback=1"
            
            try:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    # Parse response to get latest data
                    import json
                    data = json.loads(response.text)
                    
                    if "Bars" in data and len(data["Bars"]) > 0:
                        latest_bar = data["Bars"][0]
                        is_realtime = latest_bar.get("IsRealtime", False)
                        
                        print(f"✅ {contract:10} - Price: {latest_bar.get('Close', 'N/A'):>8} | Realtime: {is_realtime} | Volume: {latest_bar.get('TotalVolume', 'N/A'):>6}")
                        
                        if is_realtime:
                            active_contracts.append(contract)
                    else:
                        print(f"❌ {contract:10} - No bars data")
                else:
                    print(f"❌ {contract:10} - {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {contract:10} - Error: {str(e)[:50]}")
        
        print("\n" + "=" * 70)
        
        if active_contracts:
            print(f"🎯 ACTIVE CONTRACTS: {', '.join(active_contracts)}")
            
            # Test streaming with first active contract
            active = active_contracts[0]
            print(f"\nTesting streaming with {active}...")
            
            stream_url = f"https://api.tradestation.com/v3/marketdata/stream/barcharts/{active}?interval=1&unit=minute"
            
            print(f"Stream URL: {stream_url}")
            print("This should provide real-time bar data!")
            
        else:
            print("⚠️  No active contracts found (markets might be closed)")


if __name__ == "__main__":
    asyncio.run(find_active_contract())

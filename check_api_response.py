
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from src.data.auth_v3 import TradeStationAuthV3

async def check_api():
    auth = TradeStationAuthV3.from_file('.access_token')
    token = await auth.authenticate()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    
    symbol = "MNQM26"
    since = datetime.now(timezone.utc) - timedelta(hours=48)
    firstdate = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    url = f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}?interval=1&unit=Minute&firstdate={firstdate}"
    
    print(f"Testing URL: {url}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            bars = data.get("Bars", [])
            print(f"Number of bars: {len(bars)}")
            if bars:
                print(f"First bar keys: {bars[0].keys()}")
                print(f"First bar data: {bars[0]}")
        else:
            print(f"Response: {response.text}")

if __name__ == "__main__":
    asyncio.run(check_api())

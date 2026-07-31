import asyncio
from src.data.auth_v3 import TradeStationAuthV3

async def main():
    auth = TradeStationAuthV3.from_file('.access_token')
    print("Authenticating...")
    token = await auth.authenticate()
    print("Auth Success")
    # This should update the .access_token file
    await auth.start_auto_refresh()
    print("Auto-refresh started")
    await asyncio.sleep(2)
    print("Check file content sample:")
    print(open('.access_token').read(50))

asyncio.run(main())

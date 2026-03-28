#!/usr/bin/env python3
"""Test downloading current active MNQ contract."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from src.data.tradestation_auth import TradeStationAuth
from src.data.tradestation_client import TradeStationClient


async def test_current_contract():
    """Test downloading the current active MNQ contract."""
    # Initialize auth and client
    auth = TradeStationAuth()
    client = TradeStationClient(auth)

    try:
        # Test with current active contract (June 2026)
        symbol = "MNQM26"
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)  # Last 30 days only

        print(f"Testing download for {symbol}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")

        bars = await client.get_historical_bars(symbol, start_date, end_date)

        if bars:
            print(f"\n✅ SUCCESS! Downloaded {len(bars)} bars")
            print(f"First bar: {bars[0]}")
            print(f"Last bar: {bars[-1]}")
        else:
            print("\n❌ No bars returned")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_current_contract())

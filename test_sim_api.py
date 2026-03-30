#!/usr/bin/env python3
"""Test TradeStation SIM API endpoints."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.execution.tradestation.client import TradeStationClient
from src.data.config import load_settings


async def test_endpoints():
    """Test various TradeStation SIM API endpoints."""
    settings = load_settings()
    
    client = TradeStationClient(
        client_id=settings.tradestation_client_id,
        env="sim"
    )
    
    # Load access token
    with open(".access_token", "r") as f:
        access_token = f.read().strip()
    
    from src.execution.tradestation.auth.tokens import TokenResponse
    token_response = TokenResponse(
        access_token=access_token,
        refresh_token=settings.tradestation_refresh_token,
        expires_in=86400,
        token_type="Bearer",
    )
    client.oauth_client.token_manager.set_token(token_response)
    
    print("Testing TradeStation SIM API endpoints...")
    print(f"API Base URL: {client.api_base_url}")
    print()
    
    async with client:
        # Test 1: Get quotes snapshot
        print("Test 1: GET /quotes/AAPL")
        try:
            quotes = await client.get_quotes(["AAPL"])
            print(f"✅ Success! Got {len(quotes)} quotes")
            for quote in quotes[:2]:
                print(f"   - {quote}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 2: Get historical bars
        print("Test 2: GET /bars/AAPL")
        try:
            bars = await client.get_historical_bars(
                symbol="AAPL",
                interval="1Min",
                bar_count=5
            )
            print(f"✅ Success! Got {len(bars)} bars")
            if bars:
                print(f"   Latest: {bars[0]}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
        print()


if __name__ == "__main__":
    asyncio.run(test_endpoints())

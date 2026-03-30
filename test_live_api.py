#!/usr/bin/env python3
"""Test TradeStation LIVE API endpoints (not SIM)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.execution.tradestation.client import TradeStationClient
from src.data.config import load_settings


async def test_live_api():
    """Test TradeStation LIVE API (not SIM)."""
    settings = load_settings()
    
    # Use LIVE environment instead of SIM
    client = TradeStationClient(
        client_id=settings.tradestation_client_id,
        env="live"  # Changed from "sim"
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
    
    print("Testing TradeStation LIVE API endpoints...")
    print(f"API Base URL: {client.api_base_url}")
    print()
    
    async with client:
        # Test: Get quotes snapshot
        print("Test: GET /data/quote/AAPL (LIVE)")
        try:
            quotes = await client.get_quotes(["AAPL"])
            print(f"✅ Success! Got {len(quotes)} quotes")
            for quote in quotes[:2]:
                print(f"   - {quote}")
        except Exception as e:
            print(f"❌ Failed: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(test_live_api())

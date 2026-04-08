#!/usr/bin/env python3
"""
Test TradeStation HTTP Streaming Client

This script tests the new HTTP streaming client to verify it can
connect to TradeStation API v3 and receive market data.
"""

import asyncio
import logging
import sys
from datetime import datetime

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")

from src.data.auth_v3 import TradeStationAuthV3
from src.data.http_streaming import TradeStationHTTPStreamClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


async def test_http_streaming():
    """Test HTTP streaming connection and data reception."""

    print("\n" + "="*70)
    print("🧪 TradeStation HTTP Streaming Test")
    print("="*70)
    print()

    # Step 1: Load access token
    print("📋 Step 1: Loading access token...")
    try:
        with open(".access_token", "r") as f:
            access_token = f.read().strip()
        if not access_token or len(access_token) < 50:
            print("❌ No valid access token found!")
            print("\nPlease run OAuth flow first:")
            print("  .venv/bin/python get_standard_auth_url.py")
            print("  .venv/bin/python exchange_token_simple.py <code>")
            return False

        print(f"✅ Access token loaded (length: {len(access_token)})")
    except FileNotFoundError:
        print("❌ No .access_token file found!")
        print("\nPlease run OAuth flow first:")
        print("  .venv/bin/python get_standard_auth_url.py")
        print("  .venv/bin/python exchange_token_simple.py <code>")
        return False

    # Step 2: Initialize authentication
    print("\n📋 Step 2: Initializing authentication...")
    try:
        auth = TradeStationAuthV3(access_token=access_token)
        print("✅ Authentication initialized")
    except Exception as e:
        print(f"❌ Authentication initialization failed: {e}")
        return False

    # Step 3: Initialize HTTP streaming client
    print("\n📋 Step 3: Initializing HTTP streaming client...")
    try:
        stream_client = TradeStationHTTPStreamClient(
            auth=auth,
            symbols=["MNQM26"]
        )
        print("✅ HTTP streaming client initialized")
        print(f"   Symbols: {stream_client.symbols}")
    except Exception as e:
        print(f"❌ HTTP streaming client initialization failed: {e}")
        return False

    # Step 4: Connect to HTTP stream
    print("\n📋 Step 4: Connecting to HTTP stream...")
    try:
        await stream_client.connect()
        print("✅ HTTP stream connection established")
    except Exception as e:
        print(f"❌ HTTP stream connection failed: {e}")
        print(f"   This might be due to:")
        print(f"   - Invalid access token (run OAuth flow again)")
        print(f"   - Network connectivity issues")
        print(f"   - TradeStation API downtime")
        return False

    # Step 5: Subscribe to market data
    print("\n📋 Step 5: Subscribing to market data...")
    try:
        data_queue = await stream_client.subscribe()
        print("✅ Successfully subscribed to market data")
    except Exception as e:
        print(f"❌ Subscription failed: {e}")
        await stream_client.disconnect()
        return False

    # Step 6: Receive market data
    print("\n📋 Step 6: Receiving market data (30 seconds)...")
    print("   Press Ctrl+C to stop early")

    message_count = 0
    start_time = datetime.now()
    test_duration = 30  # seconds

    try:
        while (datetime.now() - start_time).total_seconds() < test_duration:
            try:
                # Wait for market data with timeout
                market_data = await asyncio.wait_for(
                    data_queue.get(),
                    timeout=5.0
                )

                message_count += 1

                # Print received data
                print(f"\n📊 Message #{message_count}:")
                print(f"   Symbol: {market_data.symbol}")
                print(f"   Timestamp: {market_data.timestamp}")
                print(f"   Last: ${market_data.last_price:.2f}")
                print(f"   Bid: ${market_data.bid:.2f} x {market_data.bid_size}")
                print(f"   Ask: ${market_data.ask:.2f} x {market_data.ask_size}")
                print(f"   Volume: {market_data.volume}")

            except asyncio.TimeoutError:
                print(f"\n⏳ Waiting for data... ({(datetime.now() - start_time).total_seconds():.0f}s)")
                continue

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")

    # Step 7: Disconnect
    print("\n📋 Step 7: Disconnecting...")
    try:
        await stream_client.disconnect()
        print("✅ HTTP stream disconnected")
    except Exception as e:
        print(f"❌ Disconnect failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"✅ Test completed successfully!")
    print(f"📈 Total messages received: {message_count}")
    print(f"⏱️  Test duration: {(datetime.now() - start_time).total_seconds():.1f} seconds")

    if message_count > 0:
        print(f"📉 Average message rate: {message_count / (datetime.now() - start_time).total_seconds():.2f} msg/s")
        print(f"\n✨ HTTP streaming is working correctly!")
        return True
    else:
        print(f"\n⚠️  No messages received - market might be closed")
        return False


async def main():
    """Main test function."""
    try:
        success = await test_http_streaming()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
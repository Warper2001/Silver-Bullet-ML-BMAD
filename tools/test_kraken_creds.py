#!/usr/bin/env python3
"""Credential test for Kraken Futures API.

Performs a read-only account balance query against the demo (default) or live
endpoint and prints the result. No orders are placed.

Usage:
    .venv/bin/python tools/test_kraken_creds.py           # demo
    .venv/bin/python tools/test_kraken_creds.py --live    # live
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from dotenv import load_dotenv

from src.execution.kraken.auth.api_key import KrakenFuturesAuth
from src.execution.kraken.exceptions import KrakenAuthError, KrakenAPIError
from src.execution.kraken.orders.submission import KrakenOrdersClient


async def _test(live: bool) -> int:
    env = "LIVE" if live else "DEMO"
    print(f"Testing Kraken Futures credentials ({env} endpoint)...")

    try:
        auth = KrakenFuturesAuth()
    except KrakenAuthError as exc:
        print(f"❌ Auth failed: {exc}")
        print("  Set KRAKEN_FUTURES_API_KEY and KRAKEN_FUTURES_API_SECRET in .env")
        return 1

    async with httpx.AsyncClient(timeout=15.0) as http:
        client = KrakenOrdersClient(auth, http, live=live)
        try:
            acct = await client.get_account_balance()
        except KrakenAuthError as exc:
            print(f"❌ Auth failed: {exc}")
            return 1
        except KrakenAPIError as exc:
            print(f"❌ API error: {exc}")
            return 1

    equity   = acct.get("equity", "n/a")
    margin   = acct.get("availableMargin", "n/a")
    currency = acct.get("currency", "")

    print(f"  equity:          {equity} {currency}")
    print(f"  availableMargin: {margin} {currency}")
    print(f"✅ Credentials OK ({env})")
    return 0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test Kraken Futures API credentials")
    parser.add_argument("--live", action="store_true", help="Test against live endpoint (default: demo)")
    args = parser.parse_args()
    sys.exit(asyncio.run(_test(args.live)))


if __name__ == "__main__":
    main()

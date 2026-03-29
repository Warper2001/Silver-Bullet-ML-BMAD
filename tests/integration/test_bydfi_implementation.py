#!/usr/bin/env python3
"""
BYDFI Implementation Integration Tests

This script validates the complete BYDFI implementation:
1. Configuration loading
2. Client initialization and connectivity
3. REST API functionality (market data, account info)
4. Orders client initialization

Usage:
    python tests/integration/test_bydfi_implementation.py

Prerequisites:
    - .env.bydfi file with valid BYDFI API credentials
    - Network connectivity to BYDFI API
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


class BYDFIImplementationTester:
    """Test suite for BYDFI implementation."""

    def __init__(self):
        """Initialize tester."""
        self.test_results = {}
        self.bydfi_client = None

    async def run_all_tests(self):
        """Run all tests and generate report."""
        console.print(Panel.fit("BYDFI Implementation Test Suite", style="bold blue"))
        console.print(f"Start time: {datetime.now(timezone.utc).isoformat()}")
        console.print()

        # Test suite
        tests = [
            ("Configuration Loading", self.test_config_loading),
            ("Client Initialization", self.test_client_initialization),
            ("API Connectivity", self.test_api_connectivity),
            ("Market Data - Quotes", self.test_market_data_quotes),
            ("Market Data - Historical Klines", self.test_market_data_klines),
            ("Market Data - Order Book", self.test_market_data_orderbook),
            ("Account Info", self.test_account_info),
            ("Orders Client Initialization", self.test_orders_client),
        ]

        # Run tests
        for test_name, test_func in tests:
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Test '{test_name}' failed with exception: {e}", exc_info=True)
                self.test_results[test_name] = "FAILED"
                console.print(f"[red]✗[/red] {test_name}: FAILED")
            else:
                self.test_results[test_name] = "PASSED"
                console.print(f"[green]✓[/green] {test_name}: PASSED")

        # Generate summary
        self.print_summary()

    async def test_config_loading(self):
        """Test 1: Configuration loading."""
        from src.data.bydfi_config import load_bydfi_settings

        settings = load_bydfi_settings()

        # Validate required fields
        assert settings.bydfi_api_key, "API key is required"
        assert settings.bydfi_api_secret, "API secret is required"
        assert settings.bydfi_trading_symbol, "Trading symbol is required"
        assert settings.base_url, "Base URL is required"

        # Validate symbol format
        assert "-" in settings.bydfi_trading_symbol, "Symbol must contain dash (e.g., BTC-USDT)"

        logger.info(f"Config loaded: symbol={settings.bydfi_trading_symbol}, env={settings.bydfi_environment}")

    async def test_client_initialization(self):
        """Test 2: BYDFI client initialization."""
        from src.execution.bydfi.client import BYDFIClient

        self.bydfi_client = BYDFIClient()

        # Verify client attributes
        assert self.bydfi_client.api_key, "Client should have API key"
        assert self.bydfi_client.api_secret, "Client should have API secret"
        assert self.bydfi_client.base_url, "Client should have base URL"
        assert self.bydfi_client.signature_generator, "Client should have signature generator"

        logger.info(f"Client initialized: base_url={self.bydfi_client.base_url}")

    async def test_api_connectivity(self):
        """Test 3: API connectivity test."""
        if not self.bydfi_client:
            raise Exception("Client not initialized")

        # Enter context manager (tests connectivity)
        await self.bydfi_client.__aenter__()

        logger.info("API connectivity test passed")

    async def test_market_data_quotes(self):
        """Test 4: Market data - get quotes."""
        if not self.bydfi_client:
            raise Exception("Client not initialized")

        from src.execution.bydfi.models import BYDFIQuote

        quote = await self.bydfi_client.get_quotes("BTC-USDT")

        # Validate quote structure
        assert isinstance(quote, BYDFIQuote), "Should return BYDFIQuote object"
        assert quote.symbol == "BTC-USDT", "Symbol should match"
        assert hasattr(quote, 'price'), "Quote should have price"

        console.print(f"  Current price: ${float(quote.price):,.2f}")
        logger.info(f"Quotes retrieved: price={quote.price}")

    async def test_market_data_klines(self):
        """Test 5: Market data - historical klines."""
        if not self.bydfi_client:
            raise Exception("Client not initialized")

        klines = await self.bydfi_client.get_historical_klines(
            symbol="BTC-USDT",
            interval="5m",
            limit=100,
        )

        # Validate klines
        assert len(klines) > 0, "Should return klines"
        assert klines[0].symbol == "BTC-USDT", "Symbol should match"

        console.print(f"  Retrieved {len(klines)} klines")
        logger.info(f"Klines retrieved: count={len(klines)}")

    async def test_market_data_orderbook(self):
        """Test 6: Market data - order book depth."""
        if not self.bydfi_client:
            raise Exception("Client not initialized")

        from src.execution.bydfi.models import BYDFIOrderBook

        orderbook = await self.bydfi_client.get_order_book_depth(
            symbol="BTC-USDT",
            limit=20,
        )

        # Validate order book
        assert isinstance(orderbook, BYDFIOrderBook), "Should return BYDFIOrderBook object"
        assert orderbook.symbol == "BTC-USDT", "Symbol should match"
        assert len(orderbook.bids) > 0, "Should have bids"
        assert len(orderbook.asks) > 0, "Should have asks"

        best_bid = float(orderbook.bids[0][0])
        best_ask = float(orderbook.asks[0][0])
        spread = best_ask - best_bid

        console.print(f"  Best bid: ${best_bid:,.2f}")
        console.print(f"  Best ask: ${best_ask:,.2f}")
        console.print(f"  Spread: ${spread:,.2f}")
        logger.info(f"Orderbook retrieved: bids={len(orderbook.bids)}, asks={len(orderbook.asks)}")

    async def test_account_info(self):
        """Test 7: Account information."""
        if not self.bydfi_client:
            raise Exception("Client not initialized")

        accounts = await self.bydfi_client.get_account_info()

        # Validate accounts
        assert isinstance(accounts, list), "Should return list of accounts"
        assert len(accounts) > 0, "Should have at least one account"

        console.print(f"  Found {len(accounts)} account(s)")
        for account in accounts:
            if float(account.balance) > 0:
                console.print(f"    {account.currency}: {account.balance}")

        logger.info(f"Account info retrieved: {len(accounts)} accounts")

    async def test_orders_client(self):
        """Test 8: Orders client initialization."""
        from src.execution.bydfi.orders.submission import create_bydfi_orders_client

        orders_client = await create_bydfi_orders_client(self.bydfi_client)

        # Validate orders client
        assert orders_client.bydfi_client, "Should have BYDFI client reference"
        assert orders_client.circuit_breaker, "Should have circuit breaker"
        assert orders_client.default_fill_timeout > 0, "Should have fill timeout"

        logger.info("Orders client initialized successfully")

    def print_summary(self):
        """Print test summary."""
        console.print()
        console.print(Panel.fit("Test Summary", style="bold cyan"))

        # Create results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test Name", style="cyan")
        table.add_column("Status", justify="center")

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            if result == "PASSED":
                table.add_row(test_name, "[green]PASSED[/green]")
                passed += 1
            else:
                table.add_row(test_name, "[red]FAILED[/red]")
                failed += 1

        console.print(table)

        # Summary
        total = len(self.test_results)
        console.print()
        console.print(f"Total tests: {total}")
        console.print(f"[green]Passed: {passed}[/green]")
        console.print(f"[red]Failed: {failed}[/red]")

        if failed == 0:
            console.print()
            console.print(Panel.fit("All tests PASSED! ✓", style="bold green"))
        else:
            console.print()
            console.print(Panel.fit(f"{failed} test(s) FAILED", style="bold red"))

    async def cleanup(self):
        """Cleanup resources."""
        if self.bydfi_client:
            try:
                await self.bydfi_client.__aexit__(None, None, None)
                logger.info("Client cleanup complete")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point."""
    tester = BYDFIImplementationTester()

    try:
        await tester.run_all_tests()
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite error: {e}", exc_info=True)
        sys.exit(1)

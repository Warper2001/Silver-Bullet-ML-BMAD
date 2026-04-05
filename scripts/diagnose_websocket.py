#!/usr/bin/env python3
"""
WebSocket Connection Diagnostic Script

This script diagnoses TradeStation WebSocket connection issues for the
paper trading system. It tests authentication, connection, subscription,
and data flow.

Usage:
    python scripts/diagnose_websocket.py

Output:
    Detailed diagnostic report with pass/fail status for each test
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.data.tradestation_auth import TradeStationAuth
from src.data.tradestation_websocket import TradeStationWebSocket


class WebSocketDiagnostic:
    """Diagnostic tool for WebSocket connection issues"""

    def __init__(self):
        self.results = []
        self.auth = None
        self.ws = None

    def log_result(self, test_name: str, passed: bool, message: str, details: str = ""):
        """Log a diagnostic result"""
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        print(f"  {message}")
        if details:
            print(f"  Details: {details}")
        print()

    async def test_environment_variables(self):
        """Test 1: Check environment variables"""
        print("=" * 60)
        print("Test 1: Environment Variables")
        print("=" * 60)

        load_dotenv()

        required_vars = [
            "TRADESTATION_APP_ID",
            "TRADESTATION_APP_SECRET",
            "TRADESTATION_REFRESH_TOKEN",
            "TRADESTATION_ENVIRONMENT"
        ]

        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            else:
                # Mask sensitive values for display
                if "SECRET" in var or "TOKEN" in var:
                    display_value = f"{value[:10]}...{value[-4:]}"
                else:
                    display_value = value
                print(f"  {var}: {display_value}")

        if missing_vars:
            self.log_result(
                "Environment Variables",
                False,
                "Missing required environment variables",
                f"Missing: {', '.join(missing_vars)}"
            )
            return False
        else:
            self.log_result(
                "Environment Variables",
                True,
                "All required environment variables set",
                f"Checked {len(required_vars)} variables"
            )
            return True

    async def test_authentication(self):
        """Test 2: TradeStation API authentication"""
        print("=" * 60)
        print("Test 2: Authentication")
        print("=" * 60)

        try:
            self.auth = TradeStationAuth(
                app_id=os.getenv("TRADESTATION_APP_ID"),
                app_secret=os.getenv("TRADESTATION_APP_SECRET"),
                refresh_token=os.getenv("TRADESTATION_REFRESH_TOKEN"),
                environment=os.getenv("TRADESTATION_ENVIRONMENT", "SIM")
            )

            # Test authentication
            token = self.auth.get_access_token()

            if token and len(token) > 50:
                self.log_result(
                    "Authentication",
                    True,
                    "Successfully authenticated with TradeStation API",
                    f"Token length: {len(token)}, Environment: {self.auth.environment}"
                )
                return True
            else:
                self.log_result(
                    "Authentication",
                    False,
                    "Authentication returned invalid token",
                    f"Token length: {len(token) if token else 0}"
                )
                return False

        except Exception as e:
            self.log_result(
                "Authentication",
                False,
                "Authentication failed with exception",
                str(e)
            )
            return False

    async def test_websocket_connection(self):
        """Test 3: WebSocket connection"""
        print("=" * 60)
        print("Test 3: WebSocket Connection")
        print("=" * 60)

        try:
            self.ws = TradeStationWebSocket(self.auth)

            # Try to connect (with timeout)
            print("  Attempting to connect to WebSocket...")
            print(f"  Environment: {self.auth.environment}")

            # Connect with timeout
            try:
                await asyncio.wait_for(self.ws.connect(), timeout=10)
            except asyncio.TimeoutError:
                self.log_result(
                    "WebSocket Connection",
                    False,
                    "Connection timeout after 10 seconds",
                    "Possible causes: Network issues, firewall, TradeStation services down"
                )
                return False

            # Check if connected
            if self.ws.is_connected():
                self.log_result(
                    "WebSocket Connection",
                    True,
                    "Successfully connected to TradeStation WebSocket",
                    f"Connected: {self.ws.is_connected()}"
                )
                return True
            else:
                self.log_result(
                    "WebSocket Connection",
                    False,
                    "Failed to establish WebSocket connection",
                    "Connection returned False"
                )
                return False

        except Exception as e:
            self.log_result(
                "WebSocket Connection",
                False,
                "WebSocket connection failed with exception",
                str(e)
            )
            return False

    async def test_market_data_subscription(self):
        """Test 4: Market data subscription"""
        print("=" * 60)
        print("Test 4: Market Data Subscription")
        print("=" * 60)

        if not self.ws or not self.ws.is_connected():
            self.log_result(
                "Market Data Subscription",
                False,
                "Cannot test subscription - WebSocket not connected",
                "Run Test 3 first"
            )
            return False

        try:
            symbol = os.getenv("TRADING_SYMBOL", "MNQ")

            print(f"  Subscribing to {symbol}...")

            # Subscribe to symbol
            await self.ws.subscribe_quotes([symbol])

            # Wait a moment for subscription to take effect
            await asyncio.sleep(2)

            self.log_result(
                "Market Data Subscription",
                True,
                f"Successfully subscribed to {symbol}",
                f"Symbol: {symbol}"
            )
            return True

        except Exception as e:
            self.log_result(
                "Market Data Subscription",
                False,
                "Market data subscription failed",
                str(e)
            )
            return False

    async def test_data_flow(self, timeout_seconds: int = 30):
        """Test 5: Data flow (receive messages)"""
        print("=" * 60)
        print("Test 5: Data Flow")
        print("=" * 60)

        if not self.ws or not self.ws.is_connected():
            self.log_result(
                "Data Flow",
                False,
                "Cannot test data flow - WebSocket not connected",
                "Run Test 3 first"
            )
            return False

        try:
            symbol = os.getenv("TRADING_SYMBOL", "MNQ")

            print(f"  Listening for messages (timeout: {timeout_seconds}s)...")
            print(f"  Symbol: {symbol}")
            print(f"  Market hours: Mon-Fri 9:30 AM - 4:00 PM ET")

            # Track messages
            messages_received = 0
            start_time = datetime.now()

            # Simple message handler
            async def count_messages(msg):
                nonlocal messages_received
                messages_received += 1
                print(f"    Message {messages_received}: {type(msg).__name__}")

            # Subscribe to messages
            self.ws.on_message = count_messages

            # Wait for messages
            try:
                await asyncio.wait_for(
                    self.ws.listen(),  # This would be the actual listen method
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                pass  # Expected timeout

            elapsed = (datetime.now() - start_time).total_seconds()

            if messages_received > 0:
                self.log_result(
                    "Data Flow",
                    True,
                    f"Received {messages_received} messages in {elapsed:.1f} seconds",
                    f"Rate: {messages_received/elapsed:.2f} messages/second"
                )
                return True
            else:
                self.log_result(
                    "Data Flow",
                    False,
                    f"No messages received in {timeout_seconds} seconds",
                    "Possible causes: Market closed, wrong symbol, no market data permissions"
                )
                return False

        except Exception as e:
            self.log_result(
                "Data Flow",
                False,
                "Data flow test failed with exception",
                str(e)
            )
            return False

    async def test_account_info(self):
        """Test 6: Account information"""
        print("=" * 60)
        print("Test 6: Account Information")
        print("=" * 60)

        try:
            # This would use the TradeStation API to get account info
            # For now, we'll just check if account ID is set
            account_id = os.getenv("TRADESTATION_ACCOUNT_ID")

            if account_id:
                self.log_result(
                    "Account Information",
                    True,
                    "Account ID configured",
                    f"Account: {account_id}"
                )
                return True
            else:
                self.log_result(
                    "Account Information",
                    False,
                    "TRADESTATION_ACCOUNT_ID not set in .env",
                    "Add: TRADING_ACCOUNT_ID=your_sim_account_id to .env"
                )
                return False

        except Exception as e:
            self.log_result(
                "Account Information",
                False,
                "Account info check failed",
                str(e)
            )
            return False

    async def run_all_tests(self):
        """Run all diagnostic tests"""
        print("\n" + "=" * 60)
        print("TRADESTATION WEBSOCKET DIAGNOSTIC TOOL")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run tests in sequence
        results = {}

        # Test 1: Environment variables
        results["env_vars"] = await self.test_environment_variables()
        if not results["env_vars"]:
            print("\n⚠️  Cannot proceed without environment variables")
            print("Please set required variables in .env file")
            return self.generate_report()

        # Test 2: Authentication
        results["auth"] = await self.test_authentication()
        if not results["auth"]:
            print("\n⚠️  Cannot proceed without authentication")
            print("Please verify your TradeStation API credentials")
            return self.generate_report()

        # Test 3: WebSocket connection
        results["websocket"] = await self.test_websocket_connection()
        if not results["websocket"]:
            print("\n⚠️  Cannot proceed without WebSocket connection")
            return self.generate_report()

        # Test 4: Market data subscription
        results["subscription"] = await self.test_market_data_subscription()

        # Test 5: Data flow
        results["data_flow"] = await self.test_data_flow(timeout_seconds=10)

        # Test 6: Account info
        results["account"] = await self.test_account_info()

        # Cleanup
        if self.ws and self.ws.is_connected():
            print("\n  Closing WebSocket connection...")
            await self.ws.disconnect()
            print("  ✅ Connection closed")

        return self.generate_report()

    def generate_report(self):
        """Generate final diagnostic report"""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC REPORT")
        print("=" * 60)

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")

        print("\n" + "-" * 60)
        print("DETAILED RESULTS:")
        print("-" * 60)

        for result in self.results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"\n{status}: {result['test']}")
            print(f"  Message: {result['message']}")
            if result['details']:
                print(f"  Details: {result['details']}")

        print("\n" + "-" * 60)
        print("RECOMMENDATIONS:")
        print("-" * 60)

        if passed == total:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nYour WebSocket connection is working correctly.")
            print("You can proceed with starting the paper trading system:")
            print("  ./deploy_paper_trading.sh start")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("\nCommon issues and solutions:")

            for result in self.results:
                if not result["passed"]:
                    print(f"\n❌ {result['test']}:")
                    if "Environment" in result['test']:
                        print("  Solution: Set missing environment variables in .env file")
                        print("  Required: TRADESTATION_APP_ID, TRADESTATION_APP_SECRET,")
                        print("            TRADESTATION_REFRESH_TOKEN, TRADESTATION_ENVIRONMENT")
                    elif "Authentication" in result['test']:
                        print("  Solution: Verify your API credentials in TradeStation dashboard")
                        print("  - Check App ID and App Secret are correct")
                        print("  - Regenerate Refresh Token if expired")
                        print("  - Verify API app has 'Read' and 'Trade' scopes")
                    elif "WebSocket" in result['test']:
                        print("  Solution: Check network connectivity")
                        print("  - Verify internet connection")
                        print("  - Check firewall rules (WebSocket uses wss:// on port 443)")
                        print("  - Verify TradeStation services operational")
                    elif "Data Flow" in result['test']:
                        print("  Solution: Check market data permissions and market hours")
                        print("  - Verify account has market data for MNQ (Micro Nasdaq-100)")
                        print("  - Check market is open (Mon-Fri 9:30 AM - 4:00 PM ET)")
                        print("  - Verify symbol subscription is correct")

        print("\n" + "=" * 60)
        print(f"Diagnostic completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        return passed == total


async def main():
    """Main entry point"""
    diagnostic = WebSocketDiagnostic()
    success = await diagnostic.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

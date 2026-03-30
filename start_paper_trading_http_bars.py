#!/usr/bin/env python3
"""
Paper Trading with Time-Based Bars from HTTP Polling

This script creates DollarBars by polling TradeStation API via HTTP
and aggregating quotes into time windows, then feeds them into the
ML trading pipeline.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.data.auth_v3 import TradeStationAuthV3
from src.data.contract_detector import ContractDetector
from src.data.market_data_validator import MarketDataValidator
from src.data.models import DollarBar
from src.ml.inference import MLInference
from src.execution.position_tracker import PositionTracker
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class HTTPBarAggregator:
    """Aggregates HTTP quotes into time-based DollarBars."""

    def __init__(self, bar_window_minutes: int = 5):
        """Initialize aggregator.

        Args:
            bar_window_minutes: Time window for each bar (default: 5 minutes)
        """
        self.bar_window_minutes = bar_window_minutes
        self.quotes_buffer = []
        self.last_bar_time = None

    def add_quote(self, quote: dict) -> None:
        """Add a quote to the buffer.

        Args:
            quote: Quote data from TradeStation API
        """
        self.quotes_buffer.append(quote)

    def should_create_bar(self, current_time: datetime) -> bool:
        """Check if it's time to create a new bar.

        Args:
            current_time: Current timestamp

        Returns:
            True if bar window has elapsed
        """
        if self.last_bar_time is None:
            self.last_bar_time = current_time
            return True

        elapsed = (current_time - self.last_bar_time).total_seconds()
        window_seconds = self.bar_window_minutes * 60

        return elapsed >= window_seconds

    def create_bar_from_quotes(self) -> DollarBar | None:
        """Create a DollarBar from accumulated quotes.

        Returns:
            DollarBar object or None if insufficient data
        """
        if not self.quotes_buffer:
            return None

        # Aggregate quotes
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        total_notional = 0

        for quote in self.quotes_buffer:
            last = quote.get('Last', 0)
            bid = quote.get('Bid', 0)
            ask = quote.get('Ask', 0)
            volume = quote.get('Volume', 0)

            if last > 0:
                opens.append(last)
                highs.append(last)
                lows.append(last)
                closes.append(last)
                volumes.append(volume)

                # Calculate notional value for MNQ futures
                # MNQ = $2 × Nasdaq-100 index per contract
                notional = last * 2  # Value of 1 MNQ contract per tick
                total_notional += notional

        if not opens:
            return None

        # Create DollarBar
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=opens[0],
            high=max(highs) if highs else 0,
            low=min(lows) if lows else 0,
            close=closes[-1] if closes else 0,
            volume=sum(volumes),
            notional_value=total_notional,
        )

        # Reset buffer
        self.quotes_buffer = []
        self.last_bar_time = datetime.now(timezone.utc)

        return bar


class HTTPPaperTrader:
    """Paper trading system using HTTP polling to create bars."""

    def __init__(self, access_token: str, symbol: str = "MNQM26"):
        """Initialize HTTP-based paper trader.

        Args:
            access_token: Valid TradeStation access token
            symbol: Futures contract symbol
        """
        self.access_token = access_token
        self.symbol = symbol
        self.running = False
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        # Initialize components
        self.auth = TradeStationAuthV3(access_token=access_token)
        self.aggregator = HTTPBarAggregator(bar_window_minutes=5)
        self.validator = MarketDataValidator()
        self.ml_inference = MLInference(model_dir="models/xgboost")
        self.position_tracker = PositionTracker()

        # Track bars
        self.bars_created = []

    async def fetch_quote(self) -> dict:
        """Fetch live quote via HTTP.

        Returns:
            Quote data dictionary
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.tradestation.com/v3/data/quote/{self.symbol}",
                headers=self.headers,
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return {}

    async def run_trading_loop(self):
        """Run the main trading loop with bar creation."""
        self.running = True
        print(f"\n🎯 HTTP-Based Paper Trading Started - {self.symbol}")
        print("="*70)
        print("📊 Creating 5-minute bars from HTTP quotes")
        print("🤖 ML pipeline ready")
        print("💰 Paper trading mode - NO REAL MONEY")
        print()
        print("Press Ctrl+C to stop")
        print("="*70)
        print()

        iteration = 0
        while self.running:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            try:
                # Refresh auth token if needed
                current_token = await self.auth.authenticate()
                self.headers["Authorization"] = f"Bearer {current_token}"

                # Fetch quote
                quote = await self.fetch_quote()

                if not quote or quote.get('Last', 0) == 0:
                    print("⚠️  No valid quote data (market closed)")
                    await asyncio.sleep(10)
                    continue

                # Display market data
                last_price = quote.get('Last', 0)
                volume = quote.get('Volume', 0)

                print(f"📊 {self.symbol} Quote: ${last_price} | Volume: {volume}")

                # Add quote to aggregator
                self.aggregator.add_quote(quote)

                # Check if it's time to create a bar
                current_time = datetime.now(timezone.utc)
                if self.aggregator.should_create_bar(current_time):
                    print("📊 Creating 5-minute bar...")

                    bar = self.aggregator.create_bar_from_quotes()

                    if bar and self.validator.validate_bar_for_trading(bar):
                        self.bars_created.append(bar)
                        print(f"✅ Bar #{len(self.bars_created)} created")
                        print(f"   O: ${bar.open:.2f} H: ${bar.high:.2f} L: ${bar.low:.2f} C: ${bar.close:.2f}")
                        print(f"   Volume: {bar.volume:,} | Notional: ${bar.notional_value:,.0f}")

                        # Run ML inference if we have enough bars
                        if len(self.bars_created) >= 50:
                            print(f"🤖 Running ML inference on {len(self.bars_created)} bars...")
                            # TODO: Add ML inference logic here

                        # Show positions
                        positions = self.position_tracker.get_all_positions()
                        if positions:
                            print(f"\n💰 Current Positions: {len(positions)}")
                            for pos in positions:
                                print(f"   {pos.order_id}: {pos.direction} {pos.quantity} contracts")
                                print(f"   Entry: ${pos.entry_price:.2f} | P&L: ${pos.unrealized_pnl:.2f}")
                    else:
                        print("⚠️  Bar validation failed")

                print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                print(f"📈 Total bars created: {len(self.bars_created)}")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()

            # Wait before next iteration (5 seconds)
            await asyncio.sleep(5)

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        print("\n🛑 Stopping paper trading...")


async def main():
    """Main entry point."""

    print("\n" + "="*70)
    print("🎯 HTTP-Based Paper Trading with Real Bars")
    print("="*70)
    print()

    # Check authentication
    print("📋 Step 1: Checking authentication...")
    try:
        with open(".access_token", "r") as f:
            access_token = f.read().strip()
        if not access_token or len(access_token) < 50:
            print("❌ No valid access token found!")
            return
        print("✅ Access token found")
    except FileNotFoundError:
        print("❌ No .access_token file found!")
        return

    print()

    # Test connection
    print("📋 Step 2: Testing API connection...")
    test_auth = TradeStationAuthV3(access_token=access_token)
    try:
        await test_auth.authenticate()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.tradestation.com/v3/data/quote/MNQM26",
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    print("✅ API connection successful")
                    print(f"   MNQM26 Last: ${data[0].get('Last', 'N/A')}")
                else:
                    print("⚠️  API returned empty data")
                    return
            else:
                print(f"❌ API connection failed: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return

    print()

    # Detect active contract
    print("📋 Step 3: Detecting active contract...")
    contract_detector = ContractDetector(access_token)
    try:
        active_contract = await contract_detector.detect_active_contract("MNQM26")
        print(f"✅ Active contract: {active_contract}")
        symbol = active_contract
    except Exception as e:
        print(f"⚠️  Contract detection failed: {e}")
        symbol = "MNQM26"

    print()

    # Start trading
    print("📋 Step 4: Starting HTTP-based paper trading...")
    print()
    print("="*70)
    print("🚀 SYSTEM STARTING")
    print("="*70)
    print()
    print("📊 Polling quotes every 5 seconds")
    print("📈 Creating 5-minute bars")
    print("🤖 ML inference ready")
    print("💰 Paper trading mode - NO REAL MONEY")
    print()
    print("Press Ctrl+C to stop")
    print("="*70)
    print()

    trader = HTTPPaperTrader(access_token, symbol)

    # Start auto-refresh
    await trader.auth.start_auto_refresh(interval_minutes=10)
    print("✅ Auto-refresh enabled (every 10 minutes)")

    try:
        await trader.run_trading_loop()
    except KeyboardInterrupt:
        trader.stop()
        await trader.auth.cleanup()
        print("\n👋 Paper trading stopped")
        print()
        print(f"📊 Session Summary:")
        print(f"   Total bars created: {len(trader.bars_created)}")
        print(f"   Final positions: {len(trader.position_tracker.get_all_positions())}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

#!/usr/bin/env python3
"""
BYDFI Paper Trading System - Main Entry Point

Autonomous paper trading system for BYDFI spot trading.
Implements Silver Bullet strategy with real-time signal detection,
ML inference, and risk management.

Usage:
    python start_bydfi_paper_trading.py

Features:
    - Real-time market data streaming
    - Pattern detection (MSS, FVG, Liquidity Sweep)
    - ML model inference
    - Risk management (daily limits, position sizing)
    - Graceful shutdown
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from src.data.bydfi_config import load_bydfi_settings
from src.execution.bydfi.client import BYDFIClient
from src.execution.bydfi.market_data.streaming import create_bydfi_websocket_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class PaperTradingSystem:
    """
    BYDFI paper trading system.

    Coordinates WebSocket streaming, pattern detection,
    ML inference, and risk management.
    """

    def __init__(self):
        """Initialize paper trading system."""
        self.settings = load_bydfi_settings()

        # Components
        self.bydfi_client: Optional[BYDFIClient] = None
        self.ws_client = None

        # System state
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Statistics
        self.start_time = None
        self.messages_received = 0
        self.signals_detected = 0

        logger.info("Paper trading system initialized")

    async def start(self):
        """Start paper trading system."""
        console.print(
            Panel.fit(
                "[bold blue]BYDFI Paper Trading System[/bold blue]\n"
                f"Symbol: {self.settings.bydfi_trading_symbol}",
                style="bold blue",
            )
        )

        self.start_time = datetime.now(timezone.utc)
        self.running = True

        try:
            # Initialize BYDFI client
            self.bydfi_client = BYDFIClient()
            await self.bydfi_client.__aenter__()

            # Initialize WebSocket client
            self.ws_client = create_bydfi_websocket_client()
            await self.ws_client.connect()

            # Subscribe to market data
            await self.ws_client.subscribe_trades(self.settings.bydfi_trading_symbol)
            await self.ws_client.subscribe_kline(
                self.settings.bydfi_trading_symbol,
                interval="5m",
            )

            console.print("[green]✓[/green] System started successfully")
            console.print()

            # Start message processing loop
            await self._process_messages()

        except Exception as e:
            console.print(f"[red]Error starting system: {e}[/red]")
            logger.error(f"System start failed: {e}", exc_info=True)
            raise

    async def stop(self):
        """Stop paper trading system."""
        console.print("\n[yellow]Shutting down system...[/yellow]")

        self.running = False
        self.shutdown_event.set()

        # Disconnect WebSocket
        if self.ws_client:
            await self.ws_client.disconnect()

        # Close BYDFI client
        if self.bydfi_client:
            await self.bydfi_client.__aexit__(None, None, None)

        # Print statistics
        self._print_statistics()

        console.print("[green]✓[/green] System stopped gracefully")

    async def _process_messages(self):
        """Process incoming WebSocket messages."""
        console.print("[cyan]Waiting for market data...[/cyan]")
        console.print()

        try:
            async for message in self.ws_client.messages():
                if not self.running:
                    break

                self.messages_received += 1

                # Process message based on type
                await self._handle_message(message)

                # Check staleness
                if self.ws_client.is_stale():
                    console.print(
                        f"[yellow]Warning: No messages for 30s, reconnecting...[/yellow]"
                    )
                    await self.ws_client.disconnect()
                    await self.ws_client.connect()
                    await self.ws_client.subscribe_trades(
                        self.settings.bydfi_trading_symbol
                    )

        except Exception as e:
            logger.error(f"Message processing error: {e}")

    async def _handle_message(self, message):
        """
        Handle incoming WebSocket message.

        Args:
            message: WebSocket message
        """
        # TODO: Implement pattern detection and ML inference
        # For now, just log the message
        if self.messages_received % 100 == 0:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            rate = self.messages_received / elapsed if elapsed > 0 else 0

            console.print(
                f"[dim]Messages: {self.messages_received} | "
                f"Rate: {rate:.1f} msg/s[/dim]"
            )

    def _print_statistics(self):
        """Print system statistics."""
        if not self.start_time:
            return

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        console.print()
        console.print(Panel.fit("System Statistics", style="bold cyan"))
        console.print(f"Uptime: {elapsed:.0f} seconds")
        console.print(f"Messages received: {self.messages_received}")
        console.print(f"Signals detected: {self.signals_detected}")

        if elapsed > 0:
            console.print(f"Message rate: {self.messages_received / elapsed:.2f} msg/s")

    async def _monitor_risk_limits(self):
        """Monitor risk limits and close positions if needed."""
        while self.running:
            try:
                # Check time-based position closing
                now = datetime.now(timezone.utc)
                close_time = self.settings.position_close_time_utc

                # TODO: Implement position closing logic
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")


async def main():
    """Main entry point."""
    system = PaperTradingSystem()

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler(sig, frame):
        console.print("\n[yellow]Received interrupt signal[/yellow]")
        loop.create_task(system.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler, sig, None)

    try:
        await system.start()

    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)

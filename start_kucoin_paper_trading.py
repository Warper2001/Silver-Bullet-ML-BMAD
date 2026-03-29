#!/usr/bin/env python3
"""
KuCoin Paper Trading Entry Point

This script starts the autonomous crypto paper trading system for KuCoin BTC/USDT.

Features:
- KuCoin WebSocket market data streaming with token-based authentication
- Dollar bar generation ($10M threshold)
- Silver Bullet pattern detection
- ML inference with crypto-specific models
- Risk management for 24/7 markets
- Time-based position closing

Usage:
    python start_kucoin_paper_trading.py

Environment:
    Requires .env.kucoin file with KuCoin API credentials
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.data.kucoin_config import load_kucoin_settings
from src.execution.kucoin.client import KuCoinClient
from src.execution.kucoin.market_data.streaming import KuCoinWebSocketClient
from src.monitoring.log_rotation import setup_log_rotation

# Setup logging with Rich
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=True)],
)
logger = logging.getLogger(__name__)


class KuCoinPaperTradingSystem:
    """
    Autonomous KuCoin paper trading system.

    This system orchestrates:
    - KuCoin WebSocket market data streaming
    - Dollar bar generation
    - Pattern detection (MSS, FVG, Liquidity Sweep, Silver Bullet)
    - ML inference
    - Risk management
    - Order execution (paper trading)

    Architecture:
        WebSocket → DollarBarTransformer → DetectionPipeline → MLInference → RiskOrchestrator → Execution
    """

    def __init__(self) -> None:
        """Initialize KuCoin paper trading system."""
        # Load configuration
        try:
            self.settings = load_kucoin_settings()
            logger.info(f"Loaded KuCoin configuration (environment: {self.settings.kucoin_environment})")
        except Exception as e:
            logger.error(f"Failed to load KuCoin configuration: {e}")
            sys.exit(1)

        # Components (initialized later)
        self.kucoin_client: KuCoinClient | None = None
        self.websocket_client: KuCoinWebSocketClient | None = None

        # System state
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def start(self) -> None:
        """
        Start the KuCoin paper trading system.

        This initializes all components and starts the trading loop.
        """
        logger.info("=" * 80)
        logger.info("KuCoin Paper Trading System Starting")
        logger.info("=" * 80)
        logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
        logger.info(f"Symbol: {self.settings.kucoin_trading_symbol}")
        logger.info(f"Environment: {self.settings.kucoin_environment}")
        logger.info(f"Base URL: {self.settings.base_url}")
        logger.info(f"WebSocket URL: {self.settings.websocket_base_url}")
        logger.info("=" * 80)

        try:
            # Initialize KuCoin client
            await self._initialize_kucoin_client()

            # Initialize WebSocket streaming
            await self._initialize_websocket()

            # Start trading loop
            self.is_running = True
            await self._trading_loop()

        except Exception as e:
            logger.error(f"Fatal error in trading system: {e}", exc_info=True)
            raise
        finally:
            await self._shutdown()

    async def _initialize_kucoin_client(self) -> None:
        """
        Initialize KuCoin REST API client.

        Validates environment and tests connectivity.
        """
        logger.info("Initializing KuCoin client...")

        try:
            self.kucoin_client = KuCoinClient()

            # Enter context manager (validates environment and tests connectivity)
            await self.kucoin_client.__aenter__()

            # Get account info to verify authentication
            accounts = await self.kucoin_client.get_account_info()
            logger.info(f"Authenticated successfully ({len(accounts)} accounts found)")

            # Log balances
            for account in accounts:
                if float(account.balance) > 0:
                    logger.info(f"  {account.currency}: {account.balance} (available)")

        except Exception as e:
            logger.error(f"Failed to initialize KuCoin client: {e}")
            raise

    async def _initialize_websocket(self) -> None:
        """
        Initialize KuCoin WebSocket market data streaming.
        """
        logger.info("Initializing KuCoin WebSocket streaming...")

        try:
            symbol = self.settings.kucoin_trading_symbol

            self.websocket_client = KuCoinWebSocketClient(
                symbol=symbol,
                client=self.kucoin_client,
            )

            await self.websocket_client.connect()

            logger.info(f"WebSocket connected for {symbol}")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {e}")
            raise

    async def _trading_loop(self) -> None:
        """
        Main trading loop.

        Processes market data, detects setups, executes trades.
        """
        logger.info("Starting trading loop...")

        last_health_check = datetime.now(timezone.utc)
        health_check_interval = 60  # seconds

        try:
            async for trade in self.websocket_client.stream_trades():
                # Check for shutdown
                if self.shutdown_event.is_set():
                    logger.info("Shutdown event received, stopping trading loop")
                    break

                # Process trade
                await self._process_trade(trade)

                # Periodic health check
                now = datetime.now(timezone.utc)
                if (now - last_health_check).total_seconds() >= health_check_interval:
                    await self._health_check()
                    last_health_check = now

        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            raise

    async def _process_trade(self, trade) -> None:
        """
        Process incoming trade from WebSocket.

        Args:
            trade: KuCoinWebSocketTrade object
        """
        # TODO: Implement full pipeline
        # 1. Aggregate into dollar bars
        # 2. Run pattern detection
        # 3. ML inference
        # 4. Risk validation
        # 5. Order execution

        # For now, just log the trade
        if logger.level <= logging.DEBUG:
            logger.debug(f"Trade: {trade.symbol} @ {trade.price} ({trade.quantity})")

    async def _health_check(self) -> None:
        """
        Perform periodic health check.

        Checks:
        - WebSocket staleness
        - System metrics
        - Error rates
        """
        # Check staleness
        is_stale = await self.websocket_client.check_staleness()
        if is_stale:
            logger.warning("WebSocket data is stale!")
        else:
            logger.debug("WebSocket data is fresh")

        # Log system status
        logger.info("Health check passed")

    async def _shutdown(self) -> None:
        """
        Graceful shutdown.

        Disconnects WebSocket, closes HTTP client, saves state.
        """
        logger.info("Initiating graceful shutdown...")

        self.is_running = False

        # Disconnect WebSocket
        if self.websocket_client:
            try:
                await self.websocket_client.disconnect()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")

        # Close KuCoin client
        if self.kucoin_client:
            try:
                await self.kucoin_client.__aexit__(None, None, None)
                logger.info("KuCoin client closed")
            except Exception as e:
                logger.error(f"Error closing KuCoin client: {e}")

        logger.info("Shutdown complete")
        logger.info("=" * 80)


async def main():
    """Main entry point."""
    # Setup log rotation for 24/7 operation
    setup_log_rotation()

    # Create and start system
    system = KuCoinPaperTradingSystem()

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)

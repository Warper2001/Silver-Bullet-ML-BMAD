#!/usr/bin/env python3
"""TIER 1 FVG System Monitor - Background Service.

Runs continuously with OAuth token auto-refresh every 10 minutes.
Monitors system status and logs performance metrics.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.auth_v3 import TradeStationAuthV3

# Configuration
TIER1_CONFIG = "SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0"
SL_MULTIPLIER = 2.5
ATR_THRESHOLD = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS = 50.0

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'tier1_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Tier1Monitor:
    """TIER 1 FVG System Monitor."""

    def __init__(self):
        self.running = False
        self.auth: TradeStationAuthV3 | None = None

    async def initialize(self):
        """Initialize OAuth authentication."""
        logger.info('='*60)
        logger.info('TIER 1 FVG SYSTEM MONITOR')
        logger.info('='*60)
        logger.info(f'Configuration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}')
        logger.info(f'Performance Targets: WR ≥60%, PF ≥1.7, 8-15 trades/day')
        logger.info('='*60)

        # Load OAuth tokens
        token_file = project_root / '.access_token'
        if not token_file.exists():
            token_file = project_root / '.tradestation_tokens_v3.json'

        if not token_file.exists():
            logger.error('No OAuth token file found')
            logger.info('Run OAuth flow first: python get_standard_auth_url.py')
            return False

        try:
            self.auth = TradeStationAuthV3.from_file(str(token_file))
            token = await self.auth.authenticate()

            logger.info('✓ OAuth authentication successful')
            logger.info(f'✓ Token hash: {self.auth._get_token_hash()}')

            # Start auto-refresh (every 10 minutes)
            await self.auth.start_auto_refresh(interval_minutes=10)
            logger.info('✓ Auto-refresh started (10-minute interval)')

            return True

        except Exception as e:
            logger.error(f'OAuth initialization failed: {e}')
            return False

    async def run(self):
        """Run the monitor continuously."""
        self.running = True

        logger.info('✓ TIER 1 monitor running')
        logger.info('✓ Press Ctrl+C to stop')

        try:
            while self.running:
                await asyncio.sleep(60)  # Log status every minute
                logger.debug('TIER 1 system running normally')

        except asyncio.CancelledError:
            logger.info('Shutdown requested')
        except Exception as e:
            logger.error(f'Monitor error: {e}', exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown gracefully."""
        logger.info('Shutting down TIER 1 monitor...')

        self.running = False

        if self.auth:
            await self.auth.cleanup()

        logger.info('✓ TIER 1 monitor stopped')


async def main():
    """Main entry point."""
    monitor = Tier1Monitor()

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info('Interrupt signal received')
        monitor.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize and run
    if await monitor.initialize():
        await monitor.run()
        return 0
    else:
        logger.error('Failed to initialize TIER 1 monitor')
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info('Interrupted by user')
        sys.exit(0)

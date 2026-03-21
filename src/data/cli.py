"""CLI entry point for TradeStation historical data downloader.

This module provides command-line interface for downloading historical MNQ
futures data from TradeStation API with OAuth authentication.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .historical_downloader import (
    DiskSpaceError,
    HistoricalDownloader,
    LockFileExistsError,
)
from .tradestation_auth import TradeStationAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_AUTH_FAILURE = 1
EXIT_NETWORK_ERROR = 2
EXIT_DISK_FULL = 3
EXIT_LOCK_FILE_EXISTS = 4
EXIT_INVALID_CONFIG = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data.cli",
        description="Download historical MNQ futures data from TradeStation API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all 8 quarterly contracts (default)
  python -m src.data.cli

  # Download to custom directory
  python -m src.data.cli --data-dir /path/to/data

  # Dry run - show contracts without downloading
  python -m src.data.cli --dry-run

  # Force override - re-download all contracts
  python -m src.data.cli --force-override

  # Batch mode - use cached tokens, fail if not available
  python -m src.data.cli --batch-mode

Exit Codes:
  0 - Success
  1 - Authentication failure
  2 - Network error
  3 - Disk full
  4 - Lock file exists (another instance running)
  5 - Invalid configuration
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/historical/mnq"),
        help="Directory for storing HDF5 files (default: data/historical/mnq)",
    )

    parser.add_argument(
        "--months-back",
        type=int,
        default=24,
        help="Number of months to go back (default: 24 = 8 quarters)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show contracts without downloading",
    )

    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use cached tokens without OAuth flow (fail if not available)",
    )

    parser.add_argument(
        "--force-override",
        action="store_true",
        help="Ignore checkpoint and download all contracts",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    """Async main function.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("TradeStation Historical Data Downloader")
    logger.info("=" * 70)

    # Initialize authentication
    auth = TradeStationAuth()

    # Check for cached tokens in batch mode
    if args.batch_mode:
        token_cache = auth.load_tokens_from_cache()
        if token_cache is None or not token_cache.is_valid:
            logger.error("Batch mode requested but no valid cached tokens found")
            return EXIT_AUTH_FAILURE
        logger.info("Using cached tokens (batch mode)")

    # Dry run - show contracts and exit
    if args.dry_run:
        logger.info("Dry run mode - showing contracts to download:")
        from .futures_symbols import FuturesSymbolGenerator

        generator = FuturesSymbolGenerator()
        symbols = generator.generate_mnq_symbols(args.months_back)

        for symbol in symbols:
            print(f"  - {symbol}")

        logger.info(f"Total: {len(symbols)} contracts")
        return EXIT_SUCCESS

    # Attempt OAuth flow if no valid cached tokens
    if not args.batch_mode:
        token_cache = auth.load_tokens_from_cache()
        if token_cache is None or not token_cache.is_valid:
            logger.info("No valid cached tokens found, initiating OAuth flow")
            try:
                auth.reauthorize_from_scratch()
            except Exception as e:
                logger.error(f"OAuth authorization failed: {e}")
                return EXIT_AUTH_FAILURE
        else:
            logger.info("Using cached tokens")

    # Create downloader (initialize to None for safe cleanup)
    downloader = None
    try:
        downloader = HistoricalDownloader(data_dir=args.data_dir, auth=auth)
        # Start download
        await downloader.download_all_contracts(
            months_back=args.months_back,
            force_override=args.force_override,
        )

        return EXIT_SUCCESS

    except DiskSpaceError as e:
        logger.error(f"Disk space error: {e}")
        return EXIT_DISK_FULL

    except LockFileExistsError as e:
        logger.error(f"Lock file error: {e}")
        return EXIT_LOCK_FILE_EXISTS

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return EXIT_NETWORK_ERROR

    finally:
        if downloader is not None:
            await downloader.cleanup()


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code
    """
    args = parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        return EXIT_SUCCESS
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return EXIT_INVALID_CONFIG


if __name__ == "__main__":
    sys.exit(main())

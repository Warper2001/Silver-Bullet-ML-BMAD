"""Historical data downloader orchestrator for TradeStation MNQ futures.

This module orchestrates the download of historical MNQ futures data from
TradeStation API, with HDF5 storage, checkpointing, progress tracking,
and automatic token refresh scheduling.
"""

import asyncio
import fcntl
import json
import logging
import os
import signal
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from .exceptions import AuthenticationError, ConfigurationError
from .futures_symbols import FuturesSymbolGenerator
from .tradestation_auth import TradeStationAuth
from .tradestation_client import TradeStationClient
from .tradestation_models import BarData

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_DATA_DIR = Path("data/historical/mnq")
LOCK_FILE = DEFAULT_DATA_DIR / ".lock"
CHECKPOINT_FILE = DEFAULT_DATA_DIR / ".checkpoint.json"

# MNQ contract multiplier
MNQ_MULTIPLIER = 0.5

# Minimum disk space required (200 MB)
MIN_DISK_SPACE_MB = 200

# Maximum file size per contract (15 MB)
MAX_FILE_SIZE_MB = 15

# HDF5 dataset settings
COMPRESSION_LEVEL = 1


class DiskSpaceError(Exception):
    """Exception raised when insufficient disk space."""

    pass


class LockFileExistsError(Exception):
    """Exception raised when lock file already exists."""

    pass


class HistoricalDownloader:
    """Orchestrator for downloading historical MNQ futures data.

    Features:
    - Sequential download of 8 quarterly contracts
    - HDF5 storage with atomic writes
    - Checkpoint-based resume capability
    - Progress tracking with ETA
    - Automatic token refresh scheduling
    - Graceful shutdown on SIGINT/SIGTERM
    - Lock file to prevent concurrent execution
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        auth: Optional[TradeStationAuth] = None,
    ) -> None:
        """Initialize downloader.

        Args:
            data_dir: Directory for storing HDF5 files
            auth: TradeStationAuth instance (created if None)
        """
        self.data_dir = data_dir
        self.auth = auth or TradeStationAuth()
        self.client: Optional[TradeStationClient] = None
        self.symbol_generator = FuturesSymbolGenerator()

        # Scheduler for token refresh
        self.scheduler: Optional[BackgroundScheduler] = None

        # Checkpoint tracking
        self._checkpoint: dict = {}
        self._downloaded_symbols: set[str] = set()

        # Progress tracking
        self._start_time: float = 0
        self._bars_downloaded: int = 0
        self._current_contract_bars: int = 0
        self._estimated_total_bars: int = 0

        # Shutdown flag
        self._shutdown_requested = False

        # Register signal handlers
        self._register_signal_handlers()

    async def download_all_contracts(
        self,
        months_back: int = 24,
        force_override: bool = False,
    ) -> None:
        """Download all quarterly contracts.

        Args:
            months_back: Months to go back (default: 24 = 8 quarters)
            force_override: Ignore checkpoint and download all

        Raises:
            DiskSpaceError: If insufficient disk space
            LockFileExistsError: If another instance is running
        """
        logger.info("Starting historical data download")

        # Pre-flight checks
        self._check_disk_space()
        self._acquire_lock()

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint unless force override
        if not force_override:
            self._load_checkpoint()

        # Generate symbols to download
        symbols = self.symbol_generator.generate_mnq_symbols(months_back)

        # Filter out already downloaded contracts
        if not force_override:
            symbols = [s for s in symbols if s not in self._downloaded_symbols]

        if not symbols:
            logger.info("All contracts already downloaded")
            return

        logger.info(f"Downloading {len(symbols)} contracts: {symbols}")

        # Estimate total bars for progress tracking (252 trading days/year, 390 minutes/day)
        self._estimated_total_bars = len(symbols) * 252 * 390
        logger.info(f"Estimated total bars to download: ~{self._estimated_total_bars:,}")

        # Initialize client and scheduler
        self.client = TradeStationClient(self.auth)
        self._start_scheduler()

        self._start_time = time.time()

        try:
            # Download each contract sequentially
            for i, symbol in enumerate(symbols, 1):
                if self._shutdown_requested:
                    logger.info("Shutdown requested, stopping download")
                    break

                await self._download_contract(symbol, i, len(symbols))

        finally:
            # Cleanup
            self._stop_scheduler()
            await self.client.close() if self.client else None
            self._release_lock()
            self._log_summary()

    async def _download_contract(
        self, symbol: str, index: int, total: int
    ) -> None:
        """Download single contract data.

        Args:
            symbol: Contract symbol
            index: Contract index (1-based)
            total: Total number of contracts
        """
        logger.info(f"Contract {index}/{total}: {symbol}")

        # Determine date range for this contract
        start_date = self._get_contract_start_date(symbol)
        end_date = self._get_contract_end_date(symbol)

        logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")

        # Reset per-contract counters
        self._current_contract_bars = 0

        try:
            # Fetch bars from API
            bars = await self.client.get_historical_bars(
                symbol, start_date, end_date
            )

            if not bars:
                logger.warning(f"  No bars returned for {symbol}")
                return

            # Save to HDF5
            self._save_to_hdf5(symbol, bars)

            # Update counters
            self._bars_downloaded += len(bars)
            self._current_contract_bars += len(bars)

            # Log progress every 100 bars
            if self._bars_downloaded % 100 == 0:
                self._log_progress(symbol, self._bars_downloaded, self._estimated_total_bars)

            # Update checkpoint
            self._downloaded_symbols.add(symbol)
            self._save_checkpoint()

            logger.info(f"  ✅ Downloaded {len(bars)} bars for {symbol}")

        except Exception as e:
            logger.error(f"  ❌ Failed to download {symbol}: {e}")
            raise

    def _get_contract_start_date(self, symbol: str) -> datetime:
        """Get start date for contract data.

        Args:
            symbol: Contract symbol

        Returns:
            Start date (timezone-aware)
        """
        # Parse symbol to get year and quarter
        month_code = symbol[3:4]
        year_suffix = symbol[4:6]
        year = 2000 + int(year_suffix)

        # Get expiration month for this quarter
        month_mapping = {"H": 3, "M": 6, "U": 9, "Z": 12}
        expiration_month = month_mapping[month_code]

        # Start from beginning of quarter
        start_date = datetime(year, expiration_month - 2, 1, tzinfo=timezone.utc)

        return start_date

    def _get_contract_end_date(self, symbol: str) -> datetime:
        """Get end date for contract data.

        Args:
            symbol: Contract symbol

        Returns:
            End date (timezone-aware)
        """
        # Parse symbol to get year and quarter
        month_code = symbol[3:4]
        year_suffix = symbol[4:6]
        year = 2000 + int(year_suffix)

        # Get expiration month for this quarter
        month_mapping = {"H": 3, "M": 6, "U": 9, "Z": 12}
        expiration_month = month_mapping[month_code]

        # End at expiration (third Friday of month)
        # Approximate: use last day of month
        import calendar

        last_day = calendar.monthrange(year, expiration_month)[1]
        end_date = datetime(
            year, expiration_month, last_day, 23, 59, 59, tzinfo=timezone.utc
        )

        return end_date

    def _save_to_hdf5(self, symbol: str, bars: list[BarData]) -> None:
        """Save bars to HDF5 file atomically.

        Args:
            symbol: Contract symbol
            bars: List of bars to save

        Raises:
            ValueError: If file size exceeds maximum
        """
        file_path = self.data_dir / f"{symbol}.h5"

        # Write to temp file first
        temp_path = file_path.with_suffix(".tmp")

        try:
            # Prepare structured numpy array
            dt = np.dtype(
                [
                    ("timestamp", "i8"),
                    ("open", "f8"),
                    ("high", "f8"),
                    ("low", "f8"),
                    ("close", "f8"),
                    ("volume", "i8"),
                    ("notional_value", "f8"),
                ]
            )

            # Convert bars to numpy array
            data = np.zeros(len(bars), dtype=dt)

            for i, bar in enumerate(bars):
                data[i] = (
                    int(bar.timestamp.timestamp() * 1e9),  # nanoseconds
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.close * bar.volume * MNQ_MULTIPLIER,  # notional
                )

            # Write to temp file
            with h5py.File(temp_path, "w") as h5file:
                dataset = h5file.create_dataset(
                    "historical_bars",
                    data=data,
                    compression="gzip",
                    compression_opts=COMPRESSION_LEVEL,
                )

                # Add metadata
                dataset.attrs["symbol"] = symbol
                dataset.attrs["count"] = len(bars)
                dataset.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
                dataset.attrs["multiplier"] = MNQ_MULTIPLIER

            # Validate file size
            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                temp_path.unlink()
                raise ValueError(
                    f"File size {file_size_mb:.2f} MB exceeds maximum {MAX_FILE_SIZE_MB} MB"
                )

            # Atomic rename
            temp_path.replace(file_path)

            logger.debug(
                f"Saved {len(bars)} bars to {file_path} ({file_size_mb:.2f} MB)"
            )

        except Exception as e:
            # Cleanup temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk."""
        if not CHECKPOINT_FILE.exists():
            return

        try:
            with open(CHECKPOINT_FILE, "r") as f:
                self._checkpoint = json.load(f)

            self._downloaded_symbols = set(self._checkpoint.get("symbols", []))

            logger.info(
                f"Loaded checkpoint: {len(self._downloaded_symbols)} contracts already downloaded"
            )

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk atomically with retry."""
        # Ensure directory exists
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "symbols": list(self._downloaded_symbols),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write to temp file first
        temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

        for attempt in range(3):
            try:
                with open(temp_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)

                # Atomic rename
                temp_file.replace(CHECKPOINT_FILE)

                logger.debug(f"Saved checkpoint: {len(self._downloaded_symbols)} contracts")
                return

            except IOError as e:
                if attempt < 2:
                    logger.warning(f"Checkpoint save failed (attempt {attempt + 1}): {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to save checkpoint after 3 attempts: {e}")

    def _check_disk_space(self) -> None:
        """Check for sufficient disk space.

        Raises:
            DiskSpaceError: If insufficient space
        """
        # Get disk stats
        stat = shutil.disk_usage(self.data_dir.parent)

        # Convert to MB
        free_mb = stat.free / (1024 * 1024)

        if free_mb < MIN_DISK_SPACE_MB:
            raise DiskSpaceError(
                f"Insufficient disk space: {free_mb:.0f} MB available, "
                f"{MIN_DISK_SPACE_MB} MB required"
            )

        logger.info(f"Disk space check passed: {free_mb:.0f} MB available")

    def _acquire_lock(self) -> None:
        """Acquire lock file to prevent concurrent execution.

        Raises:
            LockFileExistsError: If lock file exists
        """
        if LOCK_FILE.exists():
            # Check if PID is still running
            try:
                with open(LOCK_FILE, "r") as f:
                    pid = int(f.read().strip())

                # Check if process exists
                try:
                    os.kill(pid, 0)  # Signal 0 just checks if process exists
                    raise LockFileExistsError(
                        f"Another instance is running (PID {pid})"
                    )
                except OSError:
                    # Process not running, stale lock
                    logger.warning("Removing stale lock file")
                    LOCK_FILE.unlink()

            except (ValueError, IOError) as e:
                logger.warning(f"Could not read lock file: {e}")

        # Create lock file with our PID
        try:
            with open(LOCK_FILE, "w") as f:
                f.write(str(os.getpid()))

            # Set file permissions
            LOCK_FILE.chmod(0o644)

            logger.debug(f"Acquired lock file: {LOCK_FILE}")

        except IOError as e:
            raise LockFileExistsError(f"Failed to create lock file: {e}")

    def _release_lock(self) -> None:
        """Release lock file."""
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
                logger.debug("Released lock file")
        except OSError as e:
            logger.warning(f"Failed to remove lock file: {e}")

    def _start_scheduler(self) -> None:
        """Start token refresh scheduler."""
        self.scheduler = BackgroundScheduler()

        # Add token refresh job every 10 minutes
        self.scheduler.add_job(
            self._refresh_token_task,
            "interval",
            minutes=10,
            id="token_refresh",
        )

        self.scheduler.start()
        logger.info("Token refresh scheduler started (every 10 minutes)")

    def _stop_scheduler(self) -> None:
        """Stop token refresh scheduler."""
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            logger.info("Token refresh scheduler stopped")

    def _refresh_token_task(self) -> None:
        """Token refresh job for scheduler."""
        try:
            logger.info("Running scheduled token refresh")
            self.auth.refresh_access_token()
        except AuthenticationError as e:
            logger.error(f"Scheduled token refresh failed: {e}")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def _log_progress(
        self, symbol: str, current: int, total: int
    ) -> None:
        """Log download progress with ETA.

        Args:
            symbol: Contract symbol
            current: Current bar count
            total: Expected total bars (estimated)
        """
        elapsed = time.time() - self._start_time
        rate = current / elapsed if elapsed > 0 else 0

        if current < 100:
            eta_str = "Calculating ETA..."
        elif rate > 0:
            remaining_bars = total - current
            eta_seconds = remaining_bars / rate
            eta_str = f"ETA: {timedelta(seconds=int(eta_seconds))}"
        else:
            eta_str = "Calculating ETA..."

        percent = current / total * 100 if total > 0 else 0

        logger.info(f"  Progress: {current:,}/{total:,} ({percent:.1f}%) - {eta_str} ({rate:.0f} bars/sec)")

    def _log_summary(self) -> None:
        """Log download summary."""
        elapsed = time.time() - self._start_time
        minutes = elapsed / 60

        logger.info("=" * 70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total bars downloaded: {self._bars_downloaded}")
        logger.info(f"Total time: {minutes:.1f} minutes")
        logger.info(
            f"Throughput: {self._bars_downloaded / minutes:.0f} bars/minute"
            if minutes > 0
            else "Throughput: N/A"
        )
        logger.info(f"Contracts completed: {len(self._downloaded_symbols)}")
        logger.info("=" * 70)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._stop_scheduler()
        await self.client.close() if self.client else None
        self._release_lock()

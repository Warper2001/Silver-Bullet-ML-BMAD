"""HDF5 persistence for Dollar Bar data stream."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .models import DollarBar

logger = logging.getLogger(__name__)

# Default compression level (1 = fast compression)
DEFAULT_COMPRESSION_LEVEL = 1


class HDF5DataSink:
    """Persist DollarBar stream to HDF5 files.

    Handles:
    - Async consumption from Gap-Filled DollarBar queue
    - Date-based file rotation (one file per day)
    - Append-only writes with HDF5 format
    - Gzip compression for storage efficiency
    - Timestamp indexing for rapid retrieval
    """

    def __init__(
        self,
        gap_filled_queue: asyncio.Queue[DollarBar],
        data_directory: str,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ) -> None:
        """Initialize HDF5 data sink.

        Args:
            gap_filled_queue: Queue receiving gap-filled DollarBar from Story 1.6
            data_directory: Root directory for HDF5 files
                (e.g., 'data/processed/dollar_bars')
            compression_level: Gzip compression level (0-9, default 1 for speed)
        """
        self._gap_filled_queue = gap_filled_queue
        self._data_directory = Path(data_directory)
        self.compression_level = compression_level

        # Create data directory if it doesn't exist
        self._data_directory.mkdir(parents=True, exist_ok=True)

        # Metrics
        self._bars_written = 0
        self._last_log_time = datetime.now()
        self._current_file_path: Path | None = None
        self._current_date: datetime.date | None = None

        # HDF5 file handles (will be opened as needed)
        self._h5_files: dict[datetime.date, any] = {}

    async def consume(self) -> None:
        """Consume DollarBar stream and persist to HDF5.

        This runs in a background task and:
        1. Receives DollarBar from gap-filled queue
        2. Determines target file based on bar date
        3. Opens/creates HDF5 file for that date
        4. Appends bar to storage
        5. Tracks write latency and throughput metrics
        """
        logger.info("HDF5DataSink started")

        while True:
            try:
                # Receive DollarBar with timeout
                bar = await asyncio.wait_for(
                    self._gap_filled_queue.get(),
                    timeout=5.0,
                )

                write_start = asyncio.get_event_loop().time()
                await self._persist_bar(bar)
                write_latency_ms = (
                    asyncio.get_event_loop().time() - write_start
                ) * 1000

                # Log write latency if exceeds threshold
                if write_latency_ms > 10:
                    logger.warning(
                        f"Write latency exceeded 10ms: {write_latency_ms:.2f}ms"
                    )

                # Log metrics periodically
                self._log_persistence_metrics_periodically()

            except asyncio.TimeoutError:
                # No bars received - continue waiting
                continue

            except Exception as e:
                logger.error(f"Persistence error: {e}")
                # Continue processing - don't let one error stop the pipeline

    async def _persist_bar(self, bar: DollarBar) -> None:
        """Persist single DollarBar to HDF5 file.

        Args:
            bar: DollarBar to persist
        """
        bar_date = bar.timestamp.date()

        # Check if we need to rotate to a new file
        if self._current_date != bar_date:
            # Close current file if open
            if self._current_file_path is not None:
                self._close_current_file()

            # Open new file for new date
            self._current_date = bar_date
            self._current_file_path = self._get_file_path(bar.timestamp)
            self._open_file(bar_date)

        # Write bar to current file
        self._write_bar_to_file(bar)

        # Update metrics
        self._bars_written += 1

    def _get_file_path(self, timestamp: datetime) -> Path:
        """Get HDF5 file path for specific date.

        Args:
            timestamp: Bar timestamp

        Returns:
            Path to HDF5 file (e.g., 'data/processed/dollar_bars/2026/03-15.h5')
        """
        year_dir = self._data_directory / str(timestamp.year)
        year_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{timestamp.month:02d}-{timestamp.day:02d}.h5"
        return year_dir / filename

    def _open_file(self, date: datetime.date) -> None:
        """Open HDF5 file for specific date.

        Args:
            date: Bar date for file to open
        """
        import h5py

        file_path = self._current_file_path

        logger.info(f"Opening HDF5 file: {file_path}")

        # Create new file or open existing
        if not file_path.exists():
            # Create new HDF5 file
            import numpy as np

            with h5py.File(file_path, "w") as h5file:
                # Create dataset for DollarBar data
                # Note: timestamp stored as int64 (nanoseconds since Unix epoch)
                dt = np.dtype(
                    [
                        ("timestamp", "i8"),
                        ("open", "f8"),
                        ("high", "f8"),
                        ("low", "f8"),
                        ("close", "f8"),
                        ("volume", "i8"),
                        ("notional_value", "f8"),
                        ("is_forward_filled", "bool"),
                    ]
                )

                # Create dataset with initial size
                max_rows = 10000  # Expected max bars per day
                h5file.create_dataset(
                    "dollar_bars",
                    shape=(0,),
                    maxshape=(max_rows,),
                    dtype=dt,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )

        # Open file for appending
        self._h5_files[date] = h5py.File(file_path, "a")

    def _write_bar_to_file(self, bar: DollarBar) -> None:
        """Write bar to current HDF5 file.

        Args:
            bar: DollarBar to write
        """
        if self._current_date is None:
            raise RuntimeError("No file is currently open")

        h5file = self._h5_files[self._current_date]
        dataset = h5file["dollar_bars"]

        # Convert timestamp to int64 (nanoseconds since Unix epoch)
        timestamp_ns = int(bar.timestamp.timestamp() * 1e9)

        # Prepare row data
        row = (
            timestamp_ns,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.notional_value,
            bar.is_forward_filled,
        )

        # Resize dataset and append row
        current_size = dataset.shape[0]
        dataset.resize((current_size + 1,))
        dataset[current_size] = row

        # Flush to ensure data is written
        h5file.flush()

    def _close_current_file(self) -> None:
        """Close currently open HDF5 file if any."""
        if self._current_date is not None and self._current_date in self._h5_files:
            try:
                self._h5_files[self._current_date].close()
                logger.info(f"Closed HDF5 file: {self._current_file_path}")
            except Exception as e:
                logger.error(f"Error closing HDF5 file: {e}")
            finally:
                del self._h5_files[self._current_date]
                self._current_date = None
                self._current_file_path = None

    def _close_all_files(self) -> None:
        """Close all open HDF5 files."""
        for date in list(self._h5_files.keys()):
            try:
                self._h5_files[date].close()
            except Exception as e:
                logger.error(f"Error closing HDF5 file for {date}: {e}")

        self._h5_files.clear()
        self._current_date = None
        self._current_file_path = None

    def _log_persistence_metrics_periodically(self) -> None:
        """Log persistence metrics periodically (every 60 seconds)."""
        now = datetime.now()
        if (now - self._last_log_time).total_seconds() >= 60:
            current_file_size_mb = (
                self._get_current_file_size_bytes() / (1024 * 1024)
                if self._current_file_path and self._current_file_path.exists()
                else 0
            )

            logger.info(
                f"Persistence metrics: "
                f"bars_written={self._bars_written} "
                f"current_file="
                f"{self._current_file_path.name if self._current_file_path else 'None'} "  # noqa: E501
                f"file_size_mb={current_file_size_mb:.2f} "
                f"queue_depth={self._gap_filled_queue.qsize()}"
            )
            self._last_log_time = now

    def _get_current_file_size_bytes(self) -> int:
        """Get current HDF5 file size in bytes."""
        if self._current_file_path and self._current_file_path.exists():
            return self._current_file_path.stat().st_size
        return 0

    @property
    def bars_written(self) -> int:
        """Get total bars written since start."""
        return self._bars_written

    async def close(self) -> None:
        """Close all open HDF5 files and cleanup."""
        logger.info("Closing HDF5DataSink")
        self._close_all_files()

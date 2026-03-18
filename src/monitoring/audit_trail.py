"""Immutable Audit Trail for Trading System.

Maintains append-only log of all system actions with checksums
for integrity, daily rotation, gzip compression, and query capabilities.
"""

import csv
import gzip
import hashlib
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from typing import List


class AuditTrail:
    """Maintains immutable audit trail of all system actions.

    Provides append-only logging with checksums for integrity,
    daily file rotation, gzip compression, and query capabilities.
    """

    def __init__(
        self,
        log_directory: str = "data/logs",
        retention_days: int = 90,
        checksum_enabled: bool = True
    ):
        """Initialize audit trail system.

        Args:
            log_directory: Directory for log files
            retention_days: Days to keep log files
            checksum_enabled: Whether to validate checksums
        """
        self._log_directory = Path(log_directory)
        self._retention_days = retention_days
        self._checksum_enabled = checksum_enabled
        self._logger = logging.getLogger(__name__)

        # Create log directory
        self._log_directory.mkdir(parents=True, exist_ok=True)

    def log_action(
        self,
        action_type: str,
        entity_id: str,
        entity_type: str,
        details: dict
    ) -> None:
        """Log action to immutable audit trail.

        Args:
            action_type: Type of action (e.g., "SIGNAL_GENERATED")
            entity_id: Unique identifier for entity
            entity_type: Type of entity (signal, order, position, etc.)
            details: Action-specific data as dictionary
        """
        # Get current log file path
        log_file = self._get_current_log_file()

        # Prepare log entry
        timestamp = self._get_current_time().isoformat()
        details_json = json.dumps(details)
        row = [timestamp, action_type, entity_id, entity_type, details_json]

        # Calculate checksum if enabled
        if self._checksum_enabled:
            checksum = self._calculate_checksum(row)
            row.append(checksum)
        else:
            row.append("")

        # Append to log file (append-only mode)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not log_file.exists() or log_file.stat().st_size == 0:
                writer.writerow([
                    "timestamp", "action_type", "entity_id", "entity_type",
                    "details_json", "checksum"
                ])

            # Write log entry
            writer.writerow(row)

        self._logger.debug("Logged action: {} for entity {}".format(
            action_type, entity_id
        ))

    def query_by_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[dict]:
        """Query logs by date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of log entries as dictionaries
        """
        results = []
        current_date = start_date

        while current_date <= end_date:
            log_file = self._get_log_file_for_date(current_date)

            # Check for compressed file
            compressed_file = log_file.with_suffix('.csv.gz')

            if compressed_file.exists():
                results.extend(self._read_log_file(compressed_file))
            elif log_file.exists():
                results.extend(self._read_log_file(log_file))

            current_date += timedelta(days=1)

        return results

    def query_by_action_type(
        self,
        action_type: str,
        start_date: date,
        end_date: date
    ) -> List[dict]:
        """Query logs by action type within date range.

        Args:
            action_type: Type of action to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of matching log entries
        """
        all_entries = self.query_by_date_range(start_date, end_date)

        return [
            entry for entry in all_entries
            if entry["action_type"] == action_type
        ]

    def query_by_entity_id(
        self,
        entity_id: str,
        start_date: date,
        end_date: date
    ) -> List[dict]:
        """Query logs by entity ID within date range.

        Args:
            entity_id: Entity ID to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of matching log entries
        """
        all_entries = self.query_by_date_range(start_date, end_date)

        return [
            entry for entry in all_entries
            if entry["entity_id"] == entity_id
        ]

    def validate_integrity(self, log_file: Path) -> bool:
        """Validate log file integrity using checksums.

        Args:
            log_file: Path to log file to validate

        Returns:
            True if all checksums valid, False otherwise
        """
        if not self._checksum_enabled:
            return True

        # Handle compressed files
        if log_file.suffix == '.gz':
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'

        with open_func(log_file, mode, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header

            if header is None:
                return True

            for row in reader:
                if len(row) < 6:
                    return False

                # Recalculate checksum
                row_data = row[:5]
                stored_checksum = row[5]
                calculated_checksum = self._calculate_checksum(row_data)

                if stored_checksum != calculated_checksum:
                    self._logger.error(
                        "Checksum mismatch in {}: expected {}, got {}".format(
                            log_file, stored_checksum, calculated_checksum
                        )
                    )
                    return False

        return True

    def rotate_daily_logs(self) -> None:
        """Compress previous day's logs and clean up old files."""
        yesterday = (self._get_current_date() - timedelta(days=1))
        yesterday_log = self._get_log_file_for_date(yesterday)

        # Compress yesterday's log
        if yesterday_log.exists():
            compressed_file = yesterday_log.with_suffix('.csv.gz')

            if not compressed_file.exists():
                with open(yesterday_log, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Delete uncompressed file
                yesterday_log.unlink()
                self._logger.info("Compressed log file: {}".format(compressed_file))

        # Clean up old logs
        self._cleanup_old_logs()

    def _get_current_log_file(self) -> Path:
        """Get path to current day's log file.

        Returns:
            Path to today's log file
        """
        today = self._get_current_date()
        return self._log_directory / "audit_{}.csv".format(
            today.strftime("%Y-%m-%d")
        )

    def _get_log_file_for_date(self, query_date: date) -> Path:
        """Get path to log file for specific date.

        Args:
            query_date: Date to get log file for

        Returns:
            Path to log file (may be compressed)
        """
        return self._log_directory / "audit_{}.csv".format(
            query_date.strftime("%Y-%m-%d")
        )

    def _read_log_file(self, log_file: Path) -> List[dict]:
        """Read log file and return entries as dictionaries.

        Args:
            log_file: Path to log file

        Returns:
            List of log entries as dictionaries
        """
        entries = []

        # Check if file is compressed
        if log_file.suffix == '.gz':
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'

        with open_func(log_file, mode, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(dict(row))

        return entries

    def _calculate_checksum(self, row: List[str]) -> str:
        """Calculate MD5 checksum of row data.

        Args:
            row: List of row fields

        Returns:
            Hexadecimal MD5 checksum
        """
        # Join row data with delimiter
        row_str = ",".join(row)

        # Calculate MD5 hash
        return hashlib.md5(row_str.encode()).hexdigest()

    def _cleanup_old_logs(self) -> None:
        """Remove log files older than retention period."""
        cutoff_date = self._get_current_date() - timedelta(days=self._retention_days)

        for log_file in self._log_directory.glob("audit_*.csv*"):
            try:
                # Extract date from filename
                # Format: audit_YYYY-MM-DD.csv or audit_YYYY-MM-DD.csv.gz
                stem = log_file.stem  # Removes .gz if present
                parts = stem.split('_')[1].split('-')

                if len(parts) == 3:
                    file_date = date(int(parts[0]), int(parts[1]), int(parts[2]))

                    if file_date < cutoff_date:
                        log_file.unlink()
                        self._logger.info("Removed old log file: {}".format(log_file))
            except (ValueError, IndexError):
                # Skip files that don't match expected format
                continue

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)

    def _get_current_date(self) -> date:
        """Get current date.

        Returns:
            Current date
        """
        return self._get_current_time().date()

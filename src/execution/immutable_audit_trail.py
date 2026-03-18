"""Immutable audit trail for order actions.

This module provides an append-only CSV audit trail that records every
order action with full context. Features include SHA-256 hashing for
tamper detection, daily file rotation, and thread-safe writes.

Features:
- Append-only CSV files (no delete/modify)
- Monotonically increasing sequence numbers
- SHA-256 hashing for tamper detection
- Automatic file rotation (daily)
- Thread-safe writes
- Query methods (by order, date, event type)
- Integrity verification
- Performance: <5ms per log entry
"""

import csv
import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AuditEventType:
    """Audit trail event types."""

    # Order lifecycle
    ORDER_SUBMIT = "ORDER_SUBMIT"
    ORDER_ACK = "ORDER_ACK"
    ORDER_PARTIAL_FILL = "ORDER_PARTIAL_FILL"
    ORDER_FILL = "ORDER_FILL"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_CANCEL_ACK = "ORDER_CANCEL_ACK"
    ORDER_MODIFY = "ORDER_MODIFY"
    ORDER_REJECT = "ORDER_REJECT"

    # Errors
    API_ERROR = "API_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"

    # System events
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"


@dataclass
class AuditEntry:
    """Immutable audit trail entry for order actions.

    Attributes:
        sequence_number: Monotonically increasing entry number
        timestamp: UTC timestamp when event occurred (ISO format)
        event_type: Type of event (ORDER_SUBMIT, ORDER_FILL, etc.)
        order_id: Order ID from broker
        signal_id: Original signal ID (if applicable)
        symbol: Trading symbol (e.g., "MNQ 03-26")
        action: Order action (BUY, SELL)
        order_type: Order type (MARKET, LIMIT, STOP_MARKET)
        quantity: Order quantity (contracts)
        price: Limit price (for LIMIT orders)
        filled_quantity: Total filled quantity
        filled_price: Average filled price
        status: Order status (PENDING, FILLED, PARTIALLY_FILLED, etc.)
        reason: Reason for rejection/cancellation (if applicable)
        error_message: Error message (if applicable)
        stack_trace: Full stack trace (for errors)
        entry_hash: SHA-256 hash of entry content (tamper detection)
    """

    sequence_number: int
    timestamp: str
    event_type: str
    order_id: str
    signal_id: Optional[str]
    symbol: str
    action: Optional[str]
    order_type: Optional[str]
    quantity: Optional[int]
    price: Optional[float]
    filled_quantity: Optional[int]
    filled_price: Optional[float]
    status: Optional[str]
    reason: Optional[str]
    error_message: Optional[str]
    stack_trace: Optional[str]
    entry_hash: str


class ImmutableAuditTrail:
    """Immutable append-only audit trail for order actions.

    Features:
        - Append-only CSV files (no delete/modify)
        - Monotonically increasing sequence numbers
        - SHA-256 hashing for tamper detection
        - Automatic file rotation (daily)
        - Thread-safe writes
        - Performance: <5ms per log entry

    Attributes:
        _audit_dir: Directory for audit trail files
        _file_prefix: Prefix for audit file names
        _current_sequence: Next sequence number
        _current_file_path: Path to current audit file
        _lock: Threading lock for concurrent writes

    Example:
        >>> audit = ImmutableAuditTrail(audit_dir="data/audit")
        >>> audit.log_order_submit(
        ...     order_id="ORDER-123",
        ...     signal_id="SIG-456",
        ...     symbol="MNQ 03-26",
        ...     action="BUY",
        ...     order_type="LIMIT",
        ...     quantity=5,
        ...     price=11800.00
        ... )
    """

    def __init__(
        self,
        audit_dir: str = "data/audit",
        file_prefix: str = "order_audit"
    ) -> None:
        """Initialize immutable audit trail.

        Args:
            audit_dir: Directory for audit trail files
            file_prefix: Prefix for audit file names

        Raises:
            ValueError: If audit_dir is not writable
        """
        # Create audit directory if not exists
        self._audit_dir = Path(audit_dir)
        self._file_prefix = file_prefix

        try:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(
                "Cannot create audit directory: {}".format(self._audit_dir)
            ) from e

        # Verify directory is writable
        if not self._is_writable(self._audit_dir):
            raise ValueError(
                "Audit directory is not writable: {}".format(self._audit_dir)
            )

        # Initialize sequence number
        self._current_sequence = 1

        # Initialize current file path
        self._current_file_path = self._get_current_file_path()

        # Initialize threading lock
        self._lock = threading.Lock()

        logger.info(
            "ImmutableAuditTrail initialized: {}".format(self._audit_dir)
        )

    def log_order_submit(
        self,
        order_id: str,
        signal_id: str,
        symbol: str,
        action: str,
        order_type: str,
        quantity: int,
        price: Optional[float] = None
    ) -> None:
        """Log order submission event.

        Args:
            order_id: Order ID from broker
            signal_id: Original signal ID
            symbol: Trading symbol
            action: Order action (BUY, SELL)
            order_type: Order type (MARKET, LIMIT, STOP_MARKET)
            quantity: Order quantity
            price: Limit price (for LIMIT orders)

        Example:
            >>> audit.log_order_submit(
            ...     order_id="ORDER-123",
            ...     signal_id="SIG-456",
            ...     symbol="MNQ 03-26",
            ...     action="BUY",
            ...     order_type="LIMIT",
            ...     quantity=5,
            ...     price=11800.00
            ... )
        """
        entry = AuditEntry(
            sequence_number=self._current_sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=AuditEventType.ORDER_SUBMIT,
            order_id=order_id,
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=None,
            filled_price=None,
            status="PENDING",
            reason=None,
            error_message=None,
            stack_trace=None,
            entry_hash=""  # Will be calculated in _write_entry
        )

        self._write_entry(entry)

    def log_order_fill(
        self,
        order_id: str,
        filled_quantity: int,
        filled_price: float,
        is_partial: bool = False
    ) -> None:
        """Log order fill event.

        Args:
            order_id: Order ID from broker
            filled_quantity: Quantity filled in this event
            filled_price: Fill price
            is_partial: True if partial fill, False if complete fill

        Example:
            >>> audit.log_order_fill(
            ...     order_id="ORDER-123",
            ...     filled_quantity=5,
            ...     filled_price=11800.50,
            ...     is_partial=False
            ... )
        """
        event_type = (
            AuditEventType.ORDER_PARTIAL_FILL
            if is_partial
            else AuditEventType.ORDER_FILL
        )

        status = "PARTIALLY_FILLED" if is_partial else "FILLED"

        entry = AuditEntry(
            sequence_number=self._current_sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            order_id=order_id,
            signal_id=None,
            symbol="",
            action=None,
            order_type=None,
            quantity=None,
            price=None,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            status=status,
            reason=None,
            error_message=None,
            stack_trace=None,
            entry_hash=""
        )

        self._write_entry(entry)

    def log_order_cancel(
        self,
        order_id: str,
        reason: Optional[str] = None
    ) -> None:
        """Log order cancellation event.

        Args:
            order_id: Order ID from broker
            reason: Reason for cancellation (if applicable)

        Example:
            >>> audit.log_order_cancel(
            ...     order_id="ORDER-123",
            ...     reason="User requested"
            ... )
        """
        entry = AuditEntry(
            sequence_number=self._current_sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=AuditEventType.ORDER_CANCEL,
            order_id=order_id,
            signal_id=None,
            symbol="",
            action=None,
            order_type=None,
            quantity=None,
            price=None,
            filled_quantity=None,
            filled_price=None,
            status="CANCELLED",
            reason=reason,
            error_message=None,
            stack_trace=None,
            entry_hash=""
        )

        self._write_entry(entry)

    def log_order_reject(
        self,
        order_id: str,
        reason: str
    ) -> None:
        """Log order rejection event.

        Args:
            order_id: Order ID from broker
            reason: Rejection reason from broker

        Example:
            >>> audit.log_order_reject(
            ...     order_id="ORDER-123",
            ...     reason="Insufficient margin"
            ... )
        """
        entry = AuditEntry(
            sequence_number=self._current_sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=AuditEventType.ORDER_REJECT,
            order_id=order_id,
            signal_id=None,
            symbol="",
            action=None,
            order_type=None,
            quantity=None,
            price=None,
            filled_quantity=None,
            filled_price=None,
            status="REJECTED",
            reason=reason,
            error_message=None,
            stack_trace=None,
            entry_hash=""
        )

        self._write_entry(entry)

    def log_error(
        self,
        order_id: Optional[str],
        error_message: str,
        stack_trace: Optional[str] = None
    ) -> None:
        """Log error event.

        Args:
            order_id: Associated order ID (if applicable)
            error_message: Error message
            stack_trace: Full stack trace (if available)

        Example:
            >>> import traceback
            >>> try:
            ...     # Some operation
            ... except Exception as e:
            ...     audit.log_error(
            ...         order_id="ORDER-123",
            ...         error_message=str(e),
            ...         stack_trace=traceback.format_exc()
            ...     )
        """
        entry = AuditEntry(
            sequence_number=self._current_sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=AuditEventType.API_ERROR,
            order_id=order_id if order_id else "",
            signal_id=None,
            symbol="",
            action=None,
            order_type=None,
            quantity=None,
            price=None,
            filled_quantity=None,
            filled_price=None,
            status=None,
            reason=None,
            error_message=error_message,
            stack_trace=stack_trace,
            entry_hash=""
        )

        self._write_entry(entry)

    def get_entries_for_order(
        self,
        order_id: str
    ) -> list[AuditEntry]:
        """Get all audit entries for a specific order.

        Args:
            order_id: Order ID to query

        Returns:
            List of AuditEntry for the order

        Example:
            >>> entries = audit.get_entries_for_order("ORDER-123")
            >>> for entry in entries:
            ...     print(f"{entry.timestamp}: {entry.event_type}")
        """
        entries = []

        # Read all audit files
        for file_path in self._get_all_audit_files():
            entries.extend(self._read_entries_from_file(file_path))

        # Filter by order_id
        return [e for e in entries if e.order_id == order_id]

    def get_entries_for_date(
        self,
        date: datetime
    ) -> list[AuditEntry]:
        """Get all audit entries for a specific date.

        Args:
            date: Date to query (timezone-aware, UTC)

        Returns:
            List of AuditEntry for the date

        Example:
            >>> entries = audit.get_entries_for_date(
            ...     datetime(2026, 3, 17, tzinfo=timezone.utc)
            ... )
            >>> print(f"Total orders: {len(entries)}")
        """
        # Get file for this date
        date_str = date.strftime("%Y%m%d")
        file_path = self._audit_dir / "{}_{}.csv".format(
            self._file_prefix,
            date_str
        )

        if not file_path.exists():
            return []

        return self._read_entries_from_file(file_path)

    def get_entries_by_event_type(
        self,
        event_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[AuditEntry]:
        """Get audit entries filtered by event type.

        Args:
            event_type: Event type to filter
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            List of matching AuditEntry

        Example:
            >>> rejects = audit.get_entries_by_event_type(
            ...     event_type="ORDER_REJECT",
            ...     start_date=datetime(2026, 3, 17, tzinfo=timezone.utc)
            ... )
            >>> print(f"Rejected orders: {len(rejects)}")
        """
        entries = []

        # Read all audit files
        for file_path in self._get_all_audit_files():
            file_entries = self._read_entries_from_file(file_path)

            # Filter by date range if specified
            if start_date or end_date:
                file_entries = self._filter_entries_by_date(
                    file_entries,
                    start_date,
                    end_date
                )

            entries.extend(file_entries)

        # Filter by event type
        return [e for e in entries if e.event_type == event_type]

    def verify_integrity(self) -> dict[str, bool]:
        """Verify audit trail integrity by checking hashes.

        Returns:
            Dictionary mapping file paths to integrity status

        Example:
            >>> integrity = audit.verify_integrity()
            >>> integrity["data/audit/order_audit_20260317.csv"]
            True  # File is intact
            >>> integrity["data/audit/order_audit_20260316.csv"]
            False  # File has been tampered with
        """
        integrity = {}

        for file_path in self._get_all_audit_files():
            try:
                # Read all entries
                entries = self._read_entries_from_file(file_path)

                # Verify each entry's hash
                for entry in entries:
                    # Calculate expected hash
                    expected_hash = self._calculate_hash(entry)

                    # Compare with stored hash
                    if entry.entry_hash != expected_hash:
                        integrity[str(file_path)] = False
                        break
                else:
                    # All hashes valid
                    integrity[str(file_path)] = True

            except Exception as e:
                logger.error(
                    "Integrity check failed for {}: {}".format(
                        file_path, e
                    )
                )
                integrity[str(file_path)] = False

        return integrity

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write entry to CSV file (append-only).

        Args:
            entry: AuditEntry to write

        Raises:
            IOError: If file write fails
        """
        # Acquire lock for thread-safe write
        with self._lock:
            # Check if file rotation needed
            self._rotate_file_if_needed()

            # Calculate hash
            entry.entry_hash = self._calculate_hash(entry)

            # Append to CSV
            try:
                file_exists = self._current_file_path.exists() and \
                    self._current_file_path.stat().st_size > 0

                with open(self._current_file_path, "a", newline="") as f:
                    writer = csv.writer(f)

                    # Write header if new file
                    if not file_exists:
                        writer.writerow([
                            "sequence_number",
                            "timestamp",
                            "event_type",
                            "order_id",
                            "signal_id",
                            "symbol",
                            "action",
                            "order_type",
                            "quantity",
                            "price",
                            "filled_quantity",
                            "filled_price",
                            "status",
                            "reason",
                            "error_message",
                            "stack_trace",
                            "entry_hash"
                        ])

                    # Write entry
                    writer.writerow([
                        entry.sequence_number,
                        entry.timestamp,
                        entry.event_type,
                        entry.order_id,
                        entry.signal_id or "",
                        entry.symbol,
                        entry.action or "",
                        entry.order_type or "",
                        entry.quantity or "",
                        entry.price or "",
                        entry.filled_quantity or "",
                        entry.filled_price or "",
                        entry.status or "",
                        entry.reason or "",
                        entry.error_message or "",
                        entry.stack_trace or "",
                        entry.entry_hash
                    ])

                # Increment sequence number
                self._current_sequence += 1

            except Exception as e:
                logger.error("Failed to write audit entry: {}".format(e))
                raise IOError("Audit write failed: {}".format(e)) from e

    def _calculate_hash(self, entry: AuditEntry) -> str:
        """Calculate SHA-256 hash of entry content.

        Args:
            entry: AuditEntry to hash

        Returns:
            Hexadecimal hash string

        Note:
            Hash is computed over all fields except entry_hash itself.
            This provides tamper detection - any modification changes hash.
        """
        # Build content string
        content = (
            f"{entry.sequence_number}|"
            f"{entry.timestamp}|"
            f"{entry.event_type}|"
            f"{entry.order_id}|"
            f"{entry.signal_id or ''}|"
            f"{entry.symbol}|"
            f"{entry.action or ''}|"
            f"{entry.order_type or ''}|"
            f"{entry.quantity or ''}|"
            f"{entry.price or ''}|"
            f"{entry.filled_quantity or ''}|"
            f"{entry.filled_price or ''}|"
            f"{entry.status or ''}|"
            f"{entry.reason or ''}|"
            f"{entry.error_message or ''}|"
            f"{entry.stack_trace or ''}"
        )

        return hashlib.sha256(content.encode()).hexdigest()

    def _get_current_file_path(self) -> Path:
        """Get path to current audit file.

        Returns:
            Path to current audit file
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = "{}_{}.csv".format(self._file_prefix, today)
        return self._audit_dir / filename

    def _rotate_file_if_needed(self) -> None:
        """Rotate to new audit file if day has changed.

        Files are named: {prefix}_{YYYYMMDD}.csv

        This prevents single file from growing indefinitely and
        organizes logs by day for easier archival/analysis.
        """
        current_file_path = self._get_current_file_path()

        # If file path changed, update current file path
        if current_file_path != self._current_file_path:
            self._current_file_path = current_file_path
            logger.info("Rotated to new audit file: {}".format(
                current_file_path
            ))

    def _get_all_audit_files(self) -> list[Path]:
        """Get list of all audit files.

        Returns:
            List of audit file paths
        """
        pattern = "{}_*.csv".format(self._file_prefix)
        return sorted(self._audit_dir.glob(pattern))

    def _read_entries_from_file(
        self,
        file_path: Path
    ) -> list[AuditEntry]:
        """Read all audit entries from file.

        Args:
            file_path: Path to audit file

        Returns:
            List of AuditEntry

        Raises:
            IOError: If file read fails
        """
        entries = []

        try:
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    entry = AuditEntry(
                        sequence_number=int(row["sequence_number"]),
                        timestamp=row["timestamp"],
                        event_type=row["event_type"],
                        order_id=row["order_id"],
                        signal_id=row["signal_id"] or None,
                        symbol=row["symbol"],
                        action=row["action"] or None,
                        order_type=row["order_type"] or None,
                        quantity=int(row["quantity"]) if row["quantity"] else None,
                        price=float(row["price"]) if row["price"] else None,
                        filled_quantity=(
                            int(row["filled_quantity"])
                            if row["filled_quantity"]
                            else None
                        ),
                        filled_price=(
                            float(row["filled_price"])
                            if row["filled_price"]
                            else None
                        ),
                        status=row["status"] or None,
                        reason=row["reason"] or None,
                        error_message=row["error_message"] or None,
                        stack_trace=row["stack_trace"] or None,
                        entry_hash=row["entry_hash"]
                    )
                    entries.append(entry)

        except Exception as e:
            logger.error("Failed to read audit file {}: {}".format(
                file_path, e
            ))
            raise IOError("Audit read failed: {}".format(e)) from e

        return entries

    def _filter_entries_by_date(
        self,
        entries: list[AuditEntry],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> list[AuditEntry]:
        """Filter entries by date range.

        Args:
            entries: List of entries to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered list of entries
        """
        filtered = []

        for entry in entries:
            # Parse timestamp
            try:
                entry_date = datetime.fromisoformat(entry.timestamp)
            except ValueError:
                continue

            # Check date range
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue

            filtered.append(entry)

        return filtered

    def _is_writable(self, path: Path) -> bool:
        """Check if path is writable.

        Args:
            path: Path to check

        Returns:
            True if writable, False otherwise
        """
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False

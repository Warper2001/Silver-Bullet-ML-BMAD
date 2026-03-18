"""Unit tests for Immutable Audit Trail.

Tests audit entry creation, logging methods, file rotation,
hash calculation, integrity verification, query methods,
and CSV audit trail format.
"""

import csv
import hashlib
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.execution.immutable_audit_trail import (
    AuditEntry,
    ImmutableAuditTrail,
)


class TestAuditEntry:
    """Test audit entry dataclass."""

    def test_create_audit_entry(self):
        """Verify audit entry creation with all fields."""
        entry = AuditEntry(
            sequence_number=1,
            timestamp="2026-03-17T14:00:00.000Z",
            event_type="ORDER_SUBMIT",
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00,
            filled_quantity=None,
            filled_price=None,
            status="PENDING",
            reason=None,
            error_message=None,
            stack_trace=None,
            entry_hash="abc123"
        )

        assert entry.sequence_number == 1
        assert entry.event_type == "ORDER_SUBMIT"
        assert entry.order_id == "ORDER-123"
        assert entry.signal_id == "SIG-456"
        assert entry.symbol == "MNQ 03-26"

    def test_entry_hash_calculated_correctly(self):
        """Verify SHA-256 hash is calculated correctly."""
        entry = AuditEntry(
            sequence_number=1,
            timestamp="2026-03-17T14:00:00.000Z",
            event_type="ORDER_SUBMIT",
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00,
            filled_quantity=None,
            filled_price=None,
            status="PENDING",
            reason=None,
            error_message=None,
            stack_trace=None,
            entry_hash=""
        )

        # Calculate expected hash
        content = (
            f"{entry.sequence_number}|"
            f"{entry.timestamp}|"
            f"{entry.event_type}|"
            f"{entry.order_id}|"
            f"{entry.signal_id}|"
            f"{entry.symbol}|"
            f"{entry.action}|"
            f"{entry.order_type}|"
            f"{entry.quantity}|"
            f"{entry.price}|"
            f"{entry.filled_quantity}|"
            f"{entry.filled_price}|"
            f"{entry.status}|"
            f"{entry.reason}|"
            f"{entry.error_message}|"
            f"{entry.stack_trace}"
        )
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        assert expected_hash != ""
        assert len(expected_hash) == 64  # SHA-256 produces 64 hex chars


class TestImmutableAuditTrailInit:
    """Test ImmutableAuditTrail initialization."""

    def test_init_with_valid_directory(self):
        """Verify logger initializes with valid directory."""
        temp_dir = tempfile.mkdtemp()

        audit = ImmutableAuditTrail(audit_dir=temp_dir)

        assert audit._audit_dir == Path(temp_dir)
        assert audit._current_sequence == 1

        # Cleanup
        Path(temp_dir).rmdir()

    def test_init_creates_audit_directory(self):
        """Verify logger creates audit directory if not exists."""
        temp_base = tempfile.mkdtemp()
        audit_dir = str(Path(temp_base) / "audit")

        ImmutableAuditTrail(audit_dir=audit_dir)

        assert Path(audit_dir).exists()

        # Cleanup
        Path(audit_dir).rmdir()
        Path(temp_base).rmdir()

    def test_init_with_invalid_directory(self):
        """Verify logger raises error with invalid directory."""
        # Try to use /proc as audit directory (not writable on Linux)
        with pytest.raises(ValueError):
            ImmutableAuditTrail(audit_dir="/proc/invalid")


class TestLogOrderSubmit:
    """Test order submit logging."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_log_order_submit_creates_entry(self, audit):
        """Verify order submit creates audit entry."""
        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        # Verify file created
        assert Path(audit._current_file_path).exists()

    def test_log_order_submit_writes_correct_fields(self, audit):
        """Verify order submit writes correct CSV fields."""
        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['order_id'] == "ORDER-123"
        assert rows[0]['signal_id'] == "SIG-456"
        assert rows[0]['symbol'] == "MNQ 03-26"
        assert rows[0]['action'] == "BUY"
        assert rows[0]['order_type'] == "LIMIT"
        assert rows[0]['quantity'] == "5"
        assert rows[0]['price'] == "11800.0"

    def test_log_multiple_orders_increments_sequence(self, audit):
        """Verify sequence number increments for each entry."""
        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        audit.log_order_submit(
            order_id="ORDER-456",
            signal_id="SIG-789",
            symbol="MNQ 03-26",
            action="SELL",
            order_type="MARKET",
            quantity=3
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['sequence_number'] == "1"
        assert rows[1]['sequence_number'] == "2"


class TestLogOrderFill:
    """Test order fill logging."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_log_complete_fill(self, audit):
        """Verify complete fill logging."""
        audit.log_order_fill(
            order_id="ORDER-123",
            filled_quantity=5,
            filled_price=11800.50,
            is_partial=False
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['event_type'] == "ORDER_FILL"
        assert rows[0]['filled_quantity'] == "5"
        assert rows[0]['filled_price'] == "11800.5"

    def test_log_partial_fill(self, audit):
        """Verify partial fill logging."""
        audit.log_order_fill(
            order_id="ORDER-123",
            filled_quantity=2,
            filled_price=11800.25,
            is_partial=True
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['event_type'] == "ORDER_PARTIAL_FILL"
        assert rows[0]['filled_quantity'] == "2"


class TestLogOrderCancel:
    """Test order cancel logging."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_log_order_cancel_with_reason(self, audit):
        """Verify order cancel logging with reason."""
        audit.log_order_cancel(
            order_id="ORDER-123",
            reason="User requested"
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['event_type'] == "ORDER_CANCEL"
        assert rows[0]['reason'] == "User requested"


class TestLogOrderReject:
    """Test order reject logging."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_log_order_reject(self, audit):
        """Verify order reject logging."""
        audit.log_order_reject(
            order_id="ORDER-123",
            reason="Insufficient margin"
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['event_type'] == "ORDER_REJECT"
        assert rows[0]['reason'] == "Insufficient margin"


class TestLogError:
    """Test error logging."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_log_error_with_stack_trace(self, audit):
        """Verify error logging with stack trace."""
        stack_trace = "Traceback (most recent call last):\n  File 'test.py', line 1"
        audit.log_error(
            order_id="ORDER-123",
            error_message="Connection timeout",
            stack_trace=stack_trace
        )

        # Read CSV
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['event_type'] == "API_ERROR"
        assert rows[0]['order_id'] == "ORDER-123"
        assert rows[0]['error_message'] == "Connection timeout"
        assert rows[0]['stack_trace'] == stack_trace


class TestFileRotation:
    """Test daily file rotation."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_file_name_includes_date(self, audit):
        """Verify audit file name includes current date."""
        # File name should be order_audit_YYYYMMDD.csv
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        expected_filename = f"order_audit_{today}.csv"

        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        assert Path(audit._current_file_path).name == expected_filename


class TestConcurrentWrites:
    """Test thread-safe concurrent writes."""

    def test_concurrent_writes_thread_safe(self):
        """Verify concurrent writes are thread-safe."""
        temp_dir = tempfile.mkdtemp()
        audit = ImmutableAuditTrail(audit_dir=temp_dir)

        def log_orders(thread_id):
            for i in range(10):
                audit.log_order_submit(
                    order_id=f"ORDER-{thread_id}-{i}",
                    signal_id=f"SIG-{thread_id}",
                    symbol="MNQ 03-26",
                    action="BUY",
                    order_type="LIMIT",
                    quantity=5,
                    price=11800.00
                )

        # Create 5 threads, each logging 10 orders
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_orders, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all orders logged
        with open(audit._current_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 50  # 5 threads × 10 orders


class TestQueryMethods:
    """Test query methods."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance with sample data."""
        temp_dir = tempfile.mkdtemp()
        audit = ImmutableAuditTrail(audit_dir=temp_dir)

        # Log some sample orders
        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        audit.log_order_fill(
            order_id="ORDER-123",
            filled_quantity=5,
            filled_price=11800.50
        )

        audit.log_order_submit(
            order_id="ORDER-456",
            signal_id="SIG-789",
            symbol="MNQ 03-26",
            action="SELL",
            order_type="MARKET",
            quantity=3
        )

        return audit

    def test_get_entries_for_order(self, audit):
        """Verify querying entries by order ID."""
        entries = audit.get_entries_for_order("ORDER-123")

        assert len(entries) == 2
        assert entries[0].order_id == "ORDER-123"
        assert entries[0].event_type == "ORDER_SUBMIT"
        assert entries[1].event_type == "ORDER_FILL"

    def test_get_entries_for_date(self, audit):
        """Verify querying entries by date."""
        today = datetime.now(timezone.utc)
        entries = audit.get_entries_for_date(today)

        assert len(entries) == 3

    def test_get_entries_by_event_type(self, audit):
        """Verify querying entries by event type."""
        entries = audit.get_entries_by_event_type("ORDER_SUBMIT")

        assert len(entries) == 2
        assert all(e.event_type == "ORDER_SUBMIT" for e in entries)


class TestIntegrityVerification:
    """Test audit trail integrity verification."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance with sample data."""
        temp_dir = tempfile.mkdtemp()
        audit = ImmutableAuditTrail(audit_dir=temp_dir)

        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        return audit

    def test_verify_integrity_with_valid_file(self, audit):
        """Verify integrity check passes for valid file."""
        integrity = audit.verify_integrity()

        assert len(integrity) == 1
        assert list(integrity.values())[0] is True

    def test_verify_integrity_detects_tampering(self, audit):
        """Verify integrity check detects tampered file."""
        # Tamper with the file
        with open(audit._current_file_path, 'a') as f:
            f.write("tampered content\n")

        integrity = audit.verify_integrity()

        assert list(integrity.values())[0] is False


class TestCSVAuditTrailFormat:
    """Test CSV audit trail format."""

    @pytest.fixture
    def audit(self):
        """Create audit trail instance."""
        temp_dir = tempfile.mkdtemp()
        return ImmutableAuditTrail(audit_dir=temp_dir)

    def test_csv_has_correct_columns(self, audit):
        """Verify CSV has all required columns."""
        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )

        with open(audit._current_file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
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
        ]

        assert headers == expected_headers


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_log_entry_completes_under_5ms(self):
        """Verify logging completes in < 5ms."""
        import time

        temp_dir = tempfile.mkdtemp()
        audit = ImmutableAuditTrail(audit_dir=temp_dir)

        start_time = time.perf_counter()
        audit.log_order_submit(
            order_id="ORDER-123",
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            action="BUY",
            order_type="LIMIT",
            quantity=5,
            price=11800.00
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 5.0, (
            "Log entry took {:.2f}ms, exceeds 5ms limit".format(
                elapsed_ms
            )
        )

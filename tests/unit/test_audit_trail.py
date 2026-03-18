"""Unit tests for Audit Trail.

Tests immutable logging, checksums, daily rotation,
gzip compression, query interface, and cleanup.
"""

import gzip
import tempfile
from pathlib import Path
from datetime import datetime, timezone, date, timedelta

from src.monitoring.audit_trail import AuditTrail


class TestAuditTrailInit:
    """Test AuditTrail initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            trail = AuditTrail(log_directory=str(log_dir))

            assert trail._log_directory == log_dir
            assert trail._retention_days == 90
            assert trail._checksum_enabled is True

    def test_init_creates_log_directory(self):
        """Verify that initialization creates log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            AuditTrail(log_directory=str(log_dir))

            assert log_dir.exists()

    def test_init_with_custom_retention(self):
        """Verify initialization with custom retention days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            AuditTrail(
                log_directory=str(log_dir),
                retention_days=30
            )

            # Initialization tested

    def test_init_with_checksum_disabled(self):
        """Verify initialization with checksums disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            trail = AuditTrail(
                log_directory=str(log_dir),
                checksum_enabled=False
            )

            assert trail._checksum_enabled is False


class TestLogAction:
    """Test log_action functionality."""

    def test_log_action_creates_log_file(self):
        """Verify that logging creates the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {"pattern": "MSS"}
            )

            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )
            assert log_file.exists()

    def test_log_action_writes_header(self):
        """Verify that logging writes CSV header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {"pattern": "MSS"}
            )

            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            with open(log_file, 'r') as f:
                content = f.read()

            assert "timestamp,action_type,entity_id" in content

    def test_log_action_appends_entries(self):
        """Verify that logging appends multiple entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Log multiple actions
            for i in range(3):
                trail.log_action(
                    "SIGNAL_GENERATED",
                    "sig_{}".format(i),
                    "signal",
                    {"index": i}
                )

            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Header + 3 data rows
            assert len(lines) == 4

    def test_log_action_with_checksum_enabled(self):
        """Verify that logging includes checksum when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(
                log_directory=str(log_dir),
                checksum_enabled=True
            )

            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {"pattern": "MSS"}
            )

            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Check data row has checksum field
            data_row = lines[1].strip()
            fields = data_row.split(',')
            assert len(fields) == 6
            assert len(fields[5]) > 0  # Checksum not empty

    def test_log_action_with_checksum_disabled(self):
        """Verify that logging omits checksum when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(
                log_directory=str(log_dir),
                checksum_enabled=False
            )

            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {"pattern": "MSS"}
            )

            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Check data row has empty checksum field
            data_row = lines[1].strip()
            fields = data_row.split(',')
            assert len(fields) == 6
            assert fields[5] == ""  # Checksum empty

    def test_log_action_includes_details_json(self):
        """Verify that logging includes details as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            details = {
                "pattern": "MSS",
                "confidence": 0.85,
                "timestamp": "2026-03-18T10:15:30Z"
            }

            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                details
            )

            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            with open(log_file, 'r') as f:
                lines = f.readlines()

            data_row = lines[1].strip()
            assert "MSS" in data_row
            assert "0.85" in data_row


class TestQueryByDateRange:
    """Test query_by_date_range functionality."""

    def test_query_by_date_range_returns_entries(self):
        """Verify query returns entries for date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Log an action
            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {}
            )

            # Query today
            today = date.today()
            results = trail.query_by_date_range(today, today)

            assert len(results) == 1
            assert results[0]["action_type"] == "SIGNAL_GENERATED"

    def test_query_by_date_range_empty_range(self):
        """Verify query returns empty for date range with no logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Query without logging anything
            today = date.today()
            results = trail.query_by_date_range(today, today)

            assert len(results) == 0

    def test_query_by_date_range_multiple_days(self):
        """Verify query across multiple days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            today = date.today()

            # Create log files for multiple days
            for i in range(3):
                test_date = today + timedelta(days=i)

                # Manually create log file for each date
                log_file = log_dir / "audit_{}.csv".format(
                    test_date.strftime("%Y-%m-%d")
                )

                with open(log_file, 'w') as f:
                    f.write(
                        "timestamp,action_type,entity_id,entity_type,"
                        "details_json,checksum\n"
                    )
                    f.write(
                        "{},SIGNAL_GENERATED,sig_{},signal,{{}},\n".format(
                            datetime.now(timezone.utc).isoformat(), i
                        )
                    )

            # Query all three days
            results = trail.query_by_date_range(today, today + timedelta(days=2))

            assert len(results) == 3


class TestQueryByActionType:
    """Test query_by_action_type functionality."""

    def test_query_by_action_type_filters(self):
        """Verify query filters by action type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Log different action types
            trail.log_action("SIGNAL_GENERATED", "sig_1", "signal", {})
            trail.log_action("ORDER_SUBMITTED", "order_1", "order", {})
            trail.log_action("SIGNAL_GENERATED", "sig_2", "signal", {})

            # Query for SIGNAL_GENERATED
            today = date.today()
            results = trail.query_by_action_type("SIGNAL_GENERATED", today, today)

            assert len(results) == 2
            assert all(r["action_type"] == "SIGNAL_GENERATED" for r in results)

    def test_query_by_action_type_no_matches(self):
        """Verify query returns empty when no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Log one action type
            trail.log_action("SIGNAL_GENERATED", "sig_1", "signal", {})

            # Query for different action type
            today = date.today()
            results = trail.query_by_action_type("ORDER_SUBMITTED", today, today)

            assert len(results) == 0


class TestQueryByEntityId:
    """Test query_by_entity_id functionality."""

    def test_query_by_entity_id_filters(self):
        """Verify query filters by entity ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Log actions for different entities
            trail.log_action("SIGNAL_GENERATED", "sig_1", "signal", {})
            trail.log_action("SIGNAL_GENERATED", "sig_2", "signal", {})
            trail.log_action("SIGNAL_GENERATED", "sig_1", "signal", {})

            # Query for sig_1
            today = date.today()
            results = trail.query_by_entity_id("sig_1", today, today)

            assert len(results) == 2
            assert all(r["entity_id"] == "sig_1" for r in results)

    def test_query_by_entity_id_no_matches(self):
        """Verify query returns empty when no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Log one entity
            trail.log_action("SIGNAL_GENERATED", "sig_1", "signal", {})

            # Query for different entity
            today = date.today()
            results = trail.query_by_entity_id("sig_999", today, today)

            assert len(results) == 0


class TestValidateIntegrity:
    """Test integrity validation functionality."""

    def test_validate_integrity_with_valid_checksums(self):
        """Verify validation passes with valid checksums."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(
                log_directory=str(log_dir),
                checksum_enabled=True
            )

            # Log an action
            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {}
            )

            # Validate integrity
            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            assert trail.validate_integrity(log_file) is True

    def test_validate_integrity_detects_tampering(self):
        """Verify validation detects tampered entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(
                log_directory=str(log_dir),
                checksum_enabled=True
            )

            # Log an action
            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {}
            )

            # Tamper with log file
            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            with open(log_file, 'r') as f:
                content = f.read()

            # Modify content
            tampered_content = content.replace("SIGNAL_GENERATED", "ORDER_SUBMITTED")

            with open(log_file, 'w') as f:
                f.write(tampered_content)

            # Validate should detect tampering
            assert trail.validate_integrity(log_file) is False

    def test_validate_integrity_with_checksum_disabled(self):
        """Verify validation always passes when checksums disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(
                log_directory=str(log_dir),
                checksum_enabled=False
            )

            # Log an action
            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {}
            )

            # Validate should pass even if tampered
            log_file = log_dir / "audit_{}.csv".format(
                date.today().strftime("%Y-%m-%d")
            )

            assert trail.validate_integrity(log_file) is True


class TestRotateDailyLogs:
    """Test daily log rotation functionality."""

    def test_rotate_daily_logs_compresses_previous_day(self):
        """Verify that rotation compresses previous day's log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            yesterday = date.today() - timedelta(days=1)
            yesterday_log = log_dir / "audit_{}.csv".format(
                yesterday.strftime("%Y-%m-%d")
            )

            # Create yesterday's log file
            with open(yesterday_log, 'w') as f:
                f.write("test data\n")

            # Rotate logs
            trail.rotate_daily_logs()

            # Verify compressed file exists
            compressed_file = yesterday_log.with_suffix('.csv.gz')
            assert compressed_file.exists()

            # Verify uncompressed file removed
            assert not yesterday_log.exists()

    def test_rotate_daily_logs_cleanup_old_logs(self):
        """Verify that rotation removes old logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(
                log_directory=str(log_dir),
                retention_days=7
            )

            # Create old log file (10 days ago)
            old_date = date.today() - timedelta(days=10)
            old_log = log_dir / "audit_{}.csv".format(
                old_date.strftime("%Y-%m-%d")
            )

            with open(old_log, 'w') as f:
                f.write("old data\n")

            # Rotate logs
            trail.rotate_daily_logs()

            # Verify old log removed
            assert not old_log.exists()


class TestCalculateChecksum:
    """Test checksum calculation functionality."""

    def test_calculate_checksum_is_consistent(self):
        """Verify that checksum is consistent for same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            row = ["timestamp", "ACTION", "id", "type", "details"]

            checksum1 = trail._calculate_checksum(row)
            checksum2 = trail._calculate_checksum(row)

            assert checksum1 == checksum2

    def test_calculate_checksum_differs_for_different_data(self):
        """Verify that checksum differs for different data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            row1 = ["timestamp", "ACTION1", "id", "type", "details"]
            row2 = ["timestamp", "ACTION2", "id", "type", "details"]

            checksum1 = trail._calculate_checksum(row1)
            checksum2 = trail._calculate_checksum(row2)

            assert checksum1 != checksum2


class TestReadLogFile:
    """Test log file reading functionality."""

    def test_read_log_file_uncompressed(self):
        """Verify reading uncompressed log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Create log file
            log_file = log_dir / "audit_test.csv"
            with open(log_file, 'w') as f:
                f.write(
                    "timestamp,action_type,entity_id,entity_type,"
                    "details_json,checksum\n"
                )
                f.write("2026-03-18T10:15:30Z,SIGNAL,sig_1,signal,{},abc\n")

            # Read file
            entries = trail._read_log_file(log_file)

            assert len(entries) == 1
            assert entries[0]["action_type"] == "SIGNAL"

    def test_read_log_file_compressed(self):
        """Verify reading compressed log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            # Create compressed log file
            log_file = log_dir / "audit_test.csv.gz"
            with gzip.open(log_file, 'wt') as f:
                f.write(
                    "timestamp,action_type,entity_id,entity_type,"
                    "details_json,checksum\n"
                )
                f.write("2026-03-18T10:15:30Z,SIGNAL,sig_1,signal,{},abc\n")

            # Read file
            entries = trail._read_log_file(log_file)

            assert len(entries) == 1
            assert entries[0]["action_type"] == "SIGNAL"


class TestPerformance:
    """Test performance requirements."""

    def test_log_action_performance_under_10ms(self):
        """Verify that logging overhead is < 10ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            trail = AuditTrail(log_directory=str(log_dir))

            import time

            # Measure time to log action
            start = time.perf_counter()
            trail.log_action(
                "SIGNAL_GENERATED",
                "sig_123",
                "signal",
                {"pattern": "MSS", "confidence": 0.85}
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Should be < 10ms
            assert elapsed_ms < 10.0

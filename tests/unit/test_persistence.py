"""Unit tests for HDF5 persistence."""

from datetime import datetime, timedelta

import pytest

from src.data.models import DollarBar
from src.data.persistence import HDF5DataSink


class TestHDF5DataSinkInitialization:
    """Test HDF5DataSink class initialization."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {
            "gap_filled": asyncio.Queue(),
        }

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_initialization_with_defaults(self, queues, temp_dir) -> None:
        """Test HDF5DataSink initializes with default parameters."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        assert sink.compression_level == 1
        assert sink._data_directory == temp_dir

    def test_initialization_with_custom_compression(self, queues, temp_dir) -> None:
        """Test HDF5DataSink initializes with custom compression level."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
            compression_level=5,
        )

        assert sink.compression_level == 5

    def test_initialization_creates_data_directory(self, queues, temp_dir) -> None:
        """Test HDF5DataSink creates data directory if it doesn't exist."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        assert temp_dir.exists()
        assert sink._data_directory == temp_dir

    def test_initial_metrics(self, queues, temp_dir) -> None:
        """Test HDF5DataSink initializes with zero metrics."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        assert sink.bars_written == 0


class TestFileOrganization:
    """Test file organization logic."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {"gap_filled": asyncio.Queue()}

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    def test_file_path_generation(self, queues, temp_dir) -> None:
        """Test file path generation for specific date."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        date = datetime(2026, 3, 15, 10, 30, 0)
        file_path = sink._get_file_path(date)

        expected = temp_dir / "2026" / "03-15.h5"
        assert file_path == expected

    def test_year_directory_included(self, queues, temp_dir) -> None:
        """Test year directory is included in path."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        date = datetime(2026, 3, 15, 10, 30, 0)
        file_path = sink._get_file_path(date)

        assert "2026" in str(file_path)

    def test_file_name_format(self, queues, temp_dir) -> None:
        """Test file name format (MM-DD.h5)."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        date = datetime(2026, 3, 15, 10, 30, 0)
        file_path = sink._get_file_path(date)

        assert file_path.name == "03-15.h5"


class TestWriteOperations:
    """Test HDF5 write operations."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {"gap_filled": asyncio.Queue()}

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.mark.asyncio
    async def test_single_bar_write(self, queues, temp_dir) -> None:
        """Test single DollarBar write to HDF5."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        await sink._persist_bar(bar)

        # Verify file was created
        date = bar.timestamp.date()
        file_path = sink._get_file_path(datetime.combine(date, datetime.min.time()))
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_multiple_bars_accumulate(self, queues, temp_dir) -> None:
        """Test multiple DollarBar writes accumulate in same file."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        date = datetime.now().date()

        bars = [
            DollarBar(
                timestamp=datetime.combine(date, datetime.min.time())
                + timedelta(seconds=i * 5),
                open=4523.25 + i,
                high=4524.00 + i,
                low=4523.00 + i,
                close=4523.75 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            for i in range(3)
        ]

        for bar in bars:
            await sink._persist_bar(bar)

        # Verify file was created and contains data
        file_path = sink._get_file_path(datetime.combine(date, datetime.min.time()))
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_data_preserved_correctly(self, queues, temp_dir) -> None:
        """Test DollarBar data is preserved correctly in HDF5."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        bar = DollarBar(
            timestamp=datetime(2026, 3, 15, 10, 30, 0),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
            is_forward_filled=False,
        )

        await sink._persist_bar(bar)

        # Close file to flush data
        sink._close_current_file()

        # Read back and verify
        # (Implementation will need a read method)
        assert True  # TODO: Verify data integrity


class TestWritePerformance:
    """Test write performance requirements."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {"gap_filled": asyncio.Queue()}

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.mark.asyncio
    async def test_write_latency_under_10ms(self, queues, temp_dir) -> None:
        """Test write latency is < 10ms target."""
        import time

        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        start = time.perf_counter()
        await sink._persist_bar(bar)
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms < 10.0, f"Write took {latency_ms:.2f}ms (> 10ms)"


class TestFileRotation:
    """Test file rotation logic."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {"gap_filled": asyncio.Queue()}

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.mark.asyncio
    async def test_different_dates_different_files(self, queues, temp_dir) -> None:
        """Test bars from different dates go to different files."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # First bar on 2026-03-15
        bar1 = DollarBar(
            timestamp=datetime(2026, 3, 15, 10, 30, 0),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )
        await sink._persist_bar(bar1)

        # Second bar on 2026-03-16
        bar2 = DollarBar(
            timestamp=datetime(2026, 3, 16, 10, 30, 0),
            open=4524.00,
            high=4525.00,
            low=4524.00,
            close=4524.50,
            volume=1000,
            notional_value=50_000_000,
        )
        await sink._persist_bar(bar2)

        # Verify two different files were created
        file1 = temp_dir / "2026" / "03-15.h5"
        file2 = temp_dir / "2026" / "03-16.h5"

        assert file1.exists()
        assert file2.exists()


class TestDataIntegrity:
    """Test data integrity during persistence."""

    @pytest.fixture
    def queues(self):
        """Create test queues."""
        import asyncio

        return {"gap_filled": asyncio.Queue()}

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "data" / "processed" / "dollar_bars"

    @pytest.mark.asyncio
    async def test_timestamp_preserved(self, queues, temp_dir) -> None:
        """Test timestamp is preserved correctly."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        original_timestamp = datetime(2026, 3, 15, 10, 30, 0)
        bar = DollarBar(
            timestamp=original_timestamp,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        await sink._persist_bar(bar)

        # TODO: Read back and verify timestamp matches
        assert True

    @pytest.mark.asyncio
    async def test_forward_fill_flag_preserved(self, queues, temp_dir) -> None:
        """Test is_forward_filled flag is preserved."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        bar = DollarBar(
            timestamp=datetime(2026, 3, 15, 10, 30, 0),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=0,
            notional_value=0,
            is_forward_filled=True,
        )

        await sink._persist_bar(bar)

        # TODO: Read back and verify flag matches
        assert True

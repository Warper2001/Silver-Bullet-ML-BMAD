"""Integration tests for HDF5 persistence pipeline."""

from datetime import datetime, timedelta

import pytest

from src.data.models import DollarBar
from src.data.persistence import HDF5DataSink


class TestPersistencePipelineIntegration:
    """Test end-to-end persistence pipeline."""

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
    async def test_end_to_end_persistence(self, queues, temp_dir) -> None:
        """Test complete persistence pipeline from queue to HDF5 file."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # Simulate gap-filled bars being produced
        date = datetime(2026, 3, 15, 10, 0, 0)
        bars = [
            DollarBar(
                timestamp=date + timedelta(seconds=i * 5),
                open=4523.25 + i * 0.5,
                high=4524.00 + i * 0.5,
                low=4523.00 + i * 0.5,
                close=4523.75 + i * 0.5,
                volume=1000,
                notional_value=50_000_000,
                is_forward_filled=False,
            )
            for i in range(10)
        ]

        # Persist all bars
        for bar in bars:
            await sink._persist_bar(bar)

        # Close file to flush
        sink._close_current_file()

        # Verify file was created
        file_path = temp_dir / "2026" / "03-15.h5"
        assert file_path.exists()
        assert file_path.stat().st_size > 0

        # Verify metrics
        assert sink.bars_written == 10

    @pytest.mark.asyncio
    async def test_multiple_days_file_rotation(self, queues, temp_dir) -> None:
        """Test file rotation across multiple days."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # Day 1 bars
        day1_date = datetime(2026, 3, 15, 10, 0, 0)
        for i in range(5):
            bar = DollarBar(
                timestamp=day1_date + timedelta(seconds=i * 5),
                open=4523.25,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )
            await sink._persist_bar(bar)

        # Day 2 bars
        day2_date = datetime(2026, 3, 16, 10, 0, 0)
        for i in range(5):
            bar = DollarBar(
                timestamp=day2_date + timedelta(seconds=i * 5),
                open=4524.00,
                high=4525.00,
                low=4524.00,
                close=4524.50,
                volume=1000,
                notional_value=50_000_000,
            )
            await sink._persist_bar(bar)

        # Close all files
        sink._close_all_files()

        # Verify two files were created
        file1 = temp_dir / "2026" / "03-15.h5"
        file2 = temp_dir / "2026" / "03-16.h5"

        assert file1.exists()
        assert file2.exists()

        # Verify bars written count
        assert sink.bars_written == 10

    @pytest.mark.asyncio
    async def test_mixed_forward_and_real_bars(self, queues, temp_dir) -> None:
        """Test persistence of mixed real and forward-filled bars."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        date = datetime(2026, 3, 15, 10, 0, 0)

        # Real bar
        bar1 = DollarBar(
            timestamp=date,
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
            is_forward_filled=False,
        )
        await sink._persist_bar(bar1)

        # Forward-filled bar (gap fill)
        bar2 = DollarBar(
            timestamp=date + timedelta(seconds=5),
            open=4523.75,
            high=4523.75,
            low=4523.75,
            close=4523.75,
            volume=0,
            notional_value=0,
            is_forward_filled=True,
        )
        await sink._persist_bar(bar2)

        # Another real bar
        bar3 = DollarBar(
            timestamp=date + timedelta(seconds=10),
            open=4524.00,
            high=4525.00,
            low=4524.00,
            close=4524.50,
            volume=1000,
            notional_value=50_000_000,
            is_forward_filled=False,
        )
        await sink._persist_bar(bar3)

        # Close file to flush
        sink._close_current_file()

        # Verify file was created
        file_path = temp_dir / "2026" / "03-15.h5"
        assert file_path.exists()

        # Verify all bars were written
        assert sink.bars_written == 3


class TestPersistencePerformanceIntegration:
    """Test persistence performance at scale."""

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
    async def test_high_throughput_persistence(self, queues, temp_dir) -> None:
        """Test persistence under high throughput (simulating market hours)."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # Simulate 1 hour of market data at 5-second intervals
        # = 720 bars
        date = datetime(2026, 3, 15, 10, 0, 0)
        bars = [
            DollarBar(
                timestamp=date + timedelta(seconds=i * 5),
                open=4523.25 + (i % 100) * 0.01,  # Small price variations
                high=4524.00 + (i % 100) * 0.01,
                low=4523.00 + (i % 100) * 0.01,
                close=4523.75 + (i % 100) * 0.01,
                volume=1000,
                notional_value=50_000_000,
            )
            for i in range(720)
        ]

        # Persist all bars and measure time
        import time

        start = time.perf_counter()
        for bar in bars:
            await sink._persist_bar(bar)
        duration = time.perf_counter() - start

        # Close file to flush
        sink._close_current_file()

        # Verify all bars were written
        assert sink.bars_written == 720

        # Verify performance: 720 bars in reasonable time
        # (< 7.2 seconds = < 10ms per bar average)
        assert duration < 7.2, f"Took {duration:.2f}s for 720 bars"

        # Calculate throughput
        bars_per_second = 720 / duration
        print(f"\nThroughput: {bars_per_second:.2f} bars/second")

    @pytest.mark.asyncio
    async def test_large_file_size(self, queues, temp_dir) -> None:
        """Test persistence creates appropriately sized files."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # Simulate full trading day (6.5 hours = 4680 bars at 5-second intervals)
        date = datetime(2026, 3, 15, 10, 0, 0)
        bars = [
            DollarBar(
                timestamp=date + timedelta(seconds=i * 5),
                open=4523.25 + (i % 100) * 0.01,
                high=4524.00 + (i % 100) * 0.01,
                low=4523.00 + (i % 100) * 0.01,
                close=4523.75 + (i % 100) * 0.01,
                volume=1000,
                notional_value=50_000_000,
            )
            for i in range(4680)
        ]

        # Persist all bars
        for bar in bars:
            await sink._persist_bar(bar)

        # Close file to flush
        sink._close_current_file()

        # Check file size
        file_path = temp_dir / "2026" / "03-15.h5"
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Verify all bars were written
        assert sink.bars_written == 4680

        # File should be reasonably sized (< 10 MB for 4680 bars with compression)
        assert file_size_mb < 10.0, f"File size {file_size_mb:.2f}MB exceeds 10MB"
        print(f"\nFile size for 4680 bars: {file_size_mb:.2f}MB")


class TestPersistenceMetrics:
    """Test persistence metrics tracking."""

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
    async def test_bars_written_counter(self, queues, temp_dir) -> None:
        """Test bars_written counter increments correctly."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        date = datetime(2026, 3, 15, 10, 0, 0)

        # Initially zero
        assert sink.bars_written == 0

        # Write some bars
        for i in range(5):
            bar = DollarBar(
                timestamp=date + timedelta(seconds=i * 5),
                open=4523.25,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )
            await sink._persist_bar(bar)

        # Verify counter
        assert sink.bars_written == 5

    @pytest.mark.asyncio
    async def test_queue_depth_monitoring(self, queues, temp_dir) -> None:
        """Test queue depth is monitored during persistence."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # Add bars to queue
        date = datetime(2026, 3, 15, 10, 0, 0)
        for i in range(10):
            bar = DollarBar(
                timestamp=date + timedelta(seconds=i * 5),
                open=4523.25,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )
            await queues["gap_filled"].put(bar)

        # Verify queue depth
        assert sink._gap_filled_queue.qsize() == 10

    @pytest.mark.asyncio
    async def test_periodic_metrics_logging(self, queues, temp_dir) -> None:
        """Test periodic metrics logging doesn't cause errors."""
        sink = HDF5DataSink(
            queues["gap_filled"],
            str(temp_dir),
        )

        # Write some bars
        date = datetime(2026, 3, 15, 10, 0, 0)
        for i in range(3):
            bar = DollarBar(
                timestamp=date + timedelta(seconds=i * 5),
                open=4523.25,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )
            await sink._persist_bar(bar)

        # Call periodic metrics logging directly
        sink._log_persistence_metrics_periodically()

        # Should not raise any errors
        assert sink.bars_written == 3

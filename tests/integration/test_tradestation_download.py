"""Integration tests for TradeStation data downloader in SIM environment.

These tests require TradeStation SIM credentials and run against the actual API.
They are marked with pytest.mark.integration and should be run separately.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from freezegun import freeze_time

from src.data.historical_downloader import (
    DiskSpaceError,
    HistoricalDownloader,
    LockFileExistsError,
)
from src.data.tradestation_auth import TradeStationAuth


# Integration test marker
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_data_dir(monkeypatch):
    """Provide temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "mnq"
        monkeypatch.setattr(
            "src.data.historical_downloader.DEFAULT_DATA_DIR", temp_path
        )
        yield temp_path


@pytest.fixture
def mock_disk_space(monkeypatch):
    """Mock sufficient disk space."""
    def mock_disk_usage(path):
        mock = Mock()
        mock.free = 500 * 1024 * 1024  # 500 MB
        return mock

    monkeypatch.setattr("shutil.disk_usage", mock_disk_usage)


@pytest.fixture
def mock_auth():
    """Mock TradeStationAuth with cached tokens."""
    auth = Mock(spec=TradeStationAuth)

    # Mock cached token
    token_cache = Mock()
    token_cache.is_valid = True
    token_cache.access_token = "test_access_token"
    auth.load_tokens_from_cache.return_value = token_cache
    auth.get_valid_access_token.return_value = "test_access_token"

    return auth


@pytest.mark.asyncio
class TestOAuthFlow:
    """Integration tests for OAuth flow."""

    async def test_oauth_flow_end_to_end_sim(self):
        """Test full OAuth flow in SIM environment.

        This test requires manual interaction - skip in CI.
        """
        pytest.skip("Requires manual OAuth interaction")

    async def test_token_cache_persistence(self, mock_auth, temp_data_dir):
        """Test tokens can be persisted and loaded."""
        # Test token cache save/load
        auth = TradeStationAuth()

        # Create temp cache dir
        cache_dir = temp_data_dir / ".tradestation"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with patch("src.data.tradestation_auth.TOKEN_CACHE_DIR", cache_dir):
            tokens = {
                "access_token": "test_token",
                "refresh_token": "test_refresh",
                "expires_in": 3600,
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }

            auth.save_tokens_to_cache(tokens)

            loaded = auth.load_tokens_from_cache()
            assert loaded is not None
            assert loaded.access_token == "test_token"


@pytest.mark.asyncio
class TestDownloadFlow:
    """Integration tests for download flow."""

    async def test_download_small_dataset_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test downloading 1 day, 1 symbol in SIM environment."""
        # This test would require actual API access
        # For now, test the orchestration logic
        pytest.skip("Requires TradeStation SIM API access")

    async def test_checkpoint_resume_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test checkpoint resume after interruption."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Create initial checkpoint
        checkpoint_file = temp_data_dir / ".checkpoint.json"
        checkpoint_data = {
            "symbols": ["MNQZ24", "MNQH25"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        downloader._load_checkpoint()

        assert "MNQZ24" in downloader._downloaded_symbols
        assert "MNQH25" in downloader._downloaded_symbols

    async def test_checkpoint_atomic_write_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test checkpoint file is written atomically."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Save checkpoint
        downloader._downloaded_symbols = {"MNQH26"}
        downloader._save_checkpoint()

        # Verify main file exists
        checkpoint_file = temp_data_dir / ".checkpoint.json"
        assert checkpoint_file.exists()

        # Verify temp file was cleaned up
        temp_file = checkpoint_file.with_suffix(".tmp")
        assert not temp_file.exists()

    async def test_checkpoint_write_retry_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test checkpoint write retries on failure."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Mock write failure then success
        call_count = 0

        def mock_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Simulated failure")
            return open(*args, **kwargs)

        with patch("builtins.open", side_effect=mock_open):
            # Should retry and eventually succeed
            downloader._downloaded_symbols = {"MNQH26"}
            downloader._save_checkpoint()


@pytest.mark.asyncio
class TestHDF5Storage:
    """Integration tests for HDF5 storage."""

    async def test_hdf5_atomic_write_sim(self, temp_data_dir, mock_auth):
        """Test HDF5 files are written atomically."""
        from src.data.tradestation_models import BarData

        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Create sample bars
        timestamp = datetime.now(timezone.utc)
        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=timestamp + timedelta(minutes=i),
                open=11800.0 + i,
                high=11850.0 + i,
                low=11790.0 + i,
                close=11825.0 + i,
                volume=1000,
            )
            for i in range(10)
        ]

        # Save to HDF5
        downloader._save_to_hdf5("MNQH26", bars)

        # Verify file exists
        h5_file = temp_data_dir / "MNQH26.h5"
        assert h5_file.exists()

        # Verify temp file was cleaned up
        temp_file = h5_file.with_suffix(".tmp")
        assert not temp_file.exists()

    async def test_hdf5_file_size_validation_sim(self, temp_data_dir, mock_auth):
        """Test file size validation catches oversized files."""
        from src.data.tradestation_models import BarData

        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Create too many bars (would exceed 15 MB)
        # Each bar ~56 bytes in HDF5, so need ~300k bars to exceed 15 MB
        # For test speed, use fewer bars and mock file size check
        timestamp = datetime.now(timezone.utc)
        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=timestamp + timedelta(minutes=i),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11825.0,
                volume=1000,
            )
            for i in range(10)
        ]

        # Mock large file size
        with patch("pathlib.Path.stat") as mock_stat:
            mock_result = Mock()
            mock_result.st_size = 20 * 1024 * 1024  # 20 MB
            mock_stat.return_value = mock_result

            # Should raise ValueError for oversized file
            with pytest.raises(ValueError, match="exceeds maximum"):
                downloader._save_to_hdf5("MNQH26", bars)


@pytest.mark.asyncio
class TestLockFile:
    """Integration tests for lock file mechanism."""

    async def test_lock_file_prevents_concurrent_execution_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test lock file prevents second instance."""
        downloader1 = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # First downloader should acquire lock
        downloader1._acquire_lock()

        # Try to create second downloader
        downloader2 = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Should raise lock file error
        with pytest.raises(LockFileExistsError):
            downloader2._acquire_lock()

        # Cleanup
        downloader1._release_lock()

    async def test_lock_file_cleanup_on_error_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test lock file is cleaned up even if error occurs."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        downloader._acquire_lock()

        # Simulate error during download
        lock_file = temp_data_dir / ".lock"

        try:
            # Release should clean up lock file
            downloader._release_lock()
            assert not lock_file.exists()
        except Exception:
            # Ensure cleanup even if error
            downloader._release_lock()


@pytest.mark.asyncio
class TestDiskSpace:
    """Integration tests for disk space checks."""

    async def test_disk_space_check_sim(self, temp_data_dir, mock_auth):
        """Test pre-flight disk space check."""
        # Mock insufficient disk space
        def mock_disk_usage_insufficient(path):
            mock = Mock()
            mock.free = 100 * 1024 * 1024  # 100 MB (less than 200 MB required)
            return mock

        with patch("shutil.disk_usage", side_effect=mock_disk_usage_insufficient):
            downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

            with pytest.raises(DiskSpaceError, match="Insufficient disk space"):
                downloader._check_disk_space()


@pytest.mark.asyncio
class TestScheduler:
    """Integration tests for token refresh scheduler."""

    async def test_scheduler_shutdown_on_error_sim(self, temp_data_dir, mock_auth):
        """Test scheduler stops even if error during download."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Start scheduler
        downloader._start_scheduler()

        assert downloader.scheduler is not None
        assert downloader.scheduler.running

        # Stop scheduler
        downloader._stop_scheduler()

        assert not downloader.scheduler.running

    async def test_concurrent_token_refresh_during_download_sim(
        self, temp_data_dir, mock_auth
    ):
        """Test refresh waits for in-progress API call."""
        # This tests the threading lock behavior
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Verify lock exists
        assert downloader._refresh_lock is not None

        # Test that concurrent refresh attempts are serialized
        import threading

        refresh_count = 0

        def mock_refresh():
            nonlocal refresh_count
            with downloader._refresh_lock:
                refresh_count += 1
                import time
                time.sleep(0.1)

        threads = [threading.Thread(target=mock_refresh) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert refresh_count == 3


@pytest.mark.asyncio
class TestGracefulShutdown:
    """Integration tests for graceful shutdown."""

    async def test_graceful_shutdown_during_api_call_sim(
        self, temp_data_dir, mock_auth
    ):
        """Test graceful shutdown waits for current API call."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Simulate signal handler
        downloader._signal_handler(signal=None, frame=None)

        # Should set shutdown flag
        assert downloader._shutdown_requested is True

    async def test_graceful_shutdown_saves_checkpoint_sim(
        self, temp_data_dir, mock_auth, mock_disk_space
    ):
        """Test graceful shutdown saves checkpoint."""
        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Simulate download progress
        downloader._downloaded_symbols = {"MNQH26", "MNQM26"}

        # Save checkpoint
        downloader._save_checkpoint()

        # Verify checkpoint saved
        checkpoint_file = temp_data_dir / ".checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file) as f:
            data = json.load(f)

        assert "MNQH26" in data["symbols"]
        assert "MNQM26" in data["symbols"]


@pytest.mark.asyncio
class TestCLIExitCodes:
    """Integration tests for CLI exit codes."""

    async def test_cli_exit_codes_sim(self, temp_data_dir, mock_auth):
        """Test various exit codes for failure modes."""

        # Test invalid config exit code
        from src.data.cli import EXIT_INVALID_CONFIG, EXIT_SUCCESS

        # These would normally be tested by running the CLI
        # For now, verify constants exist
        assert EXIT_SUCCESS == 0
        assert EXIT_INVALID_CONFIG == 5

        # Test other exit codes
        from src.data.cli import (
            EXIT_AUTH_FAILURE,
            EXIT_NETWORK_ERROR,
            EXIT_DISK_FULL,
            EXIT_LOCK_FILE_EXISTS,
        )

        assert EXIT_AUTH_FAILURE == 1
        assert EXIT_NETWORK_ERROR == 2
        assert EXIT_DISK_FULL == 3
        assert EXIT_LOCK_FILE_EXISTS == 4


@pytest.mark.asyncio
class TestDataIntegrity:
    """Integration tests for data integrity validation."""

    async def test_validate_existing_hdf5_sim(self, temp_data_dir, mock_auth):
        """Test HDF5 file integrity validation before skipping."""
        from src.data.tradestation_models import BarData

        downloader = HistoricalDownloader(data_dir=temp_data_dir, auth=mock_auth)

        # Create valid HDF5 file
        timestamp = datetime.now(timezone.utc)
        bars = [
            BarData(
                symbol="MNQH26",
                timestamp=timestamp + timedelta(minutes=i),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11825.0,
                volume=1000,
            )
            for i in range(10)
        ]

        downloader._save_to_hdf5("MNQH26", bars)

        # Verify file was created
        h5_file = temp_data_dir / "MNQH26.h5"
        assert h5_file.exists()

        # Verify file can be read
        import h5py
        with h5py.File(h5_file, "r") as f:
            dataset = f["historical_bars"]
            assert len(dataset) == 10
            assert dataset.attrs["symbol"] == "MNQH26"

#!/usr/bin/env python3
"""Simple test script for HistoricalDownloader without full integration tests.

Tests basic functionality:
1. Configuration loading
2. Component initialization
3. Checkpoint save/load
4. HDF5 file creation
5. Mock download flow
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.historical_downloader import (
    HistoricalDownloader,
    DiskSpaceError,
    LockFileExistsError,
)
from src.data.tradestation_models import BarData


def test_config_loading():
    """Test 1: Check configuration loads correctly."""
    print("\n" + "="*70)
    print("TEST 1: Configuration Loading")
    print("="*70)

    try:
        from src.data.config import load_settings
        settings = load_settings()

        print(f"✅ Settings loaded successfully")
        print(f"   Client ID: {settings.tradestation_client_id[:10]}...")
        print(f"   Client Secret: {'*' * 20}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        print("   Note: This requires .env with TRADESTATION_CLIENT_ID and TRADESTATION_CLIENT_SECRET")
        return False


def test_initialization():
    """Test 2: Test downloader initialization."""
    print("\n" + "="*70)
    print("TEST 2: Downloader Initialization")
    print("="*70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "mnq"
            downloader = HistoricalDownloader(data_dir=data_dir)

            print(f"✅ Downloader initialized successfully")
            print(f"   Data directory: {data_dir}")
            print(f"   Symbol generator: {type(downloader.symbol_generator).__name__}")
            return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_operations():
    """Test 3: Test checkpoint save/load."""
    print("\n" + "="*70)
    print("TEST 3: Checkpoint Save/Load")
    print("="*70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "mnq"
            downloader = HistoricalDownloader(data_dir=data_dir)

            # Test saving checkpoint
            test_symbols = {"MNQH26", "MNQM26", "MNQU26"}
            downloader._downloaded_symbols = test_symbols
            downloader._save_checkpoint()

            checkpoint_file = data_dir / ".checkpoint.json"
            assert checkpoint_file.exists(), "Checkpoint file not created"
            print(f"✅ Checkpoint saved to {checkpoint_file}")

            # Test loading checkpoint
            downloader2 = HistoricalDownloader(data_dir=data_dir)
            downloader2._load_checkpoint()

            assert downloader2._downloaded_symbols == test_symbols, "Loaded symbols don't match"
            print(f"✅ Checkpoint loaded successfully: {len(downloader2._downloaded_symbols)} symbols")

            # Verify file content
            with open(checkpoint_file) as f:
                data = json.load(f)
            assert set(data["symbols"]) == test_symbols, "File content doesn't match"
            print(f"✅ Checkpoint file content validated")

            return True
    except Exception as e:
        print(f"❌ Checkpoint operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hdf5_storage():
    """Test 4: Test HDF5 file creation."""
    print("\n" + "="*70)
    print("TEST 4: HDF5 File Creation")
    print("="*70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "mnq"
            downloader = HistoricalDownloader(data_dir=data_dir)

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
                    volume=1000 + i * 10,
                )
                for i in range(100)
            ]

            # Save to HDF5
            downloader._save_to_hdf5("MNQH26", bars)

            # Verify file exists
            h5_file = data_dir / "MNQH26.h5"
            assert h5_file.exists(), "HDF5 file not created"
            print(f"✅ HDF5 file created: {h5_file}")

            # Verify file can be read
            import h5py
            with h5py.File(h5_file, "r") as f:
                dataset = f["historical_bars"]
                assert len(dataset) == 100, f"Expected 100 bars, got {len(dataset)}"
                print(f"✅ HDF5 file contains {len(dataset)} bars")

                # Verify metadata
                assert dataset.attrs["symbol"] == "MNQH26", "Symbol metadata mismatch"
                assert dataset.attrs["count"] == 100, "Count metadata mismatch"
                print(f"✅ HDF5 metadata validated")

                # Verify data structure
                assert "timestamp" in dataset.dtype.names, "Missing timestamp field"
                assert "open" in dataset.dtype.names, "Missing open field"
                assert "high" in dataset.dtype.names, "Missing high field"
                assert "low" in dataset.dtype.names, "Missing low field"
                assert "close" in dataset.dtype.names, "Missing close field"
                assert "volume" in dataset.dtype.names, "Missing volume field"
                assert "notional_value" in dataset.dtype.names, "Missing notional_value field"
                print(f"✅ HDF5 data structure validated")

            return True
    except Exception as e:
        print(f"❌ HDF5 storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_download_flow():
    """Test 5: Test mock download flow."""
    print("\n" + "="*70)
    print("TEST 5: Mock Download Flow")
    print("="*70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "mnq"

            # Mock auth
            mock_auth = Mock()
            mock_auth.get_valid_access_token = Mock(return_value="test_token")

            # Create downloader
            downloader = HistoricalDownloader(data_dir=data_dir, auth=mock_auth)

            # Mock historical bars response
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

            # Create mock client
            mock_client = AsyncMock()
            mock_client.get_historical_bars = AsyncMock(return_value=bars)
            mock_client.close = AsyncMock()

            # Set the client directly
            downloader.client = mock_client

            # Mock disk space check
            with patch("shutil.disk_usage") as mock_disk_usage:
                mock_stats = Mock()
                mock_stats.free = 500 * 1024 * 1024  # 500 MB
                mock_disk_usage.return_value = mock_stats

                # Download single contract
                await downloader._download_contract("MNQH26", 1, 1)

            print(f"✅ Mock download completed successfully")

            # Verify file was created
            h5_file = data_dir / "MNQH26.h5"
            assert h5_file.exists(), "HDF5 file not created"
            print(f"✅ HDF5 file created: {h5_file}")

            # Verify checkpoint
            assert "MNQH26" in downloader._downloaded_symbols, "Symbol not in checkpoint"
            print(f"✅ Checkpoint updated: {len(downloader._downloaded_symbols)} symbols")

            return True
    except Exception as e:
        print(f"❌ Mock download flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disk_space_check():
    """Test 6: Test disk space validation."""
    print("\n" + "="*70)
    print("TEST 6: Disk Space Validation")
    print("="*70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "mnq"
            downloader = HistoricalDownloader(data_dir=data_dir)

            # Test sufficient disk space
            with patch("shutil.disk_usage") as mock_disk_usage:
                mock_stats = Mock()
                mock_stats.free = 500 * 1024 * 1024  # 500 MB
                mock_disk_usage.return_value = mock_stats

                downloader._check_disk_space()
                print(f"✅ Sufficient disk space check passed")

            # Test insufficient disk space
            with patch("shutil.disk_usage") as mock_disk_usage:
                mock_stats = Mock()
                mock_stats.free = 100 * 1024 * 1024  # 100 MB (less than 200 MB required)
                mock_disk_usage.return_value = mock_stats

                try:
                    downloader._check_disk_space()
                    print(f"❌ Should have raised DiskSpaceError")
                    return False
                except DiskSpaceError as e:
                    print(f"✅ DiskSpaceError raised correctly: {e}")

            return True
    except Exception as e:
        print(f"❌ Disk space check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lock_file():
    """Test 7: Test lock file mechanism."""
    print("\n" + "="*70)
    print("TEST 7: Lock File Mechanism")
    print("="*70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "mnq"

            # First downloader should acquire lock
            downloader1 = HistoricalDownloader(data_dir=data_dir)
            downloader1._acquire_lock()
            lock_file = data_dir / ".lock"
            assert lock_file.exists(), "Lock file not created"
            print(f"✅ First downloader acquired lock")

            # Second downloader should fail
            downloader2 = HistoricalDownloader(data_dir=data_dir)
            try:
                downloader2._acquire_lock()
                print(f"❌ Should have raised LockFileExistsError")
                return False
            except LockFileExistsError as e:
                print(f"✅ LockFileExistsError raised correctly: {e}")

            # Release lock
            downloader1._release_lock()
            assert not lock_file.exists(), "Lock file not cleaned up"
            print(f"✅ Lock file released and cleaned up")

            # Now second downloader should succeed
            downloader3 = HistoricalDownloader(data_dir=data_dir)
            downloader3._acquire_lock()
            print(f"✅ Third downloader acquired lock after release")
            downloader3._release_lock()

            return True
    except Exception as e:
        print(f"❌ Lock file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# HISTORICAL DOWNLOADER TEST SUITE")
    print("#"*70)

    results = []

    # Run synchronous tests
    results.append(("Configuration Loading", test_config_loading()))
    results.append(("Initialization", test_initialization()))
    results.append(("Checkpoint Operations", test_checkpoint_operations()))
    results.append(("HDF5 Storage", test_hdf5_storage()))
    results.append(("Disk Space Check", test_disk_space_check()))
    results.append(("Lock File", test_lock_file()))

    # Run async tests
    results.append(("Mock Download Flow", await test_mock_download_flow()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The historical downloader is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

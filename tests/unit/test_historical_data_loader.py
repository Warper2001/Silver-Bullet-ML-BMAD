"""Unit tests for HistoricalDataLoader.

Tests loading of historical MNQ Dollar Bars from HDF5 storage for backtesting.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.research.historical_data_loader import HistoricalDataLoader


class TestHistoricalDataLoaderInit:
    """Test HistoricalDataLoader initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        loader = HistoricalDataLoader()

        assert loader._data_directory == Path("data/processed/dollar_bars")
        assert loader._min_completeness == 99.99
        assert loader._max_gap_minutes == 5

    def test_init_with_custom_data_directory(self):
        """Verify initialization with custom data directory."""
        loader = HistoricalDataLoader(data_directory="custom/data/path")

        assert loader._data_directory == Path("custom/data/path")

    def test_init_with_custom_completeness_threshold(self):
        """Verify initialization with custom completeness threshold."""
        loader = HistoricalDataLoader(min_completeness=95.0)

        assert loader._min_completeness == 95.0

    def test_init_with_custom_max_gap_minutes(self):
        """Verify initialization with custom max gap minutes."""
        loader = HistoricalDataLoader(max_gap_minutes=10)

        assert loader._max_gap_minutes == 10


class TestFileSelection:
    """Test HDF5 file selection for date ranges."""

    @patch('src.research.historical_data_loader.Path')
    def test_get_files_for_single_month(self, mock_path):
        """Verify file selection for single month range."""
        loader = HistoricalDataLoader()

        # Mock file exists
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file

        start_dt = pd.Timestamp("2024-03-01")
        end_dt = pd.Timestamp("2024-03-31")

        files = loader._get_files_for_date_range(start_dt, end_dt)

        assert len(files) == 1

    def test_get_files_for_multiple_months(self):
        """Verify file selection for multi-month range."""
        loader = HistoricalDataLoader()

        # Create actual temporary directory with mock files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            loader._data_directory = Path(tmpdir)

            # Create mock files
            (loader._data_directory / "MNQ_dollar_bars_202401.h5").touch()
            (loader._data_directory / "MNQ_dollar_bars_202402.h5").touch()
            (loader._data_directory / "MNQ_dollar_bars_202403.h5").touch()

            start_dt = pd.Timestamp("2024-01-15")
            end_dt = pd.Timestamp("2024-03-15")

            files = loader._get_files_for_date_range(start_dt, end_dt)

            # Should include Jan, Feb, Mar files
            assert len(files) == 3

    def test_get_files_handles_year_boundary(self):
        """Verify file selection handles year boundary."""
        loader = HistoricalDataLoader()

        # Create actual temporary directory with mock files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            loader._data_directory = Path(tmpdir)

            # Create mock files
            (loader._data_directory / "MNQ_dollar_bars_202312.h5").touch()
            (loader._data_directory / "MNQ_dollar_bars_202401.h5").touch()

            start_dt = pd.Timestamp("2023-12-01")
            end_dt = pd.Timestamp("2024-01-31")

            files = loader._get_files_for_date_range(start_dt, end_dt)

            # Should include Dec 2023 and Jan 2024
            assert len(files) == 2

    def test_get_files_skips_missing_files(self):
        """Verify file selection skips missing HDF5 files."""
        loader = HistoricalDataLoader()

        # Create actual temporary directory with partial files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            loader._data_directory = Path(tmpdir)

            # Create only Jan and Mar files (Feb missing)
            (loader._data_directory / "MNQ_dollar_bars_202401.h5").touch()
            # Skip Feb
            (loader._data_directory / "MNQ_dollar_bars_202403.h5").touch()

            start_dt = pd.Timestamp("2024-01-01")
            end_dt = pd.Timestamp("2024-03-31")

            files = loader._get_files_for_date_range(start_dt, end_dt)

            # Should skip February (missing)
            assert len(files) == 2


class TestDataLoading:
    """Test data loading from HDF5 files."""

    @patch('src.research.historical_data_loader.h5py.File')
    def test_load_single_file_returns_dataframe(self, mock_h5py):
        """Verify loading single file returns DataFrame."""
        loader = HistoricalDataLoader()

        # Mock HDF5 file structure with proper column names
        mock_data = np.array([
            [1704067200000, 2100.0, 2105.0, 2098.0, 2102.0, 1000, 2103000.0],
            [1704067205000, 2102.0, 2108.0, 2100.0, 2106.0, 1200, 2527200.0]
        ], dtype=object)

        mock_file = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_data)
        mock_file.__getitem__ = MagicMock(return_value=mock_dataset)

        mock_h5py.return_value.__enter__.return_value = mock_file

        file_path = Path("test.h5")
        df = loader._load_single_file(file_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert df['timestamp'].dtype.name == 'datetime64[ns]'

    @patch('src.research.historical_data_loader.h5py.File')
    def test_load_data_filters_by_date_range(self, mock_h5py):
        """Verify load_data filters by date range."""
        loader = HistoricalDataLoader()

        # Mock data with timestamps spanning multiple days
        timestamps = pd.date_range("2024-03-01", periods=100, freq="5s")
        mock_data = np.column_stack([
            timestamps.astype(np.int64),
            np.full(100, 2100.0),
            np.full(100, 2105.0),
            np.full(100, 2098.0),
            np.full(100, 2102.0),
            np.full(100, 1000),
            np.full(100, 2100000.0)
        ])

        mock_file = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_data)
        mock_file.__getitem__ = MagicMock(return_value=mock_dataset)

        mock_h5py.return_value.__enter__.return_value = mock_file

        with patch.object(loader, '_get_files_for_date_range') as mock_files:
            mock_files.return_value = [Path("test.h5")]

            with patch.object(loader, '_load_single_file') as mock_load:
                mock_df = pd.DataFrame(mock_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'dollar_volume'
                ])
                mock_load.return_value = mock_df

                with patch.object(loader, '_validate_data_completeness'):
                    with patch.object(loader, '_detect_and_report_gaps'):
                        with patch.object(loader, '_log_data_statistics'):
                            df = loader.load_data(
                                "2024-03-01", "2024-03-02"
                            )

                            assert df.index.min() >= pd.Timestamp(
                                "2024-03-01"
                            )
                            assert df.index.max() <= pd.Timestamp(
                                "2024-03-02"
                            )

    @patch('src.research.historical_data_loader.h5py.File')
    def test_load_data_concatenates_multiple_files(self, mock_h5py):
        """Verify load_data concatenates data from multiple files."""
        loader = HistoricalDataLoader()

        # Mock two files with different data
        timestamps1 = pd.date_range("2024-03-01", periods=50, freq="5s")
        timestamps2 = pd.date_range("2024-03-02", periods=50, freq="5s")

        mock_file = MagicMock()
        mock_dataset = MagicMock()

        mock_h5py.return_value.__enter__.return_value = mock_file
        mock_file.__getitem__ = MagicMock(return_value=mock_dataset)

        with patch.object(loader, '_get_files_for_date_range') as mock_files:
            mock_files.return_value = [
                Path("file1.h5"),
                Path("file2.h5")
            ]

            call_count = [0]

            def mock_load_side(file_path):
                count = call_count[0]
                call_count[0] += 1

                if count == 0:
                    timestamps = timestamps1
                else:
                    timestamps = timestamps2

                mock_data = np.column_stack([
                    timestamps.astype(np.int64),
                    np.full(50, 2100.0),
                    np.full(50, 2105.0),
                    np.full(50, 2098.0),
                    np.full(50, 2102.0),
                    np.full(50, 1000),
                    np.full(50, 2100000.0)
                ])

                df = pd.DataFrame(mock_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'dollar_volume'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df

            with patch.object(loader, '_load_single_file', side_effect=mock_load_side):
                with patch.object(loader, '_validate_data_completeness'):
                    with patch.object(loader, '_detect_and_report_gaps'):
                        with patch.object(loader, '_log_data_statistics'):
                            df = loader.load_data("2024-03-01", "2024-03-03")

                            assert len(df) == 100  # 50 + 50

    @patch('src.research.historical_data_loader.h5py.File')
    def test_load_data_sorts_by_timestamp(self, mock_h5py):
        """Verify load_data sorts data by timestamp."""
        loader = HistoricalDataLoader()

        # Create unsorted timestamps
        timestamps = pd.to_datetime([
            "2024-03-01 10:00:00",
            "2024-03-01 09:00:00",
            "2024-03-01 11:00:00"
        ])

        mock_data = np.column_stack([
            timestamps.astype(np.int64),
            np.full(3, 2100.0),
            np.full(3, 2105.0),
            np.full(3, 2098.0),
            np.full(3, 2102.0),
            np.full(3, 1000),
            np.full(3, 2100000.0)
        ])

        mock_file = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_data)
        mock_file.__getitem__ = MagicMock(return_value=mock_dataset)

        mock_h5py.return_value.__enter__.return_value = mock_file

        with patch.object(loader, '_get_files_for_date_range') as mock_files:
            mock_files.return_value = [Path("test.h5")]

            with patch.object(loader, '_load_single_file') as mock_load:
                mock_df = pd.DataFrame(mock_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'dollar_volume'
                ])
                mock_df['timestamp'] = pd.to_datetime(mock_df['timestamp'])
                mock_load.return_value = mock_df

                with patch.object(loader, '_validate_data_completeness'):
                    with patch.object(loader, '_detect_and_report_gaps'):
                        with patch.object(loader, '_log_data_statistics'):
                            df = loader.load_data("2024-03-01", "2024-03-02")

                            # Verify sorted
                            assert df.index.is_monotonic_increasing


class TestDataValidation:
    """Test data completeness validation."""

    def test_validate_completeness_passes_for_99_99_percent(self):
        """Verify validation passes for 99.99% completeness."""
        loader = HistoricalDataLoader(min_completeness=99.99)

        # Create DataFrame with 99.99% completeness
        start_dt = pd.Timestamp("2024-03-01")
        end_dt = pd.Timestamp("2024-03-02")

        # Expected bars: 2 days * 24 hours * 720 bars/hour = 34,560
        # 99.99% = 34,556 bars
        timestamps = pd.date_range(start_dt, periods=34556, freq="5s")
        df = pd.DataFrame({"close": np.full(34556, 2100.0)}, index=timestamps)

        # Should not raise
        loader._validate_data_completeness(df, start_dt, end_dt)

    def test_validate_completeness_raises_error_below_threshold(self):
        """Verify validation raises error for < 99.99% completeness."""
        loader = HistoricalDataLoader(min_completeness=99.99)

        # Use specific time range for accurate expected bars
        start_dt = pd.Timestamp("2024-03-01 00:00:00")
        end_dt = pd.Timestamp("2024-03-01 23:59:59")  # 1 day = 86,399 seconds

        # Expected bars: 86,399 / 5 = 17,279.8 = 17,279 bars
        # 90% of 17,279 = 15,551 bars
        timestamps = pd.date_range(start_dt, periods=15551, freq="5s")
        df = pd.DataFrame({"close": np.full(15551, 2100.0)}, index=timestamps)

        with pytest.raises(ValueError) as exc_info:
            loader._validate_data_completeness(df, start_dt, end_dt)

        assert "completeness" in str(exc_info.value).lower()

    def test_validate_completeness_calculates_expected_bars(self):
        """Verify validation calculates expected bars correctly."""
        loader = HistoricalDataLoader(min_completeness=99.99)

        start_dt = pd.Timestamp("2024-03-01 00:00:00")
        end_dt = pd.Timestamp("2024-03-01 00:10:00")  # 10 minutes

        # Expected: 10 minutes * 60 seconds / 5 seconds = 120 bars
        timestamps = pd.date_range(start_dt, periods=120, freq="5s")
        df = pd.DataFrame({"close": np.full(120, 2100.0)}, index=timestamps)

        # Should not raise (100% completeness)
        loader._validate_data_completeness(df, start_dt, end_dt)

    def test_validate_completeness_handles_exact_minimum(self):
        """Verify validation handles exact minimum threshold."""
        loader = HistoricalDataLoader(min_completeness=99.0)

        start_dt = pd.Timestamp("2024-03-01")
        end_dt = pd.Timestamp("2024-03-02")

        # Create DataFrame with exactly 99% completeness
        timestamps = pd.date_range(start_dt, periods=34214, freq="5s")
        df = pd.DataFrame({"close": np.full(34214, 2100.0)}, index=timestamps)

        # Should not raise (exactly 99%)
        loader._validate_data_completeness(df, start_dt, end_dt)


class TestGapDetection:
    """Test gap detection in time series data."""

    def test_detect_gaps_finds_gaps_over_threshold(self):
        """Verify gap detection finds gaps > 5 minutes."""
        loader = HistoricalDataLoader(max_gap_minutes=5)

        # Create timestamps with a gap > 5 minutes
        # The gap is from 10:00:05 to 10:15:00 = 14.916... minutes
        timestamps = pd.to_datetime([
            "2024-03-01 10:00:00",
            "2024-03-01 10:00:05",
            "2024-03-01 10:15:00",  # Gap from previous
            "2024-03-01 10:15:05"
        ])

        df = pd.DataFrame({"close": [2100, 2101, 2102, 2103]}, index=timestamps)

        gaps = loader._detect_and_report_gaps(df)

        assert len(gaps) == 1
        assert gaps[0]["duration_minutes"] > 5

    def test_detect_gaps_ignores_gaps_under_threshold(self):
        """Verify gap detection ignores gaps ≤ 5 minutes."""
        loader = HistoricalDataLoader(max_gap_minutes=5)

        # Create timestamps with 3-minute gap
        timestamps = pd.to_datetime([
            "2024-03-01 10:00:00",
            "2024-03-01 10:03:05",  # 3-minute gap
            "2024-03-01 10:03:10"
        ])

        df = pd.DataFrame({"close": [2100, 2101, 2102]}, index=timestamps)

        gaps = loader._detect_and_report_gaps(df)

        assert len(gaps) == 0

    def test_detect_gaps_returns_gap_information(self):
        """Verify gap detection returns detailed gap info."""
        loader = HistoricalDataLoader(max_gap_minutes=5)

        timestamps = pd.to_datetime([
            "2024-03-01 10:00:00",
            "2024-03-01 10:20:00",  # 20-minute gap
            "2024-03-01 10:20:05"
        ])

        df = pd.DataFrame({"close": [2100, 2101, 2102]}, index=timestamps)

        gaps = loader._detect_and_report_gaps(df)

        assert len(gaps) == 1
        assert "start" in gaps[0]
        assert "end" in gaps[0]
        assert "duration_minutes" in gaps[0]
        assert gaps[0]["duration_minutes"] > 5

    def test_detect_gaps_logs_warnings(self):
        """Verify gap detection logs warnings."""
        loader = HistoricalDataLoader(max_gap_minutes=5)

        timestamps = pd.to_datetime([
            "2024-03-01 10:00:00",
            "2024-03-01 10:15:00",  # 15-minute gap
            "2024-03-01 10:15:05"
        ])

        df = pd.DataFrame({"close": [2100, 2101, 2102]}, index=timestamps)

        # Patch the module logger
        with patch('src.research.historical_data_loader.logger') as mock_logger:
            loader._detect_and_report_gaps(df)

            # Verify warning logged
            assert mock_logger.warning.called


class TestStatistics:
    """Test statistics calculation and logging."""

    @patch('src.research.historical_data_loader.logging.getLogger')
    def test_log_statistics_calculates_all_metrics(self, mock_logger):
        """Verify statistics calculation includes all metrics."""
        loader = HistoricalDataLoader()

        start_dt = pd.Timestamp("2024-03-01 00:00:00")
        end_dt = pd.Timestamp("2024-03-02 00:00:00")

        # Create 1 day of data (86,400 seconds / 5 = 17,280 bars)
        timestamps = pd.date_range(start_dt, periods=17280, freq="5s")
        df = pd.DataFrame({"close": np.full(17280, 2100.0)}, index=timestamps)

        stats = loader._log_data_statistics(df, start_dt, end_dt)

        assert "date_range" in stats
        assert "total_bars" in stats
        assert "expected_bars" in stats
        assert "missing_bars" in stats
        assert "completeness_percent" in stats
        assert "time_range_days" in stats

    def test_log_statistics_logs_correctly(self):
        """Verify statistics are logged."""
        loader = HistoricalDataLoader()

        start_dt = pd.Timestamp("2024-03-01")
        end_dt = pd.Timestamp("2024-03-02")

        timestamps = pd.date_range(start_dt, periods=10000, freq="5s")
        df = pd.DataFrame({"close": np.full(10000, 2100.0)}, index=timestamps)

        # Patch the module logger
        with patch('src.research.historical_data_loader.logger') as mock_logger:
            loader._log_data_statistics(df, start_dt, end_dt)

            # Verify info logged
            assert mock_logger.info.called

    def test_log_statistics_returns_dict(self):
        """Verify statistics returns dictionary."""
        loader = HistoricalDataLoader()

        start_dt = pd.Timestamp("2024-03-01")
        end_dt = pd.Timestamp("2024-03-02")

        timestamps = pd.date_range(start_dt, periods=5000, freq="5s")
        df = pd.DataFrame({"close": np.full(5000, 2100.0)}, index=timestamps)

        stats = loader._log_data_statistics(df, start_dt, end_dt)

        assert isinstance(stats, dict)

    def test_log_statistics_handles_single_day(self):
        """Verify statistics handles single day range."""
        loader = HistoricalDataLoader()

        start_dt = pd.Timestamp("2024-03-01 00:00:00")
        end_dt = pd.Timestamp("2024-03-01 23:59:59")

        timestamps = pd.date_range(start_dt, periods=17278, freq="5s")
        df = pd.DataFrame({"close": np.full(17278, 2100.0)}, index=timestamps)

        stats = loader._log_data_statistics(df, start_dt, end_dt)

        # Should be at least 0 days, maybe 0 or 1 depending on calculation
        assert stats["time_range_days"] >= 0

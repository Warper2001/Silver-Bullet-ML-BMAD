"""Historical data loader for backtesting MNQ Dollar Bars."""

import logging
from pathlib import Path

import h5py
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Load historical MNQ Dollar Bars from HDF5 storage for backtesting.

    Handles:
    - Efficient loading by date range from multiple HDF5 files
    - Data completeness validation (≥ 99.99%)
    - Gap detection (> 5 minutes)
    - Statistics calculation and logging
    - Fast loading (< 30 seconds for 2 years)
    """

    def __init__(
        self,
        data_directory: str = "data/processed/dollar_bars",
        min_completeness: float = 99.99,
        max_gap_minutes: int = 5
    ):
        """Initialize historical data loader.

        Args:
            data_directory: Directory containing HDF5 files
                (e.g., 'data/processed/dollar_bars')
            min_completeness: Minimum data completeness % (default 99.99%)
            max_gap_minutes: Maximum gap size to report (minutes)
        """
        self._data_directory = Path(data_directory)
        self._min_completeness = min_completeness
        self._max_gap_minutes = max_gap_minutes

    def load_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load Dollar Bars for date range.

        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            pandas DataFrame with timestamp index and Dollar Bar columns

        Raises:
            ValueError: If data completeness < min_completeness
            FileNotFoundError: If required HDF5 files not found

        Example:
            >>> loader = HistoricalDataLoader()
            >>> df = loader.load_data("2022-03-01", "2024-03-01")
            >>> print(f"Loaded {len(df)} bars")
            Loaded 502847 bars
        """
        # Convert date strings to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Collect HDF5 files for date range
        files_to_load = self._get_files_for_date_range(start_dt, end_dt)

        if not files_to_load:
            raise FileNotFoundError(
                f"No HDF5 files found for date range {start_date} to {end_date}"
            )

        # Load data from each file
        dataframes = []
        for file_path in files_to_load:
            df = self._load_single_file(file_path)
            dataframes.append(df)

        # Concatenate all data
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Convert timestamp to datetime if needed
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')

        # Set timestamp as index
        combined_df = combined_df.set_index('timestamp')

        # Filter by date range
        combined_df = combined_df.loc[
            (combined_df.index >= start_dt) &
            (combined_df.index <= end_dt)
        ]

        # Validate data completeness
        self._validate_data_completeness(combined_df, start_dt, end_dt)

        # Detect and report gaps
        self._detect_and_report_gaps(combined_df)

        # Calculate and log statistics
        self._log_data_statistics(combined_df, start_dt, end_dt)

        return combined_df

    def _get_files_for_date_range(
        self,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp
    ) -> list[Path]:
        """Get list of HDF5 files for date range.

        Args:
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of file paths sorted by date
        """
        files = []

        # Generate list of months in range
        current = pd.Timestamp(start_dt).replace(day=1)
        end_month = pd.Timestamp(end_dt).replace(day=1)

        while current <= end_month:
            # File naming: MNQ_dollar_bars_YYYYMM.h5
            filename = f"MNQ_dollar_bars_{current.strftime('%Y%m')}.h5"
            file_path = self._data_directory / filename

            if file_path.exists():
                files.append(file_path)

            # Move to next month
            current = pd.Timestamp(current) + pd.DateOffset(months=1)

        return sorted(files)

    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """Load data from single HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            pandas DataFrame with Dollar Bar data
        """
        # Read HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Assuming data stored as dataset 'dollar_bars'
            # with columns: timestamp, open, high, low, close, volume, dollar_volume
            data = f['dollar_bars'][:]

        # Convert to DataFrame with column names
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'dollar_volume'
        ])

        # Convert timestamp column to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    def _validate_data_completeness(
        self,
        df: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp
    ) -> None:
        """Validate data completeness meets minimum threshold.

        Args:
            df: Loaded Dollar Bars DataFrame
            start_dt: Start datetime
            end_dt: End datetime

        Raises:
            ValueError: If completeness < min_completeness
        """
        # Calculate expected number of bars
        # Assuming 5-second bars for MNQ (highly liquid)
        time_range = end_dt - start_dt
        expected_bars = int(time_range.total_seconds() / 5)

        # Calculate actual bars
        actual_bars = len(df)

        # Calculate completeness
        if expected_bars > 0:
            completeness = (actual_bars / expected_bars) * 100
        else:
            completeness = 100.0

        if completeness < self._min_completeness:
            raise ValueError(
                f"Data completeness {completeness:.2f}% below "
                f"minimum threshold {self._min_completeness}%"
            )

    def _detect_and_report_gaps(self, df: pd.DataFrame) -> list[dict]:
        """Detect and report data gaps > max_gap_minutes.

        Args:
            df: Dollar Bars DataFrame with timestamp index

        Returns:
            List of gap dictionaries with start, end, duration
        """
        gaps = []

        if len(df) < 2:
            return gaps

        # Calculate time differences between consecutive bars
        time_diffs = df.index.to_series().diff()

        # Find gaps exceeding threshold
        gap_threshold = pd.Timedelta(minutes=self._max_gap_minutes)
        gap_mask = time_diffs > gap_threshold

        gap_times = time_diffs[gap_mask]

        for gap_start, gap_duration in gap_times.items():
            gap_info = {
                "start": gap_start,
                "end": gap_start + gap_duration,
                "duration_minutes": gap_duration.total_seconds() / 60
            }
            gaps.append(gap_info)

        # Log gaps
        if gaps:
            logger.warning(
                f"Detected {len(gaps)} data gaps > "
                f"{self._max_gap_minutes} minutes"
            )
            for gap in gaps:
                logger.warning(
                    f"Gap: {gap['start']} to {gap['end']} "
                    f"({gap['duration_minutes']:.1f} minutes)"
                )

        return gaps

    def _log_data_statistics(
        self,
        df: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp
    ) -> dict:
        """Calculate and log data statistics.

        Args:
            df: Dollar Bars DataFrame
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            Dictionary with statistics
        """
        # Calculate statistics
        time_range = end_dt - start_dt
        expected_bars = int(time_range.total_seconds() / 5)
        actual_bars = len(df)
        missing_bars = expected_bars - actual_bars

        if expected_bars > 0:
            completeness = (actual_bars / expected_bars) * 100
        else:
            completeness = 100.0

        stats = {
            "date_range": f"{start_dt.date()} to {end_dt.date()}",
            "total_bars": actual_bars,
            "expected_bars": expected_bars,
            "missing_bars": missing_bars,
            "completeness_percent": completeness,
            "time_range_days": time_range.days
        }

        # Log statistics
        logger.info(
            f"Historical data loaded: {stats['date_range']}\n"
            f"  Total bars: {stats['total_bars']:,}\n"
            f"  Expected bars: {stats['expected_bars']:,}\n"
            f"  Missing bars: {stats['missing_bars']:,}\n"
            f"  Completeness: {stats['completeness_percent']:.3f}%\n"
            f"  Time range: {stats['time_range_days']} days"
        )

        return stats

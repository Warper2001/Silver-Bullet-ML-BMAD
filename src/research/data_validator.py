"""Data validation for MNQ historical dollar bar data.

This module provides comprehensive data validation for MNQ futures historical data,
including data period verification, completeness checks, gap detection, and dollar
bar validation.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import h5py
import pandas as pd
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class GapCategory(str, Enum):
    """Category of data gap."""

    WEEKEND_HOLIDAY = "weekend_holiday"
    PROBLEMATIC = "problematic"


class GapInfo(BaseModel):
    """Information about a detected data gap."""

    start_timestamp: datetime
    end_timestamp: datetime
    duration_hours: float
    category: GapCategory


class DataPeriodValidation(BaseModel):
    """Result of data period validation."""

    is_sufficient: bool
    start_date: datetime
    end_date: datetime
    total_days: int
    expected_min_days: int
    errors: list[str]


class DataQualityValidation(BaseModel):
    """Result of data completeness validation."""

    passed: bool
    completeness_percent: float
    actual_bars: int
    expected_bars: int
    missing_bars: int
    warnings: list[str]


class DollarBarValidation(BaseModel):
    """Result of dollar bar validation."""

    exists: bool
    bar_count: int
    avg_bars_per_day: float
    threshold: float
    threshold_compliant: bool
    errors: list[str]
    warnings: list[str] = []


class DataValidator:
    """Comprehensive data validator for MNQ historical data.

    Validates:
    - Data period coverage (minimum 2 years)
    - Data completeness (>99.99% target)
    - Gap detection and categorization
    - Dollar bar availability and threshold compliance

    Example:
        >>> validator = DataValidator(
        ...     hdf5_path="data/processed/dollar_bars/MNQ_dollar_bars_202401.h5",
        ...     min_completeness=99.99,
        ...     dollar_bar_threshold=50_000_000
        ... )
        >>> period_result = validator.validate_data_period()
        >>> quality_result = validator.check_completeness()
        >>> gaps = validator.detect_gaps()
        >>> dollar_result = validator.validate_dollar_bars()
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        min_completeness: float = 99.99,
        dollar_bar_threshold: float = 50_000_000,
    ):
        """Initialize DataValidator.

        Args:
            hdf5_path: Path to HDF5 file containing dollar bars
            min_completeness: Minimum completeness percentage (default 99.99%)
            dollar_bar_threshold: Expected dollar bar notional value threshold
        """
        self.hdf5_path = Path(hdf5_path)
        self.min_completeness = min_completeness
        self.dollar_bar_threshold = dollar_bar_threshold
        self._data: pd.DataFrame | None = None

        logger.info(
            f"Initialized DataValidator for {self.hdf5_path}\n"
            f"  Min completeness: {self.min_completeness}%\n"
            f"  Dollar bar threshold: ${self.dollar_bar_threshold:,.0f}"
        )

    def _load_data(self) -> pd.DataFrame:
        """Load dollar bar data from HDF5 file.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, notional_value

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            OSError: If file is corrupted or unreadable
        """
        if self._data is not None:
            return self._data

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'dollar_bars' not in f:
                    raise KeyError("Dataset 'dollar_bars' not found in HDF5 file")

                data = f['dollar_bars'][:]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'notional_value'
            ])

            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            self._data = df
            logger.info(f"Loaded {len(df)} dollar bars from {self.hdf5_path}")

            return df

        except OSError as e:
            logger.error(f"Failed to read HDF5 file: {e}")
            raise

    def validate_data_period(self) -> DataPeriodValidation:
        """Validate data period covers at least 2 years.

        Returns:
            DataPeriodValidation with validation results

        Example:
            >>> result = validator.validate_data_period()
            >>> if result.is_sufficient:
            ...     print(f"Data covers {result.total_days} days")
        """
        logger.info("Validating data period coverage...")

        errors = []

        try:
            df = self._load_data()

            if len(df) == 0:
                errors.append("No data found in HDF5 file")
                return DataPeriodValidation(
                    is_sufficient=False,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    total_days=0,
                    expected_min_days=730,
                    errors=errors
                )

            # Get date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            total_days = (end_date - start_date).days

            # Expected minimum: 2 years (730 days)
            expected_min_days = 730
            is_sufficient = total_days >= expected_min_days

            if not is_sufficient:
                errors.append(
                    f"Insufficient data period: {total_days} days found, "
                    f"{expected_min_days} days required (2 years)"
                )

            logger.info(
                f"Data period validation: {'PASS' if is_sufficient else 'FAIL'}\n"
                f"  Start: {start_date}\n"
                f"  End: {end_date}\n"
                f"  Total days: {total_days}\n"
                f"  Required: {expected_min_days} days"
            )

            return DataPeriodValidation(
                is_sufficient=is_sufficient,
                start_date=start_date,
                end_date=end_date,
                total_days=total_days,
                expected_min_days=expected_min_days,
                errors=errors
            )

        except FileNotFoundError as e:
            logger.error(f"Data period validation failed: {e}")
            errors.append(f"File not found: {e}")
            return DataPeriodValidation(
                is_sufficient=False,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_days=0,
                expected_min_days=730,
                errors=errors
            )
        except Exception as e:
            logger.error(f"Unexpected error during data period validation: {e}")
            errors.append(f"Validation error: {e}")
            return DataPeriodValidation(
                is_sufficient=False,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_days=0,
                expected_min_days=730,
                errors=errors
            )

    def check_completeness(self) -> DataQualityValidation:
        """Check data completeness meets minimum threshold.

        Calculates completeness as (actual_bars / expected_bars) × 100,
        where expected_bars is based on unique dates in the data.

        For dollar bars, completeness measures whether we have data for
        all expected trading days in the period, not the total number of bars
        (which varies based on market activity).

        Returns:
            DataQualityValidation with completeness results

        Example:
            >>> result = validator.check_completeness()
            >>> print(f"Completeness: {result.completeness_percent:.2f}%")
        """
        logger.info("Checking data completeness...")

        warnings = []

        try:
            df = self._load_data()

            if len(df) == 0:
                return DataQualityValidation(
                    passed=False,
                    completeness_percent=0.0,
                    actual_bars=0,
                    expected_bars=0,
                    missing_bars=0,
                    warnings=["No data found"]
                )

            # Get date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            total_days = (end_date - start_date).days

            # Count unique dates in the data
            df_dates = df['timestamp'].dt.date
            actual_unique_dates = df_dates.nunique()

            # Calculate expected trading days in the period
            # MNQ futures trade Monday-Friday, excluding holidays
            # Approximately 252 trading days per year
            calendar_days = (end_date - start_date).days + 1
            weeks = calendar_days / 7
            expected_trading_days = int(weeks * 5)  # 5 trading days per week

            # For dollar bars, we check if we have at least some data for each trading day
            # This is more meaningful than counting total bars (which varies)
            actual_bars = len(df)
            expected_bars = expected_trading_days

            # Calculate completeness based on trading day coverage
            if expected_trading_days > 0:
                completeness_percent = (actual_unique_dates / expected_trading_days) * 100
            else:
                completeness_percent = 100.0

            missing_bars = max(0, expected_trading_days - actual_unique_dates)
            passed = completeness_percent >= self.min_completeness

            if not passed:
                warnings.append(
                    f"Completeness {completeness_percent:.2f}% below "
                    f"threshold {self.min_completeness}%"
                )

            logger.info(
                f"Completeness check: {'PASS' if passed else 'FAIL'}\n"
                f"  Unique dates with data: {actual_unique_dates}\n"
                f"  Expected trading days: {expected_trading_days}\n"
                f"  Missing trading days: {missing_bars}\n"
                f"  Total bars: {actual_bars}\n"
                f"  Completeness: {completeness_percent:.3f}%\n"
                f"  Threshold: {self.min_completeness}%"
            )

            return DataQualityValidation(
                passed=passed,
                completeness_percent=completeness_percent,
                actual_bars=actual_unique_dates,
                expected_bars=expected_trading_days,
                missing_bars=missing_bars,
                warnings=warnings
            )

        except FileNotFoundError as e:
            logger.error(f"Completeness check failed: {e}")
            return DataQualityValidation(
                passed=False,
                completeness_percent=0.0,
                actual_bars=0,
                expected_bars=0,
                missing_bars=0,
                warnings=[f"File not found: {e}"]
            )
        except Exception as e:
            logger.error(f"Unexpected error during completeness check: {e}")
            return DataQualityValidation(
                passed=False,
                completeness_percent=0.0,
                actual_bars=0,
                expected_bars=0,
                missing_bars=0,
                warnings=[f"Check error: {e}"]
            )

    def detect_gaps(self) -> list[GapInfo]:
        """Detect temporal gaps in data.

        Identifies gaps between consecutive bars and categorizes them:
        - Weekend/holiday gaps: 1-2 days (acceptable)
        - Problematic gaps: >3 days (missing data)

        Returns:
            List of GapInfo objects with gap details

        Example:
            >>> gaps = validator.detect_gaps()
            >>> for gap in gaps:
            ...     print(f"Gap: {gap.start_timestamp} to {gap.end_timestamp}")
        """
        logger.info("Detecting data gaps...")

        try:
            df = self._load_data()

            if len(df) < 2:
                logger.warning("Not enough data to detect gaps")
                return []

            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')

            # Calculate time differences
            time_diffs = df_sorted['timestamp'].diff()

            # Find gaps: differences > 1 hour (assuming at least hourly bars)
            # For daily bars, look for gaps > 1 day
            gap_threshold = timedelta(hours=24)  # 1 day

            gaps = []
            for idx, time_diff in time_diffs.items():
                if time_diff > gap_threshold:
                    gap_start = df_sorted.loc[idx - 1, 'timestamp']
                    gap_end = df_sorted.loc[idx, 'timestamp']
                    duration_hours = time_diff.total_seconds() / 3600

                    # Categorize gap
                    # 1-2 days (24-72 hours) = weekend/holiday (acceptable)
                    # >3 days (>72 hours) = problematic (missing data)
                    if duration_hours <= 72:
                        category = GapCategory.WEEKEND_HOLIDAY
                    else:
                        category = GapCategory.PROBLEMATIC

                    gap_info = GapInfo(
                        start_timestamp=gap_start,
                        end_timestamp=gap_end,
                        duration_hours=duration_hours,
                        category=category
                    )
                    gaps.append(gap_info)

            # Log summary
            problematic_count = sum(1 for g in gaps if g.category == GapCategory.PROBLEMATIC)
            logger.info(
                f"Gap detection complete:\n"
                f"  Total gaps: {len(gaps)}\n"
                f"  Problematic gaps: {problematic_count}\n"
                f"  Weekend/holiday gaps: {len(gaps) - problematic_count}"
            )

            return gaps

        except FileNotFoundError as e:
            logger.error(f"Gap detection failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during gap detection: {e}")
            return []

    def validate_dollar_bars(self) -> DollarBarValidation:
        """Validate dollar bar availability and threshold compliance.

        Checks:
        - Dollar bar file exists and is readable
        - Notional value threshold matches expected ($50M default)
        - Calculates bar statistics

        Returns:
            DollarBarValidation with validation results

        Example:
            >>> result = validator.validate_dollar_bars()
            >>> if result.exists:
            ...     print(f"Bars: {result.bar_count}, Threshold: ${result.threshold}")
        """
        logger.info("Validating dollar bars...")

        errors = []
        warnings = []

        try:
            # Check file exists
            if not self.hdf5_path.exists():
                logger.error(f"Dollar bar file not found: {self.hdf5_path}")
                errors.append(f"Dollar bar file not found: {self.hdf5_path}")
                return DollarBarValidation(
                    exists=False,
                    bar_count=0,
                    avg_bars_per_day=0.0,
                    threshold=self.dollar_bar_threshold,
                    threshold_compliant=False,
                    errors=errors,
                    warnings=warnings
                )

            # Load data
            df = self._load_data()

            if len(df) == 0:
                errors.append("Dollar bar file contains no data")
                return DollarBarValidation(
                    exists=True,
                    bar_count=0,
                    avg_bars_per_day=0.0,
                    threshold=self.dollar_bar_threshold,
                    threshold_compliant=False,
                    errors=errors,
                    warnings=warnings
                )

            # Check threshold compliance
            notional_values = df['notional_value']
            avg_notional = notional_values.mean()

            # Check if most bars use the expected threshold
            # Allow 10% tolerance for rounding/variations
            tolerance = self.dollar_bar_threshold * 0.10
            threshold_compliant = abs(avg_notional - self.dollar_bar_threshold) <= tolerance

            if not threshold_compliant:
                warnings.append(
                    f"Dollar bar threshold mismatch: "
                    f"expected ${self.dollar_bar_threshold:,.0f}, "
                    f"found ${avg_notional:,.0f} (avg)"
                )

            # Calculate statistics
            bar_count = len(df)

            # Calculate bars per day
            if len(df) > 0:
                start_date = df['timestamp'].min()
                end_date = df['timestamp'].max()
                total_days = max(1, (end_date - start_date).days)
                avg_bars_per_day = bar_count / total_days
            else:
                avg_bars_per_day = 0.0

            logger.info(
                f"Dollar bar validation: {'PASS' if threshold_compliant else 'WARNING'}\n"
                f"  Total bars: {bar_count}\n"
                f"  Avg bars/day: {avg_bars_per_day:.1f}\n"
                f"  Threshold: ${self.dollar_bar_threshold:,.0f}\n"
                f"  Avg notional: ${avg_notional:,.0f}\n"
                f"  Threshold compliant: {threshold_compliant}"
            )

            return DollarBarValidation(
                exists=True,
                bar_count=bar_count,
                avg_bars_per_day=avg_bars_per_day,
                threshold=self.dollar_bar_threshold,
                threshold_compliant=threshold_compliant,
                errors=errors,
                warnings=warnings
            )

        except FileNotFoundError as e:
            logger.error(f"Dollar bar validation failed: {e}")
            errors.append(f"File not found: {e}")
            return DollarBarValidation(
                exists=False,
                bar_count=0,
                avg_bars_per_day=0.0,
                threshold=self.dollar_bar_threshold,
                threshold_compliant=False,
                errors=errors,
                warnings=warnings
            )
        except Exception as e:
            logger.error(f"Unexpected error during dollar bar validation: {e}")
            errors.append(f"Validation error: {e}")
            return DollarBarValidation(
                exists=False,
                bar_count=0,
                avg_bars_per_day=0.0,
                threshold=self.dollar_bar_threshold,
                threshold_compliant=False,
                errors=errors,
                warnings=warnings
            )

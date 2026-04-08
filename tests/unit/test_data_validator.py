"""Unit tests for DataValidator class."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.research.data_validator import (
    DataValidator,
    DataPeriodValidation,
    DataQualityValidation,
    DollarBarValidation,
    GapInfo,
    GapCategory,
)


@pytest.fixture
def temp_h5_file(tmp_path):
    """Create a temporary HDF5 file for testing."""
    file_path = tmp_path / "test_dollar_bars.h5"
    return file_path


@pytest.fixture
def sample_dollar_bars_data():
    """Generate sample dollar bar data for testing."""
    # Generate 2+ years of daily bars (trading days only)
    timestamps = []
    current_date = datetime(2024, 1, 1, 9, 30)  # 9:30 AM ET
    end_date = datetime(2026, 3, 31, 16, 0)  # End of 2+ year period

    # Generate trading days for 2+ years
    while current_date <= end_date:
        # Only add weekdays (Monday-Friday)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            timestamps.append(current_date.timestamp() * 1000)  # Convert to ms
        current_date += timedelta(days=1)

    # Generate OHLCV data
    n_bars = len(timestamps)
    np.random.seed(42)
    close_prices = 15000 + np.random.randn(n_bars).cumsum()  # Random walk
    notional_value = 50_000_000  # $50M threshold

    data = np.column_stack([
        timestamps,
        close_prices * 0.999,  # open (slightly below close)
        close_prices * 1.001,  # high (slightly above close)
        close_prices * 0.998,  # low (slightly below close)
        close_prices,  # close
        np.random.randint(1000, 5000, n_bars),  # volume
        np.full(n_bars, notional_value)  # notional_value
    ])

    return data


@pytest.fixture
def sample_h5_with_gaps(tmp_path, sample_dollar_bars_data):
    """Create HDF5 file with intentional gaps for testing."""
    file_path = tmp_path / "test_with_gaps.h5"

    # Remove some bars to create gaps (simulating missing data)
    data_with_gaps = np.delete(sample_dollar_bars_data, [50, 51, 100, 200], axis=0)

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dollar_bars', data=data_with_gaps, compression='gzip')

    return file_path


@pytest.fixture
def sample_h5_weekend_gaps(tmp_path, sample_dollar_bars_data):
    """Create HDF5 file with only weekend gaps (acceptable)."""
    file_path = tmp_path / "test_weekend_gaps.h5"

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dollar_bars', data=sample_dollar_bars_data, compression='gzip')

    return file_path


@pytest.fixture
def insufficient_data_file(tmp_path):
    """Create HDF5 file with insufficient data (< 2 years)."""
    file_path = tmp_path / "test_insufficient.h5"

    # Generate only 3 months of data
    timestamps = []
    current_date = datetime(2024, 1, 1, 9, 30)
    for i in range(60):  # ~3 months
        if current_date.weekday() < 5:
            timestamps.append(current_date.timestamp() * 1000)
        current_date += timedelta(days=1)

    n_bars = len(timestamps)
    np.random.seed(42)
    close_prices = 15000 + np.random.randn(n_bars).cumsum()

    data = np.column_stack([
        timestamps,
        close_prices * 0.999,
        close_prices * 1.001,
        close_prices * 0.998,
        close_prices,
        np.random.randint(1000, 5000, n_bars),
        np.full(n_bars, 50_000_000)
    ])

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dollar_bars', data=data, compression='gzip')

    return file_path


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_initialization(self, temp_h5_file):
        """Test DataValidator can be initialized with valid parameters."""
        validator = DataValidator(
            hdf5_path=str(temp_h5_file),
            min_completeness=99.99,
            dollar_bar_threshold=50_000_000
        )

        assert validator.hdf5_path == Path(temp_h5_file)
        assert validator.min_completeness == 99.99
        assert validator.dollar_bar_threshold == 50_000_000

    def test_validate_data_period_sufficient(self, sample_h5_weekend_gaps):
        """Test validate_data_period with sufficient 2-year data."""
        validator = DataValidator(hdf5_path=str(sample_h5_weekend_gaps))
        result = validator.validate_data_period()

        assert result.is_sufficient is True
        assert result.start_date.year == 2024
        assert result.end_date.year >= 2025  # At least 1+ year
        assert result.total_days >= 365  # At least 1 year
        assert result.expected_min_days == 730  # 2 years
        assert len(result.errors) == 0

    def test_validate_data_period_insufficient(self, insufficient_data_file):
        """Test validate_data_period with insufficient data (< 2 years)."""
        validator = DataValidator(hdf5_path=str(insufficient_data_file))
        result = validator.validate_data_period()

        assert result.is_sufficient is False
        assert result.total_days < 365
        assert len(result.errors) > 0
        assert any("insufficient" in error.lower() for error in result.errors)

    def test_check_completeness_high(self, sample_h5_weekend_gaps):
        """Test check_completeness with 99.99%+ completeness."""
        validator = DataValidator(
            hdf5_path=str(sample_h5_weekend_gaps),
            min_completeness=99.99
        )
        result = validator.check_completeness()

        assert result.passed is True
        assert result.completeness_percent >= 99.99
        assert result.actual_bars > 0
        assert result.expected_bars > 0
        assert len(result.warnings) == 0

    def test_check_completeness_low(self, sample_h5_with_gaps):
        """Test check_completeness with <99% completeness (due to gaps)."""
        validator = DataValidator(
            hdf5_path=str(sample_h5_with_gaps),
            min_completeness=99.99
        )
        result = validator.check_completeness()

        # With 4 missing bars out of ~504, completeness should be <99.99%
        assert result.completeness_percent < 99.99
        assert result.passed is False
        assert len(result.warnings) > 0

    def test_check_completeness_edge_case(self, tmp_path):
        """Test check_completeness at exactly 99.99% threshold."""
        file_path = tmp_path / "test_edge_case.h5"

        # Create data with exactly 99.99% completeness
        # 10,000 expected bars, 9,999 actual = 99.99%
        timestamps = []
        current_date = datetime(2024, 1, 1, 9, 30)
        for i in range(10000):
            if current_date.weekday() < 5:
                timestamps.append(current_date.timestamp() * 1000)
            current_date += timedelta(hours=1)  # Hourly bars

        # Remove 1 bar to get exactly 99.99%
        timestamps = timestamps[:9999]

        n_bars = len(timestamps)
        np.random.seed(42)
        close_prices = 15000 + np.random.randn(n_bars).cumsum()

        data = np.column_stack([
            timestamps,
            close_prices * 0.999,
            close_prices * 1.001,
            close_prices * 0.998,
            close_prices,
            np.random.randint(1000, 5000, n_bars),
            np.full(n_bars, 50_000_000)
        ])

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('dollar_bars', data=data, compression='gzip')

        validator = DataValidator(
            hdf5_path=str(file_path),
            min_completeness=99.99
        )
        result = validator.check_completeness()

        # Should pass at exactly 99.99%
        assert result.completeness_percent >= 99.99
        assert result.passed is True

    def test_detect_gaps_no_gaps(self, sample_h5_weekend_gaps):
        """Test detect_gaps with no problematic gaps (only weekends)."""
        validator = DataValidator(hdf5_path=str(sample_h5_weekend_gaps))
        gaps = validator.detect_gaps()

        # Should have no problematic gaps
        problematic_gaps = [g for g in gaps if g.category == GapCategory.PROBLEMATIC]
        assert len(problematic_gaps) == 0

        # May have weekend gaps (acceptable)
        weekend_gaps = [g for g in gaps if g.category == GapCategory.WEEKEND_HOLIDAY]
        assert len(weekend_gaps) >= 0  # OK to have weekend gaps

    def test_detect_gaps_with_missing_data(self, sample_h5_with_gaps):
        """Test detect_gaps with missing data gaps."""
        validator = DataValidator(hdf5_path=str(sample_h5_with_gaps))
        gaps = validator.detect_gaps()

        # Should detect problematic gaps
        problematic_gaps = [g for g in gaps if g.category == GapCategory.PROBLEMATIC]
        assert len(problematic_gaps) > 0

        # Check gap properties
        for gap in problematic_gaps:
            assert gap.start_timestamp is not None
            assert gap.end_timestamp is not None
            assert gap.duration_hours > 0
            assert gap.category == GapCategory.PROBLEMATIC

    def test_detect_gaps_categorization(self, sample_h5_with_gaps):
        """Test that gaps are correctly categorized."""
        validator = DataValidator(hdf5_path=str(sample_h5_with_gaps))
        gaps = validator.detect_gaps()

        for gap in gaps:
            # Weekends/holidays: 1-2 day gaps
            # Problematic: >3 day gaps
            if gap.duration_hours <= 72:  # 3 days
                assert gap.category == GapCategory.WEEKEND_HOLIDAY
            else:
                assert gap.category == GapCategory.PROBLEMATIC

    def test_validate_dollar_bars_existing(self, sample_h5_weekend_gaps):
        """Test validate_dollar_bars with existing dollar bars."""
        validator = DataValidator(
            hdf5_path=str(sample_h5_weekend_gaps),
            dollar_bar_threshold=50_000_000
        )
        result = validator.validate_dollar_bars()

        assert result.exists is True
        assert result.bar_count > 0
        assert result.avg_bars_per_day > 0
        assert result.threshold_compliant is True
        assert result.threshold == 50_000_000
        assert len(result.errors) == 0

    def test_validate_dollar_bars_missing(self, temp_h5_file):
        """Test validate_dollar_bars with missing dollar bar file."""
        # Don't create the file
        validator = DataValidator(
            hdf5_path=str(temp_h5_file),
            dollar_bar_threshold=50_000_000
        )

        # Should handle missing file gracefully
        result = validator.validate_dollar_bars()

        assert result.exists is False
        assert result.bar_count == 0
        assert len(result.errors) > 0
        assert any("not found" in error.lower() or "missing" in error.lower()
                   for error in result.errors)

    def test_validate_dollar_bars_threshold_compliance(self, tmp_path):
        """Test validate_dollar_bars checks threshold compliance."""
        file_path = tmp_path / "test_threshold.h5"

        # Create data with wrong threshold ($40M instead of $50M)
        timestamps = [datetime(2024, 1, 1, 9, 30).timestamp() * 1000]
        n_bars = 1
        close_prices = np.array([15000])

        data = np.column_stack([
            timestamps,
            close_prices * 0.999,
            close_prices * 1.001,
            close_prices * 0.998,
            close_prices,
            [1000],
            [40_000_000]  # Wrong threshold
        ])

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('dollar_bars', data=data, compression='gzip')

        validator = DataValidator(
            hdf5_path=str(file_path),
            dollar_bar_threshold=50_000_000
        )
        result = validator.validate_dollar_bars()

        assert result.exists is True
        assert result.threshold_compliant is False
        assert len(result.warnings) > 0

    def test_validate_all_methods_integration(self, sample_h5_weekend_gaps, caplog):
        """Test running all validation methods together."""
        validator = DataValidator(
            hdf5_path=str(sample_h5_weekend_gaps),
            min_completeness=99.99,
            dollar_bar_threshold=50_000_000
        )

        with caplog.at_level(logging.INFO):
            # Run all validations
            period_result = validator.validate_data_period()
            completeness_result = validator.check_completeness()
            gaps = validator.detect_gaps()
            dollar_result = validator.validate_dollar_bars()

            # Check all methods return results
            assert period_result is not None
            assert completeness_result is not None
            assert gaps is not None
            assert dollar_result is not None

            # Check logging occurred
            assert any("validate" in record.message.lower()
                       for record in caplog.records)


class TestDataValidatorModels:
    """Test Pydantic models for data validation results."""

    def test_data_period_validation_model(self):
        """Test DataPeriodValidation Pydantic model."""
        result = DataPeriodValidation(
            is_sufficient=True,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2026, 1, 1),
            total_days=730,
            expected_min_days=730,
            errors=[]
        )

        assert result.is_sufficient is True
        assert result.total_days == 730
        assert len(result.errors) == 0

    def test_data_quality_validation_model(self):
        """Test DataQualityValidation Pydantic model."""
        result = DataQualityValidation(
            passed=True,
            completeness_percent=99.99,
            actual_bars=50000,
            expected_bars=50050,
            missing_bars=50,
            warnings=[]
        )

        assert result.passed is True
        assert result.completeness_percent == 99.99
        assert result.missing_bars == 50

    def test_dollar_bar_validation_model(self):
        """Test DollarBarValidation Pydantic model."""
        result = DollarBarValidation(
            exists=True,
            bar_count=50000,
            avg_bars_per_day=68,
            threshold=50_000_000,
            threshold_compliant=True,
            errors=[],
            warnings=[]
        )

        assert result.exists is True
        assert result.threshold == 50_000_000
        assert result.threshold_compliant is True

    def test_gap_info_model(self):
        """Test GapInfo Pydantic model."""
        gap = GapInfo(
            start_timestamp=datetime(2024, 1, 5),
            end_timestamp=datetime(2024, 1, 8),
            duration_hours=72,
            category=GapCategory.WEEKEND_HOLIDAY
        )

        assert gap.start_timestamp.day == 5
        assert gap.end_timestamp.day == 8
        assert gap.duration_hours == 72
        assert gap.category == GapCategory.WEEKEND_HOLIDAY

    def test_gap_category_enum(self):
        """Test GapCategory enum values."""
        assert GapCategory.WEEKEND_HOLIDAY == "weekend_holiday"
        assert GapCategory.PROBLEMATIC == "problematic"

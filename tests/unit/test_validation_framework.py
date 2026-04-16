"""Unit tests for validation framework.

Tests for TemporalSplitValidator, DataLeakageDetector, PerformanceValidator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.validation_framework import (
    TemporalSplitValidator,
    DataLeakageDetector,
    PerformanceValidator,
    lock_validation_data
)


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    dates = pd.date_range('2025-01-01', periods=1000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 1000)
    })
    return df


@pytest.fixture
def sample_val_data():
    """Create sample validation data."""
    dates = pd.date_range('2025-10-01', periods=500, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 500)
    })
    return df


@pytest.fixture
def sample_trades():
    """Create sample trade results."""
    dates = pd.date_range('2025-10-01', periods=50, freq='1H')
    df = pd.DataFrame({
        'entry_time': dates,
        'exit_time': dates + timedelta(minutes=30),
        'pnl': np.random.randn(50) * 100  # Mix of wins and losses
    })
    return df


class TestTemporalSplitValidator:
    """Test TemporalSplitValidator class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        validator = TemporalSplitValidator(
            train_start='2025-01-01',
            train_end='2025-09-30',
            val_start='2025-10-01',
            val_end='2025-12-31'
        )
        assert validator.train_start == pd.to_datetime('2025-01-01')
        assert validator.train_end == pd.to_datetime('2025-09-30')
        assert validator.val_start == pd.to_datetime('2025-10-01')
        assert validator.val_end == pd.to_datetime('2025-12-31')

    def test_initialization_with_test(self):
        """Test initialization with test period."""
        validator = TemporalSplitValidator(
            train_start='2025-01-01',
            train_end='2025-09-30',
            val_start='2025-10-01',
            val_end='2025-12-31',
            test_start='2026-01-01',
            test_end='2026-03-31'
        )
        assert validator.test_start == pd.to_datetime('2026-01-01')
        assert validator.test_end == pd.to_datetime('2026-03-31')

    def test_temporal_overlap_detection(self):
        """Test detection of temporal overlap."""
        with pytest.raises(ValueError, match="DATA LEAKAGE RISK"):
            TemporalSplitValidator(
                train_start='2025-01-01',
                train_end='2025-10-15',  # Extends into validation
                val_start='2025-10-01',
                val_end='2025-12-31'
            )

    def test_invalid_date_ranges(self):
        """Test invalid date ranges."""
        with pytest.raises(ValueError, match="Train start >= train end"):
            TemporalSplitValidator(
                train_start='2025-12-31',
                train_end='2025-01-01',
                val_start='2025-10-01',
                val_end='2025-12-31'
            )

    def test_validate_no_leakage_no_overlap(self, sample_train_data, sample_val_data):
        """Test validation with no temporal overlap."""
        validator = TemporalSplitValidator(
            train_start='2025-01-01',
            train_end='2025-09-30',
            val_start='2025-10-01',
            val_end='2025-12-31'
        )

        results = validator.validate_no_leakage(sample_train_data, sample_val_data)
        assert results['passed'] is True
        assert len(results['overlaps']) == 0

    def test_validate_no_leakage_with_overlap(self, sample_train_data, sample_val_data):
        """Test validation with temporal overlap."""
        # Create overlapping data that extends INTO October
        # Create data with 25000 rows starting from Sept 15
        train_data_overlap = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-15 00:00', periods=25000, freq='1min'),
            'close': np.random.randn(25000).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 25000)
        })

        validator = TemporalSplitValidator(
            train_start='2025-01-01',
            train_end='2025-09-30',
            val_start='2025-10-01',
            val_end='2025-12-31'
        )

        results = validator.validate_no_leakage(train_data_overlap, sample_val_data)
        # Should detect overlap because train data extends past Sept 30
        assert results['passed'] is False
        assert len(results['overlaps']) > 0

    def test_summary(self):
        """Test summary generation."""
        validator = TemporalSplitValidator(
            train_start='2025-01-01',
            train_end='2025-09-30',
            val_start='2025-10-01',
            val_end='2025-12-31'
        )

        summary = validator.summary()
        assert 'Train: 2025-01-01 to 2025-09-30' in summary
        assert 'Validation: 2025-10-01 to 2025-12-31' in summary
        assert 'Train: 272 days' in summary
        assert 'Validation: 91 days' in summary


class TestDataLeakageDetector:
    """Test DataLeakageDetector class."""

    def test_detect_temporal_leakage_no_leakage(self):
        """Test temporal leakage detection with no leakage."""
        detector = DataLeakageDetector()

        # Create safe rolling features
        dates = pd.date_range('2025-01-01', periods=1000, freq='1min')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(1000).cumsum() + 100,
            'sma_50': np.nan,  # Rolling features have NaN at start
            'rsi_14': np.nan
        })

        feature_windows = {'sma_50': 50, 'rsi_14': 14}
        results = detector.detect_temporal_leakage(df, feature_windows)

        assert results['leakage_detected'] is False

    def test_detect_temporal_leakage_with_leakage(self):
        """Test temporal leakage detection with leakage."""
        detector = DataLeakageDetector()

        # Create features that shouldn't have NaN but window > 0
        dates = pd.date_range('2025-01-01', periods=1000, freq='1min')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(1000).cumsum() + 100,
            'feature_a': np.random.randn(1000)  # No NaNs despite window
        })

        feature_windows = {'feature_a': 50}
        results = detector.detect_temporal_leakage(df, feature_windows)

        # Feature has no NaNs at start but window > 0 - suspicious
        assert results['leakage_detected'] is True
        assert len(results['leaky_features']) > 0

    def test_detect_target_leakage_no_leakage(self):
        """Test target leakage detection with no leakage."""
        detector = DataLeakageDetector()

        df = pd.DataFrame({
            'target': np.random.randn(1000),
            'feature_a': np.random.randn(1000),
            'feature_b': np.random.randn(1000)
        })

        results = detector.detect_target_leakage(df, 'target', ['feature_a', 'feature_b'])
        assert results['leakage_detected'] is False

    def test_detect_target_leakage_with_leakage(self):
        """Test target leakage detection with leakage."""
        detector = DataLeakageDetector()

        # Create feature nearly identical to target
        target = np.random.randn(1000)
        df = pd.DataFrame({
            'target': target,
            'feature_leaky': target * 0.98  # 98% correlated
        })

        results = detector.detect_target_leakage(df, 'target', ['feature_leaky'])
        assert results['leakage_detected'] is True
        assert results['correlations'][0]['risk'] == 'HIGH'


class TestPerformanceValidator:
    """Test PerformanceValidator class."""

    def test_calculate_metrics_with_trades(self, sample_trades):
        """Test metrics calculation with trades."""
        validator = PerformanceValidator()
        metrics = validator.calculate_metrics(sample_trades)

        assert metrics['trades'] == 50
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'profit_factor' in metrics
        assert 'max_drawdown' in metrics

    def test_calculate_metrics_empty(self):
        """Test metrics calculation with no trades."""
        validator = PerformanceValidator()
        df = pd.DataFrame({'entry_time': [], 'exit_time': [], 'pnl': []})
        metrics = validator.calculate_metrics(df)

        assert metrics['trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0

    def test_validate_against_targets_pass(self, sample_trades):
        """Test validation against targets (passing)."""
        validator = PerformanceValidator()

        # Create metrics that should pass
        metrics = {
            'trades': 100,
            'win_rate': 50.0,
            'trades_per_day': 15.0,
            'expectation_per_trade': 30.0,
            'sharpe_ratio': 0.8,
            'profit_factor': 1.5,
            'max_drawdown': 500.0
        }

        results = validator.validate_against_targets(metrics)
        assert results['overall_passed'] is True
        assert len(results['passed']) > 0

    def test_validate_against_targets_fail(self):
        """Test validation against targets (failing)."""
        validator = PerformanceValidator()

        # Create metrics that should fail
        metrics = {
            'trades': 5,
            'win_rate': 40.0,
            'trades_per_day': 2.0,
            'expectation_per_trade': 10.0,
            'sharpe_ratio': 0.3,
            'profit_factor': 1.0,
            'max_drawdown': 1500.0
        }

        results = validator.validate_against_targets(metrics)
        assert results['overall_passed'] is False
        assert len(results['failed']) > 0

    def test_check_red_flags_unrealistic_win_rate(self):
        """Test red flag detection for unrealistic win rate."""
        validator = PerformanceValidator()

        metrics = {
            'trades': 100,
            'win_rate': 85.0,  # Unrealistic
            'trades_per_day': 15.0,
            'sharpe_ratio': 1.0,
            'expectation_per_trade': 50.0
        }

        flags = validator.check_red_flags(metrics)
        assert len(flags) > 0
        assert any('UNREALISTIC' in flag for flag in flags)

    def test_check_red_flags_unrealistic_sharpe(self):
        """Test red flag detection for unrealistic Sharpe ratio."""
        validator = PerformanceValidator()

        metrics = {
            'trades': 100,
            'win_rate': 55.0,
            'trades_per_day': 15.0,
            'sharpe_ratio': 5.0,  # Unrealistic
            'expectation_per_trade': 50.0
        }

        flags = validator.check_red_flags(metrics)
        assert len(flags) > 0
        assert any('OVERFITTING' in flag for flag in flags)

    def test_check_red_flags_no_trades(self):
        """Test red flag detection for zero trades."""
        validator = PerformanceValidator()

        metrics = {
            'trades': 0,
            'win_rate': 0.0,
            'trades_per_day': 0.0,
            'sharpe_ratio': 0.0,
            'expectation_per_trade': 0.0
        }

        flags = validator.check_red_flags(metrics)
        assert len(flags) > 0
        assert any('Zero trades' in flag for flag in flags)

    def test_check_red_flags_too_frequent(self):
        """Test red flag detection for excessive trade frequency."""
        validator = PerformanceValidator()

        metrics = {
            'trades': 1000,
            'win_rate': 50.0,
            'trades_per_day': 40.0,  # Too frequent
            'sharpe_ratio': 1.0,
            'expectation_per_trade': 20.0
        }

        flags = validator.check_red_flags(metrics)
        assert len(flags) > 0
        assert any('TOO FREQUENT' in flag for flag in flags)

    def test_generate_validation_report(self, sample_trades, tmp_path):
        """Test validation report generation."""
        validator = PerformanceValidator()
        output_path = tmp_path / "validation_report.md"

        report = validator.generate_validation_report(sample_trades, output_path)

        assert len(report) > 0
        assert '# Performance Validation Report' in report
        assert output_path.exists()

    def test_generate_validation_report_no_trades(self, tmp_path):
        """Test validation report with no trades."""
        validator = PerformanceValidator()
        df = pd.DataFrame({'entry_time': [], 'exit_time': [], 'pnl': []})
        output_path = tmp_path / "validation_report_empty.md"

        report = validator.generate_validation_report(df, output_path)

        assert len(report) > 0
        assert 'Zero trades generated' in report


class TestLockValidationData:
    """Test validation data locking functionality."""

    def test_lock_validation_data(self, sample_val_data, tmp_path):
        """Test locking validation data with checksum."""
        output_path = tmp_path / "validation_data.csv"

        checksum = lock_validation_data(sample_val_data, output_path)

        assert len(checksum) == 64  # SHA256 hash length
        assert output_path.exists()
        assert (output_path.with_suffix('.sha256')).exists()

    def test_locked_data_integrity(self, sample_val_data, tmp_path):
        """Test that locked data maintains integrity."""
        output_path = tmp_path / "validation_data.csv"

        original_checksum = lock_validation_data(sample_val_data, output_path)

        # Read back and verify (parse timestamps back to datetime)
        loaded_df = pd.read_csv(output_path, parse_dates=['timestamp'])
        pd.testing.assert_frame_equal(sample_val_data.reset_index(drop=True), loaded_df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

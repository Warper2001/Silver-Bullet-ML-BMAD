"""Integration tests for end-to-end data preparation workflow."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import pytest_asyncio


@pytest.fixture
def integration_test_h5_file(tmp_path):
    """Create HDF5 file with 1 month of MNQ data for integration testing."""
    file_path = tmp_path / "test_integration_mnq.h5"

    # Generate 1 month of daily bars (trading days only)
    timestamps = []
    current_date = datetime(2024, 1, 1, 9, 30)  # 9:30 AM ET
    end_date = datetime(2024, 1, 31, 16, 0)  # End of January

    while current_date <= end_date:
        # Only add weekdays (Monday-Friday)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            timestamps.append(current_date.timestamp() * 1000)  # Convert to ms
        current_date = current_date.replace(hour=9, minute=30) + __import__('datetime').timedelta(days=1)

    # Generate OHLCV data with some intentional gaps
    n_bars = len(timestamps)

    # Remove some bars to create gaps (simulate missing data)
    # Remove bars at indices 5, 6, 15, 16 to create gaps
    indices_to_remove = [5, 6, 15, 16]
    timestamps = [t for i, t in enumerate(timestamps) if i not in indices_to_remove]

    n_bars = len(timestamps)
    np.random.seed(42)
    close_prices = 15000 + np.random.randn(n_bars).cumsum()  # Random walk
    notional_value = 50_000_000  # $50M threshold

    data = np.column_stack([
        timestamps,
        close_prices * 0.999,  # open
        close_prices * 1.001,  # high
        close_prices * 0.998,  # low
        close_prices,  # close
        np.random.randint(1000, 5000, n_bars),  # volume
        np.full(n_bars, notional_value)  # notional_value
    ])

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dollar_bars', data=data, compression='gzip')

    return file_path


@pytest.fixture
def integration_test_output_dir(tmp_path):
    """Create temporary output directory for integration tests."""
    output_dir = tmp_path / "integration_reports"
    output_dir.mkdir(exist_ok=True)
    return output_dir


class TestDataPreparationIntegration:
    """Integration tests for complete data preparation workflow."""

    @pytest.mark.integration
    def test_end_to_end_validation_workflow(
        self,
        integration_test_h5_file,
        integration_test_output_dir
    ):
        """Test complete end-to-end validation workflow."""
        from src.research.data_validator import DataValidator
        from src.research.data_quality_report import DataQualityReport

        # Step 1: Initialize validator
        validator = DataValidator(
            hdf5_path=str(integration_test_h5_file),
            min_completeness=99.99,
            dollar_bar_threshold=50_000_000
        )

        # Step 2: Run all validations
        period_result = validator.validate_data_period()
        quality_result = validator.check_completeness()
        gaps = validator.detect_gaps()
        dollar_result = validator.validate_dollar_bars()

        # Step 3: Verify all validation results are returned
        assert period_result is not None
        assert quality_result is not None
        assert gaps is not None
        assert dollar_result is not None

        # Step 4: Generate report
        report_generator = DataQualityReport(output_dir=str(integration_test_output_dir))
        report_path = report_generator.generate_report(
            period_result=period_result,
            quality_result=quality_result,
            dollar_result=dollar_result,
            gaps=gaps
        )

        # Step 5: Verify report file exists and contains expected sections
        assert report_path.exists()
        assert report_path.parent == integration_test_output_dir

        content = report_path.read_text()

        # Check required sections
        required_sections = [
            "# Data Quality Report",
            "## Executive Summary",
            "## Data Period Coverage",
            "## Completeness Analysis",
            "## Dollar Bar Analysis",
            "## Gap Analysis",
            "## Recommendations"
        ]

        for section in required_sections:
            assert section in content, f"Missing section: {section}"

        # Step 6: Verify recommendation is generated
        recommendation = report_generator.recommend_for_backtesting(
            period_result=period_result,
            quality_result=quality_result,
            dollar_result=dollar_result,
            gaps=gaps
        )

        assert recommendation is not None
        assert recommendation.status in ["GO", "CAUTION", "NO_GO"]
        assert recommendation.reasoning is not None

    @pytest.mark.integration
    def test_workflow_with_missing_data(self, tmp_path):
        """Test workflow handles missing data gracefully."""
        from src.research.data_validator import DataValidator
        from src.research.data_quality_report import DataQualityReport

        # Create file with significant gaps
        file_path = tmp_path / "test_missing_data.h5"
        timestamps = [
            datetime(2024, 1, 1, 9, 30).timestamp() * 1000,
            datetime(2024, 1, 2, 9, 30).timestamp() * 1000,
            # Missing Jan 3-10 (large gap)
            datetime(2024, 1, 11, 9, 30).timestamp() * 1000,
        ]

        n_bars = len(timestamps)
        close_prices = np.array([15000, 15050, 15100])

        data = np.column_stack([
            timestamps,
            close_prices * 0.999,
            close_prices * 1.001,
            close_prices * 0.998,
            close_prices,
            [1000, 1000, 1000],
            np.full(n_bars, 50_000_000)
        ])

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('dollar_bars', data=data, compression='gzip')

        # Run workflow
        validator = DataValidator(hdf5_path=str(file_path))
        period_result = validator.validate_data_period()
        gaps = validator.detect_gaps()

        # Should detect problematic gap
        assert len(gaps) > 0
        # Gap should be categorized as problematic (>3 days)
        problematic_gaps = [g for g in gaps if g.category.value == "problematic"]
        assert len(problematic_gaps) > 0

    @pytest.mark.integration
    def test_cli_script_invocation(
        self,
        integration_test_h5_file,
        integration_test_output_dir
    ):
        """Test CLI script can be invoked and produces expected output."""
        # Invoke CLI script
        result = subprocess.run(
            [
                sys.executable, "-m", "src.cli.validate_data",
                "--data-path", str(integration_test_h5_file),
                "--output-dir", str(integration_test_output_dir),
                "--min-completeness", "99.99"
            ],
            capture_output=True,
            text=True
        )

        # Check exit code (0 = success, 1 = validation failure, both are acceptable)
        assert result.returncode in [0, 1]

        # Check stdout contains expected information
        stdout = result.stdout

        # Should contain validation summary
        assert "VALIDATION SUMMARY" in stdout or "Data Quality" in stdout

        # Should contain key metrics
        assert "Data Period" in stdout or "Completeness" in stdout

        # Should contain recommendation
        assert "GO" in stdout or "CAUTION" in stdout or "NO-GO" in stdout

    @pytest.mark.integration
    def test_cli_creates_report_file(
        self,
        integration_test_h5_file,
        integration_test_output_dir
    ):
        """Test CLI script creates report file in output directory."""
        import subprocess
        import sys

        # Run CLI
        subprocess.run(
            [
                sys.executable, "-m", "src.cli.validate_data",
                "--data-path", str(integration_test_h5_file),
                "--output-dir", str(integration_test_output_dir)
            ],
            capture_output=True
        )

        # Check report file was created
        report_files = list(integration_test_output_dir.glob("data_quality_*.md"))
        assert len(report_files) > 0, "No report files were created"

        # Verify report content
        report_path = report_files[0]
        content = report_path.read_text()

        assert "# Data Quality Report" in content
        assert "## Recommendations" in content

    @pytest.mark.integration
    def test_cli_with_real_mnq_data(self):
        """Test CLI with actual MNQ data if available.

        This test is marked as slow and will be skipped if no data is available.
        """
        # Check if real MNQ data exists
        data_dir = Path("data/processed/dollar_bars")

        if not data_dir.exists():
            pytest.skip("No MNQ data directory found")

        # Find any MNQ HDF5 file
        h5_files = list(data_dir.glob("MNQ_dollar_bars_*.h5"))

        if not h5_files:
            pytest.skip("No MNQ HDF5 files found")

        # Use the first available file
        test_file = h5_files[0]

        # Run validation on real data
        from src.research.data_validator import DataValidator

        validator = DataValidator(
            hdf5_path=str(test_file),
            min_completeness=99.99
        )

        # Should not raise any exceptions
        period_result = validator.validate_data_period()
        quality_result = validator.check_completeness()
        gaps = validator.detect_gaps()
        dollar_result = validator.validate_dollar_bars()

        # Verify results are returned
        assert period_result is not None
        assert quality_result is not None
        assert gaps is not None
        assert dollar_result is not None

        # Verify dollar bar data exists
        assert dollar_result.exists is True

    @pytest.mark.integration
    def test_recommendation_logic_integration(
        self,
        integration_test_h5_file,
        integration_test_output_dir
    ):
        """Test recommendation logic integration with validation results."""
        from src.research.data_validator import DataValidator
        from src.research.data_quality_report import DataQualityReport

        # Run validation
        validator = DataValidator(
            hdf5_path=str(integration_test_h5_file),
            min_completeness=99.99
        )

        period_result = validator.validate_data_period()
        quality_result = validator.check_completeness()
        gaps = validator.detect_gaps()
        dollar_result = validator.validate_dollar_bars()

        # Get recommendation
        report_generator = DataQualityReport(output_dir=str(integration_test_output_dir))
        recommendation = report_generator.recommend_for_backtesting(
            period_result=period_result,
            quality_result=quality_result,
            dollar_result=dollar_result,
            gaps=gaps
        )

        # Verify recommendation structure
        assert hasattr(recommendation, 'status')
        assert hasattr(recommendation, 'reasoning')
        assert hasattr(recommendation, 'issues')

        # Verify status is valid
        assert recommendation.status.value in ['GO', 'CAUTION', 'NO_GO']

        # Verify reasoning is non-empty
        assert len(recommendation.reasoning) > 0

        # Verify issues list (should be empty for GO, populated for CAUTION/NO-GO)
        if recommendation.status.value == 'GO':
            # May have 0 issues or minor issues
            assert isinstance(recommendation.issues, list)
        else:
            # Should have at least one issue
            assert len(recommendation.issues) > 0

    @pytest.mark.integration
    def test_error_handling_missing_file(self, tmp_path):
        """Test error handling when HDF5 file doesn't exist."""
        from src.research.data_validator import DataValidator

        non_existent_file = tmp_path / "does_not_exist.h5"

        validator = DataValidator(hdf5_path=str(non_existent_file))

        # Should handle gracefully
        period_result = validator.validate_data_period()

        assert period_result is not None
        assert period_result.is_sufficient is False
        assert len(period_result.errors) > 0

    @pytest.mark.integration
    def test_error_handling_corrupted_file(self, tmp_path):
        """Test error handling when HDF5 file is corrupted."""
        from src.research.data_validator import DataValidator

        # Create invalid HDF5 file
        corrupted_file = tmp_path / "corrupted.h5"
        corrupted_file.write_text("This is not a valid HDF5 file")

        validator = DataValidator(hdf5_path=str(corrupted_file))

        # Should handle gracefully
        dollar_result = validator.validate_dollar_bars()

        assert dollar_result is not None
        assert dollar_result.exists is False
        assert len(dollar_result.errors) > 0 or len(dollar_result.warnings) > 0

"""Unit tests for DataQualityReport class."""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.research.data_quality_report import (
    DataQualityReport,
    BacktestRecommendation,
    RecommendationStatus,
)
from src.research.data_validator import (
    DataPeriodValidation,
    DataQualityValidation,
    DollarBarValidation,
    GapInfo,
    GapCategory,
)


@pytest.fixture
def validation_results_passing():
    """Create validation results that pass all checks."""
    period_result = DataPeriodValidation(
        is_sufficient=True,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2026, 1, 1),
        total_days=730,
        expected_min_days=730,
        errors=[]
    )

    quality_result = DataQualityValidation(
        passed=True,
        completeness_percent=99.99,
        actual_bars=500,
        expected_bars=500,
        missing_bars=0,
        warnings=[]
    )

    dollar_result = DollarBarValidation(
        exists=True,
        bar_count=50000,
        avg_bars_per_day=68,
        threshold=50_000_000,
        threshold_compliant=True,
        errors=[],
        warnings=[]
    )

    gaps = []  # No problematic gaps

    return {
        'period': period_result,
        'quality': quality_result,
        'dollar_bars': dollar_result,
        'gaps': gaps
    }


@pytest.fixture
def validation_results_caution():
    """Create validation results with minor issues (CAUTION)."""
    period_result = DataPeriodValidation(
        is_sufficient=True,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2026, 1, 1),
        total_days=730,
        expected_min_days=730,
        errors=[]
    )

    quality_result = DataQualityValidation(
        passed=False,  # Below threshold
        completeness_percent=99.5,
        actual_bars=497,
        expected_bars=500,
        missing_bars=3,
        warnings=["Completeness below threshold"]
    )

    dollar_result = DollarBarValidation(
        exists=True,
        bar_count=50000,
        avg_bars_per_day=68,
        threshold=50_000_000,
        threshold_compliant=True,
        errors=[],
        warnings=[]
    )

    gaps = [
        GapInfo(
            start_timestamp=datetime(2024, 5, 1),
            end_timestamp=datetime(2024, 5, 5),
            duration_hours=96,
            category=GapCategory.PROBLEMATIC
        )
    ]

    return {
        'period': period_result,
        'quality': quality_result,
        'dollar_bars': dollar_result,
        'gaps': gaps
    }


@pytest.fixture
def validation_results_no_go():
    """Create validation results with major issues (NO-GO)."""
    period_result = DataPeriodValidation(
        is_sufficient=False,  # Insufficient data period
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        total_days=365,
        expected_min_days=730,
        errors=["Insufficient data period"]
    )

    quality_result = DataQualityValidation(
        passed=False,
        completeness_percent=95.0,
        actual_bars=475,
        expected_bars=500,
        missing_bars=25,
        warnings=["Completeness 95.0% below threshold 99.99%"]
    )

    dollar_result = DollarBarValidation(
        exists=False,  # Dollar bars missing
        bar_count=0,
        avg_bars_per_day=0.0,
        threshold=50_000_000,
        threshold_compliant=False,
        errors=["Dollar bar file not found"],
        warnings=[]
    )

    gaps = [
        GapInfo(
            start_timestamp=datetime(2024, 3, 1),
            end_timestamp=datetime(2024, 3, 10),
            duration_hours=216,
            category=GapCategory.PROBLEMATIC
        ),
        GapInfo(
            start_timestamp=datetime(2024, 7, 1),
            end_timestamp=datetime(2024, 7, 15),
            duration_hours=336,
            category=GapCategory.PROBLEMATIC
        )
    ]

    return {
        'period': period_result,
        'quality': quality_result,
        'dollar_bars': dollar_result,
        'gaps': gaps
    }


class TestDataQualityReport:
    """Test suite for DataQualityReport class."""

    def test_initialization(self, tmp_path):
        """Test DataQualityReport can be initialized."""
        report = DataQualityReport(output_dir=str(tmp_path))

        assert report.output_dir == Path(tmp_path)

    def test_generate_report_passing(self, validation_results_passing, tmp_path):
        """Test generate_report with passing validation results."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_passing['period'],
            quality_result=validation_results_passing['quality'],
            dollar_result=validation_results_passing['dollar_bars'],
            gaps=validation_results_passing['gaps']
        )

        # Check report file was created
        assert report_path.exists()
        assert report_path.parent == Path(tmp_path)

        # Read and verify report content
        content = report_path.read_text()

        # Check sections exist
        assert "# Data Quality Report" in content
        assert "## Executive Summary" in content
        assert "## Data Period Coverage" in content
        assert "## Completeness Analysis" in content
        assert "## Dollar Bar Analysis" in content
        assert "## Gap Analysis" in content
        assert "## Recommendations" in content

        # Check passing indicators
        assert "**PASS**" in content or "✅" in content

    def test_generate_report_caution(self, validation_results_caution, tmp_path):
        """Test generate_report with caution results."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_caution['period'],
            quality_result=validation_results_caution['quality'],
            dollar_result=validation_results_caution['dollar_bars'],
            gaps=validation_results_caution['gaps']
        )

        content = report_path.read_text()

        # Check warnings are present
        assert "CAUTION" in content or "⚠️" in content or "WARNING" in content

        # Check gaps are documented
        assert "Gap Analysis" in content
        assert "1" in content or "problematic" in content.lower()

    def test_generate_report_no_go(self, validation_results_no_go, tmp_path):
        """Test generate_report with NO-GO results."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_no_go['period'],
            quality_result=validation_results_no_go['quality'],
            dollar_result=validation_results_no_go['dollar_bars'],
            gaps=validation_results_no_go['gaps']
        )

        content = report_path.read_text()

        # Check failure indicators
        assert "FAIL" in content or "❌" in content or "NO-GO" in content

        # Check errors are documented
        assert "Insufficient data period" in content
        assert "Dollar bar file not found" in content

        # Check gaps are listed
        assert "Gap Analysis" in content

    def test_recommend_for_backtesting_go(self, validation_results_passing):
        """Test recommend_for_backtesting returns GO for passing data."""
        report = DataQualityReport()
        recommendation = report.recommend_for_backtesting(
            period_result=validation_results_passing['period'],
            quality_result=validation_results_passing['quality'],
            dollar_result=validation_results_passing['dollar_bars'],
            gaps=validation_results_passing['gaps']
        )

        assert recommendation.status == RecommendationStatus.GO
        assert len(recommendation.issues) == 0
        assert "backtesting" in recommendation.reasoning.lower()

    def test_recommend_for_backtesting_caution(self, validation_results_caution):
        """Test recommend_for_backtesting returns CAUTION for minor issues."""
        report = DataQualityReport()
        recommendation = report.recommend_for_backtesting(
            period_result=validation_results_caution['period'],
            quality_result=validation_results_caution['quality'],
            dollar_result=validation_results_caution['dollar_bars'],
            gaps=validation_results_caution['gaps']
        )

        assert recommendation.status == RecommendationStatus.CAUTION
        assert len(recommendation.issues) > 0
        assert "caution" in recommendation.reasoning.lower() or "but" in recommendation.reasoning.lower()

    def test_recommend_for_backtesting_no_go(self, validation_results_no_go):
        """Test recommend_for_backtesting returns NO-GO for major issues."""
        report = DataQualityReport()
        recommendation = report.recommend_for_backtesting(
            period_result=validation_results_no_go['period'],
            quality_result=validation_results_no_go['quality'],
            dollar_result=validation_results_no_go['dollar_bars'],
            gaps=validation_results_no_go['gaps']
        )

        assert recommendation.status == RecommendationStatus.NO_GO
        assert len(recommendation.issues) > 0
        assert any("insufficient" in issue.lower() for issue in recommendation.issues)
        assert any("dollar bar" in issue.lower() or "missing" in issue.lower()
                   for issue in recommendation.issues)

    def test_report_filename_format(self, validation_results_passing, tmp_path):
        """Test report filename follows expected format."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_passing['period'],
            quality_result=validation_results_passing['quality'],
            dollar_result=validation_results_passing['dollar_bars'],
            gaps=validation_results_passing['gaps']
        )

        # Check filename format: data_quality_YYYY-MM-DD.md
        assert report_path.name.startswith("data_quality_")
        assert report_path.name.endswith(".md")

        # Check date format in filename
        # Extract date part: data_quality_YYYY-MM-DD.md
        date_part = report_path.stem.replace("data_quality_", "")
        assert len(date_part.split("-")) == 3  # YYYY-MM-DD format

    def test_report_contains_all_sections(self, validation_results_passing, tmp_path):
        """Test report contains all required sections."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_passing['period'],
            quality_result=validation_results_passing['quality'],
            dollar_result=validation_results_passing['dollar_bars'],
            gaps=validation_results_passing['gaps']
        )

        content = report_path.read_text()

        # Required sections
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

    def test_report_formats_numbers_correctly(self, validation_results_passing, tmp_path):
        """Test report formats percentages and statistics correctly."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_passing['period'],
            quality_result=validation_results_passing['quality'],
            dollar_result=validation_results_passing['dollar_bars'],
            gaps=validation_results_passing['gaps']
        )

        content = report_path.read_text()

        # Check percentage formatting (should have 2 decimal places)
        assert "99.99%" in content

        # Check large numbers are formatted (with commas)
        assert "50,000" in content or "50000" in content

    def test_report_handles_gaps_table(self, validation_results_caution, tmp_path):
        """Test report creates proper table for gap analysis."""
        report = DataQualityReport(output_dir=str(tmp_path))

        report_path = report.generate_report(
            period_result=validation_results_caution['period'],
            quality_result=validation_results_caution['quality'],
            dollar_result=validation_results_caution['dollar_bars'],
            gaps=validation_results_caution['gaps']
        )

        content = report_path.read_text()

        # Check gap information is present
        assert "2024-05-01" in content or "May" in content
        assert "96" in content or "96.0" in content  # duration_hours
        assert "problematic" in content.lower()


class TestBacktestRecommendation:
    """Test Pydantic model for backtest recommendation."""

    def test_recommendation_model_go(self):
        """Test BacktestRecommendation model with GO status."""
        recommendation = BacktestRecommendation(
            status=RecommendationStatus.GO,
            reasoning="Data quality meets all requirements",
            issues=[]
        )

        assert recommendation.status == RecommendationStatus.GO
        assert len(recommendation.issues) == 0

    def test_recommendation_model_caution(self):
        """Test BacktestRecommendation model with CAUTION status."""
        recommendation = BacktestRecommendation(
            status=RecommendationStatus.CAUTION,
            reasoning="Minor issues detected",
            issues=["Completeness at 99.5%", "1 problematic gap"]
        )

        assert recommendation.status == RecommendationStatus.CAUTION
        assert len(recommendation.issues) == 2

    def test_recommendation_model_no_go(self):
        """Test BacktestRecommendation model with NO-GO status."""
        recommendation = BacktestRecommendation(
            status=RecommendationStatus.NO_GO,
            reasoning="Critical data quality issues",
            issues=["Insufficient data period", "Missing dollar bars"]
        )

        assert recommendation.status == RecommendationStatus.NO_GO
        assert len(recommendation.issues) > 0

    def test_recommendation_status_enum(self):
        """Test RecommendationStatus enum values."""
        assert RecommendationStatus.GO == "GO"
        assert RecommendationStatus.CAUTION == "CAUTION"
        assert RecommendationStatus.NO_GO == "NO_GO"

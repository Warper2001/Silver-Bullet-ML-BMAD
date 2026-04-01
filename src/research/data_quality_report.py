"""Data quality report generation for MNQ historical data validation.

This module generates comprehensive markdown reports from data validation results,
including recommendations for backtesting suitability.
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from src.research.data_validator import (
    DataPeriodValidation,
    DataQualityValidation,
    DollarBarValidation,
    GapInfo,
)


logger = logging.getLogger(__name__)


class RecommendationStatus(str, Enum):
    """Recommendation status for backtesting."""

    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO_GO"


class BacktestRecommendation(BaseModel):
    """Recommendation for backtesting suitability."""

    status: RecommendationStatus
    reasoning: str
    issues: list[str]


class DataQualityReport:
    """Generate comprehensive data quality reports from validation results.

    Creates structured markdown reports with sections:
    - Executive Summary
    - Data Period Coverage
    - Completeness Analysis
    - Dollar Bar Analysis
    - Gap Analysis
    - Recommendations

    Example:
        >>> report = DataQualityReport(output_dir="data/reports")
        >>> report_path = report.generate_report(
        ...     period_result=period_validation,
        ...     quality_result=quality_validation,
        ...     dollar_result=dollar_validation,
        ...     gaps=gaps
        ... )
        >>> recommendation = report.recommend_for_backtesting(...)
    """

    def __init__(self, output_dir: str | Path = "data/reports"):
        """Initialize DataQualityReport.

        Args:
            output_dir: Directory to save reports (default: data/reports)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DataQualityReport with output dir: {self.output_dir}")

    def generate_report(
        self,
        period_result: DataPeriodValidation,
        quality_result: DataQualityValidation,
        dollar_result: DollarBarValidation,
        gaps: list[GapInfo],
    ) -> Path:
        """Generate comprehensive data quality report.

        Args:
            period_result: Data period validation results
            quality_result: Data completeness validation results
            dollar_result: Dollar bar validation results
            gaps: List of detected gaps

        Returns:
            Path to generated report file

        Example:
            >>> report_path = report.generate_report(
            ...     period_result=period_validation,
            ...     quality_result=quality_validation,
            ...     dollar_result=dollar_validation,
            ...     gaps=gaps
            ... )
        """
        logger.info("Generating data quality report...")

        # Generate filename with current date
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_filename = f"data_quality_{report_date}.md"
        report_path = self.output_dir / report_filename

        # Build report content
        content = self._build_report_content(
            period_result, quality_result, dollar_result, gaps
        )

        # Write report
        with open(report_path, 'w') as f:
            f.write(content)

        logger.info(f"Data quality report saved to: {report_path}")

        return report_path

    def _build_report_content(
        self,
        period_result: DataPeriodValidation,
        quality_result: DataQualityValidation,
        dollar_result: DollarBarValidation,
        gaps: list[GapInfo],
    ) -> str:
        """Build complete report content.

        Args:
            period_result: Data period validation results
            quality_result: Data completeness validation results
            dollar_result: Dollar bar validation results
            gaps: List of detected gaps

        Returns:
            Complete markdown report content
        """
        lines = []

        # Title and metadata
        lines.append("# Data Quality Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n---\n")

        # Executive Summary
        lines.extend(self._format_executive_summary(
            period_result, quality_result, dollar_result, gaps
        ))

        # Data Period Coverage
        lines.extend(self._format_data_period_coverage(period_result))

        # Completeness Analysis
        lines.extend(self._format_completeness_analysis(quality_result))

        # Dollar Bar Analysis
        lines.extend(self._format_dollar_bar_analysis(dollar_result))

        # Gap Analysis
        lines.extend(self._format_gap_analysis(gaps))

        # Recommendations
        lines.extend(self._format_recommendations(
            period_result, quality_result, dollar_result, gaps
        ))

        return "\n".join(lines)

    def _format_executive_summary(
        self,
        period_result: DataPeriodValidation,
        quality_result: DataQualityValidation,
        dollar_result: DollarBarValidation,
        gaps: list[GapInfo],
    ) -> list[str]:
        """Format executive summary section."""
        lines = ["## Executive Summary\n"]

        # Overall status
        all_pass = (
            period_result.is_sufficient and
            quality_result.passed and
            dollar_result.exists and
            dollar_result.threshold_compliant and
            all(gap.category.value != "problematic" for gap in gaps)
        )

        if all_pass:
            lines.append("**Overall Status:** ✅ PASS\n")
        elif quality_result.passed and dollar_result.exists:
            lines.append("**Overall Status:** ⚠️ CAUTION\n")
        else:
            lines.append("**Overall Status:** ❌ FAIL\n")

        # Key metrics
        lines.append("### Key Metrics")
        lines.append(f"- **Data Period:** {period_result.start_date.date()} to {period_result.end_date.date()} ({period_result.total_days} days)")
        lines.append(f"- **Completeness:** {quality_result.completeness_percent:.2f}%")
        lines.append(f"- **Total Dollar Bars:** {dollar_result.bar_count:,}")
        lines.append(f"- **Avg Bars/Day:** {dollar_result.avg_bars_per_day:.1f}")

        # Count gaps by category
        problematic_gaps = [g for g in gaps if g.category.value == "problematic"]
        lines.append(f"- **Problematic Gaps:** {len(problematic_gaps)}")

        lines.append("\n---\n")
        return lines

    def _format_data_period_coverage(self, result: DataPeriodValidation) -> list[str]:
        """Format data period coverage section."""
        lines = ["## Data Period Coverage\n"]

        if result.is_sufficient:
            lines.append("✅ **Status:** PASS\n")
        else:
            lines.append("❌ **Status:** FAIL\n")

        lines.append("### Details")
        lines.append(f"- **Start Date:** {result.start_date.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"- **End Date:** {result.end_date.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"- **Total Days:** {result.total_days}")
        lines.append(f"- **Required Minimum:** {result.expected_min_days} days (2 years)")

        if result.errors:
            lines.append("\n### Errors")
            for error in result.errors:
                lines.append(f"- ❌ {error}")

        lines.append("\n---\n")
        return lines

    def _format_completeness_analysis(self, result: DataQualityValidation) -> list[str]:
        """Format completeness analysis section."""
        lines = ["## Completeness Analysis\n"]

        if result.passed:
            lines.append(f"✅ **Status:** PASS ({result.completeness_percent:.2f}%)\n")
        else:
            lines.append(f"❌ **Status:** FAIL ({result.completeness_percent:.2f}%)\n")

        lines.append("### Details")
        lines.append(f"- **Actual Unique Dates:** {result.actual_bars}")
        lines.append(f"- **Expected Trading Days:** {result.expected_bars}")
        lines.append(f"- **Missing Trading Days:** {result.missing_bars}")
        lines.append(f"- **Completeness:** {result.completeness_percent:.3f}%")
        lines.append(f"- **Target Threshold:** 99.99%")

        if result.warnings:
            lines.append("\n### Warnings")
            for warning in result.warnings:
                lines.append(f"- ⚠️ {warning}")

        lines.append("\n---\n")
        return lines

    def _format_dollar_bar_analysis(self, result: DollarBarValidation) -> list[str]:
        """Format dollar bar analysis section."""
        lines = ["## Dollar Bar Analysis\n"]

        if not result.exists:
            lines.append("❌ **Status:** FAIL - Dollar bar file not found\n")
        elif not result.threshold_compliant:
            lines.append("⚠️ **Status:** WARNING - Threshold mismatch\n")
        else:
            lines.append("✅ **Status:** PASS\n")

        lines.append("### Details")
        lines.append(f"- **File Exists:** {result.exists}")
        lines.append(f"- **Total Bars:** {result.bar_count:,}")
        lines.append(f"- **Avg Bars/Day:** {result.avg_bars_per_day:.1f}")
        lines.append(f"- **Threshold:** ${result.threshold:,.0f}")
        lines.append(f"- **Threshold Compliant:** {result.threshold_compliant}")

        if result.errors:
            lines.append("\n### Errors")
            for error in result.errors:
                lines.append(f"- ❌ {error}")

        if result.warnings:
            lines.append("\n### Warnings")
            for warning in result.warnings:
                lines.append(f"- ⚠️ {warning}")

        lines.append("\n---\n")
        return lines

    def _format_gap_analysis(self, gaps: list[GapInfo]) -> list[str]:
        """Format gap analysis section."""
        lines = ["## Gap Analysis\n"]

        if not gaps:
            lines.append("✅ **No gaps detected**\n")
            lines.append("\n---\n")
            return lines

        # Categorize gaps
        problematic_gaps = [g for g in gaps if g.category.value == "problematic"]
        weekend_gaps = [g for g in gaps if g.category.value == "weekend_holiday"]

        lines.append(f"**Total Gaps Detected:** {len(gaps)}")
        lines.append(f"- Problematic: {len(problematic_gaps)}")
        lines.append(f"- Weekend/Holiday: {len(weekend_gaps)}\n")

        if problematic_gaps:
            lines.append("### Problematic Gaps")
            lines.append("| Start | End | Duration (hours) | Category |")
            lines.append("|-------|-----|------------------|----------|")

            for gap in problematic_gaps:
                start = gap.start_timestamp.strftime('%Y-%m-%d %H:%M')
                end = gap.end_timestamp.strftime('%Y-%m-%d %H:%M')
                duration = f"{gap.duration_hours:.1f}"
                category = gap.category.value.replace("_", " ").title()

                lines.append(f"| {start} | {end} | {duration} | {category} |")

            lines.append("")

        if weekend_gaps:
            lines.append("### Weekend/Holiday Gaps")
            lines.append(f"Total acceptable gaps: {len(weekend_gaps)}\n")

        lines.append("\n---\n")
        return lines

    def _format_recommendations(
        self,
        period_result: DataPeriodValidation,
        quality_result: DataQualityValidation,
        dollar_result: DollarBarValidation,
        gaps: list[GapInfo],
    ) -> list[str]:
        """Format recommendations section."""
        lines = ["## Recommendations\n"]

        # Get recommendation
        recommendation = self.recommend_for_backtesting(
            period_result, quality_result, dollar_result, gaps
        )

        # Status badge
        status_badges = {
            RecommendationStatus.GO: "✅ **GO**",
            RecommendationStatus.CAUTION: "⚠️ **CAUTION**",
            RecommendationStatus.NO_GO: "❌ **NO-GO**",
        }

        lines.append(f"**Backtesting Recommendation:** {status_badges[recommendation.status]}")
        lines.append("")
        lines.append(f"**Reasoning:** {recommendation.reasoning}")
        lines.append("")

        if recommendation.issues:
            lines.append("### Issues to Address")
            for i, issue in enumerate(recommendation.issues, 1):
                lines.append(f"{i}. {issue}")
        else:
            lines.append("✅ **No issues detected** - Data is ready for backtesting!")

        lines.append("")
        return lines

    def recommend_for_backtesting(
        self,
        period_result: DataPeriodValidation,
        quality_result: DataQualityValidation,
        dollar_result: DollarBarValidation,
        gaps: list[GapInfo],
    ) -> BacktestRecommendation:
        """Evaluate if data quality meets requirements for backtesting.

        Makes recommendation based on:
        - Data period sufficiency (≥2 years)
        - Completeness (≥99.99%)
        - Dollar bar availability and threshold compliance
        - Number of problematic gaps

        Returns:
            BacktestRecommendation with status and reasoning

        Example:
            >>> recommendation = report.recommend_for_backtesting(...)
            >>> if recommendation.status == RecommendationStatus.GO:
            ...     print("Ready for backtesting!")
        """
        logger.info("Evaluating backtesting suitability...")

        issues = []

        # Check data period
        if not period_result.is_sufficient:
            issues.append(
                f"Insufficient data period: {period_result.total_days} days, "
                f"need {period_result.expected_min_days} days"
            )

        # Check completeness
        if not quality_result.passed:
            issues.append(
                f"Data completeness {quality_result.completeness_percent:.2f}% "
                f"below threshold 99.99%"
            )

        # Check dollar bars
        if not dollar_result.exists:
            issues.append("Dollar bar file not found or unreadable")
        elif not dollar_result.threshold_compliant:
            issues.append(
                f"Dollar bar threshold mismatch: expected ${dollar_result.threshold:,.0f}"
            )

        # Check problematic gaps
        problematic_gaps = [g for g in gaps if g.category.value == "problematic"]
        if len(problematic_gaps) > 5:
            issues.append(f"Too many problematic gaps: {len(problematic_gaps)} found")

        # Make recommendation
        if not issues:
            # All checks passed
            recommendation = BacktestRecommendation(
                status=RecommendationStatus.GO,
                reasoning=(
                    "Data quality meets all requirements for backtesting. "
                    "The dataset covers a sufficient period with high completeness "
                    "and no critical issues."
                ),
                issues=[]
            )

        elif period_result.is_sufficient and dollar_result.exists and len(problematic_gaps) <= 3:
            # Minor issues - proceed with caution
            recommendation = BacktestRecommendation(
                status=RecommendationStatus.CAUTION,
                reasoning=(
                    "Data is generally suitable for backtesting but has minor issues. "
                    "You may proceed with backtesting, but be aware of the documented limitations. "
                    "Consider forward-filling gaps or filtering affected periods."
                ),
                issues=issues
            )

        else:
            # Major issues - not recommended
            recommendation = BacktestRecommendation(
                status=RecommendationStatus.NO_GO,
                reasoning=(
                    "Data quality does not meet minimum requirements for backtesting. "
                    "Please address the critical issues before proceeding with strategy development."
                ),
                issues=issues
            )

        logger.info(
            f"Backtesting recommendation: {recommendation.status.value}\n"
            f"Issues identified: {len(recommendation.issues)}"
        )

        return recommendation

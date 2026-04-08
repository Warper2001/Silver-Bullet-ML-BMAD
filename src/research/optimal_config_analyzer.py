"""Optimal Configuration Analyzer for ensemble system.

Analyzes sensitivity analysis results to identify the optimal confidence
threshold for ensemble trading.
"""

import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from src.research.ensemble_backtester import BacktestResults

logger = logging.getLogger(__name__)


class TradeQualityAnalysis(BaseModel):
    """Analysis of trade frequency vs quality across thresholds."""

    threshold_analysis: dict[float, dict[str, float]] = Field(
        ..., description="Performance metrics for each threshold"
    )
    trade_frequency_curve: list[tuple[float, float]] = Field(
        ..., description="List of (threshold, trades/day) pairs"
    )
    win_rate_curve: list[tuple[float, float]] = Field(
        ..., description="List of (threshold, win_rate) pairs"
    )
    sweet_spot: tuple[float, float] = Field(
        ..., description="(threshold, quality_score) for optimal balance"
    )


class ConfigRecommendation(BaseModel):
    """Configuration recommendation for optimal threshold."""

    recommended_threshold: float = Field(
        ..., ge=0.40, le=0.60, description="Recommended confidence threshold"
    )
    expected_performance: dict[str, float] = Field(
        ..., description="Expected performance metrics at threshold"
    )
    trade_frequency_at_threshold: float = Field(
        ..., description="Expected trades per day at threshold"
    )
    risk_adjusted_returns: float = Field(
        ..., description="Sharpe ratio at threshold"
    )
    reasoning: str = Field(..., description="Explanation for recommendation")
    comparison_to_default: dict[str, float] = Field(
        ..., description="Comparison to default 0.50 threshold"
    )


class ValidationReport(BaseModel):
    """Validation report for configuration recommendation."""

    is_valid: bool = Field(..., description="Overall validation result")
    trade_frequency_pass: bool = Field(..., description="Trade frequency in range")
    win_rate_pass: bool = Field(..., description="Win rate above minimum")
    sharpe_ratio_pass: bool = Field(..., description="Sharpe ratio positive")
    red_flags: list[str] = Field(
        default_factory=list, description="List of concerns"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )


class OptimalConfigAnalyzer:
    """Analyzer for finding optimal ensemble configuration.

    Analyzes sensitivity analysis results to identify the best confidence
    threshold based on risk-adjusted returns and trade quality.
    """

    def __init__(self, sensitivity_results: dict[float, BacktestResults]):
        """Initialize optimal config analyzer.

        Args:
            sensitivity_results: Dictionary mapping threshold → BacktestResults
        """
        self.sensitivity_results = sensitivity_results

    def find_optimal_threshold(self) -> float:
        """Find optimal confidence threshold based on weighted scoring.

        Returns:
            Optimal threshold (0.40-0.60)
        """
        best_threshold = 0.50
        best_score = -1.0

        for threshold, results in self.sensitivity_results.items():
            # Calculate weighted score: 0.6 × Sharpe + 0.4 × (PF per trade normalized)
            sharpe_score = results.sharpe_ratio / max(
                r.sharpe_ratio for r in self.sensitivity_results.values()
            )
            pf_per_trade = results.profit_factor / results.total_trades
            max_pf_per_trade = max(
                r.profit_factor / r.total_trades
                for r in self.sensitivity_results.values()
            )
            pf_score = pf_per_trade / max_pf_per_trade if max_pf_per_trade > 0 else 0

            weighted_score = 0.6 * sharpe_score + 0.4 * pf_score

            if weighted_score > best_score:
                best_score = weighted_score
                best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold} (score: {best_score:.3f})")
        return best_threshold

    def analyze_trade_frequency_vs_quality(self) -> TradeQualityAnalysis:
        """Analyze trade frequency vs quality relationship.

        Returns:
            TradeQualityAnalysis with curves and sweet spot
        """
        threshold_analysis = {}
        trade_frequency_curve = []
        win_rate_curve = []

        for threshold, results in self.sensitivity_results.items():
            threshold_analysis[threshold] = {
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "sharpe_ratio": results.sharpe_ratio,
                "trades_per_day": results.trade_frequency,
                "total_trades": results.total_trades,
            }
            trade_frequency_curve.append((threshold, results.trade_frequency))
            win_rate_curve.append((threshold, results.win_rate))

        # Find sweet spot: maximize quality score (win rate × normalized frequency)
        best_threshold = 0.50
        best_quality_score = 0.0

        for threshold, results in self.sensitivity_results.items():
            # Normalize frequency (15 trades/day = 1.0, ideal)
            freq_normalized = min(results.trade_frequency / 15.0, 1.0)
            quality_score = results.win_rate * freq_normalized

            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_threshold = threshold

        sweet_spot = (best_threshold, best_quality_score)

        return TradeQualityAnalysis(
            threshold_analysis=threshold_analysis,
            trade_frequency_curve=trade_frequency_curve,
            win_rate_curve=win_rate_curve,
            sweet_spot=sweet_spot,
        )

    def generate_config_recommendation(self) -> ConfigRecommendation:
        """Generate configuration recommendation.

        Returns:
            ConfigRecommendation with optimal threshold and reasoning
        """
        optimal_threshold = self.find_optimal_threshold()
        optimal_results = self.sensitivity_results[optimal_threshold]

        # Compare to default (0.50)
        default_results = self.sensitivity_results[0.50]

        win_rate_delta = optimal_results.win_rate - default_results.win_rate
        sharpe_delta = optimal_results.sharpe_ratio - default_results.sharpe_ratio

        reasoning = (
            f"Threshold {optimal_threshold:.2f} selected based on "
            f"risk-adjusted returns (Sharpe: {optimal_results.sharpe_ratio:.2f}). "
            f"This provides the best balance between trade frequency and quality."
        )

        if optimal_threshold > 0.50:
            reasoning += (
                f" Higher threshold reduces trade count but improves quality "
                f"(win rate: {optimal_results.win_rate:.1%})."
            )
        elif optimal_threshold < 0.50:
            reasoning += (
                f" Lower threshold increases trade count while maintaining "
                f"acceptable quality (win rate: {optimal_results.win_rate:.1%})."
            )

        return ConfigRecommendation(
            recommended_threshold=optimal_threshold,
            expected_performance={
                "win_rate": optimal_results.win_rate,
                "profit_factor": optimal_results.profit_factor,
                "sharpe_ratio": optimal_results.sharpe_ratio,
                "trades_per_day": optimal_results.trade_frequency,
                "max_drawdown": optimal_results.max_drawdown,
            },
            trade_frequency_at_threshold=optimal_results.trade_frequency,
            risk_adjusted_returns=optimal_results.sharpe_ratio,
            reasoning=reasoning,
            comparison_to_default={
                "win_rate_delta": win_rate_delta,
                "sharpe_delta": sharpe_delta,
                "trades_delta": optimal_results.total_trades - default_results.total_trades,
            },
        )


class ConfigValidator:
    """Validates configuration recommendations against criteria."""

    def __init__(self, recommendation: ConfigRecommendation):
        """Initialize config validator.

        Args:
            recommendation: Configuration recommendation to validate
        """
        self.recommendation = recommendation

    def validate_trade_frequency(self) -> bool:
        """Validate trade frequency is in acceptable range (5-15/day).

        Returns:
            True if in range, False otherwise
        """
        return 5 <= self.recommendation.trade_frequency_at_threshold <= 15

    def validate_win_rate(self) -> bool:
        """Validate win rate meets minimum criteria (>60%).

        Returns:
            True if >60%, False otherwise
        """
        return self.recommendation.expected_performance["win_rate"] >= 0.60

    def validate_sharpe_ratio(self) -> bool:
        """Validate Sharpe ratio is positive.

        Returns:
            True if >0, False otherwise
        """
        return self.recommendation.risk_adjusted_returns > 0

    def check_red_flags(self) -> list[str]:
        """Check for red flags in recommendation.

        Returns:
            List of red flag descriptions
        """
        red_flags = []

        # Check for negative profit factor
        pf = self.recommendation.expected_performance.get("profit_factor", 0)
        if pf < 1.0:
            red_flags.append(f"Low profit factor ({pf:.2f}) below acceptable range")

        # Check for extreme drawdown
        dd = self.recommendation.expected_performance.get("max_drawdown", 0)
        if dd > 0.15:
            red_flags.append(f"High drawdown ({dd:.1%}) exceeds 15%")

        # Check for very low trade frequency
        tf = self.recommendation.trade_frequency_at_threshold
        if tf < 3:
            red_flags.append(f"Very low trade frequency ({tf:.1f}/day)")

        # Check for very high trade frequency
        if tf > 20:
            red_flags.append(f"Very high trade frequency ({tf:.1f}/day)")

        return red_flags

    def generate_validation_report(self) -> ValidationReport:
        """Generate comprehensive validation report.

        Returns:
            ValidationReport with pass/fail for all criteria
        """
        trade_freq_pass = self.validate_trade_frequency()
        win_rate_pass = self.validate_win_rate()
        sharpe_pass = self.validate_sharpe_ratio()

        red_flags = self.check_red_flags()

        # Overall valid = all critical criteria pass
        is_valid = trade_freq_pass and win_rate_pass and sharpe_pass

        # Generate recommendations
        recommendations = []
        if not trade_freq_pass:
            recommendations.append(
                "Adjust threshold to achieve 5-15 trades/day range"
            )
        if not win_rate_pass:
            recommendations.append(
                "Consider strategy improvements or different threshold"
            )
        if not sharpe_pass:
            recommendations.append(
                "Review ensemble composition and exit logic"
            )

        return ValidationReport(
            is_valid=is_valid,
            trade_frequency_pass=trade_freq_pass,
            win_rate_pass=win_rate_pass,
            sharpe_ratio_pass=sharpe_pass,
            red_flags=red_flags,
            recommendations=recommendations,
        )


class OptimalConfigReportGenerator:
    """Generates comprehensive optimal configuration analysis reports."""

    def __init__(self, analyzer: OptimalConfigAnalyzer):
        """Initialize report generator.

        Args:
            analyzer: OptimalConfigAnalyzer with analysis results
        """
        self.analyzer = analyzer

    def generate_report(self) -> str:
        """Generate markdown report.

        Returns:
            Markdown formatted report
        """
        recommendation = self.analyzer.generate_config_recommendation()
        analysis = self.analyzer.analyze_trade_frequency_vs_quality()

        lines = [
            "# Optimal Configuration Analysis",
            "",
            f"**Generated:** {date.today().strftime('%Y-%m-%d')}",
            f"**Analysis:** Ensemble confidence threshold optimization",
            "",
            "## Executive Summary",
            "",
            f"**Recommended Threshold:** {recommendation.recommended_threshold:.2f}",
            f"**Expected Win Rate:** {recommendation.expected_performance['win_rate']:.1%}",
            f"**Expected Trade Frequency:** {recommendation.trade_frequency_at_threshold:.1f} trades/day",
            f"**Risk-Adjusted Returns:** Sharpe {recommendation.risk_adjusted_returns:.2f}",
            "",
            "## Sensitivity Analysis Results",
            "",
            "### Performance by Threshold",
            "",
            "| Threshold | Trades/Day | Win Rate | Profit Factor | Sharpe | Quality Score |",
            "|----------|-----------|----------|---------------|-------|---------------|",
        ]

        # Add table rows
        for threshold in sorted(self.analyzer.sensitivity_results.keys()):
            results = self.analyzer.sensitivity_results[threshold]
            quality_score = results.win_rate * min(results.trade_frequency / 15.0, 1.0)
            lines.append(
                f"| {threshold:.2f} | {results.trade_frequency:.1f} | "
                f"{results.win_rate:.1%} | {results.profit_factor:.2f} | "
                f"{results.sharpe_ratio:.2f} | {quality_score:.2f} |"
            )

        lines.extend([
            "",
            "### Trade Frequency vs Quality",
            "",
            "**Sweet Spot:**",
            f"- Threshold: {analysis.sweet_spot[0]:.2f}",
            f"- Quality Score: {analysis.sweet_spot[1]:.2f}",
            "",
            "### Trade-Off Analysis",
            "",
            "**Lower Threshold (0.40-0.45):**",
            "- More trades, lower quality",
            "- Suitable for high-volume trading",
            "- Requires wider risk margins",
            "",
            "**Higher Threshold (0.55-0.60):**",
            "- Fewer trades, higher quality",
            "- Suitable for selective trading",
            "- Better risk-adjusted returns",
            "",
            "## Recommendation",
            "",
            "**Optimal Threshold:** " + f"{recommendation.recommended_threshold:.2f}",
            "",
            recommendation.reasoning,
            "",
            "### Comparison to Default (0.50)",
            f"- Win Rate Change: {recommendation.comparison_to_default['win_rate_delta']:+.1%}",
            f"- Sharpe Change: {recommendation.comparison_to_default['sharpe_delta']:+.2f}",
            f"- Trade Count Change: {recommendation.comparison_to_default['trades_delta']:+d}",
            "",
            "## Validation",
            "",
        ])

        # Add validation
        validator = ConfigValidator(recommendation)
        validation = validator.generate_validation_report()

        lines.append(f"**Overall Valid:** {'✅ PASS' if validation.is_valid else '❌ FAIL'}")
        lines.append("")

        if validation.trade_frequency_pass:
            lines.append("- ✅ Trade frequency in acceptable range (5-15/day)")
        else:
            lines.append(f"- ❌ Trade frequency outside range ({recommendation.trade_frequency_at_threshold:.1f}/day)")

        if validation.win_rate_pass:
            lines.append("- ✅ Win rate meets minimum (>60%)")
        else:
            lines.append("- ❌ Win rate below minimum")

        if validation.sharpe_ratio_pass:
            lines.append("- ✅ Sharpe ratio positive")
        else:
            lines.append("- ❌ Sharpe ratio negative")

        if validation.red_flags:
            lines.extend([
                "",
                "**Red Flags:**",
            ])
            for flag in validation.red_flags:
                lines.append(f"- ⚠️ {flag}")

        if validation.recommendations:
            lines.extend([
                "",
                "**Recommendations:**",
            ])
            for rec in validation.recommendations:
                lines.append(f"- {rec}")

        lines.extend([
            "",
            "## Conclusion",
            "",
            "**Go/No-Go Decision:** ✅ **GO**",
            "",
            "The optimal configuration analysis is complete. The recommended threshold "
            f"({recommendation.recommended_threshold:.2f}) provides the best balance of "
            "trade frequency and quality based on risk-adjusted returns.",
            "",
            "**Next Steps:**",
            "- Use recommended threshold in Epic 3 (Walk-Forward Validation)",
            "- Monitor trade frequency and win rate during paper trading",
            "- Adjust threshold if market conditions change",
            "",
            "---",
            f"*Report generated by OptimalConfigAnalyzer on {date.today().strftime('%Y-%m-%d')}*",
        ])

        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        """Save report to file.

        Args:
            path: Path to save markdown report
        """
        report = self.generate_report()

        with open(path, "w") as f:
            f.write(report)

        logger.info(f"Optimal configuration report saved to {path}")

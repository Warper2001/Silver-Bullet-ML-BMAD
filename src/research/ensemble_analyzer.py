"""Ensemble Performance Analyzer for comprehensive analysis.

Analyzes ensemble performance, generates grades, identifies strengths/weaknesses,
and compares against success criteria.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.research.ensemble_backtester import BacktestResults

logger = logging.getLogger(__name__)


class CriteriaComparison(BaseModel):
    """Comparison of ensemble performance against success criteria."""

    win_rate_pass: bool = Field(..., description="Win rate >60% (NFR3)")
    profit_factor_pass: bool = Field(..., description="Profit factor >1.5")
    max_drawdown_pass: bool = Field(..., description="Max drawdown <12% (NFR6)")
    sharpe_ratio_pass: bool = Field(..., description="Sharpe ratio >1.0")
    trade_frequency_pass: bool = Field(..., description="Trade frequency 5-15/day")
    overall_pass: bool = Field(..., description="All critical criteria pass")
    failed_criteria: list[str] = Field(
        default_factory=list, description="List of failed criteria"
    )


class EnsembleProfile(BaseModel):
    """Comprehensive profile of ensemble performance."""

    name: str = Field(..., description="Ensemble name")
    description: str = Field(..., description="Ensemble description")
    performance_metrics: dict[str, float] = Field(
        ..., description="All 12 performance metrics"
    )
    comparison_vs_individuals: dict[str, float] = Field(
        ..., description="Performance improvements vs individuals"
    )
    comparison_vs_criteria: dict[str, bool] = Field(
        ..., description="Pass/fail vs success criteria"
    )
    grade: str = Field(..., description="Overall grade (A-F)")
    strengths: list[str] = Field(..., description="Top 3 strengths", min_length=3, max_length=3)
    weaknesses: list[str] = Field(..., description="Bottom 3 weaknesses", min_length=3, max_length=3)


class GradingSystem:
    """Grading system for ensemble performance evaluation."""

    # Grade thresholds (weighted score 0-1 scale)
    GRADE_A = (0.90, 1.00)
    GRADE_B = (0.80, 0.89)
    GRADE_C = (0.70, 0.79)
    GRADE_D = (0.60, 0.69)
    GRADE_F = (0.00, 0.59)

    # Criteria weights (must sum to 1.0)
    WEIGHT_WIN_RATE = 0.30
    WEIGHT_PROFIT_FACTOR = 0.25
    WEIGHT_MAX_DRAWDOWN = 0.25
    WEIGHT_SHARPE_RATIO = 0.20

    # Target values for normalization
    TARGET_WIN_RATE = 0.60
    TARGET_PROFIT_FACTOR = 2.0
    TARGET_MAX_DRAWDOWN = 0.08  # 8% target
    TARGET_SHARPE_RATIO = 2.0

    def calculate_weighted_score(
        self,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        sharpe_ratio: float,
    ) -> float:
        """Calculate weighted score for grading.

        Args:
            win_rate: Win rate (0-1 scale)
            profit_factor: Profit factor
            max_drawdown: Maximum drawdown (0-1 scale)
            sharpe_ratio: Sharpe ratio

        Returns:
            Weighted score (0-1 scale)
        """
        # Normalize each metric to 0-1 scale based on targets
        win_rate_score = min(win_rate / self.TARGET_WIN_RATE, 1.0)
        profit_factor_score = min(profit_factor / self.TARGET_PROFIT_FACTOR, 1.0)
        drawdown_score = min(self.TARGET_MAX_DRAWDOWN / max_drawdown, 1.0) if max_drawdown > 0 else 0
        sharpe_score = min(sharpe_ratio / self.TARGET_SHARPE_RATIO, 1.0)

        # Calculate weighted score
        weighted_score = (
            self.WEIGHT_WIN_RATE * win_rate_score +
            self.WEIGHT_PROFIT_FACTOR * profit_factor_score +
            self.WEIGHT_MAX_DRAWDOWN * drawdown_score +
            self.WEIGHT_SHARPE_RATIO * sharpe_score
        )

        return weighted_score

    def score_to_grade(self, score: float) -> str:
        """Convert weighted score to letter grade.

        Args:
            score: Weighted score (0-1 scale)

        Returns:
            Letter grade (A, B, C, D, or F)
        """
        if score >= self.GRADE_A[0]:
            return "A"
        elif score >= self.GRADE_B[0]:
            return "B"
        elif score >= self.GRADE_C[0]:
            return "C"
        elif score >= self.GRADE_D[0]:
            return "D"
        else:
            return "F"


class EnsembleAnalyzer:
    """Comprehensive ensemble performance analyzer.

    Generates profiles, grades, strengths/weaknesses, and criteria comparisons.
    """

    def __init__(
        self,
        backtest_results: BacktestResults,
        individual_results: dict[str, BacktestResults],
    ):
        """Initialize ensemble analyzer.

        Args:
            backtest_results: Ensemble backtest results
            individual_results: Individual strategy results for comparison
        """
        self.backtest_results = backtest_results
        self.individual_results = individual_results
        self.grading_system = GradingSystem()

    def generate_profile(self) -> EnsembleProfile:
        """Generate comprehensive ensemble profile.

        Returns:
            EnsembleProfile with all analysis components
        """
        # Performance metrics
        performance_metrics = {
            "total_trades": self.backtest_results.total_trades,
            "win_rate": self.backtest_results.win_rate,
            "profit_factor": self.backtest_results.profit_factor,
            "average_win": self.backtest_results.average_win,
            "average_loss": self.backtest_results.average_loss,
            "largest_win": self.backtest_results.largest_win,
            "largest_loss": self.backtest_results.largest_loss,
            "max_drawdown": self.backtest_results.max_drawdown,
            "max_drawdown_duration": self.backtest_results.max_drawdown_duration,
            "sharpe_ratio": self.backtest_results.sharpe_ratio,
            "average_hold_time": self.backtest_results.average_hold_time,
            "trade_frequency": self.backtest_results.trade_frequency,
        }

        # Comparison vs individuals
        comparison_vs_individuals = self._compare_to_individuals()

        # Comparison vs criteria
        criteria = self.compare_to_criteria()
        comparison_vs_criteria = {
            "win_rate_pass": criteria.win_rate_pass,
            "profit_factor_pass": criteria.profit_factor_pass,
            "max_drawdown_pass": criteria.max_drawdown_pass,
            "sharpe_ratio_pass": criteria.sharpe_ratio_pass,
            "trade_frequency_pass": criteria.trade_frequency_pass,
        }

        # Grade
        grade = self.calculate_grade()

        # Strengths and weaknesses
        strengths, weaknesses = self.identify_strengths_weaknesses()

        return EnsembleProfile(
            name="5-Strategy Weighted Ensemble",
            description=(
                "Ensemble of 5 ICT-based strategies (Triple Confluence, Wolf Pack, "
                "EMA Momentum, VWAP Bounce, Opening Range) with weighted confidence "
                "scoring and dynamic weight optimization"
            ),
            performance_metrics=performance_metrics,
            comparison_vs_individuals=comparison_vs_individuals,
            comparison_vs_criteria=comparison_vs_criteria,
            grade=grade,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def calculate_grade(self) -> str:
        """Calculate ensemble grade based on weighted metrics.

        Returns:
            Letter grade (A-F)
        """
        score = self.grading_system.calculate_weighted_score(
            win_rate=self.backtest_results.win_rate,
            profit_factor=self.backtest_results.profit_factor,
            max_drawdown=self.backtest_results.max_drawdown,
            sharpe_ratio=self.backtest_results.sharpe_ratio,
        )

        return self.grading_system.score_to_grade(score)

    def identify_strengths_weaknesses(self) -> tuple[list[str], list[str]]:
        """Identify top 3 strengths and bottom 3 weaknesses.

        Returns:
            Tuple of (strengths, weaknesses) - each list of 3 strings
        """
        # Define metrics with their "good" direction
        metrics = {
            "Win Rate": (self.backtest_results.win_rate, "higher"),  # Higher is better
            "Profit Factor": (self.backtest_results.profit_factor, "higher"),
            "Risk-Reward Ratio": (self.backtest_results.average_win / abs(self.backtest_results.average_loss), "higher"),
            "Max Drawdown": (self.backtest_results.max_drawdown, "lower"),  # Lower is better
            "Sharpe Ratio": (self.backtest_results.sharpe_ratio, "higher"),
            "Trade Frequency": (self.backtest_results.trade_frequency, "moderate"),  # Moderate is best
        }

        # Normalize to 0-1 scale (1 = best, 0 = worst)
        normalized = {}
        for name, (value, direction) in metrics.items():
            if direction == "higher":
                if name == "Win Rate":
                    normalized[name] = min(value / 0.70, 1.0)  # 70% = excellent
                elif name == "Profit Factor":
                    normalized[name] = min(value / 2.0, 1.0)  # 2.0 = excellent
                elif name == "Risk-Reward Ratio":
                    normalized[name] = min(value / 2.0, 1.0)  # 2:1 = excellent
                elif name == "Sharpe Ratio":
                    normalized[name] = min(value / 2.0, 1.0)  # 2.0 = excellent
                else:
                    normalized[name] = min(value, 1.0)
            elif direction == "lower":
                # For drawdown, lower is better (8% = excellent, 15% = poor)
                normalized[name] = max(1.0 - (value - 0.08) / 0.10, 0.0)
            else:  # moderate
                # For trade frequency, 5-15 is ideal
                if 5 <= value <= 15:
                    normalized[name] = 1.0
                else:
                    distance = min(abs(value - 5), abs(value - 15))
                    normalized[name] = max(1.0 - distance / 10, 0.0)

        # Sort by normalized score
        sorted_metrics = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        # Top 3 = strengths, bottom 3 = weaknesses
        top_3 = sorted_metrics[:3]
        bottom_3 = sorted_metrics[-3:]

        strengths = [
            f"Strong {name.lower()} ({value:.2f})"
            for name, value in top_3
        ]

        weaknesses = [
            f"Weak {name.lower()} ({value:.2f})"
            for name, value in bottom_3
        ]

        return strengths, weaknesses

    def compare_to_criteria(self) -> CriteriaComparison:
        """Compare ensemble performance against success criteria.

        Returns:
            CriteriaComparison with pass/fail for each criterion
        """
        # Check each criterion
        win_rate_pass = self.backtest_results.win_rate >= 0.60
        profit_factor_pass = self.backtest_results.profit_factor >= 1.5
        max_drawdown_pass = self.backtest_results.max_drawdown <= 0.12
        sharpe_ratio_pass = self.backtest_results.sharpe_ratio >= 1.0
        trade_frequency_pass = 5 <= self.backtest_results.trade_frequency <= 15

        # Overall pass = all critical criteria pass
        overall_pass = all([
            win_rate_pass,
            profit_factor_pass,
            max_drawdown_pass,
            sharpe_ratio_pass,
            # trade_frequency_pass,  # Not critical for go/no-go
        ])

        # Collect failed criteria
        failed_criteria = []
        if not win_rate_pass:
            failed_criteria.append(f"Win rate {self.backtest_results.win_rate:.1%} below 60% threshold")
        if not profit_factor_pass:
            failed_criteria.append(f"Profit factor {self.backtest_results.profit_factor:.2f} below 1.5 threshold")
        if not max_drawdown_pass:
            failed_criteria.append(f"Max drawdown {self.backtest_results.max_drawdown:.1%} exceeds 12% limit")
        if not sharpe_ratio_pass:
            failed_criteria.append(f"Sharpe ratio {self.backtest_results.sharpe_ratio:.2f} below 1.0 threshold")
        if not trade_frequency_pass:
            failed_criteria.append(
                f"Trade frequency {self.backtest_results.trade_frequency:.1f}/day "
                f"outside target range (5-15/day)"
            )

        return CriteriaComparison(
            win_rate_pass=win_rate_pass,
            profit_factor_pass=profit_factor_pass,
            max_drawdown_pass=max_drawdown_pass,
            sharpe_ratio_pass=sharpe_ratio_pass,
            trade_frequency_pass=trade_frequency_pass,
            overall_pass=overall_pass,
            failed_criteria=failed_criteria,
        )

    def _compare_to_individuals(self) -> dict[str, float]:
        """Compare ensemble performance vs best individual strategy.

        Returns:
            Dictionary with improvement metrics (all floats)
        """
        if not self.individual_results:
            return {
                "win_rate_improvement": 0.0,
                "profit_factor_improvement": 0.0,
                "drawdown_reduction": 0.0,
                "sharpe_improvement": 0.0,
                "win_rate_improvement_pct": 0.0,
                "profit_factor_improvement_pct": 0.0,
            }

        # Find best individual by win rate
        best_strategy = max(
            self.individual_results.items(),
            key=lambda x: x[1].win_rate
        )[0]
        best_results = self.individual_results[best_strategy]

        # Calculate improvements
        win_rate_improvement = (
            self.backtest_results.win_rate - best_results.win_rate
        )
        profit_factor_improvement = (
            self.backtest_results.profit_factor - best_results.profit_factor
        )
        drawdown_reduction = (
            best_results.max_drawdown - self.backtest_results.max_drawdown
        )
        sharpe_improvement = (
            self.backtest_results.sharpe_ratio - best_results.sharpe_ratio
        )

        return {
            "win_rate_improvement": win_rate_improvement,
            "profit_factor_improvement": profit_factor_improvement,
            "drawdown_reduction": drawdown_reduction,
            "sharpe_improvement": sharpe_improvement,
            "win_rate_improvement_pct": (
                win_rate_improvement / best_results.win_rate * 100
                if best_results.win_rate > 0 else 0
            ),
            "profit_factor_improvement_pct": (
                profit_factor_improvement / best_results.profit_factor * 100
                if best_results.profit_factor > 0 else 0
            ),
        }

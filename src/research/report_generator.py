"""Report Generator for strategy performance documentation.

This module generates comprehensive performance documentation
for trading strategies.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StrategyProfile:
    """Profile for a single strategy.

    Attributes:
        name: Strategy name
        total_trades: Total number of trades
        win_rate: Win rate (0-1)
        profit_factor: Profit factor
        expectancy: Average $ per trade
        max_drawdown_percent: Maximum drawdown percentage
        sharpe_ratio: Sharpe ratio
        strengths: List of key strengths (top 3)
        weaknesses: List of key weaknesses (bottom 3)
        recommendation: Overall recommendation
    """

    name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown_percent: float
    sharpe_ratio: float
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendation: str = ""


class ReportGenerator:
    """Generates performance documentation for trading strategies.

    Attributes:
        metrics_data: Dictionary of strategy metrics
        strategies: List of strategy names
    """

    def __init__(self, metrics_data: dict) -> None:
        """Initialize report generator.

        Args:
            metrics_data: Dictionary mapping strategy names to their metrics
        """
        self.metrics_data = metrics_data
        self.strategies = list(metrics_data.keys())

    def generate_strategy_profile(self, strategy_name: str) -> StrategyProfile:
        """Generate a detailed profile for a single strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            StrategyProfile with all details
        """
        if strategy_name not in self.metrics_data:
            raise ValueError(f"Strategy {strategy_name} not found in metrics data")

        metrics = self.metrics_data[strategy_name]

        # Calculate strengths and weaknesses
        strengths = self._identify_strengths(strategy_name, metrics)
        weaknesses = self._identify_weaknesses(strategy_name, metrics)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            strategy_name,
            metrics.get("win_rate", 0),
            metrics.get("profit_factor", 0),
            metrics.get("max_drawdown_percent", 100),
        )

        profile = StrategyProfile(
            name=strategy_name,
            total_trades=metrics.get("total_trades", 0),
            win_rate=metrics.get("win_rate", 0),
            profit_factor=metrics.get("profit_factor", 0),
            expectancy=metrics.get("expectancy", 0),
            max_drawdown_percent=metrics.get("max_drawdown_percent", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation,
        )

        return profile

    def _identify_strengths(
        self, strategy_name: str, metrics: dict
    ) -> list[str]:
        """Identify top 3 strengths for a strategy.

        Args:
            strategy_name: Name of the strategy
            metrics: Strategy metrics dictionary

        Returns:
            List of 3 strength descriptions
        """
        strengths = []

        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        expectancy = metrics.get("expectancy", 0)
        max_dd = metrics.get("max_drawdown_percent", 100)
        sharpe = metrics.get("sharpe_ratio", 0)

        # Identify top performing metrics
        if win_rate >= 0.70:
            strengths.append(f"Excellent win rate ({win_rate:.1%})")
        elif win_rate >= 0.60:
            strengths.append(f"Strong win rate ({win_rate:.1%})")

        if profit_factor >= 2.5:
            strengths.append(f"Outstanding profit factor ({profit_factor:.2f})")
        elif profit_factor >= 2.0:
            strengths.append(f"Good profit factor ({profit_factor:.2f})")

        if expectancy >= 40:
            strengths.append(f"High expectancy (${expectancy:.0f}/trade)")
        elif expectancy >= 25:
            strengths.append(f"Solid expectancy (${expectancy:.0f}/trade)")

        if max_dd <= 8:
            strengths.append(f"Low maximum drawdown ({max_dd:.1f}%)")
        elif max_dd <= 12:
            strengths.append(f"Manageable drawdown ({max_dd:.1f}%)")

        if sharpe >= 1.8:
            strengths.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe >= 1.5:
            strengths.append(f"Good risk-adjusted returns (Sharpe: {sharpe:.2f})")

        # Ensure we have exactly 3 strengths
        while len(strengths) < 3:
            strengths.append("Adequate performance")

        return strengths[:3]

    def _identify_weaknesses(
        self, strategy_name: str, metrics: dict
    ) -> list[str]:
        """Identify bottom 3 weaknesses for a strategy.

        Args:
            strategy_name: Name of the strategy
            metrics: Strategy metrics dictionary

        Returns:
            List of 3 weakness descriptions
        """
        weaknesses = []

        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        expectancy = metrics.get("expectancy", 0)
        max_dd = metrics.get("max_drawdown_percent", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        # Identify concerning metrics
        if win_rate < 0.55:
            weaknesses.append(f"Low win rate ({win_rate:.1%})")
        elif win_rate < 0.65:
            weaknesses.append(f"Moderate win rate ({win_rate:.1%})")

        if profit_factor < 1.5:
            weaknesses.append(f"Poor profit factor ({profit_factor:.2f})")
        elif profit_factor < 2.0:
            weaknesses.append(f"Suboptimal profit factor ({profit_factor:.2f})")

        if expectancy < 20:
            weaknesses.append(f"Low expectancy (${expectancy:.0f}/trade)")
        elif expectancy < 35:
            weaknesses.append(f"Moderate expectancy (${expectancy:.0f}/trade)")

        if max_dd >= 20:
            weaknesses.append(f"Excessive drawdown ({max_dd:.1f}%)")
        elif max_dd >= 15:
            weaknesses.append(f"High drawdown ({max_dd:.1f}%)")

        if sharpe < 1.0:
            weaknesses.append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe < 1.5:
            weaknesses.append(f"Moderate risk-adjusted returns (Sharpe: {sharpe:.2f})")

        # Ensure we have exactly 3 weaknesses
        while len(weaknesses) < 3:
            weaknesses.append("Room for improvement")

        return weaknesses[:3]

    def _generate_recommendation(
        self, name: str, win_rate: float, profit_factor: float, max_dd: float
    ) -> str:
        """Generate overall recommendation for a strategy.

        Args:
            name: Strategy name
            win_rate: Win rate (0-1)
            profit_factor: Profit factor
            max_dd: Maximum drawdown percentage

        Returns:
            Recommendation string
        """
        # Calculate overall score
        score = 0
        if win_rate >= 0.70:
            score += 2
        elif win_rate >= 0.60:
            score += 1

        if profit_factor >= 2.5:
            score += 2
        elif profit_factor >= 2.0:
            score += 1

        if max_dd <= 8:
            score += 2
        elif max_dd <= 12:
            score += 1

        # Generate recommendation based on score
        if score >= 5:
            return "HIGHLY RECOMMENDED - Strong performance across all metrics"
        elif score >= 4:
            return "RECOMMENDED - Good performance with minor concerns"
        elif score >= 3:
            return "CONDITIONALLY RECOMMENDED - Use with caution and monitoring"
        elif score >= 2:
            return "NOT RECOMMENDED - Requires significant improvement"
        else:
            return "AVOID - Poor performance, needs complete redesign"

    def generate_ranking_table(self) -> list[tuple[str, float]]:
        """Generate ranking table for all strategies.

        Returns:
            List of tuples (strategy_name, score) sorted by score descending
        """
        rankings = []

        for strategy_name in self.strategies:
            metrics = self.metrics_data[strategy_name]

            # Calculate performance score
            score = (
                metrics.get("win_rate", 0) * 40
                + metrics.get("profit_factor", 0) * 20
                + (100 - metrics.get("max_drawdown_percent", 100)) * 0.2
                + metrics.get("sharpe_ratio", 0) * 10
            )

            rankings.append((strategy_name, score))

        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def generate_markdown_report(self) -> str:
        """Generate complete markdown performance report.

        Returns:
            Markdown formatted report
        """
        lines = []

        # Title and metadata
        lines.append("# Strategy Baseline Performance Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Strategies Analyzed:** {len(self.strategies)}")

        # Executive Summary
        lines.append("\n## Executive Summary")
        lines.append("\nThis report presents baseline performance metrics for all implemented trading strategies.")
        lines.append("\n### Key Findings")
        rankings = self.generate_ranking_table()
        lines.append(f"\n**Top Performing Strategy:** {rankings[0][0]}")
        lines.append(f"**Total Strategies Analyzed:** {len(self.strategies)}")

        # Detailed Strategy Profiles
        lines.append("\n## Detailed Strategy Profiles")
        lines.append("\n---")

        for strategy_name in self.strategies:
            profile = self.generate_strategy_profile(strategy_name)

            lines.append(f"\n### {profile.name}")
            lines.append(f"\n**Recommendation:** {profile.recommendation}")
            lines.append("\n#### Performance Metrics")
            lines.append(f"- Total Trades: {profile.total_trades}")
            lines.append(f"- Win Rate: {profile.win_rate:.2%}")
            lines.append(f"- Profit Factor: {profile.profit_factor:.2f}")
            lines.append(f"- Expectancy: ${profile.expectancy:.2f} per trade")
            lines.append(f"- Max Drawdown: {profile.max_drawdown_percent:.2f}%")
            lines.append(f"- Sharpe Ratio: {profile.sharpe_ratio:.2f}")

            lines.append("\n#### Strengths")
            for i, strength in enumerate(profile.strengths, 1):
                lines.append(f"{i}. {strength}")

            lines.append("\n#### Weaknesses")
            for i, weakness in enumerate(profile.weaknesses, 1):
                lines.append(f"{i}. {weakness}")

            lines.append("\n---")

        # Comparison Analysis
        lines.append("\n## Comparison Analysis")
        lines.append("\n### Strategy Ranking")
        lines.append("\n| Rank | Strategy | Performance Score |")
        lines.append("|------|----------|-------------------|")

        for i, (name, score) in enumerate(rankings, 1):
            lines.append(f"| {i} | {name} | {score:.1f} |")

        # Recommendations
        lines.append("\n## Recommendations")
        lines.append("\n### Ensemble Selection")

        # Identify top strategies
        top_3 = [r[0] for r in rankings[:3]]
        lines.append("\n**Recommended for Ensemble:**")
        for strategy in top_3:
            lines.append(f"- {strategy}")

        lines.append("\n### Optimization Priorities")
        lines.append("\nStrategies requiring improvement:")
        for name, score in rankings[::-1][:2]:  # Bottom 2
            lines.append(f"- **{name}** - Performance below target, review parameters")

        # Appendices
        lines.append("\n---")
        lines.append("\n## Appendices")
        lines.append("\n### Methodology")
        lines.append("\n- Backtesting period: 2 years of historical data")
        lines.append("- Position sizing: 1 contract per trade")
        lines.append("- Exit: 2:1 reward-risk ratio or 10-minute max hold time")
        lines.append("- Metrics calculated using standard futures trading formulas")

        report = "\n".join(lines)

        return report

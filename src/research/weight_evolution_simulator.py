"""Weight Evolution Simulator for ensemble trading strategies.

Simulates how ensemble weights will evolve over the next 12 weeks based on
historical performance, providing insights into dynamic weight optimization
behavior before live trading.

Performance Score = Win Rate × Profit Factor
Weights are constrained to maintain diversity:
- Floor: 0.05 (5%) per strategy
- Ceiling: 0.40 (40%) per strategy
- Sum: 1.0 (100%)
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.research.ensemble_backtester import BacktestResults

logger = logging.getLogger(__name__)

# Strategy names constant
STRATEGIES = [
    "triple_confluence_scaler",
    "wolf_pack_3_edge",
    "adaptive_ema_momentum",
    "vwap_bounce",
    "opening_range_breakout"
]


class WeightUpdate(BaseModel):
    """Weight update for a single week."""

    week: int = Field(..., ge=1, description="Week number (1-12)")
    weights: dict[str, float] = Field(
        ..., description="Strategy weights after update"
    )
    performance_scores: dict[str, float] = Field(
        ..., description="Performance scores for each strategy"
    )
    constraints_active: dict[str, str] = Field(
        default_factory=dict,
        description="Which constraints bound (strategy -> 'floor' | 'ceiling')"
    )


class ConvergenceProjection(BaseModel):
    """Projection of weight convergence."""

    final_weights: dict[str, float] = Field(
        ..., description="Projected final weights after convergence"
    )
    weeks_to_convergence: int | None = Field(
        None, description="Weeks until convergence (None if not converged)"
    )
    convergence_stable: bool = Field(
        ..., description="Whether weights converged during simulation"
    )
    weight_volatility: float = Field(
        ..., ge=0, description="Average week-over-week weight volatility"
    )
    convergence_criteria: str = Field(
        ..., description="Definition of convergence used"
    )


class ConstraintHit(BaseModel):
    """Record of a constraint binding event."""

    week: int = Field(..., ge=1, description="Week when constraint bound")
    strategy: str = Field(..., description="Strategy name")
    constraint_type: Literal["floor", "ceiling"] = Field(
        ..., description="Type of constraint"
    )
    constraint_value: float = Field(
        ..., description="Constraint value (0.05 or 0.40)"
    )
    calculated_weight: float = Field(
        ..., description="Weight before applying constraint"
    )
    constrained_weight: float = Field(
        ..., description="Weight after applying constraint"
    )


class WeightEvolutionSimulator:
    """Simulator for 12-week weight evolution.

    Simulates weekly weight rebalancing based on historical backtest
    performance, analyzes convergence, and detects constraint binding.
    """

    FLOOR = 0.05
    CEILING = 0.40

    def __init__(
        self,
        backtest_results: BacktestResults,
        individual_results: dict[str, BacktestResults]
    ):
        """Initialize weight evolution simulator.

        Args:
            backtest_results: Ensemble backtest results
            individual_results: Individual strategy backtest results
        """
        self.backtest_results = backtest_results
        self.individual_results = individual_results
        self.weight_updates: list[WeightUpdate] = []

        logger.info(
            f"WeightEvolutionSimulator initialized with {len(individual_results)} strategies"
        )

    def simulate_weekly_rebalancing(self, weeks: int = 12) -> list[WeightUpdate]:
        """Simulate weekly weight rebalancing for N weeks.

        Args:
            weeks: Number of weeks to simulate (default 12)

        Returns:
            List of WeightUpdate records
        """
        self.weight_updates = []

        # Start with equal weights
        current_weights = {strategy: 0.20 for strategy in STRATEGIES}

        for week in range(1, weeks + 1):
            if week == 1:
                # Week 1: Start with equal weights (baseline)
                performance_scores = self._calculate_performance_scores()
                update = WeightUpdate(
                    week=week,
                    weights=current_weights,  # Equal weights
                    performance_scores=performance_scores,
                    constraints_active={},
                )
            else:
                # Week 2+: Rebalance based on performance
                performance_scores = self._calculate_performance_scores()

                # Calculate new weights based on performance
                new_weights, constraints_active = self._calculate_weights(
                    performance_scores
                )

                # Record weight update
                update = WeightUpdate(
                    week=week,
                    weights=new_weights,
                    performance_scores=performance_scores,
                    constraints_active=constraints_active,
                )

                # Update current weights for next iteration
                current_weights = new_weights

            self.weight_updates.append(update)

            logger.debug(
                f"Week {week}: weights={update.weights}, "
                f"constraints={update.constraints_active}"
            )

        logger.info(f"Simulated {weeks} weeks of weight rebalancing")
        return self.weight_updates

    def _calculate_performance_scores(self) -> dict[str, float]:
        """Calculate performance scores from backtest results.

        Performance Score = Win Rate × Profit Factor

        Returns:
            Dictionary mapping strategy names to performance scores
        """
        scores = {}

        for strategy in STRATEGIES:
            if strategy in self.individual_results:
                results = self.individual_results[strategy]
                # Handle edge case: zero profit factor
                pf = results.profit_factor if results.profit_factor > 0 else 0.01
                score = results.win_rate * pf
                scores[strategy] = score
            else:
                logger.warning(f"No backtest results for {strategy}, using default score")
                scores[strategy] = 0.5  # Default low score

        return scores

    def _calculate_weights(
        self, performance_scores: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Calculate new weights based on performance scores.

        Args:
            performance_scores: Strategy performance scores

        Returns:
            Tuple of (new_weights, constraint_adjustments)
        """
        # Calculate raw weights from performance scores
        total_score = sum(performance_scores.values())

        if total_score == 0:
            # All scores zero, use equal weights
            logger.warning("All performance scores zero, using equal weights")
            return {s: 0.20 for s in STRATEGIES}, {}

        # Normalize to get raw weights
        raw_weights = {
            strategy: score / total_score
            for strategy, score in performance_scores.items()
        }

        # Apply floor/ceiling constraints
        constrained_weights = {}
        constraint_adjustments = {}

        for strategy, weight in raw_weights.items():
            if weight < self.FLOOR:
                constrained_weights[strategy] = self.FLOOR
                constraint_adjustments[strategy] = "floor"
            elif weight > self.CEILING:
                constrained_weights[strategy] = self.CEILING
                constraint_adjustments[strategy] = "ceiling"
            else:
                constrained_weights[strategy] = weight

        # Check if sum is 1.0 (within tolerance)
        total = sum(constrained_weights.values())
        tolerance = 0.0001

        if abs(total - 1.0) <= tolerance:
            return constrained_weights, constraint_adjustments

        # Need to redistribute excess/deficit among unconstrained strategies
        unconstrained_strategies = [
            s for s in STRATEGIES if s not in constraint_adjustments
        ]

        if not unconstrained_strategies:
            # All strategies constrained, normalize anyway
            logger.warning("All strategies hit constraints, normalizing")
            total = sum(constrained_weights.values())
            return {
                s: w / total for s, w in constrained_weights.items()
            }, constraint_adjustments

        # Calculate excess/deficit
        excess = 1.0 - total

        if excess > 0:
            # Distribute excess to unconstrained strategies
            unconstrained_total = sum(
                constrained_weights[s] for s in unconstrained_strategies
            )

            if unconstrained_total > 0:
                for strategy in unconstrained_strategies:
                    proportion = constrained_weights[strategy] / unconstrained_total
                    constrained_weights[strategy] += proportion * excess
            else:
                # All unconstrained have 0 weight, distribute equally
                equal_addition = excess / len(unconstrained_strategies)
                for strategy in unconstrained_strategies:
                    constrained_weights[strategy] += equal_addition

        # Re-normalize after redistribution
        total = sum(constrained_weights.values())
        final_weights = {
            strategy: weight / total
            for strategy, weight in constrained_weights.items()
        }

        return final_weights, constraint_adjustments

    def project_convergence(self) -> ConvergenceProjection:
        """Project weight convergence from simulation results.

        Returns:
            ConvergenceProjection with convergence analysis
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        # Get final weights
        final_weights = self.weight_updates[-1].weights.copy()

        # Detect convergence
        converged = self.detect_convergence(window=3)
        weeks_to_convergence = self.estimate_weeks_to_convergence()

        # Calculate volatility
        volatility = self.calculate_weight_volatility()

        return ConvergenceProjection(
            final_weights=final_weights,
            weeks_to_convergence=weeks_to_convergence,
            convergence_stable=converged,
            weight_volatility=volatility,
            convergence_criteria="Weight changes < 0.01 for 3 consecutive weeks",
        )

    def calculate_weight_volatility(self) -> float:
        """Calculate average week-over-week weight volatility.

        Returns:
            Average volatility (standard deviation of weight changes)
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        if len(self.weight_updates) < 2:
            return 0.0

        # Calculate week-over-week changes
        all_changes = []

        for i in range(1, len(self.weight_updates)):
            prev_weights = self.weight_updates[i - 1].weights
            curr_weights = self.weight_updates[i].weights

            # Calculate absolute changes for each strategy
            for strategy in STRATEGIES:
                change = abs(curr_weights[strategy] - prev_weights[strategy])
                all_changes.append(change)

        if not all_changes:
            return 0.0

        # Calculate standard deviation
        import statistics
        return statistics.stdev(all_changes) if len(all_changes) > 1 else 0.0

    def detect_convergence(self, window: int = 3) -> bool:
        """Detect if weights have converged.

        Convergence: Weight changes < threshold (0.01) for N consecutive weeks.

        Args:
            window: Number of consecutive weeks to check (default 3)

        Returns:
            True if converged, False otherwise
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        if len(self.weight_updates) < window + 1:
            return False

        threshold = 0.01

        # Check last N weeks
        for i in range(len(self.weight_updates) - window, len(self.weight_updates)):
            prev_weights = self.weight_updates[i - 1].weights
            curr_weights = self.weight_updates[i].weights

            # Calculate maximum weight change
            max_change = max(
                abs(curr_weights[s] - prev_weights[s]) for s in STRATEGIES
            )

            if max_change >= threshold:
                return False

        return True

    def estimate_weeks_to_convergence(self) -> int | None:
        """Estimate week number when convergence occurred.

        Returns:
            Week number of convergence, or None if not converged
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        threshold = 0.01
        window = 3

        # Search for convergence point
        for i in range(window, len(self.weight_updates)):
            # Check if stable for 'window' weeks ending at week i
            stable = True

            for j in range(i - window + 1, i + 1):
                prev_weights = self.weight_updates[j - 1].weights
                curr_weights = self.weight_updates[j].weights

                max_change = max(
                    abs(curr_weights[s] - prev_weights[s]) for s in STRATEGIES
                )

                if max_change >= threshold:
                    stable = False
                    break

            if stable:
                return i

        return None

    def detect_constraint_hits(self) -> list[ConstraintHit]:
        """Detect constraint binding events from simulation.

        Returns:
            List of ConstraintHit records
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        constraint_hits = []

        # Calculate what weights would be without constraints
        for update in self.weight_updates:
            performance_scores = update.performance_scores
            total_score = sum(performance_scores.values())

            if total_score == 0:
                continue

            # Calculate unconstrained weights
            unconstrained_weights = {
                strategy: score / total_score
                for strategy, score in performance_scores.items()
            }

            # Check for constraint hits
            for strategy in STRATEGIES:
                actual_weight = update.weights[strategy]
                unconstrained_weight = unconstrained_weights[strategy]

                # Check floor
                if actual_weight == self.FLOOR and unconstrained_weight < self.FLOOR:
                    constraint_hits.append(
                        ConstraintHit(
                            week=update.week,
                            strategy=strategy,
                            constraint_type="floor",
                            constraint_value=self.FLOOR,
                            calculated_weight=unconstrained_weight,
                            constrained_weight=actual_weight,
                        )
                    )

                # Check ceiling
                if actual_weight == self.CEILING and unconstrained_weight > self.CEILING:
                    constraint_hits.append(
                        ConstraintHit(
                            week=update.week,
                            strategy=strategy,
                            constraint_type="ceiling",
                            constraint_value=self.CEILING,
                            calculated_weight=unconstrained_weight,
                            constrained_weight=actual_weight,
                        )
                    )

        logger.info(f"Detected {len(constraint_hits)} constraint hits")
        return constraint_hits

    def generate_weight_trajectory_data(self) -> dict:
        """Generate data for weight trajectory visualization.

        Returns:
            Dictionary with x (weeks), y (weights), series (strategies)
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        weeks = [update.week for update in self.weight_updates]

        trajectories = {
            "weeks": weeks,
            "strategies": {},
            "floor": [self.FLOOR] * len(weeks),
            "ceiling": [self.CEILING] * len(weeks),
        }

        for strategy in STRATEGIES:
            trajectories["strategies"][strategy] = [
                update.weights[strategy] for update in self.weight_updates
            ]

        return trajectories

    def generate_performance_trajectory_data(self) -> dict:
        """Generate data for performance trajectory visualization.

        Returns:
            Dictionary with weeks and performance scores per strategy
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        weeks = [update.week for update in self.weight_updates]

        trajectories = {
            "weeks": weeks,
            "strategies": {},
        }

        for strategy in STRATEGIES:
            trajectories["strategies"][strategy] = [
                update.performance_scores[strategy] for update in self.weight_updates
            ]

        return trajectories

    def generate_constraint_markers(self) -> list[dict]:
        """Generate data for constraint hit visualization.

        Returns:
            List of marker dictionaries with x, y, color, label
        """
        if not self.weight_updates:
            raise ValueError("Must run simulate_weekly_rebalancing() first")

        constraint_hits = self.detect_constraint_hits()

        markers = []

        for hit in constraint_hits:
            # Find the weight at that week
            update = self.weight_updates[hit.week - 1]

            marker = {
                "x": hit.week,
                "y": update.weights[hit.strategy],
                "strategy": hit.strategy,
                "constraint_type": hit.constraint_type,
                "color": "red" if hit.constraint_type == "floor" else "blue",
                "label": f"{hit.strategy} hit {hit.constraint_type} (Week {hit.week})",
            }

            markers.append(marker)

        return markers


class WeightEvolutionReportGenerator:
    """Generate comprehensive weight evolution analysis report."""

    def __init__(self, simulator: WeightEvolutionSimulator):
        """Initialize report generator.

        Args:
            simulator: WeightEvolutionSimulator with simulation results
        """
        self.simulator = simulator

    def generate_report(self) -> str:
        """Generate markdown report.

        Returns:
            Markdown formatted report
        """
        from datetime import date

        projection = self.simulator.project_convergence()
        constraint_hits = self.simulator.detect_constraint_hits()
        trajectory_data = self.simulator.generate_weight_trajectory_data()

        lines = [
            "# Weight Evolution Simulation Report",
            "",
            f"**Generated:** {date.today().strftime('%Y-%m-%d')}",
            f"**Simulation:** 12-week weight projection for ensemble trading",
            "",
            "## Executive Summary",
            "",
            f"**Convergence Status:** {'✅ Converged' if projection.convergence_stable else '❌ Not Converged'}",
            f"**Weeks to Convergence:** {projection.weeks_to_convergence if projection.weeks_to_convergence else 'N/A (did not converge)'}",
            f"**Weight Volatility:** {projection.weight_volatility:.4f}",
            "",
            "### Final Weights",
            "",
            "| Strategy | Final Weight |",
            "|----------|--------------|",
        ]

        # Sort strategies by final weight (descending)
        sorted_strategies = sorted(
            projection.final_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for strategy, weight in sorted_strategies:
            lines.append(f"| {strategy} | {weight:.3f} |")

        lines.extend([
            "",
            "## Weight Evolution",
            "",
            "### Weight Trajectory",
            "",
            "**Week-by-week weights:**",
            "",
            "| Week | " + " | ".join(STRATEGIES) + " |",
            "|" + "------|" * (len(STRATEGIES) + 1),
        ])

        # Add weight trajectory table
        for update in self.simulator.weight_updates:
            row = f"| {update.week} |"
            for strategy in STRATEGIES:
                weight = update.weights[strategy]
                # Highlight constraint hits
                if strategy in update.constraints_active:
                    constraint = update.constraints_active[strategy]
                    row += f" **{weight:.3f}* ({constraint[0].upper()})** |"
                else:
                    row += f" {weight:.3f} |"
            lines.append(row)

        lines.extend([
            "",
            "*Floor (F) = 0.05, Ceiling (C) = 0.40",
            "",
            "## Convergence Analysis",
            "",
            f"**Convergence Criteria:** {projection.convergence_criteria}",
            f"**Converged:** {'Yes' if projection.convergence_stable else 'No'}",
            f"**Weeks to Convergence:** {projection.weeks_to_convergence if projection.weeks_to_convergence else 'Did not converge in 12 weeks'}",
            f"**Weight Volatility:** {projection.weight_volatility:.4f} (average week-over-week change)",
            "",
        ])

        if projection.convergence_stable:
            lines.extend([
                "### Interpretation",
                "",
                "Weights have stabilized, suggesting the ensemble has found an optimal ",
                "allocation based on strategy performance. Future weight changes are expected ",
                "to be minimal unless market conditions or strategy performance changes significantly.",
                "",
            ])
        else:
            lines.extend([
                "### Interpretation",
                "",
                "Weights have NOT converged within 12 weeks, suggesting ongoing adjustment ",
                "as the system seeks optimal allocation. Consider:",
                "- Extending simulation period to 16-20 weeks",
                "- Reviewing constraint settings (floor/ceiling may be preventing convergence)",
                "- Analyzing if strategy performance is stable enough for convergence",
                "",
            ])

        lines.extend([
            "## Constraint Analysis",
            "",
            f"**Total Constraint Hits:** {len(constraint_hits)}",
            "",
        ])

        if constraint_hits:
            # Group by strategy
            by_strategy = {}
            for hit in constraint_hits:
                if hit.strategy not in by_strategy:
                    by_strategy[hit.strategy] = []
                by_strategy[hit.strategy].append(hit)

            lines.extend([
                "### Constraint Binding Events",
                "",
                "| Week | Strategy | Type | Calculated | Constrained |",
                "|------|----------|------|------------|-------------|",
            ])

            for hit in constraint_hits:
                lines.append(
                    f"| {hit.week} | {hit.strategy} | {hit.constraint_type} | "
                    f"{hit.calculated_weight:.3f} | {hit.constrained_weight:.3f} |"
                )

            lines.extend([
                "",
                "### Strategies Most Constrained",
                "",
                "| Strategy | Floor Hits | Ceiling Hits | Total |",
                "|----------|------------|--------------|-------|",
            ])

            for strategy in STRATEGIES:
                floor_hits = sum(1 for h in constraint_hits
                               if h.strategy == strategy and h.constraint_type == "floor")
                ceiling_hits = sum(1 for h in constraint_hits
                                 if h.strategy == strategy and h.constraint_type == "ceiling")
                total = floor_hits + ceiling_hits
                lines.append(f"| {strategy} | {floor_hits} | {ceiling_hits} | {total} |")

        else:
            lines.extend([
                "No constraints were hit during the 12-week simulation. ",
                "The floor (0.05) and ceiling (0.40) constraints did not bind, ",
                "suggesting weights naturally stayed within bounds.",
                "",
            ])

        lines.extend([
            "## Recommendations for Epic 4 (Paper Trading)",
            "",
        ])

        # Generate recommendations based on analysis
        if projection.convergence_stable:
            lines.extend([
                "### Weight Optimization",
                "",
                "- Weights are expected to stabilize quickly in paper trading",
                "- Monitor weekly rebalancing to confirm convergence behavior matches simulation",
                "- If convergence occurs faster than expected, consider reducing rebalancing frequency",
                "",
            ])
        else:
            lines.extend([
                "### Weight Optimization",
                "",
                "- Weights may continue to adjust for several weeks before stabilizing",
                "- Monitor weight volatility closely in first 8 weeks of paper trading",
                "- Consider more frequent rebalancing (e.g., every 3-5 days) if volatility is high",
                "",
            ])

        if constraint_hits:
            high_freq = [s for s in STRATEGIES
                        if sum(1 for h in constraint_hits if h.strategy == s) >= 3]

            if high_freq:
                lines.extend([
                    "### Constraint Management",
                    "",
                    f"**Strategies hitting constraints frequently:** {', '.join(high_freq)}",
                    "- These strategies may benefit from constraint adjustment",
                    "- Consider widening ceiling if strategy consistently outperforms",
                    "- Consider lowering floor if strategy consistently underperforms",
                    "- Monitor diversity impact if constraints are adjusted",
                    "",
                ])
            else:
                lines.extend([
                    "### Constraint Management",
                    "",
                    "- Constraints are working as designed to maintain diversity",
                    "- Current floor/ceiling settings appear appropriate",
                    "- No immediate adjustments needed",
                    "",
                ])
        else:
            lines.extend([
                "### Constraint Management",
                "",
                "- No constraints hit during simulation",
                "- Current settings (0.05 floor, 0.40 ceiling) are appropriate",
                "- Monitor if paper trading shows different behavior",
                "",
            ])

        lines.extend([
            "### Monitoring Priorities",
            "",
            "- Track weight evolution weekly to confirm simulation accuracy",
            "- Monitor if convergence timeline matches projection",
            "- Watch for strategies that may need constraint adjustment",
            "- Verify ensemble performance maintains diversification benefits",
            "",
            "## Conclusion",
            "",
            f"The 12-week weight evolution simulation projects a " +
            ("**converged**" if projection.convergence_stable else "**dynamic**") + " weight allocation. " +
            f"Final weights range from {min(projection.final_weights.values()):.3f} to " +
            f"{max(projection.final_weights.values()):.3f}.",
            "",
            "**Next Steps:**",
            "- Use this projection as baseline for Epic 4 paper trading",
            "- Compare actual weight evolution to simulation",
            "- Adjust constraints if paper trading deviates significantly",
            "- Re-run simulation with updated performance data after 12 weeks",
            "",
            "---",
            f"*Report generated by WeightEvolutionSimulator on {date.today().strftime('%Y-%m-%d')}*",
        ])

        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        """Save report to file.

        Args:
            path: Path to save markdown report
        """
        from datetime import date

        report = self.generate_report()

        with open(path, "w") as f:
            f.write(report)

        logger.info(f"Weight evolution report saved to {path}")



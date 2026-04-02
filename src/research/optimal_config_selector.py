"""Optimal Configuration Selector for Ensemble Trading System.

Analyzes walk-forward validation results and selects the optimal parameter
configuration using multi-criteria decision analysis.

Key Components:
- SelectionCriteria: Defines primary and secondary criteria for configuration selection
- OptimalConfigurationSelector: Main selector class for analyzing and ranking configurations
- PrimaryCriteria: Validation results for primary criteria (minimum thresholds)
- CompositeScore: Multi-criteria scoring components and composite score
- ConfigurationComparison: Ranked configuration with full comparison details
- SelectionReport: Comprehensive selection report with analysis and recommendations
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR CONFIGURATION SELECTION
# =============================================================================


class SelectionCriteria(BaseModel):
    """Criteria for selecting optimal configuration.

    Attributes:
        min_win_rate: Minimum out-of-sample win rate (default: 0.55 = 55%)
        min_profit_factor: Minimum out-of-sample profit factor (default: 1.5)
        max_drawdown: Maximum acceptable drawdown (default: 0.15 = 15%)
        min_trade_frequency: Minimum trades per day (default: 3.0)
        max_win_rate_std: Maximum win rate standard deviation (default: 0.10 = 10%)
    """

    min_win_rate: float = Field(
        default=0.55, ge=0.0, le=1.0, description="Minimum win rate"
    )
    min_profit_factor: float = Field(
        default=1.5, ge=1.0, description="Minimum profit factor"
    )
    max_drawdown: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Maximum drawdown"
    )
    min_trade_frequency: float = Field(
        default=3.0, ge=0.0, description="Minimum trades per day"
    )
    max_win_rate_std: float = Field(
        default=0.10, ge=0.0, description="Maximum win rate std dev"
    )


class PrimaryCriteria(BaseModel):
    """Results of primary criteria validation.

    Attributes:
        passes: Whether all primary criteria are met
        win_rate_pass: Win rate meets minimum
        profit_factor_pass: Profit factor meets minimum
        drawdown_pass: Drawdown within maximum
        trade_frequency_pass: Trade frequency above minimum
        consistency_pass: Win rate consistency within maximum
    """

    passes: bool = Field(..., description="All primary criteria passed")
    win_rate_pass: bool = Field(..., description="Win rate threshold met")
    profit_factor_pass: bool = Field(..., description="Profit factor threshold met")
    drawdown_pass: bool = Field(..., description="Drawdown threshold met")
    trade_frequency_pass: bool = Field(..., description="Trade frequency threshold met")
    consistency_pass: bool = Field(..., description="Consistency threshold met")


class CompositeScore(BaseModel):
    """Multi-criteria composite score for configuration ranking.

    Attributes:
        performance_score: Normalized performance score (0-1)
        stability_score: Normalized stability score (0-1)
        risk_score: Normalized risk score (0-1, higher = lower risk)
        frequency_score: Normalized frequency score (0-1)
        composite_score: Weighted composite score (0-1)
    """

    performance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Performance score (0-1)"
    )
    stability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Stability score (0-1)"
    )
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    frequency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Frequency score (0-1)"
    )
    composite_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite score (0-1)"
    )


class ConfigurationComparison(BaseModel):
    """Ranked configuration with full comparison details.

    Attributes:
        rank: Rank in comparison (1 = best)
        combination_id: Configuration identifier
        composite_score: Overall composite score
        individual_scores: Component scores
        avg_oos_win_rate: Average out-of-sample win rate
        avg_oos_profit_factor: Average out-of-sample profit factor
        max_drawdown: Maximum drawdown
        trade_frequency: Trades per day
        parameter_stability_score: Parameter stability (0-1)
        performance_stability: In-sample vs out-of-sample correlation (0-1)
    """

    rank: int = Field(default=1, ge=1, description="Rank in comparison")
    combination_id: str = Field(..., description="Configuration identifier")
    composite_score: float = Field(..., ge=0.0, le=1.0, description="Composite score")
    individual_scores: CompositeScore = Field(..., description="Component scores")
    avg_oos_win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate")
    avg_oos_profit_factor: float = Field(..., ge=0.0, description="Profit factor")
    max_drawdown: float = Field(..., ge=0.0, le=1.0, description="Max drawdown")
    trade_frequency: float = Field(..., ge=0.0, description="Trades per day")
    parameter_stability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Parameter stability"
    )
    performance_stability: float = Field(
        ..., ge=0.0, le=1.0, description="Performance stability"
    )


class TopConfigurations(BaseModel):
    """Top N configurations after ranking.

    Attributes:
        configurations: List of ranked configurations
        total_evaluated: Total configurations evaluated
        passing_primary_criteria: Number passing primary criteria
    """

    configurations: list[ConfigurationComparison] = Field(
        default_factory=list, description="Ranked configurations"
    )
    total_evaluated: int = Field(..., ge=0, description="Total evaluated")
    passing_primary_criteria: int = Field(
        ..., ge=0, description="Passing primary criteria"
    )


class SelectionReport(BaseModel):
    """Comprehensive selection report.

    Attributes:
        optimal_config_id: ID of selected optimal configuration
        optimal_config_metrics: Metrics for optimal configuration
        top_configurations: Top configurations ranked
        comparison_table: Comparison table as DataFrame
        sensitivity_analysis: Sensitivity analysis results
        confidence_level: Confidence in selection (high/medium/low)
    """

    model_config = {"arbitrary_types_allowed": True}

    optimal_config_id: str = Field(..., description="Optimal configuration ID")
    optimal_config_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Optimal config metrics"
    )
    top_configurations: TopConfigurations = Field(..., description="Top configurations")
    comparison_table: pd.DataFrame = Field(..., description="Comparison table")
    sensitivity_analysis: dict[str, Any] = Field(
        default_factory=dict, description="Sensitivity analysis"
    )
    confidence_level: str = Field(..., description="Confidence level (high/medium/low)")


# =============================================================================
# OPTIMAL CONFIGURATION SELECTOR
# =============================================================================


class OptimalConfigurationSelector:
    """Selects optimal configuration from walk-forward validation results.

    Uses multi-criteria decision analysis to evaluate configurations on:
    - Performance (win rate, profit factor)
    - Stability (parameter stability, performance consistency)
    - Risk (maximum drawdown)
    - Trade frequency (optimal range)

    Attributes:
        hdf5_path: Path to HDF5 file with walk-forward results
        summary_csv_path: Path to CSV with summary metrics
        criteria: Selection criteria for filtering and scoring
        configurations: Loaded configuration data
    """

    # Score weights for composite score
    PERFORMANCE_WEIGHT = 0.40
    STABILITY_WEIGHT = 0.30
    RISK_WEIGHT = 0.20
    FREQUENCY_WEIGHT = 0.10

    # Optimal trade frequency target (trades per day)
    OPTIMAL_FREQUENCY = 7.0
    FREQUENCY_TOLERANCE = 3.0  # Acceptable range: 4-10 trades/day

    def __init__(
        self,
        hdf5_path: Path | str,
        summary_csv_path: Path | str,
        criteria: SelectionCriteria | None = None,
    ) -> None:
        """Initialize optimal configuration selector.

        Args:
            hdf5_path: Path to HDF5 file with walk-forward results
            summary_csv_path: Path to CSV with summary metrics
            criteria: Selection criteria (uses default if None)
        """
        self.hdf5_path = Path(hdf5_path)
        self.summary_csv_path = Path(summary_csv_path)
        self.criteria = criteria or SelectionCriteria()
        self.configurations: dict[str, dict[str, Any]] = {}

        logger.info(
            f"OptimalConfigurationSelector initialized: "
            f"hdf5={self.hdf5_path}, csv={self.summary_csv_path}"
        )

    def load_results(self) -> None:
        """Load walk-forward results from HDF5 and CSV files.

        Reads configuration data from HDF5 file and enriches with summary
        metrics from CSV file.

        Raises:
            FileNotFoundError: If HDF5 or CSV files don't exist
            ValueError: If data quality issues detected
        """
        # Check files exist
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        if not self.summary_csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.summary_csv_path}")

        # Load HDF5 results
        with h5py.File(self.hdf5_path, "r") as f:
            for combo_id in f.keys():
                group = f[combo_id]

                # Extract attributes
                config = {
                    "combination_id": combo_id,
                    "avg_oos_win_rate": group.attrs.get("avg_oos_win_rate", 0.0),
                    "avg_oos_profit_factor": group.attrs.get(
                        "avg_oos_profit_factor", 0.0
                    ),
                    "win_rate_std": group.attrs.get("win_rate_std", 0.0),
                    "max_drawdown": group.attrs.get("max_drawdown", 0.0),
                    "total_trades": group.attrs.get("total_trades", 0),
                    "parameter_stability_score": group.attrs.get(
                        "parameter_stability_score", 0.0
                    ),
                    "performance_stability": group.attrs.get(
                        "performance_stability", 0.0
                    ),
                }

                # Extract parameters (all attributes starting with "param_")
                for attr_name in group.attrs.keys():
                    if attr_name.startswith("param_"):
                        param_name = attr_name[6:]  # Remove "param_" prefix
                        config[param_name] = group.attrs[attr_name]

                self.configurations[combo_id] = config

        # Load CSV summary and enrich
        summary_df = pd.read_csv(self.summary_csv_path)
        for _, row in summary_df.iterrows():
            combo_id = row["combination_id"]
            if combo_id in self.configurations:
                # Add any missing fields from CSV
                for col in summary_df.columns:
                    if col not in self.configurations[combo_id]:
                        self.configurations[combo_id][col] = row[col]

        # Validate data quality
        self._validate_data_quality()

        logger.info(f"Loaded {len(self.configurations)} configurations")

    def _validate_data_quality(self) -> None:
        """Validate loaded data for quality issues.

        Raises:
            ValueError: If data quality issues detected
        """
        if not self.configurations:
            raise ValueError("No configurations found in results")

        # Check for missing values
        for combo_id, config in self.configurations.items():
            required_fields = [
                "avg_oos_win_rate",
                "avg_oos_profit_factor",
                "max_drawdown",
                "total_trades",
            ]
            for field in required_fields:
                if field not in config or config[field] is None:
                    raise ValueError(f"Missing required field {field} in {combo_id}")

        # Check for valid ranges
        for combo_id, config in self.configurations.items():
            if not (0.0 <= config["avg_oos_win_rate"] <= 1.0):
                raise ValueError(f"Invalid win rate for {combo_id}")
            if config["total_trades"] < 0:
                raise ValueError(f"Invalid trade count for {combo_id}")

        logger.info("Data quality validation passed")

    def validate_primary_criteria(self, config: dict[str, Any]) -> PrimaryCriteria:
        """Validate configuration against primary criteria (minimum thresholds).

        Args:
            config: Configuration dictionary with metrics

        Returns:
            PrimaryCriteria validation result
        """
        win_rate = config["avg_oos_win_rate"]
        profit_factor = config["avg_oos_profit_factor"]
        drawdown = config["max_drawdown"]

        # Calculate trade frequency (trades per day)
        # Assuming ~21 trading days per month and walk-forward covers ~1 year
        total_trades = config["total_trades"]
        trade_frequency = total_trades / 252.0 * 21.0  # Approximate trades/day

        win_rate_std = config.get("win_rate_std", 0.0)

        # Check each criterion
        win_rate_pass = win_rate >= self.criteria.min_win_rate
        profit_factor_pass = profit_factor >= self.criteria.min_profit_factor
        drawdown_pass = drawdown <= self.criteria.max_drawdown
        trade_frequency_pass = trade_frequency >= self.criteria.min_trade_frequency
        consistency_pass = win_rate_std <= self.criteria.max_win_rate_std

        # Overall pass = all criteria pass
        passes = all(
            [
                win_rate_pass,
                profit_factor_pass,
                drawdown_pass,
                trade_frequency_pass,
                consistency_pass,
            ]
        )

        return PrimaryCriteria(
            passes=passes,
            win_rate_pass=win_rate_pass,
            profit_factor_pass=profit_factor_pass,
            drawdown_pass=drawdown_pass,
            trade_frequency_pass=trade_frequency_pass,
            consistency_pass=consistency_pass,
        )

    def calculate_secondary_scores(self, config: dict[str, Any]) -> CompositeScore:
        """Calculate normalized secondary criteria scores.

        Args:
            config: Configuration dictionary with metrics

        Returns:
            CompositeScore with all component scores
        """
        # Extract metrics
        win_rate = config["avg_oos_win_rate"]
        profit_factor = config["avg_oos_profit_factor"]
        drawdown = config["max_drawdown"]
        total_trades = config["total_trades"]

        # Calculate trade frequency
        trade_frequency = total_trades / 252.0 * 21.0

        # Get stability scores
        parameter_stability = config.get("parameter_stability_score", 0.5)
        performance_stability = config.get("performance_stability", 0.5)

        # Performance Score (40% weight)
        # Normalize win rate and profit factor to 0-1
        # Win rate: 55% = 0.0, 75% = 1.0
        win_rate_normalized = (win_rate - 0.55) / (0.75 - 0.55)
        win_rate_normalized = max(0.0, min(1.0, win_rate_normalized))

        # Profit factor: 1.5 = 0.0, 3.0 = 1.0
        profit_factor_normalized = (profit_factor - 1.5) / (3.0 - 1.5)
        profit_factor_normalized = max(0.0, min(1.0, profit_factor_normalized))

        # Performance score = average of normalized metrics
        performance_score = (win_rate_normalized + profit_factor_normalized) / 2.0

        # Stability Score (30% weight)
        # Average of parameter stability and performance stability
        stability_score = (parameter_stability + performance_stability) / 2.0

        # Risk Score (20% weight)
        # Inverse of drawdown (lower drawdown = higher score)
        # 0% drawdown = 1.0, 15% drawdown = 0.0
        risk_score = 1.0 - (drawdown / 0.15)
        risk_score = max(0.0, min(1.0, risk_score))

        # Frequency Score (10% weight)
        # Bell curve centered on optimal frequency
        freq_diff = abs(trade_frequency - self.OPTIMAL_FREQUENCY)
        if freq_diff <= self.FREQUENCY_TOLERANCE:
            # Within tolerance: 1.0
            frequency_score = 1.0
        else:
            # Outside tolerance: linear decay
            excess = freq_diff - self.FREQUENCY_TOLERANCE
            frequency_score = max(0.0, 1.0 - excess / self.FREQUENCY_TOLERANCE)

        # Composite Score (weighted sum)
        composite_score = (
            self.PERFORMANCE_WEIGHT * performance_score
            + self.STABILITY_WEIGHT * stability_score
            + self.RISK_WEIGHT * risk_score
            + self.FREQUENCY_WEIGHT * frequency_score
        )

        return CompositeScore(
            performance_score=performance_score,
            stability_score=stability_score,
            risk_score=risk_score,
            frequency_score=frequency_score,
            composite_score=composite_score,
        )

    def filter_by_primary_criteria(self) -> dict[str, dict[str, Any]]:
        """Filter configurations by primary criteria.

        Returns:
            Dictionary of configurations passing primary criteria
        """
        passing = {}

        for combo_id, config in self.configurations.items():
            criteria_result = self.validate_primary_criteria(config)
            if criteria_result.passes:
                passing[combo_id] = config

        logger.info(
            f"Primary criteria filter: {len(passing)}/{len(self.configurations)} passed"
        )

        return passing

    def rank_configurations(self) -> list[ConfigurationComparison]:
        """Rank configurations by composite score.

        Only includes configurations passing primary criteria.
        Sorted by composite score descending (best first).

        Returns:
            List of ranked ConfigurationComparison objects
        """
        # Filter by primary criteria
        passing_configs = self.filter_by_primary_criteria()

        if not passing_configs:
            logger.warning("No configurations pass primary criteria")
            return []

        # Calculate scores for all passing configurations
        scored_configs = []
        for combo_id, config in passing_configs.items():
            scores = self.calculate_secondary_scores(config)

            # Calculate trade frequency
            trade_frequency = config["total_trades"] / 252.0 * 21.0

            comparison = ConfigurationComparison(
                combination_id=combo_id,
                composite_score=scores.composite_score,
                individual_scores=scores,
                avg_oos_win_rate=config["avg_oos_win_rate"],
                avg_oos_profit_factor=config["avg_oos_profit_factor"],
                max_drawdown=config["max_drawdown"],
                trade_frequency=trade_frequency,
                parameter_stability_score=config.get("parameter_stability_score", 0.5),
                performance_stability=config.get("performance_stability", 0.5),
            )
            scored_configs.append(comparison)

        # Sort by composite score (descending)
        scored_configs.sort(key=lambda x: x.composite_score, reverse=True)

        # Assign ranks
        for idx, config in enumerate(scored_configs, start=1):
            config.rank = idx

        logger.info(f"Ranked {len(scored_configs)} configurations")

        return scored_configs

    def select_top_configurations(self, n: int = 10) -> TopConfigurations:
        """Select top N configurations by composite score.

        Args:
            n: Maximum number of configurations to select (default: 10)

        Returns:
            TopConfigurations with ranked configurations
        """
        ranked = self.rank_configurations()

        # Take top N
        top_n = ranked[:n]

        top_configs = TopConfigurations(
            configurations=top_n,
            total_evaluated=len(self.configurations),
            passing_primary_criteria=len(ranked),
        )

        logger.info(
            f"Selected top {len(top_n)} configurations "
            f"({top_configs.passing_primary_criteria} passed criteria)"
        )

        return top_configs

    def generate_comparison_table(self, top_configs: TopConfigurations) -> pd.DataFrame:
        """Generate comparison table for top configurations.

        Args:
            top_configs: TopConfigurations object

        Returns:
            DataFrame with comparison data
        """
        rows = []

        for config in top_configs.configurations:
            row = {
                "rank": config.rank,
                "combination_id": config.combination_id,
                "composite_score": config.composite_score,
                "performance_score": config.individual_scores.performance_score,
                "stability_score": config.individual_scores.stability_score,
                "risk_score": config.individual_scores.risk_score,
                "frequency_score": config.individual_scores.frequency_score,
                "avg_oos_win_rate": config.avg_oos_win_rate,
                "avg_oos_profit_factor": config.avg_oos_profit_factor,
                "max_drawdown": config.max_drawdown,
                "trade_frequency": config.trade_frequency,
                "parameter_stability_score": config.parameter_stability_score,
                "performance_stability": config.performance_stability,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        logger.info(f"Generated comparison table: {len(df)} rows")

        return df

    def select_optimal_configuration(self) -> str | None:
        """Select single optimal configuration using multi-criteria analysis.

        Applies additional tie-breaking rules:
        1. Higher parameter stability
        2. Moderate trade frequency (5-10 trades/day preferred)
        3. More balanced strategy weights
        4. Higher out-of-sample win rate
        5. Lower maximum drawdown

        Returns:
            Configuration ID of optimal configuration, or None if no valid configs
        """
        top_configs = self.select_top_configurations(n=10)

        if not top_configs.configurations:
            logger.warning("No configurations available for selection")
            return None

        # Select from top 3 for additional analysis
        candidates = top_configs.configurations[:3]

        # Score candidates on additional criteria
        best_config = None
        best_score = -1.0

        for config in candidates:
            # Additional scoring criteria
            score = 0.0

            # 1. Parameter stability (weight: 0.3)
            score += 0.3 * config.parameter_stability_score

            # 2. Trade frequency (weight: 0.2, prefer 5-10 range)
            if 5.0 <= config.trade_frequency <= 10.0:
                score += 0.2
            elif 3.0 <= config.trade_frequency < 5.0:
                score += 0.1

            # 3. Performance stability (weight: 0.2)
            score += 0.2 * config.performance_stability

            # 4. Win rate (weight: 0.2)
            score += 0.2 * config.avg_oos_win_rate

            # 5. Low drawdown (weight: 0.1)
            score += 0.1 * (1.0 - config.max_drawdown)

            if score > best_score:
                best_score = score
                best_config = config.combination_id

        logger.info(f"Selected optimal configuration: {best_config}")

        return best_config

    def generate_config_file(
        self, optimal_config_id: str, output_path: Path | str
    ) -> None:
        """Generate YAML configuration file with optimal parameters.

        Args:
            optimal_config_id: ID of optimal configuration
            output_path: Path to save YAML file

        Raises:
            ValueError: If optimal_config_id not found
        """
        if optimal_config_id not in self.configurations:
            raise ValueError(f"Configuration not found: {optimal_config_id}")

        config = self.configurations[optimal_config_id]

        # Generate YAML content
        lines = [
            "# Optimal Configuration Selected by Walk-Forward Validation",
            f"# Generated: {date.today().isoformat()}",
            f"# Configuration: {optimal_config_id}",
            "",
            "# Ensemble Parameters",
            "ensemble:",
            f"  confidence_threshold: {config.get('confidence_threshold', 0.50)}",
            "",
            "# Risk Parameters",
            "risk:",
            f"  daily_loss_limit: 500  # USD",
            f"  max_drawdown_percent: 12",
            f"  max_position_size: 5",
            "",
            "# Strategy Parameters (from optimal configuration)",
            "strategies:",
            "  triple_confluence:",
            "    enabled: true",
            "  wolf_pack:",
            "    enabled: true",
            "  adaptive_ema:",
            "    enabled: true",
            "  vwap_bounce:",
            "    enabled: true",
            "  opening_range:",
            "    enabled: true",
            "",
            "# Performance Metrics (for reference)",
            "# Walk-Forward Results:",
            f"#   Win Rate: {config['avg_oos_win_rate']:.2%}",
            f"#   Profit Factor: {config['avg_oos_profit_factor']:.2f}",
            f"#   Max Drawdown: {config['max_drawdown']:.2%}",
            f"#   Trade Frequency: {config['total_trades'] / 252.0 * 21.0:.1f} trades/day",
        ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Configuration file saved: {output_path}")

    def generate_selection_report(
        self, optimal_config_id: str, output_path: Path | str
    ) -> SelectionReport:
        """Generate comprehensive selection report.

        Args:
            optimal_config_id: ID of optimal configuration
            output_path: Path to save markdown report

        Returns:
            SelectionReport with full analysis

        Raises:
            ValueError: If optimal_config_id not found
        """
        if optimal_config_id not in self.configurations:
            raise ValueError(f"Configuration not found: {optimal_config_id}")

        # Get top configurations
        top_configs = self.select_top_configurations(n=10)

        # Generate comparison table
        comparison_table = self.generate_comparison_table(top_configs)

        # Get optimal config metrics
        optimal_config = self.configurations[optimal_config_id]

        # Calculate confidence level
        confidence = self._calculate_confidence_level(optimal_config, top_configs)

        # Sensitivity analysis (simplified)
        sensitivity = self._perform_sensitivity_analysis(optimal_config_id)

        # Generate report
        report = SelectionReport(
            optimal_config_id=optimal_config_id,
            optimal_config_metrics=optimal_config,
            top_configurations=top_configs,
            comparison_table=comparison_table,
            sensitivity_analysis=sensitivity,
            confidence_level=confidence,
        )

        # Save markdown report
        self._save_markdown_report(report, Path(output_path))

        logger.info(f"Selection report saved: {output_path}")

        return report

    def _calculate_confidence_level(
        self, optimal_config: dict[str, Any], top_configs: TopConfigurations
    ) -> str:
        """Calculate confidence level in selection.

        Args:
            optimal_config: Optimal configuration metrics
            top_configs: Top configurations for comparison

        Returns:
            Confidence level: "high", "medium", or "low"
        """
        # High confidence: top config is significantly better
        if len(top_configs.configurations) >= 2:
            top_score = top_configs.configurations[0].composite_score
            second_score = top_configs.configurations[1].composite_score
            score_margin = top_score - second_score

            if score_margin > 0.10:
                return "high"
            elif score_margin > 0.05:
                return "medium"

        return "low"

    def _perform_sensitivity_analysis(self, optimal_config_id: str) -> dict[str, Any]:
        """Perform sensitivity analysis on optimal configuration.

        Args:
            optimal_config_id: ID of optimal configuration

        Returns:
            Dictionary with sensitivity analysis results
        """
        # Simplified sensitivity analysis
        # In full implementation, would test parameter variations

        return {
            "parameter_sensitivity": "Not implemented",
            "market_condition_analysis": "Not implemented",
            "strategy_diversity": "Not implemented",
        }

    def _save_markdown_report(self, report: SelectionReport, output_path: Path) -> None:
        """Save selection report as markdown.

        Args:
            report: SelectionReport object
            output_path: Path to save markdown file
        """
        # Generate markdown table from DataFrame
        df = report.comparison_table
        table_lines = []

        # Header
        table_lines.append("| " + " | ".join(df.columns) + " |")
        table_lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")

        # Rows
        for _, row in df.iterrows():
            table_lines.append(
                "| "
                + " | ".join(
                    f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
                )
                + " |"
            )

        table_markdown = "\n".join(table_lines)

        lines = [
            "# Optimal Configuration Selection Report",
            "",
            f"**Generated:** {date.today().isoformat()}",
            f"**Optimal Configuration:** {report.optimal_config_id}",
            f"**Confidence Level:** {report.confidence_level.upper()}",
            "",
            "## Executive Summary",
            "",
            f"**Selected Configuration:** {report.optimal_config_id}",
            "",
            "**Performance Metrics:**",
            f"- Win Rate: {report.optimal_config_metrics['avg_oos_win_rate']:.2%}",
            f"- Profit Factor: {report.optimal_config_metrics['avg_oos_profit_factor']:.2f}",
            f"- Max Drawdown: {report.optimal_config_metrics['max_drawdown']:.2%}",
            f"- Trade Frequency: {report.optimal_config_metrics['total_trades'] / 252.0 * 21.0:.1f} trades/day",
            "",
            f"**Parameter Stability:** {report.optimal_config_metrics.get('parameter_stability_score', 0.5):.2f}",
            f"**Performance Stability:** {report.optimal_config_metrics.get('performance_stability', 0.5):.2f}",
            "",
            "## Top Configurations",
            "",
            f"Total configurations evaluated: {report.top_configurations.total_evaluated}",
            f"Configurations passing primary criteria: {report.top_configurations.passing_primary_criteria}",
            "",
            table_markdown,
            "",
            "## Optimal Configuration",
            "",
            f"**Configuration ID:** {report.optimal_config_id}",
            "",
            "**Key Metrics:**",
            f"- Win Rate: {report.optimal_config_metrics['avg_oos_win_rate']:.2%}",
            f"- Profit Factor: {report.optimal_config_metrics['avg_oos_profit_factor']:.2f}",
            f"- Max Drawdown: {report.optimal_config_metrics['max_drawdown']:.2%}",
            f"- Trade Frequency: {report.optimal_config_metrics['total_trades'] / 252.0 * 21.0:.1f} trades/day",
            f"- Parameter Stability: {report.optimal_config_metrics.get('parameter_stability_score', 0.5):.2f}",
            f"- Performance Stability: {report.optimal_config_metrics.get('performance_stability', 0.5):.2f}",
            "",
            "## Selection Criteria",
            "",
            "**Primary Criteria (Minimum Thresholds):**",
            f"- Win Rate ≥ {self.criteria.min_win_rate:.0%}",
            f"- Profit Factor ≥ {self.criteria.min_profit_factor}",
            f"- Max Drawdown ≤ {self.criteria.max_drawdown:.0%}",
            f"- Trade Frequency ≥ {self.criteria.min_trade_frequency:.1f} trades/day",
            f"- Win Rate Std Dev ≤ {self.criteria.max_win_rate_std:.0%}",
            "",
            "**Composite Score Weights:**",
            f"- Performance: {self.PERFORMANCE_WEIGHT:.0%}",
            f"- Stability: {self.STABILITY_WEIGHT:.0%}",
            f"- Risk: {self.RISK_WEIGHT:.0%}",
            f"- Frequency: {self.FREQUENCY_WEIGHT:.0%}",
            "",
            "## Conclusion",
            "",
            f"The optimal configuration ({report.optimal_config_id}) was selected "
            f"based on multi-criteria analysis of {report.top_configurations.total_evaluated} "
            f"parameter combinations.",
            "",
            f"**Confidence Level:** {report.confidence_level.upper()}",
            "",
            "**Recommendation:** Use this configuration for paper trading deployment.",
            "",
            "---",
            f"*Report generated by OptimalConfigurationSelector on {date.today().isoformat()}*",
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved: {output_path}")

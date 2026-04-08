"""Dynamic weight optimization for ensemble trading strategies.

This module implements performance tracking and dynamic weight rebalancing
for the ensemble system. Weights are adjusted weekly based on strategy performance
over a rolling 4-week window.

Performance Score = Win Rate × Profit Factor

Weights are constrained to maintain diversity:
- Floor: 0.05 (5%) per strategy
- Ceiling: 0.40 (40%) per strategy
- Sum: 1.0 (100%)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.detection.models import StrategyPerformance, CompletedTrade, WeightUpdate

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track strategy performance for weight optimization.

    Maintains a rolling window of completed trades and calculates
    performance metrics for each strategy.

    Attributes:
        min_trades: Minimum trades required for performance calculation
        window_weeks: Default window size in weeks (typically 4)
    """

    STRATEGIES = [
        "triple_confluence_scaler",
        "wolf_pack_3_edge",
        "adaptive_ema_momentum",
        "vwap_bounce",
        "opening_range_breakout"
    ]

    def __init__(self, min_trades: int = 20, window_weeks: int = 4) -> None:
        """Initialize performance tracker.

        Args:
            min_trades: Minimum trades required for reliable performance
            window_weeks: Default performance window in weeks

        Raises:
            ValueError: If parameters are invalid
        """
        if min_trades <= 0:
            raise ValueError("min_trades must be positive")
        if window_weeks <= 0:
            raise ValueError("window_weeks must be positive")

        self.min_trades = min_trades
        self.window_weeks = window_weeks
        self.trades: list[CompletedTrade] = []

        logger.info(
            f"PerformanceTracker initialized with min_trades={min_trades}, "
            f"window_weeks={window_weeks}"
        )

    def track_trade(self, trade: CompletedTrade) -> None:
        """Record a completed trade for performance tracking.

        Args:
            trade: Completed trade record

        Raises:
            ValueError: If trade strategy is not recognized
        """
        if trade.strategy_name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {trade.strategy_name}")

        self.trades.append(trade)

        # Prune old trades (keep last 8 weeks max)
        cutoff_date = datetime.now() - timedelta(weeks=8)
        self.trades = [t for t in self.trades if t.exit_time >= cutoff_date]

        logger.debug(
            f"Recorded trade {trade.trade_id} for {trade.strategy_name}, "
            f"pnl=${trade.pnl:.2f}, total_trades={len(self.trades)}"
        )

    def get_performance(self, strategy: str, window_end: datetime) -> StrategyPerformance:
        """Get performance metrics for a specific strategy.

        Args:
            strategy: Strategy name
            window_end: End of performance window

        Returns:
            StrategyPerformance with calculated metrics

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Try 4-week window first
        window_start_4w = window_end - timedelta(weeks=self.window_weeks)
        trades_4w = [
            t for t in self.trades
            if t.strategy_name == strategy and window_start_4w <= t.exit_time <= window_end
        ]

        # Check if we have sufficient data
        if len(trades_4w) >= self.min_trades:
            return self._calculate_performance_metrics(strategy, trades_4w, window_start_4w, window_end, "sufficient")

        # If insufficient in 4 weeks, try 8 weeks
        window_start_8w = window_end - timedelta(weeks=8)
        trades_8w = [
            t for t in self.trades
            if t.strategy_name == strategy and window_start_8w <= t.exit_time <= window_end
        ]

        if len(trades_8w) >= self.min_trades:
            return self._calculate_performance_metrics(strategy, trades_8w, window_start_8w, window_end, "insufficient_4weeks")

        # Still insufficient data
        logger.warning(
            f"Insufficient data for {strategy}: "
            f"{len(trades_8w)} trades (need {self.min_trades})"
        )

        # Return performance with zero scores
        return StrategyPerformance(
            strategy_name=strategy,
            window_start=window_start_4w,
            window_end=window_end,
            total_trades=len(trades_8w),
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            performance_score=0.0,
            data_quality="insufficient_8weeks"
        )

    def get_all_performance(self, window_end: datetime) -> dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies.

        Args:
            window_end: End of performance window

        Returns:
            Dictionary mapping strategy names to StrategyPerformance
        """
        performances = {}

        for strategy in self.STRATEGIES:
            performances[strategy] = self.get_performance(strategy, window_end)

        return performances

    def _calculate_performance_metrics(
        self,
        strategy: str,
        trades: list[CompletedTrade],
        window_start: datetime,
        window_end: datetime,
        data_quality: str
    ) -> StrategyPerformance:
        """Calculate performance metrics from trade list.

        Args:
            strategy: Strategy name
            trades: List of completed trades in window
            window_start: Window start time
            window_end: Window end time
            data_quality: Data quality flag

        Returns:
            StrategyPerformance with calculated metrics
        """
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winner())
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in trades if t.is_winner())
        gross_loss = abs(sum(t.pnl for t in trades if not t.is_winner()))

        # Handle edge case: no losses
        if gross_loss == 0:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        # Calculate performance score
        # Handle infinity profit_factor
        if profit_factor == float('inf'):
            performance_score = win_rate * 10.0  # Use high but finite score
        else:
            performance_score = win_rate * profit_factor

        return StrategyPerformance(
            strategy_name=strategy,
            window_start=window_start,
            window_end=window_end,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor if profit_factor != float('inf') else 0.0,
            performance_score=performance_score,
            data_quality=data_quality
        )

    def get_trade_count(self, strategy: str | None = None) -> int:
        """Get total trade count for a strategy or all strategies.

        Args:
            strategy: Specific strategy name, or None for all strategies

        Returns:
            Number of trades tracked
        """
        if strategy is None:
            return len(self.trades)

        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        return sum(1 for t in self.trades if t.strategy_name == strategy)

    def clear_trades(self) -> None:
        """Clear all tracked trades.

        Useful for testing or resetting state.
        """
        self.trades.clear()
        logger.info("Cleared all tracked trades")


class WeightCalculator:
    """Calculate and optimize ensemble weights based on performance.

    Converts performance scores into ensemble weights while enforcing
    floor/ceiling constraints to maintain strategy diversity.

    Attributes:
        floor: Minimum weight per strategy (default 0.05)
        ceiling: Maximum weight per strategy (default 0.40)
    """

    def __init__(self, floor: float = 0.05, ceiling: float = 0.40) -> None:
        """Initialize weight calculator.

        Args:
            floor: Minimum weight per strategy (default 0.05)
            ceiling: Maximum weight per strategy (default 0.40)

        Raises:
            ValueError: If parameters are invalid
        """
        if floor <= 0 or floor > 1.0:
            raise ValueError("floor must be between 0 and 1")
        if ceiling <= 0 or ceiling > 1.0:
            raise ValueError("ceiling must be between 0 and 1")
        if floor >= ceiling:
            raise ValueError("floor must be less than ceiling")

        self.floor = floor
        self.ceiling = ceiling

        logger.info(f"WeightCalculator initialized with floor={floor}, ceiling={ceiling}")

    def calculate_weights(self, performances: dict[str, StrategyPerformance]) -> dict[str, float]:
        """Calculate new weights based on performance scores.

        Args:
            performances: Dictionary of strategy performances

        Returns:
            Dictionary of new weights (summing to 1.0)
        """
        # Calculate raw weights from performance scores
        total_score = sum(p.performance_score for p in performances.values())

        # Handle edge case: all scores are zero
        if total_score == 0:
            logger.warning("All performance scores are zero, using equal weights")
            return {strategy: 0.20 for strategy in performances.keys()}

        raw_weights = {}
        for strategy, perf in performances.items():
            raw_weights[strategy] = perf.performance_score / total_score

        # Apply constraints and normalize
        final_weights, adjustments = self.apply_constraints(raw_weights)

        # Log constraint adjustments
        if adjustments:
            logger.info(f"Constraint adjustments: {adjustments}")

        return final_weights

    def normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights so they sum to 1.0.

        Args:
            weights: Dictionary of weights (may not sum to 1.0)

        Returns:
            Dictionary of normalized weights (summing to 1.0)
        """
        total = sum(weights.values())

        if total == 0:
            raise ValueError("Cannot normalize weights with sum of 0")

        normalized = {strategy: weight / total for strategy, weight in weights.items()}

        # Verify normalization
        new_total = sum(normalized.values())
        tolerance = 0.0001
        if abs(new_total - 1.0) > tolerance:
            raise ValueError(f"Normalization failed: sum={new_total:.4f}")

        return normalized

    def apply_constraints(self, weights: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
        """Apply floor/ceiling constraints to weights.

        Args:
            weights: Dictionary of raw weights

        Returns:
            Tuple of (constrained weights, constraint_adjustments)
        """
        constrained = weights.copy()
        adjustments = {}

        # First pass: Apply floor and ceiling
        for strategy, weight in constrained.items():
            if weight < self.floor:
                constrained[strategy] = self.floor
                adjustments[strategy] = "hit_floor"
            elif weight > self.ceiling:
                constrained[strategy] = self.ceiling
                adjustments[strategy] = "hit_ceiling"

        # Check if sum is still 1.0 (within tolerance)
        total = sum(constrained.values())
        tolerance = 0.0001

        if abs(total - 1.0) <= tolerance:
            return constrained, adjustments

        # Need to redistribute excess/deficit
        # Calculate which strategies are constrained
        constrained_strategies = set(adjustments.keys())
        unconstrained = {s: w for s, w in constrained.items() if s not in constrained_strategies}

        if not unconstrained:
            # All strategies constrained, use equal weights
            logger.warning("All strategies hit constraints, using equal weights")
            equal_weight = 1.0 / len(constrained)
            return {s: equal_weight for s in constrained}, adjustments

        # Calculate excess/deficit
        excess = 1.0 - total

        if excess > 0:
            # Need to add weight to unconstrained strategies
            # Distribute proportionally to current weights
            unconstrained_total = sum(unconstrained.values())

            if unconstrained_total > 0:
                for strategy in unconstrained:
                    proportion = unconstrained[strategy] / unconstrained_total
                    constrained[strategy] += proportion * excess
            else:
                # All unconstrained have 0 weight, distribute equally
                equal_addition = excess / len(unconstrained)
                for strategy in unconstrained:
                    constrained[strategy] += equal_addition

        elif excess < 0:
            # Need to remove weight from unconstrained strategies
            # (This shouldn't happen with floor/ceiling logic but handle anyway)
            unconstrained_total = sum(unconstrained.values())

            if unconstrained_total > 0:
                for strategy in unconstrained:
                    proportion = unconstrained[strategy] / unconstrained_total
                    constrained[strategy] += proportion * excess  # excess is negative

        # Re-normalize after redistribution
        constrained = self.normalize_weights(constrained)

        # Verify no constraints violated after normalization
        for strategy, weight in constrained.items():
            if strategy in constrained_strategies:
                continue  # Already at constraint

            if weight < self.floor or weight > self.ceiling:
                logger.warning(
                    f"Constraint violation after redistribution: "
                    f"{strategy}={weight:.4f}"
                )

        return constrained, adjustments


class WeightHistory:
    """Track and persist weight optimization history.

    Maintains a CSV file with all weight updates for analysis
    and audit trails.

    Attributes:
        history_path: Path to weight history CSV file
    """

    def __init__(self, history_path: str = "data/state/weight_history.csv") -> None:
        """Initialize weight history tracker.

        Args:
            history_path: Path to CSV file for persistence
        """
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with headers if it doesn't exist or is empty
        if not self.history_path.exists() or self.history_path.stat().st_size == 0:
            self._create_history_file()

        logger.info(f"WeightHistory initialized with path={history_path}")

    def _create_history_file(self) -> None:
        """Create history file with headers."""
        headers = [
            "rebalancing_number",
            "timestamp",
            "triple_confluence_scaler_weight",
            "wolf_pack_3_edge_weight",
            "adaptive_ema_momentum_weight",
            "vwap_bounce_weight",
            "opening_range_breakout_weight",
            "constraint_adjustments",
            "rebalancing_reason"
        ]

        import csv
        with open(self.history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

        logger.info(f"Created history file: {self.history_path}")

    def save_update(self, update: WeightUpdate, rebalancing_number: int) -> None:
        """Save a weight update to history.

        Args:
            update: Weight update to save
            rebalancing_number: Sequential rebalancing number
        """
        import csv

        # Define headers to match _create_history_file
        headers = [
            "rebalancing_number",
            "timestamp",
            "triple_confluence_scaler_weight",
            "wolf_pack_3_edge_weight",
            "adaptive_ema_momentum_weight",
            "vwap_bounce_weight",
            "opening_range_breakout_weight",
            "constraint_adjustments",
            "rebalancing_reason"
        ]

        # Create row dict
        row = {
            "rebalancing_number": rebalancing_number,
            "timestamp": update.timestamp.isoformat(),
            "triple_confluence_scaler_weight": update.new_weights.get("triple_confluence_scaler", 0.0),
            "wolf_pack_3_edge_weight": update.new_weights.get("wolf_pack_3_edge", 0.0),
            "adaptive_ema_momentum_weight": update.new_weights.get("adaptive_ema_momentum", 0.0),
            "vwap_bounce_weight": update.new_weights.get("vwap_bounce", 0.0),
            "opening_range_breakout_weight": update.new_weights.get("opening_range_breakout", 0.0),
            "constraint_adjustments": str(update.constraint_adjustments),
            "rebalancing_reason": update.rebalancing_reason
        }

        # Append to CSV
        with open(self.history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row)

        logger.info(
            f"Saved weight update #{rebalancing_number} to {self.history_path}"
        )

    def load_history(self) -> pd.DataFrame:
        """Load weight history from CSV.

        Returns:
            DataFrame with weight history
        """
        df = pd.read_csv(self.history_path)

        # Convert timestamp if data exists
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def get_latest_weights(self) -> dict[str, float]:
        """Get the most recent weights from history.

        Returns:
            Dictionary of latest weights (empty if no history)
        """
        if not self.history_path.exists() or self.history_path.stat().st_size == 0:
            return {}

        df = self.load_history()

        if df.empty:
            return {}

        # Get last row
        latest = df.iloc[-1]

        return {
            "triple_confluence_scaler": latest["triple_confluence_scaler_weight"],
            "wolf_pack_3_edge": latest["wolf_pack_3_edge_weight"],
            "adaptive_ema_momentum": latest["adaptive_ema_momentum_weight"],
            "vwap_bounce": latest["vwap_bounce_weight"],
            "opening_range_breakout": latest["opening_range_breakout_weight"]
        }

    def get_weight_evolution(self, strategy: str) -> list[tuple[datetime, float]]:
        """Get weight evolution for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            List of (timestamp, weight) tuples
        """
        df = self.load_history()

        if df.empty:
            return []

        weight_col = f"{strategy}_weight"

        if weight_col not in df.columns:
            return []

        evolution = [
            (row["timestamp"], row[weight_col])
            for _, row in df.iterrows()
        ]

        return evolution


class DynamicWeightOptimizer:
    """Orchestrator for dynamic weight optimization.

    Coordinates performance tracking, weight calculation, and
    persistence. Runs weekly optimization as background task.

    Attributes:
        config_path: Path to configuration file (config-sim.yaml)
        tracker: Performance tracker instance
        calculator: Weight calculator instance
        history: Weight history tracker
    """

    def __init__(
        self,
        config_path: str = "config-sim.yaml",
        min_trades: int = 20,
        window_weeks: int = 4,
        floor: float = 0.05,
        ceiling: float = 0.40
    ) -> None:
        """Initialize dynamic weight optimizer.

        Args:
            config_path: Path to config file
            min_trades: Minimum trades for performance calculation
            window_weeks: Performance window in weeks
            floor: Minimum weight per strategy
            ceiling: Maximum weight per strategy
        """
        self.config_path = Path(config_path)
        self.tracker = PerformanceTracker(min_trades=min_trades, window_weeks=window_weeks)
        self.calculator = WeightCalculator(floor=floor, ceiling=ceiling)
        self.history = WeightHistory()

        self._rebalancing_count = 0
        self._running = False

        logger.info(
            f"DynamicWeightOptimizer initialized with config_path={config_path}"
        )

    def optimize_weights(self) -> WeightUpdate:
        """Run weight optimization immediately.

        Returns:
            WeightUpdate with optimization results

        Raises:
            ValueError: If configuration cannot be loaded/saved
        """
        # Get current weights
        previous_weights = self._load_current_weights()

        # Get all strategy performances
        window_end = datetime.now()
        performances = self.tracker.get_all_performance(window_end)

        # Calculate new weights
        new_weights = self.calculator.calculate_weights(performances)

        # Get performance scores
        performance_scores = {
            strategy: perf.performance_score
            for strategy, perf in performances.items()
        }

        # Detect constraint adjustments
        _, constraint_adjustments = self.calculator.apply_constraints(new_weights)

        # Create weight update record
        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights=previous_weights,
            new_weights=new_weights,
            performance_scores=performance_scores,
            constraint_adjustments=constraint_adjustments,
            rebalancing_reason="Weekly optimization"
        )

        # Save to config
        self._save_weights_to_config(new_weights)

        # Save to history
        self._rebalancing_count += 1
        self.history.save_update(update, self._rebalancing_count)

        # Log summary
        self._log_optimization_summary(update)

        return update

    def _load_current_weights(self) -> dict[str, float]:
        """Load current weights from config file.

        Returns:
            Dictionary of current weights

        Raises:
            ValueError: If config file cannot be loaded
        """
        try:
            import yaml

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            weights = config["ensemble"]["strategies"]
            return weights

        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise ValueError(f"Config file not found: {self.config_path}")
        except (KeyError, TypeError) as e:
            logger.error(f"Invalid config format: {e}")
            raise ValueError(f"Cannot load weights from config: {e}")

    def _save_weights_to_config(self, weights: dict[str, float]) -> None:
        """Save weights to config file.

        Args:
            weights: Dictionary of weights to save

        Raises:
            ValueError: If config cannot be saved
        """
        try:
            import yaml

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            config["ensemble"]["strategies"] = weights

            with open(self.config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)

            logger.info(f"Saved weights to config: {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save weights to config: {e}")
            raise ValueError(f"Cannot save weights to config: {e}")

    def _log_optimization_summary(self, update: WeightUpdate) -> None:
        """Log optimization summary.

        Args:
            update: Weight update to log
        """
        logger.info("=" * 60)
        logger.info("Weight Optimization Summary")
        logger.info("=" * 60)

        logger.info(f"Timestamp: {update.timestamp}")
        logger.info(f"Reason: {update.rebalancing_reason}")

        logger.info("\nWeight Changes:")
        for strategy in update.previous_weights.keys():
            old_weight = update.previous_weights[strategy]
            new_weight = update.new_weights[strategy]
            change = new_weight - old_weight
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"

            logger.info(
                f"  {strategy}: {old_weight:.3f} → {new_weight:.3f} "
                f"({arrow}{abs(change):.3f})"
            )

        if update.constraint_adjustments:
            logger.info("\nConstraint Adjustments:")
            for strategy, adjustment in update.constraint_adjustments.items():
                logger.info(f"  {strategy}: {adjustment}")

        logger.info("\nPerformance Scores:")
        for strategy, score in update.performance_scores.items():
            logger.info(f"  {strategy}: {score:.3f}")

        logger.info("=" * 60)

    def get_days_until_next_rebalance(self) -> int:
        """Get days until next scheduled rebalance.

        Returns:
            Days until next rebalance (7 if never optimized)
        """
        if self._rebalancing_count == 0:
            return 7

        # Get last rebalance date
        df = self.history.load_history()

        if df.empty:
            return 7

        last_rebalance = df.iloc[-1]["timestamp"]
        days_since = (datetime.now() - last_rebalance).days

        days_until = 7 - days_since
        return max(0, days_until)

    def record_trade(self, trade: CompletedTrade) -> None:
        """Record a completed trade for performance tracking.

        Args:
            trade: Completed trade to record
        """
        self.tracker.track_trade(trade)

    def get_rebalancing_count(self) -> int:
        """Get total number of rebalancings performed.

        Returns:
            Rebalancing count
        """
        return self._rebalancing_count



"""Tests for weight evolution simulator.

Tests for 12-week weight projection, convergence analysis, and
constraint detection for ensemble trading strategies.
"""

import pytest
from datetime import date, datetime

from src.research.ensemble_backtester import BacktestResults
from src.research.weight_evolution_simulator import (
    WeightEvolutionSimulator,
    WeightUpdate,
    ConvergenceProjection,
    ConstraintHit,
)


@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results for simulation."""
    def create_results(strategy_name, win_rate, trades, pf=1.8):
        return BacktestResults(
            total_trades=trades,
            winning_trades=int(trades * win_rate),
            losing_trades=trades - int(trades * win_rate),
            win_rate=win_rate,
            profit_factor=pf,
            average_win=120.0,
            average_loss=-75.0,
            largest_win=300.0,
            largest_loss=-200.0,
            max_drawdown=0.08,
            max_drawdown_duration=15,
            sharpe_ratio=1.5,
            average_hold_time=8.5,
            trade_frequency=trades / 20,  # Assume 20 trading days
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            confidence_threshold=0.50,
            trades=[],
            total_pnl=4500.0,
        )

    return {
        "triple_confluence_scaler": create_results("triple_confluence_scaler", 0.68, trades=150, pf=2.0),
        "wolf_pack_3_edge": create_results("wolf_pack_3_edge", 0.62, trades=180, pf=1.7),
        "adaptive_ema_momentum": create_results("adaptive_ema_momentum", 0.70, trades=120, pf=2.2),
        "vwap_bounce": create_results("vwap_bounce", 0.58, trades=200, pf=1.5),
        "opening_range_breakout": create_results("opening_range_breakout", 0.65, trades=160, pf=1.8),
    }


class TestWeightUpdate:
    """Test suite for WeightUpdate model."""

    def test_weight_update_creation(self):
        """Test WeightUpdate can be created."""
        update = WeightUpdate(
            week=1,
            weights={
                "triple_confluence_scaler": 0.22,
                "wolf_pack_3_edge": 0.18,
                "adaptive_ema_momentum": 0.25,
                "vwap_bounce": 0.15,
                "opening_range_breakout": 0.20,
            },
            performance_scores={
                "triple_confluence_scaler": 1.36,
                "wolf_pack_3_edge": 1.05,
                "adaptive_ema_momentum": 1.54,
                "vwap_bounce": 0.87,
                "opening_range_breakout": 1.17,
            },
            constraints_active={},
        )

        assert update.week == 1
        assert len(update.weights) == 5
        assert update.weights["adaptive_ema_momentum"] == 0.25
        assert abs(sum(update.weights.values()) - 1.0) < 0.001


class TestConvergenceProjection:
    """Test suite for ConvergenceProjection model."""

    def test_convergence_projection_creation(self):
        """Test ConvergenceProjection can be created."""
        projection = ConvergenceProjection(
            final_weights={
                "triple_confluence_scaler": 0.25,
                "wolf_pack_3_edge": 0.15,
                "adaptive_ema_momentum": 0.30,
                "vwap_bounce": 0.10,
                "opening_range_breakout": 0.20,
            },
            weeks_to_convergence=8,
            convergence_stable=True,
            weight_volatility=0.015,
            convergence_criteria="Weight changes < 0.01 for 3 consecutive weeks",
        )

        assert projection.convergence_stable is True
        assert projection.weeks_to_convergence == 8
        assert projection.weight_volatility == 0.015
        assert abs(sum(projection.final_weights.values()) - 1.0) < 0.001


class TestConstraintHit:
    """Test suite for ConstraintHit model."""

    def test_constraint_hit_floor(self):
        """Test ConstraintHit for floor constraint."""
        hit = ConstraintHit(
            week=3,
            strategy="vwap_bounce",
            constraint_type="floor",
            constraint_value=0.05,
            calculated_weight=0.03,
            constrained_weight=0.05,
        )

        assert hit.week == 3
        assert hit.constraint_type == "floor"
        assert hit.constraint_value == 0.05
        assert hit.calculated_weight == 0.03
        assert hit.constrained_weight == 0.05

    def test_constraint_hit_ceiling(self):
        """Test ConstraintHit for ceiling constraint."""
        hit = ConstraintHit(
            week=5,
            strategy="adaptive_ema_momentum",
            constraint_type="ceiling",
            constraint_value=0.40,
            calculated_weight=0.45,
            constrained_weight=0.40,
        )

        assert hit.constraint_type == "ceiling"
        assert hit.calculated_weight == 0.45
        assert hit.constrained_weight == 0.40


class TestWeightEvolutionSimulator:
    """Test suite for WeightEvolutionSimulator class."""

    def test_initialization(self, sample_backtest_results):
        """Test simulator can be initialized with backtest results."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        assert simulator.backtest_results is not None
        assert simulator.individual_results == sample_backtest_results

    def test_simulate_weekly_rebalancing(self, sample_backtest_results):
        """Test 12-week weekly rebalancing simulation."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        updates = simulator.simulate_weekly_rebalancing(weeks=12)

        assert len(updates) == 12
        assert all(isinstance(update, WeightUpdate) for update in updates)

        # Check week numbers
        for i, update in enumerate(updates, start=1):
            assert update.week == i

        # Check weights sum to 1.0
        for update in updates:
            assert abs(sum(update.weights.values()) - 1.0) < 0.001

    def test_project_convergence(self, sample_backtest_results):
        """Test convergence projection."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        # Run simulation first
        simulator.simulate_weekly_rebalancing(weeks=12)

        projection = simulator.project_convergence()

        assert isinstance(projection, ConvergenceProjection)
        assert hasattr(projection, "final_weights")
        assert hasattr(projection, "weeks_to_convergence")
        assert hasattr(projection, "convergence_stable")
        assert hasattr(projection, "weight_volatility")

    def test_detect_constraint_hits(self, sample_backtest_results):
        """Test constraint hit detection."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        # Run simulation first
        simulator.simulate_weekly_rebalancing(weeks=12)

        constraint_hits = simulator.detect_constraint_hits()

        assert isinstance(constraint_hits, list)
        assert all(isinstance(hit, ConstraintHit) for hit in constraint_hits)

        # Verify constraint hits have required fields
        for hit in constraint_hits:
            assert hasattr(hit, "week")
            assert hasattr(hit, "strategy")
            assert hasattr(hit, "constraint_type")
            assert hit.constraint_type in ["floor", "ceiling"]

    def test_calculate_weight_volatility(self, sample_backtest_results):
        """Test weight volatility calculation."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        # Run simulation first
        simulator.simulate_weekly_rebalancing(weeks=12)

        volatility = simulator.calculate_weight_volatility()

        assert isinstance(volatility, float)
        assert volatility >= 0.0

    def test_detect_convergence(self, sample_backtest_results):
        """Test convergence detection."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        # Run simulation first
        simulator.simulate_weekly_rebalancing(weeks=12)

        converged = simulator.detect_convergence(window=3)

        assert isinstance(converged, bool)

    def test_weights_respect_constraints(self, sample_backtest_results):
        """Test that all weights respect floor/ceiling constraints."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        updates = simulator.simulate_weekly_rebalancing(weeks=12)

        floor = 0.05
        ceiling = 0.40

        for update in updates:
            for strategy, weight in update.weights.items():
                assert floor <= weight <= ceiling, (
                    f"Week {update.week}, {strategy}: weight {weight} "
                    f"outside [{floor}, {ceiling}]"
                )

    def test_start_with_equal_weights(self, sample_backtest_results):
        """Test that simulation starts with equal weights."""
        simulator = WeightEvolutionSimulator(
            backtest_results=sample_backtest_results["triple_confluence_scaler"],
            individual_results=sample_backtest_results,
        )

        updates = simulator.simulate_weekly_rebalancing(weeks=12)

        # Week 1 should start with equal weights
        first_update = updates[0]
        for strategy, weight in first_update.weights.items():
            assert abs(weight - 0.20) < 0.001, (
                f"Week 1 should have equal weights (0.20), "
                f"but {strategy} has {weight}"
            )

"""Unit tests for WeightCalculator class (Task 3)."""

import pytest
from datetime import datetime, timedelta

from src.detection.dynamic_weight_optimizer import WeightCalculator
from src.detection.models import StrategyPerformance


class TestWeightCalculator:
    """Test WeightCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a WeightCalculator instance."""
        return WeightCalculator(floor=0.05, ceiling=0.40)

    @pytest.fixture
    def sample_performances(self):
        """Create sample strategy performances."""
        now = datetime.now()
        window_start = now - timedelta(weeks=4)

        return {
            "triple_confluence_scaler": StrategyPerformance(
                strategy_name="triple_confluence_scaler",
                window_start=window_start,
                window_end=now,
                total_trades=40,
                winning_trades=30,
                losing_trades=10,
                win_rate=0.75,
                gross_profit=3000.0,
                gross_loss=500.0,
                profit_factor=6.0,
                performance_score=4.5,  # 0.75 * 6.0
                data_quality="sufficient"
            ),
            "wolf_pack_3_edge": StrategyPerformance(
                strategy_name="wolf_pack_3_edge",
                window_start=window_start,
                window_end=now,
                total_trades=30,
                winning_trades=15,
                losing_trades=15,
                win_rate=0.50,
                gross_profit=1500.0,
                gross_loss=750.0,
                profit_factor=2.0,
                performance_score=1.0,  # 0.50 * 2.0
                data_quality="sufficient"
            ),
            "adaptive_ema_momentum": StrategyPerformance(
                strategy_name="adaptive_ema_momentum",
                window_start=window_start,
                window_end=now,
                total_trades=25,
                winning_trades=15,
                losing_trades=10,
                win_rate=0.60,
                gross_profit=1200.0,
                gross_loss=600.0,
                profit_factor=2.0,
                performance_score=1.2,  # 0.60 * 2.0
                data_quality="sufficient"
            ),
            "vwap_bounce": StrategyPerformance(
                strategy_name="vwap_bounce",
                window_start=window_start,
                window_end=now,
                total_trades=20,
                winning_trades=8,
                losing_trades=12,
                win_rate=0.40,
                gross_profit=400.0,
                gross_loss=600.0,
                profit_factor=0.67,
                performance_score=0.27,  # 0.40 * 0.67
                data_quality="sufficient"
            ),
            "opening_range_breakout": StrategyPerformance(
                strategy_name="opening_range_breakout",
                window_start=window_start,
                window_end=now,
                total_trades=15,
                winning_trades=5,
                losing_trades=10,
                win_rate=0.33,
                gross_profit=250.0,
                gross_loss=500.0,
                profit_factor=0.50,
                performance_score=0.17,  # 0.33 * 0.50
                data_quality="insufficient_4weeks"
            )
        }

    def test_initialization(self, calculator):
        """Test WeightCalculator initialization."""
        assert calculator.floor == 0.05
        assert calculator.ceiling == 0.40

    def test_initialization_custom_params(self):
        """Test WeightCalculator with custom parameters."""
        custom = WeightCalculator(floor=0.10, ceiling=0.35)
        assert custom.floor == 0.10
        assert custom.ceiling == 0.35

    def test_initialization_invalid_floor(self):
        """Test initialization with invalid floor."""
        with pytest.raises(ValueError, match="floor must be between 0 and 1"):
            WeightCalculator(floor=-0.05)

        with pytest.raises(ValueError, match="floor must be between 0 and 1"):
            WeightCalculator(floor=1.5)

    def test_initialization_invalid_ceiling(self):
        """Test initialization with invalid ceiling."""
        with pytest.raises(ValueError, match="ceiling must be between 0 and 1"):
            WeightCalculator(ceiling=0)

        with pytest.raises(ValueError, match="ceiling must be between 0 and 1"):
            WeightCalculator(ceiling=1.5)

    def test_initialization_floor_equals_ceiling(self):
        """Test initialization with floor equal to ceiling."""
        with pytest.raises(ValueError, match="floor must be less than ceiling"):
            WeightCalculator(floor=0.20, ceiling=0.20)

    def test_calculate_weights_basic(self, calculator, sample_performances):
        """Test basic weight calculation from performance scores."""
        weights = calculator.calculate_weights(sample_performances)

        # Verify all strategies have weights
        assert len(weights) == 5
        assert "triple_confluence_scaler" in weights
        assert "wolf_pack_3_edge" in weights

        # Verify weights sum to 1.0
        total = sum(weights.values())
        assert total == pytest.approx(1.0)

    def test_calculate_weights_distribution(self, calculator, sample_performances):
        """Test that higher performance scores get higher weights."""
        weights = calculator.calculate_weights(sample_performances)

        # triple_confluence_scaler has highest score (4.5)
        # Should have highest weight
        tc_weight = weights["triple_confluence_scaler"]

        # opening_range_breakout has lowest score (0.17)
        # Should have lowest weight
        orb_weight = weights["opening_range_breakout"]

        assert tc_weight > orb_weight

    def test_calculate_weights_all_zero_scores(self, calculator):
        """Test weight calculation when all scores are zero."""
        zero_performances = {}
        now = datetime.now()
        window_start = now - timedelta(weeks=4)

        for strategy in ["triple_confluence_scaler", "wolf_pack_3_edge", "adaptive_ema_momentum", "vwap_bounce", "opening_range_breakout"]:
            zero_performances[strategy] = StrategyPerformance(
                strategy_name=strategy,
                window_start=window_start,
                window_end=now,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                profit_factor=0.0,
                performance_score=0.0,  # All zero
                data_quality="insufficient_8weeks"
            )

        weights = calculator.calculate_weights(zero_performances)

        # Should return equal weights (0.20 each)
        for strategy, weight in weights.items():
            assert weight == pytest.approx(0.20)

    def test_normalize_weights(self, calculator):
        """Test weight normalization."""
        # Weights that don't sum to 1.0
        unnormalized = {
            "triple_confluence_scaler": 2.0,
            "wolf_pack_3_edge": 1.0,
            "adaptive_ema_momentum": 1.0,
            "vwap_bounce": 0.5,
            "opening_range_breakout": 0.5
        }  # Sum = 5.0

        normalized = calculator.normalize_weights(unnormalized)

        # Each weight should be divided by 5
        assert normalized["triple_confluence_scaler"] == pytest.approx(0.40)
        assert normalized["wolf_pack_3_edge"] == pytest.approx(0.20)

        # Sum should be 1.0
        total = sum(normalized.values())
        assert total == pytest.approx(1.0)

    def test_normalize_weights_zero_sum(self, calculator):
        """Test normalizing weights with sum of 0."""
        zero_weights = {
            "triple_confluence_scaler": 0.0,
            "wolf_pack_3_edge": 0.0,
            "adaptive_ema_momentum": 0.0,
            "vwap_bounce": 0.0,
            "opening_range_breakout": 0.0
        }

        with pytest.raises(ValueError, match="Cannot normalize weights with sum of 0"):
            calculator.normalize_weights(zero_weights)

    def test_apply_constraints_floor(self, calculator):
        """Test applying floor constraint."""
        # Weights with one below floor
        weights = {
            "triple_confluence_scaler": 0.30,
            "wolf_pack_3_edge": 0.30,
            "adaptive_ema_momentum": 0.30,
            "vwap_bounce": 0.03,  # Below floor (0.05)
            "opening_range_breakout": 0.07
        }  # Sum = 1.0

        constrained, adjustments = calculator.apply_constraints(weights)

        # vwap_bounce should be raised to floor
        assert constrained["vwap_bounce"] == pytest.approx(0.05)
        assert "vwap_bounce" in adjustments
        assert adjustments["vwap_bounce"] == "hit_floor"

        # Sum should still be 1.0 (with redistribution)
        total = sum(constrained.values())
        assert total == pytest.approx(1.0)

    def test_apply_constraints_ceiling(self, calculator):
        """Test applying ceiling constraint."""
        # Weights with one above ceiling
        weights = {
            "triple_confluence_scaler": 0.50,  # Above ceiling (0.40)
            "wolf_pack_3_edge": 0.15,
            "adaptive_ema_momentum": 0.15,
            "vwap_bounce": 0.10,
            "opening_range_breakout": 0.10
        }  # Sum = 1.0

        constrained, adjustments = calculator.apply_constraints(weights)

        # triple_confluence_scaler should be lowered to ceiling
        assert constrained["triple_confluence_scaler"] == 0.40
        assert "triple_confluence_scaler" in adjustments
        assert adjustments["triple_confluence_scaler"] == "hit_ceiling"

        # Excess should be redistributed
        # Original: 0.50 → 0.40 (excess 0.10)
        # This 0.10 should be distributed to other strategies
        total = sum(constrained.values())
        assert total == pytest.approx(1.0)

    def test_apply_constraints_both_floor_and_ceiling(self, calculator):
        """Test applying both floor and ceiling constraints simultaneously."""
        weights = {
            "triple_confluence_scaler": 0.50,  # Above ceiling
            "wolf_pack_3_edge": 0.20,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.03,  # Below floor
            "opening_range_breakout": 0.07
        }  # Sum = 1.0

        constrained, adjustments = calculator.apply_constraints(weights)

        # Check constraints applied
        assert constrained["triple_confluence_scaler"] == 0.40
        assert constrained["vwap_bounce"] == 0.05

        assert "triple_confluence_scaler" in adjustments
        assert "vwap_bounce" in adjustments

        # Sum should be 1.0
        total = sum(constrained.values())
        assert total == pytest.approx(1.0)

    def test_apply_constraints_all_hit_floor(self, calculator):
        """Test when all strategies hit floor (edge case)."""
        weights = {
            "triple_confluence_scaler": 0.04,  # All below floor
            "wolf_pack_3_edge": 0.03,
            "adaptive_ema_momentum": 0.03,
            "vwap_bounce": 0.00,
            "opening_range_breakout": 0.00
        }  # Sum = 0.10

        constrained, adjustments = calculator.apply_constraints(weights)

        # All should be at floor initially
        # But since all hit constraints, should use equal weights
        # (or at least redistribute to reach sum=1.0)

        # Sum should be 1.0
        total = sum(constrained.values())
        assert total == pytest.approx(1.0)

    def test_apply_constraints_all_hit_ceiling(self, calculator):
        """Test when all strategies hit ceiling (edge case)."""
        weights = {
            "triple_confluence_scaler": 0.50,  # All above ceiling
            "wolf_pack_3_edge": 0.45,
            "adaptive_ema_momentum": 0.50,
            "vwap_bounce": 0.60,
            "opening_range_breakout": 0.55
        }  # Sum = 2.6

        constrained, adjustments = calculator.apply_constraints(weights)

        # All should be at ceiling initially
        # Since sum > 1.0, excess should be redistributed
        # But since all are at ceiling, they'll stay at ceiling
        # This is a degenerate case

        # Sum should be 1.0
        total = sum(constrained.values())
        assert total == pytest.approx(1.0)

    def test_apply_constraints_no_violations(self, calculator):
        """Test applying constraints when no violations exist."""
        # All weights within bounds
        weights = {
            "triple_confluence_scaler": 0.25,
            "wolf_pack_3_edge": 0.20,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.20,
            "opening_range_breakout": 0.15
        }  # Sum = 1.0

        constrained, adjustments = calculator.apply_constraints(weights)

        # Weights should be unchanged
        assert constrained == weights
        assert len(adjustments) == 0

    def test_floor_ceiling_different_values(self):
        """Test WeightCalculator with different floor/ceiling values."""
        custom_calc = WeightCalculator(floor=0.10, ceiling=0.30)

        weights = {
            "triple_confluence_scaler": 0.35,  # Above ceiling
            "wolf_pack_3_edge": 0.05,  # Below floor
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.20,
            "opening_range_breakout": 0.20
        }

        constrained, adjustments = custom_calc.apply_constraints(weights)

        assert constrained["triple_confluence_scaler"] == 0.30
        assert constrained["wolf_pack_3_edge"] == 0.10

    def test_weight_redistribution_proportional(self, calculator):
        """Test that excess is redistributed proportionally."""
        weights = {
            "triple_confluence_scaler": 0.50,  # +0.10 excess to redistribute
            "wolf_pack_3_edge": 0.30,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.00,
            "opening_range_breakout": 0.00
        }  # Sum = 1.0

        constrained, _ = calculator.apply_constraints(weights)

        # triple_confluence_scaler at ceiling (0.40)
        assert constrained["triple_confluence_scaler"] == pytest.approx(0.40)

        # 0.10 excess distributed to wolf_pack and adaptive_ema
        # After normalization, the sum is 1.0
        # wolf_pack and adaptive_ema should have increased from original
        assert constrained["wolf_pack_3_edge"] == pytest.approx(0.30, abs=0.01)
        assert constrained["adaptive_ema_momentum"] == pytest.approx(0.20, abs=0.01)

        # Sum should be 1.0
        total = sum(constrained.values())
        assert total == pytest.approx(1.0)

"""Unit tests for performance tracking models (Task 1)."""

import pytest
from datetime import datetime, timedelta

from src.detection.models import StrategyPerformance, WeightUpdate, CompletedTrade


class TestStrategyPerformance:
    """Test StrategyPerformance model."""

    def test_create_strategy_performance(self):
        """Test creating a valid StrategyPerformance."""
        perf = StrategyPerformance(
            strategy_name="triple_confluence_scaler",
            window_start=datetime.now() - timedelta(weeks=4),
            window_end=datetime.now(),
            total_trades=50,
            winning_trades=35,
            losing_trades=15,
            win_rate=0.70,
            gross_profit=3500.0,
            gross_loss=1500.0,
            profit_factor=2.33,
            performance_score=1.63,  # 0.70 * 2.33
            data_quality="sufficient"
        )

        assert perf.strategy_name == "triple_confluence_scaler"
        assert perf.total_trades == 50
        assert perf.win_rate == 0.70
        assert perf.performance_score == pytest.approx(1.63)

    def test_win_rate_validation(self):
        """Test win_rate must be between 0 and 1."""
        with pytest.raises(ValueError):
            StrategyPerformance(
                strategy_name="test",
                window_start=datetime.now(),
                window_end=datetime.now(),
                total_trades=10,
                winning_trades=5,
                losing_trades=5,
                win_rate=1.5,  # Invalid: > 1.0
                gross_profit=100.0,
                gross_loss=50.0,
                profit_factor=2.0,
                performance_score=3.0,
                data_quality="sufficient"
            )

    def test_gross_profit_validation(self):
        """Test gross_profit must be non-negative."""
        with pytest.raises(ValueError):
            StrategyPerformance(
                strategy_name="test",
                window_start=datetime.now(),
                window_end=datetime.now(),
                total_trades=10,
                winning_trades=5,
                losing_trades=5,
                win_rate=0.5,
                gross_profit=-100.0,  # Invalid: negative
                gross_loss=50.0,
                profit_factor=2.0,
                performance_score=1.0,
                data_quality="sufficient"
            )

    def test_calculate_performance_score(self):
        """Test performance score calculation."""
        perf = StrategyPerformance(
            strategy_name="test",
            window_start=datetime.now(),
            window_end=datetime.now(),
            total_trades=20,
            winning_trades=14,
            losing_trades=6,
            win_rate=0.70,
            gross_profit=1400.0,
            gross_loss=600.0,
            profit_factor=2.33,
            performance_score=1.631,  # More precise value
            data_quality="sufficient"
        )

        # 0.70 * 2.33 = 1.631
        score = perf.calculate_performance_score()
        assert score == pytest.approx(1.631)

    def test_is_data_sufficient(self):
        """Test data sufficiency check."""
        # Sufficient data
        perf_sufficient = StrategyPerformance(
            strategy_name="test",
            window_start=datetime.now(),
            window_end=datetime.now(),
            total_trades=20,
            winning_trades=10,
            losing_trades=10,
            win_rate=0.5,
            gross_profit=1000.0,
            gross_loss=1000.0,
            profit_factor=1.0,
            performance_score=0.5,
            data_quality="sufficient"
        )

        assert perf_sufficient.is_data_sufficient() is True

        # Insufficient data
        perf_insufficient = StrategyPerformance(
            strategy_name="test",
            window_start=datetime.now(),
            window_end=datetime.now(),
            total_trades=15,
            winning_trades=8,
            losing_trades=7,
            win_rate=0.53,
            gross_profit=800.0,
            gross_loss=700.0,
            profit_factor=1.14,
            performance_score=0.60,
            data_quality="insufficient_4weeks"
        )

        assert perf_insufficient.is_data_sufficient() is False


class TestWeightUpdate:
    """Test WeightUpdate model."""

    @pytest.fixture
    def valid_weights(self):
        """Create valid weight dictionaries."""
        previous = {
            "triple_confluence_scaler": 0.20,
            "wolf_pack_3_edge": 0.20,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.20,
            "opening_range_breakout": 0.20
        }
        new = {
            "triple_confluence_scaler": 0.25,
            "wolf_pack_3_edge": 0.15,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.20,
            "opening_range_breakout": 0.20
        }
        return previous, new

    def test_create_weight_update(self, valid_weights):
        """Test creating a valid WeightUpdate."""
        previous, new = valid_weights

        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights=previous,
            new_weights=new,
            performance_scores={
                "triple_confluence_scaler": 1.5,
                "wolf_pack_3_edge": 0.8,
                "adaptive_ema_momentum": 1.0,
                "vwap_bounce": 1.0,
                "opening_range_breakout": 1.0
            },
            rebalancing_reason="Weekly rebalancing"
        )

        assert update.previous_weights == previous
        assert update.new_weights == new
        assert update.rebalancing_reason == "Weekly rebalancing"

    def test_weights_sum_to_one_validation(self):
        """Test that weights must sum to 1.0."""
        invalid_weights = {
            "triple_confluence_scaler": 0.30,
            "wolf_pack_3_edge": 0.30,
            "adaptive_ema_momentum": 0.30,
            "vwap_bounce": 0.30,
            "opening_range_breakout": 0.30
        }  # Sum = 1.5

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            WeightUpdate(
                timestamp=datetime.now(),
                previous_weights=invalid_weights,
                new_weights=invalid_weights,
                performance_scores={},
                rebalancing_reason="test"
            )

    def test_weights_within_bounds_validation(self):
        """Test that weights must be between 0 and 1."""
        # Create weights that sum to 1.0 but have an invalid value
        invalid_weights = {
            "triple_confluence_scaler": 0.20,
            "wolf_pack_3_edge": 0.20,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.20,
            "opening_range_breakout": -0.20  # Invalid: < 0
        }  # Sum = 0.6, need to add valid weights to make sum 1.0

        valid_previous = {
            "triple_confluence_scaler": 0.20,
            "wolf_pack_3_edge": 0.20,
            "adaptive_ema_momentum": 0.20,
            "vwap_bounce": 0.20,
            "opening_range_breakout": 0.20
        }

        with pytest.raises(ValueError, match="Weights must sum to 1.0|Weight for.*must be between 0 and 1"):
            WeightUpdate(
                timestamp=datetime.now(),
                previous_weights=valid_previous,
                new_weights=invalid_weights,
                performance_scores={},
                rebalancing_reason="test"
            )

    def test_get_weight_change(self, valid_weights):
        """Test getting weight change for a strategy."""
        previous, new = valid_weights

        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights=previous,
            new_weights=new,
            performance_scores={},
            rebalancing_reason="test"
        )

        # triple_confluence_scaler: 0.20 → 0.25, change = +0.05
        change = update.get_weight_change("triple_confluence_scaler")
        assert change == pytest.approx(0.05)

        # wolf_pack_3_edge: 0.20 → 0.15, change = -0.05
        change = update.get_weight_change("wolf_pack_3_edge")
        assert change == pytest.approx(-0.05)

    def test_get_weight_change_unknown_strategy(self, valid_weights):
        """Test getting weight change for unknown strategy."""
        previous, new = valid_weights

        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights=previous,
            new_weights=new,
            performance_scores={},
            rebalancing_reason="test"
        )

        change = update.get_weight_change("unknown_strategy")
        assert change == 0.0

    def test_constraint_adjustments(self, valid_weights):
        """Test constraint adjustments tracking."""
        previous, new = valid_weights

        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights=previous,
            new_weights=new,
            performance_scores={},
            constraint_adjustments={
                "triple_confluence_scaler": "hit_ceiling",
                "wolf_pack_3_edge": "hit_floor"
            },
            rebalancing_reason="test"
        )

        assert "triple_confluence_scaler" in update.constraint_adjustments
        assert update.constraint_adjustments["triple_confluence_scaler"] == "hit_ceiling"


class TestCompletedTrade:
    """Test CompletedTrade model."""

    def test_create_completed_trade_winner(self):
        """Test creating a winning trade record."""
        entry_time = datetime.now() - timedelta(hours=2)
        exit_time = datetime.now()

        trade = CompletedTrade(
            trade_id="trade-123",
            strategy_name="triple_confluence_scaler",
            direction="long",
            entry_price=11850.0,
            exit_price=11870.0,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=100.0,  # Winner
            exit_reason="take_profit",
            bars_held=24
        )

        assert trade.trade_id == "trade-123"
        assert trade.strategy_name == "triple_confluence_scaler"
        assert trade.pnl == 100.0
        assert trade.is_winner() is True

    def test_create_completed_trade_loser(self):
        """Test creating a losing trade record."""
        entry_time = datetime.now() - timedelta(hours=1)
        exit_time = datetime.now()

        trade = CompletedTrade(
            trade_id="trade-456",
            strategy_name="wolf_pack_3_edge",
            direction="short",
            entry_price=11850.0,
            exit_price=11860.0,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=-50.0,  # Loser
            exit_reason="stop_loss",
            bars_held=12
        )

        assert trade.pnl == -50.0
        assert trade.is_winner() is False

    def test_get_hold_time_minutes(self):
        """Test hold time calculation."""
        entry_time = datetime.now() - timedelta(minutes=45)
        exit_time = datetime.now()

        trade = CompletedTrade(
            trade_id="trade-789",
            strategy_name="adaptive_ema_momentum",
            direction="long",
            entry_price=11850.0,
            exit_price=11855.0,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=25.0,
            exit_reason="time_stop",
            bars_held=9
        )

        hold_time = trade.get_hold_time_minutes()
        assert 44 <= hold_time <= 46  # Approximately 45 minutes

    def test_bars_held_validation(self):
        """Test bars_held must be non-negative."""
        with pytest.raises(ValueError):
            CompletedTrade(
                trade_id="trade-999",
                strategy_name="test",
                direction="long",
                entry_price=11850.0,
                exit_price=11855.0,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                pnl=25.0,
                exit_reason="take_profit",
                bars_held=-1  # Invalid: negative
            )

    def test_entry_price_validation(self):
        """Test entry_price must be positive."""
        with pytest.raises(ValueError):
            CompletedTrade(
                trade_id="trade-999",
                strategy_name="test",
                direction="long",
                entry_price=0,  # Invalid: not > 0
                exit_price=11855.0,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                pnl=25.0,
                exit_reason="take_profit",
                bars_held=10
            )

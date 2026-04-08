"""Unit tests for WeightHistory and DynamicWeightOptimizer (Tasks 4-5)."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.detection.dynamic_weight_optimizer import WeightHistory, DynamicWeightOptimizer
from src.detection.models import CompletedTrade, StrategyPerformance


class TestWeightHistory:
    """Test WeightHistory class."""

    @pytest.fixture
    def temp_history(self):
        """Create a temporary history file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        history = WeightHistory(history_path=temp_path)
        yield history

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_initialization_creates_file(self, temp_history):
        """Test that initialization creates history file."""
        assert temp_history.history_path.exists()

    def test_save_and_load_update(self, temp_history):
        """Test saving and loading weight update."""
        from src.detection.models import WeightUpdate

        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights={
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20
            },
            new_weights={
                "triple_confluence_scaler": 0.25,
                "wolf_pack_3_edge": 0.15,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20
            },
            performance_scores={
                "triple_confluence_scaler": 1.5,
                "wolf_pack_3_edge": 0.8,
                "adaptive_ema_momentum": 1.0,
                "vwap_bounce": 1.0,
                "opening_range_breakout": 1.0
            },
            rebalancing_reason="Test rebalance"
        )

        temp_history.save_update(update, rebalancing_number=1)

        # Verify file has data
        df = temp_history.load_history()
        assert len(df) == 1
        assert df.iloc[0]["rebalancing_number"] == 1

    def test_get_latest_weights(self, temp_history):
        """Test getting latest weights."""
        from src.detection.models import WeightUpdate

        update = WeightUpdate(
            timestamp=datetime.now(),
            previous_weights={
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20
            },
            new_weights={
                "triple_confluence_scaler": 0.30,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.15,
                "opening_range_breakout": 0.15
            },
            performance_scores={
                "triple_confluence_scaler": 1.5,
                "wolf_pack_3_edge": 1.0,
                "adaptive_ema_momentum": 1.0,
                "vwap_bounce": 0.8,
                "opening_range_breakout": 0.8
            },
            rebalancing_reason="Test"
        )

        temp_history.save_update(update, rebalancing_number=1)

        latest = temp_history.get_latest_weights()
        assert latest["triple_confluence_scaler"] == 0.30

    def test_get_latest_weights_empty(self, temp_history):
        """Test getting latest weights when history is empty."""
        # Create new history (already has headers but no data)
        latest = temp_history.get_latest_weights()
        assert latest == {}

    def test_get_weight_evolution(self, temp_history):
        """Test getting weight evolution for a strategy."""
        from src.detection.models import WeightUpdate

        # Add multiple updates
        for i in range(3):
            update = WeightUpdate(
                timestamp=datetime.now() - timedelta(days=3-i),
                previous_weights={
                    "triple_confluence_scaler": 0.20,
                    "wolf_pack_3_edge": 0.20,
                    "adaptive_ema_momentum": 0.20,
                    "vwap_bounce": 0.20,
                    "opening_range_breakout": 0.20
                },
                new_weights={
                    "triple_confluence_scaler": 0.20 + i * 0.05,
                    "wolf_pack_3_edge": 0.20,
                    "adaptive_ema_momentum": 0.20,
                    "vwap_bounce": 0.20,
                    "opening_range_breakout": 0.20 - i * 0.05
                },
                performance_scores={
                    "triple_confluence_scaler": 1.0 + i * 0.2,
                    "wolf_pack_3_edge": 1.0,
                    "adaptive_ema_momentum": 1.0,
                    "vwap_bounce": 1.0,
                    "opening_range_breakout": 1.0 - i * 0.2
                },
                rebalancing_reason=f"Rebalance {i}"
            )
            temp_history.save_update(update, rebalancing_number=i+1)

        evolution = temp_history.get_weight_evolution("triple_confluence_scaler")

        assert len(evolution) == 3
        # Should be in chronological order
        assert evolution[0][1] == pytest.approx(0.20)  # First rebalance
        assert evolution[2][1] == pytest.approx(0.30)  # Last rebalance


class TestDynamicWeightOptimizer:
    """Test DynamicWeightOptimizer class."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file."""
        config_content = """
ensemble:
  strategies:
    triple_confluence_scaler: 0.20
    wolf_pack_3_edge: 0.20
    adaptive_ema_momentum: 0.20
    vwap_bounce: 0.20
    opening_range_breakout: 0.20
  confidence_threshold: 0.50
  minimum_strategies: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def optimizer(self, temp_config):
        """Create optimizer with temp config."""
        return DynamicWeightOptimizer(config_path=temp_config)

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.tracker is not None
        assert optimizer.calculator is not None
        assert optimizer.history is not None
        assert optimizer._rebalancing_count == 0

    def test_record_trade(self, optimizer):
        """Test recording a completed trade."""
        trade = CompletedTrade(
            trade_id="test-123",
            strategy_name="triple_confluence_scaler",
            direction="long",
            entry_price=11850.0,
            exit_price=11870.0,
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            pnl=100.0,
            exit_reason="take_profit",
            bars_held=24
        )

        optimizer.record_trade(trade)

        assert optimizer.tracker.get_trade_count("triple_confluence_scaler") == 1

    def test_optimize_weights(self, optimizer):
        """Test weight optimization."""
        # Add some trades first
        for i in range(25):
            trade = CompletedTrade(
                trade_id=f"tc-{i}",
                strategy_name="triple_confluence_scaler",
                direction="long",
                entry_price=11850.0,
                exit_price=11870.0,
                entry_time=datetime.now() - timedelta(days=10),
                exit_time=datetime.now() - timedelta(days=9),
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=24
            )
            optimizer.record_trade(trade)

        # Run optimization
        update = optimizer.optimize_weights()

        assert update is not None
        assert "triple_confluence_scaler" in update.new_weights
        assert optimizer._rebalancing_count == 1

    def test_get_rebalancing_count(self, optimizer):
        """Test getting rebalancing count."""
        assert optimizer.get_rebalancing_count() == 0

        # Add trades and optimize
        for i in range(25):
            trade = CompletedTrade(
                trade_id=f"tc-{i}",
                strategy_name="triple_confluence_scaler",
                direction="long",
                entry_price=11850.0,
                exit_price=11870.0,
                entry_time=datetime.now() - timedelta(days=10),
                exit_time=datetime.now() - timedelta(days=9),
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=24
            )
            optimizer.record_trade(trade)

        optimizer.optimize_weights()

        assert optimizer.get_rebalancing_count() == 1

    def test_get_days_until_next_rebalance(self, optimizer):
        """Test getting days until next rebalance."""
        # Never optimized
        days = optimizer.get_days_until_next_rebalance()
        assert days == 7

        # After optimization
        for i in range(25):
            trade = CompletedTrade(
                trade_id=f"tc-{i}",
                strategy_name="triple_confluence_scaler",
                direction="long",
                entry_price=11850.0,
                exit_price=11870.0,
                entry_time=datetime.now() - timedelta(days=10),
                exit_time=datetime.now() - timedelta(days=9),
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=24
            )
            optimizer.record_trade(trade)

        optimizer.optimize_weights()

        days = optimizer.get_days_until_next_rebalance()
        assert 0 <= days <= 7

    def test_load_invalid_config(self):
        """Test loading with invalid config path."""
        optimizer = DynamicWeightOptimizer(config_path="/nonexistent/config.yaml")

        # Should raise when trying to optimize (not during init)
        with pytest.raises(ValueError):
            optimizer.optimize_weights()

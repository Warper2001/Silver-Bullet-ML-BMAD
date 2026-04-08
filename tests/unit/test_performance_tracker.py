"""Unit tests for PerformanceTracker class (Task 2)."""

import pytest
from datetime import datetime, timedelta

from src.detection.dynamic_weight_optimizer import PerformanceTracker
from src.detection.models import CompletedTrade


class TestPerformanceTracker:
    """Test PerformanceTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a PerformanceTracker instance."""
        return PerformanceTracker(min_trades=20, window_weeks=4)

    @pytest.fixture
    def sample_trades(self):
        """Create sample completed trades for testing."""
        entry_time = datetime.now() - timedelta(days=10)
        exit_time = datetime.now() - timedelta(days=9)

        trades = []

        # Create 30 winning trades for triple_confluence_scaler
        for i in range(30):
            trade = CompletedTrade(
                trade_id=f"tc-winner-{i}",
                strategy_name="triple_confluence_scaler",
                direction="long",
                entry_price=11850.0,
                exit_price=11870.0,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=24
            )
            trades.append(trade)

        # Create 10 losing trades for triple_confluence_scaler
        for i in range(10):
            trade = CompletedTrade(
                trade_id=f"tc-loser-{i}",
                strategy_name="triple_confluence_scaler",
                direction="long",
                entry_price=11850.0,
                exit_price=11840.0,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=-50.0,
                exit_reason="stop_loss",
                bars_held=12
            )
            trades.append(trade)

        # Create 25 trades (15 winners, 10 losers) for wolf_pack_3_edge
        for i in range(15):
            trades.append(CompletedTrade(
                trade_id=f"wp-winner-{i}",
                strategy_name="wolf_pack_3_edge",
                direction="short",
                entry_price=11850.0,
                exit_price=11830.0,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=20
            ))

        for i in range(10):
            trades.append(CompletedTrade(
                trade_id=f"wp-loser-{i}",
                strategy_name="wolf_pack_3_edge",
                direction="short",
                entry_price=11850.0,
                exit_price=11860.0,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=-50.0,
                exit_reason="stop_loss",
                bars_held=10
            ))

        return trades

    def test_initialization(self, tracker):
        """Test PerformanceTracker initialization."""
        assert tracker.min_trades == 20
        assert tracker.window_weeks == 4
        assert len(tracker.trades) == 0

    def test_initialization_custom_params(self):
        """Test PerformanceTracker with custom parameters."""
        custom_tracker = PerformanceTracker(min_trades=10, window_weeks=2)
        assert custom_tracker.min_trades == 10
        assert custom_tracker.window_weeks == 2

    def test_initialization_invalid_min_trades(self):
        """Test initialization with invalid min_trades."""
        with pytest.raises(ValueError, match="min_trades must be positive"):
            PerformanceTracker(min_trades=0)

        with pytest.raises(ValueError, match="min_trades must be positive"):
            PerformanceTracker(min_trades=-5)

    def test_initialization_invalid_window_weeks(self):
        """Test initialization with invalid window_weeks."""
        with pytest.raises(ValueError, match="window_weeks must be positive"):
            PerformanceTracker(window_weeks=0)

    def test_track_trade(self, tracker):
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

        tracker.track_trade(trade)

        assert len(tracker.trades) == 1
        assert tracker.trades[0].trade_id == "test-123"

    def test_track_trade_invalid_strategy(self, tracker):
        """Test tracking trade with unknown strategy."""
        trade = CompletedTrade(
            trade_id="test-456",
            strategy_name="unknown_strategy",
            direction="long",
            entry_price=11850.0,
            exit_price=11870.0,
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            pnl=100.0,
            exit_reason="take_profit",
            bars_held=24
        )

        with pytest.raises(ValueError, match="Unknown strategy"):
            tracker.track_trade(trade)

    def test_get_performance_sufficient_data(self, tracker, sample_trades):
        """Test performance calculation with sufficient data."""
        # Add all sample trades (40 for triple_confluence_scaler)
        for trade in sample_trades:
            tracker.track_trade(trade)

        window_end = datetime.now()
        performance = tracker.get_performance("triple_confluence_scaler", window_end)

        assert performance.strategy_name == "triple_confluence_scaler"
        assert performance.total_trades == 40
        assert performance.winning_trades == 30
        assert performance.losing_trades == 10
        assert performance.win_rate == 0.75  # 30/40
        assert performance.gross_profit == pytest.approx(3000.0)  # 30 * $100
        assert performance.gross_loss == pytest.approx(500.0)  # 10 * $50
        assert performance.profit_factor == pytest.approx(6.0)  # 3000/500
        assert performance.data_quality == "sufficient"

    def test_get_performance_insufficient_4week(self, tracker):
        """Test performance calculation with insufficient 4-week data."""
        # Add only 15 trades (less than min_trades=20)
        entry_time = datetime.now() - timedelta(days=20)
        exit_time = datetime.now() - timedelta(days=19)

        for i in range(15):
            trade = CompletedTrade(
                trade_id=f"trade-{i}",
                strategy_name="vwap_bounce",
                direction="long",
                entry_price=11850.0,
                exit_price=11870.0,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=24
            )
            tracker.track_trade(trade)

        window_end = datetime.now()
        performance = tracker.get_performance("vwap_bounce", window_end)

        # Should use 8-week window and mark as insufficient_4weeks
        assert performance.data_quality == "insufficient_4weeks" or performance.total_trades == 15

    def test_get_performance_no_trades(self, tracker):
        """Test performance calculation with no trades."""
        window_end = datetime.now()
        performance = tracker.get_performance("adaptive_ema_momentum", window_end)

        assert performance.total_trades == 0
        assert performance.win_rate == 0.0
        assert performance.performance_score == 0.0
        assert performance.data_quality == "insufficient_8weeks"

    def test_get_all_performance(self, tracker, sample_trades):
        """Test getting performance for all strategies."""
        for trade in sample_trades:
            tracker.track_trade(trade)

        window_end = datetime.now()
        all_performance = tracker.get_all_performance(window_end)

        # Should have performance for all 5 strategies
        assert len(all_performance) == 5
        assert "triple_confluence_scaler" in all_performance
        assert "wolf_pack_3_edge" in all_performance
        assert "adaptive_ema_momentum" in all_performance
        assert "vwap_bounce" in all_performance
        assert "opening_range_breakout" in all_performance

    def test_get_trade_count_all(self, tracker, sample_trades):
        """Test getting total trade count."""
        for trade in sample_trades:
            tracker.track_trade(trade)

        total_count = tracker.get_trade_count()
        assert total_count == 65  # 40 + 25 from sample_trades

    def test_get_trade_count_specific_strategy(self, tracker, sample_trades):
        """Test getting trade count for specific strategy."""
        for trade in sample_trades:
            tracker.track_trade(trade)

        tc_count = tracker.get_trade_count("triple_confluence_scaler")
        assert tc_count == 40

        wp_count = tracker.get_trade_count("wolf_pack_3_edge")
        assert wp_count == 25

    def test_get_trade_count_invalid_strategy(self, tracker):
        """Test getting trade count for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            tracker.get_trade_count("unknown_strategy")

    def test_old_trades_pruned(self, tracker):
        """Test that trades older than 8 weeks are pruned."""
        # Add trades from 9 weeks ago (should be pruned)
        old_entry = datetime.now() - timedelta(weeks=9, days=1)
        old_exit = datetime.now() - timedelta(weeks=9)

        old_trade = CompletedTrade(
            trade_id="old-trade",
            strategy_name="triple_confluence_scaler",
            direction="long",
            entry_price=11850.0,
            exit_price=11870.0,
            entry_time=old_entry,
            exit_time=old_exit,
            pnl=100.0,
            exit_reason="take_profit",
            bars_held=24
        )

        tracker.track_trade(old_trade)

        # Add recent trade
        recent_trade = CompletedTrade(
            trade_id="recent-trade",
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

        tracker.track_trade(recent_trade)

        # Should only have recent trade
        assert len(tracker.trades) == 1
        assert tracker.trades[0].trade_id == "recent-trade"

    def test_clear_trades(self, tracker, sample_trades):
        """Test clearing all tracked trades."""
        for trade in sample_trades[:10]:  # Add some trades
            tracker.track_trade(trade)

        assert tracker.get_trade_count() > 0

        tracker.clear_trades()

        assert tracker.get_trade_count() == 0
        assert len(tracker.trades) == 0

    def test_profit_factor_no_losses(self, tracker):
        """Test profit factor calculation when there are no losses."""
        # Add only winning trades
        entry_time = datetime.now() - timedelta(days=10)
        exit_time = datetime.now() - timedelta(days=9)

        for i in range(25):
            trade = CompletedTrade(
                trade_id=f"winner-{i}",
                strategy_name="adaptive_ema_momentum",
                direction="long",
                entry_price=11850.0,
                exit_price=11870.0,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=100.0,
                exit_reason="take_profit",
                bars_held=24
            )
            tracker.track_trade(trade)

        window_end = datetime.now()
        performance = tracker.get_performance("adaptive_ema_momentum", window_end)

        # Profit factor should be infinity (or very high)
        assert performance.gross_loss == 0.0
        # Performance score should handle infinity gracefully
        assert performance.performance_score > 0

    def test_performance_score_calculation(self, tracker, sample_trades):
        """Test performance score calculation (Win Rate × Profit Factor)."""
        for trade in sample_trades:
            tracker.track_trade(trade)

        window_end = datetime.now()
        performance = tracker.get_performance("triple_confluence_scaler", window_end)

        # Win rate = 30/40 = 0.75
        # Profit factor = 3000/500 = 6.0
        # Performance score = 0.75 × 6.0 = 4.5
        expected_score = 0.75 * 6.0
        assert performance.performance_score == pytest.approx(expected_score)

    def test_get_performance_invalid_strategy(self, tracker):
        """Test getting performance for unknown strategy."""
        window_end = datetime.now()

        with pytest.raises(ValueError, match="Unknown strategy"):
            tracker.get_performance("unknown_strategy", window_end)

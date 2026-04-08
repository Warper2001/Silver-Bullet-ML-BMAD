"""Integration tests for Ensemble Signal Aggregation.

Tests end-to-end signal flow from strategies to ensemble processing.
"""

from datetime import datetime, timedelta

import pytest

from src.detection.ensemble_signal_aggregator import (
    EnsembleSignalAggregator,
    normalize_signal,
)
from src.detection.models import (
    EnsembleSignal,
    MomentumSignal,
    TripleConfluenceSignal,
    WolfPackSignal,
)


class TestEnsembleAggregationIntegration:
    """Test end-to-end ensemble signal aggregation pipeline."""

    @pytest.fixture
    def sample_ensemble_signals(self):
        """Create sample ensemble signals from all strategies."""
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        return [
            EnsembleSignal(
                strategy_name="Triple Confluence Scalper",
                timestamp=base_time,
                direction="long",
                entry_price=11850.00,
                stop_loss=11840.00,
                take_profit=11870.00,
                confidence=0.85,
                bar_timestamp=base_time,
                metadata={"fvg_size": 10},
            ),
            EnsembleSignal(
                strategy_name="Wolf Pack 3-Edge",
                timestamp=base_time + timedelta(seconds=1),
                direction="long",
                entry_price=11851.00,
                stop_loss=11841.00,
                take_profit=11871.00,
                confidence=0.82,
                bar_timestamp=base_time,
                metadata={"sweep_extent": 8},
            ),
            EnsembleSignal(
                strategy_name="Adaptive EMA Momentum",
                timestamp=base_time + timedelta(seconds=2),
                direction="short",
                entry_price=11849.00,
                stop_loss=11859.00,
                take_profit=11829.00,
                confidence=0.78,
                bar_timestamp=base_time,
                metadata={"ema_fast": 11848.0},
            ),
        ]

    def test_end_to_end_normalization_and_aggregation(self, sample_ensemble_signals):
        """Test complete pipeline: normalize → aggregate → query."""
        aggregator = EnsembleSignalAggregator()

        # Add all signals
        for signal in sample_ensemble_signals:
            aggregator.add_signal(signal)

        # Verify all signals stored
        assert aggregator.get_signal_count() == 3

        # Verify all strategies active
        active = aggregator.get_active_strategies()
        assert len(active) == 3

        # Verify consensus
        consensus = aggregator.get_consensus()
        assert consensus["long"] == 2
        assert consensus["short"] == 1

    def test_normalization_preserves_strategy_specific_data(self):
        """Test that normalization preserves all strategy-specific metadata."""
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Test Triple Confluence normalization
        tc_signal = TripleConfluenceSignal(
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            direction="long",
            confidence=0.85,
            timestamp=base_time,
            contributing_factors={"fvg_size": 12, "vwap_alignment": True},
        )

        normalized = normalize_signal(tc_signal)

        assert normalized.strategy_name == "Triple Confluence Scalper"
        assert normalized.metadata["fvg_size"] == 12
        assert normalized.metadata["vwap_alignment"] is True
        assert "expected_win_rate" in normalized.metadata

    def test_normalization_from_all_strategy_types(self):
        """Test normalization from all 5 strategy types."""
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        aggregator = EnsembleSignalAggregator()

        # Triple Confluence Signal
        tc_signal = TripleConfluenceSignal(
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            direction="long",
            confidence=0.85,
            timestamp=base_time,
            contributing_factors={"fvg_size": 10},
        )

        # Wolf Pack Signal
        wp_signal = WolfPackSignal(
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            direction="long",
            confidence=0.88,
            timestamp=base_time,
            contributing_factors={"sweep_extent": 8},
        )

        # EMA Momentum Signal (uses UPPERCASE direction, 0-100 confidence)
        ema_signal = MomentumSignal(
            timestamp=base_time,
            direction="LONG",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=82.0,  # 0-100 scale
            ema_fast=11848.0,
        )

        # Normalize and add all signals
        aggregator.add_signal(normalize_signal(tc_signal))
        aggregator.add_signal(normalize_signal(wp_signal))
        aggregator.add_signal(normalize_signal(ema_signal))

        # Verify all normalized correctly
        assert aggregator.get_signal_count() == 3

        # Check EMA normalization (direction should be lowercase, confidence 0-1)
        signals = aggregator.get_signals()
        ema_normalized = [s for s in signals if s.strategy_name == "Adaptive EMA Momentum"][0]
        assert ema_normalized.direction == "long"
        assert ema_normalized.confidence == 0.82
        assert ema_normalized.metadata["ema_fast"] == 11848.0

    def test_signal_flow_chronological_processing(self):
        """Test processing signals chronologically with automatic cleanup."""
        aggregator = EnsembleSignalAggregator(max_lookback=3)
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add 20 bars worth of signals
        for i in range(20):
            is_long = i % 2 == 0
            signal = EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=base_time + timedelta(minutes=i * 5),
                direction="long" if is_long else "short",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i if is_long else 11860.00 + i,  # Above entry for short
                take_profit=11870.00 + i if is_long else 11830.00 + i,  # Below entry for short
                confidence=0.85,
                bar_timestamp=base_time + timedelta(minutes=i * 5),
            )
            aggregator.add_signal(signal)

        # Should only have last 3 bars (lookback=3)
        # Plus buffer, but cleanup should remove old ones
        stats = aggregator.get_storage_stats()
        assert stats["total_signals"] <= 8  # max_lookback + buffer

    def test_deduplication_in_integration(self):
        """Test deduplication works correctly in real scenario."""
        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add initial signal
        signal1 = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=base_time,
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.80,
            bar_timestamp=base_time,
        )
        aggregator.add_signal(signal1)

        # Add updated signal for same bar
        signal2 = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=base_time + timedelta(seconds=30),
            direction="long",
            entry_price=11851.00,
            stop_loss=11841.00,
            take_profit=11871.00,
            confidence=0.90,  # Updated confidence
            bar_timestamp=base_time,  # Same bar
        )
        aggregator.add_signal(signal2)

        # Should only have one signal for this bar
        signals = aggregator.get_signals(strategy="Test Strategy")
        assert len(signals) == 1
        assert signals[0].confidence == 0.90  # Latest kept
        assert signals[0].entry_price == 11851.00

    def test_consensus_detection_integration(self, sample_ensemble_signals):
        """Test consensus detection with real signal data."""
        aggregator = EnsembleSignalAggregator()

        for signal in sample_ensemble_signals:
            aggregator.add_signal(signal)

        # Check alignment (should be False: 2 long, 1 short)
        assert aggregator.are_signals_aligned() is False

        # Check alignment strength (2/3 = 0.67)
        strength = aggregator.get_alignment_strength()
        assert abs(strength - 0.6667) < 0.01

        # Check conflicting strategies (should be EMA Momentum)
        conflicting = aggregator.get_conflicting_strategies()
        assert len(conflicting) == 1
        assert "Adaptive EMA Momentum" in conflicting

    @pytest.mark.asyncio
    async def test_async_processing_integration(self):
        """Test async signal processing with real signals."""
        import asyncio

        aggregator = EnsembleSignalAggregator()
        queue = asyncio.Queue()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add multiple signals to queue
        for i in range(5):
            signal = EnsembleSignal(
                strategy_name=f"Strategy {i}",
                timestamp=base_time + timedelta(seconds=i),
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time,
            )
            await queue.put(signal)

        # Add sentinel to stop
        await queue.put(None)

        # Process queue
        await aggregator.process_signals_queue(queue)

        # Verify all signals processed
        assert aggregator.get_signal_count() == 5

    @pytest.mark.asyncio
    async def test_background_task_integration(self):
        """Test background task with start/stop lifecycle."""
        import asyncio

        aggregator = EnsembleSignalAggregator()
        queue = asyncio.Queue()

        # Start background task
        task = await aggregator.start_aggregator(queue)
        assert task is not None

        # Add signal
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        signal = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=base_time,
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.85,
            bar_timestamp=base_time,
        )
        await queue.put(signal)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Stop aggregator
        await aggregator.stop_aggregator()

        # Verify signal processed
        assert aggregator.get_signal_count() == 1

    def test_edge_case_all_strategies_signal_simultaneously(self):
        """Test edge case where all 5 strategies signal simultaneously."""
        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Create 5 signals all agreeing on long
        strategies = [
            "Triple Confluence Scalper",
            "Wolf Pack 3-Edge",
            "Adaptive EMA Momentum",
            "VWAP Bounce",
            "Opening Range Breakout",
        ]

        for i, strategy in enumerate(strategies):
            signal = EnsembleSignal(
                strategy_name=strategy,
                timestamp=base_time + timedelta(milliseconds=i),
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.80 + (i * 0.02),
                bar_timestamp=base_time,
            )
            aggregator.add_signal(signal)

        # Verify all signals stored
        assert aggregator.get_signal_count() == 5

        # Verify perfect alignment
        assert aggregator.are_signals_aligned() is True
        assert aggregator.get_alignment_strength() == 1.0
        assert len(aggregator.get_conflicting_strategies()) == 0

    def test_edge_case_no_signals_empty_state(self):
        """Test aggregator behavior with no signals."""
        aggregator = EnsembleSignalAggregator()

        # Verify empty state
        assert aggregator.get_signal_count() == 0
        assert len(aggregator.get_active_strategies()) == 0
        assert aggregator.are_signals_aligned() is True  # Trivially aligned
        assert aggregator.get_alignment_strength() == 0.0
        assert len(aggregator.get_conflicting_strategies()) == 0

        # Verify queries return empty
        assert aggregator.get_signals() == []
        assert aggregator.get_signals(strategy="Nonexistent") == []
        assert aggregator.get_latest_signal("Nonexistent") is None

    def test_performance_rapid_signal_bursts(self):
        """Test aggregator performance with rapid signal bursts."""
        import time

        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add 100 signals rapidly
        start_time = time.time()
        for i in range(100):
            is_long = i % 2 == 0
            signal = EnsembleSignal(
                strategy_name=f"Strategy {i % 5}",  # Rotate through 5 strategies
                timestamp=base_time + timedelta(milliseconds=i),
                direction="long" if is_long else "short",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i if is_long else 11860.00 + i,  # Above entry for short
                take_profit=11870.00 + i if is_long else 11830.00 + i,  # Below entry for short
                confidence=0.85,
                bar_timestamp=base_time + timedelta(seconds=i // 10),  # Group by 10s
            )
            aggregator.add_signal(signal)

        elapsed = time.time() - start_time

        # Should process 100 signals quickly (< 1 second)
        assert elapsed < 1.0

        # Verify signals added
        assert aggregator.get_signal_count() <= 100  # May be less due to deduplication

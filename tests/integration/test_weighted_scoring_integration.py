"""Integration tests for Weighted Confidence Scorer.

Tests the complete weighted scoring pipeline with ensemble signals
from multiple strategies.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from src.detection.models import EnsembleSignal, EnsembleTradeSignal
from src.detection.weighted_confidence_scorer import (
    StrategyWeights,
    WeightManager,
    WeightedConfidenceScorer,
)


class TestWeightedScoringIntegration:
    """Integration tests for weighted confidence scoring."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def config_path(self, temp_config_dir):
        """Create config file for testing."""
        config = {
            "ensemble": {
                "strategies": {
                    "triple_confluence_scaler": 0.20,
                    "wolf_pack_3_edge": 0.20,
                    "adaptive_ema_momentum": 0.20,
                    "vwap_bounce": 0.20,
                    "opening_range_breakout": 0.20,
                },
                "confidence_threshold": 0.50,
                "minimum_strategies": 1,
            }
        }

        config_path = Path(temp_config_dir) / "config-sim.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def all_five_strategies_signals(self):
        """Create signals from all 5 strategies (unanimous)."""
        return [
            EnsembleSignal(
                strategy_name="triple_confluence_scaler",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                confidence=0.80,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
            EnsembleSignal(
                strategy_name="wolf_pack_3_edge",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11852.0,
                stop_loss=11832.0,
                take_profit=11892.0,
                confidence=0.75,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
            EnsembleSignal(
                strategy_name="adaptive_ema_momentum",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11851.0,
                stop_loss=11831.0,
                take_profit=11891.0,
                confidence=0.78,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
            EnsembleSignal(
                strategy_name="vwap_bounce",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.5,
                stop_loss=11830.5,
                take_profit=11890.5,
                confidence=0.72,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
            EnsembleSignal(
                strategy_name="opening_range_breakout",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11853.0,
                stop_loss=11833.0,
                take_profit=11893.0,
                confidence=0.76,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
        ]

    def test_full_pipeline_with_all_five_strategies(self, config_path, all_five_strategies_signals):
        """Test complete weighted scoring pipeline with all 5 strategies."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        ensemble_signal = scorer.score_signals(all_five_strategies_signals)

        assert ensemble_signal is not None
        assert ensemble_signal.direction == "long"
        assert ensemble_signal.composite_confidence > 0.50  # Should pass threshold
        assert len(ensemble_signal.contributing_strategies) == 5
        assert ensemble_signal.is_unanimous() is True
        assert ensemble_signal.strategy_name == "Ensemble-Weighted Confidence"

    def test_full_pipeline_with_two_strategies(self, config_path):
        """Test complete weighted scoring pipeline with 2 strategies."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        signals = [
            EnsembleSignal(
                strategy_name="triple_confluence_scaler",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                confidence=0.85,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
            EnsembleSignal(
                strategy_name="wolf_pack_3_edge",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11852.0,
                stop_loss=11832.0,
                take_profit=11892.0,
                confidence=0.82,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
        ]

        ensemble_signal = scorer.score_signals(signals)

        # Composite: 0.20 * 0.85 + 0.20 * 0.82 = 0.334, below 0.50 threshold
        assert ensemble_signal is None  # Should be rejected due to low confidence

    def test_configuration_change_affects_scoring(self, config_path, all_five_strategies_signals):
        """Test that changing configuration affects scoring behavior."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        # First, verify signal passes with default threshold
        scorer.set_threshold(0.30)
        ensemble_signal = scorer.score_signals(all_five_strategies_signals)
        assert ensemble_signal is not None

        # Now increase threshold above composite confidence
        scorer.set_threshold(0.90)
        ensemble_signal = scorer.score_signals(all_five_strategies_signals)
        assert ensemble_signal is None  # Should be rejected with higher threshold

    def test_weight_change_affects_composite_confidence(self, config_path, all_five_strategies_signals):
        """Test that changing weights affects composite confidence."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        # Get initial composite confidence
        scorer.set_threshold(0.10)  # Low threshold to allow signal
        initial_signal = scorer.score_signals(all_five_strategies_signals)
        initial_confidence = initial_signal.composite_confidence

        # Change weights to favor triple_confluence_scaler
        new_weights = StrategyWeights(
            triple_confluence_scaler=0.40,
            wolf_pack_3_edge=0.15,
            adaptive_ema_momentum=0.15,
            vwap_bounce=0.15,
            opening_range_breakout=0.15,
        )
        scorer.weight_manager.save_weights(new_weights)
        scorer.reload_config()

        # Get new composite confidence
        new_signal = scorer.score_signals(all_five_strategies_signals)
        new_confidence = new_signal.composite_confidence

        # Confidences should be different due to weight change
        # Since triple_confluence_scaler has high confidence (0.80) and higher weight (0.40),
        # the new composite should be higher
        assert new_confidence != initial_confidence

    def test_mixed_direction_signals_rejected(self, config_path):
        """Test that mixed long/short signals are rejected."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        signals = [
            EnsembleSignal(
                strategy_name="triple_confluence_scaler",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                confidence=0.90,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
            EnsembleSignal(
                strategy_name="wolf_pack_3_edge",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="short",
                entry_price=11845.0,
                stop_loss=11865.0,
                take_profit=11805.0,  # 2:1 ratio for short (40pt reward / 20pt risk)
                confidence=0.90,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
        ]

        ensemble_signal = scorer.score_signals(signals)
        assert ensemble_signal is None  # Should be rejected due to direction conflict

    def test_ensemble_signal_contains_all_required_fields(self, config_path, all_five_strategies_signals):
        """Test that ensemble signal contains all required fields."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))
        scorer.set_threshold(0.30)

        ensemble_signal = scorer.score_signals(all_five_strategies_signals)

        # Check all required fields exist
        assert ensemble_signal.strategy_name == "Ensemble-Weighted Confidence"
        assert ensemble_signal.timestamp is not None
        assert ensemble_signal.direction in ["long", "short"]
        assert ensemble_signal.entry_price > 0
        assert ensemble_signal.stop_loss > 0
        assert ensemble_signal.take_profit > 0
        assert 0 <= ensemble_signal.composite_confidence <= 1
        assert len(ensemble_signal.contributing_strategies) > 0
        assert len(ensemble_signal.strategy_confidences) > 0
        assert len(ensemble_signal.strategy_weights) > 0
        assert ensemble_signal.bar_timestamp is not None

    def test_performance_target(self, config_path, all_five_strategies_signals):
        """Test that scoring meets performance target (<10ms per bar)."""
        import time

        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        start_time = time.perf_counter()
        ensemble_signal = scorer.score_signals(all_five_strategies_signals)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        assert ensemble_signal is not None
        assert elapsed_ms < 10.0, f"Scoring took {elapsed_ms:.2f}ms, target is <10ms"

    def test_empty_signals_returns_none(self, config_path):
        """Test that empty signals list returns None."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        ensemble_signal = scorer.score_signals([])
        assert ensemble_signal is None

    def test_single_strategy_signal(self, config_path):
        """Test scoring with only one strategy signal."""
        scorer = WeightedConfidenceScorer(config_path=str(config_path))

        signals = [
            EnsembleSignal(
                strategy_name="triple_confluence_scaler",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                confidence=0.90,
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
                metadata={},
            ),
        ]

        ensemble_signal = scorer.score_signals(signals)

        # Composite: 0.20 * 0.90 = 0.18, below 0.50 threshold
        assert ensemble_signal is None  # Should be rejected due to low confidence

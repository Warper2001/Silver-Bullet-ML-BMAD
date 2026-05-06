"""Unit tests for Weighted Confidence Scorer.

Tests for weight management, configuration loading, validation, and
strategy weight normalization.
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
import yaml
from pydantic import ValidationError

from src.detection.weighted_confidence_scorer import (
    StrategyWeights,
    WeightManager,
)
from src.detection.models import EnsembleTradeSignal, EnsembleSignal


class TestStrategyWeights:
    """Test StrategyWeights Pydantic model."""

    def test_valid_equal_weights(self):
        """Test creation with valid equal weights (0.20 each)."""
        weights = StrategyWeights(
            strategies={
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20,
            }
        )
        assert weights.strategies["triple_confluence_scaler"] == 0.20
        assert weights.strategies["wolf_pack_3_edge"] == 0.20
        assert weights.strategies["adaptive_ema_momentum"] == 0.20
        assert weights.strategies["vwap_bounce"] == 0.20
        assert weights.strategies["opening_range_breakout"] == 0.20

    def test_weights_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValidationError, match="sum to 1.0"):
            StrategyWeights(
                strategies={
                    "triple_confluence_scaler": 0.30,
                    "wolf_pack_3_edge": 0.30,
                    "adaptive_ema_momentum": 0.30,
                    "vwap_bounce": 0.30,
                    "opening_range_breakout": 0.30,  # Sum = 1.5, should fail
                }
            )

    def test_weights_must_be_positive(self):
        """Test that weights must be between 0 and 1."""
        with pytest.raises(ValidationError):
            StrategyWeights(
                strategies={
                    "triple_confluence_scaler": -0.10,
                    "wolf_pack_3_edge": 0.30,
                    "adaptive_ema_momentum": 0.30,
                    "vwap_bounce": 0.30,
                    "opening_range_breakout": 0.20,
                }
            )

    def test_weights_must_not_exceed_one(self):
        """Test that individual weights cannot exceed 1.0."""
        with pytest.raises(ValidationError):
            StrategyWeights(
                strategies={
                    "triple_confluence_scaler": 1.5,
                    "wolf_pack_3_edge": 0.0,
                    "adaptive_ema_momentum": 0.0,
                    "vwap_bounce": 0.0,
                    "opening_range_breakout": 0.0,
                }
            )

    def test_valid_unequal_weights(self):
        """Test creation with valid unequal weights that sum to 1.0."""
        weights = StrategyWeights(
            strategies={
                "triple_confluence_scaler": 0.25,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.15,
                "vwap_bounce": 0.30,
                "opening_range_breakout": 0.10,
            }
        )
        assert weights.strategies["triple_confluence_scaler"] == 0.25
        assert weights.strategies["vwap_bounce"] == 0.30

    def test_weights_sum_with_floating_point_tolerance(self):
        """Test that weights sum validation has floating point tolerance."""
        # Small floating point errors should be acceptable
        weights = StrategyWeights(
            strategies={
                "triple_confluence_scaler": 0.2000000001,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.1999999999,
            }
        )
        assert weights is not None


class TestWeightManager:
    """Test WeightManager class for configuration-driven weight management."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def valid_config_path(self, temp_config_dir):
        """Create a valid config file for testing."""
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
    def invalid_config_path(self, temp_config_dir):
        """Create an invalid config file (weights don't sum to 1.0)."""
        config = {
            "ensemble": {
                "strategies": {
                    "triple_confluence_scaler": 0.50,
                    "wolf_pack_3_edge=0.30": 0.30,
                    "adaptive_ema_momentum": 0.30,
                    "vwap_bounce": 0.30,
                    "opening_range_breakout": 0.30,
                },
                "confidence_threshold": 0.50,
                "minimum_strategies": 1,
            }
        }

        config_path = Path(temp_config_dir) / "config-invalid.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_load_weights_from_config(self, valid_config_path):
        """Test loading weights from valid config file."""
        manager = WeightManager(config_path=str(valid_config_path))
        weights = manager.load_weights()

        assert weights.strategies["triple_confluence_scaler"] == 0.20
        assert weights.strategies["wolf_pack_3_edge"] == 0.20
        assert weights.strategies["adaptive_ema_momentum"] == 0.20
        assert weights.strategies["vwap_bounce"] == 0.20
        assert weights.strategies["opening_range_breakout"] == 0.20

    def test_load_weights_raises_error_on_invalid_config(self, invalid_config_path):
        """Test that loading invalid config raises ValueError."""
        manager = WeightManager(config_path=str(invalid_config_path))

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            manager.load_weights()

    def test_save_weights_persists_to_config(self, valid_config_path, temp_config_dir):
        """Test saving weights to config file."""
        manager = WeightManager(config_path=str(valid_config_path))

        new_weights = StrategyWeights(strategies={"triple_confluence_scaler": 0.25, "wolf_pack_3_edge": 0.20, "adaptive_ema_momentum": 0.15, "vwap_bounce": 0.30, "opening_range_breakout": 0.10})

        manager.save_weights(new_weights)

        # Reload and verify
        manager2 = WeightManager(config_path=str(valid_config_path))
        reloaded = manager2.load_weights()

        assert reloaded.strategies["triple_confluence_scaler"] == 0.25
        assert reloaded.strategies["wolf_pack_3_edge"] == 0.20
        assert reloaded.strategies["adaptive_ema_momentum"] == 0.15

    def test_update_single_weight(self, valid_config_path):
        """Test updating a single strategy weight.

        Note: update_weight changes one weight in the config file, but
        the weights may no longer sum to 1.0. This is expected - the
        caller should normalize_weights() after updating if needed.
        """
        manager = WeightManager(config_path=str(valid_config_path))

        # Update weight - this changes the config directly
        manager.update_weight("triple_confluence_scaler", 0.25)

        # Load the raw config to verify the update
        config = manager.get_config()
        assert config["ensemble"]["strategies"]["triple_confluence_scaler"] == 0.25

    def test_normalize_weights_adjusts_to_sum_to_one(self, valid_config_path):
        """Test that normalize_weights adjusts weights to sum to 1.0."""
        manager = WeightManager(config_path=str(valid_config_path))

        weights = StrategyWeights.model_construct(
            strategies={
                "triple_confluence_scaler": 0.30,
                "wolf_pack_3_edge": 0.30,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.15,
                "opening_range_breakout": 0.15,  # Sum = 1.1
            }
        )

        normalized = manager.normalize_weights(weights)

        # Should sum to 1.0
        total = sum(normalized.strategies.values())

        assert abs(total - 1.0) < 1e-6

        # Proportions should be preserved
        # 0.30/0.30 = 1.0, so normalized values should be equal
        assert (
            normalized.strategies["triple_confluence_scaler"] / normalized.strategies["wolf_pack_3_edge"]
            == 0.30 / 0.30
        )

    def test_get_config_returns_dict(self, valid_config_path):
        """Test that get_config returns configuration as dict."""
        manager = WeightManager(config_path=str(valid_config_path))
        config = manager.get_config()

        assert isinstance(config, dict)
        assert "ensemble" in config
        assert "strategies" in config["ensemble"]
        assert config["ensemble"]["confidence_threshold"] == 0.50

    def test_missing_config_file_raises_error(self, temp_config_dir):
        """Test that missing config file raises FileNotFoundError."""
        missing_path = Path(temp_config_dir) / "nonexistent.yaml"
        manager = WeightManager(config_path=str(missing_path))

        with pytest.raises(FileNotFoundError):
            manager.load_weights()


class TestEnsembleTradeSignal:
    """Test EnsembleTradeSignal Pydantic model."""

    def test_valid_ensemble_signal(self):
        """Test creation of valid ensemble trade signal."""
        signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="long",
            entry_price=11850.0,
            stop_loss=11830.0,
            take_profit=11890.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
            strategy_confidences={"triple_confluence_scaler": 0.80, "wolf_pack_3_edge": 0.70},
            strategy_weights={"triple_confluence_scaler": 0.20, "wolf_pack_3_edge": 0.20},
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
        )

        assert signal.strategy_name == "Ensemble-Weighted Confidence"
        assert signal.direction == "long"
        assert signal.composite_confidence == 0.75
        assert len(signal.contributing_strategies) == 2

    def test_composite_confidence_must_be_between_0_and_1(self):
        """Test that composite_confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            EnsembleTradeSignal(
                strategy_name="Ensemble-Weighted Confidence",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                composite_confidence=1.5,  # Invalid: > 1.0
                contributing_strategies=["triple_confluence_scaler"],
                strategy_confidences={"triple_confluence_scaler": 0.80},
                strategy_weights={"triple_confluence_scaler": 0.20},
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
            )

    def test_at_least_one_contributing_strategy_required(self):
        """Test that at least one contributing strategy is required."""
        with pytest.raises(ValidationError):
            EnsembleTradeSignal(
                strategy_name="Ensemble-Weighted Confidence",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                composite_confidence=0.75,
                contributing_strategies=[],  # Invalid: empty list
                strategy_confidences={},
                strategy_weights={},
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
            )

    def test_all_contributing_strategies_must_have_confidence(self):
        """Test that all contributing strategies have confidence values."""
        with pytest.raises(ValidationError):
            EnsembleTradeSignal(
                strategy_name="Ensemble-Weighted Confidence",
                timestamp=datetime(2026, 3, 31, 14, 30, 0),
                direction="long",
                entry_price=11850.0,
                stop_loss=11830.0,
                take_profit=11890.0,
                composite_confidence=0.75,
                contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
                strategy_confidences={"triple_confluence_scaler": 0.80},  # Missing wolf_pack_3_edge
                strategy_weights={"triple_confluence_scaler": 0.20, "wolf_pack_3_edge": 0.20},
                bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
            )

    def test_contributing_count_method(self):
        """Test contributing_count helper method."""
        signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="long",
            entry_price=11850.0,
            stop_loss=11830.0,
            take_profit=11890.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge", "adaptive_ema_momentum"],
            strategy_confidences={
                "triple_confluence_scaler": 0.80,
                "wolf_pack_3_edge": 0.70,
                "adaptive_ema_momentum": 0.75,
            },
            strategy_weights={
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
            },
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
        )

        assert signal.contributing_count() == 3

    def test_is_unanimous_method(self):
        """Test is_unanimous helper method."""
        # Test with all 5 strategies
        signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="long",
            entry_price=11850.0,
            stop_loss=11830.0,
            take_profit=11890.0,
            composite_confidence=0.90,
            contributing_strategies=[
                "triple_confluence_scaler",
                "wolf_pack_3_edge",
                "adaptive_ema_momentum",
                "vwap_bounce",
                "opening_range_breakout",
            ],
            strategy_confidences={
                "triple_confluence_scaler": 0.80,
                "wolf_pack_3_edge": 0.75,
                "adaptive_ema_momentum": 0.70,
                "vwap_bounce": 0.72,
                "opening_range_breakout": 0.78,
            },
            strategy_weights={
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20,
            },
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
        )

        assert signal.is_unanimous() is True

    def test_is_not_unanimous_with_fewer_strategies(self):
        """Test is_unanimous returns False with fewer than 5 strategies."""
        signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="long",
            entry_price=11850.0,
            stop_loss=11830.0,
            take_profit=11890.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
            strategy_confidences={"triple_confluence_scaler": 0.80, "wolf_pack_3_edge": 0.70},
            strategy_weights={"triple_confluence_scaler": 0.20, "wolf_pack_3_edge": 0.20},
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
        )

        assert signal.is_unanimous() is False

    def test_get_weighted_entry_method(self):
        """Test get_weighted_entry helper method."""
        signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="long",
            entry_price=11850.0,
            stop_loss=11830.0,
            take_profit=11890.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
            strategy_confidences={"triple_confluence_scaler": 0.80, "wolf_pack_3_edge": 0.70},
            strategy_weights={"triple_confluence_scaler": 0.60, "wolf_pack_3_edge": 0.40},
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
        )

        # This should recalculate based on a hypothetical underlying entry prices
        # For now, it returns the entry_price since we don't store individual strategy entries
        weighted = signal.get_weighted_entry()
        assert weighted == 11850.0


class TestWeightedConfidenceScorer:
    """Test WeightedConfidenceScorer class for weighted scoring logic."""

    @pytest.fixture
    def valid_config_path(self, tmp_path):
        """Create a valid config file for testing."""
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

        config_path = tmp_path / "config-sim.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def sample_signals(self):
        """Create sample ensemble signals for testing."""
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
        ]

    def test_calculate_composite_confidence(self, valid_config_path, sample_signals):
        """Test composite confidence calculation from multiple signals."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))
        weights = scorer.weight_manager.load_weights()

        composite = scorer.calculate_composite_confidence(sample_signals, weights)

        # Expected: 0.20 * 0.80 + 0.20 * 0.75 = 0.31
        expected = 0.20 * 0.80 + 0.20 * 0.75
        assert abs(composite - expected) < 1e-6

    def test_calculate_composite_confidence_no_signals(self, valid_config_path):
        """Test composite confidence with no signals returns 0."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))
        weights = scorer.weight_manager.load_weights()

        composite = scorer.calculate_composite_confidence([], weights)
        assert composite == 0.0

    def test_check_direction_alignment_all_long(self, valid_config_path, sample_signals):
        """Test direction alignment check with all long signals."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        is_aligned = scorer.check_direction_alignment(sample_signals)
        assert is_aligned is True

    def test_check_direction_alignment_mixed(self, valid_config_path, sample_signals):
        """Test direction alignment check with mixed long/short signals."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        # Add a short signal to create conflict
        short_signal = EnsembleSignal(
            strategy_name="adaptive_ema_momentum",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="short",
            entry_price=11855.0,
            stop_loss=11875.0,
            take_profit=11815.0,
            confidence=0.70,
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
            metadata={},
        )

        mixed_signals = sample_signals + [short_signal]
        is_aligned = scorer.check_direction_alignment(mixed_signals)

        assert is_aligned is False

    def test_calculate_weighted_entry(self, valid_config_path, sample_signals):
        """Test weighted entry price calculation."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))
        weights = scorer.weight_manager.load_weights()

        weighted_entry = scorer.calculate_weighted_entry(sample_signals, weights)

        # Expected: 0.20 * 11850 + 0.20 * 11852 = 2370.2 / 0.40 = 11851.0
        expected = (0.20 * 11850.0 + 0.20 * 11852.0) / 0.40
        assert abs(weighted_entry - expected) < 1e-6

    def test_score_signals_generates_ensemble_signal(self, valid_config_path, sample_signals):
        """Test that score_signals generates valid EnsembleTradeSignal."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        # Lower threshold to allow signal generation (0.31 > 0.30)
        scorer.set_threshold(0.30)
        ensemble_signal = scorer.score_signals(sample_signals)

        assert ensemble_signal is not None
        assert ensemble_signal.direction == "long"
        assert ensemble_signal.composite_confidence > 0
        assert len(ensemble_signal.contributing_strategies) == 2
        assert ensemble_signal.strategy_name == "Ensemble-Weighted Confidence"

    def test_score_signals_returns_none_for_no_signals(self, valid_config_path):
        """Test that score_signals returns None when no signals provided."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        ensemble_signal = scorer.score_signals([])
        assert ensemble_signal is None

    def test_apply_threshold_filter_pass(self, valid_config_path):
        """Test threshold filter passes when confidence above threshold."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        result = scorer.apply_threshold_filter(0.75, 0.50)
        assert result is True

    def test_apply_threshold_filter_fail(self, valid_config_path):
        """Test threshold filter fails when confidence below threshold."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        result = scorer.apply_threshold_filter(0.40, 0.50)
        assert result is False

    def test_check_minimum_strategies_pass(self, valid_config_path, sample_signals):
        """Test minimum strategies check passes."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        result = scorer.check_minimum_strategies(sample_signals, 1)
        assert result is True

    def test_check_minimum_strategies_fail(self, valid_config_path, sample_signals):
        """Test minimum strategies check fails."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        result = scorer.check_minimum_strategies(sample_signals, 3)
        assert result is False

    def test_filter_signals_with_conflict(self, valid_config_path, sample_signals):
        """Test that filter_signals returns None for conflicting directions."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        # Add a short signal to create conflict
        short_signal = EnsembleSignal(
            strategy_name="adaptive_ema_momentum",
            timestamp=datetime(2026, 3, 31, 14, 30, 0),
            direction="short",
            entry_price=11855.0,
            stop_loss=11875.0,
            take_profit=11815.0,
            confidence=0.70,
            bar_timestamp=datetime(2026, 3, 31, 14, 30, 0),
            metadata={},
        )

        mixed_signals = sample_signals + [short_signal]
        weights = scorer.weight_manager.load_weights()

        result = scorer.filter_signals(mixed_signals, weights, 0.50)
        assert result is None  # Should be rejected due to direction conflict

    def test_filter_signals_below_threshold(self, valid_config_path, sample_signals):
        """Test that filter_signals returns None when below threshold."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        # Use very high threshold that won't be met
        weights = scorer.weight_manager.load_weights()
        result = scorer.filter_signals(sample_signals, weights, 0.95)

        assert result is None  # Should be rejected due to low confidence

    def test_filter_signals_success(self, valid_config_path, sample_signals):
        """Test that filter_signals generates valid signal."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        weights = scorer.weight_manager.load_weights()
        result = scorer.filter_signals(sample_signals, weights, 0.30)

        assert result is not None
        assert isinstance(result, EnsembleTradeSignal)
        assert result.direction == "long"
        assert result.composite_confidence > 0.30

    def test_reload_config(self, valid_config_path):
        """Test configuration reload."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        # Get initial threshold
        initial_threshold = scorer.get_threshold()
        assert initial_threshold == 0.50

        # Modify config file
        with open(valid_config_path) as f:
            config = yaml.safe_load(f)
        config["ensemble"]["confidence_threshold"] = 0.60
        with open(valid_config_path, "w") as f:
            yaml.dump(config, f)

        # Reload config
        scorer.reload_config()

        # Check new threshold
        new_threshold = scorer.get_threshold()
        assert new_threshold == 0.60

    def test_set_threshold(self, valid_config_path):
        """Test setting threshold directly."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        scorer.set_threshold(0.70)
        assert scorer.get_threshold() == 0.70

    def test_set_threshold_invalid(self, valid_config_path):
        """Test that setting invalid threshold raises ValueError."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            scorer.set_threshold(1.5)

    def test_get_config(self, valid_config_path):
        """Test getting current configuration."""
        from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

        scorer = WeightedConfidenceScorer(config_path=str(valid_config_path))

        config = scorer.get_config()
        assert isinstance(config, dict)
        assert "ensemble" in config
        assert config["ensemble"]["confidence_threshold"] == 0.50

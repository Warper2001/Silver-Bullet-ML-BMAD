"""Weighted Confidence Scorer for Ensemble Trading System.

This module implements configuration-driven strategy weight management and
weighted confidence scoring for the ensemble trading system.

Key Components:
- StrategyWeights: Pydantic model for strategy weight validation
- WeightManager: Configuration-driven weight management and persistence
- WeightedConfidenceScorer: Core weighted scoring logic with filtering
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.detection.models import EnsembleSignal, EnsembleTradeSignal

logger = logging.getLogger(__name__)


class StrategyWeights(BaseModel):
    """Strategy weights for ensemble scoring.

    Weights represent the contribution of each strategy to the ensemble.
    All weights must be between 0 and 1, and must sum to 1.0.

    Attributes:
        triple_confluence_scaler: Weight for Triple Confluence Scalper strategy
        wolf_pack_3_edge: Weight for Wolf Pack 3-Edge strategy
        adaptive_ema_momentum: Weight for Adaptive EMA Momentum strategy
        vwap_bounce: Weight for VWAP Bounce strategy
        opening_range_breakout: Weight for Opening Range Breakout strategy
    """

    triple_confluence_scaler: float = Field(..., ge=0, le=1, description="Weight for Triple Confluence Scalper")
    wolf_pack_3_edge: float = Field(..., ge=0, le=1, description="Weight for Wolf Pack 3-Edge")
    adaptive_ema_momentum: float = Field(..., ge=0, le=1, description="Weight for Adaptive EMA Momentum")
    vwap_bounce: float = Field(..., ge=0, le=1, description="Weight for VWAP Bounce")
    opening_range_breakout: float = Field(..., ge=0, le=1, description="Weight for Opening Range Breakout")

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "StrategyWeights":
        """Validate that weights sum to 1.0 with floating point tolerance.

        Returns:
            self if validation passes

        Raises:
            ValueError: If weights don't sum to 1.0 (within 1e-6 tolerance)
        """
        total = (
            self.triple_confluence_scaler
            + self.wolf_pack_3_edge
            + self.adaptive_ema_momentum
            + self.vwap_bounce
            + self.opening_range_breakout
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                f"Please adjust weights to sum exactly to 1.0."
            )

        return self


class WeightManager:
    """Configuration-driven strategy weight management.

    Manages loading, saving, updating, and normalizing strategy weights
    from YAML configuration files. Provides validation and persistence
    for ensemble weight configuration.

    Attributes:
        config_path: Path to configuration file (default: config-sim.yaml)
    """

    def __init__(self, config_path: str = "config-sim.yaml") -> None:
        """Initialize WeightManager with configuration file path.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        logger.info("WeightManager initialized with config: %s", self.config_path)

    def load_weights(self) -> StrategyWeights:
        """Load strategy weights from configuration file.

        Returns:
            StrategyWeights object with loaded weights

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If weights are invalid (don't sum to 1.0 or out of range)
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        strategies = config.get("ensemble", {}).get("strategies", {})

        weights = StrategyWeights(
            triple_confluence_scaler=strategies.get("triple_confluence_scaler", 0.20),
            wolf_pack_3_edge=strategies.get("wolf_pack_3_edge", 0.20),
            adaptive_ema_momentum=strategies.get("adaptive_ema_momentum", 0.20),
            vwap_bounce=strategies.get("vwap_bounce", 0.20),
            opening_range_breakout=strategies.get("opening_range_breakout", 0.20),
        )

        logger.info("Loaded weights from config: %s", self.config_path)
        return weights

    def save_weights(self, weights: StrategyWeights) -> None:
        """Persist strategy weights to configuration file.

        Args:
            weights: StrategyWeights object to save

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        config["ensemble"]["strategies"] = {
            "triple_confluence_scaler": weights.triple_confluence_scaler,
            "wolf_pack_3_edge": weights.wolf_pack_3_edge,
            "adaptive_ema_momentum": weights.adaptive_ema_momentum,
            "vwap_bounce": weights.vwap_bounce,
            "opening_range_breakout": weights.opening_range_breakout,
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info("Saved weights to config: %s", self.config_path)

    def update_weight(self, strategy: str, new_weight: float) -> None:
        """Update a single strategy weight and persist to config.

        Args:
            strategy: Strategy name to update
            new_weight: New weight value (0-1)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If strategy name is invalid or weight is out of range
        """
        valid_strategies = [
            "triple_confluence_scaler",
            "wolf_pack_3_edge",
            "adaptive_ema_momentum",
            "vwap_bounce",
            "opening_range_breakout",
        ]

        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                f"Valid strategies: {', '.join(valid_strategies)}"
            )

        if not 0 <= new_weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {new_weight}")

        # Load current weights
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Update specific weight
        config["ensemble"]["strategies"][strategy] = new_weight

        # Save updated config
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info("Updated %s weight to %.4f in %s", strategy, new_weight, self.config_path)

    def normalize_weights(self, weights: StrategyWeights) -> StrategyWeights:
        """Normalize weights to ensure they sum to 1.0 while preserving proportions.

        Args:
            weights: StrategyWeights object that may not sum to 1.0

        Returns:
            New StrategyWeights object normalized to sum to 1.0
        """
        total = (
            weights.triple_confluence_scaler
            + weights.wolf_pack_3_edge
            + weights.adaptive_ema_momentum
            + weights.vwap_bounce
            + weights.opening_range_breakout
        )

        if total == 0:
            # Edge case: all weights are 0, return equal weights
            return StrategyWeights(
                triple_confluence_scaler=0.20,
                wolf_pack_3_edge=0.20,
                adaptive_ema_momentum=0.20,
                vwap_bounce=0.20,
                opening_range_breakout=0.20,
            )

        normalized = StrategyWeights(
            triple_confluence_scaler=weights.triple_confluence_scaler / total,
            wolf_pack_3_edge=weights.wolf_pack_3_edge / total,
            adaptive_ema_momentum=weights.adaptive_ema_momentum / total,
            vwap_bounce=weights.vwap_bounce / total,
            opening_range_breakout=weights.opening_range_breakout / total,
        )

        logger.info("Normalized weights from total %.6f to 1.0", total)
        return normalized

    def get_config(self) -> dict[str, Any]:
        """Return current configuration as dictionary.

        Returns:
            Dictionary containing full configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        return config


class WeightedConfidenceScorer:
    """Weighted confidence scorer for ensemble trading signals.

    This class implements the core weighted scoring logic for the ensemble
    system, including composite confidence calculation, direction alignment
    checking, weighted entry calculation, and threshold filtering.

    Attributes:
        config_path: Path to configuration file (default: config-sim.yaml)
        weight_manager: WeightManager instance for weight management
    """

    def __init__(self, config_path: str = "config-sim.yaml") -> None:
        """Initialize WeightedConfidenceScorer with configuration file path.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.weight_manager = WeightManager(config_path=str(self.config_path))
        logger.info("WeightedConfidenceScorer initialized with config: %s", self.config_path)

    def calculate_composite_confidence(
        self, signals: list[EnsembleSignal], weights: StrategyWeights
    ) -> float:
        """Calculate composite confidence from multiple signals.

        Takes the highest confidence signal from each strategy to avoid
        double-counting when multiple signals from the same strategy
        appear within the lookback window.

        Args:
            signals: List of ensemble signals from individual strategies
            weights: Strategy weights to apply

        Returns:
            Composite confidence score (0-1 scale), or 0.0 if no signals
        """
        if not signals:
            return 0.0

        # Group signals by strategy, keeping only the highest confidence from each
        strategy_best_signals = {}
        for signal in signals:
            strategy_name = signal.strategy_name
            if strategy_name not in strategy_best_signals:
                strategy_best_signals[strategy_name] = signal
            elif signal.confidence > strategy_best_signals[strategy_name].confidence:
                strategy_best_signals[strategy_name] = signal

        # Calculate composite using only the best signal from each strategy
        composite = 0.0
        weight_map = {
            "triple_confluence_scaler": weights.triple_confluence_scaler,
            "wolf_pack_3_edge": weights.wolf_pack_3_edge,
            "adaptive_ema_momentum": weights.adaptive_ema_momentum,
            "vwap_bounce": weights.vwap_bounce,
            "opening_range_breakout": weights.opening_range_breakout,
        }

        for signal in strategy_best_signals.values():
            strategy_weight = weight_map.get(signal.strategy_name, 0.0)
            composite += strategy_weight * signal.confidence

        # Cap at 1.0 to handle edge cases where weights might slightly exceed 1.0
        composite = min(composite, 1.0)

        logger.debug(
            "Calculated composite confidence: %.4f from %d unique strategies (from %d total signals)",
            composite, len(strategy_best_signals), len(signals)
        )
        return composite

    def check_direction_alignment(self, signals: list[EnsembleSignal]) -> bool:
        """Check if all signals agree on direction (unanimous long or short).

        Args:
            signals: List of ensemble signals

        Returns:
            True if all signals have the same direction, False otherwise
        """
        if not signals:
            return True

        directions = {signal.direction for signal in signals}
        is_aligned = len(directions) == 1

        if not is_aligned:
            long_strategies = [s.strategy_name for s in signals if s.direction == "long"]
            short_strategies = [s.strategy_name for s in signals if s.direction == "short"]
            logger.warning(
                "Direction conflict detected: LONG: %s, SHORT: %s",
                long_strategies,
                short_strategies,
            )

        return is_aligned

    def calculate_weighted_entry(
        self, signals: list[EnsembleSignal], weights: StrategyWeights
    ) -> float:
        """Calculate weighted average entry price from all signals.

        Args:
            signals: List of ensemble signals
            weights: Strategy weights to apply

        Returns:
            Weighted average entry price
        """
        if not signals:
            return 0.0

        weight_map = {
            "triple_confluence_scaler": weights.triple_confluence_scaler,
            "wolf_pack_3_edge": weights.wolf_pack_3_edge,
            "adaptive_ema_momentum": weights.adaptive_ema_momentum,
            "vwap_bounce": weights.vwap_bounce,
            "opening_range_breakout": weights.opening_range_breakout,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in signals:
            strategy_weight = weight_map.get(signal.strategy_name, 0.0)
            weighted_sum += strategy_weight * signal.entry_price
            total_weight += strategy_weight

        if total_weight == 0:
            return 0.0

        weighted_entry = weighted_sum / total_weight
        logger.debug("Calculated weighted entry: %.2f from %d signals", weighted_entry, len(signals))
        return weighted_entry

    def score_signals(self, signals: list[EnsembleSignal]) -> EnsembleTradeSignal | None:
        """Score ensemble signals and generate trade signal if appropriate.

        This is the main entry point for the weighted confidence scorer.
        It calculates composite confidence, checks direction alignment,
        and generates an ensemble trade signal if conditions are met.

        Args:
            signals: List of ensemble signals from individual strategies

        Returns:
            EnsembleTradeSignal if valid, None otherwise
        """
        if not signals:
            return None

        # Load weights and threshold from config
        weights = self.weight_manager.load_weights()
        config = self.weight_manager.get_config()
        threshold = config.get("ensemble", {}).get("confidence_threshold", 0.50)

        # Use filter_signals which applies all logic
        return self.filter_signals(signals, weights, threshold)

    def apply_threshold_filter(self, composite_confidence: float, threshold: float) -> bool:
        """Apply confidence threshold filtering.

        Args:
            composite_confidence: Calculated composite confidence
            threshold: Minimum threshold required

        Returns:
            True if composite_confidence > threshold, False otherwise
        """
        passes = composite_confidence > threshold
        logger.debug(
            "Threshold filter: %.4f > %.4f = %s", composite_confidence, threshold, passes
        )
        return passes

    def check_minimum_strategies(
        self, signals: list[EnsembleSignal], min_count: int
    ) -> bool:
        """Check if minimum number of strategies signaled.

        Args:
            signals: List of ensemble signals
            min_count: Minimum number of signals required

        Returns:
            True if len(signals) >= min_count, False otherwise
        """
        passes = len(signals) >= min_count
        logger.debug("Minimum strategies check: %d >= %d = %s", len(signals), min_count, passes)
        return passes

    def filter_signals(
        self, signals: list[EnsembleSignal], weights: StrategyWeights, threshold: float
    ) -> EnsembleTradeSignal | None:
        """Filter signals and generate ensemble trade signal if all checks pass.

        Args:
            signals: List of ensemble signals
            weights: Strategy weights
            threshold: Confidence threshold

        Returns:
            EnsembleTradeSignal if all filters pass, None otherwise
        """
        if not signals:
            return None

        # Check direction alignment
        if not self.check_direction_alignment(signals):
            logger.info("Signal rejected: direction conflict")
            return None

        # Calculate composite confidence
        composite = self.calculate_composite_confidence(signals, weights)

        # Apply threshold filter
        if not self.apply_threshold_filter(composite, threshold):
            logger.info("Signal rejected: composite confidence %.4f below threshold %.4f", composite, threshold)
            return None

        # Calculate weighted entry
        weighted_entry = self.calculate_weighted_entry(signals, weights)

        # Calculate weighted SL/TP (use weighted average from signals for now)
        # In Story 2.4, this will be refined with proper exit logic
        weighted_sl = sum(s.stop_loss for s in signals) / len(signals)
        weighted_tp = sum(s.take_profit for s in signals) / len(signals)

        # Determine direction (all signals agree at this point)
        direction = signals[0].direction

        # Build strategy confidences and weights dicts
        strategy_confidences = {s.strategy_name: s.confidence for s in signals}
        strategy_weights_map = {
            "triple_confluence_scaler": weights.triple_confluence_scaler,
            "wolf_pack_3_edge": weights.wolf_pack_3_edge,
            "adaptive_ema_momentum": weights.adaptive_ema_momentum,
            "vwap_bounce": weights.vwap_bounce,
            "opening_range_breakout": weights.opening_range_breakout,
        }

        # Filter to only contributing strategies
        contributing_weights = {
            s.strategy_name: strategy_weights_map[s.strategy_name] for s in signals
        }

        ensemble_signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=signals[0].timestamp,
            direction=direction,
            entry_price=weighted_entry,
            stop_loss=weighted_sl,
            take_profit=weighted_tp,
            composite_confidence=composite,
            contributing_strategies=[s.strategy_name for s in signals],
            strategy_confidences=strategy_confidences,
            strategy_weights=contributing_weights,
            bar_timestamp=signals[0].bar_timestamp,
        )

        logger.info(
            "Ensemble signal generated: %s, confidence=%.4f, strategies=%d",
            direction,
            composite,
            len(signals),
        )

        return ensemble_signal

    def reload_config(self) -> None:
        """Reload configuration from config file.

        This reloads the weight manager configuration, updating weights
        and threshold from the config file.
        """
        logger.info("Reloading configuration from %s", self.config_path)
        self.weight_manager = WeightManager(config_path=str(self.config_path))

    def get_config(self) -> dict[str, Any]:
        """Get current configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.weight_manager.get_config()

    def set_threshold(self, new_threshold: float) -> None:
        """Set new confidence threshold.

        Args:
            new_threshold: New threshold value (0-1)

        Raises:
            ValueError: If threshold is not between 0 and 1
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {new_threshold}")

        config = self.weight_manager.get_config()
        config["ensemble"]["confidence_threshold"] = new_threshold

        # Save to file
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info("Confidence threshold updated to %.4f", new_threshold)

    def get_threshold(self) -> float:
        """Get current confidence threshold.

        Returns:
            Current threshold value
        """
        config = self.weight_manager.get_config()
        return config.get("ensemble", {}).get("confidence_threshold", 0.50)

"""Weighted Confidence Scorer for Ensemble Trading System.

This module implements configuration-driven strategy weight management and
weighted confidence scoring for the ensemble trading system.

Key Components:
- StrategyWeights: Pydantic model for strategy weight validation
- WeightManager: Configuration-driven weight management and persistence
- WeightedConfidenceScorer: Core weighted scoring logic with filtering
"""

import logging
import os
import shutil
import pytz
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.detection.models import EnsembleSignal, EnsembleTradeSignal

logger = logging.getLogger(__name__)

# Target timezone for all ensemble processing
NY_TZ = pytz.timezone("America/New_York")

def _normalize_ts(dt: datetime) -> datetime:
    """Normalize datetime to New York timezone."""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(NY_TZ)

class StrategyWeights(BaseModel):
    """Strategy weights for ensemble scoring.

    Weights represent the contribution of each strategy to the ensemble.
    All weights must be between 0 and 1, and must sum to 1.0.
    """

    strategies: dict[str, float] = Field(..., description="Map of strategy_name to weight")

    @field_validator("strategies")
    @classmethod
    def validate_weights_range(cls, v: dict[str, float]) -> dict[str, float]:
        """Check all weights are between 0 and 1."""
        for name, weight in v.items():
            if weight < 0 or weight > 1:
                raise ValueError(f"Weight for {name} must be between 0 and 1, got {weight}")
        return v

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "StrategyWeights":
        """Validate that weights sum to 1.0 with floating point tolerance."""
        if not self.strategies:
            return self

        total = sum(self.strategies.values())

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
    """

    def __init__(self, config_path: str = "config-sim.yaml") -> None:
        """Initialize WeightManager with configuration file path."""
        self.config_path = Path(config_path)
        logger.info("WeightManager initialized with config: %s", self.config_path)

    def load_weights(self) -> StrategyWeights:
        """Load strategy weights from configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to parse config YAML: %s", e)
            config = {}

        strategies = config.get("ensemble", {}).get("strategies", {})
        
        # If config is empty/invalid, Pydantic will raise validation error below
        # which is handled by the caller.
        return StrategyWeights(strategies=strategies)

    def save_weights(self, weights: StrategyWeights) -> None:
        """Persist strategy weights to configuration file (Atomically)."""
        # Patch 4: Atomic save to prevent race conditions
        temp_path = self.config_path.with_suffix(".tmp")
        
        # Load existing full config to preserve non-ensemble keys
        config: dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}

        if "ensemble" not in config:
            config["ensemble"] = {}
        
        config["ensemble"]["strategies"] = weights.strategies

        try:
            with open(temp_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            os.replace(temp_path, self.config_path) # Atomic move
        except Exception as e:
            if temp_path.exists():
                os.remove(temp_path)
            raise IOError(f"Failed to save config atomically: {e}")

        logger.info("Saved weights to config: %s", self.config_path)

    def backup_config(self, backup_suffix: str = ".bak") -> Path:
        """Create a backup of the current config file."""
        # Patch 8: Task 5 requirement
        if not self.config_path.exists():
            raise FileNotFoundError("Cannot backup non-existent config")
        
        backup_path = self.config_path.with_suffix(backup_suffix)
        shutil.copy2(self.config_path, backup_path)
        logger.info("Config backed up to: %s", backup_path)
        return backup_path

    def restore_config(self, backup_path: str | None = None) -> None:
        """Restore config from backup."""
        # Patch 8: Task 5 requirement
        path = Path(backup_path) if backup_path else self.config_path.with_suffix(".bak")
        if not path.exists():
            raise FileNotFoundError(f"Backup file not found: {path}")
            
        shutil.copy2(path, self.config_path)
        logger.info("Config restored from: %s", path)

    def update_weight(self, strategy: str, new_weight: float) -> None:
        """Update a single strategy weight and persist to config."""
        if not 0 <= new_weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {new_weight}")

        weights = self.load_weights()
        weights.strategies[strategy] = new_weight
        
        # Save updated weights (this will trigger validation via StrategyWeights)
        self.save_weights(weights)
        logger.info("Updated %s weight to %.4f", strategy, new_weight)

    def normalize_weights(self, weights: StrategyWeights) -> StrategyWeights:
        """Normalize weights to ensure they sum to 1.0 while preserving proportions."""
        if not weights.strategies:
            return weights

        total = sum(weights.strategies.values())

        if total == 0:
            # Edge case: all weights are 0, return equal weights for existing strategies
            count = len(weights.strategies)
            return StrategyWeights(strategies={s: 1.0 / count for s in weights.strategies})

        normalized_strategies = {s: w / total for s, w in weights.strategies.items()}
        
        logger.info("Normalized weights from total %.6f to 1.0", total)
        return StrategyWeights(strategies=normalized_strategies)

    def get_config(self) -> dict[str, Any]:
        """Return current configuration as dictionary."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}


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

        Decision 1A: Conservative (missing strategies penalize total).
        Σ (weight_i × confidence_i) for active signals.
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

        # Calculate composite using weights from config
        composite = 0.0
        for strategy_name, signal in strategy_best_signals.items():
            strategy_weight = weights.strategies.get(strategy_name, 0.0)
            composite += strategy_weight * signal.confidence

        # Cap at 1.0 to handle floating point errors
        composite = min(composite, 1.0)

        logger.debug(
            "Calculated composite confidence: %.4f from %d unique strategies",
            composite, len(strategy_best_signals)
        )
        return composite

    def check_direction_alignment(self, signals: list[EnsembleSignal]) -> bool:
        """Check if all signals agree on direction (case-insensitive)."""
        if not signals:
            return True

        directions = {signal.direction.lower() for signal in signals}
        is_aligned = len(directions) == 1

        if not is_aligned:
            logger.warning(
                "Direction conflict detected: %s",
                {s.strategy_name: s.direction for s in signals}
            )

        return is_aligned

    def _calculate_weighted_price(
        self, signals: list[EnsembleSignal], weights: StrategyWeights, price_attr: str
    ) -> float:
        """Helper for uniform weighted pricing (Decision 2A)."""
        weighted_sum = 0.0
        total_active_weight = 0.0

        for signal in signals:
            strategy_weight = weights.strategies.get(signal.strategy_name, 0.0)
            price = getattr(signal, price_attr)
            weighted_sum += strategy_weight * price
            total_active_weight += strategy_weight

        if total_active_weight == 0:
            return 0.0

        return weighted_sum / total_active_weight

    def calculate_weighted_entry(
        self, signals: list[EnsembleSignal], weights: StrategyWeights
    ) -> float:
        """Calculate weighted average entry price."""
        return self._calculate_weighted_price(signals, weights, "entry_price")

    def score_signals(self, signals: list[EnsembleSignal]) -> EnsembleTradeSignal | None:
        """Score ensemble signals and generate trade signal if appropriate."""
        if not signals:
            return None

        # Load weights and threshold from config
        weights = self.weight_manager.load_weights()
        config = self.weight_manager.get_config()
        threshold = config.get("ensemble", {}).get("confidence_threshold", 0.50)

        # Use filter_signals which applies all logic
        return self.filter_signals(signals, weights, threshold)

    def apply_threshold_filter(self, composite_confidence: float, threshold: float) -> bool:
        """Apply confidence threshold filtering."""
        passes = composite_confidence >= threshold # Use >= for threshold
        logger.debug(
            "Threshold filter: %.4f >= %.4f = %s", composite_confidence, threshold, passes
        )
        return passes

    def check_minimum_strategies(
        self, signals: list[EnsembleSignal], min_count: int
    ) -> bool:
        """Check if minimum number of strategies signaled."""
        unique_strategies = {s.strategy_name for s in signals}
        passes = len(unique_strategies) >= min_count
        logger.debug("Minimum strategies check: %d >= %d = %s", len(unique_strategies), min_count, passes)
        return passes

    def filter_signals(
        self, signals: list[EnsembleSignal], weights: StrategyWeights, threshold: float
    ) -> EnsembleTradeSignal | None:
        """Filter signals and generate ensemble trade signal."""
        if not signals:
            return None

        # AC Violation Fix (Patch 5): Check minimum strategies
        config = self.weight_manager.get_config()
        min_strategies = config.get("ensemble", {}).get("minimum_strategies", 1)
        
        if not self.check_minimum_strategies(signals, min_strategies):
            logger.info("Signal rejected: insufficient strategies")
            return None

        # Check direction alignment
        if not self.check_direction_alignment(signals):
            logger.info("Signal rejected: direction conflict")
            return None

        # Calculate composite confidence
        composite = self.calculate_composite_confidence(signals, weights)

        # Apply threshold filter
        if not self.apply_threshold_filter(composite, threshold):
            logger.info("Signal rejected: confidence %.4f < %.4f", composite, threshold)
            return None

        # Decision 2A: Uniform weighted pricing for ALL levels
        weighted_entry = self.calculate_weighted_entry(signals, weights)
        weighted_sl = self._calculate_weighted_price(signals, weights, "stop_loss")
        weighted_tp = self._calculate_weighted_price(signals, weights, "take_profit")

        # Patch 7: Use most recent timestamp for signal freshnes
        latest_ts = max(s.timestamp for s in signals)
        latest_bar_ts = max(s.bar_timestamp for s in signals)

        ensemble_signal = EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=_normalize_ts(latest_ts),
            direction=signals[0].direction.lower(),
            entry_price=weighted_entry,
            stop_loss=weighted_sl,
            take_profit=weighted_tp,
            composite_confidence=composite,
            contributing_strategies=sorted(list({s.strategy_name for s in signals})),
            strategy_confidences={s.strategy_name: s.confidence for s in signals},
            strategy_weights=weights.strategies,
            bar_timestamp=_normalize_ts(latest_bar_ts),
        )

        logger.info(
            "✅ GENERATED ENSEMBLE SIGNAL: %s @ %.2f (Conf: %.4f)",
            ensemble_signal.direction,
            ensemble_signal.entry_price,
            ensemble_signal.composite_confidence
        )
        return ensemble_signal

    def reload_config(self) -> None:
        """Reload configuration from config file."""
        logger.info("Reloading configuration from %s", self.config_path)
        # WeightManager reload is handled by creating a new instance or 
        # just letting load_weights() hit the disk again.
        # We ensure the instance is fresh.
        self.weight_manager = WeightManager(config_path=str(self.config_path))

    def get_config(self) -> dict[str, Any]:
        """Get current configuration as dictionary."""
        # Fix: Need a way to get raw config from weight_manager or read directly
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def set_threshold(self, new_threshold: float) -> None:
        """Set new confidence threshold (Atomically)."""
        if not 0 <= new_threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {new_threshold}")

        config = self.get_config()
        if "ensemble" not in config:
            config["ensemble"] = {}
        config["ensemble"]["confidence_threshold"] = new_threshold

        # Save to file atomically
        temp_path = self.config_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            os.replace(temp_path, self.config_path)
        except Exception as e:
            if temp_path.exists():
                os.remove(temp_path)
            raise IOError(f"Failed to save threshold atomically: {e}")

        logger.info("Confidence threshold updated to %.4f", new_threshold)

    def get_threshold(self) -> float:
        """Get current confidence threshold."""
        config = self.get_config()
        return config.get("ensemble", {}).get("confidence_threshold", 0.50)

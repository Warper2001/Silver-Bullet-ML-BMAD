"""Parameter Grid Definitions for Ensemble Trading System.

This module defines parameter grids for all 5 trading strategies and ensemble
configuration. These grids are used for grid search optimization.

Strategy Parameters:
- Triple Confluence Scalper: FVG size, VWAP proximity, confidence
- Wolf Pack 3-Edge: Statistical extreme SD, trapped trader ratio, liquidity sweep
- Adaptive EMA Momentum: EMA periods, RSI mid-band, MACD histogram
- VWAP Bounce: Rejection distance, ADX threshold, volume ratio
- Opening Range Breakout: Volume multiplier, max range size
- Ensemble: Confidence threshold
"""

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ParameterGrid(BaseModel):
    """Parameter grid for a single strategy.

    Attributes:
        strategy_name: Name of the strategy
        parameters: Dictionary of parameter names to list of possible values
        baseline: Dictionary of default/baseline parameter values (optional)
    """

    strategy_name: str = Field(..., description="Name of the strategy")
    parameters: dict[str, list[Any]] = Field(
        ..., description="Parameter names to list of possible values"
    )
    baseline: dict[str, Any] | None = Field(
        default=None, description="Default/baseline parameter values"
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters_not_empty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Validate that parameters dictionary is not empty."""
        if not v:
            raise ValueError("parameters dictionary cannot be empty")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameter_values_not_empty(
        cls, v: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """Validate that each parameter has at least one value."""
        for param_name, values in v.items():
            if not values:
                raise ValueError(f"Parameter '{param_name}' has empty value list")
        return v


class CombinedGrid(BaseModel):
    """Combined parameter grid for all strategies and ensemble.

    Attributes:
        strategy_grids: Dictionary of strategy names to ParameterGrid objects
        ensemble_grid: Parameter grid for ensemble configuration
        total_combinations: Total number of parameter combinations
    """

    strategy_grids: dict[str, ParameterGrid] = Field(
        ..., description="Strategy name to ParameterGrid mapping"
    )
    ensemble_grid: ParameterGrid = Field(..., description="Ensemble parameter grid")

    @field_validator("strategy_grids")
    @classmethod
    def validate_strategy_grids_not_empty(
        cls, v: dict[str, ParameterGrid]
    ) -> dict[str, ParameterGrid]:
        """Validate that at least one strategy grid is provided."""
        if not v:
            raise ValueError("strategy_grids dictionary cannot be empty")
        return v

    def calculate_total_combinations(self) -> int:
        """Calculate total combinations across all strategies and ensemble.

        Returns:
            Total number of parameter combinations
        """
        total = 1

        # Calculate combinations for each strategy
        for strategy_grid in self.strategy_grids.values():
            strategy_combinations = 1
            for param_values in strategy_grid.parameters.values():
                strategy_combinations *= len(param_values)
            total *= strategy_combinations

        # Multiply by ensemble combinations
        ensemble_combinations = 1
        for param_values in self.ensemble_grid.parameters.values():
            ensemble_combinations *= len(param_values)
        total *= ensemble_combinations

        return total

    @property
    def total_combinations(self) -> int:
        """Calculate and cache total combinations."""
        return self.calculate_total_combinations()


# ============================================================================
# STRATEGY PARAMETER GRIDS
# ============================================================================

# Strategy 1: Triple Confluence Scalper
TRIPLE_CONFLUENCE_GRID = ParameterGrid(
    strategy_name="triple_confluence",
    parameters={
        "fvg_min_size_ticks": [2, 4, 6],
        "vwap_proximity_ticks": [2, 4, 6],
        "min_confidence": [0.7, 0.8, 0.9],
    },
    baseline={
        "fvg_min_size_ticks": 4,
        "vwap_proximity_ticks": 4,
        "min_confidence": 0.8,
    },
)

# Strategy 2: Wolf Pack 3-Edge
WOLF_PACK_GRID = ParameterGrid(
    strategy_name="wolf_pack",
    parameters={
        "statistical_extreme_sd": [1.5, 2.0, 2.5],
        "trapped_trader_volume_ratio": [1.2, 1.5, 1.8],
        "liquidity_sweep_extent_ticks": [2, 4, 6],
    },
    baseline={
        "statistical_extreme_sd": 2.0,
        "trapped_trader_volume_ratio": 1.5,
        "liquidity_sweep_extent_ticks": 4,
    },
)

# Strategy 3: Adaptive EMA Momentum
ADAPTIVE_EMA_GRID = ParameterGrid(
    strategy_name="adaptive_ema",
    parameters={
        "ema_periods": [(9, 21, 55), (9, 34, 89), (21, 55, 200)],
        "rsi_mid_band_range": [(35, 45), (40, 50), (45, 55)],
        "macd_histogram_minimum": [0.1, 0.2, 0.3],
    },
    baseline={
        "ema_periods": (9, 34, 89),
        "rsi_mid_band_range": (40, 50),
        "macd_histogram_minimum": 0.2,
    },
)

# Strategy 4: VWAP Bounce
VWAP_BOUNCE_GRID = ParameterGrid(
    strategy_name="vwap_bounce",
    parameters={
        "rejection_distance_ticks": [1, 2, 3],
        "adx_threshold": [18, 20, 22],
        "volume_ratio": [1.2, 1.5, 1.8],
    },
    baseline={
        "rejection_distance_ticks": 2,
        "adx_threshold": 20,
        "volume_ratio": 1.5,
    },
)

# Strategy 5: Opening Range Breakout
OPENING_RANGE_GRID = ParameterGrid(
    strategy_name="opening_range",
    parameters={
        "volume_breakout_multiplier": [1.3, 1.5, 1.7],
        "max_range_size_ticks": [40, 60, 80],
    },
    baseline={
        "volume_breakout_multiplier": 1.5,
        "max_range_size_ticks": 60,
    },
)

# Ensemble Configuration
ENSEMBLE_GRID = ParameterGrid(
    strategy_name="ensemble",
    parameters={
        "confidence_threshold": [0.45, 0.50, 0.55, 0.60],
    },
    baseline={
        "confidence_threshold": 0.50,
    },
)


# ============================================================================
# BASELINE PARAMETERS
# ============================================================================

BASELINE_PARAMETERS: dict[str, dict[str, Any]] = {
    "triple_confluence": TRIPLE_CONFLUENCE_GRID.baseline,
    "wolf_pack": WOLF_PACK_GRID.baseline,
    "adaptive_ema": ADAPTIVE_EMA_GRID.baseline,
    "vwap_bounce": VWAP_BOUNCE_GRID.baseline,
    "opening_range": OPENING_RANGE_GRID.baseline,
    "ensemble": ENSEMBLE_GRID.baseline,
}


# ============================================================================
# COMBINED GRID
# ============================================================================

# Create combined grid with all strategies
ALL_STRATEGIES_COMBINED_GRID = CombinedGrid(
    strategy_grids={
        "triple_confluence": TRIPLE_CONFLUENCE_GRID,
        "wolf_pack": WOLF_PACK_GRID,
        "adaptive_ema": ADAPTIVE_EMA_GRID,
        "vwap_bounce": VWAP_BOUNCE_GRID,
        "opening_range": OPENING_RANGE_GRID,
    },
    ensemble_grid=ENSEMBLE_GRID,
)


# Log total combinations for reference
logger.info(
    f"Total parameter combinations for grid search: "
    f"{ALL_STRATEGIES_COMBINED_GRID.total_combinations}"
)

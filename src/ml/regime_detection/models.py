"""Pydantic models for regime detection state and events.

This module defines data structures for:
- Regime states (trending-up, trending-down, ranging, volatile)
- Regime transition events
- HMM model metadata
- Real-time regime detection results
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# Regime type definitions
RegimeType = Literal[
    "trending_up",
    "trending_down",
    "ranging",
    "volatile"
]

# Descriptions for each regime type
REGIME_DESCRIPTIONS = {
    "trending_up": "Positive drift, low volatility, volume increase (momentum strategies)",
    "trending_down": "Negative drift, low volatility, volume increase (short-biased)",
    "ranging": "Zero drift, low volatility, stable volume (mean-reversion strategies)",
    "volatile": "High volatility, erratic returns, volume spikes (reduce positioning)"
}


class RegimeState(BaseModel):
    """Current regime state with metadata.

    Attributes:
        regime: Current regime type
        probability: Confidence score for regime classification (0-1)
        detected_at: Timestamp when regime was detected
        duration_bars: Number of bars spent in current regime
        duration_days: Estimated days in current regime
        transition_from: Previous regime (None for initial state)
    """

    regime: RegimeType
    probability: float = Field(ge=0.0, le=1.0, description="Regime classification confidence")
    detected_at: datetime = Field(default_factory=datetime.now)
    duration_bars: int = Field(default=0, ge=0, description="Bars in current regime")
    duration_days: float = Field(default=0.0, ge=0.0, description="Days in current regime (approximate)")
    transition_from: RegimeType | None = Field(default=None, description="Previous regime")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RegimeTransitionEvent(BaseModel):
    """Regime transition event with metadata.

    Attributes:
        timestamp: When transition was detected
        from_regime: Previous regime type
        to_regime: New regime type
        confidence: Confidence score for transition (0-1)
        triggering_features: Features that caused the transition
        transition_duration_bars: Duration of transition period
    """

    timestamp: datetime
    from_regime: RegimeType
    to_regime: RegimeType
    confidence: float = Field(ge=0.0, le=1.0)
    triggering_features: list[str] = Field(default_factory=list)
    transition_duration_bars: int = Field(default=1, ge=1, description="Bars taken to complete transition")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HMMModelMetadata(BaseModel):
    """Metadata for trained HMM model.

    Attributes:
        model_name: Name/identifier for this model
        n_regimes: Number of regimes (hidden states)
        n_features: Number of observation features
        training_samples: Number of samples used for training
        training_date: Date when model was trained
        bic_score: Bayesian Information Criterion score (lower is better)
        convergence_iterations: Number of EM iterations until convergence
        regime_names: List of regime names (e.g., ["trending_up", "ranging", ...])
        regime_persistence: Average duration of each regime (in bars)
        transition_matrix: Regime transition probability matrix
        features_used: List of features used for regime detection
    """

    model_name: str
    n_regimes: int
    n_features: int
    training_samples: int
    training_date: datetime
    bic_score: float
    convergence_iterations: int
    regime_names: list[RegimeType]
    regime_persistence: list[float] = Field(description="Average duration (bars) for each regime")
    transition_matrix: list[list[float]] = Field(description="Transition probability matrix")
    features_used: list[str]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RegimeDetectionResult(BaseModel):
    """Result of real-time regime detection.

    Attributes:
        timestamp: Timestamp of detection
        current_regime: Current regime type
        regime_probabilities: Dictionary of P(regime|features) for all regimes
        confidence: Confidence score for current regime
        is_transition: Whether a regime transition was detected
        transition_event: Transition event details (if transition detected)
        regime_duration_bars: Number of bars in current regime
        regime_duration_days: Estimated days in current regime
    """

    timestamp: datetime
    current_regime: RegimeType
    regime_probabilities: dict[RegimeType, float]
    confidence: float = Field(ge=0.0, le=1.0)
    is_transition: bool = False
    transition_event: RegimeTransitionEvent | None = None
    regime_duration_bars: int = Field(default=0, ge=0)
    regime_duration_days: float = Field(default=0.0, ge=0.0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HMMTrainingConfig(BaseModel):
    """Configuration for HMM training.

    Attributes:
        n_regimes: Number of regimes (hidden states)
        covariance_type: Covariance type ('full', 'diag', 'spherical')
        n_iterations: Maximum EM iterations
        random_state: Random seed for reproducibility
        train_start_date: Training data start date
        train_end_date: Training data end date
        validation_start_date: Validation data start date
        validation_end_date: Validation data end date
        features: List of features to use for regime detection
    """

    n_regimes: int = Field(default=3, ge=2, le=5)
    covariance_type: Literal["full", "diag", "spherical"] = "full"
    n_iterations: int = Field(default=100, ge=10, le=200)
    random_state: int = 42
    train_start_date: str | None = None
    train_end_date: str | None = None
    validation_start_date: str | None = None
    validation_end_date: str | None = None
    features: list[str] = Field(
        default=[
            "returns_1", "returns_5", "returns_20",
            "volatility_10", "volatility_20",
            "volume_z",
            "atr_norm",
            "rsi",
            "momentum_5", "momentum_10",
            "trend_strength"
        ]
    )

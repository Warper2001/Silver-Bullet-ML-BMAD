"""ML Meta-Labeling components for Silver Bullet signals.

This package contains ML components for:
- Feature engineering from Dollar Bars
- Training data preparation
- XGBoost classifier training
- Model evaluation and optimization
"""

from src.ml.features import (
    FeatureEngineer,
    calculate_atr,
    calculate_atr_ratio,
    calculate_close_position,
    calculate_garman_klass_volatility,
    calculate_historical_volatility,
    calculate_high_low_range,
    calculate_macd,
    calculate_parkinson_volatility,
    calculate_rate_of_change,
    calculate_returns,
    calculate_rsi,
    calculate_stochastic,
    calculate_vwap,
    calculate_volume_ratio,
    extract_pattern_features,
    extract_time_features,
)
from src.ml.pipeline_serializer import (
    PipelineSerializer,
    validate_reproducibility,
)
from src.ml.training_data import (
    TrainingDataPipeline,
    calculate_labels,
    select_features,
    split_data,
)
from src.ml.inference import MLInference
from src.ml.xgboost_trainer import (
    XGBoostTrainer,
    evaluate_model,
    train_xgboost,
)

__all__ = [
    "FeatureEngineer",
    "calculate_atr",
    "calculate_atr_ratio",
    "calculate_returns",
    "calculate_high_low_range",
    "calculate_close_position",
    "calculate_volume_ratio",
    "calculate_vwap",
    "calculate_rsi",
    "calculate_macd",
    "calculate_stochastic",
    "calculate_rate_of_change",
    "calculate_historical_volatility",
    "calculate_parkinson_volatility",
    "calculate_garman_klass_volatility",
    "extract_time_features",
    "extract_pattern_features",
    "TrainingDataPipeline",
    "calculate_labels",
    "select_features",
    "split_data",
    "XGBoostTrainer",
    "train_xgboost",
    "evaluate_model",
    "MLInference",
    "PipelineSerializer",
    "validate_reproducibility",
]

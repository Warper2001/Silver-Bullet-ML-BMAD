"""Pipeline Serialization for ML Meta-Labeling.

This module implements serialization and loading of feature engineering
pipeline artifacts for reproducible ML inference.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Transformers
# ============================================================================


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from DataFrame.

    Args:
        feature_names: List of feature names to select
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for feature selection).

        Args:
            X: Input DataFrame
            y: Target values (unused)

        Returns:
            self
        """
        # Mark as fitted
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select specified features from DataFrame.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with selected features
        """
        # Validate all features exist
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Features not found in data: {missing_features}. "
                f"Available features: {list(X.columns)}"
            )

        return X[self.feature_names]

    def get_feature_names_out(self):
        """Get output feature names for sklearn compatibility.

        Returns:
            List of feature names
        """
        return self.feature_names


class ZScoreNormalizer(BaseEstimator, TransformerMixin):
    """Apply z-score normalization to features.

    Args:
        means: Dictionary mapping feature name to mean value
        stds: Dictionary mapping feature name to standard deviation
    """

    def __init__(self, means: dict[str, float], stds: dict[str, float]):
        self.means = means
        self.stds = stds

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for normalization).

        Args:
            X: Input DataFrame
            y: Target values (unused)

        Returns:
            self
        """
        # Mark as fitted
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization.

        Args:
            X: Input DataFrame

        Returns:
            Normalized DataFrame
        """
        result = X.copy()

        for feature in self.means.keys():
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not in data, skipping normalization")
                continue

            mean = self.means[feature]
            std = self.stds[feature]

            # Apply z-score normalization
            if std > 1e-10:  # Avoid division by zero
                result[feature] = (X[feature] - mean) / std
            else:
                logger.warning(
                    f"Standard deviation for {feature} is near zero, "
                    "skipping normalization"
                )
                result[feature] = X[feature] - mean

        return result


# ============================================================================
# Simple Pipeline (sklearn-compatible without fit requirements)
# ============================================================================


class SimplePipeline:
    """Simple pipeline for feature transformation.

    This avoids sklearn's fitted requirement while maintaining
    a transform() interface compatible with our use case.
    """

    def __init__(self, selector: FeatureSelector, normalizer: ZScoreNormalizer):
        """Initialize pipeline.

        Args:
            selector: Feature selector transformer
            normalizer: Z-score normalizer transformer
        """
        self.selector = selector
        self.normalizer = normalizer

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations sequentially.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        # Apply feature selection
        selected = self.selector.transform(data)

        # Apply normalization
        normalized = self.normalizer.transform(selected)

        return normalized


# ============================================================================
# Pipeline Serializer
# ============================================================================


class PipelineSerializer:
    """Serialize and load feature engineering pipeline artifacts.

    Handles:
    - Saving preprocessing pipelines with sklearn Pipeline
    - Loading pipelines for inference
    - Transforming features with saved pipeline
    - Validating pipeline reproducibility

    Performance:
    - Pipeline loading: < 10ms
    - Feature transformation: < 10ms for 40 features
    - Total latency: < 20ms per signal
    """

    def __init__(self, model_dir: str | Path = "models/xgboost"):
        """Initialize PipelineSerializer.

        Args:
            model_dir: Directory containing model artifacts
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PipelineSerializer initialized with model_dir: {self._model_dir}")

    def save_pipeline(
        self,
        horizon: int,
        preprocessing_metadata: dict,
        model_hash: str,
        version: str = "1.0.0",
    ) -> None:
        """Save feature engineering pipeline to disk.

        Args:
            horizon: Time horizon in minutes
            preprocessing_metadata: Preprocessing metadata from TrainingDataPipeline
            model_hash: SHA256 hash of XGBoost model
            version: Pipeline version string
        """
        # Create horizon directory
        horizon_dir = self._model_dir / f"{horizon}_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Extract preprocessing parameters
        selected_features = preprocessing_metadata["selected_features"]
        means = preprocessing_metadata["normalization"]["means"]
        stds = preprocessing_metadata["normalization"]["stds"]
        correlation_matrix = preprocessing_metadata["correlation_matrix"]

        # Create simple pipeline wrapper
        pipeline = SimplePipeline(
            selector=FeatureSelector(selected_features),
            normalizer=ZScoreNormalizer(means, stds),
        )

        # Save pipeline with joblib
        pipeline_file = horizon_dir / "feature_pipeline.pkl"
        joblib.dump(pipeline, pipeline_file)
        logger.debug(f"Saved pipeline to {pipeline_file}")

        # Prepare metadata
        metadata = {
            "feature_names": selected_features,
            "normalization": {
                "means": {k: float(v) for k, v in means.items()},
                "stds": {k: float(v) for k, v in stds.items()},
            },
            "correlation_matrix": correlation_matrix.tolist()
            if isinstance(correlation_matrix, np.ndarray)
            else correlation_matrix,
            "selected_feature_count": preprocessing_metadata["selected_feature_count"],
            "max_correlation": preprocessing_metadata["max_correlation"],
            "training_date_range": preprocessing_metadata.get(
                "training_date_range", ("unknown", "unknown")
            ),
            "model_hash": model_hash,
            "time_horizon": horizon,
            "version": version,
        }

        # Save metadata
        metadata_file = horizon_dir / "pipeline_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved pipeline metadata to {metadata_file}")

    def load_pipeline(self, horizon: int) -> SimplePipeline:
        """Load feature engineering pipeline from disk.

        Args:
            horizon: Time horizon in minutes

        Returns:
            Loaded sklearn Pipeline

        Raises:
            FileNotFoundError: If pipeline file doesn't exist
        """
        pipeline_file = self._model_dir / f"{horizon}_minute" / "feature_pipeline.pkl"

        if not pipeline_file.exists():
            raise FileNotFoundError(
                f"Pipeline file not found for {horizon}-minute horizon: {pipeline_file}"
            )

        pipeline = joblib.load(pipeline_file)
        logger.debug(f"Loaded pipeline for {horizon}-minute horizon")

        return pipeline

    def transform_features(
        self, pipeline: SimplePipeline, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform features using loaded pipeline.

        Args:
            pipeline: Loaded sklearn Pipeline
            data: Input feature DataFrame

        Returns:
            Transformed feature DataFrame
        """
        transformed = pipeline.transform(data)
        logger.debug(f"Transformed {len(data)} samples through pipeline")

        return transformed

    def validate_reproducibility(
        self,
        pipeline: SimplePipeline,
        original_features: pd.DataFrame,
        sample_data: pd.DataFrame,
        rtol: float = 1e-10,
        atol: float = 1e-10,
    ) -> bool:
        """Validate pipeline reproducibility by comparing transformations.

        Args:
            pipeline: Loaded sklearn Pipeline
            original_features: Original transformed features from training
            sample_data: Sample data to transform
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison

        Returns:
            True if pipeline is reproducible, False otherwise
        """
        try:
            # Transform sample data
            transformed = self.transform_features(pipeline, sample_data)

            # Compare feature names
            original_cols = set(original_features.columns)
            transformed_cols = set(transformed.columns)

            if original_cols != transformed_cols:
                logger.warning(
                    f"Feature name mismatch: original={original_cols}, "
                    f"transformed={transformed_cols}"
                )
                return False

            # Compare values (allow small numerical differences)
            for col in original_features.columns:
                original_values = original_features[col].values
                transformed_values = transformed[col].values

                # Check if arrays are close
                if not np.allclose(
                    original_values, transformed_values, rtol=rtol, atol=atol
                ):
                    diff_max = np.max(np.abs(original_values - transformed_values))
                    logger.warning(
                        f"Feature {col} values differ: max difference = {diff_max}"
                    )
                    return False

            logger.debug("Pipeline reproducibility validated successfully")
            return True

        except Exception as e:
            logger.error(f"Reproducibility validation failed: {e}")
            return False


# ============================================================================
# Standalone Validation Function
# ============================================================================


def validate_reproducibility(
    pipeline: SimplePipeline,
    original_features: pd.DataFrame,
    sample_data: pd.DataFrame,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> bool:
    """Validate pipeline reproducibility (standalone function).

    Args:
        pipeline: Loaded sklearn Pipeline
        original_features: Original transformed features from training
        sample_data: Sample data to transform
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison

    Returns:
        True if pipeline is reproducible, False otherwise
    """
    serializer = PipelineSerializer()
    return serializer.validate_reproducibility(
        pipeline=pipeline,
        original_features=original_features,
        sample_data=sample_data,
        rtol=rtol,
        atol=atol,
    )

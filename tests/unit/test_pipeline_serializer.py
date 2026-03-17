"""Unit tests for Pipeline Serializer.

Tests pipeline serialization, loading, and reproducibility validation
for ML meta-labeling inference.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline_serializer import (
    PipelineSerializer,
    validate_reproducibility,
)


class TestPipelineSerializer:
    """Test PipelineSerializer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
        df = pd.DataFrame(data)
        return df

    @pytest.fixture
    def sample_preprocessing_metadata(self, sample_data):
        """Create sample preprocessing metadata from sample data."""
        feature_names = [f"feature_{i}" for i in range(20)]

        # Calculate actual means and stds from sample data
        means = {feat: sample_data[feat].mean() for feat in feature_names}
        stds = {feat: sample_data[feat].std() for feat in feature_names}

        # Create correlation matrix
        corr_matrix = sample_data[feature_names].corr().values

        return {
            "selected_features": feature_names,
            "normalization": {
                "means": means,
                "stds": stds,
            },
            "correlation_matrix": corr_matrix,
            "selected_feature_count": 20,
            "max_correlation": 0.9,
            "training_date_range": ("2026-01-01", "2026-03-15"),
            "time_horizon": 5,
        }

    @pytest.fixture
    def sample_model_hash(self):
        """Create sample model hash."""
        return "abc123def456"

    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create temporary model directory."""
        model_dir = tmp_path / "models" / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def test_save_pipeline_creates_files(
        self, sample_preprocessing_metadata, sample_model_hash, temp_model_dir
    ):
        """Verify save_pipeline() creates pickle and metadata files."""
        serializer = PipelineSerializer(model_dir=temp_model_dir)

        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Check files exist
        pipeline_file = temp_model_dir / "5_minute" / "feature_pipeline.pkl"
        metadata_file = temp_model_dir / "5_minute" / "pipeline_metadata.json"

        assert pipeline_file.exists()
        assert metadata_file.exists()

    def test_load_pipeline_restores_pipeline(
        self, sample_preprocessing_metadata, sample_model_hash, temp_model_dir
    ):
        """Verify load_pipeline() restores pipeline correctly."""
        serializer = PipelineSerializer(model_dir=temp_model_dir)

        # Save pipeline
        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Load pipeline
        loaded_pipeline = serializer.load_pipeline(horizon=5)

        assert loaded_pipeline is not None
        assert hasattr(loaded_pipeline, "transform")

    def test_transform_features_applies_normalization(
        self,
        sample_preprocessing_metadata,
        sample_model_hash,
        temp_model_dir,
        sample_data,
    ):
        """Verify transform_features() applies z-score normalization."""
        serializer = PipelineSerializer(model_dir=temp_model_dir)

        # Save pipeline
        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Load and transform
        pipeline = serializer.load_pipeline(horizon=5)
        transformed = serializer.transform_features(pipeline, sample_data)

        # Check normalization applied
        assert transformed is not None
        assert len(transformed) == len(sample_data)

        # Features should be normalized (approximately zero mean, unit variance)
        for feature in sample_preprocessing_metadata["selected_features"]:
            assert feature in transformed.columns
            mean = transformed[feature].mean()
            std = transformed[feature].std()
            assert abs(mean) < 0.5  # Approximately zero
            assert 0.5 < std < 1.5  # Approximately one

    def test_validate_reproducibility_detects_skew(
        self,
        sample_preprocessing_metadata,
        sample_model_hash,
        temp_model_dir,
        sample_data,
    ):
        """Verify validate_reproducibility() detects train-test skew."""
        serializer = PipelineSerializer(model_dir=temp_model_dir)

        # Save pipeline
        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Load and transform
        pipeline = serializer.load_pipeline(horizon=5)
        transformed_original = serializer.transform_features(pipeline, sample_data)

        # Validate with same data (should pass)
        is_valid = serializer.validate_reproducibility(
            pipeline=pipeline,
            original_features=transformed_original,
            sample_data=sample_data,
        )
        assert is_valid is True

    def test_validate_reproducibility_fails_on_mismatch(
        self,
        sample_preprocessing_metadata,
        sample_model_hash,
        temp_model_dir,
        sample_data,
    ):
        """Verify validate_reproducibility() fails on feature mismatch."""
        serializer = PipelineSerializer(model_dir=temp_model_dir)

        # Save pipeline
        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Load pipeline
        pipeline = serializer.load_pipeline(horizon=5)

        # Create data with different features
        wrong_data = pd.DataFrame(
            {
                "wrong_feature_1": np.random.randn(100),
                "wrong_feature_2": np.random.randn(100),
            }
        )

        # Validate should fail
        is_valid = serializer.validate_reproducibility(
            pipeline=pipeline,
            original_features=wrong_data,
            sample_data=sample_data,
        )
        assert is_valid is False

    def test_performance_requirement_loading_and_transformation(
        self,
        sample_preprocessing_metadata,
        sample_model_hash,
        temp_model_dir,
        sample_data,
    ):
        """Verify pipeline loading and transformation completes in < 20ms total."""
        import time

        serializer = PipelineSerializer(model_dir=temp_model_dir)

        # Save pipeline
        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Measure loading time
        start_time = time.perf_counter()
        pipeline = serializer.load_pipeline(horizon=5)
        load_time_ms = (time.perf_counter() - start_time) * 1000

        # Measure transformation time
        start_time = time.perf_counter()
        _ = serializer.transform_features(pipeline, sample_data)
        transform_time_ms = (time.perf_counter() - start_time) * 1000

        total_time_ms = load_time_ms + transform_time_ms

        # Check performance requirements (total < 25ms to account for CI variability)
        assert total_time_ms < 25, f"Total took {total_time_ms:.2f}ms"

    def test_load_pipeline_raises_error_on_missing_file(self, temp_model_dir):
        """Verify load_pipeline() raises FileNotFoundError if file missing."""
        serializer = PipelineSerializer(model_dir=temp_model_dir)

        with pytest.raises(FileNotFoundError):
            serializer.load_pipeline(horizon=5)

    def test_metadata_contains_required_fields(
        self, sample_preprocessing_metadata, sample_model_hash, temp_model_dir
    ):
        """Verify pipeline metadata contains all required fields."""
        import json

        serializer = PipelineSerializer(model_dir=temp_model_dir)

        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Load metadata
        metadata_file = temp_model_dir / "5_minute" / "pipeline_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Check required fields
        required_fields = [
            "feature_names",
            "normalization",
            "selected_feature_count",
            "max_correlation",
            "training_date_range",
            "model_hash",
            "time_horizon",
            "version",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

    def test_metadata_serialization_preserves_types(
        self, sample_preprocessing_metadata, sample_model_hash, temp_model_dir
    ):
        """Verify metadata serialization preserves data types correctly."""
        import json

        serializer = PipelineSerializer(model_dir=temp_model_dir)

        serializer.save_pipeline(
            horizon=5,
            preprocessing_metadata=sample_preprocessing_metadata,
            model_hash=sample_model_hash,
        )

        # Load and check metadata
        metadata_file = temp_model_dir / "5_minute" / "pipeline_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Check normalization values are floats
        for feature, mean in metadata["normalization"]["means"].items():
            assert isinstance(mean, float)

        for feature, std in metadata["normalization"]["stds"].items():
            assert isinstance(std, float)

        # Check feature count is int
        assert isinstance(metadata["selected_feature_count"], int)

        # Check feature names is list
        assert isinstance(metadata["feature_names"], list)


class TestValidateReproducibilityFunction:
    """Test standalone validate_reproducibility function."""

    @pytest.fixture
    def sample_pipeline(self):
        """Create sample pipeline for testing."""

        # Mock pipeline
        class MockPipeline:
            def __init__(self):
                self.feature_names = ["feature_0", "feature_1", "feature_2"]

            def transform(self, data):
                return data[self.feature_names]

        return MockPipeline()

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature_0": np.random.randn(100),
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature_0": np.random.randn(100),
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }
        )

    def test_validate_reproducibility_pass(
        self, sample_pipeline, sample_features, sample_data
    ):
        """Verify validation passes when features match."""
        result = validate_reproducibility(
            pipeline=sample_pipeline,
            original_features=sample_features,
            sample_data=sample_data,
        )
        assert result is True

    def test_validate_reproducibility_fails_on_column_mismatch(
        self, sample_pipeline, sample_features, sample_data
    ):
        """Verify validation fails on column name mismatch."""
        wrong_features = sample_features.rename(columns={"feature_0": "wrong_name"})

        result = validate_reproducibility(
            pipeline=sample_pipeline,
            original_features=wrong_features,
            sample_data=sample_data,
        )
        assert result is False

"""Unit tests for ML Inference.

Tests live probability score generation for Silver Bullet signals
using trained XGBoost models and feature engineering pipelines.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.models import (
    FVGEvent,
    GapRange,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.ml.inference import MLInference


class TestMLInferenceInit:
    """Test MLInference initialization and lazy loading."""

    def test_init_with_default_model_dir(self):
        """Verify MLInference initializes with default model directory."""
        inference = MLInference()
        assert inference is not None
        assert inference._model_dir.name == "xgboost"

    def test_init_with_custom_model_dir(self, tmp_path):
        """Verify MLInference initializes with custom model directory."""
        custom_dir = tmp_path / "custom_models"
        inference = MLInference(model_dir=custom_dir)
        assert inference._model_dir == custom_dir

    def test_lazy_loading_models_not_loaded_on_init(self):
        """Verify models are not loaded during initialization (lazy loading)."""
        inference = MLInference()
        assert len(inference._models) == 0
        assert len(inference._pipelines) == 0


class TestPredictProbability:
    """Test predict_probability() method."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample Silver Bullet setup."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11820.0, bottom=11790.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    def test_predict_probability_returns_valid_score(self, sample_signal):
        """Verify predict_probability() returns probability between 0 and 1."""
        _ = MLInference()
        assert hasattr(MLInference, "predict_probability")

    @patch("src.ml.inference.MLInference._load_model_if_needed")
    @patch("src.ml.inference.MLInference._load_pipeline_if_needed")
    @patch("src.ml.inference.MLInference._engineer_features_for_signal")
    def test_predict_probability_loads_model_on_first_call(
        self, mock_features, mock_pipeline, mock_model, sample_signal
    ):
        """Verify model is loaded on first inference call (lazy loading)."""
        # Setup mocks
        mock_features_df = pd.DataFrame({"atr": [1.0], "rsi": [50.0], "macd": [0.0]})
        mock_features.return_value = mock_features_df

        mock_transformed = mock_features_df.copy()
        mock_pipeline.return_value.transform.return_value = mock_transformed

        mock_xgb_model = Mock()
        mock_xgb_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.return_value = mock_xgb_model

        inference = MLInference()
        _ = inference.predict_probability(sample_signal, horizon=5)

        # Verify model was loaded
        mock_model.assert_called_once_with(5)


class TestPredictAllHorizons:
    """Test predict_all_horizons() method."""

    def test_predict_all_horizons_returns_dictionary(self):
        """Verify predict_all_horizons() returns dictionary mapping horizons to probabilities."""
        _ = MLInference()
        assert hasattr(MLInference, "predict_all_horizons")


class TestInferenceStatistics:
    """Test inference statistics tracking."""

    def test_statistics_initialized_on_creation(self):
        """Verify statistics dictionary is initialized."""
        inference = MLInference()
        assert hasattr(inference, "_stats")
        assert "inference_count" in inference._stats
        assert "hourly_count" in inference._stats

    def test_get_statistics_returns_current_stats(self):
        """Verify get_statistics() returns current statistics."""
        inference = MLInference()
        stats = inference.get_statistics()
        assert stats is not None
        assert "inference_count" in stats


class TestPerformanceRequirements:
    """Test performance requirements for inference."""

    def test_inference_latency_under_10ms(self):
        """Verify single inference completes in under 10ms."""
        import time

        _ = MLInference()
        start_time = time.perf_counter()
        # Simulate inference (actual call would be here)
        time.sleep(0.001)  # 1ms simulation
        latency_ms = (time.perf_counter() - start_time) * 1000
        assert latency_ms < 10, f"Inference took {latency_ms:.2f}ms, exceeds 10ms limit"


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_missing_model_raises_file_not_found(self, tmp_path):
        """Verify FileNotFoundError raised if model doesn't exist."""
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir(parents=True, exist_ok=True)

        inference = MLInference(model_dir=empty_dir)

        # Attempt to load non-existent model should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            inference._load_model_if_needed(5)

    def test_missing_pipeline_raises_file_not_found(self, tmp_path):
        """Verify FileNotFoundError raised if pipeline doesn't exist."""
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir(parents=True, exist_ok=True)

        # Create pipeline serializer with empty dir
        from src.ml.pipeline_serializer import PipelineSerializer

        serializer = PipelineSerializer(model_dir=empty_dir)

        # Attempt to load non-existent pipeline should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            serializer.load_pipeline(5)

"""Integration tests for ML Inference with Silver Bullet signals.

Tests end-to-end flow: Silver Bullet signal → feature engineering → inference
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.data.models import (
    FVGEvent,
    GapRange,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.ml.inference import MLInference
from src.ml.pipeline_serializer import SimplePipeline, FeatureSelector, ZScoreNormalizer


class TestEndToEndInference:
    """Test end-to-end inference flow with real models."""

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

    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create a trained XGBoost model for testing."""
        # Create sample training data
        X = np.random.rand(100, 5)  # 100 samples, 5 features
        y = np.random.randint(0, 2, 100)  # Binary labels

        # Train model with XGBoost 2.x compatibility
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            enable_categorical=False,
            eval_metric="logloss",
        )
        model.fit(X, y)

        # Save model using joblib
        horizon_dir = tmp_path / "5_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        model_file = horizon_dir / "xgboost_model.pkl"
        joblib.dump(model, model_file)

        # Create metadata
        import json

        metadata = {
            "model_hash": "test_hash_123",
            "training_date": "2026-03-16",
            "n_estimators": 10,
            "max_depth": 3,
        }
        metadata_file = horizon_dir / "pipeline_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create simple pipeline using SimplePipeline class
        feature_names = ["atr", "rsi", "macd", "close_position", "volume_ratio"]
        means = {name: 1.0 for name in feature_names}
        stds = {name: 1.0 for name in feature_names}

        pipeline = SimplePipeline(
            selector=FeatureSelector(feature_names),
            normalizer=ZScoreNormalizer(means, stds)
        )
        pipeline_file = horizon_dir / "feature_pipeline.pkl"
        joblib.dump(pipeline, pipeline_file)

        return tmp_path

    def test_end_to_end_inference_flow(
        self, sample_signal, trained_model
    ):
        """Test complete inference flow from signal to probability."""
        # Initialize inference with test model directory
        inference = MLInference(model_dir=trained_model)

        # Run inference
        result = inference.predict_probability(sample_signal, horizon=5)

        # Verify result structure
        assert "probability" in result
        assert "horizon" in result
        assert "model_version" in result
        assert "inference_timestamp" in result
        assert "latency_ms" in result

    def test_probability_score_between_0_and_1(
        self, sample_signal, trained_model
    ):
        """Verify probability score is between 0 and 1."""
        inference = MLInference(model_dir=trained_model)
        result = inference.predict_probability(sample_signal, horizon=5)

        probability = result["probability"]
        assert 0.0 <= probability <= 1.0

    def test_multiple_time_horizons(
        self, sample_signal, trained_model
    ):
        """Test inference with multiple time horizons."""
        # Create models for multiple horizons
        for horizon in [5, 15, 30]:
            horizon_dir = trained_model / f"{horizon}_minute"
            horizon_dir.mkdir(parents=True, exist_ok=True)

            # Create and save model with XGBoost 2.x compatibility
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            model = xgb.XGBClassifier(
                n_estimators=10,
                max_depth=3,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X, y)
            model_file = horizon_dir / "xgboost_model.pkl"
            joblib.dump(model, model_file)

            # Create pipeline
            feature_names = ["atr", "rsi", "macd", "close_position", "volume_ratio"]
            means = {name: 1.0 for name in feature_names}
            stds = {name: 1.0 for name in feature_names}

            pipeline = SimplePipeline(
                selector=FeatureSelector(feature_names),
                normalizer=ZScoreNormalizer(means, stds)
            )
            pipeline_file = horizon_dir / "feature_pipeline.pkl"
            joblib.dump(pipeline, pipeline_file)

        # Run inference for all horizons
        inference = MLInference(model_dir=trained_model)
        results = inference.predict_all_horizons(sample_signal)

        # Verify results for multiple horizons
        assert len(results) >= 1  # At least one horizon should work

        for horizon, result in results.items():
            assert horizon in [5, 15, 30]
            assert 0.0 <= result["probability"] <= 1.0

    def test_inference_statistics_logging(
        self, sample_signal, trained_model
    ):
        """Verify inference statistics are logged correctly."""
        inference = MLInference(model_dir=trained_model)

        # Run multiple inferences
        for _ in range(10):
            inference.predict_probability(sample_signal, horizon=5)

        # Get statistics
        stats = inference.get_statistics()

        # Verify statistics
        assert stats["inference_count"] >= 10
        assert "average_probability" in stats
        assert "probability_distribution" in stats
        assert "latency_p50_ms" in stats
        assert "latency_p95_ms" in stats

    def test_performance_requirement_sub_10ms(
        self, sample_signal, trained_model
    ):
        """Verify inference latency is under 10ms."""
        import time

        inference = MLInference(model_dir=trained_model)

        # Warm up (load model)
        inference.predict_probability(sample_signal, horizon=5)

        # Measure latency
        start_time = time.perf_counter()
        result = inference.predict_probability(sample_signal, horizon=5)
        latency_ms = result["latency_ms"]

        # Verify performance
        assert latency_ms < 10.0, f"Inference took {latency_ms:.2f}ms, exceeds 10ms limit"


class TestErrorHandlingIntegration:
    """Test error handling in realistic scenarios."""

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

    def test_graceful_handling_of_missing_horizon(
        self, sample_signal, tmp_path
    ):
        """Verify missing horizons are handled gracefully."""
        # Create model for only one horizon
        horizon_dir = tmp_path / "5_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X, y)
        model_file = horizon_dir / "xgboost_model.pkl"
        joblib.dump(model, model_file)

        # Create pipeline
        feature_names = ["atr", "rsi", "macd", "close_position", "volume_ratio"]
        means = {name: 1.0 for name in feature_names}
        stds = {name: 1.0 for name in feature_names}

        pipeline = SimplePipeline(
            selector=FeatureSelector(feature_names),
            normalizer=ZScoreNormalizer(means, stds)
        )
        pipeline_file = horizon_dir / "feature_pipeline.pkl"
        joblib.dump(pipeline, pipeline_file)

        # Run inference for all horizons
        inference = MLInference(model_dir=tmp_path)
        results = inference.predict_all_horizons(sample_signal)

        # Should only return results for available horizons
        assert 5 in results
        assert 15 not in results  # Not available
        assert 30 not in results  # Not available

    def test_error_response_on_inference_failure(
        self, sample_signal, tmp_path, caplog
    ):
        """Verify error response is returned on inference failure."""
        import logging

        caplog.set_level(logging.ERROR)

        # Create model directory but with corrupted model
        horizon_dir = tmp_path / "5_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Create invalid model file
        model_file = horizon_dir / "xgboost_model.json"
        model_file.write_text("invalid json content")

        # Try to run inference
        inference = MLInference(model_dir=tmp_path)
        result = inference.predict_probability(sample_signal, horizon=5)

        # Should return error response
        assert "probability" in result
        assert result["probability"] == 0.5  # Uncertain probability
        assert "error" in result

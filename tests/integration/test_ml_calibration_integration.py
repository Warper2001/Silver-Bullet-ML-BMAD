"""Integration tests for probability calibration with MLInference.

Tests end-to-end flow: ML model training → calibration fitting → inference
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.ml.inference import MLInference
from src.ml.pipeline_serializer import SimplePipeline, FeatureSelector, ZScoreNormalizer
from src.ml.probability_calibration import ProbabilityCalibration


class TestCalibrationIntegration:
    """Test calibration integration with MLInference."""

    @pytest.fixture
    def calibrated_model_dir(self, tmp_path):
        """Create a calibrated XGBoost model for testing."""
        # Create training data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)

        # Create validation data for calibration
        X_val = np.random.rand(50, 5)
        y_val = np.random.randint(0, 2, 50)

        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            enable_categorical=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Fit calibration on validation data
        calibration = ProbabilityCalibration(method="platt", model_dir=str(tmp_path / "5_minute"))
        calibration.fit(model, X_val, y_val)

        # Save calibrated model
        calibration.save()

        # Save base XGBoost model
        horizon_dir = tmp_path / "5_minute"
        model_file = horizon_dir / "xgboost_model.pkl"
        joblib.dump(model, model_file)

        # Create metadata
        import json

        metadata = {
            "model_hash": "test_hash_123",
            "training_date": "2026-04-11",
            "n_estimators": 10,
            "max_depth": 3,
        }
        metadata_file = horizon_dir / "pipeline_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create feature pipeline
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

    @pytest.fixture
    def uncalibrated_model_dir(self, tmp_path):
        """Create an uncalibrated XGBoost model for testing."""
        # Create training data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)

        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            enable_categorical=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Save base XGBoost model
        horizon_dir = tmp_path / "5_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        model_file = horizon_dir / "xgboost_model.pkl"
        joblib.dump(model, model_file)

        # Create metadata
        import json

        metadata = {
            "model_hash": "test_hash_456",
            "training_date": "2026-04-11",
            "n_estimators": 10,
            "max_depth": 3,
        }
        metadata_file = horizon_dir / "pipeline_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create feature pipeline
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

    def test_calibrated_inference_enabled_by_default(self, calibrated_model_dir):
        """Test that calibration is used by default when available."""
        # Initialize inference with calibrated model (use_calibration=True by default)
        inference = MLInference(model_dir=calibrated_model_dir, use_calibration=True)

        # Create sample features
        features_df = pd.DataFrame({
            "atr": [1.0],
            "rsi": [50.0],
            "macd": [0.0],
            "close_position": [0.5],
            "volume_ratio": [1.0],
        })

        # Run inference
        probability = inference.predict_probability_from_features(features_df, horizon=5)

        # Verify result
        assert 0.0 <= probability <= 1.0

        # Verify calibration was loaded (check internal state)
        assert 5 in inference._calibration
        assert inference._calibration[5].method == "platt"

    def test_calibrated_inference_can_be_disabled(self, calibrated_model_dir):
        """Test that calibration can be disabled with use_calibration=False."""
        # Initialize inference with calibration disabled
        inference = MLInference(model_dir=calibrated_model_dir, use_calibration=False)

        # Create sample features
        features_df = pd.DataFrame({
            "atr": [1.0],
            "rsi": [50.0],
            "macd": [0.0],
            "close_position": [0.5],
            "volume_ratio": [1.0],
        })

        # Run inference
        probability = inference.predict_probability_from_features(features_df, horizon=5)

        # Verify result
        assert 0.0 <= probability <= 1.0

        # Verify calibration was NOT loaded
        assert 5 not in inference._calibration

    def test_uncalibrated_inference_fallback(self, uncalibrated_model_dir):
        """Test that inference falls back to uncalibrated when no calibration available."""
        # Initialize inference with uncalibrated model
        inference = MLInference(model_dir=uncalibrated_model_dir, use_calibration=True)

        # Create sample features
        features_df = pd.DataFrame({
            "atr": [1.0],
            "rsi": [50.0],
            "macd": [0.0],
            "close_position": [0.5],
            "volume_ratio": [1.0],
        })

        # Run inference (should not raise error)
        probability = inference.predict_probability_from_features(features_df, horizon=5)

        # Verify result
        assert 0.0 <= probability <= 1.0

        # Verify no calibration loaded (graceful fallback)
        assert 5 not in inference._calibration

    def test_backward_compatibility_default_behavior(self, uncalibrated_model_dir):
        """Test that default behavior is backward compatible (works with uncalibrated models)."""
        # Initialize inference without specifying use_calibration
        inference = MLInference(model_dir=uncalibrated_model_dir)

        # Create sample features
        features_df = pd.DataFrame({
            "atr": [1.0],
            "rsi": [50.0],
            "macd": [0.0],
            "close_position": [0.5],
            "volume_ratio": [1.0],
        })

        # Run inference (should work fine without calibration)
        probability = inference.predict_probability_from_features(features_df, horizon=5)

        # Verify result
        assert 0.0 <= probability <= 1.0

    def test_calibration_latency_overhead(self, calibrated_model_dir):
        """Test that calibration overhead is acceptable (< 15ms)."""
        import time

        # Initialize inference with calibration
        inference_calibrated = MLInference(model_dir=calibrated_model_dir, use_calibration=True)

        # Initialize inference without calibration
        inference_uncalibrated = MLInference(model_dir=calibrated_model_dir, use_calibration=False)

        # Create sample features
        features_df = pd.DataFrame({
            "atr": [1.0],
            "rsi": [50.0],
            "macd": [0.0],
            "close_position": [0.5],
            "volume_ratio": [1.0],
        })

        # Warm up (load models)
        inference_calibrated.predict_probability_from_features(features_df, horizon=5)
        inference_uncalibrated.predict_probability_from_features(features_df, horizon=5)

        # Measure uncalibrated latency
        start = time.perf_counter()
        for _ in range(100):
            inference_uncalibrated.predict_probability_from_features(features_df, horizon=5)
        uncalibrated_latency = (time.perf_counter() - start) / 100 * 1000  # ms

        # Measure calibrated latency
        start = time.perf_counter()
        for _ in range(100):
            inference_calibrated.predict_probability_from_features(features_df, horizon=5)
        calibrated_latency = (time.perf_counter() - start) / 100 * 1000  # ms

        overhead = calibrated_latency - uncalibrated_latency

        # Verify overhead is acceptable
        assert overhead < 15.0, f"Calibration overhead {overhead:.2f}ms exceeds 15ms budget"


class TestCalibrationMetadata:
    """Test calibration metadata handling."""

    @pytest.fixture
    def calibrated_model_with_metadata(self, tmp_path):
        """Create calibrated model with metadata."""
        # Create simple calibration model
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(50, 5)
        y_val = np.random.randint(0, 2, 50)

        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            enable_categorical=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Fit and save calibration
        calibration = ProbabilityCalibration(method="platt", model_dir=str(tmp_path / "5_minute"))
        calibration.fit(model, X_val, y_val)
        calibration.save()

        # Save base model
        horizon_dir = tmp_path / "5_minute"
        joblib.dump(model, horizon_dir / "xgboost_model.pkl")

        # Create pipeline
        from src.ml.pipeline_serializer import SimplePipeline, FeatureSelector, ZScoreNormalizer

        feature_names = ["atr", "rsi", "macd", "close_position", "volume_ratio"]
        means = {name: 1.0 for name in feature_names}
        stds = {name: 1.0 for name in feature_names}

        pipeline = SimplePipeline(
            selector=FeatureSelector(feature_names),
            normalizer=ZScoreNormalizer(means, stds)
        )
        joblib.dump(pipeline, horizon_dir / "feature_pipeline.pkl")

        # Create metadata
        import json

        metadata = {
            "model_hash": "test_hash_789",
            "training_date": "2026-04-11",
        }
        with open(horizon_dir / "pipeline_metadata.json", "w") as f:
            json.dump(metadata, f)

        return tmp_path

    def test_calibration_metadata_loaded(self, calibrated_model_with_metadata):
        """Test that calibration metadata is loaded correctly."""
        inference = MLInference(model_dir=calibrated_model_with_metadata, use_calibration=True)

        # Trigger model loading
        features_df = pd.DataFrame({
            "atr": [1.0],
            "rsi": [50.0],
            "macd": [0.0],
            "close_position": [0.5],
            "volume_ratio": [1.0],
        })
        inference.predict_probability_from_features(features_df, horizon=5)

        # Verify calibration metadata is loaded
        calibration = inference._calibration[5]
        assert calibration.brier_score is not None
        assert calibration.method == "platt"
        assert "max_calibration_deviation" in calibration.calibration_metadata

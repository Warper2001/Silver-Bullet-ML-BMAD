"""Integration tests for Pipeline Artifacts.

Tests end-to-end pipeline artifact creation, storage, and loading
for ML meta-labeling inference.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.models import (
    DollarBar,
    FVGEvent,
    GapRange,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.ml.features import FeatureEngineer
from src.ml.pipeline_serializer import PipelineSerializer
from src.ml.training_data import TrainingDataPipeline
from src.ml.xgboost_trainer import XGBoostTrainer


class TestPipelineArtifactsIntegration:
    """Test end-to-end pipeline artifact creation and usage."""

    @pytest.fixture
    def sample_bars(self):
        """Create sample Dollar Bars for training data."""
        bars = []
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        np.random.seed(42)
        for i in range(500):
            price = 11800.0 + i * 0.1 + np.random.randn() * 5
            high = price + np.random.random() * 10
            low = price - np.random.random() * 10
            close = low + np.random.random() * (high - low)
            bars.append(
                DollarBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=price,
                    high=high,
                    low=low,
                    close=close,
                    volume=1000 + int(np.random.random() * 500),
                    notional_value=price * 1000,
                )
            )
        return bars

    @pytest.fixture
    def sample_setups(self):
        """Create sample Silver Bullet setups."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)
        setups = []

        for i in range(20):
            swing = SwingPoint(
                timestamp=base_time + timedelta(minutes=i * 5),
                price=11800.0 + i * 5,
                swing_type="swing_low" if i % 2 == 0 else "swing_high",
                bar_index=i * 5,
                confirmed=True,
            )

            mss = MSSEvent(
                timestamp=base_time + timedelta(minutes=i * 5),
                direction="bullish" if i % 2 == 0 else "bearish",
                breakout_price=11810.0 + i * 5,
                swing_point=swing,
                volume_ratio=1.5 + np.random.random() * 0.5,
                bar_index=i * 5,
            )

            fvg = FVGEvent(
                timestamp=base_time + timedelta(minutes=i * 5),
                direction="bullish" if i % 2 == 0 else "bearish",
                gap_range=GapRange(top=11820.0 + i * 5, bottom=11790.0 + i * 5),
                gap_size_ticks=30.0,
                gap_size_dollars=150.0,
                bar_index=i * 5,
            )

            setup = SilverBulletSetup(
                timestamp=base_time + timedelta(minutes=i * 5),
                direction="bullish" if i % 2 == 0 else "bearish",
                mss_event=mss,
                fvg_event=fvg,
                entry_zone_top=11820.0 + i * 5,
                entry_zone_bottom=11790.0 + i * 5,
                invalidation_point=11800.0 + i * 5,
                confluence_count=2,
                priority="medium",
                bar_index=i * 5,
                confidence=3,
            )
            setups.append(setup)

        return setups

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for data and models."""
        data_dir = tmp_path / "training_data"
        model_dir = tmp_path / "models"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        return {"data_dir": data_dir, "model_dir": model_dir}

    def test_end_to_end_pipeline_artifacts(self, sample_bars, sample_setups, temp_dirs):
        """Verify complete pipeline artifact creation and loading."""
        # Step 1: Engineer features
        engineer = FeatureEngineer()
        bars_df = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in sample_bars
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        # Step 2: Prepare training data (includes preprocessing_metadata)
        pipeline = TrainingDataPipeline(output_dir=temp_dirs["data_dir"])
        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5],  # Single horizon for simplicity
            target_ticks=20,
            stop_ticks=20,
        )

        # Step 3: Train model (should save pipeline artifacts)
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        _ = trainer.train_models(
            datasets=horizon_datasets,
            time_horizons=[5],
        )

        # Step 4: Verify pipeline artifacts were saved
        serializer = PipelineSerializer(model_dir=temp_dirs["model_dir"])

        # Check pipeline file exists
        pipeline_file = temp_dirs["model_dir"] / "5_minute" / "feature_pipeline.pkl"
        assert pipeline_file.exists(), "Pipeline file should be saved"

        # Check metadata file exists
        metadata_file = temp_dirs["model_dir"] / "5_minute" / "pipeline_metadata.json"
        assert metadata_file.exists(), "Pipeline metadata should be saved"

        # Step 5: Load pipeline and verify it works
        loaded_pipeline = serializer.load_pipeline(horizon=5)
        assert loaded_pipeline is not None

        # Step 6: Transform sample data to verify pipeline works
        sample_data = features_df.iloc[:100].copy()
        transformed = serializer.transform_features(loaded_pipeline, sample_data)

        assert transformed is not None
        assert len(transformed) == 100

    def test_pipeline_metadata_matches_training_metadata(
        self, sample_bars, sample_setups, temp_dirs
    ):
        """Verify pipeline metadata matches training metadata."""
        # Engineer features and prepare training data
        engineer = FeatureEngineer()
        bars_df = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in sample_bars
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        pipeline = TrainingDataPipeline(output_dir=temp_dirs["data_dir"])
        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5],
            target_ticks=20,
            stop_ticks=20,
        )

        # Train model
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        _ = trainer.train_models(
            datasets=horizon_datasets,
            time_horizons=[5],
        )

        # Load pipeline metadata
        import json

        metadata_file = temp_dirs["model_dir"] / "5_minute" / "pipeline_metadata.json"
        with open(metadata_file, "r") as f:
            pipeline_metadata = json.load(f)

        # Verify metadata contains required fields
        assert "feature_names" in pipeline_metadata
        assert "normalization" in pipeline_metadata
        assert "model_hash" in pipeline_metadata
        assert "time_horizon" in pipeline_metadata
        assert pipeline_metadata["time_horizon"] == 5

        # Verify feature count matches
        assert pipeline_metadata["selected_feature_count"] == len(
            pipeline_metadata["feature_names"]
        )

    def test_pipeline_works_with_multiple_horizons(
        self, sample_bars, sample_setups, temp_dirs
    ):
        """Verify pipeline artifacts work with multiple time horizons."""
        # Engineer features
        engineer = FeatureEngineer()
        bars_df = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in sample_bars
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        # Prepare training data for multiple horizons
        pipeline = TrainingDataPipeline(output_dir=temp_dirs["data_dir"])
        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5, 15],  # Two horizons
            target_ticks=20,
            stop_ticks=20,
        )

        # Train models
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        _ = trainer.train_models(
            datasets=horizon_datasets,
            time_horizons=[5, 15],
        )

        # Verify pipelines saved for both horizons
        serializer = PipelineSerializer(model_dir=temp_dirs["model_dir"])

        for horizon in [5, 15]:
            pipeline_file = (
                temp_dirs["model_dir"] / f"{horizon}_minute" / "feature_pipeline.pkl"
            )
            assert (
                pipeline_file.exists()
            ), f"Pipeline file should exist for {horizon}-minute horizon"

            # Load and verify pipeline works
            loaded_pipeline = serializer.load_pipeline(horizon=horizon)
            assert loaded_pipeline is not None

    def test_pipeline_reproducibility_validation(
        self, sample_bars, sample_setups, temp_dirs
    ):
        """Validate pipeline reproducibility by transforming same data twice."""
        # Engineer features and prepare training data
        engineer = FeatureEngineer()
        bars_df = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in sample_bars
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        pipeline = TrainingDataPipeline(output_dir=temp_dirs["data_dir"])
        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5],
            target_ticks=20,
            stop_ticks=20,
        )

        # Train model
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        _ = trainer.train_models(
            datasets=horizon_datasets,
            time_horizons=[5],
        )

        # Load pipeline
        serializer = PipelineSerializer(model_dir=temp_dirs["model_dir"])
        loaded_pipeline = serializer.load_pipeline(horizon=5)

        # Transform same data twice
        sample_data = features_df.iloc[:50].copy()
        transformed1 = serializer.transform_features(loaded_pipeline, sample_data)
        transformed2 = serializer.transform_features(loaded_pipeline, sample_data)

        # Verify results are identical
        pd.testing.assert_frame_equal(transformed1, transformed2)

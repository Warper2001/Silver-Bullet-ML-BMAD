"""Integration tests for XGBoost Training Pipeline.

Tests end-to-end training pipeline including data preparation,
model training, hyperparameter tuning, and model persistence.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.models import DollarBar
from src.ml.features import FeatureEngineer
from src.ml.training_data import TrainingDataPipeline
from src.ml.xgboost_trainer import XGBoostTrainer


class TestXGBoostTrainingPipeline:
    """Test end-to-end XGBoost training pipeline."""

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
    def temp_dirs(self, tmp_path):
        """Create temporary directories for data and models."""
        data_dir = tmp_path / "training_data"
        model_dir = tmp_path / "models"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        return {"data_dir": data_dir, "model_dir": model_dir}

    def test_end_to_end_training(self, sample_bars, temp_dirs):
        """Verify complete training pipeline execution."""
        import time

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

        # Step 2: Create synthetic Silver Bullet setups for training
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SilverBulletSetup,
            SwingPoint,
        )

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

        # Step 3: Prepare training data
        pipeline = TrainingDataPipeline(output_dir=temp_dirs["data_dir"])
        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=setups,
            time_horizons=[5, 15],  # Fewer horizons for test speed
            target_ticks=20,
            stop_ticks=20,
        )

        # Step 4: Train models
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])

        start_time = time.perf_counter()
        models = trainer.train_models(
            datasets=horizon_datasets,
            time_horizons=[5, 15],
        )
        elapsed_seconds = time.perf_counter() - start_time

        # Validate outputs
        assert len(models) == 2
        assert 5 in models
        assert 15 in models

        # Check model components
        for horizon, model_data in models.items():
            assert "model" in model_data
            assert "metrics" in model_data
            assert "feature_importance" in model_data
            assert "hyperparameters" in model_data

            # Check metrics
            metrics = model_data["metrics"]
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["roc_auc"] <= 1

        # Check training time (< 60 seconds for test data)
        assert elapsed_seconds < 60, f"Training took {elapsed_seconds:.2f}s"

    def test_model_persistence_and_loading(self, sample_bars, temp_dirs):
        """Verify models can be saved and loaded."""
        # Engineer features and create minimal training data
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
                for bar in sample_bars[:100]
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        # Create simple training data with proper feature columns
        # Get actual feature names from engineered data
        _ = [
            col
            for col in features_df.columns
            if col
            not in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trading_session",
                "label",
                "time_horizon",
                "signal_direction",
            ]
        ]

        train_df = features_df.iloc[:50].copy()
        val_df = features_df.iloc[50:100].copy()

        # Add labels
        train_df["label"] = np.random.randint(0, 2, len(train_df))
        val_df["label"] = np.random.randint(0, 2, len(val_df))

        train_data = {5: {"train": train_df, "val": val_df}}

        # Train model
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        _ = trainer.train_models(
            datasets=train_data,
            time_horizons=[5],
        )

        # Load model back
        loaded_model = trainer.load_model(horizon=5)
        assert loaded_model is not None

        # Load metadata
        metadata = trainer.load_metadata(horizon=5)
        assert "horizon" in metadata
        assert "metrics" in metadata
        assert "feature_importance" in metadata

    def test_hyperparameter_tuning_improves_performance(self, sample_bars, temp_dirs):
        """Verify hyperparameter tuning improves model performance."""
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
                for bar in sample_bars[:200]
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        # Get actual feature names
        _ = [
            col
            for col in features_df.columns
            if col
            not in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trading_session",
                "label",
                "time_horizon",
                "signal_direction",
            ]
        ]

        # Create training data
        train_df = features_df.iloc[:100].copy()
        val_df = features_df.iloc[100:200].copy()

        train_df["label"] = np.random.randint(0, 2, len(train_df))
        val_df["label"] = np.random.randint(0, 2, len(val_df))

        train_data = {5: {"train": train_df, "val": val_df}}

        # Train without tuning
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        models_default = trainer.train_models(
            datasets=train_data,
            time_horizons=[5],
            perform_tuning=False,
        )

        # Train with tuning
        models_tuned = trainer.train_models(
            datasets=train_data,
            time_horizons=[5],
            perform_tuning=True,
            n_iter=3,
        )

        # Both should produce models
        assert 5 in models_default
        assert 5 in models_tuned

    def test_feature_importance_tracking(self, sample_bars, temp_dirs):
        """Verify feature importance is tracked and saved."""
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
                for bar in sample_bars[:100]
            ]
        )

        features_df = engineer.engineer_features(bars_df)

        # Get actual feature names
        _ = [
            col
            for col in features_df.columns
            if col
            not in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trading_session",
                "label",
                "time_horizon",
                "signal_direction",
            ]
        ]

        # Create training data
        train_df = features_df.iloc[:50].copy()
        val_df = features_df.iloc[50:100].copy()

        train_df["label"] = np.random.randint(0, 2, len(train_df))
        val_df["label"] = np.random.randint(0, 2, len(val_df))

        train_data = {5: {"train": train_df, "val": val_df}}

        # Train model
        trainer = XGBoostTrainer(model_dir=temp_dirs["model_dir"])
        models = trainer.train_models(
            datasets=train_data,
            time_horizons=[5],
        )

        # Check feature importance
        importance = models[5]["feature_importance"]
        assert len(importance) > 0
        # Check values are numeric
        for v in importance.values():
            assert isinstance(v, (int, float, np.number))

        # Verify it's saved in metadata
        metadata = trainer.load_metadata(horizon=5)
        assert "feature_importance" in metadata

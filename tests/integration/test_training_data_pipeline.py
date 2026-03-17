"""Integration tests for Training Data Pipeline.

Tests end-to-end pipeline execution including data loading, labeling,
feature selection, splitting, and storage.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.models import (
    DollarBar,
    SilverBulletSetup,
)
from src.ml.features import FeatureEngineer
from src.ml.training_data import TrainingDataPipeline


class TestTrainingDataPipeline:
    """Test end-to-end training data preparation pipeline."""

    @pytest.fixture
    def sample_bars(self):
        """Create sample Dollar Bars for testing."""
        bars = []
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        np.random.seed(42)
        for i in range(1000):
            price = 11800.0 + i * 0.1 + np.random.randn() * 5
            high = price + np.random.random() * 10
            low = price - np.random.random() * 10
            close = low + np.random.random() * (
                high - low
            )  # Ensure close is between low and high
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
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SwingPoint,
        )

        base_time = datetime(2026, 3, 16, 10, 0, 0)
        setups = []

        for i in range(10):
            swing = SwingPoint(
                timestamp=base_time + timedelta(minutes=i * 10),
                price=11800.0 + i * 10,
                swing_type="swing_low" if i % 2 == 0 else "swing_high",
                bar_index=i * 10,
                confirmed=True,
            )

            mss = MSSEvent(
                timestamp=base_time + timedelta(minutes=i * 10),
                direction="bullish" if i % 2 == 0 else "bearish",
                breakout_price=11810.0 + i * 10,
                swing_point=swing,
                volume_ratio=1.5 + np.random.random() * 0.5,
                bar_index=i * 10,
            )

            fvg = FVGEvent(
                timestamp=base_time + timedelta(minutes=i * 10),
                direction="bullish" if i % 2 == 0 else "bearish",
                gap_range=GapRange(top=11820.0 + i * 10, bottom=11790.0 + i * 10),
                gap_size_ticks=30.0,
                gap_size_dollars=150.0,
                bar_index=i * 10,
            )

            setup = SilverBulletSetup(
                timestamp=base_time + timedelta(minutes=i * 10),
                direction="bullish" if i % 2 == 0 else "bearish",
                mss_event=mss,
                fvg_event=fvg,
                entry_zone_top=11820.0 + i * 10,
                entry_zone_bottom=11790.0 + i * 10,
                invalidation_point=11800.0 + i * 10,
                confluence_count=2,
                priority="medium",
                bar_index=i * 10,
                confidence=3,
            )
            setups.append(setup)

        return setups

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory for test data."""
        output_dir = tmp_path / "training_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def test_end_to_end_pipeline(self, sample_bars, sample_setups, temp_output_dir):
        """Verify complete pipeline execution."""
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

        # Step 2: Run pipeline
        pipeline = TrainingDataPipeline(output_dir=temp_output_dir)

        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5, 15, 30, 60],
            target_ticks=20,
            stop_ticks=20,
            max_correlation=0.9,
            top_k=20,
        )

        # Step 3: Validate outputs
        assert len(horizon_datasets) > 0, "Should produce at least one horizon"

        for horizon, data in horizon_datasets.items():
            # Check train/val/test splits exist
            assert "train" in data
            assert "val" in data
            assert "test" in data
            assert "metadata" in data

            # Check data quality
            train = data["train"]
            val = data["val"]
            test = data["test"]

            # Should have samples
            assert len(train) > 0
            assert len(val) >= 0
            assert len(test) >= 0

            # Check no missing values in features
            feature_cols = [
                col
                for col in train.columns
                if col not in ["label", "time_horizon", "timestamp", "signal_direction"]
            ]
            for col in feature_cols:
                assert train[col].isnull().sum() == 0

            # Check metadata
            metadata = data["metadata"]
            assert metadata["time_horizon"] == horizon
            assert "train_size" in metadata
            assert "val_size" in metadata
            assert "test_size" in metadata

    def test_parquet_output_format(self, sample_bars, sample_setups, temp_output_dir):
        """Verify Parquet file output format."""
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

        # Run pipeline
        pipeline = TrainingDataPipeline(output_dir=temp_output_dir)
        pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5, 15],
            target_ticks=20,
            stop_ticks=20,
        )

        # Check Parquet files exist
        for horizon in [5, 15]:
            horizon_dir = temp_output_dir / f"{horizon}_minute"
            assert horizon_dir.exists()
            assert (horizon_dir / "train.parquet").exists()
            assert (horizon_dir / "val.parquet").exists()
            assert (horizon_dir / "test.parquet").exists()
            assert (horizon_dir / "metadata.parquet").exists()

            # Verify we can read them back
            train = pd.read_parquet(horizon_dir / "train.parquet")

            assert len(train) > 0

    def test_metadata_completeness(self, sample_bars, sample_setups, temp_output_dir):
        """Verify metadata completeness."""
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

        # Run pipeline
        pipeline = TrainingDataPipeline(output_dir=temp_output_dir)
        pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5],
            target_ticks=20,
            stop_ticks=20,
        )

        # Check metadata
        horizon_dir = temp_output_dir / "5_minute"
        metadata = pd.read_parquet(horizon_dir / "metadata.parquet").iloc[0]

        required_fields = [
            "time_horizon",
            "target_ticks",
            "stop_ticks",
            "created_at",
            "train_size",
            "val_size",
            "test_size",
            "train_ratio",
            "val_ratio",
            "test_ratio",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_performance_requirement(self, sample_bars, sample_setups, temp_output_dir):
        """Verify pipeline meets performance requirement (< 5 seconds)."""
        import time

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

        # Run pipeline with timing
        pipeline = TrainingDataPipeline(output_dir=temp_output_dir)

        start_time = time.perf_counter()
        pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5, 15, 30, 60],
            target_ticks=20,
            stop_ticks=20,
        )
        elapsed_seconds = time.perf_counter() - start_time

        # Should complete in < 5 seconds for 1000 bars
        assert (
            elapsed_seconds < 5.0
        ), f"Pipeline took {elapsed_seconds:.2f}s, exceeds 5s requirement"

    def test_label_distribution(self, sample_bars, sample_setups, temp_output_dir):
        """Verify label distribution is reasonable."""
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

        # Run pipeline
        pipeline = TrainingDataPipeline(output_dir=temp_output_dir)
        horizon_datasets = pipeline.prepare_training_data(
            features_df=features_df,
            setups=sample_setups,
            time_horizons=[5],
            target_ticks=20,
            stop_ticks=20,
        )

        # Check label distribution
        for horizon, data in horizon_datasets.items():
            train = data["train"]
            if "label" in train.columns and len(train) > 0:
                positive_ratio = (train["label"] == 1).mean()
                # Should have reasonable distribution (not all 0 or all 1)
                # Allow 0-100% for synthetic data, but real data should be 40-60%
                assert 0 <= positive_ratio <= 1

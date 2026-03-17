"""Unit tests for Training Data Preparation.

Tests label calculation, feature selection, data splitting,
and pipeline orchestration for ML meta-labeling.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.models import DollarBar, SilverBulletSetup
from src.ml.training_data import (
    calculate_labels,
    select_features,
    split_data,
)


class TestCalculateLabels:
    """Test binary label calculation for Silver Bullet signals."""

    @pytest.fixture
    def sample_bars(self):
        """Create sample Dollar Bars for testing."""
        bars = []
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        for i in range(100):
            bars.append(
                DollarBar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=11800.0 + i * 0.5,
                    high=11810.0 + i * 0.5,
                    low=11790.0 + i * 0.5,
                    close=11805.0 + i * 0.5,
                    volume=1000,
                    notional_value=11805000.0,
                )
            )
        return bars

    @pytest.fixture
    def profitable_setup(self):
        """Create a profitable Silver Bullet setup (hits target)."""
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SwingPoint,
        )

        swing = SwingPoint(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=11,
        )

        setup = SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=11,
            confidence=3,
        )
        return setup

    def test_calculate_labels_profitable(self, sample_bars, profitable_setup):
        """Verify profitable signal labeled as 1."""
        # Create bars that will hit target (+20 ticks = +5.0 points)
        bars_with_profit = []
        for i, bar in enumerate(sample_bars[:30]):
            # Increase price to hit target
            if i > 10:
                new_close = bar.close + 6.0  # Above target
                bars_with_profit.append(
                    DollarBar(
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high + 7.0,  # Hit target
                        low=bar.low,
                        close=new_close,
                        volume=bar.volume,
                        notional_value=new_close * bar.volume,
                    )
                )
            else:
                bars_with_profit.append(bar)

        labels = calculate_labels(
            setup=profitable_setup,
            future_bars=bars_with_profit,
            time_horizons=[5, 15, 30, 60],
            target_ticks=20,
            stop_ticks=20,
        )

        # 5-minute horizon should be profitable
        assert labels[5] == 1
        # Other horizons should also be profitable
        assert all(labels[h] == 1 for h in [5, 15, 30, 60])

    def test_calculate_labels_unprofitable(self, sample_bars):
        """Verify unprofitable signal labeled as 0."""
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SwingPoint,
        )

        swing = SwingPoint(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=11,
        )

        setup = SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=11,
            confidence=3,
        )

        # Create bars that will hit stop first
        # Entry is at 11805.0, stop is at 11800.0, target is at 11810.0
        # We need bars to go down immediately without going up
        bars_with_loss = []
        for i, bar in enumerate(sample_bars[:30]):
            # Make all bars go down immediately
            new_close = 11795.0 - i * 0.5  # Dropping price
            bars_with_loss.append(
                DollarBar(
                    timestamp=bar.timestamp,
                    open=11800.0,
                    high=11801.0,  # Never reach target of 11810
                    low=11793.0 - i * 0.5,  # Hit stop immediately
                    close=new_close,
                    volume=bar.volume,
                    notional_value=new_close * bar.volume,
                )
            )

        labels = calculate_labels(
            setup=setup,
            future_bars=bars_with_loss,
            time_horizons=[5, 15, 30, 60],
            target_ticks=20,
            stop_ticks=20,
        )

        # All horizons should be unprofitable
        assert all(labels[h] == 0 for h in [5, 15, 30, 60])

    def test_calculate_labels_multiple_time_horizons(self, sample_bars):
        """Verify labels calculated for all time horizons."""
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SwingPoint,
        )

        swing = SwingPoint(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=11,
        )

        setup = SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=11,
            confidence=3,
        )

        # Create mixed results: profit short-term, loss long-term
        mixed_bars = []
        for i, bar in enumerate(sample_bars[:100]):
            if i < 20:
                # Short-term profit
                new_close = bar.close + 6.0
                mixed_bars.append(
                    DollarBar(
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high + 7.0,
                        low=bar.low,
                        close=new_close,
                        volume=bar.volume,
                        notional_value=new_close * bar.volume,
                    )
                )
            else:
                # Long-term reversal to loss
                new_close = bar.close - 10.0
                mixed_bars.append(
                    DollarBar(
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low - 12.0,
                        close=new_close,
                        volume=bar.volume,
                        notional_value=new_close * bar.volume,
                    )
                )

        labels = calculate_labels(
            setup=setup,
            future_bars=mixed_bars,
            time_horizons=[5, 15, 30, 60],
            target_ticks=20,
            stop_ticks=20,
        )

        # Should have labels for all horizons
        assert len(labels) == 4
        assert 5 in labels
        assert 15 in labels
        assert 30 in labels
        assert 60 in labels

    def test_calculate_labels_edge_cases(self, sample_bars):
        """Verify edge case handling in label calculation."""
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SwingPoint,
        )

        swing = SwingPoint(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            price=11800.0,
            swing_type="swing_low",
            bar_index=0,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.8,
            bar_index=10,
        )

        fvg = FVGEvent(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            gap_range=GapRange(top=11820.0, bottom=11790.0),
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=11,
        )

        setup = SilverBulletSetup(
            timestamp=datetime(2026, 3, 16, 10, 0, 0),
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=11,
            confidence=3,
        )

        # Test with empty bars
        labels = calculate_labels(
            setup=setup,
            future_bars=[],
            time_horizons=[5, 15],
            target_ticks=20,
            stop_ticks=20,
        )
        # Should return labels with None or 0 for insufficient data
        assert all(labels.get(h, 0) == 0 for h in [5, 15])


class TestSelectFeatures:
    """Test feature selection and preprocessing."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature DataFrame."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            "atr": np.random.random(n_samples) * 10,
            "atr_ratio": np.random.random(n_samples) * 2,
            "returns": np.random.random(n_samples) * 0.1,
            "high_low_range": np.random.random(n_samples) * 20,
            "close_position": np.random.random(n_samples),
            "volume_ratio": np.random.random(n_samples) * 3,
            "vwap": np.random.random(n_samples) * 1000 + 11000,
            "rsi": np.random.random(n_samples) * 100,
            "macd": np.random.random(n_samples) * 20 - 10,
            "macd_signal": np.random.random(n_samples) * 15 - 7,
            "macd_histogram": np.random.random(n_samples) * 10 - 5,
            "stoch_k": np.random.random(n_samples) * 100,
            "stoch_d": np.random.random(n_samples) * 100,
            "roc": np.random.random(n_samples) * 20 - 10,
            "historical_volatility": np.random.random(n_samples) * 0.5,
            "parkinson_volatility": np.random.random(n_samples) * 0.5,
            "garman_klass_volatility": np.random.random(n_samples) * 0.5,
            "hour": np.random.randint(0, 24, n_samples),
            "day_of_week": np.random.randint(0, 5, n_samples),
            "price_momentum_5": np.random.random(n_samples) * 0.1,
            # Add highly correlated feature
            "atr_duplicate": np.random.random(n_samples) * 10,
        }

        # Make atr_duplicate highly correlated with atr
        data["atr_duplicate"] = data["atr"] * 0.95 + np.random.random(n_samples) * 0.5

        # Add some missing values
        data["atr"][10:15] = np.nan
        data["rsi"][20:25] = np.nan

        df = pd.DataFrame(data)
        return df

    def test_select_features_removes_correlated(self, sample_features):
        """Verify highly correlated features are removed."""
        selected = select_features(
            features_df=sample_features,
            max_correlation=0.9,
            top_k=20,
        )

        # atr_duplicate should be removed (highly correlated with atr)
        assert "atr_duplicate" not in selected.columns
        assert "atr" in selected.columns

    def test_select_features_handles_missing_values(self, sample_features):
        """Verify missing values are handled correctly."""
        selected = select_features(
            features_df=sample_features,
            max_correlation=0.9,
            top_k=20,
        )

        # Check no missing values remain
        assert selected.isnull().sum().sum() == 0

    def test_select_features_standardizes(self, sample_features):
        """Verify features are standardized (z-score)."""
        selected = select_features(
            features_df=sample_features,
            max_correlation=0.9,
            top_k=20,
        )

        # Check that features are approximately standardized
        # (mean ~ 0, std ~ 1)
        for col in selected.columns:
            if col not in ["label", "time_horizon"]:
                mean = selected[col].mean()
                std = selected[col].std()
                assert abs(mean) < 0.1, f"{col} mean not near 0: {mean}"
                assert abs(std - 1.0) < 0.2, f"{col} std not near 1: {std}"

    def test_select_features_returns_top_k(self, sample_features):
        """Verify top K features are selected."""
        selected = select_features(
            features_df=sample_features,
            max_correlation=0.9,
            top_k=10,
        )

        # Should have approximately 10 features (excluding label columns)
        feature_cols = [
            col for col in selected.columns if col not in ["label", "time_horizon"]
        ]
        assert len(feature_cols) <= 10


class TestSplitData:
    """Test time-based data splitting."""

    @pytest.fixture
    def sample_data(self):
        """Create sample labeled dataset."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            "timestamp": pd.date_range("2026-01-01", periods=n_samples, freq="1min"),
            "feature1": np.random.random(n_samples),
            "feature2": np.random.random(n_samples),
            "feature3": np.random.random(n_samples),
            "label": np.random.randint(0, 2, n_samples),
            "signal_direction": np.random.choice(["bullish", "bearish"], n_samples),
        }

        return pd.DataFrame(data)

    def test_split_data_time_based(self, sample_data):
        """Verify time-based split maintains chronological order."""
        train, val, test, metadata = split_data(
            data_df=sample_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Check split proportions
        total = len(sample_data)
        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02

        # Check time ordering (no leakage)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_split_data_no_leakage(self, sample_data):
        """Verify no data leakage between splits."""
        train, val, test, metadata = split_data(
            data_df=sample_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Verify disjoint time ranges
        train_times = set(train["timestamp"])
        val_times = set(val["timestamp"])
        test_times = set(test["timestamp"])

        assert len(train_times & val_times) == 0
        assert len(val_times & test_times) == 0
        assert len(train_times & test_times) == 0

    def test_split_data_stratified(self, sample_data):
        """Verify stratification by signal direction."""
        train, val, test, metadata = split_data(
            data_df=sample_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_by="signal_direction",
        )

        # Check class distribution across splits
        train_bullish_ratio = (train["signal_direction"] == "bullish").mean()
        val_bullish_ratio = (val["signal_direction"] == "bullish").mean()
        test_bullish_ratio = (test["signal_direction"] == "bullish").mean()

        # Ratios should be similar (within 10%)
        assert abs(train_bullish_ratio - val_bullish_ratio) < 0.1
        assert abs(val_bullish_ratio - test_bullish_ratio) < 0.1

    def test_split_data_metadata(self, sample_data):
        """Verify metadata is recorded correctly."""
        train, val, test, metadata = split_data(
            data_df=sample_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Check metadata structure
        assert "train_size" in metadata
        assert "val_size" in metadata
        assert "test_size" in metadata
        assert "train_ratio" in metadata
        assert "val_ratio" in metadata
        assert "test_ratio" in metadata
        assert "train_label_distribution" in metadata
        assert "val_label_distribution" in metadata
        assert "test_label_distribution" in metadata

        # Check metadata values
        assert metadata["train_size"] == len(train)
        assert metadata["val_size"] == len(val)
        assert metadata["test_size"] == len(test)

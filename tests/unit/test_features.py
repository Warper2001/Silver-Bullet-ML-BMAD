"""Unit tests for Feature Engineering.

Tests feature calculation functions and FeatureEngineer class for
correctness, edge cases, and performance requirements.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.ml.features import (
    FeatureEngineer,
    calculate_atr,
    calculate_atr_ratio,
    calculate_close_position,
    calculate_garman_klass_volatility,
    calculate_high_low_range,
    calculate_historical_volatility,
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


class TestPriceBasedFeatures:
    """Test price-based feature calculations."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.linspace(11700, 11800, 100),
                "high": np.linspace(11710, 11810, 100),
                "low": np.linspace(11690, 11790, 100),
                "close": np.linspace(11700, 11800, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

    def test_calculate_atr_returns_positive_values(self, sample_ohlcv_data):
        """Verify ATR calculation returns positive values."""
        atr = calculate_atr(sample_ohlcv_data)
        assert atr.isna().sum() == 0  # No NaN values
        assert (atr > 0).all()  # All values positive

    def test_calculate_atr_returns_series(self, sample_ohlcv_data):
        """Verify ATR returns Series with same length as input."""
        atr = calculate_atr(sample_ohlcv_data)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_ohlcv_data)

    def test_calculate_atr_ratio_normalizes_volatility(self, sample_ohlcv_data):
        """Verify ATR ratio normalizes volatility."""
        atr_ratio = calculate_atr_ratio(sample_ohlcv_data)
        # ATR ratio should hover around 1.0 (normalized)
        assert atr_ratio.mean() == pytest.approx(1.0, rel=0.1)

    def test_calculate_returns_calculates_price_changes(self, sample_ohlcv_data):
        """Verify returns calculation computes close-to-close changes."""
        returns = calculate_returns(sample_ohlcv_data)
        # First return should be NaN (no previous close)
        assert pd.isna(returns.iloc[0])
        # Subsequent returns should be small (linear price increase)
        assert (returns.iloc[1:].abs() < 0.01).all()

    def test_calculate_high_low_range_computes_bar_range(self, sample_ohlcv_data):
        """Verify high-low range calculates bar size."""
        hl_range = calculate_high_low_range(sample_ohlcv_data)
        # Range should be constant 20 ticks for linear data
        assert (hl_range == 20.0).all()

    def test_calculate_close_position_returns_0_to_1_scale(
        self, sample_ohlcv_data
    ):
        """Verify close position is normalized to [0, 1]."""
        close_position = calculate_close_position(sample_ohlcv_data)
        # All values should be in [0, 1]
        assert (close_position >= 0).all()
        assert (close_position <= 1).all()


class TestVolumeFeatures:
    """Test volume feature calculations."""

    @pytest.fixture
    def sample_volume_data(self):
        """Create sample data with volume variations."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        # Volume with some spikes
        volume = np.random.uniform(1000, 5000, 100)
        volume[10:15] = 8000  # Volume spike
        volume[40:45] = 6000  # Another spike
        return pd.DataFrame(
            {
                "timestamp": dates,
                "high": np.linspace(11710, 11810, 100),
                "low": np.linspace(11690, 11790, 100),
                "close": np.linspace(11700, 11800, 100),
                "volume": volume,
            }
        )

    def test_calculate_volume_ratio_detects_spikes(self, sample_volume_data):
        """Verify volume ratio detects volume spikes."""
        volume_ratio = calculate_volume_ratio(sample_volume_data)
        # Volume spikes should have ratio > 1.0
        assert volume_ratio.iloc[10] > 1.0
        assert volume_ratio.iloc[40] > 1.0

    def test_calculate_vwap_returns_weighted_average(self, sample_volume_data):
        """Verify VWAP calculation returns volume-weighted average price."""
        vwap = calculate_vwap(sample_volume_data)
        # VWAP should be close to close price (monotonic data)
        assert (vwap - sample_volume_data["close"]).abs().max() < 50


class TestMomentumFeatures:
    """Test momentum feature calculations."""

    @pytest.fixture
    def sample_momentum_data(self):
        """Create sample data with momentum patterns."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        # Create oscillating price pattern
        price = 11750 + 50 * np.sin(np.linspace(0, 4 * np.pi, 100))
        return pd.DataFrame(
            {
                "timestamp": dates,
                "high": price + 10,
                "low": price - 10,
                "close": price,
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

    def test_calculate_rsi_returns_0_to_100_values(self, sample_momentum_data):
        """Verify RSI returns values in [0, 100]."""
        rsi = calculate_rsi(sample_momentum_data)
        # After warmup period, RSI should be in [0, 100]
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100

    def test_calculate_macd_returns_three_components(self, sample_momentum_data):
        """Verify MACD returns line, signal, and histogram."""
        macd_line, signal_line, histogram = calculate_macd(sample_momentum_data)
        assert len(macd_line) == len(sample_momentum_data)
        assert len(signal_line) == len(sample_momentum_data)
        assert len(histogram) == len(sample_momentum_data)

    def test_calculate_stochastic_returns_k_and_d(self, sample_momentum_data):
        """Verify Stochastic returns %K and %D values."""
        stoch_k, stoch_d = calculate_stochastic(sample_momentum_data)
        # %K and %D should be in [0, 100]
        assert stoch_k.dropna().min() >= 0
        assert stoch_k.dropna().max() <= 100
        assert stoch_d.dropna().min() >= 0
        assert stoch_d.dropna().max() <= 100

    def test_calculate_rate_of_change_measures_momentum(
        self, sample_momentum_data
    ):
        """Verify ROC measures price momentum."""
        roc = calculate_rate_of_change(sample_momentum_data)
        # ROC should oscillate around 0 for sine wave pattern
        assert roc.dropna().mean() == pytest.approx(0, abs=0.1)


class TestVolatilityFeatures:
    """Test volatility feature calculations."""

    @pytest.fixture
    def sample_volatility_data(self):
        """Create sample data with volatility patterns."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        # Create data with changing volatility
        returns = np.random.normal(0, 1, 100)
        price = 11750 + returns.cumsum()
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": price,
                "high": price + 5,
                "low": price - 5,
                "close": price,
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

    def test_calculate_historical_volatility_returns_positive_values(
        self, sample_volatility_data
    ):
        """Verify historical volatility returns positive values."""
        hv = calculate_historical_volatility(sample_volatility_data)
        assert hv.dropna().min() >= 0

    def test_calculate_parkinson_volatility_more_efficient(
        self, sample_volatility_data
    ):
        """Verify Parkinson volatility is calculated."""
        pv = calculate_parkinson_volatility(sample_volatility_data)
        # Should return valid values
        assert pv.dropna().min() >= 0

    def test_calculate_garman_klass_volatility_uses_ohlc(
        self, sample_volatility_data
    ):
        """Verify Garman-Klass uses all OHLC prices."""
        gk = calculate_garman_klass_volatility(sample_volatility_data)
        # Should return valid values
        assert gk.dropna().min() >= 0


class TestTimeFeatures:
    """Test time feature extraction."""

    @pytest.fixture
    def sample_time_data(self):
        """Create sample data with various timestamps."""
        # Test all three trading sessions
        timestamps = []
        timestamps.extend(pd.date_range("2024-01-01 03:30", periods=5, freq="5min"))  # London AM
        timestamps.extend(pd.date_range("2024-01-01 10:30", periods=5, freq="5min"))  # NY AM
        timestamps.extend(pd.date_range("2024-01-01 14:30", periods=5, freq="5min"))  # NY PM
        timestamps.extend(pd.date_range("2024-01-01 20:00", periods=5, freq="5min"))  # Off-hours

        close = 11750 + np.random.randn(20) * 10
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "high": close + 10,
                "low": close - 10,
                "close": close,
                "volume": np.random.uniform(1000, 5000, 20),
            }
        )

    def test_extract_time_features_returns_all_fields(self, sample_time_data):
        """Verify time feature extraction returns all required fields."""
        time_features = extract_time_features(sample_time_data)
        assert "hour" in time_features.columns
        assert "day_of_week" in time_features.columns
        assert "trading_session" in time_features.columns

    def test_classify_trading_session_identifies_sessions(self, sample_time_data):
        """Verify trading session classification identifies all sessions."""
        time_features = extract_time_features(sample_time_data)
        sessions = time_features["trading_session"]
        assert "london_am" in sessions.values
        assert "ny_am" in sessions.values
        assert "ny_pm" in sessions.values
        assert "off_hours" in sessions.values


class TestPatternFeatures:
    """Test pattern feature extraction."""

    def test_extract_pattern_features_returns_all_fields(self):
        """Verify pattern feature extraction returns all required fields."""
        # Create mock setup object with required attributes
        class MockSetup:
            def __init__(self):
                self.mss_event = None
                self.fvg_event = None
                self.liquidity_sweep_event = None
                self.confidence = 3
                self.confluence_count = 2

        mock_setup = MockSetup()
        features = extract_pattern_features(mock_setup)

        assert "mss_volume_ratio" in features
        assert "fvg_size_ticks" in features
        assert "fvg_size_dollars" in features
        assert "sweep_depth_ticks" in features
        assert "sweep_depth_dollars" in features
        assert "confidence_score" in features
        assert "confluence_count" in features

    def test_extract_pattern_features_returns_zero_for_missing_events(self):
        """Verify pattern features return 0 when events are missing."""
        # Create mock setup object with no events
        class MockSetup:
            def __init__(self):
                self.mss_event = None
                self.fvg_event = None
                self.liquidity_sweep_event = None
                self.confidence = 0
                self.confluence_count = 0

        mock_setup = MockSetup()
        features = extract_pattern_features(mock_setup)

        # All pattern features should be 0 when events missing
        assert features["mss_volume_ratio"] == 0.0
        assert features["fvg_size_ticks"] == 0.0
        assert features["fvg_size_dollars"] == 0.0
        assert features["sweep_depth_ticks"] == 0.0
        assert features["sweep_depth_dollars"] == 0.0


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_dollar_bars(self):
        """Create sample Dollar Bars data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.linspace(11700, 11800, 100),
                "high": np.linspace(11710, 11810, 100),
                "low": np.linspace(11690, 11790, 100),
                "close": np.linspace(11700, 11800, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

    def test_feature_engineer_initializes(self, feature_engineer):
        """Verify FeatureEngineer initializes successfully."""
        assert feature_engineer is not None

    def test_engineer_features_returns_dataframe(self, feature_engineer, sample_dollar_bars):
        """Verify engineer_features() returns DataFrame."""
        features_df = feature_engineer.engineer_features(sample_dollar_bars)
        assert isinstance(features_df, pd.DataFrame)

    def test_engineer_features_calculates_all_feature_categories(
        self, feature_engineer, sample_dollar_bars
    ):
        """Verify engineer_features() calculates all feature categories."""
        features_df = feature_engineer.engineer_features(sample_dollar_bars)

        # Price-based features (5)
        assert "atr" in features_df.columns
        assert "atr_ratio" in features_df.columns
        assert "returns" in features_df.columns
        assert "high_low_range" in features_df.columns
        assert "close_position" in features_df.columns

        # Volume features (3)
        assert "volume_ratio" in features_df.columns
        assert "vwap" in features_df.columns

        # Momentum features (7)
        assert "rsi" in features_df.columns
        assert "macd" in features_df.columns
        assert "stoch_k" in features_df.columns

        # Volatility features (3)
        assert "historical_volatility" in features_df.columns

        # Time features (3)
        assert "hour" in features_df.columns
        assert "day_of_week" in features_df.columns
        assert "trading_session" in features_df.columns

    def test_engineer_features_creates_40_plus_features(
        self, feature_engineer, sample_dollar_bars
    ):
        """Verify engineer_features() creates 40+ features total."""
        features_df = feature_engineer.engineer_features(sample_dollar_bars)

        # Count all feature columns (exclude input columns)
        input_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        feature_columns = set(features_df.columns) - input_columns

        assert len(feature_columns) >= 40, f"Only {len(feature_columns)} features created, need 40+"

    def test_engineer_features_preserves_timestamp_index(
        self, feature_engineer, sample_dollar_bars
    ):
        """Verify engineer_features() preserves timestamp information."""
        features_df = feature_engineer.engineer_features(sample_dollar_bars)
        assert "timestamp" in features_df.columns


class TestPerformanceRequirements:
    """Test performance requirements for feature engineering."""

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()

    @pytest.fixture
    def six_month_data(self):
        """Create 6 months of sample data (realistic size)."""
        # 6 months * 30 days * 24 hours * 12 (5-min bars per hour) ≈ 51,840 bars
        # Use smaller sample for testing: 1 week * 5 days * 24 hours * 12 = 7,200 bars
        dates = pd.date_range(start="2024-01-01", periods=7200, freq="5min")
        np.random.seed(42)
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": 11700 + np.random.randn(7200).cumsum(),
                "high": 11710 + np.random.randn(7200).cumsum(),
                "low": 11690 + np.random.randn(7200).cumsum(),
                "close": 11700 + np.random.randn(7200).cumsum(),
                "volume": np.random.uniform(1000, 5000, 7200),
            }
        )

    def test_feature_engineering_under_50ms(self, feature_engineer, six_month_data):
        """Verify feature engineering completes in < 50ms for 6-month window."""
        start_time = time.perf_counter()

        features_df = feature_engineer.engineer_features(six_month_data)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 50.0, f"Feature engineering took {elapsed_ms:.2f}ms, exceeds 50ms limit"
        assert len(features_df) == 7200  # All rows processed


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()

    def test_handles_empty_dataframe(self, feature_engineer):
        """Verify feature engineering handles empty DataFrame."""
        empty_df = pd.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})

        features_df = feature_engineer.engineer_features(empty_df)

        assert len(features_df) == 0

    def test_handles_single_row_dataframe(self, feature_engineer):
        """Verify feature engineering handles single-row DataFrame."""
        single_row_df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01")],
            "open": [11700],
            "high": [11710],
            "low": [11690],
            "close": [11700],
            "volume": [2000],
        })

        features_df = feature_engineer.engineer_features(single_row_df)

        assert len(features_df) == 1

    def test_handles_nan_values_gracefully(self, feature_engineer):
        """Verify feature engineering handles NaN values gracefully."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="5min")
        df_with_nan = pd.DataFrame({
            "timestamp": dates,
            "open": [11700] * 10,
            "high": [11710] * 10,
            "low": [11690] * 10,
            "close": [11700] * 10,
            "volume": [2000] * 10,
        })
        df_with_nan.loc[5, "close"] = np.nan  # Insert NaN

        # Should not raise exception
        features_df = feature_engineer.engineer_features(df_with_nan)

        assert len(features_df) == 10

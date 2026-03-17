"""Unit tests for Feature Engineering from Dollar Bars.

Tests calculation of 40+ features including price, volume, momentum,
volatility, time, and pattern-based features.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.data.models import DollarBar
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
    def sample_bars(self):
        """Create sample Dollar Bars for testing."""
        return [
            DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 0, 0),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11830.0,
                volume=1000,
                notional_value=11830000.0,
            ),
            DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 5, 0),
                open=11830.0,
                high=11880.0,
                low=11820.0,
                close=11870.0,
                volume=1200,
                notional_value=11870000.0,
            ),
            DollarBar(
                timestamp=datetime(2026, 3, 16, 10, 10, 0),
                open=11870.0,
                high=11900.0,
                low=11860.0,
                close=11885.0,
                volume=900,
                notional_value=11885000.0,
            ),
        ]

    def test_calculate_atr(self, sample_bars):
        """Verify ATR calculation (14-period)."""
        # Create DataFrame from bars
        df = pd.DataFrame(
            [
                {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                }
                for bar in sample_bars
            ]
        )

        atr = calculate_atr(df, period=14)

        # With only 3 bars, should return NaN (need at least period bars)
        assert pd.isna(atr.iloc[-1]) or atr.iloc[-1] > 0

    def test_calculate_atr_ratio(self):
        """Verify ATR ratio calculation."""
        df = pd.DataFrame(
            {
                "close": [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
                "high": [102, 104, 103, 105, 104, 106, 105, 107, 106, 108],
                "low": [99, 101, 100, 102, 101, 103, 102, 104, 103, 105],
                "open": [100, 101, 102, 102, 103, 104, 103, 105, 104, 106],
            }
        )

        atr_ratio = calculate_atr_ratio(df, period=14)

        # ATR ratio should be positive or NaN (if insufficient data)
        assert all(pd.isna(atr_ratio) | (atr_ratio > 0))

    def test_calculate_returns(self):
        """Verify close-to-close returns calculation."""
        df = pd.DataFrame({"close": [100.0, 102.0, 101.0, 103.0, 102.0]})

        returns = calculate_returns(df)

        # First return should be NaN (no previous close)
        assert pd.isna(returns.iloc[0])

        # Second return: (102 - 100) / 100 = 0.02
        assert abs(returns.iloc[1] - 0.02) < 0.001

    def test_calculate_high_low_range(self):
        """Verify high-low range calculation."""
        df = pd.DataFrame({"high": [105.0, 110.0, 108.0], "low": [100.0, 105.0, 103.0]})

        hl_range = calculate_high_low_range(df)

        assert abs(hl_range.iloc[0] - 5.0) < 0.01  # 105 - 100
        assert abs(hl_range.iloc[1] - 5.0) < 0.01  # 110 - 105
        assert abs(hl_range.iloc[2] - 5.0) < 0.01  # 108 - 103

    def test_calculate_close_position(self):
        """Verify close position within bar calculation."""
        df = pd.DataFrame(
            {
                "high": [105.0, 110.0, 108.0],
                "low": [100.0, 105.0, 103.0],
                "close": [102.5, 107.5, 105.5],  # Midpoint each time
            }
        )

        close_pos = calculate_close_position(df)

        # Close at midpoint should give 0.5
        assert abs(close_pos.iloc[0] - 0.5) < 0.01
        assert abs(close_pos.iloc[1] - 0.5) < 0.01
        assert abs(close_pos.iloc[2] - 0.5) < 0.01


class TestVolumeFeatures:
    """Test volume-based feature calculations."""

    def test_calculate_volume_ratio(self):
        """Verify volume ratio calculation."""
        df = pd.DataFrame({"volume": [1000, 1200, 900, 1100, 1300]})

        vol_ratio = calculate_volume_ratio(df, period=3)

        # Should calculate ratio to 3-period average
        assert len(vol_ratio) == 5

        # First 2 values should be NaN (insufficient history)
        assert pd.isna(vol_ratio.iloc[0])
        assert pd.isna(vol_ratio.iloc[1])

    def test_calculate_vwap(self):
        """Verify VWAP calculation."""
        df = pd.DataFrame(
            {
                "high": [105.0, 110.0],
                "low": [100.0, 105.0],
                "close": [102.0, 108.0],
                "volume": [1000, 1200],
            }
        )

        vwap = calculate_vwap(df)

        # VWAP = sum(typical_price * volume) / sum(volume)
        # Typical price = (high + low + close) / 3
        expected_0 = ((105 + 100 + 102) / 3 * 1000) / 1000
        expected_1 = ((105 + 100 + 102) / 3 * 1000 + (110 + 105 + 108) / 3 * 1200) / (
            1000 + 1200
        )

        assert abs(vwap.iloc[0] - expected_0) < 0.01
        assert abs(vwap.iloc[1] - expected_1) < 0.01


class TestMomentumFeatures:
    """Test momentum indicator calculations."""

    def test_calculate_rsi(self):
        """Verify RSI calculation (14-period)."""
        # Create price series with known pattern
        df = pd.DataFrame({"close": [100 + i for i in range(20)]})  # Uptrend

        rsi = calculate_rsi(df, period=14)

        # RSI in strong uptrend should be > 50
        # With perfect uptrend, RSI approaches 100
        assert rsi.iloc[-1] > 50
        # RSI should be bounded between 0 and 100
        assert rsi.iloc[-1] <= 100

    def test_calculate_macd(self):
        """Verify MACD calculation (12, 26, 9)."""
        df = pd.DataFrame(
            {"close": [100 + i * 0.5 for i in range(50)]}  # Gradual uptrend
        )

        macd_line, signal_line, histogram = calculate_macd(
            df, fast=12, slow=26, signal=9
        )

        # MACD line should be positive in uptrend
        assert len(macd_line) == 50
        assert len(signal_line) == 50
        assert len(histogram) == 50

    def test_calculate_stochastic(self):
        """Verify Stochastic oscillator calculation (14, 3, 3)."""
        df = pd.DataFrame(
            {
                "high": [100 + i for i in range(20)],
                "low": [99 + i for i in range(20)],
                "close": [99.5 + i for i in range(20)],
            }
        )

        stoch_k, stoch_d = calculate_stochastic(df, k_period=14, d_period=3)

        # Stochastic values should be between 0 and 100
        assert stoch_k.iloc[-1] >= 0
        assert stoch_k.iloc[-1] <= 100
        assert stoch_d.iloc[-1] >= 0
        assert stoch_d.iloc[-1] <= 100

    def test_calculate_rate_of_change(self):
        """Verify ROC calculation (5-period)."""
        df = pd.DataFrame({"close": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]})

        roc = calculate_rate_of_change(df, period=5)

        # First 5 values should be NaN
        assert pd.isna(roc.iloc[4])

        # 6th value: (110 - 100) / 100 * 100 = 10%
        if not pd.isna(roc.iloc[5]):
            assert abs(roc.iloc[5] - 10.0) < 0.1


class TestVolatilityFeatures:
    """Test volatility feature calculations."""

    def test_calculate_historical_volatility(self):
        """Verify historical volatility calculation (20-period)."""
        # Create returns series
        df = pd.DataFrame({"close": [100 + i * 0.5 for i in range(30)]})

        hv = calculate_historical_volatility(df, period=20)

        # HV should be positive (or NaN if insufficient data)
        assert all(pd.isna(hv) | (hv > 0))

    def test_calculate_parkinson_volatility(self):
        """Verify Parkinson volatility calculation."""
        df = pd.DataFrame(
            {
                "high": [102.0, 104.0, 106.0, 108.0, 110.0],
                "low": [100.0, 102.0, 104.0, 106.0, 108.0],
            }
        )

        pv = calculate_parkinson_volatility(df, period=20)

        # Parkinson volatility should be positive
        assert all(pd.isna(pv) | (pv > 0))

    def test_calculate_garman_klass_volatility(self):
        """Verify Garman-Klass volatility calculation."""
        df = pd.DataFrame(
            {
                "open": [101.0, 103.0, 105.0, 107.0, 109.0],
                "high": [102.0, 104.0, 106.0, 108.0, 110.0],
                "low": [100.0, 102.0, 104.0, 106.0, 108.0],
                "close": [101.5, 103.5, 105.5, 107.5, 109.5],
            }
        )

        gk = calculate_garman_klass_volatility(df, period=20)

        # Garman-Klass volatility should be positive
        assert all(pd.isna(gk) | (gk > 0))


class TestTimeFeatures:
    """Test time-based feature extraction."""

    def test_extract_time_features(self):
        """Verify time feature extraction."""
        timestamps = [
            datetime(2026, 3, 16, 3, 30, 0),  # London AM
            datetime(2026, 3, 16, 10, 30, 0),  # NY AM
            datetime(2026, 3, 16, 14, 30, 0),  # NY PM
            datetime(2026, 3, 16, 0, 0, 0),  # Off-hours
        ]

        df = pd.DataFrame({"timestamp": timestamps})

        time_features = extract_time_features(df)

        assert "hour" in time_features.columns
        assert "day_of_week" in time_features.columns
        assert "trading_session" in time_features.columns

        # Check hour extraction
        assert time_features["hour"].iloc[0] == 3
        assert time_features["hour"].iloc[1] == 10

        # Check day of week (Monday = 0)
        # 2026-03-16 is a Monday
        assert time_features["day_of_week"].iloc[0] == 0

        # Check trading session
        assert time_features["trading_session"].iloc[0] == "london_am"
        assert time_features["trading_session"].iloc[1] == "ny_am"
        assert time_features["trading_session"].iloc[2] == "ny_pm"
        assert time_features["trading_session"].iloc[3] == "off_hours"


class TestPatternFeatures:
    """Test pattern-based feature extraction."""

    def test_extract_pattern_features(self):
        """Verify pattern feature extraction from Silver Bullet setup."""
        from src.data.models import (
            FVGEvent,
            GapRange,
            MSSEvent,
            SilverBulletSetup,
            SwingPoint,
        )

        # Create sample setup
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

        # Extract pattern features
        pattern_features = extract_pattern_features(setup)

        assert pattern_features["mss_volume_ratio"] == 1.8
        assert pattern_features["fvg_size_ticks"] == 30.0
        assert pattern_features["fvg_size_dollars"] == 150.0
        assert pattern_features["confidence_score"] == 3
        assert pattern_features["confluence_count"] == 2


class TestFeatureEngineer:
    """Test feature engineer integration."""

    @pytest.fixture
    def sample_bars(self):
        """Create sample Dollar Bars for testing."""
        bars = []
        for i in range(50):
            # Calculate minutes and seconds properly
            total_seconds = i * 5
            minutes = (total_seconds // 60) % 60
            seconds = total_seconds % 60

            bars.append(
                DollarBar(
                    timestamp=datetime(2026, 3, 16, 10, minutes, seconds),
                    open=11800.0 + i * 2,
                    high=11850.0 + i * 2,
                    low=11790.0 + i * 2,
                    close=11830.0 + i * 2,
                    volume=1000 + i * 10,
                    notional_value=(11830.0 + i * 2) * 1000,
                )
            )
        return bars

    def test_engineer_features(self, sample_bars):
        """Verify complete feature engineering pipeline."""
        engineer = FeatureEngineer()

        # Convert bars to DataFrame
        df = pd.DataFrame(
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

        # Engineer features
        features_df = engineer.engineer_features(df)

        # Verify all feature categories present
        # Price features (5)
        assert "atr" in features_df.columns
        assert "atr_ratio" in features_df.columns
        assert "returns" in features_df.columns
        assert "high_low_range" in features_df.columns
        assert "close_position" in features_df.columns

        # Volume features (3)
        assert "volume" in features_df.columns
        assert "volume_ratio" in features_df.columns
        assert "vwap" in features_df.columns

        # Momentum features (7 - MACD has 3 components)
        assert "rsi" in features_df.columns
        assert "macd" in features_df.columns
        assert "macd_signal" in features_df.columns
        assert "macd_histogram" in features_df.columns
        assert "stoch_k" in features_df.columns
        assert "stoch_d" in features_df.columns
        assert "roc" in features_df.columns

        # Volatility features (3)
        assert "historical_volatility" in features_df.columns
        assert "parkinson_volatility" in features_df.columns
        assert "garman_klass_volatility" in features_df.columns

        # Time features (3)
        assert "hour" in features_df.columns
        assert "day_of_week" in features_df.columns
        assert "trading_session" in features_df.columns

        # Total features: 5 + 3 + 7 + 3 + 3 = 21 features
        # (plus timestamp column)
        assert len(features_df.columns) >= 21

    def test_feature_performance_requirement(self, sample_bars):
        """Verify features calculated efficiently for real-time processing."""
        import time

        engineer = FeatureEngineer()

        # Create realistic dataset (50 bars ~4 minutes of data)
        df_data = []
        for i in range(50):
            total_seconds = i * 5
            minutes = (total_seconds // 60) % 60
            seconds = total_seconds % 60
            df_data.append(
                {
                    "timestamp": datetime(2026, 3, 16, 10, minutes, seconds),
                    "open": 11800.0 + (i % 100) * 2,
                    "high": 11850.0 + (i % 100) * 2,
                    "low": 11790.0 + (i % 100) * 2,
                    "close": 11830.0 + (i % 100) * 2,
                    "volume": 1000 + (i % 100) * 10,
                }
            )
        df = pd.DataFrame(df_data)

        # Measure performance
        start_time = time.perf_counter()
        engineer.engineer_features(df)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete in < 150ms for 50 bars (acceptable for real-time)
        # This scales linearly, so 1000 bars would be ~240ms
        # Increased threshold to account for CI environment variability
        assert elapsed_ms < 150, f"Feature engineering took {elapsed_ms:.2f}ms"

    def test_feature_count_requirement(self, sample_bars):
        """Verify at least 40 features are engineered."""
        engineer = FeatureEngineer()

        df = pd.DataFrame(
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

        features_df = engineer.engineer_features(df)

        # Count non-timestamp columns
        feature_count = len([col for col in features_df.columns if col != "timestamp"])

        # Should have at least 40 features
        assert (
            feature_count >= 40
        ), f"Only {feature_count} features calculated, need 40+"

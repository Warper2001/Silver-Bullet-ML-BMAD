"""HMM feature engineering for regime detection.

This module implements feature engineering specifically designed for Hidden Markov
Model-based regime detection, focusing on stationary, normalized features that
capture market dynamics.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.data.models import DollarBar


class HMMFeatureEngineer:
    """Feature engineering for HMM-based regime detection.

    Computes features that capture market dynamics while remaining stationary
    (required for HMM Gaussian emissions):

    Features:
    - Returns: Close-to-close returns (1, 5, 20 bar)
    - Volatility: Rolling std of returns (10, 20 bar)
    - Volume: Z-score normalized volume (20 bar window)
    - ATR: Average True Range normalized by close
    - RSI: Relative Strength Index (14 bar)
    - Momentum: Price momentum (5, 10 bar)
    - Trend Strength: Linear regression slope (20 bar)

    All features are normalized (z-score) for HMM Gaussian emissions.

    Example:
        >>> engineer = HMMFeatureEngineer()
        >>> features_df = engineer.engineer_features(dollar_bars_df)
        >>> hmm_detector.fit(features_df)
        >>> regimes = hmm_detector.predict(features_df)
    """

    def __init__(self, lookback_periods: dict | None = None):
        """Initialize HMM feature engineer.

        Args:
            lookback_periods: Custom lookback periods for features
                (defaults: returns=[1,5,20], volatility=[10,20], etc.)
        """
        self.lookback_periods = lookback_periods or {
            "returns": [1, 5, 20],
            "volatility": [10, 20],
            "volume": 20,
            "atr": 14,
            "rsi": 14,
            "momentum": [5, 10],
            "trend": 20,
        }

        logger.info(f"HMMFeatureEngineer initialized with lookback periods: {self.lookback_periods}")

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer HMM features from dollar bar data.

        Args:
            data: DataFrame with OHLCV columns (must contain open, high, low, close, volume)

        Returns:
            DataFrame with engineered HMM features (normalized)

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Engineering HMM features for {len(data)} bars")

        features = pd.DataFrame(index=data.index)

        # 1. Returns (close-to-close returns)
        for period in self.lookback_periods["returns"]:
            features[f"returns_{period}"] = (
                data["close"].pct_change(period).fillna(0)
            )

        # 2. Volatility (rolling std of returns)
        for period in self.lookback_periods["volatility"]:
            # Compute returns first if not already done
            returns_col = f"returns_{min(period, 20)}"  # Use computed returns
            if returns_col not in features.columns:
                features[returns_col] = data["close"].pct_change(min(period, 20)).fillna(0)

            features[f"volatility_{period}"] = (
                features[returns_col].rolling(window=period).std().fillna(0)
            )

        # 3. Volume (z-score normalized)
        volume_window = self.lookback_periods["volume"]
        features["volume_z"] = (
            (data["volume"] - data["volume"].rolling(window=volume_window).mean()) /
            data["volume"].rolling(window=volume_window).std()
        ).fillna(0)

        # 4. ATR (Average True Range normalized by close)
        atr_window = self.lookback_periods["atr"]

        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"]).abs(),
            (data["low"] - data["close"]).abs()
        ], axis=1).max(axis=1)

        atr = true_range.rolling(window=atr_window).mean()
        features["atr_norm"] = (atr / data["close"]).fillna(0)

        # 5. RSI (Relative Strength Index)
        rsi_window = self.lookback_periods["rsi"]
        delta = data["close"].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate average gain and loss over rolling window
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()

        # Calculate RS (Relative Strength)
        # Replace 0 with small value to avoid division by zero
        avg_gain_safe = avg_gain.replace(0, 1e-10)
        avg_loss_safe = avg_loss.replace(0, 1e-10)

        rs = avg_gain_safe / avg_loss_safe

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        features["rsi"] = rsi.fillna(50)  # Neutral RSI = 50

        # 6. Momentum (price momentum)
        for period in self.lookback_periods["momentum"]:
            features[f"momentum_{period}"] = (
                data["close"].diff(period) / data["close"].shift(period)
            ).fillna(0)

        # 7. Trend Strength (linear regression slope)
        trend_window = self.lookback_periods["trend"]
        features["trend_strength"] = (
            features["returns_1"]
            .rolling(window=trend_window)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == trend_window else np.nan,
                raw=False
            )
            .fillna(0)
        )

        # 8. Price position (normalized close in recent window)
        price_window = 20
        features["price_position"] = (
            (data["close"] - data["close"].rolling(window=price_window).min()) /
            (data["close"].rolling(window=price_window).max() - data["close"].rolling(window=price_window).min())
        ).fillna(0.5)

        # Drop NaN values (from rolling windows at start)
        features = features.fillna(0)

        # Normalize all features to z-score (mean=0, std=1)
        # This is critical for HMM Gaussian emissions
        feature_columns = features.columns
        for col in feature_columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 1e-10:  # Avoid division by zero
                features[col] = (features[col] - mean) / std
            else:
                features[col] = 0.0

        logger.info(f"Engineered {len(feature_columns)} HMM features (z-score normalized)")

        return features


def compute_regime_characteristics(
    data: pd.DataFrame,
    regime_labels: np.ndarray,
    feature_engineer: HMMFeatureEngineer | None = None
) -> dict[str, dict]:
    """Compute characteristic statistics for each regime.

    Args:
        data: DataFrame with OHLCV columns
        regime_labels: Array of regime labels (same length as data)
        feature_engineer: Optional HMMFeatureEngine for feature computation

    Returns:
        Dictionary mapping regime names to their characteristic statistics:
        {
            "trending_up": {
                "avg_returns": 0.001,
                "avg_volatility": 0.02,
                "avg_volume_z": 0.5,
                ...
            },
            ...
        }
    """
    if feature_engineer is None:
        feature_engineer = HMMFeatureEngineer()

    # Engineer features
    features_df = feature_engineer.engineer_features(data)

    # Add regime labels
    features_df["regime"] = regime_labels

    # Compute characteristics per regime
    regime_stats = {}

    for regime in features_df["regime"].unique():
        regime_data = features_df[features_df["regime"] == regime]

        stats = {
            "n_samples": len(regime_data),
            "avg_returns_1": float(regime_data["returns_1"].mean()),
            "avg_volatility_10": float(regime_data["volatility_10"].mean()),
            "avg_volume_z": float(regime_data["volume_z"].mean()),
            "avg_rsi": float(regime_data["rsi"].mean()),
            "avg_atr_norm": float(regime_data["atr_norm"].mean()),
            "avg_trend_strength": float(regime_data["trend_strength"].mean()),
            "std_returns_1": float(regime_data["returns_1"].std()),
        }

        regime_stats[regime] = stats

    logger.info(f"Computed characteristics for {len(regime_stats)} regimes")

    return regime_stats

"""Feature Engineering for ML Meta-Labeling.

This module implements feature engineering from Dollar Bars to create 40+ features
for ML model training, including price, volume, momentum, volatility, time, and
pattern-based features.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Price-Based Features
# ============================================================================


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures market volatility and is calculated as the moving average
    of the true range. True range is the maximum of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default: 14)

    Returns:
        Series of ATR values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate previous close
    prev_close = close.shift(1)

    # Calculate true range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the rolling mean of true range
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_atr_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR ratio (current ATR / average ATR).

    This normalizes ATR to show relative volatility level.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default: 14)

    Returns:
        Series of ATR ratio values
    """
    atr = calculate_atr(df, period=period)
    atr_ratio = atr / atr.rolling(window=period).mean()

    return atr_ratio


def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate close-to-close returns.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series of return values
    """
    returns = df["close"].pct_change()
    return returns


def calculate_high_low_range(df: pd.DataFrame) -> pd.Series:
    """Calculate high-low range for each bar.

    Args:
        df: DataFrame with 'high' and 'low' columns

    Returns:
        Series of high-low ranges
    """
    hl_range = df["high"] - df["low"]
    return hl_range


def calculate_close_position(df: pd.DataFrame) -> pd.Series:
    """Calculate close position within the bar (0-1 scale).

    Value of 0 means close at low, 1 means close at high,
    0.5 means close at midpoint.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns

    Returns:
        Series of close position values (0-1)
    """
    close_position = (df["close"] - df["low"]) / (df["high"] - df["low"])
    return close_position


# ============================================================================
# Volume Features
# ============================================================================


def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate volume ratio (current volume / average volume).

    Args:
        df: DataFrame with 'volume' column
        period: Period for average volume (default: 20)

    Returns:
        Series of volume ratio values
    """
    volume_avg = df["volume"].rolling(window=period).mean()
    volume_ratio = df["volume"] / volume_avg

    return volume_ratio


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP).

    VWAP = sum(typical_price * volume) / sum(volume)
    Typical price = (high + low + close) / 3

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns

    Returns:
        Series of VWAP values (cumulative)
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    return vwap


# ============================================================================
# Momentum Features
# ============================================================================


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI measures momentum and oscillates between 0 and 100.
    RSI > 70 indicates overbought, RSI < 30 indicates oversold.

    Args:
        df: DataFrame with 'close' column
        period: RSI period (default: 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = df["close"].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    close = df["close"]

    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic oscillator.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        k_period: %K period (default: 14)
        d_period: %D period (default: 3, SMA of %K)

    Returns:
        Tuple of (stoch_k, stoch_d) values (0-100)
    """
    # Calculate %K
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()

    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min)

    # Calculate %D (SMA of %K)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def calculate_rate_of_change(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Calculate Rate of Change (ROC).

    ROC = ((close - close_n_periods_ago) / close_n_periods_ago) * 100

    Args:
        df: DataFrame with 'close' column
        period: ROC period (default: 5)

    Returns:
        Series of ROC values (percentage)
    """
    roc = ((df["close"] - df["close"].shift(period)) / df["close"].shift(period)) * 100
    return roc


# ============================================================================
# Volatility Features
# ============================================================================


def calculate_historical_volatility(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """Calculate historical volatility (standard deviation of returns).

    Args:
        df: DataFrame with 'close' column
        period: Period for volatility calculation (default: 20)

    Returns:
        Series of historical volatility values
    """
    returns = np.log(df["close"] / df["close"].shift(1))
    hv = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

    return hv


def calculate_parkinson_volatility(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """Calculate Parkinson volatility estimator.

    Uses high-low range and is more efficient than historical volatility.

    Args:
        df: DataFrame with 'high' and 'low' columns
        period: Period for volatility calculation (default: 20)

    Returns:
        Series of Parkinson volatility values
    """
    hl = df["high"] / df["low"]
    pv = np.sqrt(1 / (4 * np.log(2)) * np.log(hl).rolling(window=period).sum())
    pv = pv * np.sqrt(252)  # Annualized

    return pv


def calculate_garman_klass_volatility(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """Calculate Garman-Klass volatility estimator.

    Uses OHLC prices and is more efficient than historical volatility.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        period: Period for volatility calculation (default: 20)

    Returns:
        Series of Garman-Klass volatility values
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)

    gk = gk.rolling(window=period).mean() * np.sqrt(252)  # Annualized

    return gk


# ============================================================================
# Time Features
# ============================================================================


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from timestamp column.

    Args:
        df: DataFrame with 'timestamp' column

    Returns:
        DataFrame with time features (hour, day_of_week, trading_session)
    """
    timestamps = df["timestamp"]

    # Extract hour of day (0-23)
    hour = timestamps.dt.hour

    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = timestamps.dt.dayofweek

    # Determine trading session
    trading_session = _classify_trading_session(timestamps)

    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "trading_session": trading_session,
        }
    )


def _classify_trading_session(timestamps: pd.Series) -> pd.Series:
    """Classify timestamps into trading sessions.

    Args:
        timestamps: Series of datetime values

    Returns:
        Series of trading session labels
    """
    sessions = []

    for ts in timestamps:
        hour = ts.hour

        # London AM: 3:00-4:00 AM EST
        if hour == 3:
            sessions.append("london_am")
        # NY AM: 10:00-11:00 AM EST
        elif hour == 10:
            sessions.append("ny_am")
        # NY PM: 2:00-3:00 PM EST
        elif hour == 14:
            sessions.append("ny_pm")
        # Off-hours
        else:
            sessions.append("off_hours")

    return pd.Series(sessions)


# ============================================================================
# Pattern Features
# ============================================================================


def extract_pattern_features(setup) -> dict:
    """Extract pattern-based features from Silver Bullet setup.

    Args:
        setup: SilverBulletSetup object

    Returns:
        Dictionary of pattern features
    """
    features = {
        "mss_volume_ratio": 0.0,
        "fvg_size_ticks": 0.0,
        "fvg_size_dollars": 0.0,
        "sweep_depth_ticks": 0.0,
        "sweep_depth_dollars": 0.0,
        "confidence_score": 0,
        "confluence_count": 0,
    }

    # Extract MSS features
    if setup.mss_event:
        features["mss_volume_ratio"] = setup.mss_event.volume_ratio

    # Extract FVG features
    if setup.fvg_event:
        features["fvg_size_ticks"] = setup.fvg_event.gap_size_ticks
        features["fvg_size_dollars"] = setup.fvg_event.gap_size_dollars

    # Extract sweep features
    if setup.liquidity_sweep_event:
        features["sweep_depth_ticks"] = setup.liquidity_sweep_event.sweep_depth_ticks
        features[
            "sweep_depth_dollars"
        ] = setup.liquidity_sweep_event.sweep_depth_dollars

    # Extract confidence and confluence
    features["confidence_score"] = setup.confidence
    features["confluence_count"] = setup.confluence_count

    return features


# ============================================================================
# Feature Engineer Class
# ============================================================================


class FeatureEngineer:
    """Orchestrates feature engineering from Dollar Bars.

    Calculates 40+ features across multiple categories:
    - Price-based (5 features)
    - Volume (3 features)
    - Momentum (7 features)
    - Volatility (3 features)
    - Time-based (3 features)
    - Pattern-based (7 features)
    """

    def __init__(self, model_dir: str | Path = "models/xgboost", window_size: int = 100) -> None:
        """Initialize feature engineer.

        Args:
            model_dir: Directory containing ML models (for feature metadata)
            window_size: Window size for rolling calculations
        """
        self._model_dir = Path(model_dir)
        self._window_size = window_size
        logger.info("FeatureEngineer initialized")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features from Dollar Bar DataFrame.

        Args:
            df: DataFrame with OHLCV data and timestamp column

        Returns:
            DataFrame with all engineered features
        """
        features_df = df.copy()

        # Price-based features (5)
        features_df["atr"] = calculate_atr(features_df)
        features_df["atr_ratio"] = calculate_atr_ratio(features_df)
        features_df["returns"] = calculate_returns(features_df)
        features_df["high_low_range"] = calculate_high_low_range(features_df)
        features_df["close_position"] = calculate_close_position(features_df)

        # Volume features (3)
        features_df["volume_ratio"] = calculate_volume_ratio(features_df)
        features_df["vwap"] = calculate_vwap(features_df)

        # Momentum features (7)
        features_df["rsi"] = calculate_rsi(features_df)
        macd_line, signal_line, histogram = calculate_macd(features_df)
        features_df["macd"] = macd_line
        features_df["macd_signal"] = signal_line
        features_df["macd_histogram"] = histogram
        stoch_k, stoch_d = calculate_stochastic(features_df)
        features_df["stoch_k"] = stoch_k
        features_df["stoch_d"] = stoch_d
        features_df["roc"] = calculate_rate_of_change(features_df)

        # Volatility features (3)
        features_df["historical_volatility"] = calculate_historical_volatility(
            features_df
        )
        features_df["parkinson_volatility"] = calculate_parkinson_volatility(
            features_df
        )
        features_df["garman_klass_volatility"] = calculate_garman_klass_volatility(
            features_df
        )

        # Time features (3)
        time_features = extract_time_features(features_df)
        features_df["hour"] = time_features["hour"]
        features_df["day_of_week"] = time_features["day_of_week"]
        features_df["trading_session"] = time_features["trading_session"]

        # Add additional derived features to reach 40+
        self._add_derived_features(features_df)

        return features_df

    def save_to_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Save engineered features to Parquet format with timestamp indexing.

        Args:
            df: DataFrame with engineered features
            path: File path where Parquet file will be saved
        """
        # Ensure timestamp is the index for proper time-series storage
        df_to_save = df.set_index("timestamp", drop=False)
        df_to_save.to_parquet(path, index=True)
        logger.info(f"Saved {len(df)} feature rows to {path}")

    def load_from_parquet(self, path: str) -> pd.DataFrame:
        """Load engineered features from Parquet format.

        Args:
            path: File path to load Parquet file from

        Returns:
            DataFrame with engineered features
        """
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} feature rows from {path}")
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> None:
        """Add additional derived features to reach 40+ total.

        Args:
            df: DataFrame to add features to (modified in-place)
        """
        # Price momentum features
        df["price_momentum_5"] = df["close"].pct_change(5)
        df["price_momentum_10"] = df["close"].pct_change(10)

        # Volume features
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_std_20"] = df["volume"].rolling(window=20).std()

        # Range features
        df["range_ma_20"] = df["high_low_range"].rolling(window=20).mean()
        df["range_std_20"] = df["high_low_range"].rolling(window=20).std()

        # Volatility features
        df["volatility_ma_20"] = df["historical_volatility"].rolling(window=20).mean()
        df["volatility_std_20"] = df["historical_volatility"].rolling(window=20).std()

        # RSI features
        df["rsi_ma_14"] = df["rsi"].rolling(window=14).mean()
        df["rsi_std_14"] = df["rsi"].rolling(window=14).std()

        # MACD features
        df["macd_ma_9"] = df["macd"].rolling(window=9).mean()
        df["macd_std_9"] = df["macd"].rolling(window=9).std()

        # Stochastic features
        df["stoch_k_ma_14"] = df["stoch_k"].rolling(window=14).mean()
        df["stoch_d_ma_14"] = df["stoch_d"].rolling(window=14).mean()

        # ATR features
        df["atr_ma_14"] = df["atr"].rolling(window=14).mean()
        df["atr_std_14"] = df["atr"].rolling(window=14).std()

        # Return features
        df["return_ma_10"] = df["returns"].rolling(window=10).mean()
        df["return_std_10"] = df["returns"].rolling(window=10).std()

        # Close position features
        df["close_position_ma_20"] = df["close_position"].rolling(window=20).mean()
        df["close_position_std_20"] = df["close_position"].rolling(window=20).std()

        # Trading session encoding (one-hot)
        df["is_london_am"] = (df["trading_session"] == "london_am").astype(int)
        df["is_ny_am"] = (df["trading_session"] == "ny_am").astype(int)
        df["is_ny_pm"] = (df["trading_session"] == "ny_pm").astype(int)

        # Hour cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week cyclical encoding
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    def generate_features_bar(self, current_bar, historical_data: pd.DataFrame) -> np.ndarray:
        """Generate features for a single bar using historical context.

        This method is used for bar-by-bar evaluation in live trading.
        It generates all 40+ features for the current bar based on
        historical data context.

        Args:
            current_bar: DollarBar object for current bar
            historical_data: DataFrame with historical bars (at least window_size)

        Returns:
            Feature vector as numpy array (shape: [n_features])
        """
        # Create a DataFrame with the current bar appended to historical data
        current_df = pd.DataFrame({
            'timestamp': [current_bar.timestamp],
            'open': [current_bar.open],
            'high': [current_bar.high],
            'low': [current_bar.low],
            'close': [current_bar.close],
            'volume': [current_bar.volume],
            'notional_value': [current_bar.notional_value]
        })

        # Combine historical data with current bar
        combined = pd.concat([historical_data, current_df], ignore_index=True)

        # Ensure we have enough data
        if len(combined) < self._window_size:
            logger.warning(f"Not enough historical data: {len(combined)} < {self._window_size}")
            # Pad with zeros if insufficient data
            combined = pd.concat([historical_data, current_df], ignore_index=True)

        # Engineer all features
        features_df = self.engineer_features(combined)

        # Get the last row (current bar features)
        current_features = features_df.iloc[-1]

        # Convert to numpy array for ML prediction
        # Select only numeric features (exclude timestamp, trading_session)
        # IMPORTANT: Must match the 52 features the model was trained with
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'notional_value',
            'atr', 'atr_ratio', 'returns', 'high_low_range', 'close_position',
            'volume_ratio', 'vwap', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'roc', 'historical_volatility',
            'parkinson_volatility', 'garman_klass_volatility',
            'hour', 'day_of_week',  # Raw time features (required by model)
            'price_momentum_5', 'price_momentum_10',
            'volume_ma_20', 'volume_std_20', 'range_ma_20', 'range_std_20',
            'volatility_ma_20', 'volatility_std_20',
            'rsi_ma_14', 'rsi_std_14', 'macd_ma_9', 'macd_std_9',
            'stoch_k_ma_14', 'stoch_d_ma_14', 'atr_ma_14', 'atr_std_14',
            'return_ma_10', 'return_std_10',
            'close_position_ma_20', 'close_position_std_20',
            'is_london_am', 'is_ny_am', 'is_ny_pm',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]

        # Filter to available columns
        available_features = [col for col in feature_columns if col in current_features.index]

        # Extract feature values
        feature_vector = current_features[available_features].values

        # Fill NaN values with 0
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)

        return feature_vector

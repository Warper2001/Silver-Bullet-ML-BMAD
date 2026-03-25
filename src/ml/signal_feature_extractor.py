"""Signal Feature Extractor for Meta-Labeling.

This module extracts features at signal timestamps for meta-labeling model training.
It leverages the existing FeatureEngineer to generate 40+ features.
"""

import logging
import pandas as pd
from src.ml.features import FeatureEngineer

logger = logging.getLogger(__name__)


class SignalFeatureExtractor:
    """Extract features at signal timestamps for meta-labeling.

    This class uses the existing FeatureEngineer to extract features
    from price data at the moment each signal was generated.
    """

    def __init__(self, lookback_bars: int = 100):
        """Initialize the feature extractor.

        Args:
            lookback_bars: Number of bars to use for feature calculation (default: 100)
        """
        self.lookback_bars = lookback_bars
        self.feature_engineer = FeatureEngineer()
        logger.info(f"SignalFeatureExtractor initialized with lookback={lookback_bars}")

    def extract_features_at_signal_time(
        self,
        signal_timestamp: pd.Timestamp,
        price_data: pd.DataFrame
    ) -> dict:
        """Extract features from price data at the signal timestamp.

        Gets a lookback window of price data ending at the signal time,
        engineers features using the FeatureEngineer, and returns
        the most recent feature values.

        Args:
            signal_timestamp: Timestamp when signal was generated
            price_data: DataFrame with OHLCV data (must include timestamp index)

        Returns:
            Dictionary with feature names as keys and feature values as values

        Raises:
            ValueError: If signal_timestamp not in price_data index
        """
        # Convert to pandas Timestamp if needed
        signal_time = pd.Timestamp(signal_timestamp)

        # Validate that signal time exists in price data
        if signal_time not in price_data.index:
            # Find closest previous bar
            prior_bars = price_data.index[price_data.index <= signal_time]
            if len(prior_bars) == 0:
                raise ValueError(
                    f"Signal timestamp {signal_time} is before all price data. "
                    f"Price data range: {price_data.index.min()} to {price_data.index.max()}"
                )
            signal_time = prior_bars[-1]
            logger.warning(
                f"Signal timestamp {signal_timestamp} not in price data, "
                f"using closest prior bar: {signal_time}"
            )

        # Get lookback window (data up to and including signal time)
        lookback_data = price_data.loc[:signal_time].tail(self.lookback_bars)

        # Ensure we have enough data
        if len(lookback_data) < 14:  # Minimum for ATR calculation
            raise ValueError(
                f"Insufficient data for feature extraction: "
                f"only {len(lookback_data)} bars available, "
                f"need at least 14 bars"
            )

        # Engineer features using existing FeatureEngineer
        try:
            # FeatureEngineer expects timestamp as a column, not index
            lookback_copy = lookback_data.reset_index()
            features_df = self.feature_engineer.engineer_features(lookback_copy)

            # Return most recent row (features at signal time)
            signal_features = features_df.iloc[-1].to_dict()

            # Remove features that cause model mismatch (model trained without these)
            # The XGBoost model was trained on a filtered set of features
            signal_features.pop('volume', None)  # Not in training data
            signal_features.pop('open', None)    # Not in training data
            signal_features.pop('close', None)   # Not in training data
            signal_features.pop('high', None)    # Not in training data
            signal_features.pop('low', None)     # Not in training data
            signal_features.pop('notional_value', None)  # Not in training data

            # Add signal timestamp
            signal_features['timestamp'] = signal_time

            return signal_features

        except Exception as e:
            logger.error(f"Error engineering features for signal at {signal_time}: {e}")
            raise

    def extract_for_all_signals(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Extract features for all signals in a DataFrame.

        Batch processing of feature extraction for multiple signals.

        Args:
            signals_df: DataFrame with signals (index is signal timestamps)
            price_data: DataFrame with OHLCV data
            verbose: Whether to print progress messages

        Returns:
            DataFrame with features for each signal
            Index is signal timestamp, columns are feature names
        """
        logger.info(f"Extracting features for {len(signals_df)} signals...")

        features_list = []
        failed_count = 0

        for i, (signal_timestamp, signal_row) in enumerate(signals_df.iterrows()):
            if verbose and (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{len(signals_df)} signals...")

            try:
                features = self.extract_features_at_signal_time(
                    signal_timestamp=signal_timestamp,
                    price_data=price_data
                )
                features_list.append(features)
            except Exception as e:
                logger.warning(
                    f"Failed to extract features for signal at {signal_timestamp}: {e}"
                )
                failed_count += 1

        if failed_count > 0:
            logger.warning(
                f"Failed to extract features for {failed_count}/{len(signals_df)} signals"
            )

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        # Set timestamp as index
        if 'timestamp' in features_df.columns:
            features_df = features_df.set_index('timestamp')
        else:
            features_df.index = signals_df.index[:len(features_df)]

        logger.info(f"Feature extraction complete: {len(features_df)} signals × {len(features_df.columns)} features")

        return features_df

"""ML Meta-Labeling Backtester for evaluating XGBoost model performance.

Evaluates improvement in win rate from probability-based filtering
using trained XGBoost model and feature engineering pipeline.

Performance: Completes in < 5 minutes for 2 years of data.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLMetaLabelingBacktester:
    """Backtest ML meta-labeling model on historical Silver Bullet signals.

    Evaluates improvement in win rate from probability-based filtering
    using trained XGBoost model and feature engineering pipeline.

    Performance: Completes in < 5 minutes for 2 years of data.
    """

    # Default triple-barrier configuration
    DEFAULT_TAKE_PROFIT_PCT = 0.5  # 0.5% profit target
    DEFAULT_STOP_LOSS_PCT = 0.25  # 0.25% stop loss
    DEFAULT_MAX_BARS = 50  # Max 50 bars (4+ hours for 5-min bars)

    def __init__(
        self,
        model_path: str = "data/models/xgboost_classifier.pkl",
        pipeline_path: str = "data/models/feature_pipeline.pkl",
        probability_threshold: float = 0.65,
        triple_barrier_config: dict | None = None
    ):
        """Initialize ML meta-labeling backtester.

        Args:
            model_path: Path to trained XGBoost model pickle file
            pipeline_path: Path to saved feature engineering pipeline
            probability_threshold: Minimum P(Success) for signal filtering
            triple_barrier_config: Config for triple-barrier exit labeling
        """
        self._model_path = Path(model_path)
        self._pipeline_path = Path(pipeline_path)
        self._probability_threshold = probability_threshold

        # Set triple-barrier configuration
        if triple_barrier_config is None:
            self._triple_barrier_config = {
                'take_profit_pct': self.DEFAULT_TAKE_PROFIT_PCT,
                'stop_loss_pct': self.DEFAULT_STOP_LOSS_PCT,
                'max_bars': self.DEFAULT_MAX_BARS
            }
        else:
            self._triple_barrier_config = triple_barrier_config

        # Load model and pipeline
        self._model = None
        self._pipeline = None
        self._load_model_and_pipeline()

        logger.info(
            f"MLMetaLabelingBacktester initialized: "
            f"model_path={model_path}, "
            f"pipeline_path={pipeline_path}, "
            f"probability_threshold={probability_threshold}"
        )

    def run_ml_backtest(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> dict:
        """Run complete ML backtest pipeline on historical signals.

        Args:
            signals_df: Silver Bullet signals DataFrame with columns:
                timestamp, direction, confidence, mss_detected, fvg_detected,
                sweep_detected, time_window
            price_data: Historical price data with OHLCV

        Returns:
            Dictionary with performance metrics:
            {
                'total_signals': int,
                'filtered_signals': int,
                'win_rate_all': float,
                'win_rate_filtered': float,
                'improvement_pct': float,
                'signals_all': DataFrame,
                'signals_filtered': DataFrame
            }
        """
        logger.info("Starting ML meta-labeling backtest...")

        # Engineer features for all signals
        features_df = self.engineer_features(signals_df, price_data)

        # Handle case where no features could be extracted
        if features_df.empty:
            logger.warning("No features extracted, returning empty results")
            return {
                'total_signals': len(signals_df),
                'filtered_signals': 0,
                'win_rate_all': 0.0,
                'win_rate_filtered': 0.0,
                'improvement_pct': 0.0,
                'signals_all': signals_df.copy(),
                'signals_filtered': pd.DataFrame(),
                'labels_all': pd.Series(),
                'labels_filtered': pd.Series()
            }

        # Generate probability scores
        probabilities = self.generate_probability_scores(features_df)

        # Align probabilities with original signals
        # Features may have fewer rows due to insufficient data filtering
        aligned_indices = features_df.index
        signals_aligned = signals_df.loc[aligned_indices]

        # Add probabilities to signals
        signals_with_prob = signals_aligned.copy()
        signals_with_prob['probability'] = probabilities

        # Filter by probability threshold
        filtered_signals = self.filter_by_probability(
            signals_with_prob,
            probabilities
        )

        # Label outcomes using triple-barrier
        labels_all = self.label_triple_barrier_outcomes(
            signals_with_prob,
            price_data
        )

        labels_filtered = self.label_triple_barrier_outcomes(
            filtered_signals,
            price_data
        )

        # Calculate performance metrics
        metrics_all = self.calculate_performance_metrics(
            signals_with_prob,
            labels_all
        )

        metrics_filtered = self.calculate_performance_metrics(
            filtered_signals,
            labels_filtered
        )

        # Compare results
        comparison = self.compare_results(metrics_all, metrics_filtered)

        logger.info(
            f"ML backtest complete: "
            f"{comparison['total_signals']} total signals, "
            f"{comparison['filtered_signals']} filtered signals, "
            f"win rate improved from {comparison['win_rate_all']:.1f}% "
            f"to {comparison['win_rate_filtered']:.1f}% "
            f"({comparison['improvement_pct']:+.1f}%)"
        )

        # Return comprehensive results
        return {
            **comparison,
            'signals_all': signals_with_prob,
            'signals_filtered': filtered_signals,
            'labels_all': labels_all,
            'labels_filtered': labels_filtered
        }

    def _load_model_and_pipeline(self) -> None:
        """Load trained XGBoost model and feature engineering pipeline.

        Raises:
            FileNotFoundError: If model or pipeline files not found
            RuntimeError: If model/pipeline loading fails
        """
        logger.debug("Loading model and pipeline...")

        # Load XGBoost model
        try:
            with open(self._model_path, 'rb') as f:
                self._model = pickle.load(f)
            logger.debug(f"Model loaded from {self._model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Load feature pipeline
        try:
            with open(self._pipeline_path, 'rb') as f:
                self._pipeline = pickle.load(f)
            logger.debug(f"Pipeline loaded from {self._pipeline_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pipeline file not found: {self._pipeline_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline: {e}")

    def engineer_features(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Engineer features for historical signals using saved pipeline.

        Args:
            signals_df: Signals DataFrame with timestamps
            price_data: Historical price data

        Returns:
            Features DataFrame with engineered features
        """
        logger.debug(f"Engineering features for {len(signals_df)} signals...")

        # For each signal, extract features at signal timestamp
        features_list = []
        valid_indices = []

        for idx, signal in signals_df.iterrows():
            signal_timestamp = (
                idx if isinstance(idx, pd.Timestamp) else signal['timestamp']
            )

            # Find price data at signal time
            signal_price_data = price_data.loc[:signal_timestamp].tail(100)

            if len(signal_price_data) < 10:
                logger.warning(
                    f"Insufficient price data for signal at "
                    f"{signal_timestamp}"
                )
                continue

            # Extract features
            features = self._extract_signal_features(signal, signal_price_data)
            features_list.append(features)
            valid_indices.append(idx)

        if not features_list:
            logger.warning("No features extracted")
            return pd.DataFrame()

        # Create features DataFrame
        features_df = pd.DataFrame(features_list, index=valid_indices)

        # Use pipeline for transformations (if available)
        if self._pipeline is not None:
            try:
                features_transformed = self._pipeline.transform(features_df)
                feature_names = self._pipeline.get_feature_names_out()
                features_df = pd.DataFrame(
                    features_transformed,
                    columns=feature_names,
                    index=valid_indices
                )
            except Exception as e:
                logger.warning(
                    f"Pipeline transformation failed: {e}, "
                    f"using raw features"
                )

        logger.debug(f"Features engineered: {features_df.shape}")
        return features_df

    def _extract_signal_features(
        self,
        signal: pd.Series,
        price_data: pd.DataFrame
    ) -> dict:
        """Extract features for a single signal.

        Args:
            signal: Signal row with direction, confidence, etc.
            price_data: Historical price data up to signal time

        Returns:
            Dictionary of features
        """
        features = {}

        # Price-based features
        latest_price = price_data['close'].iloc[-1]
        features['price'] = latest_price
        features['returns_5'] = (
            price_data['close'].iloc[-1] / price_data['close'].iloc[-6] - 1
            if len(price_data) >= 6 else 0
        )
        features['returns_10'] = (
            price_data['close'].iloc[-1] / price_data['close'].iloc[-11] - 1
            if len(price_data) >= 11 else 0
        )
        features['returns_20'] = (
            price_data['close'].iloc[-1] / price_data['close'].iloc[-21] - 1
            if len(price_data) >= 21 else 0
        )

        # Volatility features
        features['volatility_10'] = (
            price_data['close'].iloc[-10:].std()
            if len(price_data) >= 10 else 0
        )
        features['volatility_20'] = (
            price_data['close'].iloc[-20:].std()
            if len(price_data) >= 20 else 0
        )

        # Volume features
        features['volume_ratio'] = (
            price_data['volume'].iloc[-1] /
            price_data['volume'].iloc[-20:].mean()
            if len(price_data) >= 20 else 1
        )

        # Pattern-based features (from signal)
        features['direction'] = 1 if signal.get('direction') == 'bullish' else 0
        features['confidence'] = signal.get('confidence', 60) / 100
        features['mss_detected'] = 1 if signal.get('mss_detected') else 0
        features['fvg_detected'] = 1 if signal.get('fvg_detected') else 0
        features['sweep_detected'] = 1 if signal.get('sweep_detected') else 0

        # Time-based features
        signal_time = (
            signal.get('timestamp')
            if isinstance(signal.get('timestamp'), pd.Timestamp)
            else signal.name
        )
        if isinstance(signal_time, pd.Timestamp):
            features['hour'] = signal_time.hour
            features['day_of_week'] = signal_time.dayofweek
        else:
            features['hour'] = 12
            features['day_of_week'] = 2

        return features

    def generate_probability_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate P(Success) scores using trained XGBoost model.

        Args:
            features_df: Features DataFrame

        Returns:
            Array of probability scores (0-1)
        """
        logger.debug("Generating probability scores...")

        if self._model is None:
            raise RuntimeError("Model not loaded")

        if features_df.empty:
            logger.warning("No features to score")
            return np.array([])

        try:
            # Generate probability scores
            probabilities = self._model.predict_proba(features_df)

            # Return probability of success (class 1)
            if probabilities.shape[1] == 2:
                return probabilities[:, 1]
            else:
                return probabilities[:, 0]

        except Exception as e:
            logger.error(f"Probability generation failed: {e}")
            return np.full(len(features_df), 0.5)

    def filter_by_probability(
        self,
        signals_df: pd.DataFrame,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """Filter signals by probability threshold.

        Args:
            signals_df: Signals DataFrame with probability column
            probabilities: Array of probability scores

        Returns:
            Filtered signals DataFrame
        """
        logger.debug(
            f"Filtering signals by P >= {self._probability_threshold}..."
        )

        # Add probability column if not present
        if 'probability' not in signals_df.columns:
            signals_df = signals_df.copy()
            signals_df['probability'] = probabilities

        # Filter by threshold
        filtered = signals_df[
            signals_df['probability'] >= self._probability_threshold
        ].copy()

        logger.debug(
            f"Filtered to {len(filtered)} signals "
            f"({len(filtered) / len(signals_df) * 100:.1f}%)"
        )

        return filtered

    def label_triple_barrier_outcomes(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> pd.Series:
        """Label trade outcomes using triple-barrier exit strategy.

        Args:
            signals_df: Signals DataFrame with timestamps and directions
            price_data: Historical price data

        Returns:
            Series of labels (1=success/take_profit, 0=failure/stop_loss_or_time)
        """
        logger.debug(f"Labeling outcomes for {len(signals_df)} signals...")

        labels = []

        take_profit_pct = self._triple_barrier_config['take_profit_pct'] / 100
        stop_loss_pct = self._triple_barrier_config['stop_loss_pct'] / 100
        max_bars = self._triple_barrier_config['max_bars']

        for idx, signal in signals_df.iterrows():
            signal_timestamp = (
                idx if isinstance(idx, pd.Timestamp) else signal['timestamp']
            )
            direction = signal.get('direction', 'bullish')

            # Get entry price
            try:
                entry_price = price_data.loc[signal_timestamp, 'open']
            except KeyError:
                # Use closest available price
                signal_timestamp_loc = price_data.index.get_indexer(
                    [signal_timestamp], method='ffill'
                )[0]
                entry_price = price_data.iloc[signal_timestamp_loc]['open']

            # Calculate barriers
            if direction == 'bullish':
                take_profit_price = entry_price * (1 + take_profit_pct)
                stop_loss_price = entry_price * (1 - stop_loss_pct)
            else:  # bearish
                take_profit_price = entry_price * (1 - take_profit_pct)
                stop_loss_price = entry_price * (1 + stop_loss_pct)

            # Look ahead for barrier hits
            future_data = price_data.loc[signal_timestamp:].iloc[1:max_bars+1]

            if len(future_data) == 0:
                labels.append(0)  # Time exit
                continue

            outcome = 0  # Default: time exit (failure)

            for _, bar in future_data.iterrows():
                bar_high = bar['high']
                bar_low = bar['low']

                if direction == 'bullish':
                    if bar_high >= take_profit_price:
                        outcome = 1  # Take profit hit
                        break
                    elif bar_low <= stop_loss_price:
                        outcome = 0  # Stop loss hit
                        break
                else:  # bearish
                    if bar_low <= take_profit_price:
                        outcome = 1  # Take profit hit
                        break
                    elif bar_high >= stop_loss_price:
                        outcome = 0  # Stop loss hit
                        break

            labels.append(outcome)

        return pd.Series(labels, index=signals_df.index)

    def calculate_performance_metrics(
        self,
        signals_df: pd.DataFrame,
        labels: pd.Series
    ) -> dict:
        """Calculate performance metrics for signals.

        Args:
            signals_df: Signals DataFrame
            labels: Series of outcome labels

        Returns:
            Dictionary with metrics
        """
        if len(signals_df) == 0:
            return {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0
            }

        total_signals = len(signals_df)
        wins = labels.sum()
        losses = total_signals - wins
        win_rate = (wins / total_signals * 100) if total_signals > 0 else 0.0

        return {
            'total_signals': total_signals,
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate
        }

    def compare_results(
        self,
        metrics_all: dict,
        metrics_filtered: dict
    ) -> dict:
        """Compare performance between all and filtered signals.

        Args:
            metrics_all: Metrics for all signals
            metrics_filtered: Metrics for filtered signals

        Returns:
            Comparison results dictionary
        """
        total_signals = metrics_all['total_signals']
        filtered_signals = metrics_filtered['total_signals']
        win_rate_all = metrics_all['win_rate']
        win_rate_filtered = metrics_filtered['win_rate']

        improvement = win_rate_filtered - win_rate_all
        improvement_pct = (
            (improvement / win_rate_all * 100)
            if win_rate_all > 0 else 0.0
        )

        return {
            'total_signals': total_signals,
            'filtered_signals': filtered_signals,
            'win_rate_all': win_rate_all,
            'win_rate_filtered': win_rate_filtered,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

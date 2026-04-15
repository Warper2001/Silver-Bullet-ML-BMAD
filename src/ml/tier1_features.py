"""Tier 1 Features for 1-Minute Dollar Bar Trading.

Implements the 7 core features from Phase 1 research specifically designed
for high-frequency (1-minute) data:

1. Volume Imbalance - Direct measure of order flow (3 lookbacks)
2. Delta/Cumulative Delta - Running order flow pressure (3 lookbacks)
3. Realized Volatility - Volatility regime detection (4 horizons)
4. VWAP Deviation - Mean reversion signals (3 lookbacks)
5. Bid-Ask Bounce Indicator - Noise quantification
6. Noise-Adjusted Momentum - Signal vs noise (3 lookbacks)
7. Regime Detection Integration - Already implemented

These features are designed to:
- Filter microstructure noise (60-80% of 1-minute price movement)
- Capture order flow dynamics (leads price at high frequency)
- Detect volatility regimes (critical for 1-minute behavior)
- Provide signal vs noise distinction (volatility-adjusted metrics)

Reference: Phase 1 Research Report (_bmad-output/phase1_rearket_research_1min_features.md)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import logging

logger = logging.getLogger(__name__)


class Tier1FeatureEngineer:
    """Engineers Tier 1 features for 1-minute trading.

    Features are designed specifically for high-frequency data where:
    - Microstructure noise dominates (60-80% of price movement)
    - Order flow leads price changes
    - Volatility clustering is extreme
    - Traditional technical indicators fail

    Expected Performance:
        - Feature correlation with forward returns: >0.1
        - Statistical significance: p < 0.05
        - Feature importance: >0.05 in XGBoost
    """

    def __init__(self):
        """Initialize Tier 1 feature engineer."""
        self.feature_names = self._get_all_feature_names()
        logger.info(f"Tier1FeatureEngineer initialized with {len(self.feature_names)} features")

    def _get_all_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated.

        Returns:
            List of feature names
        """
        feature_names = []

        # 1. Volume Imbalance (3 lookbacks)
        for lookback in [3, 5, 10]:
            feature_names.append(f'volume_imbalance_{lookback}')

        # 2. Delta/Cumulative Delta (3 lookbacks)
        for lookback in [20, 50, 100]:
            feature_names.append(f'cumulative_delta_{lookback}')

        # 3. Realized Volatility (4 horizons)
        for horizon in [5, 15, 30, 60]:
            feature_names.append(f'realized_vol_{horizon}')

        # 4. VWAP Deviation (3 lookbacks)
        for lookback in [5, 10, 20]:
            feature_names.append(f'vwap_deviation_{lookback}')

        # 5. Bid-Ask Bounce Indicator
        feature_names.append('bid_ask_bounce')

        # 6. Noise-Adjusted Momentum (3 lookbacks)
        for lookback in [5, 10, 20]:
            feature_names.append(f'noise_adj_momentum_{lookback}')

        return feature_names

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all Tier 1 features.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with original data + Tier 1 features
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Generate features
        logger.info("Generating Tier 1 features...")

        # 1. Volume Imbalance
        df = self._add_volume_imbalance(df)

        # 2. Cumulative Delta
        df = self._add_cumulative_delta(df)

        # 3. Realized Volatility
        df = self._add_realized_volatility(df)

        # 4. VWAP Deviation
        df = self._add_vwap_deviation(df)

        # 5. Bid-Ask Bounce
        df = self._add_bid_ask_bounce(df)

        # 6. Noise-Adjusted Momentum
        df = self._add_noise_adjusted_momentum(df)

        logger.info(f"Generated {len(self.feature_names)} Tier 1 features")

        return df

    def _add_volume_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume imbalance features.

        Volume Imbalance = (buy_volume - sell_volume) / total_volume

        Interpretation:
        - Positive = buying pressure (aggressive buyers)
        - Negative = selling pressure (aggressive sellers)
        - Magnitude = strength of directional conviction

        At 1-minute resolution, this captures aggressive market orders
        vs passive limit orders.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume imbalance features
        """
        # Estimate buy/sell volume from price movement
        # If close > open: buyers were aggressive (candle is green)
        # If close < open: sellers were aggressive (candle is red)
        # If close == open: split volume evenly

        df['price_change'] = df['close'] - df['open']
        df['bullish_candle'] = (df['price_change'] > 0).astype(int)
        df['bearish_candle'] = (df['price_change'] < 0).astype(int)
        df['doji'] = (df['price_change'] == 0).astype(int)

        # Estimate buy/sell volume
        df['estimated_buy_vol'] = df['volume'] * (
            df['bullish_candle'] * 0.7 +  # 70% of volume to buyers
            df['bearish_candle'] * 0.3 +  # 30% to sellers
            df['doji'] * 0.5  # Split evenly for dojis
        )
        df['estimated_sell_vol'] = df['volume'] - df['estimated_buy_vol']

        # Calculate volume imbalance at multiple lookbacks
        for lookback in [3, 5, 10]:
            # Rolling sum of buy/sell volume
            buy_vol_sum = df['estimated_buy_vol'].rolling(window=lookback).sum()
            sell_vol_sum = df['estimated_sell_vol'].rolling(window=lookback).sum()
            total_vol = buy_vol_sum + sell_vol_sum

            # Volume imbalance
            imb = (buy_vol_sum - sell_vol_sum) / total_vol
            df[f'volume_imbalance_{lookback}'] = imb

        # Clean up temporary columns
        df = df.drop(['price_change', 'bullish_candle', 'bearish_candle',
                      'doji', 'estimated_buy_vol', 'estimated_sell_vol'], axis=1)

        return df

    def _add_cumulative_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative delta features.

        Delta = Cumulative(buy_volume - sell_volume)

        Interpretation:
        - Shows sustained buying/selling campaigns
        - Divergence: Price makes new high but delta doesn't = reversal
        - Slope: Positive = institutional buying, Negative = selling

        At 1-minute, cumulative delta reveals if order flow supports
        the current price movement or if it's losing momentum.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with cumulative delta features
        """
        # Calculate per-bar delta
        df['per_bar_delta'] = (
            df['close'] - df['open']
        ) * df['volume']

        # Cumulative delta
        df['cum_delta_raw'] = df['per_bar_delta'].cumsum()

        # Normalize by recent average volume (to make it scale-invariant)
        avg_vol = df['volume'].rolling(window=20).mean()
        df['cum_delta_normalized'] = df['cum_delta_raw'] / avg_vol

        # Calculate delta at different lookbacks
        for lookback in [20, 50, 100]:
            # Delta over lookback period
            delta_lookback = df['cum_delta_normalized'].diff(lookback)
            df[f'cumulative_delta_{lookback}'] = delta_lookback

        # Clean up
        df = df.drop(['per_bar_delta', 'cum_delta_raw',
                      'cum_delta_normalized'], axis=1)

        return df

    def _add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility features.

        Realized Volatility = sqrt(sum(return^2))

        Interpretation:
        - Measures actual volatility (not implied)
        - Multi-horizon: Shows volatility regime
        - Clustering: High vol → High vol (persistence)

        At 1-minute, volatility clustering is 5x stronger than daily.
        Different horizons reveal regime shifts.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with realized volatility features
        """
        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Realized volatility at different horizons
        for horizon in [5, 15, 30, 60]:
            # Squared returns
            squared_returns = df['returns']**2

            # Rolling sum (integral of squared returns)
            integrated_var = squared_returns.rolling(window=horizon).sum()

            # Square root to get volatility
            realized_vol = np.sqrt(integrated_var)

            # Annualize (rough approximation for 1-minute bars)
            # sqrt(252 trading days * 6.5 hours * 60 minutes) ≈ 313
            df[f'realized_vol_{horizon}'] = realized_vol * 313

        # Clean up
        df = df.drop(['returns'], axis=1)

        return df

    def _add_vwap_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP deviation features.

        VWAP = Volume-Weighted Average Price
        Deviation = (close - VWAP) / VWAP

        Interpretation:
        - Price above VWAP = buyers in control (bullish)
        - Price below VWAP = sellers in control (bearish)
        - Extreme deviation (>2 std) = mean reversion likely

        At 1-minute, VWAP shows short-term fair value. Deviations
        from VWAP often revert quickly.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with VWAP deviation features
        """
        # Calculate typical price (HLC/3)
        df['typical_price'] = (
            df['high'] + df['low'] + df['close']
        ) / 3

        # Calculate VWAP at different lookbacks
        for lookback in [5, 10, 20]:
            # Volume * typical price
            vol_price = df['typical_price'] * df['volume']

            # Rolling sums
            vol_sum = df['volume'].rolling(window=lookback).sum()
            vol_price_sum = vol_price.rolling(window=lookback).sum()

            # VWAP
            vwap = vol_price_sum / vol_sum

            # Deviation from VWAP
            deviation = (df['close'] - vwap) / vwap
            df[f'vwap_deviation_{lookback}'] = deviation

        # Clean up
        df = df.drop(['typical_price'], axis=1)

        return df

    def _add_bid_ask_bounce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bid-ask bounce indicator.

        Bid-Ask Bounce = Autocorrelation of returns (lag 1)

        Interpretation:
        - Negative autocorrelation = bid-ask bounce present
        - Magnitude = strength of microstructure noise
        - High bounce (>0.1) = noise dominates, avoid trading

        At 1-minute, bid-ask bounce can be 60-80% of price movement.
        This feature quantifies the noise level.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with bid-ask bounce feature
        """
        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Autocorrelation at lag 1 (rolling window)
        # This measures: Does a positive return tend to be followed by negative?
        window = 20
        autocorr = df['returns'].rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= window else np.nan
        )

        # Bid-ask bounce = negative autocorrelation
        # Take absolute value (magnitude matters, not sign)
        df['bid_ask_bounce'] = -autocorr

        # Clean up
        df = df.drop(['returns'], axis=1)

        return df

    def _add_noise_adjusted_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add noise-adjusted momentum features.

        Noise-Adjusted Momentum = (price_change) / (realized_volatility * sqrt(time))

        Interpretation:
        - Normalizes price change by volatility
        - Distinguishes signal from noise
        - >2 = strong momentum, <-2 = strong reversal

        At 1-minute, must divide by volatility to filter noise.
        Traditional momentum fails because it oscillates on noise.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with noise-adjusted momentum features
        """
        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Calculate realized volatility for normalization (use different name to avoid collision)
        df['_norm_volatility'] = df['returns'].rolling(window=5).apply(
            lambda x: np.sqrt((x**2).sum()) * 313
        )

        # Noise-adjusted momentum at different lookbacks
        for lookback in [5, 10, 20]:
            # Price change over lookback
            price_change = df['close'].diff(lookback)

            # Volatility adjustment
            vol_adjustment = df['_norm_volatility'] * np.sqrt(lookback)

            # Noise-adjusted momentum
            mom = price_change / vol_adjustment
            df[f'noise_adj_momentum_{lookback}'] = mom

        # Clean up
        df = df.drop(['returns', '_norm_volatility'], axis=1)

        return df

    def validate_features(self, df: pd.DataFrame, forward_returns: pd.Series) -> Dict[str, float]:
        """Validate feature quality against forward returns.

        Args:
            df: DataFrame with Tier 1 features
            forward_returns: Series of forward returns to predict

        Returns:
            Dictionary of validation metrics
        """
        validation_results = {}

        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                logger.warning(f"Feature {feature_name} not found in DataFrame")
                continue

            # Drop NaN values
            valid_data = pd.DataFrame({
                'feature': df[feature_name],
                'forward_return': forward_returns
            }).dropna()

            if len(valid_data) < 100:
                logger.warning(f"Insufficient data for {feature_name}")
                continue

            # Calculate correlation
            corr = valid_data['feature'].corr(valid_data['forward_return'])

            # Calculate statistical significance (p-value)
            # Using simple t-test approximation
            n = len(valid_data)
            t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

            validation_results[feature_name] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'abs_correlation': abs(corr)
            }

        # Print validation results
        logger.info("\n=== Tier 1 Feature Validation ===")
        logger.info(f"{'Feature':<30} {'Corr':>10} {'P-value':>10} {'Significant':>12}")
        logger.info("-" * 70)

        for feature_name, metrics in sorted(
            validation_results.items(),
            key=lambda x: x[1]['abs_correlation'],
            reverse=True
        ):
            sig_str = "✓" if metrics['significant'] else "✗"
            logger.info(
                f"{feature_name:<30} {metrics['correlation']:>10.3f} "
                f"{metrics['p_value']:>10.4f} {sig_str:>12}"
            )

        # Summary
        significant_features = sum(
            1 for m in validation_results.values() if m['significant']
        )
        logger.info(f"\nSignificant features (p < 0.05): {significant_features}/{len(validation_results)}")

        return validation_results


def main():
    """Test Tier 1 feature generation."""
    from pathlib import Path

    # Load sample data
    data_file = Path("data/ml_training/regime_aware_1min_2025_labeled/regime_0_training_data_labeled.csv")

    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return 1

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} bars")

    # Generate features
    engineer = Tier1FeatureEngineer()
    df_features = engineer.generate_features(df)

    print(f"\nGenerated features: {len(engineer.feature_names)}")
    print(f"Output shape: {df_features.shape}")

    # Show sample features
    print("\n=== Sample Features (first 5 rows) ===")
    # Get actual feature columns that exist
    actual_features = [f for f in engineer.feature_names if f in df_features.columns]
    print(df_features[actual_features].head())

    # Validate features (if labels exist)
    if 'label' in df_features.columns:
        print("\n=== Validating Features ===")
        forward_returns = df_features['close'].pct_change(5).shift(-5)  # 5-bar forward return
        validation = engineer.validate_features(df_features, forward_returns)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

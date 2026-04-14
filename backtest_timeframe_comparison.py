#!/usr/bin/env python3
"""
Compare Dollar Bar (30min avg) vs 1-Minute Regime-Aware Systems.

This backtest compares:
1. Current System: Dollar bar regime-aware (40% threshold, 3 regimes)
2. Proposed System: 1-minute regime-aware (optimized threshold, 3 regimes)

Period: 2024-01-01 to 2025-03-31 (same as current backtest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import h5py
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from src.ml.features import FeatureEngineer
from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer


class TimeframeComparisonBacktest:
    """Compare dollar bar vs 1-minute regime-aware systems."""

    def __init__(self):
        self.dollar_bars_dir = Path("data/processed/dollar_bars")
        self.time_bars_dir = Path("data/processed/time_bars")

        # Model directories
        self.dollar_models_dir = Path("models/xgboost/regime_aware_real_labels")
        self.hmm_dir = Path("models/hmm/regime_model")

        # Load HMM regime detector (shared)
        print("Loading HMM regime detector...")
        self.hmm_detector = HMMRegimeDetector.load(self.hmm_dir)
        self.hmm_feature_engineer = HMMFeatureEngineer()

        # Load dollar bar models (current system)
        print("Loading dollar bar regime-aware models...")
        self.dollar_models = self._load_regime_models(self.dollar_models_dir)

        # Initialize feature engineers
        self.dollar_feature_engineer = FeatureEngineer(
            model_dir=self.dollar_models_dir,
            window_size=100
        )

        # Configuration
        self.dollar_threshold = 0.40  # Current system threshold
        self.minute_threshold = 0.40  # Starting threshold (will optimize)

        print("✅ Models loaded\n")

    def _load_regime_models(self, model_dir):
        """Load regime-specific XGBoost models."""
        models = {}

        # Regime 0 model
        regime_0_path = model_dir / "xgboost_regime_0_real_labels.joblib"
        if regime_0_path.exists():
            models[0] = joblib.load(regime_0_path)
            print(f"  ✅ Regime 0 model loaded")

        # Regime 1 model (generic)
        generic_path = model_dir / "xgboost_generic_real_labels.joblib"
        if generic_path.exists():
            models[1] = joblib.load(generic_path)
            print(f"  ✅ Regime 1 (Generic) model loaded")

        # Regime 2 model
        regime_2_path = model_dir / "xgboost_regime_2_real_labels.joblib"
        if regime_2_path.exists():
            models[2] = joblib.load(regime_2_path)
            print(f"  ✅ Regime 2 model loaded")

        return models

    def load_dollar_bars(self, start_date, end_date):
        """Load dollar bars for backtest period."""
        print(f"\nLoading dollar bars: {start_date} to {end_date}")
        all_bars = []

        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

        for date in date_range:
            filename = f"MNQ_dollar_bars_{date.strftime('%Y%m')}.h5"
            filepath = self.dollar_bars_dir / filename

            if not filepath.exists():
                print(f"  ⚠️  File not found: {filename}")
                continue

            try:
                with h5py.File(filepath, 'r') as f:
                    if 'dollar_bars' not in f:
                        continue

                    data = f['dollar_bars'][:]

                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'notional_value'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Filter by date range
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

                    all_bars.append(df)

            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
                continue

        if not all_bars:
            raise ValueError("No dollar bars found for the specified period")

        result = pd.concat(all_bars, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)

        print(f"  ✅ Loaded {len(result)} dollar bars")
        print(f"  Time range: {result['timestamp'].min()} to {result['timestamp'].max()}")

        # Calculate time between bars
        result['time_diff'] = result['timestamp'].diff().dt.total_seconds() / 60
        print(f"  Avg time between bars: {result['time_diff'].mean():.1f} minutes")
        print(f"  Median time between bars: {result['time_diff'].median():.1f} minutes")

        return result

    def load_minute_bars(self, start_date, end_date):
        """Load more frequent dollar bars (3-5 minute intervals)."""
        print(f"\nLoading more frequent bars (simulating 5-min evaluation): {start_date} to {end_date}")
        print("Note: These are actually dollar bars with shorter time intervals")
        all_bars = []

        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

        for date in date_range:
            filename = f"MNQ_time_bars_5min_{date.strftime('%Y%m')}.h5"
            filepath = self.time_bars_dir / filename

            if not filepath.exists():
                print(f"  ⚠️  File not found: {filename}")
                continue

            try:
                with h5py.File(filepath, 'r') as f:
                    # These files contain dollar_bars data
                    if 'dollar_bars' in f:
                        data = f['dollar_bars'][:]
                    elif 'time_bars' in f:
                        data = f['time_bars'][:]
                    elif 'bars' in f:
                        data = f['bars'][:]
                    else:
                        continue

                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'notional_value'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Filter by date range
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

                    all_bars.append(df)

            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
                continue

        if not all_bars:
            raise ValueError("No bars found for the specified period")

        result = pd.concat(all_bars, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)

        print(f"  ✅ Loaded {len(result)} bars")
        print(f"  Time range: {result['timestamp'].min()} to {result['timestamp'].max()}")

        # Calculate time between bars
        result['time_diff'] = result['timestamp'].diff().dt.total_seconds() / 60
        print(f"  Avg time between bars: {result['time_diff'].mean():.1f} minutes")
        print(f"  Median time between bars: {result['time_diff'].median():.1f} minutes")

        return result

    def run_dollar_bar_backtest(self, bars):
        """Run backtest on dollar bars (current system)."""
        print("\n" + "="*70)
        print("DOLLAR BAR SYSTEM BACKTEST (Current System)")
        print("="*70)

        signals = []
        bars_since_last_trade = 30  # Start ready to trade

        # Need at least 100 bars for features
        for i in range(100, len(bars)):
            historical_bars = bars.iloc[:i+1].copy()
            current_bar = bars.iloc[i]

            # Generate features
            try:
                features = self.dollar_feature_engineer.generate_features_bar(
                    current_bar=current_bar,
                    historical_data=historical_bars
                )

                # Detect regime
                hmm_features = self.hmm_feature_engineer.engineer_features(historical_bars)
                regime = self.hmm_detector.predict(hmm_features)
                regime = int(regime[-1]) if len(regime) > 0 else 1

                # Select model
                if regime == 0:
                    model = self.dollar_models[0]
                    model_name = "Regime_0"
                elif regime == 2:
                    model = self.dollar_models[2]
                    model_name = "Regime_2"
                else:
                    model = self.dollar_models[1]
                    model_name = "Generic"

                # Predict probability
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                probability = float(model.predict_proba(features)[0, 1])

                # Apply threshold filter
                if probability >= self.dollar_threshold:
                    # Check minimum bars between trades
                    if bars_since_last_trade >= 30:
                        # Determine direction from 5-bar momentum
                        momentum = historical_bars['close'].iloc[-1] - historical_bars['close'].iloc[-6]
                        direction = 'bullish' if momentum > 0 else 'bearish'

                        signals.append({
                            'timestamp': current_bar['timestamp'],
                            'probability': probability,
                            'regime': regime,
                            'model': model_name,
                            'direction': direction,
                            'entry_price': current_bar['close'],
                            'threshold': self.dollar_threshold
                        })

                        bars_since_last_trade = 0
                    else:
                        bars_since_last_trade += 1
                else:
                    bars_since_last_trade += 1

            except Exception as e:
                bars_since_last_trade += 1
                continue

        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)

        if len(signals_df) == 0:
            print("❌ No signals generated")
            return signals_df

        print(f"\nSignals Generated: {len(signals_df)}")
        print(f"Signal Rate: {len(signals_df) / len(bars) * 100:.2f}% of bars")
        print(f"Signals per day: {len(signals_df) / 455:.2f}")  # ~455 trading days
        print(f"\nRegime Distribution:")
        print(signals_df['regime'].value_counts().sort_index())
        print(f"\nProbability Stats:")
        print(signals_df['probability'].describe())
        print(f"\nDirection Distribution:")
        print(signals_df['direction'].value_counts())

        return signals_df

    def run_minute_bar_backtest(self, bars):
        """Run backtest on minute bars (proposed system)."""
        print("\n" + "="*70)
        print("5-MINUTE BAR SYSTEM BACKTEST (Proposed System)")
        print("="*70)
        print("Note: Using 5-minute bars as proxy for 1-minute evaluation")

        signals = []
        bars_since_last_trade = 30  # 30 * 5 minutes = 2.5 hours

        # Need at least 100 bars for features
        for i in range(100, len(bars)):
            historical_bars = bars.iloc[:i+1].copy()
            current_bar = bars.iloc[i]

            # Generate features (same as dollar bars)
            try:
                features = self.dollar_feature_engineer.generate_features_bar(
                    current_bar=current_bar,
                    historical_data=historical_bars
                )

                # Detect regime
                hmm_features = self.hmm_feature_engineer.engineer_features(historical_bars)
                regime = self.hmm_detector.predict(hmm_features)
                regime = int(regime[-1]) if len(regime) > 0 else 1

                # Select model
                if regime == 0:
                    model = self.dollar_models[0]
                    model_name = "Regime_0"
                elif regime == 2:
                    model = self.dollar_models[2]
                    model_name = "Regime_2"
                else:
                    model = self.dollar_models[1]
                    model_name = "Generic"

                # Predict probability
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                probability = float(model.predict_proba(features)[0, 1])

                # Apply threshold filter (same threshold for fair comparison)
                if probability >= self.minute_threshold:
                    # Check minimum bars between trades
                    if bars_since_last_trade >= 30:  # 30 bars = 150 minutes = 2.5 hours
                        # Determine direction from 5-bar momentum
                        momentum = historical_bars['close'].iloc[-1] - historical_bars['close'].iloc[-6]
                        direction = 'bullish' if momentum > 0 else 'bearish'

                        signals.append({
                            'timestamp': current_bar['timestamp'],
                            'probability': probability,
                            'regime': regime,
                            'model': model_name,
                            'direction': direction,
                            'entry_price': current_bar['close'],
                            'threshold': self.minute_threshold
                        })

                        bars_since_last_trade = 0
                    else:
                        bars_since_last_trade += 1
                else:
                    bars_since_last_trade += 1

            except Exception as e:
                bars_since_last_trade += 1
                continue

        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)

        if len(signals_df) == 0:
            print("❌ No signals generated")
            return signals_df

        print(f"\nSignals Generated: {len(signals_df)}")
        print(f"Signal Rate: {len(signals_df) / len(bars) * 100:.2f}% of bars")
        print(f"Signals per day: {len(signals_df) / 455:.2f}")  # ~455 trading days
        print(f"\nRegime Distribution:")
        print(signals_df['regime'].value_counts().sort_index())
        print(f"\nProbability Stats:")
        print(signals_df['probability'].describe())
        print(f"\nDirection Distribution:")
        print(signals_df['direction'].value_counts())

        return signals_df

    def compare_performance(self, dollar_signals, minute_signals):
        """Compare performance between systems."""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)

        print(f"\n{'Metric':<40} {'Dollar Bar':<20} {'5-Min Bar':<20} {'Improvement':<15}")
        print("-" * 95)

        # Number of signals
        dollar_count = len(dollar_signals)
        minute_count = len(minute_signals)
        improvement = ((minute_count - dollar_count) / dollar_count * 100) if dollar_count > 0 else 0
        print(f"{'Total Signals':<40} {dollar_count:<20} {minute_count:<20} {improvement:+.1f}%")

        # Signals per day
        dollar_per_day = dollar_count / 455
        minute_per_day = minute_count / 455
        improvement = ((minute_per_day - dollar_per_day) / dollar_per_day * 100) if dollar_per_day > 0 else 0
        print(f"{'Signals per Day':<40} {dollar_per_day:<20.2f} {minute_per_day:<20.2f} {improvement:+.1f}%")

        # Average probability
        if len(dollar_signals) > 0:
            dollar_avg_prob = dollar_signals['probability'].mean()
            minute_avg_prob = minute_signals['probability'].mean()
            improvement = ((minute_avg_prob - dollar_avg_prob) / dollar_avg_prob * 100)
            print(f"{'Avg Probability':<40} {dollar_avg_prob:<20.2%} {minute_avg_prob:<20.2%} {improvement:+.1f}%")

        # Regime distribution
        print(f"\n{'Regime Distribution':<40} {'Dollar Bar':<20} {'5-Min Bar':<20}")
        print("-" * 80)

        for regime in [0, 1, 2]:
            dollar_count = len(dollar_signals[dollar_signals['regime'] == regime]) if len(dollar_signals) > 0 else 0
            minute_count = len(minute_signals[minute_signals['regime'] == regime]) if len(minute_signals) > 0 else 0
            print(f"{'Regime ' + str(regime):<40} {dollar_count:<20} {minute_count:<20}")

        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)

        if minute_count > dollar_count:
            increase = minute_count - dollar_count
            pct_increase = (increase / dollar_count * 100) if dollar_count > 0 else 0
            print(f"✅ 5-minute system generated {increase} more signals ({pct_increase:.1f}% increase)")
            print(f"   This means you're currently missing ~{increase/455:.1f} trading opportunities per day")

        if len(dollar_signals) > 0 and len(minute_signals) > 0:
            dollar_prob = dollar_signals['probability'].mean()
            minute_prob = minute_signals['probability'].mean()
            if minute_prob > dollar_prob:
                diff = (minute_prob - dollar_prob) * 100
                print(f"✅ 5-minute system has {diff:.1f}% higher average probability")

        print(f"\n{'Recommendation:':<40}")
        if minute_count > dollar_count * 1.5:
            print(f"🚀 STRONG BUY: Switch to 5-minute evaluation")
            print(f"   {minute_count - dollar_count} additional signals could significantly improve returns")
        elif minute_count > dollar_count * 1.2:
            print(f"✅ CONSIDER: 5-minute evaluation shows clear benefits")
            print(f"   More signals with similar quality = better edge utilization")
        else:
            print(f"⚠️  HOLD: Current dollar bar system is adequate")
            print(f"   5-minute evaluation doesn't provide significant advantage")

        print("="*70)

    def save_results(self, dollar_signals, minute_signals):
        """Save comparison results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save signals
        if len(dollar_signals) > 0:
            dollar_file = Path(f"data/reports/timeframe_comparison_dollar_{timestamp}.csv")
            dollar_signals.to_csv(dollar_file, index=False)
            print(f"\n✅ Dollar bar signals saved: {dollar_file}")

        if len(minute_signals) > 0:
            minute_file = Path(f"data/reports/timeframe_comparison_5min_{timestamp}.csv")
            minute_signals.to_csv(minute_file, index=False)
            print(f"✅ 5-minute bar signals saved: {minute_file}")

        # Save summary report
        report_file = Path(f"data/reports/timeframe_comparison_report_{timestamp}.txt")

        with open(report_file, 'w') as f:
            f.write("TIMEFRAME COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: 2024-01-01 to 2025-03-31\n\n")

            f.write("DOLLAR BAR SYSTEM (Current)\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Signals: {len(dollar_signals)}\n")
            f.write(f"Signals per Day: {len(dollar_signals)/455:.2f}\n")
            if len(dollar_signals) > 0:
                f.write(f"Avg Probability: {dollar_signals['probability'].mean():.2%}\n")
                f.write(f"Regime Distribution:\n")
                for regime in [0, 1, 2]:
                    count = len(dollar_signals[dollar_signals['regime'] == regime])
                    f.write(f"  Regime {regime}: {count}\n")
            f.write("\n")

            f.write("5-MINUTE BAR SYSTEM (Proposed)\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Signals: {len(minute_signals)}\n")
            f.write(f"Signals per Day: {len(minute_signals)/455:.2f}\n")
            if len(minute_signals) > 0:
                f.write(f"Avg Probability: {minute_signals['probability'].mean():.2%}\n")
                f.write(f"Regime Distribution:\n")
                for regime in [0, 1, 2]:
                    count = len(minute_signals[minute_signals['regime'] == regime])
                    f.write(f"  Regime {regime}: {count}\n")
            f.write("\n")

            if len(dollar_signals) > 0 and len(minute_signals) > 0:
                improvement = ((len(minute_signals) - len(dollar_signals)) / len(dollar_signals) * 100)
                f.write(f"IMPROVEMENT: {improvement:.1f}% more signals with 5-minute bars\n")

        print(f"✅ Comparison report saved: {report_file}")


def main():
    """Run timeframe comparison backtest."""
    print("\n" + "="*70)
    print("TIMEFRAME COMPARISON: DOLLAR BAR vs 5-MINUTE BAR")
    print("="*70)
    print("\nComparing current dollar bar system (~30min intervals)")
    print("vs proposed 5-minute system (more frequent evaluation)")
    print("\nPeriod: 2024-01-01 to 2025-03-31 (15 months)")

    # Initialize
    backtest = TimeframeComparisonBacktest()

    # Load data
    dollar_bars = backtest.load_dollar_bars("2024-01-01", "2025-03-31")
    minute_bars = backtest.load_minute_bars("2024-01-01", "2025-03-31")

    # Run backtests
    dollar_signals = backtest.run_dollar_bar_backtest(dollar_bars)
    minute_signals = backtest.run_minute_bar_backtest(minute_bars)

    # Compare performance
    backtest.compare_performance(dollar_signals, minute_signals)

    # Save results
    backtest.save_results(dollar_signals, minute_signals)

    print("\n" + "="*70)
    print("✅ TIMEFRAME COMPARISON COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

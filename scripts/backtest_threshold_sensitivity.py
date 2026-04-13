#!/usr/bin/env python3
"""Threshold sensitivity analysis to find optimal probability threshold for target trade frequency."""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import joblib
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThresholdSensitivityAnalyzer:
    """Analyze trade frequency vs win rate across different probability thresholds."""

    def __init__(
        self,
        hmm_detector: HMMRegimeDetector,
        hmm_feature_engineer: HMMFeatureEngineer,
        feature_engineer: FeatureEngineer,
        generic_model,
        regime_0_model,
        regime_2_model,
        probability_thresholds: List[float] = None
    ):
        self.hmm_detector = hmm_detector
        self.hmm_feature_engineer = hmm_feature_engineer
        self.feature_engineer = feature_engineer
        self.generic_model = generic_model
        self.regime_0_model = regime_0_model
        self.regime_2_model = regime_2_model

        # Test thresholds from 40% to 70% in 2.5% increments
        self.probability_thresholds = probability_thresholds or [
            0.40, 0.425, 0.45, 0.475, 0.50,
            0.525, 0.55, 0.575, 0.60, 0.625,
            0.65, 0.675, 0.70
        ]

        # Trading parameters
        self.take_profit_pct = 0.003  # 0.3%
        self.stop_loss_pct = 0.002    # 0.2%
        self.max_hold_minutes = 30

    def load_dollar_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and prepare dollar bar data."""
        logger.info(f"\nLoading dollar bars from {start_date} to {end_date}...")

        data_dir = Path("data/processed/dollar_bars/")
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        dataframes = []
        current = start_dt.replace(day=1)

        while current <= end_dt:
            filename = f"MNQ_dollar_bars_{current.strftime('%Y%m')}.h5"
            file_path = data_dir / filename

            if file_path.exists():
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['dollar_bars'][:]
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    dataframes.append(df)
                except Exception as e:
                    logger.error(f"  Failed to load {filename}: {e}")

            current = current + pd.DateOffset(months=1)

        if not dataframes:
            raise ValueError(f"No data found for {start_date} to {end_date}")

        combined = pd.concat(dataframes, ignore_index=True)
        combined = combined.sort_values('timestamp')
        combined = combined.loc[
            (combined['timestamp'] >= start_dt) &
            (combined['timestamp'] <= end_dt)
        ]

        logger.info(f"✅ Loaded {len(combined):,} dollar bars")
        return combined

    def detect_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect regimes for all data."""
        logger.info("\nDetecting regimes...")

        hmm_features = self.hmm_feature_engineer.engineer_features(data)
        regimes = self.hmm_detector.predict(hmm_features)

        regime_df = pd.DataFrame({
            'regime': regimes,
            'regime_name': [self.hmm_detector.metadata.regime_names[int(r)] for r in regimes]
        }, index=data.index)

        logger.info(f"Regime distribution:")
        for regime, count in regime_df['regime_name'].value_counts().items():
            pct = count / len(regime_df) * 100
            logger.info(f"  {regime}: {count:,} bars ({pct:.1f}%)")

        return regime_df

    def generate_realistic_signals(self, data: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic Silver Bullet-like signals."""
        logger.info("\nGenerating realistic Silver Bullet-like signals...")

        # Calculate indicators
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['volume_ma'] = data['volume'].rolling(20).mean()

        # Signal criteria (lowered for more signals)
        vol_threshold = data['volatility'].quantile(0.60)  # Top 40% (lowered from 75%)
        vol_ratio = data['volume'] / data['volume_ma']

        # More lenient signal criteria
        data['signal_strength'] = (
            (data['volatility'] > vol_threshold).astype(int) +
            (vol_ratio > 1.3).astype(int) +  # Lowered from 1.5
            (abs(data['returns']) > 0.0005).astype(int)  # Lowered from 0.001
        )

        # Accept signal_strength >= 1 (more lenient)
        signals = data[data['signal_strength'] >= 1].copy()

        # Determine direction based on recent price movement
        signals['momentum_5'] = signals['close'].pct_change(5)
        signals['signal_direction'] = np.where(
            signals['momentum_5'] > 0, 1, -1
        )

        logger.info(f"Generated {len(signals)} potential signals")
        logger.info(f"  Bullish: {len(signals[signals['signal_direction'] == 1])}")
        logger.info(f"  Bearish: {len(signals[signals['signal_direction'] == -1])}")

        return signals

    def get_model_prediction(
        self,
        signal_row: pd.Series,
        features_df: pd.DataFrame,
        model,
        model_name: str
    ) -> Tuple[float, bool]:
        """Get model prediction for a signal."""
        signal_time = signal_row.name

        # Get features at signal time
        if signal_time not in features_df.index:
            return 0.0, False

        features = features_df.loc[[signal_time]]

        # Get feature columns expected by model
        expected_features = model.feature_names_in_
        available_features = [f for f in expected_features if f in features.columns]
        X = features[available_features].fillna(0)

        # Get prediction
        prediction_proba = model.predict_proba(X)[0, 1]

        return float(prediction_proba), True

    def simulate_trade_with_exits(
        self,
        signal_row: pd.Series,
        data: pd.DataFrame,
        direction: int
    ) -> Tuple[float, str, int]:
        """Simulate trade with triple-barrier exits."""
        entry_price = signal_row['close']
        take_profit_price = entry_price * (1 + self.take_profit_pct * direction)
        stop_loss_price = entry_price * (1 - self.stop_loss_pct * direction)

        entry_idx = data.index.get_loc(signal_row.name)
        max_hold_bars = self.max_hold_minutes // 5  # 5-minute bars

        exit_reason = 'time_stop'
        exit_price = entry_price

        for i in range(1, min(max_hold_bars + 1, len(data) - entry_idx)):
            bar = data.iloc[entry_idx + i]
            current_price = bar['close']

            # Check take profit first
            if direction == 1:  # Long
                if bar['high'] >= take_profit_price:
                    exit_reason = 'take_profit'
                    exit_price = take_profit_price
                    break
                if bar['low'] <= stop_loss_price:
                    exit_reason = 'stop_loss'
                    exit_price = stop_loss_price
                    break
            else:  # Short
                if bar['low'] <= take_profit_price:
                    exit_reason = 'take_profit'
                    exit_price = take_profit_price
                    break
                if bar['high'] >= stop_loss_price:
                    exit_reason = 'stop_loss'
                    exit_price = stop_loss_price
                    break

            exit_price = current_price

        # Calculate P&L
        price_change_pct = (exit_price - entry_price) / entry_price
        pnl_pct = price_change_pct * direction * 100

        return pnl_pct, exit_reason, i * 5

    def backtest_at_threshold(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        probability_threshold: float
    ) -> Dict:
        """Run backtest at a specific probability threshold."""
        results = []

        for idx, signal in signals.iterrows():
            if idx not in regime_df.index:
                continue

            regime = regime_df.loc[idx, 'regime']

            # Hybrid model selection
            if regime == 0:
                prediction, success = self.get_model_prediction(
                    signal, features_df, self.regime_0_model, 'Regime_0'
                )
                model_used = 'Regime_0'
            elif regime == 2:
                prediction, success = self.get_model_prediction(
                    signal, features_df, self.regime_2_model, 'Regime_2'
                )
                model_used = 'Regime_2'
            else:  # Regime 1
                prediction, success = self.get_model_prediction(
                    signal, features_df, self.generic_model, 'Generic'
                )
                model_used = 'Generic'

            if not success:
                continue

            # Check threshold
            if prediction < probability_threshold:
                continue

            # Simulate trade
            direction = signal['signal_direction']
            pnl_pct, exit_reason, hold_minutes = self.simulate_trade_with_exits(
                signal, data, direction
            )

            results.append({
                'timestamp': idx,
                'prediction': prediction,
                'regime': int(regime),
                'model_used': model_used,
                'direction': 'long' if direction == 1 else 'short',
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'hold_minutes': hold_minutes,
                'outcome': 'win' if pnl_pct > 0 else 'loss'
            })

        if len(results) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'total_pnl_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'max_drawdown_pct': 0
            }

        trades_df = pd.DataFrame(results)

        # Calculate metrics
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = (winning_trades / num_trades * 100)
        total_pnl = trades_df['pnl_pct'].sum()
        avg_trade = trades_df['pnl_pct'].mean()
        std_trade = trades_df['pnl_pct'].std()
        sharpe = (avg_trade / std_trade * np.sqrt(252)) if std_trade > 0 else 0

        # Profit factor
        wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        losses = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = (wins / losses) if losses > 0 else 0

        # Max drawdown
        cumulative = trades_df['pnl_pct'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max)
        max_drawdown = drawdown.min()

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown_pct': max_drawdown
        }

    def analyze_sensitivity(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        trading_days: int = 390
    ) -> pd.DataFrame:
        """Analyze performance across all thresholds."""
        logger.info("\n" + "=" * 70)
        logger.info("THRESHOLD SENSITIVITY ANALYSIS")
        logger.info("=" * 70)

        results_list = []

        for threshold in self.probability_thresholds:
            logger.info(f"\nTesting threshold: {threshold:.1%}")

            metrics = self.backtest_at_threshold(
                data, signals, features_df, regime_df, threshold
            )

            # Calculate trades per day
            trades_per_day = metrics['num_trades'] / trading_days

            results_list.append({
                'threshold': threshold,
                'num_trades': metrics['num_trades'],
                'trades_per_day': trades_per_day,
                'trades_per_month': trades_per_day * 21,
                'win_rate': metrics['win_rate'],
                'total_pnl_pct': metrics['total_pnl_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'profit_factor': metrics['profit_factor'],
                'avg_trade': metrics['avg_trade'],
                'max_drawdown_pct': metrics['max_drawdown_pct']
            })

        results_df = pd.DataFrame(results_list)

        return results_df

    def find_optimal_threshold(self, results_df: pd.DataFrame) -> Dict:
        """Find optimal threshold based on constraints."""
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMAL THRESHOLD ANALYSIS")
        logger.info("=" * 70)

        # Filter by minimum trade frequency (1 trade/day)
        min_freq = results_df[results_df['trades_per_day'] >= 1.0]

        if len(min_freq) == 0:
            logger.warning("⚠️  No thresholds meet minimum 1 trade/day requirement!")
            logger.info("  Need to expand signal generation or lower threshold further")
            return None

        # Filter by minimum win rate (50%)
        viable = min_freq[min_freq['win_rate'] >= 50.0]

        if len(viable) == 0:
            logger.warning("⚠️  No thresholds meet both 1 trade/day AND 50% win rate!")
            logger.info("  Best option at 1 trade/day:")
            best = min_freq.loc[min_freq['win_rate'].idxmax()]
            return best.to_dict()

        # Among viable options, maximize Sharpe ratio
        optimal = viable.loc[viable['sharpe_ratio'].idxmax()]

        logger.info(f"\n✅ Optimal threshold found: {optimal['threshold']:.1%}")
        logger.info(f"  Trades per day: {optimal['trades_per_day']:.2f}")
        logger.info(f"  Win rate: {optimal['win_rate']:.2f}%")
        logger.info(f"  Sharpe ratio: {optimal['sharpe_ratio']:.2f}")
        logger.info(f"  Profit factor: {optimal['profit_factor']:.2f}")
        logger.info(f"  Total P&L: {optimal['total_pnl_pct']:.2f}%")

        return optimal.to_dict()

    def generate_report(
        self,
        results_df: pd.DataFrame,
        optimal_threshold: Dict,
        output_dir: Path
    ):
        """Generate comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_path = output_dir / f"threshold_sensitivity_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n✅ Results saved to {results_path}")

        # Generate summary report
        report_path = output_dir / f"threshold_sensitivity_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("THRESHOLD SENSITIVITY ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: 2024-01-01 to 2025-03-31 (15 months)\n")
            f.write(f"Trading Days: ~390\n")
            f.write(f"Thresholds Tested: {len(self.probability_thresholds)}\n\n")

            f.write("=" * 70 + "\n")
            f.write("OPTIMAL THRESHOLD RECOMMENDATION\n")
            f.write("=" * 70 + "\n\n")

            if optimal_threshold:
                f.write(f"✅ RECOMMENDED THRESHOLD: {optimal_threshold['threshold']:.1%}\n\n")
                f.write("Expected Performance:\n")
                f.write(f"  Trades per day: {optimal_threshold['trades_per_day']:.2f}\n")
                f.write(f"  Trades per month: {optimal_threshold['trades_per_month']:.1f}\n")
                f.write(f"  Win rate: {optimal_threshold['win_rate']:.2f}%\n")
                f.write(f"  Sharpe ratio: {optimal_threshold['sharpe_ratio']:.2f}\n")
                f.write(f"  Profit factor: {optimal_threshold['profit_factor']:.2f}\n")
                f.write(f"  Total P&L: {optimal_threshold['total_pnl_pct']:.2f}%\n")
                f.write(f"  Max drawdown: {optimal_threshold['max_drawdown_pct']:.2f}%\n\n")
            else:
                f.write("⚠️  NO THRESHOLD MEETS MINIMUM REQUIREMENTS\n\n")
                f.write("Requirements:\n")
                f.write("  - Minimum 1 trade/day\n")
                f.write("  - Minimum 50% win rate\n\n")
                f.write("Recommendations:\n")
                f.write("  1. Expand signal generation (lower volatility/volume thresholds)\n")
                f.write("  2. Use regime-specific thresholds\n")
                f.write("  3. Accept lower win rate (45-50%)\n\n")

            f.write("=" * 70 + "\n")
            f.write("FULL THRESHOLD COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            for _, row in results_df.iterrows():
                meets_min = "✅" if row['trades_per_day'] >= 1.0 else "❌"
                meets_win = "✅" if row['win_rate'] >= 50.0 else "❌"

                f.write(f"Threshold: {row['threshold']:.1%} {meets_min}\n")
                f.write(f"  Trades/Day: {row['trades_per_day']:.2f} ({row['trades_per_month']:.1f}/month)\n")
                f.write(f"  Win Rate: {row['win_rate']:.2f}% {meets_win}\n")
                f.write(f"  Sharpe: {row['sharpe_ratio']:.2f}\n")
                f.write(f"  P&L: {row['total_pnl_pct']:.2f}%\n")
                f.write(f"  Profit Factor: {row['profit_factor']:.2f}\n")
                f.write(f"  Max DD: {row['max_drawdown_pct']:.2f}%\n\n")

        logger.info(f"✅ Report saved to {report_path}")

        return optimal_threshold


def main():
    """Main execution."""
    logger.info("\n" + "=" * 70)
    logger.info("THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("Finding optimal probability threshold for 1-20 trades/day target")
    logger.info("=" * 70)

    try:
        # Load models
        logger.info("\nLoading models...")
        hmm_dir = Path("models/hmm/regime_model")

        hmm_detector = HMMRegimeDetector.load(hmm_dir)
        logger.info(f"✅ HMM loaded: {hmm_detector.n_regimes} regimes")

        generic_model = joblib.load(Path("models/xgboost/regime_aware_real_labels/xgboost_generic_real_labels.joblib"))
        regime_0_model = joblib.load(Path("models/xgboost/regime_aware_real_labels/xgboost_regime_0_real_labels.joblib"))
        regime_2_model = joblib.load(Path("models/xgboost/regime_aware_real_labels/xgboost_regime_2_real_labels.joblib"))
        logger.info("✅ ML models loaded")

        # Initialize analyzers
        hmm_feature_engineer = HMMFeatureEngineer()
        feature_engineer = FeatureEngineer()

        analyzer = ThresholdSensitivityAnalyzer(
            hmm_detector=hmm_detector,
            hmm_feature_engineer=hmm_feature_engineer,
            feature_engineer=feature_engineer,
            generic_model=generic_model,
            regime_0_model=regime_0_model,
            regime_2_model=regime_2_model
        )

        # Load data
        data = analyzer.load_dollar_bars("2024-01-01", "2025-03-31")

        # Detect regimes
        regime_df = analyzer.detect_regimes(data)

        # Generate signals
        signals = analyzer.generate_realistic_signals(data, regime_df)

        # Limit for faster analysis (remove limit for full results)
        max_signals = 2000
        signals = signals.head(max_signals)
        logger.info(f"Limited to {max_signals} signals for faster analysis")

        # Engineer features
        logger.info("\nEngineering features...")
        features_df = analyzer.feature_engineer.engineer_features(data)
        logger.info(f"✅ {len(features_df.columns)} features engineered")

        # Run sensitivity analysis
        results_df = analyzer.analyze_sensitivity(
            data=data,
            signals=signals,
            features_df=features_df,
            regime_df=regime_df,
            trading_days=390
        )

        # Find optimal threshold
        optimal_threshold = analyzer.find_optimal_threshold(results_df)

        # Generate report
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer.generate_report(results_df, optimal_threshold, output_dir)

        logger.info("\n" + "=" * 70)
        logger.info("✅ THRESHOLD SENSITIVITY ANALYSIS COMPLETE")
        logger.info("=" * 70)

        if optimal_threshold:
            logger.info(f"\n🎯 RECOMMENDED THRESHOLD: {optimal_threshold['threshold']:.1%}")
            logger.info(f"   Expected trades per day: {optimal_threshold['trades_per_day']:.2f}")
            logger.info(f"   Expected win rate: {optimal_threshold['win_rate']:.2f}%")
        else:
            logger.warning("\n⚠️  No threshold meets minimum requirements")
            logger.info("   Consider expanding signal generation")

    except Exception as e:
        logger.error(f"\n❌ Analysis failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

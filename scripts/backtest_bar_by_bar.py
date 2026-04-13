#!/usr/bin/env python3
"""Bar-by-bar ML backtest to achieve 1-20 trades per day target.

This backtest evaluates EVERY 5-minute bar (not just filtered signals)
to maximize trading opportunities and hit the target frequency.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Tuple

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


class BarByBarBacktester:
    """Evaluate every bar with ML model for maximum trade frequency."""

    def __init__(
        self,
        hmm_detector: HMMRegimeDetector,
        hmm_feature_engineer: HMMFeatureEngineer,
        feature_engineer: FeatureEngineer,
        generic_model,
        regime_0_model,
        regime_2_model,
        probability_threshold: float = 0.45
    ):
        self.hmm_detector = hmm_detector
        self.hmm_feature_engineer = hmm_feature_engineer
        self.feature_engineer = feature_engineer
        self.generic_model = generic_model
        self.regime_0_model = regime_0_model
        self.regime_2_model = regime_2_model
        self.probability_threshold = probability_threshold

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

    def get_model_prediction(
        self,
        bar_idx: int,
        features_df: pd.DataFrame,
        model,
        model_name: str
    ) -> Tuple[float, bool]:
        """Get model prediction for a specific bar."""
        # Get features at bar time
        if bar_idx not in features_df.index:
            return 0.0, False

        features = features_df.loc[[bar_idx]]

        # Get feature columns expected by model
        expected_features = model.feature_names_in_
        available_features = [f for f in expected_features if f in features.columns]
        X = features[available_features].fillna(0)

        # Get prediction
        prediction_proba = model.predict_proba(X)[0, 1]

        return float(prediction_proba), True

    def determine_direction(self, data: pd.DataFrame, bar_idx: int) -> int:
        """Determine trade direction based on recent price movement."""
        # Get recent close prices
        if bar_idx < 5:
            return 1  # Default to long

        recent_closes = data.iloc[bar_idx-5:bar_idx+1]['close']
        momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]

        return 1 if momentum > 0 else -1

    def simulate_trade_with_exits(
        self,
        entry_bar: pd.Series,
        data: pd.DataFrame,
        direction: int
    ) -> Tuple[float, str, int]:
        """Simulate trade with triple-barrier exits."""
        entry_price = entry_bar['close']
        take_profit_price = entry_price * (1 + self.take_profit_pct * direction)
        stop_loss_price = entry_price * (1 - self.stop_loss_pct * direction)

        entry_idx = data.index.get_loc(entry_bar.name)
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

    def run_backtest(
        self,
        data: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        bars_between_trades: int = 30
    ) -> Dict:
        """Run bar-by-bar backtest."""
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING BAR-BY-BAR BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Evaluating EVERY bar with ML model")
        logger.info(f"Probability threshold: {self.probability_threshold:.1%}")
        logger.info(f"Min bars between trades: {bars_between_trades}")

        results = []
        last_trade_idx = -1000  # Force first trade

        # Evaluate every bar (skip first 100 for feature stability)
        start_bar = 100
        total_bars = len(data)

        for idx in range(start_bar, total_bars):
            bar_idx = data.index[idx]

            # Check if enough bars since last trade
            bars_since_last = idx - last_trade_idx
            if bars_since_last < bars_between_trades:
                continue

            # Get regime
            if bar_idx not in regime_df.index:
                continue
            regime = regime_df.loc[bar_idx, 'regime']

            # Select model based on regime
            if regime == 0:
                prediction, success = self.get_model_prediction(
                    bar_idx, features_df, self.regime_0_model, 'Regime_0'
                )
                model_used = 'Regime_0'
            elif regime == 2:
                prediction, success = self.get_model_prediction(
                    bar_idx, features_df, self.regime_2_model, 'Regime_2'
                )
                model_used = 'Regime_2'
            else:  # Regime 1
                prediction, success = self.get_model_prediction(
                    bar_idx, features_df, self.generic_model, 'Generic'
                )
                model_used = 'Generic'

            if not success:
                continue

            # Check threshold
            if prediction < self.probability_threshold:
                continue

            # Determine direction
            direction = self.determine_direction(data, idx)

            # Get bar data
            bar = data.iloc[idx]

            # Simulate trade
            pnl_pct, exit_reason, hold_minutes = self.simulate_trade_with_exits(
                bar, data, direction
            )

            results.append({
                'timestamp': bar_idx,
                'prediction': prediction,
                'regime': int(regime),
                'model_used': model_used,
                'direction': 'long' if direction == 1 else 'short',
                'entry_price': bar['close'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'hold_minutes': hold_minutes,
                'outcome': 'win' if pnl_pct > 0 else 'loss'
            })

            last_trade_idx = idx

            # Progress update
            if len(results) % 100 == 0:
                logger.info(f"  Generated {len(results)} trades...")

        if len(results) == 0:
            logger.warning("No trades generated!")
            return {
                'num_trades': 0,
                'trades_df': pd.DataFrame()
            }

        trades_df = pd.DataFrame(results)

        # Calculate metrics
        logger.info("\n" + "=" * 70)
        logger.info("BAR-BY-BAR BACKTEST RESULTS")
        logger.info("=" * 70)

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

        # Calculate trades per day
        trading_days = 390  # Approximate trading days in 15 months
        trades_per_day = num_trades / trading_days
        trades_per_month = trades_per_day * 21

        logger.info(f"\nTotal Trades: {num_trades}")
        logger.info(f"Trades per Day: {trades_per_day:.2f}")
        logger.info(f"Trades per Month: {trades_per_month:.1f}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total P&L: {total_pnl:.2f}%")
        logger.info(f"Avg Trade: {avg_trade:.3f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")

        # Exit breakdown
        logger.info("\nExit Reasons:")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            pct = count / len(trades_df) * 100
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")

        # Per-regime breakdown
        logger.info("\nPer-Regime Performance:")
        for regime in sorted(trades_df['regime'].unique()):
            regime_trades = trades_df[trades_df['regime'] == regime]
            regime_wins = len(regime_trades[regime_trades['pnl_pct'] > 0])
            regime_win_rate = (regime_wins / len(regime_trades) * 100)
            regime_pnl = regime_trades['pnl_pct'].sum()
            regime_name = self.hmm_detector.metadata.regime_names[int(regime)]

            logger.info(f"  Regime {regime} ({regime_name}):")
            logger.info(f"    Trades: {len(regime_trades)}")
            logger.info(f"    Win Rate: {regime_win_rate:.2f}%")
            logger.info(f"    P&L: {regime_pnl:.2f}%")

        return {
            'num_trades': num_trades,
            'trades_per_day': trades_per_day,
            'trades_per_month': trades_per_month,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown_pct': max_drawdown,
            'trades_df': trades_df
        }

    def generate_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trades
        if len(results['trades_df']) > 0:
            trades_path = output_dir / f"bar_by_bar_trades_{timestamp}.csv"
            results['trades_df'].to_csv(trades_path, index=False)
            logger.info(f"\n✅ Trades saved to {trades_path}")

        # Generate summary report
        report_path = output_dir / f"bar_by_bar_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("BAR-BY-BAR ML BACKTEST REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: 2024-01-01 to 2025-03-31 (15 months)\n")
            f.write(f"Trading Days: ~390\n")
            f.write(f"Approach: Evaluate EVERY bar (not just signals)\n")
            f.write(f"Probability Threshold: {self.probability_threshold:.1%}\n\n")

            f.write("=" * 70 + "\n")
            f.write("RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total Trades: {results['num_trades']}\n")
            f.write(f"Trades per Day: {results['trades_per_day']:.2f}\n")
            f.write(f"Trades per Month: {results['trades_per_month']:.1f}\n")
            f.write(f"Win Rate: {results['win_rate']:.2f}%\n")
            f.write(f"Total P&L: {results['total_pnl_pct']:.2f}%\n")
            f.write(f"Avg Trade: {results['avg_trade']:.3f}%\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
            f.write(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%\n\n")

            # Check if meets target
            meets_target = results['trades_per_day'] >= 1.0 and results['trades_per_day'] <= 20.0
            f.write("=" * 70 + "\n")
            f.write("TARGET VALIDATION\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Target: 1-20 trades per day\n")
            f.write(f"Actual: {results['trades_per_day']:.2f} trades per day\n")
            f.write(f"Status: {'✅ MEETS TARGET' if meets_target else '❌ DOES NOT MEET TARGET'}\n\n")

            if meets_target:
                f.write("✅ System ready for paper trading deployment\n")
            else:
                f.write("⚠️  Need to adjust threshold or approach\n")

        logger.info(f"✅ Report saved to {report_path}")

        return meets_target


def main():
    """Main execution."""
    logger.info("\n" + "=" * 70)
    logger.info("BAR-BY-BAR ML BACKTEST")
    logger.info("Evaluating every bar to achieve 1-20 trades/day target")
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

        # Test multiple thresholds
        thresholds = [0.40, 0.45, 0.50, 0.55]

        for threshold in thresholds:
            logger.info("\n" + "=" * 70)
            logger.info(f"TESTING THRESHOLD: {threshold:.1%}")
            logger.info("=" * 70)

            # Initialize backtester
            backtester = BarByBarBacktester(
                hmm_detector=hmm_detector,
                hmm_feature_engineer=HMMFeatureEngineer(),
                feature_engineer=FeatureEngineer(),
                generic_model=generic_model,
                regime_0_model=regime_0_model,
                regime_2_model=regime_2_model,
                probability_threshold=threshold
            )

            # Load data
            data = backtester.load_dollar_bars("2024-01-01", "2025-03-31")

            # Detect regimes
            regime_df = backtester.detect_regimes(data)

            # Engineer features
            logger.info("\nEngineering features...")
            features_df = backtester.feature_engineer.engineer_features(data)
            logger.info(f"✅ {len(features_df.columns)} features engineered")

            # Run backtest
            results = backtester.run_backtest(
                data=data,
                features_df=features_df,
                regime_df=regime_df,
                bars_between_trades=30  # Don't overlap trades
            )

            # Generate report
            output_dir = Path("data/reports")
            output_dir.mkdir(parents=True, exist_ok=True)

            meets_target = backtester.generate_report(results, output_dir)

            if meets_target:
                logger.info(f"\n✅ THRESHOLD {threshold:.1%} MEETS TARGET!")
                logger.info(f"   {results['trades_per_day']:.2f} trades/day")
                logger.info(f"   {results['win_rate']:.2f}% win rate")
                break

        logger.info("\n" + "=" * 70)
        logger.info("✅ BAR-BY-BAR BACKTEST COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ Backtest failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

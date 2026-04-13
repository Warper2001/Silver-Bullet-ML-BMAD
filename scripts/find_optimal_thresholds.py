#!/usr/bin/env python3
"""Find optimal thresholds for 1 and 2 trades per day targets."""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Tuple, List

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


class OptimalThresholdFinder:
    """Find thresholds that hit specific trade frequency targets."""

    def __init__(
        self,
        hmm_detector: HMMRegimeDetector,
        hmm_feature_engineer: HMMFeatureEngineer,
        feature_engineer: FeatureEngineer,
        generic_model,
        regime_0_model,
        regime_2_model
    ):
        self.hmm_detector = hmm_detector
        self.hmm_feature_engineer = hmm_feature_engineer
        self.feature_engineer = feature_engineer
        self.generic_model = generic_model
        self.regime_0_model = regime_0_model
        self.regime_2_model = regime_2_model

        # Trading parameters
        self.take_profit_pct = 0.003
        self.stop_loss_pct = 0.002
        self.max_hold_minutes = 30
        self.bars_between_trades = 30

    def load_dollar_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load dollar bar data."""
        logger.info(f"Loading dollar bars from {start_date} to {end_date}...")

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
                    pass

            current = current + pd.DateOffset(months=1)

        combined = pd.concat(dataframes, ignore_index=True)
        combined = combined.sort_values('timestamp')
        combined = combined.loc[
            (combined['timestamp'] >= start_dt) &
            (combined['timestamp'] <= end_dt)
        ]

        logger.info(f"✅ Loaded {len(combined):,} dollar bars")
        return combined

    def detect_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect regimes."""
        logger.info("Detecting regimes...")
        hmm_features = self.hmm_feature_engineer.engineer_features(data)
        regimes = self.hmm_detector.predict(hmm_features)

        regime_df = pd.DataFrame({
            'regime': regimes,
            'regime_name': [self.hmm_detector.metadata.regime_names[int(r)] for r in regimes]
        }, index=data.index)

        return regime_df

    def get_model_prediction(self, bar_idx: int, features_df: pd.DataFrame, model) -> Tuple[float, bool]:
        """Get model prediction."""
        if bar_idx not in features_df.index:
            return 0.0, False

        features = features_df.loc[[bar_idx]]
        expected_features = model.feature_names_in_
        available_features = [f for f in expected_features if f in features.columns]
        X = features[available_features].fillna(0)
        prediction_proba = model.predict_proba(X)[0, 1]

        return float(prediction_proba), True

    def determine_direction(self, data: pd.DataFrame, bar_idx: int) -> int:
        """Determine trade direction."""
        if bar_idx < 5:
            return 1
        recent_closes = data.iloc[bar_idx-5:bar_idx+1]['close']
        momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        return 1 if momentum > 0 else -1

    def simulate_trade(self, entry_bar: pd.Series, data: pd.DataFrame, direction: int) -> Tuple[float, str, int]:
        """Simulate trade with exits."""
        entry_price = entry_bar['close']
        take_profit_price = entry_price * (1 + self.take_profit_pct * direction)
        stop_loss_price = entry_price * (1 - self.stop_loss_pct * direction)

        entry_idx = data.index.get_loc(entry_bar.name)
        max_hold_bars = self.max_hold_minutes // 5

        exit_reason = 'time_stop'
        exit_price = entry_price

        for i in range(1, min(max_hold_bars + 1, len(data) - entry_idx)):
            bar = data.iloc[entry_idx + i]
            current_price = bar['close']

            if direction == 1:
                if bar['high'] >= take_profit_price:
                    exit_reason = 'take_profit'
                    exit_price = take_profit_price
                    break
                if bar['low'] <= stop_loss_price:
                    exit_reason = 'stop_loss'
                    exit_price = stop_loss_price
                    break
            else:
                if bar['low'] <= take_profit_price:
                    exit_reason = 'take_profit'
                    exit_price = take_profit_price
                    break
                if bar['high'] >= stop_loss_price:
                    exit_reason = 'stop_loss'
                    exit_price = stop_loss_price
                    break

            exit_price = current_price

        price_change_pct = (exit_price - entry_price) / entry_price
        pnl_pct = price_change_pct * direction * 100

        return pnl_pct, exit_reason, i * 5

    def backtest_at_threshold(
        self,
        data: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        probability_threshold: float
    ) -> Dict:
        """Run backtest at specific threshold."""
        results = []
        last_trade_idx = -1000
        start_bar = 100
        total_bars = len(data)

        for idx in range(start_bar, total_bars):
            bar_idx = data.index[idx]

            if (idx - last_trade_idx) < self.bars_between_trades:
                continue

            if bar_idx not in regime_df.index:
                continue

            regime = regime_df.loc[bar_idx, 'regime']

            # Select model
            if regime == 0:
                prediction, success = self.get_model_prediction(bar_idx, features_df, self.regime_0_model)
            elif regime == 2:
                prediction, success = self.get_model_prediction(bar_idx, features_df, self.regime_2_model)
            else:
                prediction, success = self.get_model_prediction(bar_idx, features_df, self.generic_model)

            if not success or prediction < probability_threshold:
                continue

            direction = self.determine_direction(data, idx)
            bar = data.iloc[idx]
            pnl_pct, exit_reason, hold_minutes = self.simulate_trade(bar, data, direction)

            results.append({
                'timestamp': bar_idx,
                'prediction': prediction,
                'regime': int(regime),
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'outcome': 'win' if pnl_pct > 0 else 'loss'
            })

            last_trade_idx = idx

        if len(results) == 0:
            return None

        trades_df = pd.DataFrame(results)

        num_trades = len(trades_df)
        win_rate = (len(trades_df[trades_df['pnl_pct'] > 0]) / num_trades * 100)
        total_pnl = trades_df['pnl_pct'].sum()
        avg_trade = trades_df['pnl_pct'].mean()
        std_trade = trades_df['pnl_pct'].std()
        sharpe = (avg_trade / std_trade * np.sqrt(252)) if std_trade > 0 else 0

        wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        losses = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = (wins / losses) if losses > 0 else 0

        cumulative = trades_df['pnl_pct'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max)
        max_drawdown = drawdown.min()

        trading_days = 390
        trades_per_day = num_trades / trading_days

        return {
            'threshold': probability_threshold,
            'num_trades': num_trades,
            'trades_per_day': trades_per_day,
            'trades_per_month': trades_per_day * 21,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown_pct': max_drawdown
        }

    def find_threshold_for_target(
        self,
        data: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        target_trades_per_day: float,
        tolerance: float = 0.2
    ) -> Dict:
        """Find threshold that produces target trade frequency."""
        logger.info(f"\nSearching for threshold to achieve {target_trades_per_day} trades/day...")

        # Binary search for threshold
        low = 0.30
        high = 0.70
        best_result = None
        best_distance = float('inf')

        for iteration in range(15):
            mid = (low + high) / 2
            result = self.backtest_at_threshold(data, features_df, regime_df, mid)

            if result is None:
                low = mid
                continue

            distance = abs(result['trades_per_day'] - target_trades_per_day)

            if distance < best_distance:
                best_distance = distance
                best_result = result
                logger.info(f"  Threshold {mid:.1%}: {result['trades_per_day']:.2f} trades/day (distance: {distance:.2f})")

            # Check if within tolerance
            if distance < tolerance:
                logger.info(f"  ✅ Found threshold within tolerance!")
                break

            # Adjust search range
            if result['trades_per_day'] > target_trades_per_day:
                low = mid
            else:
                high = mid

        return best_result


def main():
    """Find optimal thresholds for 1 and 2 trades per day."""
    logger.info("\n" + "=" * 70)
    logger.info("FINDING OPTIMAL THRESHOLDS FOR 1 AND 2 TRADES/DAY")
    logger.info("=" * 70)

    try:
        # Load models
        logger.info("Loading models...")
        hmm_dir = Path("models/hmm/regime_model")
        hmm_detector = HMMRegimeDetector.load(hmm_dir)

        generic_model = joblib.load(Path("models/xgboost/regime_aware_real_labels/xgboost_generic_real_labels.joblib"))
        regime_0_model = joblib.load(Path("models/xgboost/regime_aware_real_labels/xgboost_regime_0_real_labels.joblib"))
        regime_2_model = joblib.load(Path("models/xgboost/regime_aware_real_labels/xgboost_regime_2_real_labels.joblib"))

        logger.info("✅ Models loaded")

        # Initialize finder
        finder = OptimalThresholdFinder(
            hmm_detector=hmm_detector,
            hmm_feature_engineer=HMMFeatureEngineer(),
            feature_engineer=FeatureEngineer(),
            generic_model=generic_model,
            regime_0_model=regime_0_model,
            regime_2_model=regime_2_model
        )

        # Load data
        data = finder.load_dollar_bars("2024-01-01", "2025-03-31")
        regime_df = finder.detect_regimes(data)

        # Engineer features
        logger.info("Engineering features...")
        features_df = finder.feature_engineer.engineer_features(data)
        logger.info(f"✅ {len(features_df.columns)} features engineered")

        # Find thresholds for targets
        targets = [1.0, 2.0]
        results = []

        for target in targets:
            result = finder.find_threshold_for_target(
                data, features_df, regime_df, target, tolerance=0.15
            )
            if result:
                results.append(result)

        # Generate report
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMAL THRESHOLD RESULTS")
        logger.info("=" * 70)

        for result in results:
            logger.info(f"\nTarget: {result['trades_per_day']:.2f} trades/day")
            logger.info(f"Optimal Threshold: {result['threshold']:.1%}")
            logger.info(f"Actual Trades/Day: {result['trades_per_day']:.2f}")
            logger.info(f"Trades/Month: {result['trades_per_month']:.1f}")
            logger.info(f"Win Rate: {result['win_rate']:.2f}%")
            logger.info(f"Total P&L: {result['total_pnl_pct']:.2f}%")
            logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            logger.info(f"Profit Factor: {result['profit_factor']:.2f}")
            logger.info(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")

        # Save results
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"optimal_thresholds_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("OPTIMAL THRESHOLDS FOR 1 AND 2 TRADES PER DAY\n")
            f.write("=" * 70 + "\n\n")

            for result in results:
                f.write(f"Target: {result['trades_per_day']:.2f} trades/day\n")
                f.write(f"Optimal Threshold: {result['threshold']:.1%}\n\n")
                f.write(f"Performance:\n")
                f.write(f"  Trades/Day: {result['trades_per_day']:.2f}\n")
                f.write(f"  Trades/Month: {result['trades_per_month']:.1f}\n")
                f.write(f"  Win Rate: {result['win_rate']:.2f}%\n")
                f.write(f"  Total P&L: {result['total_pnl_pct']:.2f}%\n")
                f.write(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}\n")
                f.write(f"  Profit Factor: {result['profit_factor']:.2f}\n")
                f.write(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%\n\n")
                f.write("-" * 70 + "\n\n")

        logger.info(f"\n✅ Report saved to {report_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

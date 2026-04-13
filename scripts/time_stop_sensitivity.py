#!/usr/bin/env python3
"""Time stop sensitivity analysis to optimize trade frequency."""

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


class TimeStopSensitivityAnalyzer:
    """Analyze impact of different time stop durations."""

    def __init__(
        self,
        hmm_detector: HMMRegimeDetector,
        hmm_feature_engineer: HMMFeatureEngineer,
        feature_engineer: FeatureEngineer,
        generic_model,
        regime_0_model,
        regime_2_model,
        probability_threshold: float = 0.40
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

        # Test different time stops
        self.time_stops = [5, 10, 15, 20, 25, 30]  # minutes
        self.bars_between_trades_options = [10, 20, 30, 40, 50]  # bars

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

    def simulate_trade(
        self,
        entry_bar: pd.Series,
        data: pd.DataFrame,
        direction: int,
        max_hold_minutes: int
    ) -> Tuple[float, str, int]:
        """Simulate trade with exits."""
        entry_price = entry_bar['close']
        take_profit_price = entry_price * (1 + self.take_profit_pct * direction)
        stop_loss_price = entry_price * (1 - self.stop_loss_pct * direction)

        entry_idx = data.index.get_loc(entry_bar.name)
        max_hold_bars = max_hold_minutes // 5  # 5-minute bars

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

    def backtest_at_time_stop(
        self,
        data: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        max_hold_minutes: int,
        bars_between_trades: int
    ) -> Dict:
        """Run backtest at specific time stop."""
        results = []
        last_trade_idx = -1000
        start_bar = 100
        total_bars = len(data)

        for idx in range(start_bar, total_bars):
            bar_idx = data.index[idx]

            if (idx - last_trade_idx) < bars_between_trades:
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

            if not success or prediction < self.probability_threshold:
                continue

            direction = self.determine_direction(data, idx)
            bar = data.iloc[idx]
            pnl_pct, exit_reason, hold_minutes = self.simulate_trade(
                bar, data, direction, max_hold_minutes
            )

            results.append({
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'hold_minutes': hold_minutes,
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

        # Calculate cycle time
        cycle_minutes = max_hold_minutes + (bars_between_trades * 5)
        max_possible_trades_per_day = (24 * 60) / cycle_minutes

        return {
            'max_hold_minutes': max_hold_minutes,
            'bars_between_trades': bars_between_trades,
            'cycle_minutes': cycle_minutes,
            'max_trades_per_day': max_possible_trades_per_day,
            'num_trades': num_trades,
            'trades_per_day': trades_per_day,
            'trades_per_month': trades_per_day * 21,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown_pct': max_drawdown,
            'exit_tp': len(trades_df[trades_df['exit_reason'] == 'take_profit']),
            'exit_sl': len(trades_df[trades_df['exit_reason'] == 'stop_loss']),
            'exit_time': len(trades_df[trades_df['exit_reason'] == 'time_stop']),
            'avg_hold_minutes': trades_df['hold_minutes'].mean()
        }


def main():
    """Analyze time stop sensitivity."""
    logger.info("\n" + "=" * 70)
    logger.info("TIME STOP SENSITIVITY ANALYSIS")
    logger.info("Optimizing trade frequency vs performance")
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

        # Initialize analyzer
        analyzer = TimeStopSensitivityAnalyzer(
            hmm_detector=hmm_detector,
            hmm_feature_engineer=HMMFeatureEngineer(),
            feature_engineer=FeatureEngineer(),
            generic_model=generic_model,
            regime_0_model=regime_0_model,
            regime_2_model=regime_2_model,
            probability_threshold=0.40
        )

        # Load data
        data = analyzer.load_dollar_bars("2024-01-01", "2025-03-31")
        regime_df = analyzer.detect_regimes(data)

        # Engineer features
        logger.info("Engineering features...")
        features_df = analyzer.feature_engineer.engineer_features(data)
        logger.info(f"✅ {len(features_df.columns)} features engineered")

        # Test combinations
        results = []

        logger.info("\n" + "=" * 70)
        logger.info("TESTING TIME STOP + WAIT TIME COMBINATIONS")
        logger.info("=" * 70)

        for time_stop in [15, 20, 30]:
            for bars_between in [20, 30, 40]:
                logger.info(f"\nTesting: {time_stop}min hold + {bars_between}bars wait...")

                result = analyzer.backtest_at_time_stop(
                    data, features_df, regime_df,
                    max_hold_minutes=time_stop,
                    bars_between_trades=bars_between
                )

                if result:
                    results.append(result)

                    logger.info(f"  Trades/Day: {result['trades_per_day']:.2f}")
                    logger.info(f"  Win Rate: {result['win_rate']:.2f}%")
                    logger.info(f"  Sharpe: {result['sharpe_ratio']:.2f}")
                    logger.info(f"  Cycle: {result['cycle_minutes']}min")

        # Generate report
        logger.info("\n" + "=" * 70)
        logger.info("TIME STOP SENSITIVITY RESULTS")
        logger.info("=" * 70)

        for result in sorted(results, key=lambda x: x['trades_per_day'], reverse=True):
            logger.info(f"\nHold: {result['max_hold_minutes']}min, Wait: {result['bars_between_trades']}bars")
            logger.info(f"  Trades/Day: {result['trades_per_day']:.2f} (max possible: {result['max_trades_per_day']:.1f})")
            logger.info(f"  Win Rate: {result['win_rate']:.2f}%")
            logger.info(f"  Sharpe: {result['sharpe_ratio']:.2f}")
            logger.info(f"  Total P&L: {result['total_pnl_pct']:.2f}%")
            logger.info(f"  Exits: TP:{result['exit_tp']} SL:{result['exit_sl']} Time:{result['exit_time']}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

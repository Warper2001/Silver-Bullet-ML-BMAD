#!/usr/bin/env python3
"""Proper trading backtest with real ML predictions.

This script performs an accurate backtest comparing:
1. Generic XGBoost model with probability threshold
2. Hybrid regime-aware system with probability threshold

Key differences from simplified backtest:
- Uses real ML model predictions
- Filters by probability threshold (e.g., 65%)
- Compares apples-to-apples (both models use same threshold)
- Realistic signal generation based on volatility and volume
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

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


class ProperMLBacktester:
    """Proper backtest with real ML predictions."""

    def __init__(
        self,
        probability_threshold: float = 0.65,
        min_data_bars: int = 100
    ):
        """Initialize backtester.
        
        Args:
            probability_threshold: Minimum prediction probability to take trade
            min_data_bars: Minimum bars for feature engineering
        """
        self.probability_threshold = probability_threshold
        self.min_data_bars = min_data_bars

        logger.info("Loading models...")
        
        # Load HMM detector
        hmm_dir = Path("models/hmm/regime_model")
        self.hmm_detector = HMMRegimeDetector.load(hmm_dir)
        self.hmm_feature_engineer = HMMFeatureEngineer()
        logger.info(f"  ✅ HMM loaded: {self.hmm_detector.n_regimes} regimes")

        # Load ML models
        self.generic_model = joblib.load(
            Path("models/xgboost/regime_aware_real_labels/xgboost_generic_real_labels.joblib")
        )
        self.regime_0_model = joblib.load(
            Path("models/xgboost/regime_aware_real_labels/xgboost_regime_0_real_labels.joblib")
        )
        self.regime_2_model = joblib.load(
            Path("models/xgboost/regime_aware_real_labels/xgboost_regime_2_real_labels.joblib")
        )
        
        logger.info("  ✅ ML models loaded (generic + regime 0 + regime 2)")
        logger.info(f"  Probability threshold: {self.probability_threshold:.2f}")

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

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

        regime_counts = regime_df['regime_name'].value_counts()
        logger.info("Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            logger.info(f"  {regime}: {count:,} bars ({pct:.1f}%)")

        return regime_df

    def generate_realistic_signals(self, data: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic Silver Bullet-like signals.
        
        Uses volatility and volume spikes to simulate potential Silver Bullet setups.
        This is a proxy since we don't have historical pattern detection logs.
        """
        logger.info("\nGenerating realistic Silver Bullet-like signals...")
        
        # Calculate indicators
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Generate signals when:
        # 1. High volatility (potential breakout)
        # 2. Volume spike (institutional activity)
        # 3. Price movement threshold
        
        vol_threshold = data['volatility'].quantile(0.75)  # Top 25% volatility
        vol_ratio = data['volume'] / data['volume_ma']
        
        data['signal_strength'] = (
            (data['volatility'] > vol_threshold).astype(int) +
            (vol_ratio > 1.5).astype(int) +
            (abs(data['returns']) > 0.001).astype(int)
        )
        
        # Only take strong signals (2+ criteria met)
        signals = data[data['signal_strength'] >= 2].copy()
        
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
    ) -> tuple[float, bool]:
        """Get prediction and filter decision from model.
        
        Args:
            signal_row: Signal with timestamp
            features_df: Engineered features
            model: ML model
            model_name: Model name for logging
        
        Returns:
            (prediction_probability, pass_threshold)
        """
        try:
            # Get features at signal time
            signal_time = signal_row.name
            
            if signal_time in features_df.index:
                features = features_df.loc[[signal_time]]
            else:
                # Find closest prior features
                available = features_df.index[features_df.index <= signal_time]
                if len(available) == 0:
                    return 0.5, False
                features = features_df.loc[[available[-1]]]

            # Get feature columns expected by model
            expected_features = model.feature_names_in_
            
            # Filter to available features
            available_features = [f for f in expected_features if f in features.columns]
            
            if len(available_features) == 0:
                logger.warning(f"No matching features for {model_name}")
                return 0.5, False
            
            X = features[available_features].fillna(0)

            # Get prediction
            prediction_proba = model.predict_proba(X)[0, 1]
            passes_threshold = prediction_proba >= self.probability_threshold

            return float(prediction_proba), passes_threshold

        except Exception as e:
            logger.warning(f"{model_name} prediction failed: {e}")
            return 0.5, False

    def simulate_trade_with_exits(
        self,
        signal_row: pd.Series,
        data: pd.DataFrame,
        direction: int
    ) -> dict:
        """Simulate trade with proper exit logic.
        
        Uses triple-barrier exits:
        - Take profit: +0.3% move
        - Stop loss: -0.2% move
        - Time stop: 30 minutes
        """
        entry_price = signal_row['close']
        entry_time = signal_row.name
        
        take_profit_pct = 0.003  # 0.3%
        stop_loss_pct = 0.002    # 0.2%
        max_hold_minutes = 30
        
        take_profit_price = entry_price * (1 + take_profit_pct * direction)
        stop_loss_price = entry_price * (1 - stop_loss_pct * direction)
        
        # Find exit
        curr_loc = data.index.get_loc(entry_time)
        
        for i in range(1, min(max_hold_minutes, len(data) - curr_loc - 1)):
            future_idx = curr_loc + i
            if future_idx >= len(data):
                break
                
            bar = data.iloc[future_idx]
            
            # Check take profit
            if direction == 1:  # Long
                if bar['high'] >= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = 'take_profit'
                    break
                if bar['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'stop_loss'
                    break
            else:  # Short
                if bar['low'] <= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = 'take_profit'
                    break
                if bar['high'] >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'stop_loss'
                    break
        
        else:
            # Time stop
            exit_price = data.iloc[min(curr_loc + max_hold_minutes, len(data) - 1)]['close']
            exit_reason = 'time_stop'
        
        # Calculate P&L
        price_change_pct = (exit_price - entry_price) / entry_price
        pnl_pct = price_change_pct * direction * 100
        
        return {
            'entry_time': entry_time,
            'exit_time': data.index[curr_loc + i] if i < max_hold_minutes else data.index[curr_loc + max_hold_minutes],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': 'long' if direction == 1 else 'short',
            'exit_reason': exit_reason,
            'hold_minutes': i,
            'pnl_pct': pnl_pct,
            'outcome': 'win' if pnl_pct > 0 else 'loss'
        }

    def run_backtest(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-03-31",
        max_signals: int = 1000
    ):
        """Run proper backtest with ML predictions."""
        
        logger.info("\n" + "=" * 70)
        logger.info("PROPER ML BACKTEST WITH REAL PREDICTIONS")
        logger.info("=" * 70)
        
        # Load data
        data = self.load_dollar_bars(start_date, end_date)
        
        # Detect regimes
        regime_df = self.detect_regimes(data)
        
        # Engineer features once for efficiency
        logger.info("\nEngineering features...")
        features_df = self.feature_engineer.engineer_features(data)
        logger.info(f"✅ {features_df.shape[1]} features engineered")
        
        # Generate signals
        signals = self.generate_realistic_signals(data, regime_df)
        
        if max_signals:
            signals = signals.head(max_signals)
            logger.info(f"Limited to {max_signals} signals")
        
        # Run backtests for both models
        results = {
            'generic': self._run_single_backtest(
                signals, data, features_df, regime_df,
                self.generic_model, 'Generic'
            ),
            'hybrid': self._run_single_backtest(
                signals, data, features_df, regime_df,
                None, 'Hybrid'  # Will handle regime selection internally
            )
        }
        
        # Compare results
        comparison = self._compare_results(results['generic'], results['hybrid'])
        
        return {
            'generic': results['generic'],
            'hybrid': results['hybrid'],
            'comparison': comparison,
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'total_signals': len(signals),
                'probability_threshold': self.probability_threshold
            }
        }

    def _run_single_backtest(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        model,
        model_name: str
    ) -> dict:
        """Run backtest for a single model."""
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Running {model_name} Model Backtest")
        logger.info('=' * 70)
        
        trades = []
        passed_threshold = 0
        failed_threshold = 0
        
        for idx, signal in signals.iterrows():
            # Get regime at signal time
            if idx not in regime_df.index:
                continue
            regime = regime_df.loc[idx, 'regime']
            
            # Select model based on approach
            if model_name == 'Generic':
                prediction, passes = self.get_model_prediction(
                    signal, features_df, self.generic_model, 'Generic'
                )
            else:  # Hybrid
                if regime == 0:
                    prediction, passes = self.get_model_prediction(
                        signal, features_df, self.regime_0_model, 'Regime_0'
                    )
                elif regime == 2:
                    prediction, passes = self.get_model_prediction(
                        signal, features_df, self.regime_2_model, 'Regime_2'
                    )
                else:  # Regime 1 - use generic fallback
                    prediction, passes = self.get_model_prediction(
                        signal, features_df, self.generic_model, 'Generic (Regime 1 fallback)'
                    )
            
            # Track threshold filtering
            if passes:
                passed_threshold += 1
            else:
                failed_threshold += 1
                continue  # Skip low probability signals
            
            # Simulate trade
            trade_result = self.simulate_trade_with_exits(
                signal, data, signal['signal_direction']
            )
            
            trade_result['prediction'] = prediction
            trade_result['regime'] = int(regime)
            trades.append(trade_result)
        
        if len(trades) == 0:
            logger.warning(f"No trades passed {self.probability_threshold:.0%} threshold!")
            return {
                'trades': pd.DataFrame(),
                'metrics': {},
                'filtered': {
                    'passed_threshold': passed_threshold,
                    'failed_threshold': failed_threshold,
                    'total_signals': len(signals)
                }
            }
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        metrics = self._calculate_metrics(trades_df)
        metrics['filtered'] = {
            'passed_threshold': passed_threshold,
            'failed_threshold': failed_threshold,
            'total_signals': len(signals),
            'filter_rate': 1 - (passed_threshold / len(signals))
        }
        
        # Log results
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Total signals evaluated: {len(signals)}")
        logger.info(f"  Passed threshold (>{self.probability_threshold:.0%}): {passed_threshold}")
        logger.info(f"  Failed threshold: {failed_threshold}")
        logger.info(f"  Filter rate: {metrics['filtered']['filter_rate']:.1%}")
        logger.info(f"  Trades taken: {metrics['total_trades']}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"  Total P&L: {metrics['total_pnl_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        
        return {
            'trades': trades_df,
            'metrics': metrics,
            'model_name': model_name
        }

    def _calculate_metrics(self, trades_df: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics."""
        
        if len(trades_df) == 0:
            return {}
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = (winning_trades / total_trades * 100)
        
        total_pnl = trades_df['pnl_pct'].sum()
        avg_pnl = trades_df['pnl_pct'].mean()
        std_pnl = trades_df['pnl_pct'].std()
        
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if winning_trades < total_trades else 0
        
        total_wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        total_losses = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        sharpe = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0
        
        # Max drawdown
        cumulative = trades_df['pnl_pct'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Exit analysis
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Per-regime
        regime_metrics = {}
        for regime in sorted(trades_df['regime'].unique()):
            regime_trades = trades_df[trades_df['regime'] == regime]
            regime_win_rate = (len(regime_trades[regime_trades['pnl_pct'] > 0]) / len(regime_trades) * 100)
            regime_pnl = regime_trades['pnl_pct'].sum()
            
            regime_metrics[int(regime)] = {
                'trades': len(regime_trades),
                'win_rate': regime_win_rate,
                'total_pnl_pct': regime_pnl
            }
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'avg_pnl_pct': avg_pnl,
            'std_pnl_pct': std_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'exit_reasons': exit_reasons,
            'regime_metrics': regime_metrics
        }

    def _compare_results(self, generic: dict, hybrid: dict) -> dict:
        """Compare generic vs hybrid results."""
        
        if not generic['metrics'] or not hybrid['metrics']:
            return {}
        
        g = generic['metrics']
        h = hybrid['metrics']
        
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON: HYBRID VS GENERIC")
        logger.info("=" * 70)
        
        logger.info(f"  Trade Count: {h['total_trades']} vs {g['total_trades']} ({h['total_trades'] - g['total_trades']:+d})")
        logger.info(f"  Win Rate: {h['win_rate']:.2f}% vs {g['win_rate']:.2f}% ({h['win_rate'] - g['win_rate']:+.2f}%)")
        logger.info(f"  Total P&L: {h['total_pnl_pct']:+.2f}% vs {g['total_pnl_pct']:+.2f}% ({h['total_pnl_pct'] - g['total_pnl_pct']:+.2f}%)")
        logger.info(f"  Sharpe Ratio: {h['sharpe_ratio']:.2f} vs {g['sharpe_ratio']:.2f} ({h['sharpe_ratio'] - g['sharpe_ratio']:+.2f})")
        
        return {
            'trade_count_diff': h['total_trades'] - g['total_trades'],
            'win_rate_diff_pct': h['win_rate'] - g['win_rate'],
            'pnl_diff_pct': h['total_pnl_pct'] - g['total_pnl_pct'],
            'sharpe_diff': h['sharpe_ratio'] - g['sharpe_ratio']
        }

    def generate_report(self, results: dict, output_dir: Path = Path("data/reports")):
        """Generate comprehensive report."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Text report
        report_path = output_dir / f"proper_ml_backtest_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PROPER ML BACKTEST - GENERIC VS HYBRID REGIME-AWARE\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: {results['metadata']['start_date']} to {results['metadata']['end_date']}\n")
            f.write(f"Probability Threshold: {results['metadata']['probability_threshold']:.2%}\n")
            f.write(f"Total Signals: {results['metadata']['total_signals']}\n\n")
            
            # Generic
            f.write("-" * 70 + "\n")
            f.write("GENERIC MODEL RESULTS\n")
            f.write("-" * 70 + "\n")
            self._write_model_results(f, results['generic'])
            
            # Hybrid
            f.write("\n" + "-" * 70 + "\n")
            f.write("HYBRID REGIME-AWARE RESULTS\n")
            f.write("-" * 70 + "\n")
            self._write_model_results(f, results['hybrid'])
            
            # Comparison
            f.write("\n" + "-" * 70 + "\n")
            f.write("COMPARISON\n")
            f.write("-" * 70 + "\n")
            self._write_comparison(f, results['comparison'])
        
        # Save trades
        for model_name in ['generic', 'hybrid']:
            if len(results[model_name]['trades']) > 0:
                trades_path = output_dir / f"{model_name}_trades_{timestamp}.csv"
                results[model_name]['trades'].to_csv(trades_path, index=False)
        
        logger.info(f"\n✅ Report saved to {report_path}")
        logger.info(f"✅ Trades saved to {output_dir}/{model_name}_trades_{timestamp}.csv")

    def _write_model_results(self, f, model_results: dict):
        """Write model results to report."""
        
        metrics = model_results['metrics']
        
        if not metrics:
            f.write("No trades generated (all filtered by threshold)\n")
            return
        
        f.write(f"Signals Evaluated: {metrics['filtered']['total_signals']}\n")
        f.write(f"Passed Threshold: {metrics['filtered']['passed_threshold']}\n")
        f.write(f"Failed Threshold: {metrics['filtered']['failed_threshold']}\n")
        f.write(f"Filter Rate: {metrics['filtered']['filter_rate']:.1%}\n")
        f.write(f"Trades Taken: {metrics['total_trades']}\n\n")
        
        f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"Total P&L: {metrics['total_pnl_pct']:+.2f}%\n")
        f.write(f"Avg Trade: {metrics['avg_pnl_pct']:.3f}%\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n\n")
        
        f.write("Exit Reasons:\n")
        for reason, count in sorted(metrics['exit_reasons'].items(), key=lambda x: -x[1]):
            f.write(f"  {reason}: {count}\n")
        
        f.write("\nPer-Regime Performance:\n")
        for regime, rm in sorted(metrics['regime_metrics'].items()):
            regime_name = self.hmm_detector.metadata.regime_names[int(regime)]
            f.write(f"  Regime {regime} ({regime_name}):\n")
            f.write(f"    Trades: {rm['trades']}\n")
            f.write(f"    Win Rate: {rm['win_rate']:.2f}%\n")
            f.write(f"    P&L: {rm['total_pnl_pct']:+.2f}%\n")

    def _write_comparison(self, f, comparison: dict):
        """Write comparison to report."""
        
        if not comparison:
            f.write("No comparison available\n")
            return
        
        f.write(f"Trade Count Difference: {comparison['trade_count_diff']:+d}\n")
        f.write(f"Win Rate Difference: {comparison['win_rate_diff_pct']:+.2f}%\n")
        f.write(f"Total P&L Difference: {comparison['pnl_diff_pct']:+.2f}%\n")
        f.write(f"Sharpe Ratio Difference: {comparison['sharpe_diff']:+.2f}\n")
        
        if comparison['pnl_diff_pct'] > 0:
            f.write("\n✅ HYBRID SYSTEM OUTPERFORMS GENERIC\n")
        elif comparison['pnl_diff_pct'] < 0:
            f.write("\n⚠️ HYBRID SYSTEM UNDERPERFORMS GENERIC\n")
        else:
            f.write("\n= HYBRID AND GENERIC PERFORM SIMILAR\n")


def main():
    """Execute proper backtest."""
    logger.info("\n" * 70)
    logger.info("PROPER ML BACKTEST WITH REAL PREDICTIONS")
    logger.info("This compares ACTUAL ML model performance, not just accuracy")
    logger.info("=" * 70)
    
    try:
        # Initialize
        backtester = ProperMLBacktester(
            probability_threshold=0.65,
            min_data_bars=100
        )
        
        # Run backtest
        results = backtester.run_backtest(
            start_date="2024-01-01",
            end_date="2025-03-31",
            max_signals=1000  # Increased for better sample size
        )
        
        # Generate report
        backtester.generate_report(results)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ PROPER BACKTEST COMPLETE")
        logger.info("=" * 70)
        
        logger.info("\nKEY FINDINGS:")
        if results['comparison']:
            comp = results['comparison']
            logger.info(f"  • P&L Difference: {comp['pnl_diff_pct']:+.2f}%")
            logger.info(f"  • Win Rate Difference: {comp['win_rate_diff_pct']:+.2f}%")
            logger.info(f"  • Sharpe Difference: {comp['sharpe_diff']:+.2f}")
        
        logger.info("\nIMPORTANT:")
        logger.info("  • This backtest uses REAL ML model predictions")
        logger.info("  • Filters by 65% probability threshold")
        logger.info("  • Uses proper triple-barrier exit logic")
        logger.info("  • Compares actual trading performance, not just accuracy")
        
        logger.info("\nNEXT STEPS:")
        logger.info("  1. Review detailed report in data/reports/")
        logger.info("  2. Analyze trade-by-trade results")
        logger.info("  3. Validate expected +5.81% improvement")
        logger.info("  4. If results satisfactory, deploy to paper trading")
        
    except Exception as e:
        logger.error(f"\n❌ Backtest failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

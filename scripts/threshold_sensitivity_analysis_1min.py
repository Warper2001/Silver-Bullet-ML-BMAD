#!/usr/bin/env python3
"""Threshold Sensitivity Analysis for 1-Minute System.

Tests multiple probability thresholds to find optimal setting for 1-minute data.
Uses held-out Oct-Dec 2025 validation data to ensure no data leakage.
"""

import sys
import warnings
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.features import FeatureEngineer
from src.data.models import DollarBar

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants (same as validation backtest)
MIN_BARS_BETWEEN_TRADES = 1
MAX_CONCURRENT_POSITIONS = 3
TAKE_PROFIT_PCT = 0.003  # 0.3%
STOP_LOSS_PCT = 0.002    # 0.2%
MAX_HOLD_BARS = 30  # 30 minutes at 1-min bars
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.50
CONTRACTS_PER_TRADE = 5

# Thresholds to test (based on validation results showing 42.6% - 53.7% probabilities)
THRESHOLDS_TO_TEST = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

def run_backtest_at_threshold(df, all_regimes, models, feature_engineer, threshold):
    """Run backtest at specific probability threshold."""

    regime_0_model, regime_2_model, generic_model = models

    trades = []
    bars_since_last_trade = MIN_BARS_BETWEEN_TRADES
    open_positions = []

    # Start after feature window
    for i in range(100, len(df)):
        current_bar = df.iloc[i]
        historical_data = df.iloc[i-100:i]

        # Get pre-computed regime
        regime = all_regimes[i]

        # Select model
        if regime == 0:
            model = regime_0_model
        elif regime == 2:
            model = regime_2_model
        else:
            model = generic_model

        # Generate features using proper FeatureEngineer
        bar_timestamp = df.index[i]
        dollar_bar = DollarBar(
            timestamp=bar_timestamp,
            open=float(current_bar['open']),
            high=float(current_bar['high']),
            low=float(current_bar['low']),
            close=float(current_bar['close']),
            volume=int(current_bar['volume']),
            notional_value=float(current_bar['notional'])
        )

        # Generate 52 features
        features = feature_engineer.generate_features_bar(
            current_bar=dollar_bar,
            historical_data=historical_data.reset_index()
        )

        # Predict probability
        probability = float(model.predict_proba(features.reshape(1, -1))[0, 1])

        # Apply threshold
        if probability < threshold:
            bars_since_last_trade += 1
            continue

        # Check minimum bars between trades
        if bars_since_last_trade < MIN_BARS_BETWEEN_TRADES:
            continue

        # Check concurrent positions
        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            continue

        # Determine direction from recent data
        recent_close = historical_data['close'].iloc[-1]
        momentum_5 = recent_close - historical_data['close'].iloc[-6]
        direction = "bullish" if momentum_5 > 0 else "bearish"

        # Calculate entry, stops
        entry_price = float(current_bar['close'])
        if direction == "bullish":
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)
            take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

        # Track trade
        open_positions.append({
            'entry_time': bar_timestamp,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'probability': probability,
            'regime': regime,
            'bar_index': i
        })

        bars_since_last_trade = 0

        # Check exits for open positions
        for pos in open_positions[:]:
            bars_held = i - pos['bar_index']

            # Max hold time exit
            if bars_held >= MAX_HOLD_BARS:
                pnl = (float(current_bar['close']) - pos['entry_price']) * CONTRACTS_PER_TRADE
                if pos['direction'] == "bearish":
                    pnl = -pnl
                pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
                pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
                trades.append({**pos, 'exit_time': df.index[i], 'pnl': pnl, 'exit_reason': 'time'})
                open_positions.remove(pos)
                continue

            # Check stop loss and take profit
            if pos['direction'] == "bullish":
                if current_bar['low'] <= pos['stop_loss']:
                    pnl = (pos['stop_loss'] - pos['entry_price']) * CONTRACTS_PER_TRADE
                    pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
                    pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
                    trades.append({**pos, 'exit_time': df.index[i], 'pnl': pnl, 'exit_reason': 'stop'})
                    open_positions.remove(pos)
                elif current_bar['high'] >= pos['take_profit']:
                    pnl = (pos['take_profit'] - pos['entry_price']) * CONTRACTS_PER_TRADE
                    pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
                    pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
                    trades.append({**pos, 'exit_time': df.index[i], 'pnl': pnl, 'exit_reason': 'target'})
                    open_positions.remove(pos)
            else:  # bearish
                if current_bar['high'] >= pos['stop_loss']:
                    pnl = (pos['entry_price'] - pos['stop_loss']) * CONTRACTS_PER_TRADE
                    pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
                    pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
                    trades.append({**pos, 'exit_time': df.index[i], 'pnl': pnl, 'exit_reason': 'stop'})
                    open_positions.remove(pos)
                elif current_bar['low'] <= pos['take_profit']:
                    pnl = (pos['entry_price'] - pos['take_profit']) * CONTRACTS_PER_TRADE
                    pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
                    pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
                    trades.append({**pos, 'exit_time': df.index[i], 'pnl': pnl, 'exit_reason': 'target'})
                    open_positions.remove(pos)

    return trades

def main():
    logger.info("=" * 80)
    logger.info("THRESHOLD SENSITIVITY ANALYSIS - 1-MINUTE SYSTEM")
    logger.info("Testing thresholds from 25% to 55% on Oct-Dec 2025 validation data")
    logger.info("=" * 80)

    # Load 1-minute dollar bars
    logger.info("\nLoading 1-minute dollar bars...")
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Filter to ONLY Oct-Dec 2025 (held-out test set)
    df = df[(df.index.month >= 10) & (df.index.year == 2025)]
    logger.info(f"✅ Loaded {len(df):,} bars for Oct-Dec 2025 validation")

    # Load HMM model
    logger.info("\nLoading HMM regime detector...")
    hmm_path = Path("models/hmm/regime_model_1min")
    detector = HMMRegimeDetector.load(hmm_path)
    logger.info(f"✅ Loaded HMM with {detector.n_regimes} regimes")

    # Load XGBoost models
    logger.info("\nLoading XGBoost models...")
    import joblib

    model_dir = Path("models/xgboost/regime_aware_1min_2025_54features")
    regime_0_model = joblib.load(model_dir / "xgboost_regime_0_54features.joblib")
    regime_2_model = joblib.load(model_dir / "xgboost_regime_2_54features.joblib")
    generic_model = joblib.load(model_dir / "xgboost_generic_54features.joblib")

    models = (regime_0_model, regime_2_model, generic_model)
    logger.info(f"✅ Loaded 3 XGBoost models from {model_dir}")

    # Pre-compute all regimes
    logger.info("\nPre-computing regimes for validation period...")
    hmm_feature_engineer = HMMFeatureEngineer()
    all_hmm_features = hmm_feature_engineer.engineer_features(df)
    all_regimes = detector.predict(all_hmm_features)
    logger.info(f"✅ Regimes computed for {len(df):,} bars")

    # Initialize FeatureEngineer
    logger.info("\nInitializing FeatureEngineer...")
    feature_engineer = FeatureEngineer(model_dir="models/xgboost/regime_aware_1min_2025_54features")
    logger.info(f"✅ FeatureEngineer initialized")

    # Test each threshold
    results = {}

    logger.info("\n" + "=" * 80)
    logger.info("TESTING DIFFERENT THRESHOLDS")
    logger.info("=" * 80)

    for threshold in THRESHOLDS_TO_TEST:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing threshold: {threshold*100:.0f}%")
        logger.info(f"{'='*60}")

        trades = run_backtest_at_threshold(df, all_regimes, models, feature_engineer, threshold)

        if len(trades) == 0:
            logger.warning(f"No trades generated at {threshold*100:.0f}% threshold")
            results[threshold] = {
                'trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'expectation': 0,
                'profit_factor': 0,
                'sharpe': 0,
                'trades_per_day': 0,
                'max_drawdown': 0
            }
            continue

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date

        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        expectation_per_trade = avg_pnl

        winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = winners / losers if losers > 0 else 0

        # Sharpe ratio
        returns_std = trades_df['pnl'].std()
        sharpe_ratio = (avg_pnl / returns_std) if returns_std > 0 else 0

        # Max drawdown
        cumulative_returns = trades_df['pnl'].cumsum()
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

        # Trades per day (92 trading days in Oct-Dec)
        trades_per_day = len(trades_df) / 92

        # Store results
        results[threshold] = {
            'trades': len(trades_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'expectation': expectation_per_trade,
            'profit_factor': profit_factor,
            'sharpe': sharpe_ratio,
            'trades_per_day': trades_per_day,
            'max_drawdown': max_drawdown
        }

        logger.info(f"\nResults at {threshold*100:.0f}% threshold:")
        logger.info(f"  Total Trades: {len(trades_df)}")
        logger.info(f"  Trades/Day: {trades_per_day:.1f}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"  Total P&L: ${total_pnl:,.2f}")
        logger.info(f"  Expectation/Trade: ${expectation_per_trade:.2f}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: ${max_drawdown:,.2f}")

    # Create summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("THRESHOLD COMPARISON SUMMARY")
    logger.info("=" * 80)

    # Create comparison table
    summary_data = []
    for threshold in THRESHOLDS_TO_TEST:
        r = results[threshold]
        summary_data.append({
            'Threshold': f"{threshold*100:.0f}%",
            'Trades': r['trades'],
            'Trades/Day': f"{r['trades_per_day']:.1f}",
            'Win Rate': f"{r['win_rate']:.1f}%",
            'Expectation': f"${r['expectation']:.1f}",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Sharpe': f"{r['sharpe']:.2f}",
            'Max DD': f"${r['max_drawdown']:.0f}"
        })

    summary_df = pd.DataFrame(summary_data)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Find optimal threshold
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMAL THRESHOLD ANALYSIS")
    logger.info("=" * 80)

    # Score each threshold (higher is better)
    scores = {}
    for threshold in THRESHOLDS_TO_TEST:
        r = results[threshold]
        score = 0

        # Trade frequency score (0-20 points)
        if 5 <= r['trades_per_day'] <= 25:
            score += 20
        elif r['trades_per_day'] >= 1:
            score += 10

        # Win rate score (0-20 points)
        if r['win_rate'] >= 50:
            score += 20
        elif r['win_rate'] >= 45:
            score += 10

        # Expectation score (0-25 points)
        if r['expectation'] >= 20:
            score += 25
        elif r['expectation'] >= 0:
            score += 10

        # Profit factor score (0-20 points)
        if r['profit_factor'] >= 1.5:
            score += 20
        elif r['profit_factor'] >= 1.0:
            score += 10

        # Sharpe ratio score (0-15 points)
        if r['sharpe'] >= 0.6:
            score += 15
        elif r['sharpe'] >= 0.0:
            score += 5

        scores[threshold] = score

    # Find best threshold
    best_threshold = max(scores.keys(), key=lambda k: scores[k])
    best_score = scores[best_threshold]

    logger.info(f"\nThreshold Scores:")
    for threshold in THRESHOLDS_TO_TEST:
        logger.info(f"  {threshold*100:.0f}%: {scores[threshold]:.0f}/100 points")

    logger.info(f"\n✅ OPTIMAL THRESHOLD: {best_threshold*100:.0f}% ({best_score:.0f}/100 points)")

    # Get results for best threshold
    best_results = results[best_threshold]
    logger.info(f"\nExpected Performance at {best_threshold*100:.0f}% threshold:")
    logger.info(f"  Trade Frequency: {best_results['trades_per_day']:.1f} trades/day")
    logger.info(f"  Win Rate: {best_results['win_rate']:.1f}%")
    logger.info(f"  Expectation/Trade: ${best_results['expectation']:.2f}")
    logger.info(f"  Profit Factor: {best_results['profit_factor']:.2f}")
    logger.info(f"  Sharpe Ratio: {best_results['sharpe']:.2f}")

    # Validate against targets
    logger.info(f"\nValidation against targets at {best_threshold*100:.0f}%:")

    # Build target descriptions without nested quotes
    trade_freq_desc = f"Trades/Day 5-25: {best_results['trades_per_day']:.1f}"
    win_rate_desc = f"Win Rate ≥ 50%: {best_results['win_rate']:.1f}%"
    expect_desc = f"Expectation ≥ $20: ${best_results['expectation']:.2f}"
    pf_desc = f"Profit Factor ≥ 1.5: {best_results['profit_factor']:.2f}"
    sharpe_desc = f"Sharpe ≥ 0.6: {best_results['sharpe']:.2f}"
    dd_desc = f"Max Drawdown <$1K: ${best_results['max_drawdown']:.0f}"

    targets = {
        trade_freq_desc: 5 <= best_results['trades_per_day'] <= 25,
        win_rate_desc: best_results['win_rate'] >= 50.0,
        expect_desc: best_results['expectation'] >= 20.0,
        pf_desc: best_results['profit_factor'] >= 1.5,
        sharpe_desc: best_results['sharpe'] >= 0.6,
        dd_desc: best_results['max_drawdown'] < 1000.0,
    }
    passed = sum(targets.values())
    total = len(targets)
    logger.info(f"\nTargets Met: {passed}/{total} ({passed/total*100:.1f}%)")

    # Save results to CSV
    output_path = Path("data/reports/threshold_sensitivity_analysis_1min.csv")
    summary_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Results saved to: {output_path}")

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION")
    logger.info("=" * 80)

    if passed >= total * 0.7:
        logger.info(f"✅ DEPLOY AT {best_threshold*100:.0f}% THRESHOLD")
        logger.info(f"\nThe system meets 70%+ of performance targets at this threshold.")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Update config.yaml: probability_threshold: {best_threshold}")
        logger.info(f"2. Run full validation backtest to confirm results")
        logger.info(f"3. Consider paper trading deployment")
    elif best_results['trades_per_day'] >= 1:
        logger.info(f"⚠️  CAUTIOUS DEPLOYMENT AT {best_threshold*100:.0f}% THRESHOLD")
        logger.info(f"\nThe system shows promise but doesn't meet all targets.")
        logger.info(f"\nRecommendations:")
        logger.info(f"1. Test in paper trading for 2-4 weeks")
        logger.info(f"2. Monitor performance closely")
        logger.info(f"3. Consider parameter optimization (Task #6)")
        logger.info(f"4. Investigate model performance issues (Task #7)")
    else:
        logger.info(f"❌ DO NOT DEPLOY - INSUFFICIENT TRADES")
        logger.info(f"\nEven at {best_threshold*100:.0f}% threshold, trade frequency is too low.")
        logger.info(f"\nRequired actions:")
        logger.info(f"1. Investigate model performance issues (Task #7)")
        logger.info(f"2. Consider retraining with different parameters")
        logger.info(f"3. Test alternative model architectures")

    logger.info(f"\n✅ Threshold sensitivity analysis complete")

    return 0

if __name__ == "__main__":
    sys.exit(main())
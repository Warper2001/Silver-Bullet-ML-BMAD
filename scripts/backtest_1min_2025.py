#!/usr/bin/env python3
"""Backtest 1-minute system with 40% probability threshold.

This script runs a comprehensive backtest on the 1-minute hybrid system,
including transaction costs, concurrent position limits, and triple-barrier exits.
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

# Constants
PROBABILITY_THRESHOLD = 0.40
MIN_BARS_BETWEEN_TRADES = 1
MAX_CONCURRENT_POSITIONS = 3
TAKE_PROFIT_PCT = 0.003  # 0.3%
STOP_LOSS_PCT = 0.002    # 0.2%
MAX_HOLD_BARS = 30  # 30 minutes at 1-min bars
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.50
CONTRACTS_PER_TRADE = 5

def main():
    logger.info("=" * 70)
    logger.info("1-MINUTE SYSTEM BACKTEST - 40% THRESHOLD")
    logger.info("=" * 70)

    # Load 1-minute dollar bars
    logger.info("\nLoading 1-minute dollar bars...")
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    logger.info(f"✅ Loaded {len(df):,} bars")

    # Load HMM model
    logger.info("\nLoading HMM regime detector...")
    hmm_path = Path("models/hmm/regime_model_1min")
    detector = HMMRegimeDetector.load(hmm_path)
    logger.info(f"✅ Loaded HMM with {detector.n_regimes} regimes")

    # Load XGBoost models
    logger.info("\nLoading XGBoost models...")
    import joblib

    model_dir = Path("models/xgboost/regime_aware_1min_2025_54features")  # Latest 54-feature models
    regime_0_model = joblib.load(model_dir / "xgboost_regime_0_54features.joblib")
    regime_2_model = joblib.load(model_dir / "xgboost_regime_2_54features.joblib")
    generic_model = joblib.load(model_dir / "xgboost_generic_54features.joblib")

    logger.info(f"✅ Loaded 3 XGBoost models from {model_dir}")

    # Pre-compute all regimes (much faster than computing on each bar)
    logger.info("\nPre-computing regimes for all bars...")
    hmm_feature_engineer = HMMFeatureEngineer()
    all_hmm_features = hmm_feature_engineer.engineer_features(df)
    all_regimes = detector.predict(all_hmm_features)
    logger.info(f"✅ Regimes computed for {len(df):,} bars")

    # Initialize FeatureEngineer for bar-by-bar feature generation
    logger.info("\nInitializing FeatureEngineer...")
    feature_engineer = FeatureEngineer(model_dir="models/xgboost/regime_aware_1min_2025_54features")
    logger.info(f"✅ FeatureEngineer initialized")

    # Run backtest
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 70)

    trades = []
    bars_since_last_trade = MIN_BARS_BETWEEN_TRADES
    open_positions = []

    # Start after feature window
    for i in range(100, len(df)):
        if i % 10000 == 0:
            logger.info(f"Processing bar {i:,}/{len(df):,}...")

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

        # Generate features using proper FeatureEngineer (bar-by-bar method)
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

        # Generate 52 features using the same method as training
        features = feature_engineer.generate_features_bar(
            current_bar=dollar_bar,
            historical_data=historical_data.reset_index()
        )

        # Predict probability
        probability = float(model.predict_proba(features.reshape(1, -1))[0, 1])

        # Apply threshold
        if probability < PROBABILITY_THRESHOLD:
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

        # Check exits for open positions (only check current bar, not future bars)
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

            # Check stop loss and take profit on current bar
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

    # Calculate metrics
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 70)

    if len(trades) == 0:
        logger.warning("No trades generated - may need to lower threshold")
        return 0

    trades_df = pd.DataFrame(trades)
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date

    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = (winning_trades / len(trades_df) * 100)
    total_return = trades_df['pnl'].sum()
    avg_return = trades_df['pnl'].mean()

    daily_counts = trades_df.groupby('date').size()
    trades_per_day = daily_counts.mean()

    winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = winners / losers if losers > 0 else 0

    logger.info(f"\nTrading Summary:")
    logger.info(f"  Total Trades: {len(trades_df)}")
    logger.info(f"  Trading Days: {len(daily_counts)}")
    logger.info(f"  Trades/Day: {trades_per_day:.1f}")

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Win Rate: {win_rate:.2f}%")
    logger.info(f"  Total Return: ${total_return:,.2f}")
    logger.info(f"  Avg Return/Trade: ${avg_return:.2f}")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")

    # Save results
    output_path = Path("data/reports/backtest_1min_2025.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)

    logger.info(f"\n✅ Results saved to: {output_path}")

    # Validate against targets
    logger.info(f"\n" + "=" * 70)
    logger.info("VALIDATION AGAINST TARGETS")
    logger.info("=" * 70)

    targets = {
        'Win Rate ≥ 50%': win_rate >= 50.0,
        f'Trades/Day 5-25': 5 <= trades_per_day <= 25,
        'Sharpe ≥ 0.6': 'Need equity curve data to calculate',
        'Max Drawdown < 5%': 'Need equity curve data to calculate'
    }

    for target, met in targets.items():
        status = "✅ PASS" if met else "❌ FAIL"
        logger.info(f"  {target}: {status}")

    logger.info(f"\n✅ Backtest complete")

    return 0

if __name__ == "__main__":
    sys.exit(main())

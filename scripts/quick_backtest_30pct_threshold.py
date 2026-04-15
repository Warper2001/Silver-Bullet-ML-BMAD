#!/usr/bin/env python3
"""Quick backtest with 30% threshold to test trade frequency."""

import sys
import warnings
from pathlib import Path
import logging
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants - LOWER THRESHOLD FOR MORE TRADES
PROBABILITY_THRESHOLD = 0.30  # 30% instead of 40%
MIN_BARS_BETWEEN_TRADES = 1
MAX_CONCURRENT_POSITIONS = 3
TAKE_PROFIT_PCT = 0.003
STOP_LOSS_PCT = 0.002
MAX_HOLD_BARS = 30
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.50
CONTRACTS_PER_TRADE = 5

def main():
    logger.info("=" * 70)
    logger.info("QUICK BACKTEST - 30% THRESHOLD (vs 40% baseline)")
    logger.info("=" * 70)

    # Load data
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Use validation period for quick test
    df = df.loc[df.index >= '2025-10-01']
    logger.info(f"✅ Loaded {len(df):,} bars (Oct-Dec 2025)")

    # Load models
    logger.info("\nLoading models...")
    hmm_path = Path("models/hmm/regime_model_1min")
    detector = HMMRegimeDetector.load(hmm_path)
    logger.info(f"✅ Loaded HMM with {detector.n_regimes} regimes")

    import joblib
    model_dir = Path("models/xgboost/regime_aware_1min_2025")
    regime_0_model = joblib.load(model_dir / "xgboost_regime_0_real_labels.joblib")
    regime_2_model = joblib.load(model_dir / "xgboost_regime_2_real_labels.joblib")
    generic_model = joblib.load(model_dir / "xgboost_generic_real_labels.joblib")
    logger.info(f"✅ Loaded 3 XGBoost models")

    # Pre-compute regimes
    logger.info("\nPre-computing regimes...")
    hmm_feature_engineer = HMMFeatureEngineer()
    all_hmm_features = hmm_feature_engineer.engineer_features(df)
    all_regimes = detector.predict(all_hmm_features)
    logger.info(f"✅ Regimes computed")

    # Run backtest
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING BACKTEST - 30% THRESHOLD")
    logger.info("=" * 70)

    trades = []
    bars_since_last_trade = MIN_BARS_BETWEEN_TRADES
    open_positions = []

    def calculate_features(df, i):
        current_bar = df.iloc[i]
        momentum_5 = (current_bar['close'] - df['close'].iloc[i-5]) / df['close'].iloc[i-5] if i >= 5 else 0
        return np.array([
            float(current_bar['open']),
            float(current_bar['high']),
            float(current_bar['low']),
            float(current_bar['close']),
            int(current_bar['volume']),
            float(current_bar['notional']),
            float(momentum_5)
        ])

    for i in range(100, len(df)):
        if i % 5000 == 0:
            logger.info(f"Processing bar {i:,}/{len(df):,}...")

        current_bar = df.iloc[i]
        historical_data = df.iloc[i-100:i]

        # Get regime
        regime = all_regimes[i]

        # Select model
        if regime == 0:
            model = regime_0_model
        elif regime == 2:
            model = regime_2_model
        else:
            model = generic_model

        # Generate features and predict
        features = calculate_features(df, i)
        probability = float(model.predict_proba(features.reshape(1, -1))[0, 1])

        # Apply threshold - 30% instead of 40%
        if probability < PROBABILITY_THRESHOLD:
            bars_since_last_trade += 1
            continue

        if bars_since_last_trade < MIN_BARS_BETWEEN_TRADES:
            continue

        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            continue

        # Determine direction
        recent_close = historical_data['close'].iloc[-1]
        momentum_5 = recent_close - historical_data['close'].iloc[-6]
        direction = "bullish" if momentum_5 > 0 else "bearish"

        # Calculate entry and exits
        entry_price = float(current_bar['close'])
        if direction == "bullish":
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)
            take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

        # Track trade
        open_positions.append({
            'entry_time': df.index[i],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'probability': probability,
            'regime': regime,
            'bar_index': i
        })

        bars_since_last_trade = 0

        # Check exits
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

    # Calculate metrics
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST RESULTS - 30% THRESHOLD")
    logger.info("=" * 70)

    if len(trades) == 0:
        logger.warning("No trades generated")
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

    # Compare with 40% baseline
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON: 30% vs 40% Threshold")
    logger.info(f"{'=' * 70}")
    logger.info(f"  40% threshold: 3 trades, 66.67% win rate, $44.38 return")
    logger.info(f"  30% threshold: {len(trades_df)} trades, {win_rate:.2f}% win rate, ${total_return:,.2f} return")
    logger.info(f"  Trade increase: {len(trades_df) - 3} more trades ({(len(trades_df)/3 - 1)*100:.0f}% increase)")

    return 0

if __name__ == "__main__":
    sys.exit(main())

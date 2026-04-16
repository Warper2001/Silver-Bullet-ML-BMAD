#!/usr/bin/env python3
"""Complete Tier1 backtest with all trading constraints.

This script runs a comprehensive backtest using Tier1 models on Oct-Dec 2025 data
with all real-world trading constraints including:
- MIN_BARS_BETWEEN_TRADES = 1
- MAX_CONCURRENT_POSITIONS = 3
- Triple-barrier exits (TP: 0.3%, SL: 0.2%, Time: 30min)
- Transaction costs ($2.50/contract + 0.25 tick slippage)
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
from src.ml.tier1_features import Tier1FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants (matching config.yaml)
PROBABILITY_THRESHOLD = 0.40
MIN_BARS_BETWEEN_TRADES = 1
MAX_CONCURRENT_POSITIONS = 3
TAKE_PROFIT_PCT = 0.003  # 0.3%
STOP_LOSS_PCT = 0.002    # 0.2%
MAX_HOLD_BARS = 30
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.50
CONTRACTS_PER_TRADE = 5

def main():
    logger.info("=" * 80)
    logger.info("COMPLETE TIER1 BACKTEST WITH ALL CONSTRAINTS (Oct-Dec 2025)")
    logger.info("Tier1 models (16 order flow features) with full trading simulation")
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

    # Load Tier1 XGBoost models
    logger.info("\nLoading Tier1 XGBoost models...")
    import joblib

    model_dir = Path("models/xgboost/regime_aware_tier1")
    regime_0_model = joblib.load(model_dir / "xgboost_regime_0_tier1.joblib")
    regime_1_model = joblib.load(model_dir / "xgboost_regime_1_tier1.joblib")
    regime_2_model = joblib.load(model_dir / "xgboost_regime_2_tier1.joblib")

    logger.info(f"✅ Loaded 3 Tier1 XGBoost models from {model_dir}")

    # Pre-compute all regimes
    logger.info("\nPre-computing regimes for validation period...")
    hmm_feature_engineer = HMMFeatureEngineer()
    all_hmm_features = hmm_feature_engineer.engineer_features(df)
    all_regimes = detector.predict(all_hmm_features)
    logger.info(f"✅ Regimes computed for {len(df):,} bars")

    # Generate Tier1 features
    logger.info("\nGenerating Tier1 features...")
    tier1_engineer = Tier1FeatureEngineer()
    df_with_features = tier1_engineer.generate_features(df)

    # Use correct 16 features (exclude realized_vol_5 to match training data)
    feature_names = [f for f in tier1_engineer.feature_names if f != 'realized_vol_5']
    logger.info(f"✅ Using {len(feature_names)} Tier1 features")

    # Run backtest with all constraints
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING COMPLETE BACKTEST")
    logger.info("=" * 80)
    logger.info("Constraints:")
    logger.info(f"  MIN_BARS_BETWEEN_TRADES: {MIN_BARS_BETWEEN_TRADES}")
    logger.info(f"  MAX_CONCURRENT_POSITIONS: {MAX_CONCURRENT_POSITIONS}")
    logger.info(f"  MAX_HOLD_BARS: {MAX_HOLD_BARS} (30 minutes)")
    logger.info(f"  Transaction costs: ${COMMISSION_PER_CONTRACT}/contract + {SLIPPAGE_TICKS * 0.25} tick slippage")

    trades = []
    bars_since_last_trade = MIN_BARS_BETWEEN_TRADES
    open_positions = []

    # Start after feature window
    for i in range(100, len(df)):
        if i % 5000 == 0:
            logger.info(f"Processing bar {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)...")

        current_bar = df.iloc[i]

        # Get pre-computed regime
        regime = all_regimes[i]

        # Select model
        if regime == 0:
            model = regime_0_model
        elif regime == 1:
            model = regime_1_model
        else:
            model = regime_2_model

        # Get Tier1 features (16 features, exclude realized_vol_5)
        features = df_with_features.iloc[i][feature_names].values

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
        historical = df.iloc[i-100:i]
        recent_close = historical['close'].iloc[-1]
        momentum_5 = recent_close - historical['close'].iloc[-6]
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

    # Close any remaining open positions at end
    for pos in open_positions[:]:
        pnl = (float(df.iloc[-1]['close']) - pos['entry_price']) * CONTRACTS_PER_TRADE
        if pos['direction'] == "bearish":
            pnl = -pnl
        pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
        pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
        trades.append({**pos, 'exit_time': df.index[-1], 'pnl': pnl, 'exit_reason': 'end_of_data'})

    # Calculate metrics
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE TIER1 BACKTEST RESULTS (Oct-Dec 2025)")
    logger.info("=" * 80)

    if len(trades) == 0:
        logger.warning("No trades generated - system may need adjustment")
        return 0

    trades_df = pd.DataFrame(trades)
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date

    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = (winning_trades / len(trades_df) * 100)
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()

    daily_counts = trades_df.groupby('date').size()
    trades_per_day = daily_counts.mean()

    winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = winners / losers if losers > 0 else 0

    expectation_per_trade = avg_pnl

    # Sharpe ratio
    returns_std = trades_df['pnl'].std()
    sharpe_ratio = (avg_pnl / returns_std) if returns_std > 0 else 0

    # Max drawdown
    cumulative_returns = trades_df['pnl'].cumsum()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

    # Calculate equity curve metrics
    total_return = cumulative_returns.iloc[-1]
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0

    logger.info(f"\nTrading Summary:")
    logger.info(f"  Total Trades: {len(trades_df)}")
    logger.info(f"  Trading Days: {len(daily_counts)}")
    logger.info(f"  Trades/Day: {trades_per_day:.1f}")

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Win Rate: {win_rate:.2f}%")
    logger.info(f"  Total Return: ${total_pnl:,.2f}")
    logger.info(f"  Avg Return/Trade: ${avg_pnl:.2f}")
    logger.info(f"  Expectation/Trade (after costs): ${expectation_per_trade:.2f}")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")
    logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: ${max_drawdown:,.2f}")

    logger.info(f"\nTrade Details:")
    logger.info(f"  Avg Win: ${avg_win:.2f}")
    logger.info(f"  Avg Loss: ${avg_loss:.2f}")
    logger.info(f"  Avg Win/Avg Loss Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")

    # Exit reasons
    logger.info(f"\nExit Reasons:")
    logger.info(trades_df['exit_reason'].value_counts().to_string())

    # Regime analysis
    logger.info(f"\nRegime Distribution:")
    logger.info(trades_df['regime'].value_counts().sort_index().to_string())

    # Probability analysis
    logger.info(f"\nProbability Analysis:")
    logger.info(f"  Avg Probability: {trades_df['probability'].mean():.3f} ({trades_df['probability'].mean()*100:.1f}%)")
    logger.info(f"  Min Probability: {trades_df['probability'].min():.3f}")
    logger.info(f"  Max Probability: {trades_df['probability'].max():.3f}")

    # Save results
    output_path = Path("data/reports/backtest_tier1_complete_octdec2025.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)

    logger.info(f"\n✅ Results saved to: {output_path}")

    # Validate against targets
    logger.info(f"\n" + "=" * 80)
    logger.info("VALIDATION AGAINST TECH SPEC TARGETS")
    logger.info("=" * 80)

    targets = {
        f'Win Rate ≥ 50%: {win_rate:.1f}%': win_rate >= 50.0,
        f'Expectation/Trade ≥ $20: ${expectation_per_trade:.2f}': expectation_per_trade >= 20.0,
        f'Profit Factor ≥ 1.5: {profit_factor:.2f}': profit_factor >= 1.5,
        f'Trades/Day 5-25: {trades_per_day:.1f}': 5 <= trades_per_day <= 25,
        f'Sharpe ≥ 0.6: {sharpe_ratio:.2f}': sharpe_ratio >= 0.6,
        f'Max Drawdown <$1,000: ${max_drawdown:.2f}': max_drawdown < 1000.0,
    }

    for target, met in targets.items():
        status = "✅ PASS" if met else "❌ FAIL"
        logger.info(f"  {status} {target}")

    passed = sum(targets.values())
    total = len(targets)
    logger.info(f"\nTargets Met: {passed}/{total} ({passed/total*100:.1f}%)")

    # Overall assessment
    logger.info(f"\n" + "=" * 80)
    logger.info("OVERALL ASSESSMENT & DEPLOYMENT DECISION")
    logger.info("=" * 80)

    if passed >= total * 0.7:
        logger.info("🎉 DEPLOYMENT APPROVED - 70%+ targets met!")
        logger.info("\n✅ TIER1 1-MINUTE SYSTEM READY FOR PRODUCTION")
        logger.info("\nConfiguration Summary:")
        logger.info("  Model: Tier1 (16 order flow features)")
        logger.info(f"  Threshold: {PROBABILITY_THRESHOLD*100:.0f}%")
        logger.info(f"  MIN_BARS_BETWEEN_TRADES: {MIN_BARS_BETWEEN_TRADES}")
        logger.info(f"  MAX_CONCURRENT_POSITIONS: {MAX_CONCURRENT_POSITIONS}")
        logger.info(f"\nExpected Performance:")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Trades/Day: {trades_per_day:.1f}")
        logger.info(f"  Expectation/Trade: ${expectation_per_trade:.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info("\nNext Steps:")
        logger.info("1. ✅ Deploy to paper trading immediately")
        logger.info("2. Monitor performance for 2-4 weeks")
        logger.info("3. Consider optimizing threshold to 45% for better performance")
        logger.info("4. Scale to live trading after paper trading validation")

    elif passed >= total * 0.5:
        logger.info("⚠️  CAUTIOUS DEPLOYMENT - 50-70% targets met")
        logger.info("\nTier1 models show promise but need monitoring")
        logger.info("\nRecommendations:")
        logger.info("1. Deploy to paper trading for 2-4 weeks")
        logger.info("2. Monitor performance closely")
        logger.info("3. Consider threshold optimization (try 45%)")
        logger.info("4. Investigate any failing metrics")

    else:
        logger.info("❌ DEPLOYMENT NOT RECOMMENDED - <50% targets met")
        logger.info("\nTier1 system needs optimization")
        logger.info("\nRequired Actions:")
        logger.info("1. Investigate why performance is below targets")
        logger.info("2. Consider different probability threshold")
        logger.info("3. Review model training methodology")
        logger.info("4. Test alternative approaches")

    # Comparison with threshold analysis expectations
    logger.info(f"\n" + "=" * 80)
    logger.info("COMPARISON WITH THRESHOLD ANALYSIS EXPECTATIONS")
    logger.info("=" * 80)

    logger.info(f"\nExpected at {PROBABILITY_THRESHOLD*100:.0f}% threshold:")
    logger.info(f"  Trades: ~4,159")
    logger.info(f"  Win Rate: 93.8%")
    logger.info(f"  Trades/Day: 19.8")

    logger.info(f"\nActual Results:")
    logger.info(f"  Trades: {len(trades_df)}")
    logger.info(f"  Win Rate: {win_rate:.1f}%")
    logger.info(f"  Trades/Day: {trades_per_day:.1f}")

    # Calculate match percentage
    trades_match = (len(trades_df) / 4159) * 100 if len(trades_df) > 0 else 0
    winrate_match = (win_rate / 93.8) * 100 if win_rate > 0 else 0

    logger.info(f"\nMatch with Expectations:")
    logger.info(f"  Trade Count: {trades_match:.1f}% of expected")
    logger.info(f"  Win Rate: {winrate_match:.1f}% of expected")

    if trades_match > 80 and winrate_match > 80:
        logger.info("  ✅ Performance matches threshold analysis closely")
    elif trades_match > 50 and winrate_match > 50:
        logger.info("  ⚠️  Performance partially matches expectations")
    else:
        logger.info("  ❌ Performance differs significantly from expectations")

    logger.info(f"\n✅ Complete Tier1 backtest finished")

    return 0

if __name__ == "__main__":
    sys.exit(main())
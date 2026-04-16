#!/usr/bin/env python3
"""Exit parameter grid search for 1-minute Tier1 models.

Tests all combinations of:
- Stop Loss: 0.15%, 0.20%, 0.25%, 0.30%, 0.35%, 0.40%
- Take Profit: 0.20%, 0.25%, 0.30%, 0.35%, 0.40%
- Max Hold: 15min, 30min, 45min, 60min, 90min

Optimizes for Sharpe ratio on Oct-Dec 2025 validation data.
"""

import sys
import warnings
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.tier1_features import Tier1FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROBABILITY_THRESHOLD = 0.40
MIN_BARS_BETWEEN_TRADES = 1
MAX_CONCURRENT_POSITIONS = 3
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.50
CONTRACTS_PER_TRADE = 5

# Parameter grids
STOP_LOSS_PCTS = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]  # 0.15% - 0.40%
TAKE_PROFIT_PCTS = [0.0020, 0.0025, 0.0030, 0.0035, 0.0040]  # 0.20% - 0.40%
MAX_HOLD_BARS_LIST = [15, 30, 45, 60, 90]  # 15-90 minutes


def run_backtest(
    df: pd.DataFrame,
    df_with_features: pd.DataFrame,
    all_regimes: np.ndarray,
    models: Dict[int, object],
    feature_names: List[str],
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_bars: int
) -> Dict:
    """Run single backtest with given parameters."""

    trades = []
    bars_since_last_trade = MIN_BARS_BETWEEN_TRADES
    open_positions = []

    for i in range(100, len(df)):
        # Get regime and model
        regime = all_regimes[i]
        model = models.get(regime)
        if model is None:
            continue

        # Get features and predict
        features = df_with_features.iloc[i][feature_names].values
        probability = float(model.predict_proba(features.reshape(1, -1))[0, 1])

        if probability < PROBABILITY_THRESHOLD:
            bars_since_last_trade += 1
            continue

        if bars_since_last_trade < MIN_BARS_BETWEEN_TRADES:
            continue

        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            continue

        # Determine direction
        historical = df.iloc[i-100:i]
        recent_close = historical['close'].iloc[-1]
        momentum_5 = recent_close - historical['close'].iloc[-6]
        direction = "bullish" if momentum_5 > 0 else "bearish"

        # Calculate entry, stops
        entry_price = float(df.iloc[i]['close'])
        if direction == "bullish":
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

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
            if bars_held >= max_hold_bars:
                pnl = (float(df.iloc[i]['close']) - pos['entry_price']) * CONTRACTS_PER_TRADE
                if pos['direction'] == "bearish":
                    pnl = -pnl
                pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
                pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
                trades.append({**pos, 'exit_time': df.index[i], 'pnl': pnl, 'exit_reason': 'time'})
                open_positions.remove(pos)
                continue

            # Stop loss and take profit
            current_bar = df.iloc[i]
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
            else:
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

    # Close remaining positions
    for pos in open_positions[:]:
        pnl = (float(df.iloc[-1]['close']) - pos['entry_price']) * CONTRACTS_PER_TRADE
        if pos['direction'] == "bearish":
            pnl = -pnl
        pnl -= COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE
        pnl -= SLIPPAGE_TICKS * 0.25 * CONTRACTS_PER_TRADE
        trades.append({**pos, 'exit_time': df.index[-1], 'pnl': pnl, 'exit_reason': 'end_of_data'})

    # Calculate metrics
    if len(trades) == 0:
        return {
            'trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'sharpe': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'trades_per_day': 0
        }

    trades_df = pd.DataFrame(trades)
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date

    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    trades_per_day = trades_df.groupby('date').size().mean()

    winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = winners / losers if losers > 0 else 0

    returns_std = trades_df['pnl'].std()
    sharpe = (avg_pnl / returns_std) if returns_std > 0 else 0

    cumulative_returns = trades_df['pnl'].cumsum()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'trades_per_day': trades_per_day
    }


def main():
    logger.info("=" * 80)
    logger.info("EXIT PARAMETER GRID SEARCH (1-Minute Tier1 Models)")
    logger.info("Testing 6×5×5 = 150 parameter combinations")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading 1-minute dollar bars...")
    data_path = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df[(df.index.month >= 10) & (df.index.year == 2025)]
    logger.info(f"✅ Loaded {len(df):,} bars for Oct-Dec 2025")

    # Load models
    logger.info("\nLoading models...")
    import joblib

    hmm_path = Path("models/hmm/regime_model_1min")
    detector = HMMRegimeDetector.load(hmm_path)

    model_dir = Path("models/xgboost/regime_aware_tier1")
    models = {
        0: joblib.load(model_dir / "xgboost_regime_0_tier1.joblib"),
        1: joblib.load(model_dir / "xgboost_regime_1_tier1.joblib"),
        2: joblib.load(model_dir / "xgboost_regime_2_tier1.joblib")
    }

    logger.info(f"✅ Loaded HMM + 3 XGBoost models")

    # Pre-compute regimes
    logger.info("\nPre-computing regimes...")
    hmm_feature_engineer = HMMFeatureEngineer()
    all_hmm_features = hmm_feature_engineer.engineer_features(df)
    all_regimes = detector.predict(all_hmm_features)
    logger.info(f"✅ Regimes computed")

    # Generate Tier1 features
    logger.info("\nGenerating Tier1 features...")
    tier1_engineer = Tier1FeatureEngineer()
    df_with_features = tier1_engineer.generate_features(df)
    feature_names = [f for f in tier1_engineer.feature_names if f != 'realized_vol_5']
    logger.info(f"✅ Using {len(feature_names)} Tier1 features")

    # Grid search
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH STARTING")
    logger.info(f"Total combinations: {len(STOP_LOSS_PCTS) * len(TAKE_PROFIT_PCTS) * len(MAX_HOLD_BARS_LIST)}")
    logger.info("=" * 80)

    results = []
    total = len(STOP_LOSS_PCTS) * len(TAKE_PROFIT_PCTS) * len(MAX_HOLD_BARS_LIST)
    current = 0

    for stop_loss_pct, take_profit_pct, max_hold_bars in product(
        STOP_LOSS_PCTS, TAKE_PROFIT_PCTS, MAX_HOLD_BARS_LIST
    ):
        current += 1
        if current % 10 == 0:
            logger.info(f"Progress: {current}/{total} ({current/total*100:.1f}%)...")

        metrics = run_backtest(
            df, df_with_features, all_regimes, models, feature_names,
            stop_loss_pct, take_profit_pct, max_hold_bars
        )

        results.append({
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_hold_bars': max_hold_bars,
            **metrics
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe', ascending=False)

    # Display top 10
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 PARAMETER SETS (BY SHARPE RATIO)")
    logger.info("=" * 80)

    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        logger.info(
            f"\n#{row.name+1} (Sharpe: {row['sharpe']:.2f})"
            f"\n  Stop Loss: {row['stop_loss_pct']*100:.2f}%"
            f"\n  Take Profit: {row['take_profit_pct']*100:.2f}%"
            f"\n  Max Hold: {row['max_hold_bars']} min"
            f"\n  Trades: {row['trades']}"
            f"\n  Win Rate: {row['win_rate']:.1f}%"
            f"\n  Trades/Day: {row['trades_per_day']:.1f}"
            f"\n  Expectation/Trade: ${row['avg_pnl']:.2f}"
            f"\n  Total P&L: ${row['total_pnl']:,.2f}"
            f"\n  Profit Factor: {row['profit_factor']:.2f}"
            f"\n  Max Drawdown: ${row['max_drawdown']:,.2f}"
        )

    # Save results
    output_path = Path("data/reports/exit_parameter_optimization_1min.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Full results saved to: {output_path}")

    # Generate report
    report_path = Path("data/reports/exit_parameter_optimization_1min.md")
    with open(report_path, 'w') as f:
        f.write("# Exit Parameter Optimization - 1-Minute Tier1 Models\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Test Period:** Oct-Dec 2025 (held-out validation data)\n")
        f.write(f"**Total Combinations Tested:** {len(results_df)}\n\n")

        f.write("## Top 3 Parameter Sets\n\n")

        for i in range(min(3, len(results_df))):
            row = results_df.iloc[i]
            f.write(f"### #{i+1} (Sharpe: {row['sharpe']:.2f})\n\n")
            f.write(f"- **Stop Loss:** {row['stop_loss_pct']*100:.2f}%\n")
            f.write(f"- **Take Profit:** {row['take_profit_pct']*100:.2f}%\n")
            f.write(f"- **Max Hold:** {row['max_hold_bars']} minutes\n")
            f.write(f"- **Trades:** {row['trades']}\n")
            f.write(f"- **Win Rate:** {row['win_rate']:.1f}%\n")
            f.write(f"- **Trades/Day:** {row['trades_per_day']:.1f}\n")
            f.write(f"- **Expectation/Trade:** ${row['avg_pnl']:.2f}\n")
            f.write(f"- **Total P&L:** ${row['total_pnl']:,.2f}\n")
            f.write(f"- **Profit Factor:** {row['profit_factor']:.2f}\n")
            f.write(f"- **Max Drawdown:** ${row['max_drawdown']:,.2f}\n\n")

        f.write("## Comparison with Baseline\n\n")
        f.write("| Metric | Baseline (0.3%/0.2%/30min) | Best Optimized |\n")
        f.write("|--------|---------------------------|----------------|\n")

        baseline = results_df[
            (results_df['stop_loss_pct'] == 0.002) &
            (results_df['take_profit_pct'] == 0.003) &
            (results_df['max_hold_bars'] == 30)
        ].iloc[0] if len(results_df[
            (results_df['stop_loss_pct'] == 0.002) &
            (results_df['take_profit_pct'] == 0.003) &
            (results_df['max_hold_bars'] == 30)
        ]) > 0 else None

        best = results_df.iloc[0]

        if baseline is not None:
            f.write(f"| Sharpe Ratio | {baseline['sharpe']:.2f} | {best['sharpe']:.2f} ({((best['sharpe']/baseline['sharpe']-1)*100):+.1f}%) |\n")
            f.write(f"| Win Rate | {baseline['win_rate']:.1f}% | {best['win_rate']:.1f}% ({(best['win_rate']-baseline['win_rate']):+.1f}pp) |\n")
            f.write(f"| Trades/Day | {baseline['trades_per_day']:.1f} | {best['trades_per_day']:.1f} ({((best['trades_per_day']/baseline['trades_per_day']-1)*100):+.1f}%) |\n")
            f.write(f"| Expectation/Trade | ${baseline['avg_pnl']:.2f} | ${best['avg_pnl']:.2f} ({((best['avg_pnl']/baseline['avg_pnl']-1)*100):+.1f}%) |\n")

        f.write("\n## Recommendations\n\n")

        if best['sharpe'] > 0.6:
            f.write(f"✅ **RECOMMENDED:** Use #{1} parameters (Sharpe: {best['sharpe']:.2f})\n\n")
            f.write("This configuration meets the minimum Sharpe ratio threshold (≥0.6).\n")
        else:
            f.write("⚠️ **CAUTION:** No parameter combination achieved Sharpe ≥0.6\n\n")
            f.write("Exit parameter optimization alone is insufficient. Training methodology fixes required.\n")

    logger.info(f"✅ Report saved to: {report_path}")
    logger.info("\n✅ Task 1.1 complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

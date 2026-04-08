#!/usr/bin/env python3
"""
Simple FVG Strategy Backtest (2024) - Baseline Performance

This backtests a simple Fair Value Gap (FVG) trading strategy on one year of 2024 MNQ data.
FVG is one of the core ICT patterns and provides clear entry/exit levels.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import List, Dict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.models import DollarBar
from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class FVGBacktester:
    """Simple FVG strategy backtester."""

    def __init__(self, initial_capital: float = 100000.0):
        """Initialize backtester."""
        self.initial_capital = initial_capital
        self.bars: List[DollarBar] = []

    def load_2025_data(self) -> None:
        """Load all 2025 MNQ data."""
        logger.info("🔄 Loading 2025 MNQ data...")

        data_path = Path("data/processed/time_bars/")
        h5_files = sorted(data_path.glob("MNQ_time_bars_5min_2025*.h5"))

        if not h5_files:
            raise ValueError("No 2024 MNQ data files found")

        logger.info(f"📂 Found {len(h5_files)} monthly files")

        all_bars = []

        for h5_file in tqdm(h5_files, desc="Loading monthly data"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'dollar_bars' not in f:
                        continue

                    dataset = f['dollar_bars']

                    for i in range(len(dataset)):
                        try:
                            ts_ms = dataset[i, 0]
                            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

                            bar = DollarBar(
                                timestamp=ts,
                                open=float(dataset[i, 1]),
                                high=float(dataset[i, 2]),
                                low=float(dataset[i, 3]),
                                close=float(dataset[i, 4]),
                                volume=int(dataset[i, 5]),
                                notional_value=float(dataset[i, 6]) if dataset.shape[1] > 6 else 0.0,
                                is_forward_filled=False,
                            )
                            all_bars.append(bar)
                        except Exception:
                            continue

            except Exception as e:
                logger.debug(f"Error loading {h5_file.name}: {e}")

        all_bars.sort(key=lambda x: x.timestamp)
        self.bars = all_bars

        logger.info(f"✅ Loaded {len(all_bars):,} bars for 2025")
        if all_bars:
            logger.info(f"📅 Date range: {all_bars[0].timestamp} to {all_bars[-1].timestamp}")

    def detect_fvg_setups(self) -> List[Dict]:
        """Detect all FVG setups for 2024."""
        logger.info("🔍 Detecting FVG setups...")

        fvg_setups = []

        # Detect FVGs with sliding window
        for i in tqdm(range(3, len(self.bars)), desc="Scanning for FVGs"):
            try:
                # Get recent bars for context
                start_idx = max(0, i - 10)
                historical_bars = self.bars[start_idx:i+1]
                current_index = len(historical_bars) - 1

                # Detect FVGs
                bullish_fvg = detect_bullish_fvg(historical_bars, current_index)
                bearish_fvg = detect_bearish_fvg(historical_bars, current_index)

                if bullish_fvg:
                    fvg_setups.append({
                        'timestamp': self.bars[i].timestamp,
                        'bar_index': i,
                        'direction': 'bullish',
                        'entry_top': bullish_fvg.gap_range.top,
                        'entry_bottom': bullish_fvg.gap_range.bottom,
                        'gap_size': bullish_fvg.gap_size_dollars,
                        'fvg_event': bullish_fvg,
                    })

                if bearish_fvg:
                    fvg_setups.append({
                        'timestamp': self.bars[i].timestamp,
                        'bar_index': i,
                        'direction': 'bearish',
                        'entry_top': bearish_fvg.gap_range.top,
                        'entry_bottom': bearish_fvg.gap_range.bottom,
                        'gap_size': bearish_fvg.gap_size_dollars,
                        'fvg_event': bearish_fvg,
                    })

            except Exception as e:
                logger.debug(f"Error detecting FVG at bar {i}: {e}")
                continue

        logger.info(f"✅ Found {len(fvg_setups)} FVG setups")
        return fvg_setups

    def execute_backtest(self, setups: List[Dict]) -> List[Dict]:
        """Execute backtest on FVG setups.

        Strategy: Enter at FVG midpoint, exit when gap fills or stop loss hit.
        Risk: 1.5x gap size, Reward: Gap fill (conservative approach)
        """
        logger.info("💰 Executing backtest trades...")

        trades = []
        position_size = 1  # 1 contract per trade

        for setup in tqdm(setups, desc="Executing trades"):
            try:
                entry_idx = setup['bar_index']
                if entry_idx >= len(self.bars) - 1:
                    continue

                entry_bar = self.bars[entry_idx]

                # Entry at FVG midpoint
                entry_price = (setup['entry_top'] + setup['entry_bottom']) / 2

                # Calculate stop loss (1.5x gap size beyond FVG)
                gap_size = setup['entry_top'] - setup['entry_bottom']

                if setup['direction'] == 'bullish':
                    stop_loss = setup['entry_bottom'] - (gap_size * 1.5)
                    target_price = setup['entry_top']  # Gap fill
                else:  # bearish
                    stop_loss = setup['entry_top'] + (gap_size * 1.5)
                    target_price = setup['entry_bottom']  # Gap fill

                # Execute trade with 20-bar limit
                max_bars = 20
                for exit_idx in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(self.bars))):
                    exit_bar = self.bars[exit_idx]
                    pnl = 0.0
                    exit_reason = None

                    if setup['direction'] == 'bullish':
                        # Check stop loss
                        if exit_bar.low <= stop_loss:
                            pnl = (stop_loss - entry_price) * position_size * 0.5  # $0.50 per tick
                            exit_reason = "stop_loss"
                            break
                        # Check target (gap fill)
                        elif exit_bar.high >= target_price:
                            pnl = (target_price - entry_price) * position_size * 0.5
                            exit_reason = "gap_fill"
                            break
                    else:  # bearish
                        # Check stop loss
                        if exit_bar.high >= stop_loss:
                            pnl = (entry_price - stop_loss) * position_size * 0.5
                            exit_reason = "stop_loss"
                            break
                        # Check target (gap fill)
                        elif exit_bar.low <= target_price:
                            pnl = (entry_price - target_price) * position_size * 0.5
                            exit_reason = "gap_fill"
                            break

                    # Time exit
                    if exit_idx - entry_idx >= max_bars:
                        if setup['direction'] == 'bullish':
                            pnl = (exit_bar.close - entry_price) * position_size * 0.5
                        else:
                            pnl = (entry_price - exit_bar.close) * position_size * 0.5
                        exit_reason = "time_exit"
                        break

                if pnl != 0.0:
                    trade = {
                        'entry_time': entry_bar.timestamp,
                        'exit_time': exit_bar.timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_bar.close,
                        'direction': setup['direction'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'gap_size': setup['gap_size'],
                        'bars_held': exit_idx - entry_idx,
                    }
                    trades.append(trade)

            except Exception as e:
                logger.debug(f"Error executing trade: {e}")
                continue

        logger.info(f"✅ Executed {len(trades)} trades")
        return trades

    def analyze_performance(self, trades: List[Dict]) -> Dict:
        """Analyze backtest performance."""
        logger.info("📊 Analyzing performance...")

        if not trades:
            return {}

        df = pd.DataFrame(trades)

        total_trades = len(trades)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0.0

        profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0.0
        expectancy = avg_pnl

        # Drawdown analysis
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['running_max'] = df['cumulative_pnl'].cummax()
        df['drawdown'] = df['cumulative_pnl'] - df['running_max']
        max_drawdown = df['drawdown'].min()

        # Sharpe ratio
        if len(trades) > 1:
            returns = df['pnl'] / self.initial_capital
            sharpe_ratio = np.sqrt(252 * 78) * returns.mean() / returns.std() if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Monthly analysis
        df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')
        monthly_pnl = df.groupby('month')['pnl'].sum()

        # Exit analysis
        exit_reasons = df['exit_reason'].value_counts()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.initial_capital + total_pnl,
            'monthly_pnl': monthly_pnl.to_dict(),
            'exit_reasons': exit_reasons.to_dict(),
            'avg_bars_held': df['bars_held'].mean(),
        }

    def generate_report(self, performance: Dict) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("FVG STRATEGY BACKTEST REPORT (2025)")
        report.append("=" * 80)
        report.append("")

        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Trades: {performance['total_trades']}")
        report.append(f"Win Rate: {performance['win_rate']:.2%}")
        report.append(f"Total P&L: ${performance['total_pnl']:,.2f}")
        report.append(f"Return: {(performance['total_pnl'] / self.initial_capital):.2%}")
        report.append("")

        report.append("⚠️  RISK METRICS")
        report.append("-" * 80)
        report.append(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        report.append(f"Max Drawdown: ${performance['max_drawdown']:,.2f}")
        report.append(f"Profit Factor: {performance['profit_factor']:.2f}")
        report.append(f"Expectancy: ${performance['expectancy']:.2f} per trade")
        report.append("")

        report.append("📈 TRADE STATISTICS")
        report.append("-" * 80)
        report.append(f"Average Win: ${performance['avg_win']:,.2f}")
        report.append(f"Average Loss: ${performance['avg_loss']:,.2f}")
        report.append(f"Average Bars Held: {performance['avg_bars_held']:.1f}")
        report.append("")

        report.append("🚪 EXIT ANALYSIS")
        report.append("-" * 80)
        for reason, count in performance['exit_reasons'].items():
            pct = count / performance['total_trades'] * 100
            report.append(f"{reason}: {count} ({pct:.1f}%)")
        report.append("")

        report.append("📅 MONTHLY PERFORMANCE")
        report.append("-" * 80)
        for month, pnl in performance['monthly_pnl'].items():
            report.append(f"{month}: ${pnl:,.2f}")
        report.append("")

        # Institutional assessment
        report.append("🏦 INSTITUTIONAL GRADE ASSESSMENT")
        report.append("-" * 80)

        metrics = {
            'Sharpe Ratio': (performance['sharpe_ratio'], 1.0),
            'Max Drawdown': (abs(performance['max_drawdown']), 10.0),
            'Win Rate': (performance['win_rate'], 0.55),
            'Profit Factor': (performance['profit_factor'], 1.5),
        }

        passed = 0
        for metric, (value, threshold) in metrics.items():
            if metric == 'Max Drawdown':
                status = "✅ PASS" if value <= threshold else "❌ FAIL"
            else:
                status = "✅ PASS" if value >= threshold else "❌ FAIL"
                if value >= threshold:
                    passed += 1
            report.append(f"{metric}: {value:.3f} (threshold: {threshold}) {status}")

        report.append("")
        if passed >= 3:
            report.append("🎯 INSTITUTIONAL GRADE: PASS (3+ metrics met threshold)")
        else:
            report.append("⚠️  INSTITUTIONAL GRADE: FAIL (< 3 metrics met threshold)")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Run FVG strategy backtest."""
    logger.info("🚀 Starting FVG Strategy Backtest (2025)")
    logger.info("=" * 80)

    backtester = FVGBacktester(initial_capital=100000.0)

    try:
        # Load 2025 data
        backtester.load_2025_data()

        # Detect FVG setups
        setups = backtester.detect_fvg_setups()

        if not setups:
            logger.warning("⚠️  No FVG setups found.")
            return

        # Execute backtest
        trades = backtester.execute_backtest(setups)

        if not trades:
            logger.error("❌ No trades executed.")
            return

        # Analyze performance
        performance = backtester.analyze_performance(trades)

        # Generate report
        report = backtester.generate_report(performance)
        print("\n" + report)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/reports/backtest_fvg_2024_{timestamp}.txt"

        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"✅ Report saved to {report_file}")

        # Save trade data
        trades_file = f"data/reports/backtest_fvg_2024_{timestamp}.csv"
        pd.DataFrame(trades).to_csv(trades_file, index=False)
        logger.info(f"✅ Trade data saved to {trades_file}")

    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
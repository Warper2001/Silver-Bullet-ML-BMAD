#!/usr/bin/env python3
"""Walk-Forward Validation for TIER 1 FVG System.

Tests the optimal configuration (SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0) across
multiple time periods to detect overfitting and validate robustness.

Periods: 4 independent 3-week test periods
- Period 1: Dec 2023 (in-sample - already tested)
- Period 2: Jan 2024 (out-of-sample)
- Period 3: Feb 2024 (out-of-sample)
- Period 4: Mar 2024 (out-of-sample)
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import DollarBar, FVGEvent, GapRange

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MNQ_DATA_PATH = Path("/root/mnq_historical.json")
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE
DOLLAR_BAR_THRESHOLD = 50_000_000
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1

# OPTIMAL CONFIGURATION from parameter tuning
SL_MULTIPLIER = 2.5
ATR_THRESHOLD = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS = 50.0

# Trading parameters
CONTRACTS_PER_TRADE = 1
MAX_HOLD_BARS = 10

# Period definitions (start_index, end_index, period_name)
# Testing most recent 3 months for current market conditions
PERIODS = [
    (775296, 795296, "Period 1: Dec 2025 (Most Recent)"),
    (755296, 775296, "Period 2: Nov 2025 (Out-of-Sample)"),
    (735296, 755296, "Period 3: Oct 2025 (Out-of-Sample)"),
]


@dataclass
class PeriodResult:
    """Results for a single time period."""
    period_name: str
    start_date: str
    end_date: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    avg_trades_per_day: float
    total_pnl: float
    expectancy: float


def load_mnq_data_subset(data_path: Path, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Load MNQ historical data subset."""
    print(f"Loading bars {start_idx}:{end_idx} from {data_path}...")

    if not data_path.exists():
        raise FileNotFoundError(f"MNQ data file not found: {data_path}")

    # Load only the required slice to avoid memory issues
    with open(data_path, 'r') as f:
        # Read and parse JSON in streaming fashion
        data = json.loads(f.read())
        data = data[start_idx:end_idx]

    if len(data) == 0:
        raise ValueError(f"No data found for range {start_idx}:{end_idx}")

    df = pd.DataFrame(data)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    numeric_columns = ['High', 'Low', 'Open', 'Close', 'TotalVolume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('TimeStamp').reset_index(drop=True)

    print(f"✓ Loaded {len(df)} bars: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

    return df


def transform_to_dollar_bars(df: pd.DataFrame) -> list[DollarBar]:
    """Transform to Dollar Bars."""
    logger.info("Transforming to Dollar Bars...")

    df['notional'] = ((df['High'] + df['Low']) / 2) * df['TotalVolume'] * MNQ_POINT_VALUE
    df['cumulative_notional'] = df['notional'].cumsum()

    bar_boundaries = df[df['cumulative_notional'] % DOLLAR_BAR_THRESHOLD < df['notional']].index.tolist()

    if len(df) > 0 and (len(bar_boundaries) == 0 or bar_boundaries[-1] != len(df) - 1):
        bar_boundaries.append(len(df) - 1)

    dollar_bars = []
    prev_boundary = 0

    for boundary in bar_boundaries:
        if boundary == 0:
            continue

        segment = df.iloc[prev_boundary:boundary+1]

        if len(segment) == 0:
            continue

        capped_notional = min(float(segment['notional'].sum()), 1_500_000_000)

        dollar_bar = DollarBar(
            timestamp=segment.iloc[0]['TimeStamp'],
            open=float(segment.iloc[0]['Open']),
            high=float(segment['High'].max()),
            low=float(segment['Low'].min()),
            close=float(segment.iloc[-1]['Close']),
            volume=int(segment['TotalVolume'].sum()),
            notional_value=capped_notional,
            is_forward_filled=False,
        )
        dollar_bars.append(dollar_bar)

        prev_boundary = boundary + 1

    logger.info(f"Created {len(dollar_bars)} Dollar Bars")
    return dollar_bars


class Tier1Validator:
    """Validator for TIER 1 with optimal configuration."""

    def __init__(self):
        self.sl_multiplier = SL_MULTIPLIER
        self.atr_threshold = ATR_THRESHOLD
        self.volume_ratio_threshold = VOLUME_RATIO_THRESHOLD
        self.max_gap_dollars = MAX_GAP_DOLLARS
        self.trades = []

    def run_backtest(self, dollar_bars: list[DollarBar]) -> dict:
        """Run backtest with optimal configuration."""
        print(f"Running backtest with SL{self.sl_multiplier}x_ATR{self.atr_threshold}_Vol{self.volume_ratio_threshold}_MaxGap${self.max_gap_dollars}...")

        # Pre-calculate ATR (vectorized)
        df = pd.DataFrame([
            {'high': bar.high, 'low': bar.low, 'close': bar.close}
            for bar in dollar_bars
        ])

        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr_values = df['true_range'].ewm(span=14, adjust=False).mean().values
        mean_tr = df['true_range'].mean()
        atr_values = np.nan_to_num(atr_values, nan=mean_tr)

        # Pre-calculate volume ratios
        volumes = np.array([bar.volume for bar in dollar_bars])
        is_bullish = np.array([1 if bar.close > bar.open else 0 for bar in dollar_bars])
        is_bearish = np.array([1 if bar.close < bar.open else 0 for bar in dollar_bars])

        up_volumes = pd.Series(volumes * is_bullish).rolling(window=20, min_periods=1).sum().values
        down_volumes = pd.Series(volumes * is_bearish).rolling(window=20, min_periods=1).sum().values

        # Detect and simulate trades
        total_bars = len(dollar_bars)
        for i in range(2, total_bars):
            # Progress indicator every 1000 bars
            if i % 1000 == 0:
                print(f"  Processing bar {i}/{total_bars} ({100*i/total_bars:.1f}%)...")

            # Detect bullish FVG
            bullish_fvg = self._detect_bullish_fvg_with_filters(
                dollar_bars, i, atr_values, up_volumes, down_volumes
            )

            # Detect bearish FVG
            bearish_fvg = self._detect_bearish_fvg_with_filters(
                dollar_bars, i, atr_values, up_volumes, down_volumes
            )

            for fvg in [bullish_fvg, bearish_fvg]:
                if fvg is None:
                    continue

                self._simulate_fvg_trade(fvg, dollar_bars)

        return self._calculate_metrics(dollar_bars)

    def _detect_bullish_fvg_with_filters(
        self,
        bars: list[DollarBar],
        i: int,
        atr_values: np.ndarray,
        up_volumes: np.ndarray,
        down_volumes: np.ndarray,
    ) -> FVGEvent | None:
        """Detect bullish FVG with TIER 1 filters."""
        if i < 2:
            return None

        candle_1 = bars[i-2]
        candle_3 = bars[i]

        # Bullish FVG pattern: candle 1 close > candle 3 open
        if candle_1.close <= candle_3.open:
            return None

        # Calculate gap range
        gap_bottom = candle_3.low
        gap_top = candle_1.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * self.atr_threshold):
            return None

        # Max gap size filter
        if self.max_gap_dollars is not None:
            gap_dollars = gap_size * MNQ_CONTRACT_VALUE
            if gap_dollars > self.max_gap_dollars:
                return None

        # Volume filter
        up_volume = up_volumes[i]
        down_volume = down_volumes[i]

        if down_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = up_volume / down_volume

        if volume_ratio < self.volume_ratio_threshold:
            return None

        return FVGEvent(
            timestamp=bars[i].timestamp,
            direction="bullish",
            gap_range=GapRange(top=gap_top, bottom=gap_bottom),
            gap_size_ticks=gap_size / MNQ_TICK_SIZE,
            gap_size_dollars=gap_size * MNQ_CONTRACT_VALUE,
            bar_index=i,
            filled=False,
        )

    def _detect_bearish_fvg_with_filters(
        self,
        bars: list[DollarBar],
        i: int,
        atr_values: np.ndarray,
        up_volumes: np.ndarray,
        down_volumes: np.ndarray,
    ) -> FVGEvent | None:
        """Detect bearish FVG with TIER 1 filters."""
        if i < 2:
            return None

        candle_1 = bars[i-2]
        candle_3 = bars[i]

        # Bearish FVG pattern: candle 1 close < candle 3 open
        if candle_1.close >= candle_3.open:
            return None

        # Calculate gap range
        gap_bottom = candle_1.low
        gap_top = candle_3.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * self.atr_threshold):
            return None

        # Max gap size filter
        if self.max_gap_dollars is not None:
            gap_dollars = gap_size * MNQ_CONTRACT_VALUE
            if gap_dollars > self.max_gap_dollars:
                return None

        # Volume filter (bearish = down/up)
        up_volume = up_volumes[i]
        down_volume = down_volumes[i]

        if up_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = down_volume / up_volume

        if volume_ratio < self.volume_ratio_threshold:
            return None

        return FVGEvent(
            timestamp=bars[i].timestamp,
            direction="bearish",
            gap_range=GapRange(top=gap_top, bottom=gap_bottom),
            gap_size_ticks=gap_size / MNQ_TICK_SIZE,
            gap_size_dollars=gap_size * MNQ_CONTRACT_VALUE,
            bar_index=i,
            filled=False,
        )

    def _simulate_fvg_trade(self, fvg: FVGEvent, dollar_bars: list[DollarBar]) -> None:
        """Simulate FVG trade with optimal SL."""
        direction = "long" if fvg.direction == "bullish" else "short"

        if direction == "long":
            entry_price = fvg.gap_range.bottom
            take_profit = fvg.gap_range.top
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.bottom - (gap_size * self.sl_multiplier)
        else:  # short
            entry_price = fvg.gap_range.top
            take_profit = fvg.gap_range.bottom
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.top + (gap_size * self.sl_multiplier)

        if stop_loss <= 0 or take_profit <= 0:
            return

        entry_bar_index = fvg.bar_index + 1
        if entry_bar_index >= len(dollar_bars):
            return

        entry_bar = dollar_bars[entry_bar_index]

        # Simulate exit
        exit_price, exit_reason, bars_held = self._simulate_exit(
            entry_bar_index,
            entry_price,
            stop_loss,
            take_profit,
            dollar_bars,
            direction,
        )

        # Calculate P&L with transaction costs
        commission = COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2
        slippage_cost = SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * CONTRACTS_PER_TRADE * 2

        if direction == "long":
            price_diff = exit_price - entry_price
        else:  # short
            price_diff = entry_price - exit_price

        pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE * CONTRACTS_PER_TRADE
        pnl_final = pnl_before_costs - commission - slippage_cost

        # Record trade
        self.trades.append({
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl_final,
            "bars_held": bars_held,
            "exit_reason": exit_reason,
        })

    def _simulate_exit(
        self,
        entry_bar_index: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        dollar_bars: list[DollarBar],
        direction: str,
    ) -> tuple[float, str, int]:
        """Simulate trade exit."""
        sl_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
        tp_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE

        if direction == "long":
            sl_trigger = stop_loss - sl_buffer
            tp_trigger = take_profit + tp_buffer
        else:  # short
            sl_trigger = stop_loss + sl_buffer
            tp_trigger = take_profit - tp_buffer

        max_index = min(entry_bar_index + MAX_HOLD_BARS + 1, len(dollar_bars))

        for i in range(entry_bar_index + 1, max_index):
            bar = dollar_bars[i]

            if direction == "long":
                if bar.low <= sl_trigger:
                    exit_price = min(stop_loss, bar.low + sl_buffer)
                    return exit_price, "stop_loss", i - entry_bar_index
                if bar.high >= tp_trigger:
                    exit_price = max(take_profit, bar.high - tp_buffer)
                    return exit_price, "take_profit", i - entry_bar_index
            else:  # short
                if bar.high >= sl_trigger:
                    exit_price = max(stop_loss, bar.high - sl_buffer)
                    return exit_price, "stop_loss", i - entry_bar_index
                if bar.low <= tp_trigger:
                    exit_price = min(take_profit, bar.low + tp_buffer)
                    return exit_price, "take_profit", i - entry_bar_index

            if i - entry_bar_index >= MAX_HOLD_BARS:
                return bar.close, "max_time", i - entry_bar_index

        last_bar = dollar_bars[-1]
        bars_held = len(dollar_bars) - 1 - entry_bar_index
        return last_bar.close, "end_of_data", bars_held

    def _calculate_metrics(self, dollar_bars: list[DollarBar]) -> dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trades_per_day": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
            }

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] < 0]

        win_rate = len(wins) / len(self.trades) * 100
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_won = sum(t["pnl"] for t in wins)
        total_lost = sum(t["pnl"] for t in losses)
        profit_factor = abs(total_won / total_lost) if total_lost != 0 else float('inf')
        avg_win = total_won / len(wins) if wins else 0.0
        avg_loss = total_lost / len(losses) if losses else 0.0
        expectancy = total_pnl / len(self.trades)

        if len(dollar_bars) > 0:
            time_span_days = (dollar_bars[-1].timestamp - dollar_bars[0].timestamp).total_seconds() / 86400
            avg_trades_per_day = len(self.trades) / time_span_days if time_span_days > 0 else 0.0
        else:
            avg_trades_per_day = 0.0

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trades_per_day": avg_trades_per_day,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
        }


def generate_walkforward_report(period_results: list[PeriodResult]) -> str:
    """Generate walk-forward validation report."""
    report = f"""
{'=' * 80}
WALK-FORWARD VALIDATION REPORT - TIER 1 FVG System
{'=' * 80}

Configuration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}

{'=' * 80}
PERIOD RESULTS
{'=' * 80}

"""

    for result in period_results:
        report += f"""
{result.period_name}
Date Range: {result.start_date} to {result.end_date}
{'=' * 80}
Total Trades: {result.total_trades}
Wins: {result.wins} | Losses: {result.losses}
Win Rate: {result.win_rate:.2f}% {'✅' if result.win_rate >= 60.0 else '❌'}
Profit Factor: {result.profit_factor:.2f} {'✅' if result.profit_factor >= 1.7 else '❌'}
Trade Frequency: {result.avg_trades_per_day:.2f}/day {'✅' if 8.0 <= result.avg_trades_per_day <= 15.0 else '❌'}
Total P&L: ${result.total_pnl:.2f}
Expectancy: ${result.expectancy:.2f}/trade
Targets Met: {sum([result.win_rate >= 60.0, result.profit_factor >= 1.7, 8.0 <= result.avg_trades_per_day <= 15.0])}/3
"""

    # Calculate aggregate statistics
    all_results = [r for r in period_results if r.total_trades > 0]
    if not all_results:
        report += "\n❌ NO VALID RESULTS\n"
        return report

    avg_win_rate = np.mean([r.win_rate for r in all_results])
    avg_profit_factor = np.mean([r.profit_factor for r in all_results if r.profit_factor != float('inf')])
    avg_frequency = np.mean([r.avg_trades_per_day for r in all_results])
    total_pnl = sum([r.total_pnl for r in all_results])

    # Calculate consistency (standard deviation)
    wr_std = np.std([r.win_rate for r in all_results])
    pf_std = np.std([r.profit_factor for r in all_results if r.profit_factor != float('inf')])

    report += f"""
{'=' * 80}
AGGREGATE STATISTICS ({len(all_results)} periods)
{'=' * 80}
Average Win Rate: {avg_win_rate:.2f}% (±{wr_std:.2f}%)
Average Profit Factor: {avg_profit_factor:.2f} (±{pf_std:.2f})
Average Trade Frequency: {avg_frequency:.2f}/day
Total P&L (all periods): ${total_pnl:.2f}

"""

    # Assess robustness
    periods_passing_all = sum([
        1 for r in all_results
        if r.win_rate >= 60.0 and r.profit_factor >= 1.7 and 8.0 <= r.avg_trades_per_day <= 15.0
    ])

    pass_rate = periods_passing_all / len(all_results) * 100

    report += f"""
{'=' * 80}
ROBUSTNESS ASSESSMENT
{'=' * 80}
Periods Passing All 3 Targets: {periods_passing_all}/{len(all_results)} ({pass_rate:.1f}%)
"""

    if pass_rate >= 75:
        report += "Conclusion: ✅ ROBUST - System performs consistently across different market conditions\n"
    elif pass_rate >= 50:
        report += "Conclusion: ⚠️ MODERATE - System shows some inconsistency but may be viable\n"
    else:
        report += "Conclusion: ❌ FRAGILE - System is likely overfit to in-sample data\n"

    # Check for warning signs
    warnings = []
    if wr_std > 10:
        warnings.append(f"⚠️ High win rate variability (±{wr_std:.1f}%)")
    if pf_std > 0.5:
        warnings.append(f"⚠️ High profit factor variability (±{pf_std:.2f})")

    if any([r.total_trades < 100 for r in all_results]):
        warnings.append("⚠️ Some periods have insufficient trade sample size")

    if warnings:
        report += "\nWarnings:\n"
        for warning in warnings:
            report += f"  {warning}\n"

    report += f"{'=' * 80}\n"

    return report


def main():
    """Main entry point."""
    print("=" * 80)
    print("WALK-FORWARD VALIDATION - TIER 1 FVG System")
    print("=" * 80)
    print(f"\nConfiguration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}")
    print(f"Testing {len(PERIODS)} independent time periods...")

    period_results = []

    for i, (start_idx, end_idx, period_name) in enumerate(PERIODS, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(PERIODS)}] {period_name}")
        print(f"{'=' * 80}")

        try:
            # Load data subset
            df = load_mnq_data_subset(MNQ_DATA_PATH, start_idx, end_idx)

            # Transform to Dollar Bars
            dollar_bars = transform_to_dollar_bars(df)

            if len(dollar_bars) < 100:
                print(f"⚠️ Insufficient Dollar Bars: {len(dollar_bars)}")
                continue

            # Run backtest
            validator = Tier1Validator()
            metrics = validator.run_backtest(dollar_bars)

            # Create result
            result = PeriodResult(
                period_name=period_name,
                start_date=dollar_bars[0].timestamp.strftime("%Y-%m-%d") if dollar_bars else "N/A",
                end_date=dollar_bars[-1].timestamp.strftime("%Y-%m-%d") if dollar_bars else "N/A",
                total_trades=metrics['total_trades'],
                wins=metrics.get('wins', 0),
                losses=metrics.get('losses', 0),
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                avg_trades_per_day=metrics['avg_trades_per_day'],
                total_pnl=metrics['total_pnl'],
                expectancy=metrics['expectancy'],
            )

            period_results.append(result)

            # Print quick result
            print(f"Trades: {metrics['total_trades']:4d}, WR: {metrics['win_rate']:5.1f}%, "
                  f"PF: {metrics['profit_factor']:5.2f}, Freq: {metrics['avg_trades_per_day']:5.1f}/day")
            print(f"P&L: ${metrics['total_pnl']:7.2f}, Expectancy: ${metrics['expectancy']:5.2f}/trade")

            targets_met = sum([
                metrics['win_rate'] >= 60.0,
                metrics['profit_factor'] >= 1.7,
                8.0 <= metrics['avg_trades_per_day'] <= 15.0,
            ])
            print(f"Targets: {targets_met}/3 {'✅' if targets_met == 3 else '❌'}")

        except Exception as e:
            print(f"❌ Error processing period: {e}")
            logger.error(f"Period {i} failed", exc_info=True)
            continue

    print("\n" + "=" * 80)
    print("Walk-forward validation complete!")
    print("=" * 80)

    # Generate report
    report = generate_walkforward_report(period_results)
    print(report)

    # Save report
    report_path = Path("/root/Silver-Bullet-ML-BMAD/backtest_tier1_walkforward_validation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Exit code based on robustness
    periods_passing_all = sum([
        1 for r in period_results
        if r.total_trades > 0 and r.win_rate >= 60.0 and r.profit_factor >= 1.7
        and 8.0 <= r.avg_trades_per_day <= 15.0
    ])
    pass_rate = periods_passing_all / len([r for r in period_results if r.total_trades > 0]) if period_results else 0

    sys.exit(0 if pass_rate >= 0.5 else 1)


if __name__ == "__main__":
    main()

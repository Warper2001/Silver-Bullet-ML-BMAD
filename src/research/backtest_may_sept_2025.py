#!/usr/bin/env python3
"""TIER 1 FVG Backtest - May to September 2025.

Tests the optimal TIER 1 configuration (SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0)
on a 5-month period to validate performance across different market conditions.
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

# TIER 1 Optimal Configuration (from grid search)
SL_MULTIPLIER = 2.5
ATR_THRESHOLD = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS = 50.0

# Trading parameters
CONTRACTS_PER_TRADE = 1
MAX_HOLD_BARS = 10

# Sample size (May-September 2025)
# We need to estimate the right indices for this timeframe
# Total data: 795,296 bars spanning Dec 2023 to Mar 2026 (~28 months)
# May-Sep 2025 = 5 months = approximately 141,000 bars (based on distribution)
START_IDX = 250000  # Estimated start for May 2025
END_IDX = 400000      # Estimated end for September 2025


@dataclass
class BacktestResult:
    """Backtest results."""
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
    avg_win: float
    avg_loss: float


def load_mnq_data_subset(start_idx: int, end_idx: int) -> pd.DataFrame:
    """Load MNQ historical data subset."""
    print(f"Loading bars {start_idx:,}:{end_idx:,} from {MNQ_DATA_PATH}")

    if not MNQ_DATA_PATH.exists():
        raise FileNotFoundError(f"MNQ data file not found: {MNQ_DATA_PATH}")

    with open(MNQ_DATA_PATH, 'r') as f:
        content = f.read()
        data = json.loads(content)
        data = data[start_idx:end_idx]

    print(f"✓ Loaded {len(data):,} raw bars")

    df = pd.DataFrame(data)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    numeric_columns = ['High', 'Low', 'Open', 'Close', 'TotalVolume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('TimeStamp').reset_index(drop=True)

    print(f"✓ Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

    return df


def transform_to_dollar_bars(df: pd.DataFrame) -> list[DollarBar]:
    """Transform to Dollar Bars."""
    print("Transforming to Dollar Bars...")

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

    print(f"✓ Created {len(dollar_bars)} Dollar Bars")

    return dollar_bars


class Tier1Backtester:
    """Backtester for TIER 1 with optimal configuration."""

    def __init__(self):
        self.trades = []

    def run_backtest(self, dollar_bars: list[DollarBar]) -> dict:
        """Run backtest with optimal TIER 1 configuration."""
        print(f"Running TIER 1 backtest with SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}")

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

        print(f"Detecting FVG setups with TIER 1 filters...")

        # Detect and simulate trades
        for i in range(2, len(dollar_bars)):
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

            # Progress indicator
            if i % 1000 == 0:
                print(f"  Processing bar {i}/{len(dollar_bars)} ({100*i/len(dollar_bars):.1f}%)... Trades: {len(self.trades)}")

        print(f"✓ Detected and simulated {len(self.trades)} trades")

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

        # Bullish FVG pattern: candle 1 close > candle_3 open
        if candle_1.close <= candle_3.open:
            return None

        # Calculate gap range (ORIGINAL PROVEN LOGIC)
        gap_bottom = candle_3.low
        gap_top = candle_1.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * ATR_THRESHOLD):
            return None

        # Max gap size filter
        gap_dollars = gap_size * MNQ_CONTRACT_VALUE
        if gap_dollars > MAX_GAP_DOLLARS:
            return None

        # Volume filter
        up_volume = up_volumes[i]
        down_volume = down_volumes[i]

        if down_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = up_volume / down_volume

        if volume_ratio < VOLUME_RATIO_THRESHOLD:
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

        # Bearish FVG pattern: candle_1 close < candle_3 open
        if candle_1.close >= candle_3.open:
            return None

        # Calculate gap range (ORIGINAL PROVEN LOGIC)
        gap_bottom = candle_1.low
        gap_top = candle_3.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * ATR_THRESHOLD):
            return None

        # Max gap size filter
        gap_dollars = gap_size * MNQ_CONTRACT_VALUE
        if gap_dollars > MAX_GAP_DOLLARS:
            return None

        # Volume filter (bearish = down/up)
        up_volume = up_volumes[i]
        down_volume = down_volumes[i]

        if up_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = down_volume / up_volume

        if volume_ratio < VOLUME_RATIO_THRESHOLD:
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
            stop_loss = fvg.gap_range.bottom - (gap_size * SL_MULTIPLIER)
        else:  # short
            entry_price = fvg.gap_range.top
            take_profit = fvg.gap_range.bottom
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.top + (gap_size * SL_MULTIPLIER)

        if stop_loss <= 0 or take_profit <= 0:
            return

        entry_bar_index = fvg.bar_index + 1
        if entry_bar_index >= len(dollar_bars):
            return

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


def generate_report(result: BacktestResult) -> str:
    """Generate backtest report."""
    report = f"""
{'=' * 80}
TIER 1 FVG BACKTEST REPORT - {result.period_name}
{'=' * 80}

Configuration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}

Date Range: {result.start_date} to {result.end_date}

{'=' * 80}
BACKTEST RESULTS
{'=' * 80}

Total Trades: {result.total_trades}
Wins: {result.wins} | Losses: {result.losses}
Win Rate: {result.win_rate:.2f}%
Profit Factor: {result.profit_factor:.2f}
Trade Frequency: {result.avg_trades_per_day:.2f}/day
Total P&L: ${result.total_pnl:.2f}
Expectancy: ${result.expectancy:.2f}/trade
Avg Win: ${result.avg_win:.2f}
Avg Loss: ${result.avg_loss:.2f}

{'=' * 80}
TARGET VALIDATION (Win Rate >= 60%, PF >= 1.7, 8-15 trades/day)
{'=' * 80}

Win Rate (>=60%):       {'✅ PASS' if result.win_rate >= 60.0 else '❌ FAIL'} - {result.win_rate:.2f}%
Profit Factor (>=1.7):  {'✅ PASS' if result.profit_factor >= 1.7 else '❌ FAIL'} - {result.profit_factor:.2f}
Trade Freq (8-15/day): {'✅ PASS' if 8.0 <= result.avg_trades_per_day <= 15.0 else '❌ FAIL'} - {result.avg_trades_per_day:.2f}/day

{'=' * 80}
"""

    # Check if all targets met
    all_targets_met = (
        result.win_rate >= 60.0 and
        result.profit_factor >= 1.7 and
        8.0 <= result.avg_trades_per_day <= 15.0
    )

    if all_targets_met:
        report += "✅✅✅ ALL TARGETS ACHIEVED! ✅✅✅\n"
    else:
        report += f"Targets Met: {sum([result.win_rate >= 60.0, result.profit_factor >= 1.7, 8.0 <= result.avg_trades_per_day <= 15.0])}/3\n"

    report += f"{'=' * 80}\n"

    return report


def main():
    """Main entry point."""
    print("=" * 80)
    print("TIER 1 FVG BACKTEST - May to September 2025")
    print("=" * 80)
    print(f"Configuration: SL{SL_MULTIPLIER}x_ATR{ATR_THRESHOLD}_Vol{VOLUME_RATIO_THRESHOLD}_MaxGap${MAX_GAP_DOLLARS}")
    print(f"Data Range: {START_IDX:,} to {END_IDX:,} (estimated)")
    print()

    # Load data
    df = load_mnq_data_subset(START_IDX, END_IDX)

    # Transform to Dollar Bars
    dollar_bars = transform_to_dollar_bars(df)

    if len(dollar_bars) < 100:
        logger.error(f"Insufficient Dollar Bars: {len(dollar_bars)}")
        sys.exit(1)

    # Run backtest
    backtester = Tier1Backtester()
    metrics = backtester.run_backtest(dollar_bars)

    # Generate result
    result = BacktestResult(
        period_name="May-September 2025",
        start_date=dollar_bars[0].timestamp.strftime("%Y-%m-%d"),
        end_date=dollar_bars[-1].timestamp.strftime("%Y-%m-%d"),
        total_trades=metrics['total_trades'],
        wins=metrics.get('wins', 0),
        losses=metrics.get('losses', 0),
        win_rate=metrics['win_rate'],
        profit_factor=metrics['profit_factor'],
        avg_trades_per_day=metrics['avg_trades_per_day'],
        total_pnl=metrics['total_pnl'],
        expectancy=metrics['expectancy'],
        avg_win=metrics.get('avg_win', 0.0),
        avg_loss=metrics.get('avg_loss', 0.0),
    )

    # Print report
    report = generate_report(result)
    print(report)

    # Save report
    report_path = Path("/root/Silver-Bullet-ML-BMAD/backtest_may_sept_2025_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Exit code based on targets
    all_targets_met = (
        result.win_rate >= 60.0 and
        result.profit_factor >= 1.7 and
        8.0 <= result.avg_trades_per_day <= 15.0
    )

    sys.exit(0 if all_targets_met else 1)


if __name__ == "__main__":
    main()

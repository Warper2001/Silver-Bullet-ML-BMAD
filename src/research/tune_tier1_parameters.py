#!/usr/bin/env python3
"""TIER 1 Parameter Grid Search for optimal configuration.

Tests combinations of:
- Stop Loss multiplier (2.5×, 3.0×, 3.5×)
- ATR threshold (0.6, 0.7, 0.8)
- Volume ratio threshold (1.75, 2.0, 2.25)
- Max gap size cap ($50, $75, $100)

Target: Find configuration that meets all 3 targets:
- Win Rate >= 60%
- Profit Factor >= 1.7
- Trade Frequency 8-15/day
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import DollarBar, FVGEvent, GapRange
from src.detection.fvg_detection import (
    detect_bullish_fvg,
    detect_bearish_fvg,
)

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
SAMPLE_SIZE = 20000

# Trading parameters
CONTRACTS_PER_TRADE = 1
MAX_HOLD_BARS = 10


@dataclass
class Tier1Config:
    """TIER 1 configuration parameters."""
    sl_multiplier: float
    atr_threshold: float
    volume_ratio_threshold: float
    max_gap_dollars: float | None

    def __str__(self) -> str:
        return f"SL{self.sl_multiplier}x_ATR{self.atr_threshold}_Vol{self.volume_ratio_threshold}_MaxGap${self.max_gap_dollars}"


def load_mnq_data(data_path: Path = MNQ_DATA_PATH, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load MNQ historical data."""
    logger.info(f"Loading {sample_size} bars from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"MNQ data file not found: {data_path}")

    with open(data_path, 'r') as f:
        content = f.read()
        data = json.loads(content)
        data = data[:sample_size]

    df = pd.DataFrame(data)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    numeric_columns = ['High', 'Low', 'Open', 'Close', 'TotalVolume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('TimeStamp').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} bars: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

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


class Tier1ParameterBacktester:
    """Backtester for TIER 1 with configurable parameters."""

    def __init__(self, config: Tier1Config):
        self.config = config
        self.trades = []

    def run_backtest(self, dollar_bars: list[DollarBar]) -> dict:
        """Run backtest with specific configuration."""
        logger.info(f"Running backtest with {self.config}")

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

        # Calculate gap range (ORIGINAL PROVEN LOGIC)
        gap_bottom = candle_3.low
        gap_top = candle_1.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * self.config.atr_threshold):
            return None

        # Max gap size filter
        if self.config.max_gap_dollars is not None:
            gap_dollars = gap_size * MNQ_CONTRACT_VALUE
            if gap_dollars > self.config.max_gap_dollars:
                return None

        # Volume filter
        up_volume = up_volumes[i]
        down_volume = down_volumes[i]

        if down_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = up_volume / down_volume

        if volume_ratio < self.config.volume_ratio_threshold:
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

        # Calculate gap range (ORIGINAL PROVEN LOGIC)
        gap_bottom = candle_1.low
        gap_top = candle_3.high

        if gap_top <= gap_bottom:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * self.config.atr_threshold):
            return None

        # Max gap size filter
        if self.config.max_gap_dollars is not None:
            gap_dollars = gap_size * MNQ_CONTRACT_VALUE
            if gap_dollars > self.config.max_gap_dollars:
                return None

        # Volume filter (bearish = down/up)
        up_volume = up_volumes[i]
        down_volume = down_volumes[i]

        if up_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = down_volume / up_volume

        if volume_ratio < self.config.volume_ratio_threshold:
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
        """Simulate FVG trade with configurable SL."""
        direction = "long" if fvg.direction == "bullish" else "short"

        if direction == "long":
            entry_price = fvg.gap_range.bottom
            take_profit = fvg.gap_range.top
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.bottom - (gap_size * self.config.sl_multiplier)
        else:  # short
            entry_price = fvg.gap_range.top
            take_profit = fvg.gap_range.bottom
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.top + (gap_size * self.config.sl_multiplier)

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


def generate_parameter_grid_report(
    results: list[tuple[Tier1Config, dict]],
    dollar_bars: list[DollarBar],
) -> str:
    """Generate parameter tuning report."""
    if len(dollar_bars) > 0:
        start_date = dollar_bars[0].timestamp.strftime("%Y-%m-%d")
        end_date = dollar_bars[-1].timestamp.strftime("%Y-%m-%d")
    else:
        start_date = "N/A"
        end_date = "N/A"

    report = f"""
{'=' * 80}
TIER 1 PARAMETER TUNING REPORT - Grid Search Results
{'=' * 80}

Data: {SAMPLE_SIZE:,} raw bars → {len(dollar_bars)} Dollar Bars
Date Range: {start_date} to {end_date}
Combinations Tested: {len(results)}

{'=' * 80}
PARAMETER GRID RESULTS (Sorted by Profit Factor)
{'=' * 80}

"""

    # Sort by profit factor (descending)
    results_sorted = sorted(results, key=lambda x: x[1]['profit_factor'], reverse=True)

    for i, (config, metrics) in enumerate(results_sorted[:20], 1):  # Top 20
        report += f"""
{i}. {config}
   Trades: {metrics['total_trades']}, Wins: {metrics.get('wins', 0)}, Losses: {metrics.get('losses', 0)}
   WR: {metrics['win_rate']:.2f}%, PF: {metrics['profit_factor']:.2f}, Trades/Day: {metrics['avg_trades_per_day']:.2f}
   P&L: ${metrics['total_pnl']:.2f}, Expectancy: ${metrics['expectancy']:.2f}/trade
   Targets: {'✅' if metrics['win_rate'] >= 60.0 else '❌'} WR,
            {'✅' if metrics['profit_factor'] >= 1.7 else '❌'} PF,
            {'✅' if 8.0 <= metrics['avg_trades_per_day'] <= 15.0 else '❌'} Freq
   ALL TARGETS MET: {'✅ YES' if (metrics['win_rate'] >= 60.0 and metrics['profit_factor'] >= 1.7 and 8.0 <= metrics['avg_trades_per_day'] <= 15.0) else '❌ NO'}
"""

    # Find best configuration
    best_config = None
    for config, metrics in results_sorted:
        if (metrics['win_rate'] >= 60.0 and
            metrics['profit_factor'] >= 1.7 and
            8.0 <= metrics['avg_trades_per_day'] <= 15.0):
            best_config = (config, metrics)
            break

    if best_config:
        report += f"""
{'=' * 80}
BEST CONFIGURATION FOUND
{'=' * 80}
{best_config[0]}

Metrics:
Total Trades: {best_config[1]['total_trades']}
Win Rate: {best_config[1]['win_rate']:.2f}%
Profit Factor: {best_config[1]['profit_factor']:.2f}
Avg Trades/Day: {best_config[1]['avg_trades_per_day']:.2f}
Total P&L: ${best_config[1]['total_pnl']:.2f}
Expectancy: ${best_config[1]['expectancy']:.2f} per trade

ALL TARGETS MET: ✅
{'=' * 80}
"""
    else:
        report += f"""
{'=' * 80}
NO CONFIGURATION MET ALL TARGETS
{'=' * 80}
Closest configurations (by targets met):

"""

        # Find closest to meeting all targets
        results_with_scores = []
        for config, metrics in results:
            targets_met = sum([
                metrics['win_rate'] >= 60.0,
                metrics['profit_factor'] >= 1.7,
                8.0 <= metrics['avg_trades_per_day'] <= 15.0,
            ])
            results_with_scores.append((targets_met, config, metrics))

        results_with_scores.sort(key=lambda x: x[0], reverse=True)

        for targets_met, config, metrics in results_with_scores[:5]:
            report += f"""
- {config} ({targets_met}/3 targets met)
  WR: {metrics['win_rate']:.2f}%, PF: {metrics['profit_factor']:.2f}, Trades/Day: {metrics['avg_trades_per_day']:.2f}
"""

    report += f"{'=' * 80}\n"

    return report


def main():
    """Main entry point."""
    print("=" * 80)
    print("TIER 1 PARAMETER TUNING - Grid Search")
    print("=" * 80)

    # Load data
    df = load_mnq_data()

    # Transform to Dollar Bars
    dollar_bars = transform_to_dollar_bars(df)

    if len(dollar_bars) < 100:
        logger.error(f"Insufficient Dollar Bars: {len(dollar_bars)}")
        sys.exit(1)

    # Define parameter grid
    sl_multipliers = [2.5, 3.0, 3.5]
    atr_thresholds = [0.6, 0.7, 0.8]
    volume_ratios = [1.75, 2.0, 2.25]
    max_gap_sizes = [50.0, 75.0, 100.0]

    total_combinations = (
        len(sl_multipliers) *
        len(atr_thresholds) *
        len(volume_ratios) *
        len(max_gap_sizes)
    )

    print(f"\nTesting {total_combinations} parameter combinations...")
    print(f"  - SL multipliers: {sl_multipliers}")
    print(f"  - ATR thresholds: {atr_thresholds}")
    print(f"  - Volume ratios: {volume_ratios}")
    print(f"  - Max gap sizes: {max_gap_sizes}")

    results = []
    combination_num = 0

    for sl_mult, atr_thresh, vol_ratio, max_gap in product(
        sl_multipliers, atr_thresholds, volume_ratios, max_gap_sizes
    ):
        combination_num += 1

        # Skip if max_gap is not a viable cap
        if max_gap < 25.0:  # Minimum gap size is ~$25
            continue

        config = Tier1Config(
            sl_multiplier=sl_mult,
            atr_threshold=atr_thresh,
            volume_ratio_threshold=vol_ratio,
            max_gap_dollars=max_gap,
        )

        print(f"\n[{combination_num}/{total_combinations}] Testing {config}...")

        # Run backtest
        backtester = Tier1ParameterBacktester(config)
        metrics = backtester.run_backtest(dollar_bars)

        results.append((config, metrics))

        # Print quick result
        print(f"  Trades: {metrics['total_trades']:5d}, WR: {metrics['win_rate']:5.1f}%, "
              f"PF: {metrics['profit_factor']:5.2f}, Freq: {metrics['avg_trades_per_day']:5.1f}/day, "
              f"Targets: {sum([metrics['win_rate'] >= 60.0, metrics['profit_factor'] >= 1.7, 8.0 <= metrics['avg_trades_per_day'] <= 15.0])}/3")

        # Check if all targets met
        if (metrics['win_rate'] >= 60.0 and
            metrics['profit_factor'] >= 1.7 and
            8.0 <= metrics['avg_trades_per_day'] <= 15.0):
            print(f"  ✅✅✅ ALL TARGETS MET! ✅✅✅")
            break  # Early exit if we found a winner

    print("\n" + "=" * 80)
    print("Grid search complete!")
    print("=" * 80)

    # Generate report
    report = generate_parameter_grid_report(results, dollar_bars)
    print(report)

    # Save report
    report_path = Path("/root/Silver-Bullet-ML-BMAD/backtest_tier1_parameter_tuning_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Exit with appropriate code
    best_config = None
    for config, metrics in results:
        if (metrics['win_rate'] >= 60.0 and
            metrics['profit_factor'] >= 1.7 and
            8.0 <= metrics['avg_trades_per_day'] <= 15.0):
            best_config = config
            break

    sys.exit(0 if best_config else 1)


if __name__ == "__main__":
    main()

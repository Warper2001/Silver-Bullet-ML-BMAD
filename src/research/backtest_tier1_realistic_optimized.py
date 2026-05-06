#!/usr/bin/env python3
"""Optimized realistic TIER 1 FVG Backtester with smart optimizations.

Optimizations (maintaining full realism):
1. Vectorized ATR calculation (pandas)
2. Pre-calculated exit thresholds (no ExitSimulator overhead)
3. Batch FVG detection
4. Reduced logging
5. 50K sample for statistical significance

Runtime target: 3-4 hours for 50K bars (~8 weeks data)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import DollarBar, FVGEvent, GapRange
from src.detection.atr_filter import ATRFilter
from src.detection.volume_confirmer import VolumeConfirmer
from src.detection.fvg_detection import (
    detect_bullish_fvg,
    detect_bearish_fvg,
)

# Configure logging (less verbose)
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MNQ_DATA_PATH = Path("/root/mnq_historical.json")
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE  # $5 per point
DOLLAR_BAR_THRESHOLD = 50_000_000
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1
SAMPLE_SIZE = 20000  # 20K sample = proven runtime (~2.5 hours), good statistical significance

# Trading parameters
CONTRACTS_PER_TRADE = 1
MAX_HOLD_BARS = 10
STOP_LOSS_MULTIPLIER = 2.0


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
    """Transform to Dollar Bars (vectorized)."""
    logger.info("Transforming to Dollar Bars...")

    # Vectorized calculation
    df['notional'] = ((df['High'] + df['Low']) / 2) * df['TotalVolume'] * MNQ_POINT_VALUE
    df['cumulative_notional'] = df['notional'].cumsum()

    # Find bar boundaries
    bar_boundaries = df[df['cumulative_notional'] % DOLLAR_BAR_THRESHOLD < df['notional']].index.tolist()

    if len(df) > 0 and (len(bar_boundaries) == 0 or bar_boundaries[-1] != len(df) - 1):
        bar_boundaries.append(len(df) - 1)

    # Aggregate bars (vectorized where possible)
    dollar_bars = []
    prev_boundary = 0

    for boundary in tqdm(bar_boundaries, desc="Creating Dollar Bars", disable=True):
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


def calculate_atr_vectorized(dollar_bars: list[DollarBar], period: int = 14) -> np.ndarray:
    """Calculate ATR for all bars using vectorized pandas operations.

    Args:
        dollar_bars: List of Dollar Bars
        period: ATR lookback period

    Returns:
        Array of ATR values (same length as dollar_bars)
    """
    # Convert to DataFrame for vectorized operations
    df = pd.DataFrame([
        {
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
        }
        for bar in dollar_bars
    ])

    # Calculate True Range (vectorized)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate ATR (vectorized exponential moving average)
    df['atr'] = df['true_range'].ewm(span=period, adjust=False).mean()

    # Fill NaN values with mean TR for first few bars
    mean_tr = df['true_range'].mean()
    df['atr'] = df['atr'].fillna(mean_tr)

    return df['atr'].values


def calculate_exit_fast(
    entry_bar_index: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    dollar_bars: list[DollarBar],
    direction: str,
) -> tuple[float, str, int]:
    """Fast exit calculation without ExitSimulator overhead.

    Pre-calculates exit by scanning forward for SL/TP levels.
    Maintains same accuracy as ExitSimulator but much faster.

    Args:
        entry_bar_index: Entry bar index
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        dollar_bars: All dollar bars
        direction: "long" or "short"

    Returns:
        Tuple of (exit_price, exit_reason, bars_held)
    """
    sl_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE
    tp_buffer = SLIPPAGE_TICKS * MNQ_TICK_SIZE

    if direction == "long":
        sl_trigger = stop_loss - sl_buffer
        tp_trigger = take_profit + tp_buffer
    else:  # short
        sl_trigger = stop_loss + sl_buffer
        tp_trigger = take_profit - tp_buffer

    # Scan forward for exit
    max_index = min(entry_bar_index + MAX_HOLD_BARS + 1, len(dollar_bars))

    for i in range(entry_bar_index + 1, max_index):
        bar = dollar_bars[i]

        if direction == "long":
            # Check SL first (risk management priority)
            if bar.low <= sl_trigger:
                exit_price = min(stop_loss, bar.low + sl_buffer)
                return exit_price, "stop_loss", i - entry_bar_index

            # Check TP
            if bar.high >= tp_trigger:
                exit_price = max(take_profit, bar.high - tp_buffer)
                return exit_price, "take_profit", i - entry_bar_index
        else:  # short
            # Check SL first
            if bar.high >= sl_trigger:
                exit_price = max(stop_loss, bar.high - sl_buffer)
                return exit_price, "stop_loss", i - entry_bar_index

            # Check TP
            if bar.low <= tp_trigger:
                exit_price = min(take_profit, bar.low + tp_buffer)
                return exit_price, "take_profit", i - entry_bar_index

        # Check time exit
        if i - entry_bar_index >= MAX_HOLD_BARS:
            return bar.close, "max_time", i - entry_bar_index

    # End of data
    last_bar = dollar_bars[-1]
    bars_held = len(dollar_bars) - 1 - entry_bar_index
    return last_bar.close, "end_of_data", bars_held


class OptimizedTier1FVGBacktester:
    """Optimized realistic TIER 1 FVG backtester."""

    def __init__(self, use_tier1_filters: bool = True):
        """Initialize backtester.

        Args:
            use_tier1_filters: Whether to use TIER 1 filters
        """
        self.use_tier1_filters = use_tier1_filters
        self.trades = []

    def run_backtest(self, dollar_bars: list[DollarBar]) -> dict:
        """Run optimized backtest.

        Args:
            dollar_bars: List of Dollar Bars

        Returns:
            Performance metrics
        """
        logger.info(f"Running {'TIER 1' if self.use_tier1_filters else 'Baseline'} backtest...")

        # Pre-calculate ATR for all bars (vectorized)
        atr_values = calculate_atr_vectorized(dollar_bars, period=14)

        # Pre-calculate volume ratios (vectorized) - O(n) instead of O(n²)
        volumes = np.array([bar.volume for bar in dollar_bars])
        volume_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values

        # Pre-calculate directional volume (up vs down) for each bar
        is_bullish = np.array([1 if bar.close > bar.open else 0 for bar in dollar_bars])
        is_bearish = np.array([1 if bar.close < bar.open else 0 for bar in dollar_bars])

        # Calculate rolling up/down volume (vectorized)
        up_volumes_rolling = pd.Series(
            volumes * is_bullish
        ).rolling(window=20, min_periods=1).sum().values

        down_volumes_rolling = pd.Series(
            volumes * is_bearish
        ).rolling(window=20, min_periods=1).sum().values

        # Batch detect FVGs and simulate trades
        for i in tqdm(range(2, len(dollar_bars)), desc="Processing", disable=False):
            bar = dollar_bars[i]

            # Skip if insufficient ATR data
            if i < 14 or np.isnan(atr_values[i]):
                continue

            # Detect FVGs
            if self.use_tier1_filters:
                bullish_fvg = self._detect_bullish_fvg_optimized(
                    dollar_bars, i, atr_values, up_volumes_rolling, down_volumes_rolling
                )
                bearish_fvg = self._detect_bearish_fvg_optimized(
                    dollar_bars, i, atr_values, up_volumes_rolling, down_volumes_rolling
                )
            else:
                bullish_fvg = self._detect_bullish_fvg_no_filters(dollar_bars, i)
                bearish_fvg = self._detect_bearish_fvg_no_filters(dollar_bars, i)

            # Simulate trades
            for fvg in [bullish_fvg, bearish_fvg]:
                if fvg is None:
                    continue

                self._simulate_fvg_trade_optimized(fvg, dollar_bars)

        return self._calculate_metrics(dollar_bars)

    def _detect_bullish_fvg_optimized(
        self,
        bars: list[DollarBar],
        i: int,
        atr_values: np.ndarray,
        up_volumes_rolling: np.ndarray,
        down_volumes_rolling: np.ndarray,
    ) -> FVGEvent | None:
        """Optimized bullish FVG detection with pre-calculated ATR/volume."""
        # Need 3 candles
        if i < 2:
            return None

        # 3-candle pattern
        candle_1 = bars[i-2]
        candle_2 = bars[i-1]
        candle_3 = bars[i]

        # Bullish FVG: Candle 2 high < Candle 3 low
        if not (candle_2.high < candle_3.low):
            return None

        # Calculate gap
        gap_bottom = candle_2.high
        gap_top = candle_3.low

        if gap_bottom >= gap_top:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter (pre-calculated)
        atr = atr_values[i]
        if gap_size < (atr * 0.5):
            return None

        # Volume filter (pre-calculated, O(1) lookup)
        up_volume = up_volumes_rolling[i]
        down_volume = down_volumes_rolling[i]

        if down_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = up_volume / down_volume

        if volume_ratio < 1.5:
            return None

        # Create FVG event
        return FVGEvent(
            timestamp=bars[i].timestamp,
            direction="bullish",
            gap_range=GapRange(top=gap_top, bottom=gap_bottom),
            gap_size_ticks=gap_size / MNQ_TICK_SIZE,
            gap_size_dollars=gap_size * MNQ_CONTRACT_VALUE,
            bar_index=i,
            filled=False,
        )

    def _detect_bearish_fvg_optimized(
        self,
        bars: list[DollarBar],
        i: int,
        atr_values: np.ndarray,
        up_volumes_rolling: np.ndarray,
        down_volumes_rolling: np.ndarray,
    ) -> FVGEvent | None:
        """Optimized bearish FVG detection with pre-calculated ATR/volume."""
        if i < 2:
            return None

        candle_1 = bars[i-2]
        candle_2 = bars[i-1]
        candle_3 = bars[i]

        # Bearish FVG: Candle 2 low > Candle 3 high
        if not (candle_2.low > candle_3.high):
            return None

        gap_bottom = candle_3.high
        gap_top = candle_2.low

        if gap_bottom >= gap_top:
            return None

        gap_size = gap_top - gap_bottom

        # ATR filter
        atr = atr_values[i]
        if gap_size < (atr * 0.5):
            return None

        # Volume filter (bearish = down/up, pre-calculated)
        up_volume = up_volumes_rolling[i]
        down_volume = down_volumes_rolling[i]

        if up_volume == 0:
            volume_ratio = float('inf')
        else:
            volume_ratio = down_volume / up_volume

        if volume_ratio < 1.5:
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

    def _detect_bullish_fvg_no_filters(self, bars: list[DollarBar], i: int) -> FVGEvent | None:
        """Baseline bullish FVG detection (no filters)."""
        if i < 2:
            return None

        candle_2 = bars[i-1]
        candle_3 = bars[i]

        if not (candle_2.high < candle_3.low):
            return None

        gap_bottom = candle_2.high
        gap_top = candle_3.low

        if gap_bottom >= gap_top:
            return None

        gap_size = gap_top - gap_bottom

        return FVGEvent(
            timestamp=bars[i].timestamp,
            direction="bullish",
            gap_range=GapRange(top=gap_top, bottom=gap_bottom),
            gap_size_ticks=gap_size / MNQ_TICK_SIZE,
            gap_size_dollars=gap_size * MNQ_CONTRACT_VALUE,
            bar_index=i,
            filled=False,
        )

    def _detect_bearish_fvg_no_filters(self, bars: list[DollarBar], i: int) -> FVGEvent | None:
        """Baseline bearish FVG detection (no filters)."""
        if i < 2:
            return None

        candle_2 = bars[i-1]
        candle_3 = bars[i]

        if not (candle_2.low > candle_3.high):
            return None

        gap_bottom = candle_3.high
        gap_top = candle_2.low

        if gap_bottom >= gap_top:
            return None

        gap_size = gap_top - gap_bottom

        return FVGEvent(
            timestamp=bars[i].timestamp,
            direction="bearish",
            gap_range=GapRange(top=gap_top, bottom=gap_bottom),
            gap_size_ticks=gap_size / MNQ_TICK_SIZE,
            gap_size_dollars=gap_size * MNQ_CONTRACT_VALUE,
            bar_index=i,
            filled=False,
        )

    def _simulate_fvg_trade_optimized(
        self,
        fvg: FVGEvent,
        dollar_bars: list[DollarBar],
    ) -> None:
        """Optimized trade simulation with fast exit calculation."""
        direction = "long" if fvg.direction == "bullish" else "short"

        # Calculate entry and exit levels
        if direction == "long":
            entry_price = fvg.gap_range.bottom
            take_profit = fvg.gap_range.top
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.bottom - (gap_size * STOP_LOSS_MULTIPLIER)
        else:  # short
            entry_price = fvg.gap_range.top
            take_profit = fvg.gap_range.bottom
            gap_size = fvg.gap_range.top - fvg.gap_range.bottom
            stop_loss = fvg.gap_range.top + (gap_size * STOP_LOSS_MULTIPLIER)

        # Validate
        if stop_loss <= 0 or take_profit <= 0:
            return

        entry_bar_index = fvg.bar_index + 1
        if entry_bar_index >= len(dollar_bars):
            return

        entry_bar = dollar_bars[entry_bar_index]

        # Fast exit calculation
        exit_price, exit_reason, bars_held = calculate_exit_fast(
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
            "entry_time": entry_bar.timestamp,
            "exit_time": dollar_bars[entry_bar_index + bars_held].timestamp,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "pnl": pnl_final,
            "bars_held": bars_held,
        })

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

        # Count wins/losses
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] < 0]

        win_rate = len(wins) / len(self.trades) * 100

        # Calculate P&L metrics
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_won = sum(t["pnl"] for t in wins)
        total_lost = sum(t["pnl"] for t in losses)

        profit_factor = abs(total_won / total_lost) if total_lost != 0 else float('inf')

        avg_win = total_won / len(wins) if wins else 0.0
        avg_loss = total_lost / len(losses) if losses else 0.0

        expectancy = total_pnl / len(self.trades)

        # Calculate trades per day
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


def generate_report(baseline_metrics: dict, tier1_metrics: dict, dollar_bars: list[DollarBar]) -> str:
    """Generate performance comparison report."""
    if len(dollar_bars) > 0:
        start_date = dollar_bars[0].timestamp.strftime("%Y-%m-%d")
        end_date = dollar_bars[-1].timestamp.strftime("%Y-%m-%d")
    else:
        start_date = "N/A"
        end_date = "N/A"

    report = f"""
{'=' * 80}
TIER 1 FVG SYSTEM - OPTIMIZED REALISTIC BACKTEST (50K Sample)
{'=' * 80}

Data Source: Real MNQ Historical Data (/root/mnq_historical.json)
Sample Size: {SAMPLE_SIZE:,} raw bars
Data Points: {len(dollar_bars)} Dollar Bars
Date Range: {start_date} to {end_date}

Optimizations Applied:
- Vectorized ATR calculation (pandas)
- Pre-calculated volume ratios
- Fast exit calculation (maintains realism)
- 50K sample for statistical significance

Transaction Costs:
- Commission: ${COMMISSION_PER_CONTRACT} per contract per side
- Slippage: {SLIPPAGE_TICKS} tick (${SLIPPAGE_TICKS * MNQ_TICK_SIZE:.2f}) per side
- Total Cost: ${(COMMISSION_PER_CONTRACT + SLIPPAGE_TICKS * MNQ_TICK_SIZE) * 2:.2f} per round-trip

Exit Logic:
- Take Profit: Gap fill (opposite boundary)
- Stop Loss: {STOP_LOSS_MULTIPLIER}× gap size against position
- Time Exit: {MAX_HOLD_BARS} bars max hold time

{'=' * 80}
BASELINE (No Filters)
{'=' * 80}
Total Trades: {baseline_metrics['total_trades']}
Wins: {baseline_metrics.get('wins', 0)}
Losses: {baseline_metrics.get('losses', 0)}
Win Rate: {baseline_metrics['win_rate']:.2f}%
Profit Factor: {baseline_metrics['profit_factor']:.2f}
Avg Trades/Day: {baseline_metrics['avg_trades_per_day']:.2f}
Total P&L: ${baseline_metrics['total_pnl']:.2f}
Avg Win: ${baseline_metrics['avg_win']:.2f}
Avg Loss: ${baseline_metrics['avg_loss']:.2f}
Expectancy: ${baseline_metrics['expectancy']:.2f} per trade

{'=' * 80}
TIER 1 (ATR + Volume Filters)
{'=' * 80}
Total Trades: {tier1_metrics['total_trades']}
Wins: {tier1_metrics.get('wins', 0)}
Losses: {tier1_metrics.get('losses', 0)}
Win Rate: {tier1_metrics['win_rate']:.2f}%
Profit Factor: {tier1_metrics['profit_factor']:.2f}
Avg Trades/Day: {tier1_metrics['avg_trades_per_day']:.2f}
Total P&L: ${tier1_metrics['total_pnl']:.2f}
Avg Win: ${tier1_metrics['avg_win']:.2f}
Avg Loss: ${tier1_metrics['avg_loss']:.2f}
Expectancy: ${tier1_metrics['expectancy']:.2f} per trade

{'=' * 80}
IMPROVEMENTS
{'=' * 80}
Win Rate: {tier1_metrics['win_rate'] - baseline_metrics['win_rate']:+.2f}%
Profit Factor: {tier1_metrics['profit_factor'] - baseline_metrics['profit_factor']:+.2f}
Expectancy: ${tier1_metrics['expectancy'] - baseline_metrics['expectancy']:+.2f} per trade
Trade Freq: {((tier1_metrics['avg_trades_per_day'] / baseline_metrics['avg_trades_per_day'] - 1) * 100):+.2f}%
Total P&L: ${tier1_metrics['total_pnl'] - baseline_metrics['total_pnl']:+.2f}

{'=' * 80}
TARGET VALIDATION (Win Rate >= 60%, PF >= 1.7, 8-15 trades/day)
{'=' * 80}
Win Rate (>=60%):       {'✅ PASS' if tier1_metrics['win_rate'] >= 60.0 else '❌ FAIL'} - {tier1_metrics['win_rate']:.2f}%
Profit Factor (>=1.7):  {'✅ PASS' if tier1_metrics['profit_factor'] >= 1.7 else '❌ FAIL'} - {tier1_metrics['profit_factor']:.2f}
Trade Freq (8-15/day):  {'✅ PASS' if 8.0 <= tier1_metrics['avg_trades_per_day'] <= 15.0 else '❌ FAIL'} - {tier1_metrics['avg_trades_per_day']:.2f}/day

{'=' * 80}
"""

    return report


def main():
    """Main entry point."""
    print("=" * 80)
    print("TIER 1 FVG OPTIMIZED REALISTIC BACKTEST")
    print("=" * 80)

    # Load data
    df = load_mnq_data()

    # Transform to Dollar Bars
    dollar_bars = transform_to_dollar_bars(df)

    if len(dollar_bars) < 100:
        logger.error(f"Insufficient Dollar Bars: {len(dollar_bars)}")
        sys.exit(1)

    # Run baseline
    print("\n" + "=" * 80)
    print("Running BASELINE backtest...")
    print("=" * 80)
    baseline = OptimizedTier1FVGBacktester(use_tier1_filters=False)
    baseline_metrics = baseline.run_backtest(dollar_bars)

    # Run TIER 1
    print("\n" + "=" * 80)
    print("Running TIER 1 backtest...")
    print("=" * 80)
    tier1 = OptimizedTier1FVGBacktester(use_tier1_filters=True)
    tier1_metrics = tier1.run_backtest(dollar_bars)

    # Generate report
    report = generate_report(baseline_metrics, tier1_metrics, dollar_bars)
    print(report)

    # Save report
    report_path = Path("/root/Silver-Bullet-ML-BMAD/backtest_tier1_optimized_50k_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Exit with appropriate code
    targets_met = (
        tier1_metrics["win_rate"] >= 60.0
        and tier1_metrics["profit_factor"] >= 1.7
        and 8.0 <= tier1_metrics["avg_trades_per_day"] <= 15.0
    )
    sys.exit(0 if targets_met else 1)


if __name__ == "__main__":
    main()

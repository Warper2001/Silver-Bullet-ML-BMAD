#!/usr/bin/env python3
"""Hybrid optimized realistic backtest - original detection + pre-calculated filters.

Uses original proven FVG detection functions with vectorized filter calculations.
Maintains full realism while optimizing only the safe parts.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import DollarBar, FVGEvent
from src.detection.atr_filter import ATRFilter
from src.detection.volume_confirmer import VolumeConfirmer
from src.detection.fvg_detection import (
    detect_bullish_fvg,
    detect_bearish_fvg,
)
from src.research.backtest_engine import BacktestEngine

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
STOP_LOSS_MULTIPLIER = 2.0


class VectorizedATRFilter(ATRFilter):
    """ATR Filter with vectorized calculation support."""

    def __init__(self, lookback_period: int = 14, atr_threshold: float = 0.5):
        super().__init__(lookback_period, atr_threshold)
        self._atr_cache = None

    def pre_calculate_atr(self, dollar_bars: list[DollarBar]) -> np.ndarray:
        """Pre-calculate ATR for all bars (vectorized)."""
        # Convert to DataFrame
        df = pd.DataFrame([
            {'high': bar.high, 'low': bar.low, 'close': bar.close}
            for bar in dollar_bars
        ])

        # Calculate True Range (vectorized)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR
        atr_values = df['true_range'].ewm(span=self.lookback_period, adjust=False).mean().values

        # Fill NaN
        mean_tr = df['true_range'].mean()
        atr_values = np.nan_to_num(atr_values, nan=mean_tr)

        self._atr_cache = atr_values
        return atr_values

    def get_atr(self, bar_index: int) -> float:
        """Get pre-calculated ATR for specific bar."""
        if self._atr_cache is None:
            raise ValueError("Must call pre_calculate_atr() first")
        return self._atr_cache[bar_index]

    def passes_filter(self, gap_size: float, bar_index: int) -> bool:
        """Check if gap passes ATR filter (using cached ATR)."""
        atr = self.get_atr(bar_index)
        return gap_size >= (atr * self.atr_threshold)


class VectorizedVolumeConfirmer(VolumeConfirmer):
    """Volume Confirmer with vectorized calculation support."""

    def __init__(self, lookback_period: int = 20, volume_ratio_threshold: float = 1.5):
        super().__init__(lookback_period, volume_ratio_threshold)
        self._up_volume_cache = None
        self._down_volume_cache = None

    def pre_calculate_volume_ratios(self, dollar_bars: list[DollarBar]) -> tuple[np.ndarray, np.ndarray]:
        """Pre-calculate up/down volume ratios (vectorized)."""
        volumes = np.array([bar.volume for bar in dollar_bars])

        # Create directional arrays
        is_bullish = np.array([1 if bar.close > bar.open else 0 for bar in dollar_bars])
        is_bearish = np.array([1 if bar.close < bar.open else 0 for bar in dollar_bars])

        # Calculate rolling up/down volume
        up_volumes = pd.Series(volumes * is_bullish).rolling(
            window=self.lookback_period, min_periods=1
        ).sum().values

        down_volumes = pd.Series(volumes * is_bearish).rolling(
            window=self.lookback_period, min_periods=1
        ).sum().values

        self._up_volume_cache = up_volumes
        self._down_volume_cache = down_volumes

        return up_volumes, down_volumes

    def get_volume_ratio(self, bar_index: int, direction: str) -> float:
        """Get pre-calculated volume ratio for specific bar."""
        if self._up_volume_cache is None:
            raise ValueError("Must call pre_calculate_volume_ratios() first")

        up_volume = self._up_volume_cache[bar_index]
        down_volume = self._down_volume_cache[bar_index]

        if direction == "bullish":
            if down_volume == 0:
                return float('inf')
            return up_volume / down_volume
        else:  # bearish
            if up_volume == 0:
                return float('inf')
            return down_volume / up_volume

    def passes_filter(self, bar_index: int, direction: str) -> bool:
        """Check if bar passes volume filter (using cached ratios)."""
        ratio = self.get_volume_ratio(bar_index, direction)
        return ratio >= self.volume_ratio_threshold


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


def simulate_trade_exit(
    entry_bar_index: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    dollar_bars: list[DollarBar],
    direction: str,
) -> tuple[float, str, int]:
    """Simulate trade exit (fast version without ExitSimulator overhead)."""
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


class HybridTier1Backtester:
    """Hybrid backtester using original detection + pre-calculated filters."""

    def __init__(self, use_tier1_filters: bool = True):
        self.use_tier1_filters = use_tier1_filters
        self.engine = BacktestEngine(initial_capital=100000.0)
        self.trades = []

    def run_backtest(self, dollar_bars: list[DollarBar]) -> dict:
        """Run backtest with original detection + vectorized filters."""
        logger.info(f"Running {'TIER 1' if self.use_tier1_filters else 'Baseline'} backtest...")

        # Pre-calculate filters (vectorized)
        if self.use_tier1_filters:
            atr_filter = VectorizedATRFilter(lookback_period=14, atr_threshold=0.5)
            volume_confirmer = VectorizedVolumeConfirmer(lookback_period=20, volume_ratio_threshold=1.5)

            # Pre-calculate ATR and volume ratios
            atr_filter.pre_calculate_atr(dollar_bars)
            volume_confirmer.pre_calculate_volume_ratios(dollar_bars)
        else:
            atr_filter = None
            volume_confirmer = None

        # Detect and simulate trades
        for i in tqdm(range(2, len(dollar_bars)), desc="Processing", disable=False):
            # Use original proven detection functions
            bullish_fvg = detect_bullish_fvg(dollar_bars, i, atr_filter, volume_confirmer)
            bearish_fvg = detect_bearish_fvg(dollar_bars, i, atr_filter, volume_confirmer)

            # Simulate trades
            for fvg in [bullish_fvg, bearish_fvg]:
                if fvg is None:
                    continue

                self._simulate_fvg_trade(fvg, dollar_bars)

        return self._calculate_metrics(dollar_bars)

    def _simulate_fvg_trade(self, fvg: FVGEvent, dollar_bars: list[DollarBar]) -> None:
        """Simulate FVG trade with realistic exit logic."""
        direction = "long" if fvg.direction == "bullish" else "short"

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

        if stop_loss <= 0 or take_profit <= 0:
            return

        entry_bar_index = fvg.bar_index + 1
        if entry_bar_index >= len(dollar_bars):
            return

        entry_bar = dollar_bars[entry_bar_index]

        # Simulate exit
        exit_price, exit_reason, bars_held = simulate_trade_exit(
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
        self.engine.add_trade(
            entry_time=entry_bar.timestamp,
            exit_time=dollar_bars[entry_bar_index + bars_held].timestamp,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bars_held=bars_held,
        )

        # Adjust for transaction costs
        self.engine.trades[-1].pnl = pnl_final

    def _calculate_metrics(self, dollar_bars: list[DollarBar]) -> dict:
        """Calculate performance metrics."""
        trades = self.engine.get_all_trades()

        if not trades:
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

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        win_rate = len(wins) / len(trades) * 100
        total_pnl = sum(t.pnl for t in trades)
        total_won = sum(t.pnl for t in wins)
        total_lost = sum(t.pnl for t in losses)
        profit_factor = abs(total_won / total_lost) if total_lost != 0 else float('inf')
        avg_win = total_won / len(wins) if wins else 0.0
        avg_loss = total_lost / len(losses) if losses else 0.0
        expectancy = total_pnl / len(trades)

        if len(dollar_bars) > 0:
            time_span_days = (dollar_bars[-1].timestamp - dollar_bars[0].timestamp).total_seconds() / 86400
            avg_trades_per_day = len(trades) / time_span_days if time_span_days > 0 else 0.0
        else:
            avg_trades_per_day = 0.0

        return {
            "total_trades": len(trades),
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
TIER 1 FVG SYSTEM - HYBRID OPTIMIZED BACKTEST (Original Detection + Vectorized Filters)
{'=' * 80}

Data Source: Real MNQ Historical Data (/root/mnq_historical.json)
Sample Size: {SAMPLE_SIZE:,} raw bars
Data Points: {len(dollar_bars)} Dollar Bars
Date Range: {start_date} to {end_date}

Optimizations Applied:
- Original proven FVG detection (detect_bullish_fvg, detect_bearish_fvg)
- Vectorized ATR pre-calculation
- Vectorized volume ratio pre-calculation
- Fast exit simulation (maintains realism)
- 20K sample (proven runtime ~2-2.5 hours)

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
    print("TIER 1 FVG HYBRID OPTIMIZED BACKTEST")
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
    baseline = HybridTier1Backtester(use_tier1_filters=False)
    baseline_metrics = baseline.run_backtest(dollar_bars)

    # Run TIER 1
    print("\n" + "=" * 80)
    print("Running TIER 1 backtest...")
    print("=" * 80)
    tier1 = HybridTier1Backtester(use_tier1_filters=True)
    tier1_metrics = tier1.run_backtest(dollar_bars)

    # Generate report
    report = generate_report(baseline_metrics, tier1_metrics, dollar_bars)
    print(report)

    # Save report
    report_path = Path("/root/Silver-Bullet-ML-BMAD/backtest_tier1_hybrid_report.txt")
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

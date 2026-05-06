#!/usr/bin/env python3
"""
TIER 1 FVG Parameter Sensitivity Analysis

Tests robustness of optimal configuration by analyzing performance degradation
when parameters are varied slightly around the optimal values.

Optimal Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0

Tests variations:
- SL Multiplier: [2.3, 2.4, 2.5, 2.6, 2.7] (±0.2 around optimal)
- ATR Threshold: [0.65, 0.7, 0.75] (±0.05 around optimal)
- Volume Ratio: [2.1, 2.25, 2.4] (±0.15 around optimal)

Total combinations: 5 × 3 × 3 = 45
"""

import sys
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import DollarBar, FVGEvent, GapRange


# Constants
MNQ_DATA_PATH = Path("/root/mnq_historical.json")
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE
DOLLAR_BAR_THRESHOLD = 50_000_000
COMMISSION_PER_CONTRACT = 0.45
SLIPPAGE_TICKS = 1
TRANSACTION_COST = (COMMISSION_PER_CONTRACT * 2 +
                   SLIPPAGE_TICKS * MNQ_TICK_SIZE * MNQ_POINT_VALUE * 2)

# Sample size for sensitivity analysis
START_IDX = 250000
END_IDX = 280000  # 30K bars = ~18K dollar bars (faster than full backtest)


@dataclass
class SensitivityResult:
    """Results from single parameter combination."""
    sl_multiplier: float
    atr_threshold: float
    volume_ratio: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    expectancy: float
    avg_win: float
    avg_loss: float


class Tier1Backtester:
    """TIER 1 backtest with realistic P&L calculation."""

    def __init__(
        self,
        sl_multiplier: float,
        atr_threshold: float,
        volume_ratio_threshold: float,
        max_gap_dollars: float = 50.0
    ):
        self.sl_multiplier = sl_multiplier
        self.atr_threshold = atr_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        self.max_gap_dollars = max_gap_dollars

    def run_backtest(self, dollar_bars: List[DollarBar]) -> Dict:
        """Run backtest and return metrics."""
        print(f"Running backtest: SL{self.sl_multiplier}x_ATR{self.atr_threshold}_Vol{self.volume_ratio_threshold}")

        trades = []
        equity_curve = []
        running_equity = 0.0

        # Pre-calculate ATR values
        high_values = np.array([bar.high for bar in dollar_bars])
        low_values = np.array([bar.low for bar in dollar_bars])
        close_values = np.array([bar.close for bar in dollar_bars])

        tr_values = np.maximum(
            high_values[1:] - low_values[1:],
            np.maximum(
                np.abs(high_values[1:] - close_values[:-1]),
                np.abs(low_values[1:] - close_values[:-1])
            )
        )

        atr_values = pd.Series(tr_values).ewm(span=14, adjust=False).mean().values
        atr_values = np.concatenate([[np.mean(tr_values[:14])], atr_values])  # Pad first value
        atr_values = np.nan_to_num(atr_values, nan=np.mean(tr_values))

        # Pre-calculate volume ratios
        volumes = np.array([bar.volume for bar in dollar_bars])
        is_bullish = np.array([1 if bar.close > bar.open else 0 for bar in dollar_bars])
        is_bearish = np.array([1 if bar.close < bar.open else 0 for bar in dollar_bars])

        up_volumes = pd.Series(volumes * is_bullish).rolling(window=20, min_periods=1).sum().values
        down_volumes = pd.Series(volumes * is_bearish).rolling(window=20, min_periods=1).sum().values

        print("  Detecting FVG setups...")

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

                trade_result = self._simulate_fvg_trade(fvg, dollar_bars)

                if trade_result:
                    trades.append(trade_result)
                    running_equity += trade_result['pnl']
                    equity_curve.append({
                        'trade': len(trades),
                        'equity': running_equity
                    })

        # Calculate metrics
        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total_won = sum(t['pnl'] for t in wins)
        total_lost = sum(abs(t['pnl']) for t in losses)
        avg_win = total_won / len(wins) if wins else 0.0
        avg_loss = total_lost / len(losses) if losses else 0.0

        win_rate = (len(wins) / len(trades)) * 100
        profit_factor = total_won / total_lost if total_lost > 0 else 0.0
        expectancy = sum(t['pnl'] for t in trades) / len(trades)

        print(f"  ✓ {len(trades)} trades, WR: {win_rate:.1f}%, PF: {profit_factor:.2f}")

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": sum(t['pnl'] for t in trades),
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    def _detect_bullish_fvg_with_filters(
        self,
        bars: List[DollarBar],
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

        # Calculate gap range
        gap_bottom = candle_3.low
        gap_top = candle_1.high

        if gap_top <= gap_bottom:
            return None

        gap_size = (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE

        # ATR filter: gap must be >= threshold * ATR
        atr_value = atr_values[i]
        if gap_size < self.atr_threshold * atr_value:
            return None

        # Volume confirmation: up volume >= threshold * down volume
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
        bars: List[DollarBar],
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

        # Bearish FVG pattern: candle 1 close < candle_3 open
        if candle_1.close >= candle_3.open:
            return None

        # Calculate gap range
        gap_bottom = candle_1.low
        gap_top = candle_3.high

        if gap_top <= gap_bottom:
            return None

        gap_size = (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE

        # ATR filter: gap must be >= threshold * ATR
        atr_value = atr_values[i]
        if gap_size < self.atr_threshold * atr_value:
            return None

        # Volume confirmation: down volume >= threshold * up volume
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

    def _simulate_fvg_trade(self, fvg: FVGEvent, dollar_bars: List[DollarBar]) -> Dict:
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

        # Simulate trade execution
        exit_price = None
        exit_type = None
        bars_held = 0

        for j in range(fvg.bar_index + 1, min(fvg.bar_index + 12, len(dollar_bars))):
            bar = dollar_bars[j]
            bars_held = j - fvg.bar_index

            # Check time exit (max 10 bars)
            if bars_held > 10:
                exit_price = bar.close
                exit_type = "time"
                break

            # Check take profit and stop loss
            if direction == "long":
                if bar.low <= take_profit:
                    exit_price = take_profit
                    exit_type = "tp"
                    break
                if bar.high >= stop_loss:
                    exit_price = stop_loss
                    exit_type = "sl"
                    break
            else:  # short
                if bar.high >= take_profit:
                    exit_price = take_profit
                    exit_type = "tp"
                    break
                if bar.low <= stop_loss:
                    exit_price = stop_loss
                    exit_type = "sl"
                    break

        # If no exit triggered, use last available bar
        if exit_price is None:
            last_bar_idx = min(fvg.bar_index + 11, len(dollar_bars) - 1)
            exit_price = dollar_bars[last_bar_idx].close
            exit_type = "time"
            bars_held = last_bar_idx - fvg.bar_index

        # Calculate P&L
        if direction == "long":
            price_diff = exit_price - entry_price
        else:  # short
            price_diff = entry_price - exit_price

        pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE
        pnl = pnl_before_costs - TRANSACTION_COST

        return {
            'exit_type': exit_type,
            'exit_price': exit_price,
            'bars_held': bars_held,
            'pnl': pnl,
        }

    def _simulate_trade(
        self,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        direction: str,
        future_bars: List[DollarBar]
    ) -> Dict:
        """Simulate trade with triple-barrier exits."""
        if not future_bars:
            return None

        for i, bar in enumerate(future_bars):
            # Check time exit (max 10 bars)
            if i >= 10:
                return {
                    'exit_type': 'time',
                    'exit_price': bar.close,
                    'bars_held': i + 1,
                    'pnl': self._calculate_pnl(entry_price, bar.close, direction)
                }

            # Check take profit
            if direction == "LONG":
                if bar.low <= tp_price:
                    return {
                        'exit_type': 'tp',
                        'exit_price': tp_price,
                        'bars_held': i + 1,
                        'pnl': self._calculate_pnl(entry_price, tp_price, direction)
                    }
                if bar.high >= sl_price:
                    return {
                        'exit_type': 'sl',
                        'exit_price': sl_price,
                        'bars_held': i + 1,
                        'pnl': self._calculate_pnl(entry_price, sl_price, direction)
                    }
            else:  # SHORT
                if bar.high >= tp_price:
                    return {
                        'exit_type': 'tp',
                        'exit_price': tp_price,
                        'bars_held': i + 1,
                        'pnl': self._calculate_pnl(entry_price, tp_price, direction)
                    }
                if bar.low <= sl_price:
                    return {
                        'exit_type': 'sl',
                        'exit_price': sl_price,
                        'bars_held': i + 1,
                        'pnl': self._calculate_pnl(entry_price, sl_price, direction)
                    }

        # If no exit triggered, use last bar
        last_bar = future_bars[-1]
        return {
            'exit_type': 'time',
            'exit_price': last_bar.close,
            'bars_held': len(future_bars),
            'pnl': self._calculate_pnl(entry_price, last_bar.close, direction)
        }

    def _calculate_pnl(self, entry: float, exit: float, direction: str) -> float:
        """Calculate P&L with transaction costs."""
        price_diff = exit - entry if direction == "LONG" else entry - exit
        pnl_before_costs = price_diff * MNQ_CONTRACT_VALUE
        return pnl_before_costs - TRANSACTION_COST


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


def transform_to_dollar_bars(df: pd.DataFrame) -> List[DollarBar]:
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
            bar_num=len(dollar_bars)
        )

        dollar_bars.append(dollar_bar)
        prev_boundary = boundary + 1

    print(f"✓ Created {len(dollar_bars):,} Dollar Bars")

    return dollar_bars


def run_sensitivity_analysis(dollar_bars: List[DollarBar]) -> List[SensitivityResult]:
    """Run parameter sensitivity analysis."""
    print("=" * 80)
    print("TIER 1 PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("Testing variations around optimal: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0")
    print()

    # Parameter variations around optimal
    sl_multipliers = [2.3, 2.4, 2.5, 2.6, 2.7]
    atr_thresholds = [0.65, 0.7, 0.75]
    volume_ratios = [2.1, 2.25, 2.4]

    total_combinations = len(sl_multipliers) * len(atr_thresholds) * len(volume_ratios)
    print(f"Total combinations to test: {total_combinations}")
    print(f"SL multipliers: {sl_multipliers}")
    print(f"ATR thresholds: {atr_thresholds}")
    print(f"Volume ratios: {volume_ratios}")
    print()

    results = []
    combinations_tested = 0

    for sl_multiplier in sl_multipliers:
        for atr_threshold in atr_thresholds:
            for volume_ratio in volume_ratios:
                combinations_tested += 1
                print(f"[{combinations_tested}/{total_combinations}] Testing SL{sl_multiplier}x_ATR{atr_threshold}_Vol{volume_ratio}")

                backtester = Tier1Backtester(
                    sl_multiplier=sl_multiplier,
                    atr_threshold=atr_threshold,
                    volume_ratio_threshold=volume_ratio
                )

                metrics = backtester.run_backtest(dollar_bars)

                result = SensitivityResult(
                    sl_multiplier=sl_multiplier,
                    atr_threshold=atr_threshold,
                    volume_ratio=volume_ratio,
                    total_trades=metrics['total_trades'],
                    wins=metrics['wins'],
                    losses=metrics['losses'],
                    win_rate=metrics['win_rate'],
                    profit_factor=metrics['profit_factor'],
                    total_pnl=metrics['total_pnl'],
                    expectancy=metrics['expectancy'],
                    avg_win=metrics['avg_win'],
                    avg_loss=metrics['avg_loss']
                )

                results.append(result)
                print()

    return results


def analyze_sensitivity(results: List[SensitivityResult]) -> None:
    """Analyze and display sensitivity results."""
    print("=" * 80)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print()

    # Find optimal result
    optimal = max(results, key=lambda r: r.win_rate * r.profit_factor)

    print("OPTIMAL CONFIGURATION:")
    print(f"  SL{optimal.sl_multiplier}x_ATR{optimal.atr_threshold}_Vol{optimal.volume_ratio}")
    print(f"  Win Rate: {optimal.win_rate:.2f}%")
    print(f"  Profit Factor: {optimal.profit_factor:.2f}")
    print(f"  Total Trades: {optimal.total_trades}")
    print(f"  Expectancy: ${optimal.expectancy:.2f}")
    print()

    # Analyze robustness
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print()

    # Count how many combinations meet targets
    wr_pass = sum(1 for r in results if r.win_rate >= 60.0)
    pf_pass = sum(1 for r in results if r.profit_factor >= 1.7)
    both_pass = sum(1 for r in results if r.win_rate >= 60.0 and r.profit_factor >= 1.7)

    print(f"Combinations meeting Win Rate ≥60%: {wr_pass}/{len(results)} ({wr_pass/len(results)*100:.1f}%)")
    print(f"Combinations meeting Profit Factor ≥1.7: {pf_pass}/{len(results)} ({pf_pass/len(results)*100:.1f}%)")
    print(f"Combinations meeting BOTH targets: {both_pass}/{len(results)} ({both_pass/len(results)*100:.1f}%)")
    print()

    # Parameter stability analysis
    print("PARAMETER STABILITY (Performance Range):")
    print()

    # SL multiplier stability
    sl_results = {}
    for sl in [2.3, 2.4, 2.5, 2.6, 2.7]:
        sl_vals = [r.win_rate for r in results if r.sl_multiplier == sl]
        if sl_vals:
            sl_results[sl] = {
                'min': min(sl_vals),
                'max': max(sl_vals),
                'std': np.std(sl_vals)
            }

    print("  SL Multiplier Impact on Win Rate:")
    for sl, stats in sl_results.items():
        print(f"    SL{sl}x: {stats['min']:.1f}% - {stats['max']:.1f}% (std: {stats['std']:.2f})")
    print()

    # ATR threshold stability
    atr_results = {}
    for atr in [0.65, 0.7, 0.75]:
        atr_vals = [r.profit_factor for r in results if r.atr_threshold == atr]
        if atr_vals:
            atr_results[atr] = {
                'min': min(atr_vals),
                'max': max(atr_vals),
                'std': np.std(atr_vals)
            }

    print("  ATR Threshold Impact on Profit Factor:")
    for atr, stats in atr_results.items():
        print(f"    ATR{atr}: {stats['min']:.2f} - {stats['max']:.2f} (std: {stats['std']:.2f})")
    print()

    # Volume ratio stability
    vol_results = {}
    for vol in [2.1, 2.25, 2.4]:
        vol_vals = [r.expectancy for r in results if r.volume_ratio == vol]
        if vol_vals:
            vol_results[vol] = {
                'min': min(vol_vals),
                'max': max(vol_vals),
                'std': np.std(vol_vals)
            }

    print("  Volume Ratio Impact on Expectancy:")
    for vol, stats in vol_results.items():
        print(f"    Vol{vol}: ${stats['min']:.2f} - ${stats['max']:.2f} (std: ${stats['std']:.2f})")
    print()

    # Cliff-edge detection
    print("=" * 80)
    print("CLIFF-EDGE DETECTION")
    print("=" * 80)
    print()

    # Check if small parameter changes cause large performance drops
    sl_sorted = sorted([r for r in results if r.atr_threshold == 0.7 and r.volume_ratio == 2.25],
                      key=lambda x: x.sl_multiplier)

    if len(sl_sorted) > 1:
        print("SL Multiplier Sensitivity (ATR0.7, Vol2.25):")
        for i in range(len(sl_sorted) - 1):
            current = sl_sorted[i]
            next_r = sl_sorted[i + 1]
            wr_change = abs(current.win_rate - next_r.win_rate)
            pf_change = abs(current.profit_factor - next_r.profit_factor)

            status = "✓ Stable" if wr_change < 5 and pf_change < 0.3 else "⚠ VOLATILE"
            print(f"  SL{current.sl_multiplier} → SL{next_r.sl_multiplier}: "
                  f"WR Δ{wr_change:.1f}%, PF Δ{pf_change:.2f} {status}")
        print()

    # Robustness conclusion
    print("=" * 80)
    print("ROBUSTNESS CONCLUSION")
    print("=" * 80)
    print()

    robustness_score = both_pass / len(results) * 100

    if robustness_score >= 80:
        print("✅ HIGHLY ROBUST: >80% of parameter variations meet targets")
        print("   Strategy is NOT overfit to specific parameter values.")
    elif robustness_score >= 50:
        print("⚠️  MODERATELY ROBUST: 50-80% of variations meet targets")
        print("   Strategy is reasonably stable but benefits from precise tuning.")
    else:
        print("❌ FRAGILE: <50% of variations meet targets")
        print("   Strategy may be overfit to specific parameter values.")

    print()
    print(f"Robustness Score: {robustness_score:.1f}%")
    print()

    # Save results
    save_results(results, optimal)


def save_results(results: List[SensitivityResult], optimal: SensitivityResult) -> None:
    """Save sensitivity analysis results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = project_root / f"parameter_sensitivity_analysis_{timestamp}.csv"

    rows = []
    for r in results:
        rows.append({
            'sl_multiplier': r.sl_multiplier,
            'atr_threshold': r.atr_threshold,
            'volume_ratio': r.volume_ratio,
            'total_trades': r.total_trades,
            'wins': r.wins,
            'losses': r.losses,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'total_pnl': r.total_pnl,
            'expectancy': r.expectancy,
            'avg_win': r.avg_win,
            'avg_loss': r.avg_loss,
            'meets_wr_target': r.win_rate >= 60.0,
            'meets_pf_target': r.profit_factor >= 1.7,
            'meets_both': r.win_rate >= 60.0 and r.profit_factor >= 1.7,
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("TIER 1 FVG PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Optimal Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0")
    print(f"Data Range: {START_IDX:,} to {END_IDX:,} (30K bars)")
    print()

    # Load data
    df = load_mnq_data_subset(START_IDX, END_IDX)

    # Transform to Dollar Bars
    dollar_bars = transform_to_dollar_bars(df)

    if len(dollar_bars) < 100:
        print(f"ERROR: Insufficient Dollar Bars: {len(dollar_bars)}")
        sys.exit(1)

    print()

    # Run sensitivity analysis
    results = run_sensitivity_analysis(dollar_bars)

    # Analyze results
    analyze_sensitivity(results)

    print()
    print("=" * 80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

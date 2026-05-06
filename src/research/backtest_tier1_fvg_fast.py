#!/usr/bin/env python3
"""Fast backtest validation of TIER 1 FVG System using sample of MNQ data.

This is a streamlined version that processes a subset of data for quick validation.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
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
from src.detection.multi_timeframe import MultiTimeframeNester
from src.detection.fvg_detection import (
    detect_bullish_fvg,
    detect_bearish_fvg,
    check_fvg_fill,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MNQ_DATA_PATH = Path("/root/mnq_historical.json")
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
DOLLAR_BAR_THRESHOLD = 50_000_000
SAMPLE_SIZE = 5000  # Process only first 5K bars for ultra-fast validation


def load_sample_mnq_data(data_path: Path = MNQ_DATA_PATH, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load a sample of MNQ historical data for quick testing.

    Args:
        data_path: Path to MNQ historical data JSON file
        sample_size: Number of bars to load

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading sample of {sample_size} bars from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"MNQ data file not found: {data_path}")

    # Load JSON data in chunks
    logger.info("Reading JSON file...")
    data = []
    with open(data_path, 'r') as f:
        # Read line by line (it's a JSON array)
        content = f.read()
        # Parse JSON
        parsed_data = json.loads(content)
        # Take sample
        data = parsed_data[:sample_size]

    logger.info(f"Loaded {len(data)} data points")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert timestamp strings to datetime
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    # Convert numeric columns from string to float
    numeric_columns = ['High', 'Low', 'Open', 'Close', 'TotalVolume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by timestamp
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    logger.info(f"Data range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

    return df


def transform_to_dollar_bars_fast(df: pd.DataFrame) -> list[DollarBar]:
    """Fast transformation to Dollar Bars using vectorized operations.

    Args:
        df: DataFrame with raw MNQ OHLCV data

    Returns:
        List of DollarBar objects
    """
    logger.info("Fast transforming to Dollar Bars...")

    # Calculate notional value for each bar
    df['notional'] = ((df['High'] + df['Low']) / 2) * df['TotalVolume'] * MNQ_POINT_VALUE

    # Calculate cumulative notional
    df['cumulative_notional'] = df['notional'].cumsum()

    # Find bar boundaries (every $50M)
    bar_boundaries = df[df['cumulative_notional'] % DOLLAR_BAR_THRESHOLD < df['notional']].index.tolist()

    # Add the last bar boundary
    if len(df) > 0 and bar_boundaries[-1] != len(df) - 1:
        bar_boundaries.append(len(df) - 1)

    logger.info(f"Creating {len(bar_boundaries)} Dollar Bars...")

    # Create Dollar Bars by aggregating between boundaries
    dollar_bars = []
    prev_boundary = 0

    for boundary in tqdm(bar_boundaries, desc="Aggregating Dollar Bars"):
        if boundary == 0:
            continue

        # Aggregate bars between prev_boundary and boundary
        segment = df.iloc[prev_boundary:boundary+1]

        if len(segment) == 0:
            continue

        dollar_bar = DollarBar(
            timestamp=segment.iloc[0]['TimeStamp'],
            open=float(segment.iloc[0]['Open']),
            high=float(segment['High'].max()),
            low=float(segment['Low'].min()),
            close=float(segment.iloc[-1]['Close']),
            volume=int(segment['TotalVolume'].sum()),
            notional_value=float(segment['notional'].sum()),
            is_forward_filled=False,
        )
        dollar_bars.append(dollar_bar)

        prev_boundary = boundary + 1

    logger.info(f"Created {len(dollar_bars)} Dollar Bars")
    return dollar_bars


class SimpleBacktest:
    """Simplified backtest for quick validation."""

    def __init__(self, name: str, use_filters: bool = False):
        """Initialize backtest.

        Args:
            name: Strategy name
            use_filters: Whether to use TIER 1 filters
        """
        self.name = name
        self.use_filters = use_filters
        self.trades = []

        if use_filters:
            self.atr_filter = ATRFilter(lookback_period=14, atr_threshold=0.5)
            self.volume_confirmer = VolumeConfirmer(lookback_period=20, volume_ratio_threshold=1.5)

    def run(self, dollar_bars: list[DollarBar]) -> dict:
        """Run backtest.

        Args:
            dollar_bars: List of Dollar Bars

        Returns:
            Performance metrics
        """
        logger.info(f"Running {self.name}...")

        for i in tqdm(range(2, len(dollar_bars)), desc=self.name):
            # Detect FVGs
            if self.use_filters:
                bullish_fvg = detect_bullish_fvg(dollar_bars, i, self.atr_filter, self.volume_confirmer)
                bearish_fvg = detect_bearish_fvg(dollar_bars, i, self.atr_filter, self.volume_confirmer)
            else:
                bullish_fvg = detect_bullish_fvg(dollar_bars, i, None, None)
                bearish_fvg = detect_bearish_fvg(dollar_bars, i, None, None)

            # Process detected FVGs
            for fvg in [bullish_fvg, bearish_fvg]:
                if fvg is None:
                    continue

                # Check if gap fills within 5 bars
                filled = False
                fill_bar = i
                for j in range(i+1, min(i+6, len(dollar_bars))):
                    if check_fvg_fill(fvg, dollar_bars[j]):
                        filled = True
                        fill_bar = j
                        break

                self.trades.append({
                    "direction": fvg.direction,
                    "filled": filled,
                    "bars_held": fill_bar - i,
                })

        return self._calculate_metrics(dollar_bars)

    def _calculate_metrics(self, dollar_bars: list[DollarBar]) -> dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

        wins = [t for t in self.trades if t["filled"]]
        losses = [t for t in self.trades if not t["filled"]]

        win_rate = len(wins) / len(self.trades) * 100
        profit_factor = len(wins) / len(losses) if losses else float('inf')

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
        }


def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("TIER 1 FVG ULTRA-FAST VALIDATION (5K Sample)")
    logger.info("=" * 70)

    # Load sample data
    df = load_sample_mnq_data(sample_size=SAMPLE_SIZE)

    # Transform to Dollar Bars
    dollar_bars = transform_to_dollar_bars_fast(df)

    if len(dollar_bars) < 100:
        logger.error(f"Insufficient Dollar Bars: {len(dollar_bars)}")
        sys.exit(1)

    # Run baseline
    baseline = SimpleBacktest("Baseline", use_filters=False)
    baseline_metrics = baseline.run(dollar_bars)

    # Run TIER 1
    tier1 = SimpleBacktest("TIER 1", use_filters=True)
    tier1_metrics = tier1.run(dollar_bars)

    # Generate report
    report = f"""
{'=' * 70}
TIER 1 FVG SYSTEM - FAST VALIDATION (Sample Data)
{'=' * 70}

Data Points: {len(dollar_bars)} Dollar Bars
Date Range: {dollar_bars[0].timestamp.strftime('%Y-%m-%d')} to {dollar_bars[-1].timestamp.strftime('%Y-%m-%d')}

{'=' * 70}
BASELINE (No Filters)
{'=' * 70}
Total Trades: {baseline_metrics['total_trades']}
Win Rate: {baseline_metrics['win_rate']:.2f}%
Profit Factor: {baseline_metrics['profit_factor']:.2f}
Avg Trades/Day: {baseline_metrics['avg_trades_per_day']:.2f}

{'=' * 70}
TIER 1 (ATR + Volume Filters)
{'=' * 70}
Total Trades: {tier1_metrics['total_trades']}
Win Rate: {tier1_metrics['win_rate']:.2f}%
Profit Factor: {tier1_metrics['profit_factor']:.2f}
Avg Trades/Day: {tier1_metrics['avg_trades_per_day']:.2f}

{'=' * 70}
IMPROVEMENTS
{'=' * 70}
Win Rate: {tier1_metrics['win_rate'] - baseline_metrics['win_rate']:+.2f}%
Profit Factor: {tier1_metrics['profit_factor'] - baseline_metrics['profit_factor']:+.2f}
Trade Freq Change: {((tier1_metrics['avg_trades_per_day'] / baseline_metrics['avg_trades_per_day'] - 1) * 100):+.2f}%

{'=' * 70}
TARGETS (Win Rate >= 60%, PF >= 1.7, 8-15 trades/day)
{'=' * 70}
{'✅ PASS' if tier1_metrics['win_rate'] >= 60.0 else '❌ FAIL'} - Win Rate: {tier1_metrics['win_rate']:.2f}% >= 60.0%
{'✅ PASS' if tier1_metrics['profit_factor'] >= 1.7 else '❌ FAIL'} - Profit Factor: {tier1_metrics['profit_factor']:.2f} >= 1.7
{'✅ PASS' if 8.0 <= tier1_metrics['avg_trades_per_day'] <= 15.0 else '❌ FAIL'} - Trade Freq: {tier1_metrics['avg_trades_per_day']:.2f} in [8.0, 15.0]

{'=' * 70}
"""
    print(report)

    # Save report
    report_path = Path("/root/Silver-Bullet-ML-BMAD/backtest_tier1_fast_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

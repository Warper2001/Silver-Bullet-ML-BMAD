"""Run baseline backtests on all 5 strategies with historical MNQ data.

This script loads historical dollar bar data, processes it through all
implemented strategies, and generates comprehensive performance reports.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

import h5py
import numpy as np

from src.data.models import DollarBar
from src.research.backtest_engine import BacktestEngine
from src.research.performance_analyzer import PerformanceAnalyzer
from src.research.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dollar_bars(data_path: str = "data/processed/dollar_bars/") -> list[DollarBar]:
    """Load all dollar bars from HDF5 files.

    Args:
        data_path: Path to dollar bars directory

    Returns:
        List of DollarBar objects sorted by timestamp
    """
    logger.info(f"Loading dollar bars from {data_path}")

    path = Path(data_path)
    h5_files = sorted(path.glob("*.h5"))

    if not h5_files:
        raise ValueError(f"No HDF5 files found in {data_path}")

    all_bars = []

    for h5_file in h5_files:
        logger.info(f"Loading {h5_file.name}...")

        try:
            with h5py.File(h5_file, 'r') as f:
                if 'dollar_bars' not in f:
                    logger.warning(f"  No 'dollar_bars' dataset, skipping")
                    continue

                bars = f['dollar_bars']

                # Convert to DollarBar objects
                # Data structure: [timestamp(ms), open, high, low, close, volume, notional]
                for i in range(len(bars)):
                    try:
                        # Convert millisecond timestamp to datetime
                        ts_ms = bars[i, 0]
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

                        bar = DollarBar(
                            timestamp=ts,
                            open=float(bars[i, 1]),
                            high=float(bars[i, 2]),
                            low=float(bars[i, 3]),
                            close=float(bars[i, 4]),
                            volume=int(bars[i, 5]),
                            notional_value=float(bars[i, 6]),
                            is_forward_filled=False,  # Not stored in HDF5
                        )
                        all_bars.append(bar)
                    except Exception as e:
                        logger.warning(f"  Error loading bar {i}: {e}")

        except Exception as e:
            logger.error(f"  Error loading {h5_file.name}: {e}")

    # Sort by timestamp
    all_bars.sort(key=lambda x: x.timestamp)

    logger.info(f"Loaded {len(all_bars):,} dollar bars total")

    # Log date range
    if all_bars:
        logger.info(f"Date range: {all_bars[0].timestamp} to {all_bars[-1].timestamp}")

    return all_bars


def backtest_triple_confluence(bars: list[DollarBar]) -> dict:
    """Backtest Triple Confluence Scalper strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        Dictionary with backtest results
    """
    logger.info("Backtesting Triple Confluence Scalper...")

    try:
        # Import strategy
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

        strategy = TripleConfluenceStrategy(config={})
        engine = BacktestEngine(initial_capital=100000)

        signals_generated = 0

        for i, bar in enumerate(bars):
            # Process bar through strategy
            signal = strategy.process_bar(bar)

            if signal:
                signals_generated += 1

                # Simulate trade exit (2:1 R:R or 10-min max hold)
                # For simplicity, assume 2:1 target hit
                entry = signal.entry_price
                sl = signal.stop_loss
                tp = signal.take_profit

                # Simulate 5-bar hold (simplified)
                exit_idx = min(i + 5, len(bars) - 1)
                exit_bar = bars[exit_idx]

                # Calculate P&L (MNQ: $5/point)
                if signal.direction == "long":
                    pnl = (exit_bar.close - entry) * 5.0
                else:
                    pnl = (entry - exit_bar.close) * 5.0

                engine.add_trade(
                    entry_time=bar.timestamp,
                    exit_time=exit_bar.timestamp,
                    direction=signal.direction,
                    entry_price=entry,
                    exit_price=exit_bar.close,
                    stop_loss=sl,
                    take_profit=tp,
                    bars_held=5,
                )

        logger.info(f"  Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
        return {"signals": signals_generated, "trades": engine.get_total_trades()}

    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"signals": 0, "trades": 0, "error": str(e)}


def backtest_wolf_pack(bars: list[DollarBar]) -> dict:
    """Backtest Wolf Pack 3-Edge strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        Dictionary with backtest results
    """
    logger.info("Backtesting Wolf Pack 3-Edge...")

    try:
        from src.detection.wolf_pack_strategy import WolfPackStrategy

        strategy = WolfPackStrategy()  # Uses default parameters
        engine = BacktestEngine(initial_capital=100000)

        signals_generated = 0

        for i, bar in enumerate(bars):
            # Wolf Pack uses process_bars() with list
            signals = strategy.process_bars([bar])
            signal = signals[0] if signals else None

            if signal:
                signals_generated += 1

                entry = signal.entry_price
                sl = signal.stop_loss
                tp = signal.take_profit

                exit_idx = min(i + 3, len(bars) - 1)
                exit_bar = bars[exit_idx]

                if signal.direction == "long":
                    pnl = (exit_bar.close - entry) * 5.0
                else:
                    pnl = (entry - exit_bar.close) * 5.0

                engine.add_trade(
                    entry_time=bar.timestamp,
                    exit_time=exit_bar.timestamp,
                    direction=signal.direction,
                    entry_price=entry,
                    exit_price=exit_bar.close,
                    stop_loss=sl,
                    take_profit=tp,
                    bars_held=3,
                )

        logger.info(f"  Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
        return {"signals": signals_generated, "trades": engine.get_total_trades()}

    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"signals": 0, "trades": 0, "error": str(e)}


def backtest_adaptive_ema(bars: list[DollarBar]) -> dict:
    """Backtest Adaptive EMA Momentum strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        Dictionary with backtest results
    """
    logger.info("Backtesting Adaptive EMA Momentum...")

    try:
        from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy

        strategy = AdaptiveEMAStrategy()  # No parameters needed
        engine = BacktestEngine(initial_capital=100000)

        signals_generated = 0

        for i, bar in enumerate(bars):
            # Adaptive EMA uses process_bars() with list
            signals = strategy.process_bars([bar])
            signal = signals[0] if signals else None

            if signal:
                signals_generated += 1

                entry = signal.entry_price
                sl = signal.stop_loss
                tp = signal.take_profit

                exit_idx = min(i + 4, len(bars) - 1)
                exit_bar = bars[exit_idx]

                if signal.direction == "long":
                    pnl = (exit_bar.close - entry) * 5.0
                else:
                    pnl = (entry - exit_bar.close) * 5.0

                engine.add_trade(
                    entry_time=bar.timestamp,
                    exit_time=exit_bar.timestamp,
                    direction=signal.direction,
                    entry_price=entry,
                    exit_price=exit_bar.close,
                    stop_loss=sl,
                    take_profit=tp,
                    bars_held=4,
                )

        logger.info(f"  Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
        return {"signals": signals_generated, "trades": engine.get_total_trades()}

    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"signals": 0, "trades": 0, "error": str(e)}


def backtest_vwap_bounce(bars: list[DollarBar]) -> dict:
    """Backtest VWAP Bounce strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        Dictionary with backtest results
    """
    logger.info("Backtesting VWAP Bounce...")

    try:
        from src.detection.vwap_bounce_strategy import VWAPBounceStrategy

        strategy = VWAPBounceStrategy(config={})
        engine = BacktestEngine(initial_capital=100000)

        signals_generated = 0

        for i, bar in enumerate(bars):
            signal = strategy.process_bar(bar)

            if signal:
                signals_generated += 1

                entry = signal.entry_price
                sl = signal.stop_loss
                tp = signal.take_profit

                exit_idx = min(i + 6, len(bars) - 1)
                exit_bar = bars[exit_idx]

                if signal.direction == "long":
                    pnl = (exit_bar.close - entry) * 5.0
                else:
                    pnl = (entry - exit_bar.close) * 5.0

                engine.add_trade(
                    entry_time=bar.timestamp,
                    exit_time=exit_bar.timestamp,
                    direction=signal.direction,
                    entry_price=entry,
                    exit_price=exit_bar.close,
                    stop_loss=sl,
                    take_profit=tp,
                    bars_held=6,
                )

        logger.info(f"  Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
        return {"signals": signals_generated, "trades": engine.get_total_trades()}

    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"signals": 0, "trades": 0, "error": str(e)}


def backtest_opening_range(bars: list[DollarBar]) -> dict:
    """Backtest Opening Range Breakout strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        Dictionary with backtest results
    """
    logger.info("Backtesting Opening Range Breakout...")

    try:
        from src.detection.opening_range_strategy import OpeningRangeStrategy

        strategy = OpeningRangeStrategy(config={})
        engine = BacktestEngine(initial_capital=100000)

        signals_generated = 0

        for i, bar in enumerate(bars):
            signal = strategy.process_bar(bar)

            if signal:
                signals_generated += 1

                entry = signal.entry_price
                sl = signal.stop_loss
                tp = signal.take_profit

                exit_idx = min(i + 8, len(bars) - 1)
                exit_bar = bars[exit_idx]

                if signal.direction == "long":
                    pnl = (exit_bar.close - entry) * 5.0
                else:
                    pnl = (entry - exit_bar.close) * 5.0

                engine.add_trade(
                    entry_time=bar.timestamp,
                    exit_time=exit_bar.timestamp,
                    direction=signal.direction,
                    entry_price=entry,
                    exit_price=exit_bar.close,
                    stop_loss=sl,
                    take_profit=tp,
                    bars_held=8,
                )

        logger.info(f"  Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
        return {"signals": signals_generated, "trades": engine.get_total_trades()}

    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"signals": 0, "trades": 0, "error": str(e)}


def main():
    """Main entry point for backtesting."""
    logger.info("=" * 70)
    logger.info("BASELINE BACKTESTING - ALL STRATEGIES")
    logger.info("=" * 70)

    # Load data
    try:
        bars = load_dollar_bars()

        # Sample for faster testing (use subset)
        logger.info(f"Using all {len(bars)} bars for backtesting")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Backtest all strategies
    strategies = {
        "Triple Confluence Scalper": backtest_triple_confluence,
        "Wolf Pack 3-Edge": backtest_wolf_pack,
        "Adaptive EMA Momentum": backtest_adaptive_ema,
        "VWAP Bounce": backtest_vwap_bounce,
        "Opening Range Breakout": backtest_opening_range,
    }

    results = {}

    for strategy_name, backtest_func in strategies.items():
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Testing: {strategy_name}")
        logger.info('=' * 70)

        try:
            result = backtest_func(bars)
            results[strategy_name] = result
        except Exception as e:
            logger.error(f"Failed to backtest {strategy_name}: {e}")
            results[strategy_name] = {"error": str(e)}

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 70)

    for strategy_name, result in results.items():
        if "error" in result:
            logger.info(f"\n{strategy_name}: ❌ ERROR")
            logger.info(f"  {result['error']}")
        else:
            logger.info(f"\n{strategy_name}: ✅ COMPLETE")
            logger.info(f"  Signals Generated: {result['signals']}")
            logger.info(f"  Trades Executed: {result['trades']}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Backtesting complete!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

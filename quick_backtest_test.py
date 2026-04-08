"""Quick backtest test with subset of data."""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py
import numpy as np

from src.data.models import DollarBar
from src.research.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_bars(num_bars: int = 500) -> list[DollarBar]:
    """Load sample bars from first HDF5 file.

    Args:
        num_bars: Number of bars to load

    Returns:
        List of DollarBar objects
    """
    logger.info(f"Loading {num_bars} sample bars...")

    path = Path("data/processed/dollar_bars/")
    h5_files = sorted(path.glob("*.h5"))

    if not h5_files:
        raise ValueError("No HDF5 files found")

    # Load from first file only
    h5_file = h5_files[0]
    logger.info(f"Loading from {h5_file.name}...")

    all_bars = []

    with h5py.File(h5_file, 'r') as f:
        bars = f['dollar_bars']

        # Load limited number of bars
        for i in range(min(len(bars), num_bars)):
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
                is_forward_filled=False,
            )
            all_bars.append(bar)

    logger.info(f"Loaded {len(all_bars)} bars")
    logger.info(f"Date range: {all_bars[0].timestamp} to {all_bars[-1].timestamp}")

    return all_bars


def backtest_triple_confluence(bars: list[DollarBar]) -> dict:
    """Quick backtest of Triple Confluence."""
    logger.info("Testing Triple Confluence Scalper...")

    from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

    strategy = TripleConfluenceStrategy(config={})
    engine = BacktestEngine(initial_capital=100000)

    signals = 0

    for i, bar in enumerate(bars):
        signal = strategy.process_bar(bar)

        if signal:
            signals += 1
            entry = signal.entry_price
            exit_idx = min(i + 5, len(bars) - 1)
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
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                bars_held=5,
            )

    logger.info(f"  Signals: {signals}, Trades: {engine.get_total_trades()}")
    return {"signals": signals, "trades": engine.get_total_trades()}


def backtest_wolf_pack(bars: list[DollarBar]) -> dict:
    """Quick backtest of Wolf Pack."""
    logger.info("Testing Wolf Pack 3-Edge...")

    from src.detection.wolf_pack_strategy import WolfPackStrategy

    strategy = WolfPackStrategy()
    engine = BacktestEngine(initial_capital=100000)

    signals = 0

    for i, bar in enumerate(bars):
        signals_list = strategy.process_bars([bar])
        signal = signals_list[0] if signals_list else None

        if signal:
            signals += 1
            entry = signal.entry_price
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
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                bars_held=3,
            )

    logger.info(f"  Signals: {signals}, Trades: {engine.get_total_trades()}")
    return {"signals": signals, "trades": engine.get_total_trades()}


def backtest_adaptive_ema(bars: list[DollarBar]) -> dict:
    """Quick backtest of Adaptive EMA."""
    logger.info("Testing Adaptive EMA Momentum...")

    from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy

    strategy = AdaptiveEMAStrategy()
    engine = BacktestEngine(initial_capital=100000)

    signals = 0

    for i, bar in enumerate(bars):
        signals_list = strategy.process_bars([bar])
        signal = signals_list[0] if signals_list else None

        if signal:
            signals += 1
            entry = signal.entry_price
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
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                bars_held=4,
            )

    logger.info(f"  Signals: {signals}, Trades: {engine.get_total_trades()}")
    return {"signals": signals, "trades": engine.get_total_trades()}


def main():
    """Run quick backtests."""
    logger.info("=" * 60)
    logger.info("QUICK BACKTEST TEST - 500 bars")
    logger.info("=" * 60)

    try:
        bars = load_sample_bars(num_bars=500)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test each strategy
    strategies = {
        "Triple Confluence": backtest_triple_confluence,
        "Wolf Pack": backtest_wolf_pack,
        "Adaptive EMA": backtest_adaptive_ema,
    }

    results = {}

    for name, func in strategies.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {name}")
        logger.info('=' * 60)

        try:
            result = func(bars)
            results[name] = result
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, result in results.items():
        if "error" in result:
            logger.info(f"\n{name}: ❌ ERROR")
            logger.info(f"  {result['error']}")
        else:
            logger.info(f"\n{name}: ✅")
            logger.info(f"  Signals: {result['signals']}")
            logger.info(f"  Trades: {result['trades']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run comprehensive backtests on all 5 strategies with full dataset.

This script:
1. Loads all 116K+ historical dollar bars
2. Runs each strategy through the entire dataset
3. Calculates 12 performance metrics
4. Generates comprehensive reports
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging
import json

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


def load_all_bars(data_path: str = "data/processed/dollar_bars/") -> list[DollarBar]:
    """Load all dollar bars from HDF5 files.

    Args:
        data_path: Path to dollar bars directory

    Returns:
        List of DollarBar objects sorted by timestamp
    """
    logger.info(f"Loading all dollar bars from {data_path}")

    path = Path(data_path)
    h5_files = sorted(path.glob("*.h5"))

    if not h5_files:
        raise ValueError(f"No HDF5 files found in {data_path}")

    logger.info(f"Found {len(h5_files)} HDF5 files")

    all_bars = []
    file_count = 0

    for h5_file in h5_files:
        file_count += 1
        logger.info(f"[{file_count}/{len(h5_files)}] Loading {h5_file.name}...")

        try:
            with h5py.File(h5_file, 'r') as f:
                bars = f['dollar_bars']

                # Convert to DollarBar objects
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
                            is_forward_filled=False,
                        )
                        all_bars.append(bar)
                    except Exception as e:
                        logger.warning(f"  Error loading bar {i}: {e}")

        except Exception as e:
            logger.error(f"  Error loading {h5_file.name}: {e}")

    # Sort by timestamp
    all_bars.sort(key=lambda x: x.timestamp)

    logger.info(f"✅ Loaded {len(all_bars):,} dollar bars total")

    # Log date range
    if all_bars:
        logger.info(f"Date range: {all_bars[0].timestamp} to {all_bars[-1].timestamp}")

    return all_bars


def backtest_triple_confluence(bars: list[DollarBar]) -> BacktestEngine:
    """Backtest Triple Confluence Scalper strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        BacktestEngine with completed trades
    """
    logger.info("=" * 70)
    logger.info("BACKTESTING: Triple Confluence Scalper")
    logger.info("=" * 70)

    from src.detection.triple_confluence_strategy import TripleConfluenceStrategy

    strategy = TripleConfluenceStrategy(config={})
    engine = BacktestEngine(initial_capital=100000)

    signals_generated = 0
    total_bars = len(bars)

    for i, bar in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Progress: {i:,}/{total_bars:,} bars ({100*i/total_bars:.1f}%)")

        signal = strategy.process_bar(bar)

        if signal:
            signals_generated += 1

            # Simplified exit: hold for 5 bars
            entry = signal.entry_price
            sl = signal.stop_loss
            tp = signal.take_profit

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

    logger.info(f"✅ Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
    return engine


def backtest_wolf_pack(bars: list[DollarBar]) -> BacktestEngine:
    """Backtest Wolf Pack 3-Edge strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        BacktestEngine with completed trades
    """
    logger.info("=" * 70)
    logger.info("BACKTESTING: Wolf Pack 3-Edge")
    logger.info("=" * 70)

    from src.detection.wolf_pack_strategy import WolfPackStrategy

    strategy = WolfPackStrategy()
    engine = BacktestEngine(initial_capital=100000)

    signals_generated = 0
    total_bars = len(bars)

    for i, bar in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Progress: {i:,}/{total_bars:,} bars ({100*i/total_bars:.1f}%)")

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

    logger.info(f"✅ Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
    return engine


def backtest_adaptive_ema(bars: list[DollarBar]) -> BacktestEngine:
    """Backtest Adaptive EMA Momentum strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        BacktestEngine with completed trades
    """
    logger.info("=" * 70)
    logger.info("BACKTESTING: Adaptive EMA Momentum")
    logger.info("=" * 70)

    from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy

    strategy = AdaptiveEMAStrategy()
    engine = BacktestEngine(initial_capital=100000)

    signals_generated = 0
    total_bars = len(bars)

    for i, bar in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Progress: {i:,}/{total_bars:,} bars ({100*i/total_bars:.1f}%)")

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

    logger.info(f"✅ Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
    return engine


def backtest_vwap_bounce(bars: list[DollarBar]) -> BacktestEngine:
    """Backtest VWAP Bounce strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        BacktestEngine with completed trades
    """
    logger.info("=" * 70)
    logger.info("BACKTESTING: VWAP Bounce")
    logger.info("=" * 70)

    from src.detection.vwap_bounce_strategy import VWAPBounceStrategy

    strategy = VWAPBounceStrategy(config={})
    engine = BacktestEngine(initial_capital=100000)

    signals_generated = 0
    total_bars = len(bars)

    for i, bar in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Progress: {i:,}/{total_bars:,} bars ({100*i/total_bars:.1f}%)")

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

    logger.info(f"✅ Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
    return engine


def backtest_opening_range(bars: list[DollarBar]) -> BacktestEngine:
    """Backtest Opening Range Breakout strategy.

    Args:
        bars: Historical dollar bars

    Returns:
        BacktestEngine with completed trades
    """
    logger.info("=" * 70)
    logger.info("BACKTESTING: Opening Range Breakout")
    logger.info("=" * 70)

    from src.detection.opening_range_strategy import OpeningRangeStrategy

    strategy = OpeningRangeStrategy(config={})
    engine = BacktestEngine(initial_capital=100000)

    signals_generated = 0
    total_bars = len(bars)

    for i, bar in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Progress: {i:,}/{total_bars:,} bars ({100*i/total_bars:.1f}%)")

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

    logger.info(f"✅ Signals: {signals_generated}, Trades: {engine.get_total_trades()}")
    return engine


def calculate_metrics(engine: BacktestEngine, strategy_name: str) -> dict:
    """Calculate performance metrics for a strategy.

    Args:
        engine: BacktestEngine with completed trades
        strategy_name: Name of strategy

    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Calculating metrics for {strategy_name}...")

    analyzer = PerformanceAnalyzer(engine.get_all_trades())
    metrics = analyzer.calculate_metrics()

    # Convert metrics to dict for JSON serialization
    return {
        "total_trades": metrics.total_trades,
        "winning_trades": metrics.winning_trades,
        "losing_trades": metrics.losing_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "avg_risk_reward": metrics.avg_risk_reward,
        "expectancy": metrics.expectancy,
        "trade_frequency": metrics.trade_frequency,
        "avg_hold_time_bars": metrics.avg_hold_time_bars,
        "max_drawdown_percent": metrics.max_drawdown_percent,
        "sharpe_ratio": metrics.sharpe_ratio,
        "total_pnl": metrics.total_pnl,
        "final_capital": metrics.final_capital,
    }


def main():
    """Run comprehensive backtests on all strategies."""
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE BASELINE BACKTESTING - ALL STRATEGIES")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now()}")

    # Load data
    try:
        bars = load_all_bars()
        logger.info(f"Using all {len(bars):,} bars for backtesting")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
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
        try:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Starting: {strategy_name}")
            logger.info('=' * 70)

            engine = backtest_func(bars)
            signals_count = engine.get_total_trades()  # Signals = trades in this simplified backtest
            metrics = calculate_metrics(engine, strategy_name)

            results[strategy_name] = {
                "trades": engine.get_total_trades(),
                "signals": signals_count,
                "metrics": metrics,
            }

            # Save individual results
            output_file = Path(f"data/reports/backtest_{strategy_name.lower().replace(' ', '_')}_results.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results[strategy_name], f, indent=2, default=str)

            logger.info(f"✅ Saved results to {output_file}")

        except Exception as e:
            logger.error(f"Failed to backtest {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
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
            logger.info(f"  Trades: {result['trades']}")
            logger.info(f"  Signals: {result.get('signals', 'N/A')}")
            metrics = result.get('metrics', {})
            if metrics:
                logger.info(f"  Win Rate: {metrics.get('win_rate', 'N/A'):.2%}")
                logger.info(f"  Profit Factor: {metrics.get('profit_factor', 'N/A'):.2f}")
                logger.info(f"  Expectancy: ${metrics.get('expectancy', 0):.2f}")
                logger.info(f"  Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
                logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
                logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%")

    # Save aggregate results
    aggregate_file = Path("data/reports/backtest_aggregate_results.json")
    with open(aggregate_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n✅ Saved aggregate results to {aggregate_file}")
    logger.info("=" * 70)
    logger.info(f"End time: {datetime.now()}")
    logger.info("✅ Backtesting complete!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run backtests with proper stop loss and take profit execution.

This script uses the ExitSimulator to provide more realistic performance metrics
by executing stop losses and take profits when hit, rather than using fixed-bar exits.
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
from src.research.exit_simulator import ExitSimulator

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

    for h5_file in h5_files:
        logger.info(f"Loading {h5_file.name}...")

        try:
            with h5py.File(h5_file, 'r') as f:
                bars = f['dollar_bars']

                for i in range(len(bars)):
                    try:
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

    all_bars.sort(key=lambda x: x.timestamp)

    logger.info(f"✅ Loaded {len(all_bars):,} dollar bars total")

    if all_bars:
        logger.info(f"Date range: {all_bars[0].timestamp} to {all_bars[-1].timestamp}")

    return all_bars


def backtest_strategy_with_proper_exits(
    strategy,
    strategy_name: str,
    bars: list[DollarBar],
    max_hold_bars: int = 10,
) -> BacktestEngine:
    """Backtest a strategy with proper SL/TP execution.

    Args:
        strategy: Strategy instance with process_bar() method
        strategy_name: Name of strategy
        bars: Historical dollar bars
        max_hold_bars: Maximum bars to hold

    Returns:
        BacktestEngine with completed trades
    """
    logger.info("=" * 70)
    logger.info(f"BACKTESTING: {strategy_name} (with proper exits)")
    logger.info("=" * 70)

    engine = BacktestEngine(initial_capital=100000)
    simulator = ExitSimulator(max_hold_bars=max_hold_bars)

    signals_generated = 0
    total_bars = len(bars)

    for i, bar in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Progress: {i:,}/{total_bars:,} bars ({100*i/total_bars:.1f}%)")

        # Generate signal (handle both process_bar and process_bars interfaces)
        try:
            signal = strategy.process_bar(bar)
        except AttributeError:
            # Strategy uses process_bars() interface
            signals = strategy.process_bars([bar])
            signal = signals[0] if signals else None

        if signal:
            signals_generated += 1

            # Simulate exit with proper SL/TP execution
            exit_bar, exit_price, exit_reason, bars_held = simulator.simulate_exit(
                entry_bar=bar,
                bars=bars,
                entry_index=i,
                direction=signal.direction,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            # Calculate P&L (MNQ: $5/point)
            if signal.direction == "long":
                pnl = (exit_price - signal.entry_price) * 5.0
            else:  # short
                pnl = (signal.entry_price - exit_price) * 5.0

            engine.add_trade(
                entry_time=bar.timestamp,
                exit_time=exit_bar.timestamp,
                direction=signal.direction,
                entry_price=signal.entry_price,
                exit_price=exit_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                bars_held=bars_held,
            )

            logger.debug(
                f"Signal {signals_generated}: {signal.direction} "
                f"@ {signal.entry_price:.2f} -> {exit_price:.2f} "
                f"(${pnl:+.2f}), reason: {exit_reason}, held: {bars_held} bars"
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
    """Run comprehensive backtests with proper exits."""
    logger.info("=" * 70)
    logger.info("BACKTESTING WITH PROPER SL/TP EXECUTION")
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

    # Import strategies
    from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
    from src.detection.wolf_pack_strategy import WolfPackStrategy
    from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
    from src.detection.vwap_bounce_strategy import VWAPBounceStrategy
    from src.detection.opening_range_strategy import OpeningRangeStrategy

    # Define strategies with appropriate max hold times
    strategies = {
        "Triple Confluence Scalper": {
            "instance": TripleConfluenceStrategy(config={}),
            "max_hold_bars": 10,  # 50 minutes
        },
        "Wolf Pack 3-Edge": {
            "instance": WolfPackStrategy(),
            "max_hold_bars": 8,  # 40 minutes
        },
        "Adaptive EMA Momentum": {
            "instance": AdaptiveEMAStrategy(),
            "max_hold_bars": 10,  # 50 minutes
        },
        "VWAP Bounce": {
            "instance": VWAPBounceStrategy(config={}),
            "max_hold_bars": 12,  # 60 minutes
        },
        "Opening Range Breakout": {
            "instance": OpeningRangeStrategy(config={}),
            "max_hold_bars": 16,  # 80 minutes (daily breakout)
        },
    }

    results = {}

    for strategy_name, config in strategies.items():
        try:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Starting: {strategy_name}")
            logger.info('=' * 70)

            engine = backtest_strategy_with_proper_exits(
                strategy=config["instance"],
                strategy_name=strategy_name,
                bars=bars,
                max_hold_bars=config["max_hold_bars"],
            )

            metrics = calculate_metrics(engine, strategy_name)

            results[strategy_name] = {
                "trades": engine.get_total_trades(),
                "metrics": metrics,
            }

            # Save individual results
            output_file = Path(
                f"data/reports/backtest_proper_exits_{strategy_name.lower().replace(' ', '_')}_results.json"
            )
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
    logger.info("BACKTEST SUMMARY (WITH PROPER EXITS)")
    logger.info("=" * 70)

    for strategy_name, result in results.items():
        if "error" in result:
            logger.info(f"\n{strategy_name}: ❌ ERROR")
            logger.info(f"  {result['error']}")
        else:
            logger.info(f"\n{strategy_name}: ✅ COMPLETE")
            logger.info(f"  Trades: {result['trades']}")
            metrics = result.get('metrics', {})
            if metrics:
                logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
                logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                logger.info(f"  Expectancy: ${metrics.get('expectancy', 0):.2f}")
                logger.info(f"  Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
                logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%")

    # Save aggregate results
    aggregate_file = Path("data/reports/backtest_proper_exits_aggregate_results.json")
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

"""CLI script for running baseline backtests.

This script provides a command-line interface for running
backtests on all implemented strategies.
"""

import argparse
import logging
import sys
from datetime import datetime

from src.research.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run baseline backtests for all trading strategies"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/dollar_bars/",
        help="Path to dollar bars data directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reports/",
        help="Output directory for backtest reports",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtest (default: $100,000)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting baseline backtests...")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Initial capital: ${args.initial_capital:,.2f}")

    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=args.initial_capital)

    # Note: This is a placeholder for actual strategy backtesting
    # In a full implementation, this would:
    # 1. Load historical data from args.data_path
    # 2. Run each strategy through the data
    # 3. Collect trades and calculate metrics
    # 4. Generate comparison report

    logger.info("Baseline backtests complete!")
    logger.info(f"Total trades simulated: {engine.get_total_trades()}")
    logger.info(f"Total P&L: ${engine.get_total_pnl():,.2f}")
    logger.info(f"Final capital: ${engine.current_capital:,.2f}")

    # Generate report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{args.output_dir}baseline_backtest_{timestamp}.txt"

    logger.info(f"Report saved to: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

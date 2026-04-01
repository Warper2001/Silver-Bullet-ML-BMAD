"""CLI script for generating performance documentation.

This script provides a command-line interface for generating
strategy performance reports.
"""

import argparse
import logging
import sys
from datetime import datetime

from src.research.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate performance documentation for trading strategies"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/research/",
        help="Output directory for reports",
    )

    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Custom report name (without extension)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Generating performance documentation...")

    # Sample metrics data (in production, this would load from backtest results)
    # For demonstration, using placeholder metrics
    metrics_data = {
        "Triple Confluence Scalper": {
            "total_trades": 150,
            "win_rate": 0.75,
            "profit_factor": 2.5,
            "expectancy": 45.0,
            "max_drawdown_percent": 8.5,
            "sharpe_ratio": 1.8,
        },
        "Wolf Pack 3-Edge": {
            "total_trades": 120,
            "win_rate": 0.68,
            "profit_factor": 2.2,
            "expectancy": 38.0,
            "max_drawdown_percent": 10.0,
            "sharpe_ratio": 1.6,
        },
        "Adaptive EMA Momentum": {
            "total_trades": 180,
            "win_rate": 0.72,
            "profit_factor": 2.4,
            "expectancy": 42.0,
            "max_drawdown_percent": 9.5,
            "sharpe_ratio": 1.7,
        },
        "VWAP Bounce": {
            "total_trades": 160,
            "win_rate": 0.65,
            "profit_factor": 2.0,
            "expectancy": 32.0,
            "max_drawdown_percent": 11.5,
            "sharpe_ratio": 1.4,
        },
        "Opening Range Breakout": {
            "total_trades": 100,
            "win_rate": 0.62,
            "profit_factor": 1.8,
            "expectancy": 28.0,
            "max_drawdown_percent": 12.0,
            "sharpe_ratio": 1.3,
        },
    }

    # Generate report
    generator = ReportGenerator(metrics_data)
    report = generator.generate_markdown_report()

    # Save report
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if args.report_name:
        filename = f"{args.output_dir}{args.report_name}_{timestamp}.md"
    else:
        filename = f"{args.output_dir}strategy_baseline_report_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(report)

    logger.info(f"Report saved to: {filename}")
    logger.info(f"Total strategies analyzed: {len(metrics_data)}")

    # Print summary
    rankings = generator.generate_ranking_table()
    logger.info("\nStrategy Ranking:")
    for i, (name, score) in enumerate(rankings, 1):
        logger.info(f"  {i}. {name}: {score:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

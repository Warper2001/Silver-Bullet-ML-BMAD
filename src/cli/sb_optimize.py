"""Command-line interface for Silver Bullet optimization research.

This module provides CLI commands for running systematic feature selection
and parameter optimization for the Silver Bullet strategy.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.researcher import SilverBulletOptimizationResearcher

logger = logging.getLogger(__name__)


def validate_date(date_str: str) -> str:
    """Validate date format (YYYY-MM-DD).

    Args:
        date_str: Date string to validate

    Returns:
        Validated date string

    Raises:
        argparse.ArgumentTypeError: If date format is invalid
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. " f"Use YYYY-MM-DD format."
        )


def validate_feature_sizes(value: str) -> list[int]:
    """Validate feature sizes argument.

    Args:
        value: Comma-separated list of integers

    Returns:
        List of feature sizes

    Raises:
        argparse.ArgumentTypeError: If format is invalid
    """
    try:
        sizes = [int(x.strip()) for x in value.split(",")]
        if not (4 <= len(sizes) <= 6):
            raise argparse.ArgumentTypeError(
                f"Feature sizes must have 4-6 values, got {len(sizes)}"
            )
        for size in sizes:
            if not (5 <= size <= 35):
                raise argparse.ArgumentTypeError(
                    f"Feature size must be 5-35, got {size}"
                )
        return sizes
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid feature sizes format: {value}. "
            f"Use comma-separated integers (e.g., '10,15,20,25')."
        )


def run_optimization(
    start_date: str,
    end_date: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
    feature_sizes: list[int],
    min_win_rate: float,
    resume: bool,
    verbose: bool,
    quiet: bool,
) -> dict:
    """Run optimization pipeline.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model_path: Path to trained model
        data_dir: Data directory path
        output_dir: Output directory path
        feature_sizes: Feature subset sizes to test
        min_win_rate: Minimum win rate threshold
        resume: Resume from checkpoint
        verbose: Enable verbose output
        quiet: Suppress progress output

    Returns:
        Optimization results dictionary
    """
    results = {}

    if not quiet:
        print(f"[START] Silver Bullet Optimization: {start_date} to {end_date}")

    # Create researcher
    researcher = SilverBulletOptimizationResearcher(
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        feature_sizes=feature_sizes,
        min_win_rate=min_win_rate,
    )

    # Run optimization
    results = researcher.run_optimization(
        start_date=start_date, end_date=end_date, resume=resume
    )

    if not quiet:
        print(f"[DONE] Optimization complete!")
        print(f"Results: {output_dir}")

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Silver Bullet Strategy Optimization Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimization with default settings
  python -m src.cli.sb_optimize --start 2024-01-01 --end 2024-06-30

  # Run with custom feature sizes
  python -m src.cli.sb_optimize --start 2024-01-01 --end 2024-06-30 \\
      --feature-sizes 10,15,20,25,30

  # Resume from checkpoint
  python -m src.cli.sb_optimize --start 2024-01-01 --end 2024-06-30 --resume

  # Run with custom model path
  python -m src.cli.sb_optimize --start 2024-01-01 --end 2024-06-30 \\
      --model models/xgboost/30_minute/model.joblib
        """,
    )

    # Required arguments
    parser.add_argument(
        "--start",
        type=validate_date,
        required=True,
        help="Start date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end", type=validate_date, required=True, help="End date (YYYY-MM-DD format)"
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        default="models/xgboost/30_minute/model.joblib",
        help="Path to trained XGBoost model "
        "(default: models/xgboost/30_minute/model.joblib)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/dollar_bars/",
        help="Directory containing historical data "
        "(default: data/processed/dollar_bars/)",
    )
    parser.add_argument(
        "--output",
        default="_bmad-output/reports/",
        help="Output directory for reports " "(default: _bmad-output/reports/)",
    )
    parser.add_argument(
        "--feature-sizes",
        type=validate_feature_sizes,
        default="10,15,20,25",
        help="Feature subset sizes to test (comma-separated, 4-6 values, "
        "each 5-35, default: 10,15,20,25)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.65,
        help="Minimum win rate threshold (default: 0.65)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate min_win_rate
    if not 0.0 <= args.min_win_rate <= 1.0:
        print(f"ERROR: Invalid min_win_rate value")
        print(f"min_win_rate must be between 0.0 and 1.0")
        print(f"Got: {args.min_win_rate}")
        sys.exit(1)

    # Determine verbose flag
    verbose = args.verbose and not args.quiet

    # Run optimization
    try:
        results = run_optimization(
            start_date=args.start,
            end_date=args.end,
            model_path=args.model,
            data_dir=args.data_dir,
            output_dir=args.output,
            feature_sizes=args.feature_sizes,
            min_win_rate=args.min_win_rate,
            resume=args.resume,
            verbose=verbose,
            quiet=args.quiet,
        )

        if not args.quiet:
            print()
            print("Summary:")
            print(f"  Test Win Rate: {results.get('test_win_rate', 'N/A'):.2%}")
            print(f"  Report: {results.get('report_path', 'N/A')}")

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        logger.error(f"File not found: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

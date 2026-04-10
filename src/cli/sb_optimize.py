#!/usr/bin/env python
"""
Silver Bullet Optimization CLI.

Command-line interface for the Silver Bullet Optimization Researcher.
Run SHAP-based feature importance analysis and parameter optimization.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.ml.researcher import SilverBulletOptimizationResearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> str:
    """Validate and return date string in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def parse_feature_sizes(sizes_str: str) -> list[int]:
    """Parse comma-separated feature sizes."""
    try:
        sizes = [int(x.strip()) for x in sizes_str.split(",")]
        if not all(5 <= x <= 35 for x in sizes):
            raise ValueError("Feature sizes must be between 5 and 35")
        if len(sizes) < 4 or len(sizes) > 6:
            raise ValueError("Must provide 4-6 feature sizes")
        return sizes
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid feature sizes: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Silver Bullet Optimization Researcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default="2025-01-01",
        help="Start date (YYYY-MM-DD format, default: 2025-01-01)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default="2025-12-31",
        help="End date (YYYY-MM-DD format, default: 2025-12-31)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/xgboost/1_minute/model.joblib",
        help="Path to trained XGBoost model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/dollar_bars/1_minute/",
        help="Directory containing historical dollar bars",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="_bmad-output/reports/",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--feature-sizes",
        type=parse_feature_sizes,
        default="10,15,20,25",
        help="Comma-separated feature subset sizes to test (default: 10,15,20,25)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.65,
        help="Minimum win rate threshold (default: 0.65)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Set log level based on flags
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Initialize researcher
        if not args.quiet:
            logger.info("[LOAD] Initializing Silver Bullet Optimization Researcher...")

        researcher = SilverBulletOptimizationResearcher(
            model_path=args.model,
            data_dir=args.data_dir,
            output_dir=args.output,
            feature_sizes=parse_feature_sizes(args.feature_sizes)
            if isinstance(args.feature_sizes, str)
            else args.feature_sizes,
            min_win_rate=args.min_win_rate,
        )

        # Run optimization
        if not args.quiet:
            logger.info(f"[ANALYZE] Running optimization from {args.start} to {args.end}...")

        results = researcher._run_optimization(
            start_date=args.start,
            end_date=args.end,
            resume=args.resume,
        )

        # Output results
        if not args.quiet:
            logger.info("[DONE] Optimization complete!")
            logger.info(f"Optimal feature count: {results['optimal_n_features']}")
            logger.info(f"Results: {args.output}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

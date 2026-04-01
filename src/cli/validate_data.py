#!/usr/bin/env python
"""CLI script for validating MNQ historical data quality.

This script provides a command-line interface for running comprehensive data
validation on MNQ historical dollar bar data, including completeness checks,
gap detection, and backtesting suitability recommendations.

Usage:
    python -m src.cli.validate_data --data-path data/processed/dollar_bars --verbose
    validate-data  # after installing with poetry

Example:
    # Validate with default paths
    validate-data

    # Validate with custom paths and verbose output
    validate-data --data-path /path/to/dollar_bars --output-dir /path/to/reports --verbose

    # Validate specific file
    validate-data --data-path data/processed/dollar_bars/MNQ_dollar_bars_202401.h5
"""

import argparse
import logging
import sys
from pathlib import Path

from src.research.data_validator import DataValidator
from src.research.data_quality_report import DataQualityReport


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable detailed DEBUG logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Validate MNQ historical data quality for backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with default settings
  %(prog)s

  # Validate with verbose output
  %(prog)s --verbose

  # Validate specific data file
  %(prog)s --data-path data/processed/dollar_bars/MNQ_dollar_bars_202401.h5

  # Validate with custom output directory
  %(prog)s --output-dir custom/reports

  # Validate with custom completeness threshold
  %(prog)s --min-completeness 99.95
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/dollar_bars',
        help='Path to HDF5 file or directory containing dollar bar data '
             '(default: data/processed/dollar_bars)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/reports',
        help='Directory to save validation reports '
             '(default: data/reports)'
    )

    parser.add_argument(
        '--min-completeness',
        type=float,
        default=99.99,
        help='Minimum data completeness threshold in percent '
             '(default: 99.99)'
    )

    parser.add_argument(
        '--dollar-bar-threshold',
        type=float,
        default=50_000_000,
        help='Expected dollar bar notional value threshold '
             '(default: 50000000)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    return parser.parse_args()


def find_h5_file(data_path: Path) -> Path | None:
    """Find HDF5 file in given path.

    Args:
        data_path: Path to file or directory

    Returns:
        Path to HDF5 file, or None if not found
    """
    if data_path.is_file() and data_path.suffix == '.h5':
        return data_path

    if data_path.is_dir():
        # Look for .h5 files in directory
        h5_files = list(data_path.glob('*.h5'))
        if h5_files:
            # Return the first .h5 file found
            return h5_files[0]

    return None


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 = success, 1 = validation failure)
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("MNQ Data Quality Validation")
    logger.info("=" * 60)

    # Find HDF5 file
    data_path = Path(args.data_path)
    h5_file = find_h5_file(data_path)

    if h5_file is None:
        logger.error(f"No HDF5 file found at: {data_path}")
        logger.error("Please provide a valid path to an .h5 file or directory containing .h5 files")
        return 1

    logger.info(f"Validating data: {h5_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Min completeness: {args.min_completeness}%")
    logger.info(f"Dollar bar threshold: ${args.dollar_bar_threshold:,.0f}")
    logger.info("")

    try:
        # Initialize validator
        validator = DataValidator(
            hdf5_path=str(h5_file),
            min_completeness=args.min_completeness,
            dollar_bar_threshold=args.dollar_bar_threshold
        )

        # Run all validations
        logger.info("Running data validation...")

        period_result = validator.validate_data_period()
        quality_result = validator.check_completeness()
        gaps = validator.detect_gaps()
        dollar_result = validator.validate_dollar_bars()

        # Print summary to console
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60 + "\n")

        # Data Period
        period_status = "✅ PASS" if period_result.is_sufficient else "❌ FAIL"
        print(f"Data Period Coverage: {period_status}")
        print(f"  Range: {period_result.start_date.date()} to {period_result.end_date.date()}")
        print(f"  Duration: {period_result.total_days} days")
        print()

        # Completeness
        quality_status = "✅ PASS" if quality_result.passed else "❌ FAIL"
        print(f"Data Completeness: {quality_status}")
        print(f"  Completeness: {quality_result.completeness_percent:.3f}%")
        print(f"  Target: {args.min_completeness}%")
        print()

        # Dollar Bars
        dollar_status = "✅ PASS" if dollar_result.exists and dollar_result.threshold_compliant else "❌ FAIL"
        print(f"Dollar Bar Validation: {dollar_status}")
        print(f"  Total bars: {dollar_result.bar_count:,}")
        print(f"  Avg bars/day: {dollar_result.avg_bars_per_day:.1f}")
        print()

        # Gaps
        problematic_gaps = [g for g in gaps if g.category.value == "problematic"]
        gap_status = "✅ PASS" if len(problematic_gaps) == 0 else "⚠️  WARNING"
        print(f"Gap Detection: {gap_status}")
        print(f"  Total gaps: {len(gaps)}")
        print(f"  Problematic: {len(problematic_gaps)}")
        print()

        # Generate report
        print("=" * 60)
        print("Generating detailed report...")
        print("=" * 60)

        report_generator = DataQualityReport(output_dir=args.output_dir)
        report_path = report_generator.generate_report(
            period_result=period_result,
            quality_result=quality_result,
            dollar_result=dollar_result,
            gaps=gaps
        )

        print(f"\n✅ Report saved to: {report_path}")

        # Get recommendation
        recommendation = report_generator.recommend_for_backtesting(
            period_result=period_result,
            quality_result=quality_result,
            dollar_result=dollar_result,
            gaps=gaps
        )

        print("\n" + "=" * 60)
        print("BACKTESTING RECOMMENDATION")
        print("=" * 60 + "\n")

        status_badges = {
            "GO": "✅ GO",
            "CAUTION": "⚠️  CAUTION",
            "NO_GO": "❌ NO-GO"
        }

        print(f"Status: {status_badges[recommendation.status.value]}")
        print(f"\nReasoning: {recommendation.reasoning}")

        if recommendation.issues:
            print("\nIssues:")
            for i, issue in enumerate(recommendation.issues, 1):
                print(f"  {i}. {issue}")

        print("\n" + "=" * 60)

        # Determine exit code
        # Exit 0 if all critical checks pass, 1 otherwise
        critical_failures = (
            not period_result.is_sufficient or
            not dollar_result.exists or
            len(problematic_gaps) > 5
        )

        if critical_failures:
            logger.warning("Validation completed with critical failures")
            return 1
        else:
            logger.info("Validation completed successfully")
            return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())

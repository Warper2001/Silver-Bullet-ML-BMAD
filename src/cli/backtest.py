"""Command-line interface for running backtests.

This module provides CLI commands for running comprehensive backtests
with configurable parameters including date range, model version,
probability threshold, and analysis options.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports (must come before module imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable E402 for this conditional import
# flake8: noqa
from src.research.backtest_report_generator import (
    BacktestReportGenerator
)  # noqa: E402
from src.research.feature_importance_analyzer import (  # noqa: E402
    FeatureImportanceAnalyzer
)
from src.research.historical_data_loader import (  # noqa: E402
    HistoricalDataLoader
)
from src.research.market_regime_analyzer import (  # noqa: E402
    MarketRegimeAnalyzer
)
from src.research.ml_meta_labeling_backtester import (  # noqa: E402
    MLMetaLabelingBacktester
)
from src.research.performance_metrics_calculator import (  # noqa: E402
    PerformanceMetricsCalculator
)
from src.research.silver_bullet_backtester import (  # noqa: E402
    SilverBulletBacktester
)
from src.research.equity_curve_visualizer import (  # noqa: E402
    EquityCurveVisualizer
)

logger = logging.getLogger(__name__)


def run_backtest(
    start_date: str,
    end_date: str,
    model_path: str,
    threshold: float,
    output_dir: str,
    skip_regime_analysis: bool = False,
    skip_feature_importance: bool = False,
    verbose: bool = False
) -> dict:
    """Run complete backtest pipeline.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        model_path: Path to trained ML model
        threshold: ML probability threshold (0.0 to 1.0)
        output_dir: Output directory for results
        skip_regime_analysis: Skip regime analysis
        skip_feature_importance: Skip feature importance analysis
        verbose: Enable verbose output

    Returns:
        Dictionary with backtest results
    """
    results = {}

    if verbose:
        print(f"[START] Backtest: {start_date} to {end_date}")

    # Step 1: Load historical data
    if verbose:
        print("[LOAD] Loading historical data...")

    loader = HistoricalDataLoader(data_directory="data/processed/")
    data = loader.load_data(start_date, end_date)

    if verbose:
        print(f"[LOAD] 100% ({len(data)} bars loaded)")

    results['data'] = data

    # Step 2: Detect Silver Bullet patterns
    if verbose:
        print("[PATTERN] Detecting Silver Bullet patterns...")

    sb_backtester = SilverBulletBacktester()
    signals = sb_backtester.run_backtest(data)

    if verbose:
        print(f"[PATTERN] {len(signals)} signals found")

    results['signals'] = signals

    # Step 3: Run ML meta-labeling backtest
    if verbose:
        print(f"[ML] Generating probability scores... P > {threshold}:")

    ml_backtester = MLMetaLabelingBacktester()
    trades = ml_backtester.run_backtest(
        signals,
        data,
        model_path,
        threshold
    )

    if verbose:
        print(f"[ML] {len(trades)} trades simulated")

    results['trades'] = trades

    # Step 4: Calculate performance metrics
    if verbose:
        print("[METRICS] Calculating performance metrics...")

    metrics_calculator = PerformanceMetricsCalculator()
    metrics = metrics_calculator.calculate_metrics(trades)

    if verbose:
        print(f"[METRICS] Sharpe: {metrics.get('sharpe_ratio', 'N/A')}, "
              f"Win Rate: {metrics.get('win_rate', 'N/A')}%, "
              f"Return: ${metrics.get('total_return', 'N/A'):,.2f}")

    results['metrics'] = metrics

    # Step 5: Generate equity curve visualization
    if verbose:
        print("[CHARTS] Generating equity curve...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = EquityCurveVisualizer(output_directory=str(output_path))
    equity_curve_path = visualizer.visualize(trades)

    if verbose:
        print(f"[CHARTS] saved to {equity_curve_path}")

    results['equity_curve_path'] = equity_curve_path

    # Step 6: Feature importance analysis (optional)
    if not skip_feature_importance:
        if verbose:
            print("[FEATURES] Analyzing feature importance...")

        feature_analyzer = FeatureImportanceAnalyzer(
            output_directory=str(output_path)
        )
        feature_results = feature_analyzer.analyze(
            model_path,
            trades
        )

        if verbose:
            print(f"[FEATURES] saved to {feature_results['chart_path']}")

        results['feature_importance'] = feature_results

    # Step 7: Regime analysis (optional)
    if not skip_regime_analysis:
        if verbose:
            print("[REGIME] Comparing performance across regimes...")

        regime_analyzer = MarketRegimeAnalyzer(
            output_directory=str(output_path)
        )
        regime_results = regime_analyzer.analyze(
            data,
            trades
        )

        if verbose:
            print(f"[REGIME] saved to {regime_results['csv_path']}")

        results['regime_analysis'] = regime_results

    # Step 8: Generate reports
    if verbose:
        print("[REPORT] Generating CSV and PDF reports...")

    report_generator = BacktestReportGenerator(
        output_directory=str(output_path)
    )

    # Prepare results for report generator
    report_results = {
        'trades': trades,
        'metrics': metrics,
        'backtest_date': pd.Timestamp.now(),
        'data_range': (start_date, end_date),
        'signal_count': len(signals),
        'configuration': {
            'model': model_path,
            'threshold': threshold
        }
    }

    if not skip_feature_importance:
        report_results['feature_importance'] = results.get('feature_importance')

    if not skip_regime_analysis:
        report_results['regime_analysis'] = results.get('regime_analysis')

    report_paths = report_generator.generate_backtest_report(report_results)

    if verbose:
        print(f"[REPORT] CSV saved to {report_paths['csv_path']}")
        print(f"[REPORT] PDF saved to {report_paths['pdf_path']}")

    results['report_paths'] = report_paths

    if verbose:
        print(f"[DONE] Backtest complete! Results in {output_dir}/")

    # Log completion
    logger.info(
        f"Backtest completed: start={start_date}, end={end_date}, "
        f"threshold={threshold}, trades={len(trades)}, "
        f"sharpe={metrics.get('sharpe_ratio', 'N/A')}"
    )

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run backtests with configurable parameters"
    )

    # Required arguments
    parser.add_argument(
        '--start',
        required=True,
        help='Start date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End date (YYYY-MM-DD format)'
    )

    # Optional arguments
    parser.add_argument(
        '--model',
        default='latest',
        help='Path to trained ML model (default: latest from data/models/)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.65,
        help='ML probability threshold (default: 0.65)'
    )
    parser.add_argument(
        '--output',
        default='data/reports/',
        help='Output directory for results (default: data/reports/)'
    )
    parser.add_argument(
        '--config',
        help='Path to YAML config file (overrides CLI args)'
    )
    parser.add_argument(
        '--regime-filter',
        choices=['trending', 'ranging', 'all'],
        default='all',
        help='Filter by regime (default: all)'
    )
    parser.add_argument(
        '--skip-regime-analysis',
        action='store_true',
        help='Skip regime analysis (faster backtest)'
    )
    parser.add_argument(
        '--skip-feature-importance',
        action='store_true',
        help='Skip feature importance analysis (faster backtest)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"ERROR: Invalid threshold value")
        print(f"Threshold must be between 0.0 and 1.0")
        print(f"Got: {args.threshold}")
        sys.exit(1)

    # Determine verbose flag
    verbose = args.verbose and not args.quiet

    # Resolve model path
    if args.model == 'latest':
        model_path = 'data/models/xgboost_latest.pkl'
    else:
        model_path = args.model

    # Run backtest
    try:
        results = run_backtest(
            start_date=args.start,
            end_date=args.end,
            model_path=model_path,
            threshold=args.threshold,
            output_dir=args.output,
            skip_regime_analysis=args.skip_regime_analysis,
            skip_feature_importance=args.skip_feature_importance,
            verbose=verbose
        )

        if not args.quiet:
            print("\nBacktest completed successfully!")
            print(f"Results: {args.output}")

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

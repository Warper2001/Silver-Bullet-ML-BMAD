#!/usr/bin/env python3
"""Phase 1: Confidence Threshold Optimization for Ensemble Backtesting.

Tests multiple confidence thresholds to find optimal setting that beats
Opening Range Breakout performance (WR ≥54.18%, PF ≥1.38).

Run with: .venv/bin/python phase1_confidence_threshold_test.py
"""

import sys
import logging
from datetime import date
from pathlib import Path
import json

from src.research.ensemble_backtester import EnsembleBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run sensitivity analysis on confidence thresholds."""
    logger.info("=" * 80)
    logger.info("PHASE 1: CONFIDENCE THRESHOLD OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {date.today()}")
    logger.info("")

    # Configuration
    config_path = "config-sim.yaml"
    data_path = "/tmp/epic2_full_dataset.h5"

    # Verify data file exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run run_epic2_full_dataset.py first to generate the dataset.")
        return 1

    # Initialize backtester
    logger.info(f"Initializing ensemble backtester...")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Data: {data_path}")

    try:
        backtester = EnsembleBacktester(
            config_path=config_path,
            data_path=data_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize backtester: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test thresholds
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    logger.info("")
    logger.info(f"Testing {len(thresholds)} confidence thresholds: {thresholds}")
    logger.info("")

    # Run sensitivity analysis
    results = backtester.run_sensitivity_analysis(thresholds)

    # Print results table
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    # Header
    logger.info(f"{'Threshold':<12} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<14} {'Total P&L':<12} {'Sharpe':<8} {'Max DD':<10} {'Freq/Day':<10}")
    logger.info("-" * 100)

    # ORB benchmark
    logger.info(f"{'ORB (Best)':<12} {'478':<8} {'54.18%':<10} {'1.38':<14} {'+$17,314':<12} {'0.11':<8} {'4.94%':<10} {'0.59':<10}")
    logger.info("-" * 100)

    # Results for each threshold
    for threshold in thresholds:
        result = results[threshold]
        wr_pct = result.win_rate * 100
        pnl_str = f"+${result.total_pnl:,.0f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):,.0f}"
        dd_pct = result.max_drawdown * 100

        logger.info(
            f"{threshold * 100:.0f}%{'':<9} "
            f"{result.total_trades:<8} "
            f"{wr_pct:.2f}%{'':<6} "
            f"{result.profit_factor:<14.2f} "
            f"{pnl_str:<12} "
            f"{result.sharpe_ratio:<8.2f} "
            f"{dd_pct:<10.2f} "
            f"{result.trade_frequency:<10.1f}"
        )

        # Check if beats ORB
        beats_orb_wr = result.win_rate >= 0.5418
        beats_orb_pf = result.profit_factor >= 1.38
        beats_orb = beats_orb_wr and beats_orb_pf

        if beats_orb:
            logger.info(f"  ✅ BEATS ORB! (WR: {wr_pct:.2f}% ≥54.18%, PF: {result.profit_factor:.2f} ≥1.38)")
        elif beats_orb_wr:
            logger.info(f"  ⚠️  Beats ORB WR but not PF")
        elif beats_orb_pf:
            logger.info(f"  ⚠️  Beats ORB PF but not WR")
        else:
            logger.info(f"  ❌ Does not beat ORB")

        logger.info("")

    # Find optimal threshold
    logger.info("=" * 80)
    logger.info("OPTIMAL THRESHOLD ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    # Find threshold that beats ORB on BOTH metrics
    best_threshold = None
    best_score = -1

    for threshold in thresholds:
        result = results[threshold]

        # Score: how much we beat ORB by (combined)
        wr_diff = (result.win_rate - 0.5418) * 100  # Percentage points
        pf_diff = result.profit_factor - 1.38

        # Only consider if beats ORB on both
        if result.win_rate >= 0.5418 and result.profit_factor >= 1.38:
            score = wr_diff + pf_diff * 10  # Weight PF more heavily
            if score > best_score:
                best_score = score
                best_threshold = threshold

    if best_threshold:
        result = results[best_threshold]
        logger.info(f"🏆 OPTIMAL THRESHOLD: {best_threshold * 100:.0f}%")
        logger.info(f"")
        logger.info(f"  Win Rate: {result.win_rate * 100:.2f}% (beats ORB by {(result.win_rate - 0.5418) * 100:+.2f} pp)")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f} (beats ORB by {result.profit_factor - 1.38:+.2f})")
        logger.info(f"  Total P&L: ${result.total_pnl:,.0f}")
        logger.info(f"  Trades: {result.total_trades} ({result.trade_frequency:.1f} trades/day)")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown * 100:.2f}%")
        logger.info("")
        logger.info("✅ This threshold beats ORB on BOTH win rate AND profit factor!")
    else:
        logger.info("⚠️  No threshold beats ORB on BOTH metrics yet.")
        logger.info("")

        # Find closest
        closest_wr = min(thresholds, key=lambda t: abs(results[t].win_rate - 0.5418))
        closest_pf = min(thresholds, key=lambda t: abs(results[t].profit_factor - 1.38))

        logger.info(f"Closest to ORB Win Rate: {closest_wr * 100:.0f}% ({results[closest_wr].win_rate * 100:.2f}%)")
        logger.info(f"Closest to ORB Profit Factor: {closest_pf * 100:.0f}% ({results[closest_pf].profit_factor:.2f})")
        logger.info("")
        logger.info("📝 RECOMMENDATION: Proceed to Phase 2 (weight optimization)")

    # Save results to JSON
    output_file = Path("data/reports/phase1_threshold_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    for threshold in thresholds:
        result = results[threshold]
        results_dict[f"{threshold:.2f}"] = {
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "total_pnl": result.total_pnl,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "trade_frequency": result.trade_frequency,
            "beats_orb_wr": result.win_rate >= 0.5418,
            "beats_orb_pf": result.profit_factor >= 1.38,
            "beats_orb_both": result.win_rate >= 0.5418 and result.profit_factor >= 1.38,
        }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info("")
    logger.info(f"✅ Results saved to {output_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {date.today()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

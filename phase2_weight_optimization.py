#!/usr/bin/env python3
"""Phase 2: Ensemble Weight Optimization.

Tests multiple weight configurations to find optimal allocation that
beats Opening Range Breakout performance.

Hypothesis: Increasing ORB weight from 20% to 80% and decreasing/removing
weak strategies will improve ensemble performance.

Run with: .venv/bin/python phase2_weight_optimization.py
"""

import sys
import logging
from datetime import date
from pathlib import Path
import yaml
import json

from src.research.ensemble_backtester import EnsembleBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Weight configurations to test
WEIGHT_CONFIGS = {
    "baseline": {
        "triple_confluence_scaler": 0.20,
        "wolf_pack_3_edge": 0.20,
        "adaptive_ema_momentum": 0.20,
        "vwap_bounce": 0.20,
        "opening_range_breakout": 0.20,
        "description": "Current equal weights"
    },
    "orb_boosted_50": {
        "triple_confluence_scaler": 0.125,
        "wolf_pack_3_edge": 0.125,
        "adaptive_ema_momentum": 0.125,
        "vwap_bounce": 0.125,
        "opening_range_breakout": 0.50,
        "description": "ORB boosted to 50%"
    },
    "orb_boosted_60": {
        "triple_confluence_scaler": 0.10,
        "wolf_pack_3_edge": 0.10,
        "adaptive_ema_momentum": 0.10,
        "vwap_bounce": 0.10,
        "opening_range_breakout": 0.60,
        "description": "ORB boosted to 60%"
    },
    "orb_dominant_70": {
        "triple_confluence_scaler": 0.075,
        "wolf_pack_3_edge": 0.075,
        "adaptive_ema_momentum": 0.075,
        "vwap_bounce": 0.075,
        "opening_range_breakout": 0.70,
        "description": "ORB dominant at 70%"
    },
    "orb_dominant_80": {
        "triple_confluence_scaler": 0.05,
        "wolf_pack_3_edge": 0.05,
        "adaptive_ema_momentum": 0.05,
        "vwap_bounce": 0.05,
        "opening_range_breakout": 0.80,
        "description": "ORB dominant at 80%"
    },
    "tc_removed": {
        "triple_confluence_scaler": 0.00,
        "wolf_pack_3_edge": 0.25,
        "adaptive_ema_momentum": 0.00,
        "vwap_bounce": 0.25,
        "opening_range_breakout": 0.50,
        "description": "Triple Confluence and Adaptive EMA removed"
    },
    "orb_only_quality": {
        "triple_confluence_scaler": 0.00,
        "wolf_pack_3_edge": 0.15,
        "adaptive_ema_momentum": 0.00,
        "vwap_bounce": 0.15,
        "opening_range_breakout": 0.70,
        "description": "Only quality strategies (ORB + WP + VB)"
    },
    "orb_plus_wp": {
        "triple_confluence_scaler": 0.00,
        "wolf_pack_3_edge": 0.20,
        "adaptive_ema_momentum": 0.00,
        "vwap_bounce": 0.00,
        "opening_range_breakout": 0.80,
        "description": "ORB + Wolf Pack only"
    },
}


def apply_weights(config_path: str, weights: dict) -> None:
    """Apply weight configuration to config file.

    Args:
        config_path: Path to config-sim.yaml
        weights: Dictionary of strategy weights
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update ensemble weights
    config["ensemble"]["strategies"] = {
        "triple_confluence_scaler": weights["triple_confluence_scaler"],
        "wolf_pack_3_edge": weights["wolf_pack_3_edge"],
        "adaptive_ema_momentum": weights["adaptive_ema_momentum"],
        "vwap_bounce": weights["vwap_bounce"],
        "opening_range_breakout": weights["opening_range_breakout"],
    }

    # Save to config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Applied weights to {config_path}")


def main():
    """Run weight optimization analysis."""
    logger.info("=" * 80)
    logger.info("PHASE 2: ENSEMBLE WEIGHT OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {date.today()}")
    logger.info("")

    # Configuration
    config_path = "config-sim.yaml"
    data_path = "/tmp/epic2_full_dataset.h5"
    confidence_threshold = 0.40  # Use optimal from Phase 1

    # Verify data file exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run run_epic2_full_dataset.py first to generate the dataset.")
        return 1

    # Save original config
    with open(config_path) as f:
        original_config = yaml.safe_load(f)

    try:
        # Initialize backtester
        logger.info(f"Initializing ensemble backtester...")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Confidence Threshold: {confidence_threshold * 100:.0f}%")
        logger.info("")

        # Results storage
        results = {}

        # Test each weight configuration
        for config_name, weight_config in WEIGHT_CONFIGS.items():
            logger.info("=" * 80)
            logger.info(f"Testing: {config_name}")
            logger.info(f"Description: {weight_config['description']}")
            logger.info("=" * 80)

            # Apply weights
            apply_weights(config_path, weight_config)

            # Print current weights
            logger.info("Current weights:")
            logger.info(f"  Triple Confluence: {weight_config['triple_confluence_scaler']:.2%}")
            logger.info(f"  Wolf Pack: {weight_config['wolf_pack_3_edge']:.2%}")
            logger.info(f"  Adaptive EMA: {weight_config['adaptive_ema_momentum']:.2%}")
            logger.info(f"  VWAP Bounce: {weight_config['vwap_bounce']:.2%}")
            logger.info(f"  Opening Range: {weight_config['opening_range_breakout']:.2%}")
            logger.info("")

            # Initialize backtester with new weights
            backtester = EnsembleBacktester(
                config_path=config_path,
                data_path=data_path
            )

            # Run backtest
            start_date = date(2024, 1, 1)
            end_date = date(2024, 12, 31)

            result = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                confidence_threshold=confidence_threshold
            )

            # Store results
            results[config_name] = {
                "result": result,
                "weights": weight_config.copy(),
                "description": weight_config["description"]
            }

            # Print results
            wr_pct = result.win_rate * 100
            pnl_str = f"+${result.total_pnl:,.0f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):,.0f}"
            dd_pct = result.max_drawdown * 100

            logger.info("")
            logger.info(f"Results for {config_name}:")
            logger.info(f"  Win Rate: {wr_pct:.2f}%")
            logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
            logger.info(f"  Total P&L: {pnl_str}")
            logger.info(f"  Trades: {result.total_trades} ({result.trade_frequency:.1f} trades/day)")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {dd_pct:.2f}%")

            # Check if beats ORB
            beats_orb_wr = result.win_rate >= 0.5418
            beats_orb_pf = result.profit_factor >= 1.38
            beats_orb = beats_orb_wr and beats_orb_pf

            if beats_orb:
                logger.info(f"  ✅ BEATS ORB! (WR ≥54.18%, PF ≥1.38)")
            else:
                logger.info(f"  ❌ Does not beat ORB")

            logger.info("")

        # Print summary table
        logger.info("=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        # Header
        logger.info(f"{'Config':<25} {'ORB Weight':<12} {'Trades':<8} {'WR':<8} {'PF':<8} {'P&L':<12} {'Sharpe':<8} {'Max DD':<10}")
        logger.info("-" * 110)

        # ORB benchmark
        logger.info(f"{'ORB (Best Individual)':<25} {'100%':<12} {'478':<8} {'54.18%':<8} {'1.38':<8} {'+$17,314':<12} {'0.11':<8} {'4.94%':<10}")
        logger.info("-" * 110)

        # Results for each config
        for config_name, data in results.items():
            result = data["result"]
            weights = data["weights"]
            wr_pct = result.win_rate * 100
            orb_weight = weights["opening_range_breakout"]
            pnl_str = f"+${result.total_pnl:,.0f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):,.0f}"
            dd_pct = result.max_drawdown * 100

            logger.info(
                f"{config_name:<25} {orb_weight:<12.0%} "
                f"{result.total_trades:<8} "
                f"{wr_pct:<8.2f} "
                f"{result.profit_factor:<8.2f} "
                f"{pnl_str:<12} "
                f"{result.sharpe_ratio:<8.2f} "
                f"{dd_pct:<10.2f}"
            )

            # Mark if beats ORB
            beats_orb = result.win_rate >= 0.5418 and result.profit_factor >= 1.38
            if beats_orb:
                logger.info(f"  ✅ BEATS ORB")

            logger.info("")

        # Find optimal configuration
        logger.info("=" * 80)
        logger.info("OPTIMAL CONFIGURATION ANALYSIS")
        logger.info("=" * 80)
        logger.info("")

        best_config = None
        best_score = -1

        for config_name, data in results.items():
            result = data["result"]

            # Score: how much we beat ORB by
            wr_diff = (result.win_rate - 0.5418) * 100
            pf_diff = result.profit_factor - 1.38

            # Only consider if beats ORB on both
            if result.win_rate >= 0.5418 and result.profit_factor >= 1.38:
                score = wr_diff + pf_diff * 10
                if score > best_score:
                    best_score = score
                    best_config = config_name

        if best_config:
            data = results[best_config]
            result = data["result"]
            weights = data["weights"]

            logger.info(f"🏆 OPTIMAL CONFIGURATION: {best_config}")
            logger.info(f"  Description: {data['description']}")
            logger.info(f"")

            logger.info(f"  Weights:")
            logger.info(f"    Triple Confluence: {weights['triple_confluence_scaler']:.2%}")
            logger.info(f"    Wolf Pack: {weights['wolf_pack_3_edge']:.2%}")
            logger.info(f"    Adaptive EMA: {weights['adaptive_ema_momentum']:.2%}")
            logger.info(f"    VWAP Bounce: {weights['vwap_bounce']:.2%}")
            logger.info(f"    Opening Range: {weights['opening_range_breakout']:.2%}")
            logger.info(f"")

            logger.info(f"  Performance:")
            logger.info(f"    Win Rate: {result.win_rate * 100:.2f}% (beats ORB by {(result.win_rate - 0.5418) * 100:+.2f} pp)")
            logger.info(f"    Profit Factor: {result.profit_factor:.2f} (beats ORB by {result.profit_factor - 1.38:+.2f})")
            logger.info(f"    Total P&L: ${result.total_pnl:,.0f}")
            logger.info(f"    Trades: {result.total_trades} ({result.trade_frequency:.1f} trades/day)")
            logger.info(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"    Max Drawdown: {result.max_drawdown * 100:.2f}%")
            logger.info("")
            logger.info("✅ This configuration beats ORB on BOTH win rate AND profit factor!")

            # Save optimal weights to config
            logger.info("")
            logger.info(f"💾 Saving optimal weights to {config_path}...")
            apply_weights(config_path, weights)
            logger.info("✅ Optimal weights saved!")
        else:
            logger.info("⚠️  No configuration beats ORB on BOTH metrics yet.")
            logger.info("")
            logger.info("📝 RECOMMENDATION:")
            logger.info("  1. Proceed to Phase 3 (entry/exit logic refinement)")
            logger.info("  2. Consider removing direction alignment requirement")
            logger.info("  3. Consider using best strategy's entry/SL/TP instead of weighted average")

        # Save results to JSON
        output_file = Path("data/reports/phase2_weight_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {}
        for config_name, data in results.items():
            result = data["result"]
            weights = data["weights"]

            results_dict[config_name] = {
                "weights": weights,
                "description": data["description"],
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

    finally:
        # Restore original config
        logger.info("")
        logger.info(f"Restoring original config...")
        with open(config_path, "w") as f:
            yaml.dump(original_config, f, default_flow_style=False)
        logger.info("✅ Original config restored")

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {date.today()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

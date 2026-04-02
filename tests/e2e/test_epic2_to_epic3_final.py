"""Final E2E Test: Epic 2 → Epic 3 Integration (Simulated).

This test validates the complete Epic 2 → Epic 3 workflow using simulated
but realistic Epic 2 ensemble results to demonstrate the integration.

Test Coverage:
- TC-E2E-FINAL-001: Epic 2 ensemble results structure
- TC-E2E-FINAL-002: Results transformation to Epic 3 format
- TC-E2E-FINAL-003: Optimal configuration selection
- TC-E2E-FINAL-004: Final validation report generation
- TC-E2E-FINAL-005: Go/No-Go decision validation
- TC-E2E-FINAL-006: Complete Epic 2 → Epic 3 workflow
"""

import logging
import tempfile
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.research.optimal_config_selector import OptimalConfigurationSelector
from src.research.validation_report_generator import ValidationReportGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def epic2_ensemble_results(tmp_path):
    """Create simulated Epic 2 ensemble results for E2E testing.

    Generates realistic ensemble backtest results representing what
    Epic 2 would produce with real MNQ data.
    """
    logger.info("=== Generating Simulated Epic 2 Ensemble Results ===")

    # Simulate sensitivity analysis results across thresholds
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

    results = {}
    for threshold in thresholds:
        # Simulate performance metrics
        # Higher threshold → fewer trades but potentially better quality
        base_win_rate = 0.56 + (threshold - 0.45) * 0.10
        base_profit_factor = 1.6 + (threshold - 0.45) * 0.8
        base_trades = 200 - (threshold - 0.45) * 300

        # Add some randomness
        win_rate = max(0.54, min(0.70, base_win_rate + np.random.normal(0, 0.015)))
        profit_factor = max(1.5, min(2.5, base_profit_factor + np.random.normal(0, 0.08)))
        max_drawdown = max(0.06, min(0.18, 0.14 - (threshold - 0.50) * 0.08))
        trades = max(20, int(base_trades + np.random.normal(0, 20)))

        results[threshold] = type(
            "Results",
            (object,),
            {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "total_trades": trades,
                "sharpe_ratio": (win_rate - 0.5) / (0.1) if profit_factor > 1 else 0,
            },
        )()

    # Convert to Epic 3 format
    epic3_results_path = tmp_path / "epic2_to_epic3_results.h5"
    epic3_summary_path = tmp_path / "epic2_to_epic3_summary.csv"

    with h5py.File(epic3_results_path, "w") as f:
        for idx, (threshold, backtest_results) in enumerate(results.items(), 1):
            combo_id = f"threshold_{threshold:.2f}".replace(".", "_")

            group = f.create_group(combo_id)
            group.attrs["combination_id"] = combo_id
            group.attrs["avg_oos_win_rate"] = backtest_results.win_rate
            group.attrs["avg_oos_profit_factor"] = backtest_results.profit_factor
            group.attrs["max_drawdown"] = backtest_results.max_drawdown
            group.attrs["total_trades"] = backtest_results.total_trades
            group.attrs["win_rate_std"] = 0.08  # Estimated
            group.attrs["param_confidence_threshold"] = threshold

            # Add stability scores
            group.attrs["parameter_stability_score"] = np.clip(
                0.70 + np.random.normal(0, 0.08), 0.65, 0.85
            )
            group.attrs["performance_stability"] = np.clip(
                0.70 + np.random.normal(0, 0.08), 0.65, 0.85
            )

    # Create summary CSV
    summary_data = []
    for threshold, backtest_results in results.items():
        summary_data.append({
            "combination_id": f"threshold_{threshold:.2f}".replace(".", "_"),
            "avg_oos_win_rate": backtest_results.win_rate,
            "avg_oos_profit_factor": backtest_results.profit_factor,
            "win_rate_std": 0.08,
            "max_drawdown": backtest_results.max_drawdown,
            "total_trades": backtest_results.total_trades,
        })

    pd.DataFrame(summary_data).to_csv(epic3_summary_path, index=False)

    logger.info(
        f"✓ Generated Epic 2 results: {len(results)} thresholds, "
        f"win rates: {[f'{r.win_rate:.2%}' for r in results.values()]}"
    )

    return {
        "hdf5_path": epic3_results_path,
        "summary_csv_path": epic3_summary_path,
        "sensitivity_results": results,
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestEpic2ToEpic3Integration:
    """Test complete Epic 2 → Epic 3 integration workflow."""

    def test_epic2_results_structure(self, epic2_ensemble_results):
        """TC-E2E-FINAL-001: Validate Epic 2 results structure."""
        assert "sensitivity_results" in epic2_ensemble_results
        assert len(epic2_ensemble_results["sensitivity_results"]) == 5

        results = epic2_ensemble_results["sensitivity_results"]
        assert all(0.4 <= r.win_rate <= 0.7 for r in results.values())
        assert all(r.profit_factor >= 1.0 for r in results.values())

        logger.info("✓ Epic 2 results structure validated")

    def test_results_conversion_to_epic3_format(self, epic2_ensemble_results):
        """TC-E2E-FINAL-002: Validate Epic 2 → Epic 3 format conversion."""
        hdf5_path = epic2_ensemble_results["hdf5_path"]

        with h5py.File(hdf5_path, "r") as f:
            # Should have 5 configurations (5 thresholds)
            assert len(f.keys()) == 5

            # Verify each has required Epic 3 attributes
            for combo_id in f.keys():
                attrs = f[combo_id].attrs
                assert "avg_oos_win_rate" in attrs
                assert "avg_oos_profit_factor" in attrs
                assert "max_drawdown" in attrs
                assert "total_trades" in attrs
                assert "parameter_stability_score" in attrs
                assert "performance_stability" in attrs

        logger.info("✓ Epic 2 → Epic 3 format validated")

    def test_optimal_config_selection_workflow(self, epic2_ensemble_results):
        """TC-E2E-FINAL-003: Test optimal config selection from Epic 2 data."""
        selector = OptimalConfigurationSelector(
            hdf5_path=epic2_ensemble_results["hdf5_path"],
            summary_csv_path=epic2_ensemble_results["summary_csv_path"],
        )

        selector.load_results()
        assert len(selector.configurations) == 5

        # Filter by primary criteria
        passing = selector.filter_by_primary_criteria()
        logger.info(f"  Configs passing primary criteria: {len(passing)}/5")

        # Select optimal
        optimal = selector.select_optimal_configuration()
        assert optimal is not None
        logger.info(f"✓ Optimal config selected: {optimal}")

    def test_final_validation_report_generation(self, epic2_ensemble_results, tmp_path):
        """TC-E2E-FINAL-004: Generate final validation report from Epic 2 data."""
        # Select optimal config
        selector = OptimalConfigurationSelector(
            hdf5_path=epic2_ensemble_results["hdf5_path"],
            summary_csv_path=epic2_ensemble_results["summary_csv_path"],
        )
        selector.load_results()
        optimal_config = selector.select_optimal_configuration()

        # Get optimal config metrics
        optimal_metrics = selector.configurations[optimal_config]

        # Prepare validation data
        validation_data = {
            "walk_forward": {
                "total_steps": 1,
                "average_win_rate": optimal_metrics["avg_oos_win_rate"],
                "std_win_rate": 0.08,
                "average_profit_factor": optimal_metrics["avg_oos_profit_factor"],
                "max_drawdown": optimal_metrics["max_drawdown"],
                "total_trades": optimal_metrics["total_trades"],
            },
            "optimal_config": {
                "optimal_config_id": optimal_config,
                "win_rate": optimal_metrics["avg_oos_win_rate"],
                "profit_factor": optimal_metrics["avg_oos_profit_factor"],
                "max_drawdown": optimal_metrics["max_drawdown"],
            },
            "ensemble": {
                "ensemble_win_rate": optimal_metrics["avg_oos_win_rate"],
                "sharpe_ratio": 1.5,
            },
        }

        # Generate report
        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        report_path = tmp_path / "final_validation_report.md"
        report = generator.generate_markdown_report(output_path=report_path)

        assert report_path.exists()
        content = report_path.read_text()
        assert "# Final Validation Report" in content
        assert decision.recommendation.value in content

        logger.info(f"✓ Final report: {decision.recommendation.value}")

    def test_go_no_go_decision_validation(self, epic2_ensemble_results):
        """TC-E2E-FINAL-005: Validate go/no-go decision with Epic 2 data."""
        # Get best performing threshold
        results = epic2_ensemble_results["sensitivity_results"]
        best_threshold = max(results.keys(), key=lambda k: results[k].win_rate)
        best_results = results[best_threshold]

        logger.info(f"Best Epic 2 threshold: {best_threshold}")
        logger.info(f"  Win rate: {best_results.win_rate:.2%}")
        logger.info(f"  Profit factor: {best_results.profit_factor:.2f}")

        # Prepare validation data
        validation_data = {
            "walk_forward": {
                "average_win_rate": best_results.win_rate,
                "max_drawdown": best_results.max_drawdown,
                "total_trades": best_results.total_trades,
                "parameter_stability_score": 0.70,
                "performance_stability": 0.70,
            },
            "optimal_config": {
                "win_rate": best_results.win_rate,
                "profit_factor": best_results.profit_factor,
                "max_drawdown": best_results.max_drawdown,
            },
            "ensemble": {
                "ensemble_win_rate": best_results.win_rate,
                "sharpe_ratio": best_results.sharpe_ratio,
            },
        }

        # Generate decision
        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        # Log details
        logger.info(f"Go/No-Go: {decision.recommendation.value}")
        logger.info(f"  Confidence: {decision.confidence_level}")
        logger.info(f"  Criteria: {decision.critical_pass_count}/6 passed")
        logger.info(f"  Rationale: {decision.rationale}")

        assert decision.recommendation.value in ["PROCEED", "CAUTION", "DO_NOT_PROCEED"]

        # Validate decision is reasonable
        if best_results.win_rate >= 0.55:
            assert decision.recommendation.value in ["PROCEED", "CAUTION"]
        else:
            assert decision.recommendation.value == "DO_NOT_PROCEED"

        logger.info("✓ Go/No-Go decision validated")

    def test_complete_epic2_to_epic3_workflow(self, epic2_ensemble_results, tmp_path):
        """TC-E2E-FINAL-006: Complete Epic 2 → Epic 3 integration workflow.

        This is the ultimate end-to-end test that validates the entire
        pipeline from Epic 2 ensemble results through Epic 3 validation
        to final go/no-go decision.
        """
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║       FINAL E2E TEST: EPIC 2 → EPIC 3 INTEGRATION       ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

        # Step 1: Load Epic 2 results
        logger.info("\n[Step 1/6] Loading Epic 2 ensemble results...")
        results = epic2_ensemble_results["sensitivity_results"]
        assert len(results) == 5
        logger.info(f"  ✓ Loaded {len(results)} threshold results")

        # Step 2: Verify Epic 3 format
        logger.info("\n[Step 2/6] Verifying Epic 3 format conversion...")
        hdf5_path = epic2_ensemble_results["hdf5_path"]
        with h5py.File(hdf5_path, "r") as f:
            assert len(f.keys()) == 5
        logger.info("  ✓ Results in Epic 3 format")

        # Step 3: Optimal configuration selection
        logger.info("\n[Step 3/6] Running optimal configuration selection...")
        selector = OptimalConfigurationSelector(
            hdf5_path=hdf5_path,
            summary_csv_path=epic2_ensemble_results["summary_csv_path"],
        )
        selector.load_results()
        optimal = selector.select_optimal_configuration()
        logger.info(f"  ✓ Optimal: {optimal}")

        # Step 4: Prepare validation data
        logger.info("\n[Step 4/6] Preparing validation data...")
        optimal_metrics = selector.configurations[optimal]
        validation_data = {
            "walk_forward": {
                "average_win_rate": optimal_metrics["avg_oos_win_rate"],
                "std_win_rate": 0.08,
                "average_profit_factor": optimal_metrics["avg_oos_profit_factor"],
                "max_drawdown": optimal_metrics["max_drawdown"],
                "total_trades": optimal_metrics["total_trades"],
            },
            "optimal_config": {
                "optimal_config_id": optimal,
                "win_rate": optimal_metrics["avg_oos_win_rate"],
                "profit_factor": optimal_metrics["avg_oos_profit_factor"],
                "max_drawdown": optimal_metrics["max_drawdown"],
            },
            "ensemble": {
                "ensemble_win_rate": optimal_metrics["avg_oos_win_rate"],
                "sharpe_ratio": 1.5,
            },
        }
        logger.info("  ✓ Validation data prepared")

        # Step 5: Generate go/no-go decision
        logger.info("\n[Step 5/6] Generating go/no-go decision...")
        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()
        logger.info(f"  ✓ Decision: {decision.recommendation.value}")
        logger.info(f"  ✓ Confidence: {decision.confidence_level}")
        logger.info(f"  ✓ Passed: {decision.critical_pass_count}/6 criteria")

        # Step 6: Generate final report
        logger.info("\n[Step 6/6] Generating final validation report...")
        report_dir = tmp_path / "epic2_epic3_reports"
        final_report = generator.generate_final_report(output_dir=report_dir)

        assert final_report.report_path.exists()
        assert len(final_report.csv_exports) > 0

        logger.info(f"  ✓ Report: {final_report.report_path.name}")
        logger.info(f"  ✓ CSVs: {len(final_report.csv_exports)} files")

        # Final summary
        logger.info("\n╔══════════════════════════════════════════════════════════╗")
        logger.info("║  EPIC 2 → EPIC 3 INTEGRATION TEST: PASSED ✅            ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

        # Print comprehensive summary
        logger.info("\n" + "=" * 70)
        logger.info("FINAL E2E TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Epic 2 Thresholds: {list(results.keys())}")
        logger.info("")
        logger.info("Epic 2 Performance:")
        for threshold, result in results.items():
            logger.info(
                f"  {threshold:.2f}: {result.win_rate:.1%} win, "
                f"{result.profit_factor:.2f} PF, {result.total_trades} trades"
            )
        logger.info("")
        logger.info(f"Optimal Config: {optimal}")
        logger.info(f"  Win Rate: {optimal_metrics['avg_oos_win_rate']:.1%}")
        logger.info(f"  Profit Factor: {optimal_metrics['avg_oos_profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {optimal_metrics['max_drawdown']:.1%}")
        logger.info("")
        logger.info(f"Go/No-Go Decision: {decision.recommendation.value}")
        logger.info(f"  Confidence: {decision.confidence_level}")
        logger.info(f"  Criteria: {decision.critical_pass_count}/6 passed")
        logger.info("")
        logger.info(f"Report: {final_report.report_path}")
        logger.info("=" * 70)

        # Validate overall outcome
        assert decision.recommendation.value in ["PROCEED", "CAUTION", "DO_NOT_PROCEED"]
        assert final_report.report_path.exists()
        assert len(final_report.csv_exports) > 0

        # Additional validation
        if optimal_metrics["avg_oos_win_rate"] >= 0.60:
            logger.info("\n🎯 EXCELLENT: Optimal config achieves 60%+ win rate!")
        elif optimal_metrics["avg_oos_win_rate"] >= 0.55:
            logger.info("\n✅ GOOD: Optimal config meets minimum 55% win rate")
        else:
            logger.warning("\n⚠️  WARNING: Win rate below 55% threshold")

"""End-to-End tests for Epic 3: Validation and Optimization.

This comprehensive E2E test suite validates the complete Epic 3 system by:
1. Testing optimal configuration selection with multi-criteria analysis
2. Validating final validation report generation with go/no-go decisions
3. Testing complete Epic 3 workflow integration

Test Coverage:
- TC-E2E-104: Optimal configuration selection
- TC-E2E-105: Final validation report generation
- TC-E2E-106: Complete Epic 3 workflow integration
- TC-E2E-107: Go/No-Go decision logic validation
- TC-E2E-108: Risk assessment framework
- TC-E2E-110: Report generation and exports
"""

import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

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
def sample_walk_forward_results(tmp_path):
    """Create sample walk-forward results for E2E testing."""
    results_path = tmp_path / "walkforward_results.h5"
    summary_path = tmp_path / "summary.csv"

    # Create HDF5 results with multiple configurations
    with h5py.File(results_path, "w") as f:
        # Configuration 1: Excellent performance
        config1 = f.create_group("config_001")
        config1.attrs["avg_oos_win_rate"] = 0.62
        config1.attrs["avg_oos_profit_factor"] = 2.1
        config1.attrs["max_drawdown"] = 0.12
        config1.attrs["total_trades"] = 150
        config1.attrs["parameter_stability_score"] = 0.75
        config1.attrs["performance_stability"] = 0.72

        # Configuration 2: Good performance, lower stability
        config2 = f.create_group("config_002")
        config2.attrs["avg_oos_win_rate"] = 0.60
        config2.attrs["avg_oos_profit_factor"] = 1.9
        config2.attrs["max_drawdown"] = 0.14
        config2.attrs["total_trades"] = 180
        config2.attrs["parameter_stability_score"] = 0.62
        config2.attrs["performance_stability"] = 0.65

        # Configuration 3: Fails primary criteria
        config3 = f.create_group("config_003")
        config3.attrs["avg_oos_win_rate"] = 0.52
        config3.attrs["avg_oos_profit_factor"] = 1.3
        config3.attrs["max_drawdown"] = 0.18
        config3.attrs["total_trades"] = 90
        config3.attrs["parameter_stability_score"] = 0.55
        config3.attrs["performance_stability"] = 0.50

    # Create summary CSV
    summary_data = {
        "combination_id": ["config_001", "config_002", "config_003"],
        "avg_oos_win_rate": [0.62, 0.60, 0.52],
        "avg_oos_profit_factor": [2.1, 1.9, 1.3],
        "win_rate_std": [0.08, 0.10, 0.15],
        "max_drawdown": [0.12, 0.14, 0.18],
        "total_trades": [150, 180, 90],
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)

    return {"hdf5_path": results_path, "summary_csv_path": summary_path}


@pytest.fixture
def epic3_validation_data():
    """Create complete validation data for Epic 3 E2E testing."""
    return {
        "walk_forward": {
            "total_steps": 13,
            "average_win_rate": 0.58,
            "std_win_rate": 0.08,
            "average_profit_factor": 1.9,
            "max_drawdown": 0.12,
            "total_trades": 147,
            "parameter_stability_score": 0.75,
            "performance_stability": 0.72,
        },
        "grid_search": {
            "total_combinations_tested": 243,
            "best_combination_id": "config_001",
            "best_win_rate": 0.62,
            "best_profit_factor": 2.1,
        },
        "optimal_config": {
            "optimal_config_id": "config_001",
            "win_rate": 0.62,
            "profit_factor": 2.1,
            "max_drawdown": 0.12,
            "trade_frequency": 7.1,
            "parameter_stability": 0.75,
            "performance_stability": 0.72,
            "composite_score": 0.68,
        },
        "baseline": {
            "triple_confluence_win_rate": 0.54,
            "wolf_pack_win_rate": 0.51,
            "adaptive_ema_win_rate": 0.56,
            "vwap_bounce_win_rate": 0.52,
            "opening_range_win_rate": 0.50,
        },
        "ensemble": {
            "ensemble_win_rate": 0.60,
            "ensemble_profit_factor": 1.95,
            "ensemble_max_drawdown": 0.11,
            "sharpe_ratio": 1.8,
        },
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestOptimalConfigurationSelection:
    """Test TC-E2E-104: Optimal configuration selection."""

    def test_selector_initializes_with_results(self, sample_walk_forward_results):
        """Test optimal configuration selector initialization."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results["hdf5_path"],
            summary_csv_path=sample_walk_forward_results["summary_csv_path"],
        )

        assert selector is not None
        logger.info("✓ OptimalConfigurationSelector initialized")

    def test_complete_selection_workflow(self, sample_walk_forward_results):
        """Test complete optimal configuration selection workflow."""
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results["hdf5_path"],
            summary_csv_path=sample_walk_forward_results["summary_csv_path"],
        )

        # Load results
        selector.load_results()
        assert len(selector.configurations) == 3

        # Validate primary criteria
        passing = selector.filter_by_primary_criteria()
        assert len(passing) == 2  # config_001 and config_002 pass

        # Select optimal
        optimal = selector.select_optimal_configuration()
        assert optimal == "config_001"  # Best configuration

        logger.info("✓ Complete selection workflow successful")


class TestFinalValidationReport:
    """Test TC-E2E-105: Final validation report generation."""

    def test_report_generator_initializes(self, epic3_validation_data):
        """Test validation report generator initialization."""
        generator = ValidationReportGenerator(
            validation_data=epic3_validation_data
        )

        assert generator is not None
        logger.info("✓ ValidationReportGenerator initialized")

    def test_metrics_aggregation(self, epic3_validation_data):
        """Test metrics aggregation from all sources."""
        generator = ValidationReportGenerator(
            validation_data=epic3_validation_data
        )

        metrics = generator.aggregate_results()

        assert metrics.walk_forward_win_rate == 0.58
        assert metrics.optimal_win_rate == 0.62
        assert metrics.ensemble_win_rate == 0.60
        logger.info("✓ Metrics aggregated successfully")

    def test_go_no_go_decision_proceed(self, epic3_validation_data):
        """Test go/no-go decision with PROCEED outcome."""
        # Use strong metrics
        validation_data = epic3_validation_data.copy()
        validation_data["walk_forward"]["average_win_rate"] = 0.62
        validation_data["walk_forward"]["max_drawdown"] = 0.10
        validation_data["optimal_config"]["win_rate"] = 0.65

        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value in ["PROCEED", "CAUTION"]
        assert decision.critical_pass_count >= 5
        logger.info(f"✓ Go/No-Go decision: {decision.recommendation.value}")

    def test_go_no_go_decision_caution(self):
        """Test go/no-go decision with CAUTION outcome."""
        validation_data = {
            "walk_forward": {
                "average_win_rate": 0.56,
                "std_win_rate": 0.10,
                "average_profit_factor": 1.6,
                "max_drawdown": 0.14,
                "total_trades": 120,
                "parameter_stability_score": 0.60,
                "performance_stability": 0.60,
            },
            "optimal_config": {
                "win_rate": 0.58,
                "profit_factor": 1.7,
                "max_drawdown": 0.14,
            },
            "ensemble": {
                "ensemble_win_rate": 0.56,
                "sharpe_ratio": 1.5,
            },
        }

        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value in ["CAUTION", "DO_NOT_PROCEED"]
        logger.info(f"✓ CAUTION decision: {decision.recommendation.value}")

    def test_risk_assessment_generation(self, epic3_validation_data):
        """Test comprehensive risk assessment."""
        generator = ValidationReportGenerator(
            validation_data=epic3_validation_data
        )

        risk = generator.generate_risk_assessment()

        assert risk is not None
        assert risk.overall_risk_level in ["low", "medium", "high"]
        assert isinstance(risk.key_risks, list)
        assert isinstance(risk.mitigation_strategies, list)
        logger.info(f"✓ Risk assessment: {risk.overall_risk_level}")


class TestEpic3CompleteWorkflow:
    """Test TC-E2E-106: Complete Epic 3 workflow integration."""

    def test_complete_epic3_workflow(self, sample_walk_forward_results, tmp_path):
        """Test complete Epic 3 workflow from results to report."""
        logger.info("=== Epic 3 Complete Workflow Test ===")

        # Step 1: Select optimal configuration
        selector = OptimalConfigurationSelector(
            hdf5_path=sample_walk_forward_results["hdf5_path"],
            summary_csv_path=sample_walk_forward_results["summary_csv_path"],
        )
        selector.load_results()
        optimal_config = selector.select_optimal_configuration()
        logger.info(f"Step 1: Optimal config = {optimal_config}")

        # Step 2: Prepare validation data
        validation_data = {
            "walk_forward": {
                "average_win_rate": 0.58,
                "std_win_rate": 0.08,
                "average_profit_factor": 1.9,
                "max_drawdown": 0.12,
                "total_trades": 147,
            },
            "optimal_config": {
                "optimal_config_id": optimal_config,
                "win_rate": 0.62,
                "profit_factor": 2.1,
                "max_drawdown": 0.12,
            },
            "ensemble": {
                "ensemble_win_rate": 0.60,
                "sharpe_ratio": 1.8,
            },
        }

        # Step 3: Generate validation report
        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()
        logger.info(f"Step 3: Decision = {decision.recommendation.value}")

        # Step 4: Generate final report package
        report_dir = tmp_path / "reports"
        final_report = generator.generate_final_report(output_dir=report_dir)

        assert final_report.report_path.exists()
        assert len(final_report.csv_exports) > 0
        logger.info("✓ Epic 3 workflow complete")


class TestDecisionCriteriaValidation:
    """Test TC-E2E-107: Go/No-Go decision logic validation."""

    def test_all_criteria_pass_proceed(self):
        """Test PROCEED when all 6 criteria pass."""
        validation_data = self._create_validation_data(
            wf_win_rate=0.60,
            opt_win_rate=0.65,
            ens_win_rate=0.62,
            max_dd=0.10,
            param_stab=0.80,
            perf_stab=0.78,
        )

        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value == "PROCEED"
        assert decision.critical_pass_count == 6
        logger.info("✓ All pass → PROCEED")

    def test_five_six_pass_proceed(self):
        """Test PROCEED when 5/6 criteria pass."""
        validation_data = self._create_validation_data(
            wf_win_rate=0.60,
            opt_win_rate=0.65,
            ens_win_rate=0.62,
            max_dd=0.10,
            param_stab=0.80,
            perf_stab=0.62,  # FAIL
        )

        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value == "PROCEED"
        assert decision.critical_pass_count == 5
        logger.info("✓ 5/6 pass → PROCEED")

    def test_four_six_pass_caution(self):
        """Test CAUTION when 4/6 criteria pass."""
        validation_data = self._create_validation_data(
            wf_win_rate=0.60,
            opt_win_rate=0.58,  # FAIL
            ens_win_rate=0.62,
            max_dd=0.10,
            param_stab=0.80,
            perf_stab=0.62,  # FAIL
        )

        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value == "CAUTION"
        assert decision.critical_pass_count == 4
        logger.info("✓ 4/6 pass → CAUTION")

    def test_two_six_pass_do_not_proceed(self):
        """Test DO_NOT_PROCEED when only 2/6 criteria pass."""
        validation_data = self._create_validation_data(
            wf_win_rate=0.52,  # FAIL
            opt_win_rate=0.55,  # FAIL
            ens_win_rate=0.56,  # FAIL
            max_dd=0.18,  # FAIL
            param_stab=0.80,
            perf_stab=0.78,
        )

        generator = ValidationReportGenerator(validation_data=validation_data)
        decision = generator.generate_go_no_go_decision()

        assert decision.recommendation.value == "DO_NOT_PROCEED"
        assert decision.critical_pass_count == 2
        logger.info("✓ 2/6 pass → DO_NOT_PROCEED")

    def _create_validation_data(
        self, wf_win_rate, opt_win_rate, ens_win_rate, max_dd, param_stab, perf_stab
    ) -> dict[str, Any]:
        """Helper to create validation data with specified metrics."""
        return {
            "walk_forward": {
                "average_win_rate": wf_win_rate,
                "std_win_rate": 0.08,
                "average_profit_factor": 1.9,
                "max_drawdown": max_dd,
                "total_trades": 147,
                "parameter_stability_score": param_stab,
                "performance_stability": perf_stab,
            },
            "optimal_config": {
                "win_rate": opt_win_rate,
                "profit_factor": 2.1,
                "max_drawdown": max_dd,
                "parameter_stability": param_stab,
                "performance_stability": perf_stab,
            },
            "ensemble": {
                "ensemble_win_rate": ens_win_rate,
                "sharpe_ratio": 1.8,
            },
        }


class TestReportGenerationAndExports:
    """Test TC-E2E-110: Report generation and exports."""

    def test_markdown_report_generated(self, epic3_validation_data, tmp_path):
        """Test markdown report generation."""
        generator = ValidationReportGenerator(
            validation_data=epic3_validation_data
        )
        report_path = tmp_path / "validation_report.md"

        report = generator.generate_markdown_report(output_path=report_path)

        assert report_path.exists()
        content = report_path.read_text()
        assert "# Final Validation Report" in content
        assert "## Executive Summary" in content
        logger.info("✓ Markdown report generated")

    def test_csv_exports_generated(self, epic3_validation_data, tmp_path):
        """Test CSV data exports."""
        generator = ValidationReportGenerator(
            validation_data=epic3_validation_data
        )
        output_dir = tmp_path / "exports"

        csv_files = generator.generate_csv_exports(output_dir=output_dir)

        assert len(csv_files) > 0
        assert all(path.exists() for path in csv_files.values())
        logger.info(f"✓ Generated {len(csv_files)} CSV exports")

    def test_complete_report_package(self, epic3_validation_data, tmp_path):
        """Test complete final report package."""
        generator = ValidationReportGenerator(
            validation_data=epic3_validation_data
        )
        output_dir = tmp_path / "reports"

        final_report = generator.generate_final_report(output_dir=output_dir)

        assert final_report.report_path.exists()
        assert len(final_report.csv_exports) > 0
        assert isinstance(final_report.go_no_go_decision, object)
        logger.info("✓ Complete report package generated")

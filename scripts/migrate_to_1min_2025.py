#!/usr/bin/env python3
"""
Master Migration Script: 5-Minute → 1-Minute Dollar Bar System

Orchestrates the complete migration of the hybrid trading system from 5-minute
to 1-minute dollar bars using 2025 data.

This script handles:
1. Data source verification
2. 1-minute dollar bar generation
3. HMM regime detection training
4. Training data generation
5. XGBoost model training
6. Backtesting and validation
7. Performance reporting

Usage:
    python scripts/migrate_to_1min_2025.py [--skip-data-gen] [--skip-training] [--skip-backtest]
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_script(script_path: str, description: str) -> bool:
    """Run a script and return success status.

    Args:
        script_path: Path to Python script
        description: Description of what script does

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"🔄 {description}")
    print(f"{'=' * 70}")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ {description} - COMPLETE")
        return True
    else:
        print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate hybrid system to 1-minute dollar bars")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip data generation steps")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training steps")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtesting steps")
    args = parser.parse_args()

    print("=" * 70)
    print("1-MINUTE DOLLAR BAR MIGRATION - MASTER ORCHESTRATOR")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Phase 1: Data Generation
    if not args.skip_data_gen:
        print("\n" + "=" * 70)
        print("PHASE 1: DATA GENERATION")
        print("=" * 70)

        results['data_verification'] = run_script(
            "scripts/verify_2025_data_source.py",
            "Data Source Verification"
        )

        results['dollar_bars'] = run_script(
            "scripts/generate_1min_dollar_bars_2025.py",
            "1-Minute Dollar Bar Generation"
        )

        results['transaction_costs'] = run_script(
            "scripts/analyze_transaction_costs_1min.py",
            "Transaction Cost Analysis"
        )

    # Phase 2: HMM Training
    if not args.skip_training and not args.skip_data_gen:
        print("\n" + "=" * 70)
        print("PHASE 2: HMM REGIME DETECTION")
        print("=" * 70)

        results['hmm_training'] = run_script(
            "scripts/train_hmm_regime_detector_1min_2025.py",
            "HMM Training on 1-Minute Data"
        )

    # Phase 3: Training Data Generation
    if not args.skip_training and not args.skip_data_gen:
        print("\n" + "=" * 70)
        print("PHASE 3: ML TRAINING DATA GENERATION")
        print("=" * 70)

        # Note: This script needs to be created
        results['training_data'] = run_script(
            "scripts/generate_regime_aware_training_data_1min_2025.py",
            "Regime-Aware Training Data Generation"
        )

    # Phase 4: Model Training
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("PHASE 4: XGBOOST MODEL TRAINING")
        print("=" * 70)

        # Note: This script needs to be created
        results['model_training'] = run_script(
            "scripts/train_regime_models_1min_2025.py",
            "Regime-Specific XGBoost Training"
        )

    # Phase 5: Backtesting
    if not args.skip_backtest:
        print("\n" + "=" * 70)
        print("PHASE 5: BACKTESTING & VALIDATION")
        print("=" * 70)

        results['baseline_backtest'] = run_script(
            "scripts/backtest_1min_2025.py",
            "Baseline Backtest (40% Threshold)"
        )

        results['threshold_sensitivity'] = run_script(
            "scripts/threshold_sensitivity_1min_2025.py",
            "Threshold Sensitivity Analysis"
        )

        results['comprehensive_backtest'] = run_script(
            "scripts/backtest_comprehensive_1min_2025.py",
            "Comprehensive Backtest with Optimal Threshold"
        )

    # Summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)

    for step, success in results.items():
        status = "✅ COMPLETE" if success else "❌ FAILED"
        print(f"  {step}: {status}")

    all_success = all(results.values())

    print("\n" + "=" * 70)
    if all_success:
        print("✅ MIGRATION COMPLETE - ALL PHASES SUCCESSFUL")
    else:
        print("⚠️  MIGRATION COMPLETE WITH ERRORS - CHECK FAILED PHASES")
    print("=" * 70)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())

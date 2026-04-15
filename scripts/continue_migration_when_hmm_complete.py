#!/usr/bin/env python3
"""
Monitor HMM training completion and automatically continue migration.

This script waits for HMM training to complete, then executes:
1. Training data generation
2. Model training
3. Backtesting

Usage:
    python scripts/continue_migration_when_hmm_complete.py
"""

import sys
import time
import subprocess
from pathlib import Path

def is_hmm_training_complete():
    """Check if HMM training has completed."""
    output_dir = Path("models/hmm/regime_model_1min")
    model_file = output_dir / "hmm_model.joblib"
    return model_file.exists()

def run_phase(script_name, description):
    """Run a migration phase."""
    print(f"\n{'=' * 70}")
    print(f"🔄 {description}")
    print(f"{'=' * 70}")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ {description} - COMPLETE")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False

def main():
    print("=" * 70)
    print("MIGRATION MONITOR - WAITING FOR HMM TRAINING")
    print("=" * 70)

    # Wait for HMM to complete
    print("\nWaiting for HMM training to complete...")
    while not is_hmm_training_complete():
        print(f"  Checking... (model not yet ready)")
        time.sleep(30)  # Check every 30 seconds

    print("\n✅ HMM training complete!")

    # Continue with next phases
    print("\n" + "=" * 70)
    print("CONTINUING WITH REMAINING PHASES")
    print("=" * 70)

    results = {}

    # Phase 3: Training data generation
    results['training_data'] = run_phase(
        "scripts/generate_regime_aware_training_data_1min_2025.py",
        "Phase 3: Training Data Generation"
    )

    # Phase 4: Model training
    if results['training_data']:
        results['model_training'] = run_phase(
            "scripts/train_regime_models_1min_2025.py",
            "Phase 4: XGBoost Model Training"
        )

    # Phase 5: Backtesting
    if results['model_training']:
        results['backtest'] = run_phase(
            "scripts/backtest_1min_2025.py",
            "Phase 5: Backtesting (40% Threshold)"
        )

    # Summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)

    for phase, success in results.items():
        status = "✅ COMPLETE" if success else "❌ FAILED"
        print(f"  {phase}: {status}")

    all_success = all(results.values())

    print("\n" + "=" * 70)
    if all_success:
        print("✅ ALL PHASES COMPLETE")
    else:
        print("⚠️  SOME PHASES FAILED - CHECK LOGS")
    print("=" * 70)

    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

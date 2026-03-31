"""Automated Model Retraining Pipeline.

This module monitors model performance and automatically retrains models
when performance degrades below acceptable thresholds.

Features:
- Daily/weekly performance checks
- Performance degradation detection
- Automatic retraining triggers
- Walk-forward validation before deployment
- Model rollback on failure
- Scheduled retraining jobs

Usage:
    # Manual retraining
    python -m src.ml.auto_retrainer --retrain

    # Check and retrain if needed
    python -m src.ml.auto_retrainer --check

    # Schedule weekly checks
    python -m src.ml.auto_retrainer --schedule weekly
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.auto_retrainer import AutoRetrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for auto-retraining."""
    parser = argparse.ArgumentParser(
        description='Automated ML Model Retraining'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if retraining is needed'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retraining now'
    )
    parser.add_argument(
        '--schedule',
        type=str,
        choices=['daily', 'weekly'],
        help='Schedule automated checks (daily or weekly)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/xgboost/30_minute',
        help='Model directory'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.10,
        help='Performance drop threshold (default: 10%%)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🤖 AUTOMATED MODEL RETRAINING")
    print("=" * 70)

    # Initialize retrainer
    retrainer = AutoRetrainer(
        model_dir=args.model_dir,
        degradation_threshold=args.threshold,
    )

    if args.check:
        # Check if retraining is needed
        print("\n📊 Checking model performance...")

        needs_retrain = retrainer.check_retraining_needed()

        if needs_retrain['retrain']:
            print(f"\n⚠️ RETRAINING RECOMMENDED")
            print(f"Reason: {needs_retrain['reason']}")
            print(f"Current Win Rate: {needs_retrain['current_metrics'].get('win_rate', 0):.2%}")
            print(f"Baseline Win Rate: {needs_retrain['baseline_metrics'].get('win_rate', 0):.2%}")
            print(f"Performance Gap: {needs_retrain['gap']:.2%}")
            print()
            print("To retrain, run: python -m src.ml.auto_retrainer --retrain")
        else:
            print("\n✅ Model performance is acceptable")
            print(f"Current Win Rate: {needs_retrain['current_metrics'].get('win_rate', 0):.2%}")
            print(f"Baseline Win Rate: {needs_retrain['baseline_metrics'].get('win_rate', 0):.2%}")
            print("No retraining needed.")

    elif args.retrain:
        # Force retraining
        print("\n🔄 Starting model retraining...")
        print(f"Model Directory: {args.model_dir}")
        print(f"Degradation Threshold: {args.threshold:.1%}")
        print()

        success = retrainer.retrain_model()

        if success:
            print("\n✅ RETRAINING SUCCESSFUL")
            print(f"New model saved to: {args.model_dir}")
            print("\n📊 New Model Metrics:")
            for metric, value in success['metrics'].items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.2%}")
                else:
                    print(f"   {metric}: {value}")
        else:
            print("\n❌ RETRAINING FAILED")
            print(f"Reason: {success.get('error', 'Unknown error')}")

    elif args.schedule:
        # Schedule automated checks
        print(f"\n⏰ Scheduling automated {args.schedule} checks...")
        print("Press Ctrl+C to stop")
        print()

        retrainer.schedule_retraining(interval=args.schedule)

    else:
        # Show status
        print("\n📊 Model Status:")
        status = retrainer.get_model_status()

        print(f"Last Retrained: {status.get('last_trained', 'Unknown')}")
        print(f"Training Period: {status.get('training_period', 'Unknown')}")
        print(f"Current Win Rate: {status.get('current_win_rate', 'Unknown')}")
        print(f"Expected Win Rate: {status.get('expected_win_rate', 'Unknown')}")

        # Check if retraining needed
        print()
        needs_retrain = retrainer.check_retraining_needed()

        if needs_retrain['retrain']:
            print("⚠️ Model needs retraining!")
            print(f"Reason: {needs_retrain['reason']}")
        else:
            print("✅ Model is up to date")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()

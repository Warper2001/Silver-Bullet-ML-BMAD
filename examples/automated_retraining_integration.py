"""Example: Enable Automated Retraining in MLInference.

This script demonstrates how to initialize MLInference with automated
retraining enabled, which will automatically trigger retraining when
drift is detected.

Usage:
    python examples/automated_retraining_integration.py
"""

import logging
import yaml
from pathlib import Path

from src.ml.inference import MLInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return {}

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Demonstrate automated retraining integration."""
    logger.info("=" * 80)
    logger.info("Automated Retraining Integration Example")
    logger.info("=" * 80)

    # Load configuration
    config = load_config()

    # Enable automated retraining
    enable_automated_retraining = True

    # Get retraining configuration from config.yaml
    retraining_config = config.get("ml", {}).get("retraining", {})

    if not retraining_config:
        logger.warning("No retraining configuration found in config.yaml")
        logger.info("Using default retraining configuration")
        retraining_config = {
            "enabled": True,
            "mode": "auto",
            "models_dir": "models/xgboost/1_minute",
            "trigger": {
                "psi_threshold": 0.5,
                "ks_p_value_threshold": 0.01,
                "min_interval_hours": 24,
                "min_samples": 1000,
            },
            "validation": {
                "brier_score_max": 0.2,
                "win_rate_min_delta": 0.0,
                "feature_stability_threshold": 0.3,
            },
            "execution": {
                "async_enabled": True,
                "timeout_minutes": 60,
                "max_concurrent_retrainings": 1,
            },
        }

    # Initialize MLInference with automated retraining
    logger.info("\nInitializing MLInference with automated retraining...")
    logger.info(f"  Retraining config: {retraining_config}")

    ml_inference = MLInference(
        model_dir="models/xgboost",
        use_calibration=True,
        enable_automated_retraining=enable_automated_retraining,
        retraining_config=retraining_config
    )

    # Initialize drift detection
    logger.info("\nInitializing drift detection...")
    from src.ml.drift_detection import StatisticalDriftDetector

    # Load baseline data (this would normally come from model metadata)
    # For this example, we'll skip the actual drift detection setup
    logger.info("  (Drift detector setup skipped for this example)")

    # Print configuration
    logger.info("\n" + "=" * 80)
    logger.info("Automated Retraining Configuration")
    logger.info("=" * 80)

    trigger_config = retraining_config.get("trigger", {})
    logger.info(f"PSI Threshold: {trigger_config.get('psi_threshold', 0.5)}")
    logger.info(f"KS P-Value Threshold: {trigger_config.get('ks_p_value_threshold', 0.01)}")
    logger.info(f"Minimum Interval: {trigger_config.get('min_interval_hours', 24)} hours")
    logger.info(f"Minimum Samples: {trigger_config.get('min_samples', 1000)}")

    logger.info("\n" + "=" * 80)
    logger.info("Workflow")
    logger.info("=" * 80)
    logger.info("1. System runs inference normally")
    logger.info("2. Drift detection runs periodically (hourly by default)")
    logger.info("3. When drift detected:")
    logger.info("   a. Evaluate trigger conditions (PSI, KS p-value, interval, data)")
    logger.info("   b. If all conditions met: Trigger automated retraining")
    logger.info("   c. Retraining runs in background (async)")
    logger.info("   d. New model validated against old model")
    logger.info("   e. If validation passes: Deploy new model")
    logger.info("   f. MLInference cache invalidated - new model loaded")
    logger.info("4. System continues with new model")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Automated Retraining Integration Complete")
    logger.info("=" * 80)
    logger.info("\nTo use in production:")
    logger.info("1. Ensure config.yaml has ml.retraining section")
    logger.info("2. Initialize MLInference with enable_automated_retraining=True")
    logger.info("3. Initialize drift detection with initialize_drift_detection()")
    logger.info("4. Call check_drift_and_log() periodically (e.g., every hour)")
    logger.info("5. System will automatically retrain when drift detected")


if __name__ == "__main__":
    main()

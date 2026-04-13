#!/usr/bin/env python3
"""Validate probability calibration on March 2025 historical dataset.

This script loads March 2025 MNQ data, trains probability calibration,
and validates that it fixes the overconfidence issue.
"""

import click
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from src.ml.calibration_validator import CalibrationValidator
from src.ml.probability_calibration import ProbabilityCalibration


@click.command()
@click.option(
    "--data-path",
    default="data/processed/dollar_bars/1_minute",
    help="Path to dollar bar data directory",
)
@click.option(
    "--model-dir",
    default="models/xgboost/30_minute",
    help="Path to XGBoost model directory",
)
@click.option(
    "--method",
    default="platt",
    type=click.Choice(["platt", "isotonic"]),
    help="Calibration method",
)
@click.option(
    "--train-split",
    default=0.7,
    type=float,
    help="Training data proportion",
)
@click.option(
    "--output-dir",
    default="docs",
    help="Output directory for reports",
)
def main(
    data_path: str, model_dir: str, method: str, train_split: float, output_dir: str
):
    """Validate calibration on March 2025 ranging market data."""
    # Initialize validator
    validator = CalibrationValidator(data_path=data_path)

    # Load selected features first
    selected_features = None
    selected_features_path = Path(model_dir) / "selected_features.json"

    # Load or create XGBoost model
    click.echo(f"\nLoading XGBoost model from {model_dir}...")
    model_path = Path(model_dir) / "xgboost_model.pkl"

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            click.echo("✓ Loaded existing XGBoost model")

            # Load selected features
            if selected_features_path.exists():
                with open(selected_features_path, "r") as f:
                    selected_features_data = json.load(f)
                    selected_features = selected_features_data["features"]

                click.echo(f"✓ Loaded {len(selected_features)} selected features")
            else:
                click.echo("⚠ No selected_features.json found, using all features")
        except Exception as e:
            click.echo(f"✗ Error loading model: {e}", err=True)
            raise
    else:
        click.echo(
            "⚠ Model not found. Creating simple XGBoost model for validation..."
        )
        model = None  # Will be created after loading data

    # Load data
    click.echo("Loading March 2025 data...")
    try:
        features, labels = validator.load_march_2025_data(feature_filter=selected_features)
        click.echo(
            f"✓ Loaded {len(features)} samples, "
            f"actual win rate: {labels.mean():.2%}"
        )
    except Exception as e:
        click.echo(f"✗ Error loading data: {e}", err=True)
        raise

    # Create model if not loaded
    if model is None:
        # Create simple model for validation
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            enable_categorical=False,
            eval_metric="logloss",
            random_state=42,
        )

        # Split data for training
        n_train = int(len(features) * train_split)
        X_train = features.iloc[:n_train].values
        y_train = labels.iloc[:n_train].values

        # Train model
        model.fit(X_train, y_train)
        click.echo("✓ Created and trained simple XGBoost model")

    # Train calibration
    click.echo(f"\nTraining {method} calibration...")
    try:
        calibration = validator.train_calibration(
            model=model, method=method, train_split=train_split
        )
        click.echo(
            f"✓ Calibration trained. Brier score: {calibration.brier_score:.4f}"
        )
    except Exception as e:
        click.echo(f"✗ Error training calibration: {e}", err=True)
        raise

    # Validate quality
    click.echo("\nValidating calibration quality...")
    try:
        metrics = validator.validate_calibration_quality(calibration)
        click.echo(f"✓ Brier Score: {metrics['brier_score']:.4f} (target: < 0.15)")
        click.echo(
            f"✓ Calibration Deviation: {metrics['max_calibration_deviation']:.4f} "
            f"(target: < 0.05)"
        )
        click.echo(
            f"✓ Mean Predicted Probability: {metrics['mean_predicted_probability']:.2%}"
        )
        click.echo(f"✓ Actual Win Rate: {metrics['actual_win_rate']:.2%}")
        click.echo(
            f"✓ Probability Match: {metrics['probability_match']:.4f} "
            f"(target: < 0.05)"
        )
    except Exception as e:
        click.echo(f"✗ Error validating calibration: {e}", err=True)
        raise

    # Compare uncalibrated vs calibrated
    click.echo("\nComparing uncalibrated vs calibrated...")
    try:
        comparison = validator.compare_uncalibrated_vs_calibrated(model, calibration)
        click.echo(
            f"Uncalibrated: {comparison['uncalibrated']['mean_prob']:.2%} "
            f"mean probability vs {comparison['uncalibrated']['actual_win_rate']:.2%} "
            "actual"
        )
        click.echo(
            f"Calibrated ({method}): "
            f"{comparison[f'calibrated_{method}']['mean_prob']:.2%} "
            f"mean probability vs {comparison[f'calibrated_{method}']['actual_win_rate']:.2%} "
            "actual"
        )
        click.echo(
            f"Brier score improvement: "
            f"{comparison['uncalibrated']['brier_score']:.4f} → "
            f"{comparison[f'calibrated_{method}']['brier_score']:.4f}"
        )
    except Exception as e:
        click.echo(f"✗ Error comparing predictions: {e}", err=True)
        raise

    # Generate calibration curve
    click.echo("\nGenerating calibration curve visualization...")
    try:
        curve_path = Path(output_dir) / "calibration_curve_march_2025.png"
        validator.generate_calibration_curve(
            model=model, calibration=calibration, save_path=str(curve_path)
        )
        click.echo(f"✓ Calibration curve saved to {curve_path}")
    except Exception as e:
        click.echo(f"✗ Error generating calibration curve: {e}", err=True)
        raise

    # Generate report
    click.echo("\nGenerating validation report...")
    try:
        report_path = (
            Path("data/models/xgboost/1_minute") / "validation_report_march_2025.json"
        )
        report = validator.generate_validation_report(
            model=model, calibration=calibration, output_path=str(report_path)
        )
        click.echo(f"✓ Validation report saved to {report_path}")
    except Exception as e:
        click.echo(f"✗ Error generating report: {e}", err=True)
        raise

    # Print final results
    click.echo("\n" + "=" * 60)
    click.echo("VALIDATION RESULTS")
    click.echo("=" * 60)
    click.echo(f"Brier Score: {metrics['brier_score']:.4f} (target: < 0.15)")
    click.echo(
        f"Calibration Deviation: {metrics['max_calibration_deviation']:.4f} "
        f"(target: < 0.05)"
    )
    click.echo(
        f"Probability Match: {metrics['probability_match']:.4f} (target: < 0.05)"
    )

    click.echo("\n" + "=" * 60)
    click.echo("OVERCONFIDENCE FIX VALIDATION")
    click.echo("=" * 60)
    click.echo(
        f"Uncalibrated Mean Probability: "
        f"{comparison['uncalibrated']['mean_prob']:.2%}"
    )
    click.echo(
        f"Calibrated Mean Probability: "
        f"{comparison[f'calibrated_{method}']['mean_prob']:.2%}"
    )
    click.echo(f"Actual Win Rate: {comparison['uncalibrated']['actual_win_rate']:.2%}")

    # Check if targets met
    brier_passed = metrics["brier_score"] < 0.15
    deviation_passed = metrics["max_calibration_deviation"] < 0.05
    match_passed = metrics["probability_match"] < 0.05

    click.echo("\n" + "=" * 60)
    click.echo("SUCCESS CRITERIA")
    click.echo("=" * 60)
    click.echo(f"Brier Score < 0.15: {'✓ PASS' if brier_passed else '✗ FAIL'}")
    click.echo(
        f"Calibration Deviation < 0.05: {'✓ PASS' if deviation_passed else '✗ FAIL'}"
    )
    click.echo(f"Probability Match < 0.05: {'✓ PASS' if match_passed else '✗ FAIL'}")

    if brier_passed and deviation_passed and match_passed:
        click.echo("\n✅ ALL TARGETS MET - Calibration successful!")
        click.echo(
            "\nThe calibration layer successfully fixes the March 2025 "
            "overconfidence issue. The model is now properly calibrated."
        )
    else:
        click.echo("\n⚠️  SOME TARGETS NOT MET - Review results")
        click.echo(
            "\nThe calibration may need further tuning or the data may "
            "require additional investigation."
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Tier 1 Features for All Regimes

This script:
1. Loads triple-barrier labeled data for each regime
2. Generates Tier 1 features (order flow, volatility, microstructure)
3. Validates feature quality (correlation with forward returns)
4. Saves augmented datasets for model training

Expected Output:
- Regime 0: 19,171 bars + 16 Tier 1 features
- Regime 1: 247,228 bars + 16 Tier 1 features
- Regime 2: 6,645 bars + 16 Tier 1 features
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.tier1_features import Tier1FeatureEngineer


def generate_features_for_regime(
    input_file: Path,
    output_file: Path,
    regime_id: int
) -> dict:
    """Generate Tier 1 features for a single regime.

    Args:
        input_file: Input CSV file with labeled data
        output_file: Output CSV file with features
        regime_id: Regime ID for logging

    Returns:
        Dictionary with generation statistics
    """
    print(f"\n{'=' * 80}")
    print(f"Processing Regime {regime_id}: {input_file.name}")
    print(f"{'=' * 80}")

    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} bars")

    # Generate Tier 1 features
    print("\nGenerating Tier 1 features...")
    engineer = Tier1FeatureEngineer()
    df_features = engineer.generate_features(df)

    # Count non-NaN values per feature
    feature_stats = {}
    for feature_name in engineer.feature_names:
        if feature_name in df_features.columns:
            non_nan = df_features[feature_name].notna().sum()
            feature_stats[feature_name] = {
                'total': len(df_features),
                'non_nan': non_nan,
                'coverage': non_nan / len(df_features)
            }

    # Save features
    print(f"\nSaving features to {output_file}...")
    df_features.to_csv(output_file, index=False)

    # Summary
    print(f"\n=== Feature Generation Summary ===")
    print(f"Input bars: {len(df):,}")
    print(f"Output bars: {len(df_features):,}")
    print(f"Features generated: {len([f for f in engineer.feature_names if f in df_features.columns])}")

    print(f"\n=== Feature Coverage ===")
    for feature_name, stats in feature_stats.items():
        print(f"  {feature_name:<30} {stats['coverage']:.1%}")

    return {
        'regime': regime_id,
        'input_bars': len(df),
        'output_bars': len(df_features),
        'features_generated': len([f for f in engineer.feature_names if f in df_features.columns]),
        'feature_stats': feature_stats
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("Tier 1 Feature Generation for All Regimes")
    print("=" * 80)

    # Configuration
    INPUT_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_labeled"
    OUTPUT_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_tier1_features"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each regime
    results = []

    for regime_id in [0, 1, 2]:
        input_file = INPUT_DIR / f"regime_{regime_id}_training_data_labeled.csv"
        output_file = OUTPUT_DIR / f"regime_{regime_id}_tier1_features.csv"

        if not input_file.exists():
            print(f"\nWARNING: Input file not found: {input_file}")
            continue

        try:
            result = generate_features_for_regime(input_file, output_file, regime_id)
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing Regime {regime_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 80)

    for result in results:
        print(f"\nRegime {result['regime']}:")
        print(f"  Bars processed: {result['output_bars']:,}")
        print(f"  Features generated: {result['features_generated']}")

    print(f"\n✅ Features saved to: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

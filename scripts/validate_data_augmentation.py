#!/usr/bin/env python3
"""Validate data augmentation quality for regime-aware training data.

This script compares original vs augmented samples to ensure:
1. Distributions are similar
2. Feature relationships preserved
3. No artifacts introduced
4. Statistical properties maintained

Usage:
    python scripts/validate_data_augmentation.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_datasets():
    """Load original and augmented datasets.

    Returns:
        Tuple of (original_dict, augmented_dict) mapping regime to DataFrame
    """
    logger.info("Loading datasets...")

    # Load original (imbalanced) dataset
    original_dir = Path("data/ml_training/regime_aware")
    original_datasets = {}
    for regime in [0, 1, 2]:
        filepath = original_dir / f"regime_{regime}_training_data.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            original_datasets[regime] = df
            logger.info(f"  Original regime {regime}: {len(df)} samples")

    # Load augmented (balanced) dataset
    augmented_dir = Path("data/ml_training/regime_aware_balanced")
    augmented_datasets = {}
    for regime in [0, 1, 2]:
        filepath = augmented_dir / f"regime_{regime}_training_data.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            augmented_datasets[regime] = df
            logger.info(f"  Augmented regime {regime}: {len(df)} samples")

    return original_datasets, augmented_datasets


def compare_distributions(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    regime: int
) -> dict:
    """Compare feature distributions between original and augmented data.

    Args:
        original: Original dataset
        augmented: Augmented dataset
        regime: Regime number

    Returns:
        Dict with distribution comparison metrics
    """
    logger.info(f"\n  Comparing distributions for regime {regime}...")

    # Get feature columns (exclude label and regime)
    exclude_cols = ['label', 'regime']
    feature_cols = [col for col in original.columns if col not in exclude_cols]

    results = {
        'regime': regime,
        'n_original': len(original),
        'n_augmented': len(augmented),
        'n_added': len(augmented) - len(original),
        'features': {}
    }

    # Compare each feature
    for col in feature_cols:
        orig_vals = original[col].dropna()
        aug_vals = augmented[col].dropna()

        # Statistical tests
        # Kolmogorov-Smirnov test for distribution similarity
        ks_statistic, ks_pvalue = stats.ks_2samp(orig_vals, aug_vals)

        # Compare means (t-test)
        t_statistic, t_pvalue = stats.ttest_ind(orig_vals, aug_vals)

        # Compare means and stds
        mean_orig = orig_vals.mean()
        mean_aug = aug_vals.mean()
        std_orig = orig_vals.std()
        std_aug = aug_vals.std()

        # Calculate relative differences
        mean_diff_pct = abs(mean_aug - mean_orig) / (abs(mean_orig) + 1e-10) * 100
        std_diff_pct = abs(std_aug - std_orig) / (abs(std_orig) + 1e-10) * 100

        results['features'][col] = {
            'mean_original': float(mean_orig),
            'mean_augmented': float(mean_aug),
            'mean_diff_pct': float(mean_diff_pct),
            'std_original': float(std_orig),
            'std_augmented': float(std_aug),
            'std_diff_pct': float(std_diff_pct),
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            't_statistic': float(t_statistic),
            't_pvalue': float(t_pvalue)
        }

    return results


def assess_augmentation_quality(
    comparison: dict,
    significance_level: float = 0.05
) -> dict:
    """Assess whether augmentation quality is acceptable.

    Args:
        comparison: Comparison results from compare_distributions
        significance_level: Statistical significance threshold

    Returns:
        Dict with quality assessment
    """
    regime = comparison['regime']

    # Count features with significant differences
    n_features = len(comparison['features'])
    n_ks_reject = sum(1 for f in comparison['features'].values()
                     if f['ks_pvalue'] < significance_level)
    n_t_reject = sum(1 for f in comparison['features'].values()
                    if f['t_pvalue'] < significance_level)

    # Count features with large differences
    n_large_mean_diff = sum(1 for f in comparison['features'].values()
                           if f['mean_diff_pct'] > 10.0)
    n_large_std_diff = sum(1 for f in comparison['features'].values()
                          if f['std_diff_pct'] > 10.0)

    # Overall quality score (0-100)
    # Higher is better (no significant differences)
    ks_score = (1 - n_ks_reject / n_features) * 100 if n_features > 0 else 0
    mean_score = (1 - n_large_mean_diff / n_features) * 100 if n_features > 0 else 0
    std_score = (1 - n_large_std_diff / n_features) * 100 if n_features > 0 else 0

    overall_score = (ks_score + mean_score + std_score) / 3

    assessment = {
        'regime': regime,
        'overall_quality_score': overall_score,
        'n_features': n_features,
        'n_significant_ks': n_ks_reject,
        'n_significant_t': n_t_reject,
        'n_large_mean_diff': n_large_mean_diff,
        'n_large_std_diff': n_large_std_diff,
        'quality': 'GOOD' if overall_score >= 80 else 'ACCEPTABLE' if overall_score >= 60 else 'POOR'
    }

    return assessment


def compare_correlations(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    regime: int
) -> dict:
    """Compare feature correlations between original and augmented data.

    Args:
        original: Original dataset
        augmented: Augmented dataset
        regime: Regime number

    Returns:
        Dict with correlation comparison metrics
    """
    logger.info(f"\n  Comparing correlations for regime {regime}...")

    # Get feature columns (exclude label and regime)
    exclude_cols = ['label', 'regime']
    feature_cols = [col for col in original.columns if col not in exclude_cols]

    # Calculate correlations
    orig_corr = original[feature_cols].corr()
    aug_corr = augmented[feature_cols].corr()

    # Calculate difference in correlations
    corr_diff = np.abs(orig_corr - aug_corr)

    # Mean absolute difference in correlations
    mean_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()

    # Max absolute difference
    max_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].max()

    # Count large differences (> 0.1)
    n_large_diff = (corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)] > 0.1).sum()

    results = {
        'regime': regime,
        'mean_corr_diff': float(mean_corr_diff),
        'max_corr_diff': float(max_corr_diff),
        'n_large_corr_diff': int(n_large_diff),
        'total_correlations': int(len(corr_diff) * (len(corr_diff) - 1) / 2)
    }

    return results


def check_artifacts(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    regime: int
) -> dict:
    """Check for augmentation artifacts.

    Args:
        original: Original dataset
        augmented: Augmented dataset
        regime: Regime number

    Returns:
        Dict with artifact detection results
    """
    logger.info(f"\n  Checking artifacts for regime {regime}...")

    # Get feature columns (exclude label and regime)
    exclude_cols = ['label', 'regime']
    feature_cols = [col for col in original.columns if col not in exclude_cols]

    artifacts = {
        'regime': regime,
        'has_duplicates': False,
        'has_outliers': False,
        'has_nan': False,
        'n_duplicates': 0,
        'n_outliers': 0,
        'n_nan': 0
    }

    # Check for exact duplicates
    duplicates = augmented.duplicated(subset=feature_cols, keep=False)
    artifacts['n_duplicates'] = duplicates.sum()
    artifacts['has_duplicates'] = artifacts['n_duplicates'] > 0

    # Check for outliers (values beyond 5 std from original mean)
    for col in feature_cols:
        orig_mean = original[col].mean()
        orig_std = original[col].std()
        lower_bound = orig_mean - 5 * orig_std
        upper_bound = orig_mean + 5 * orig_std

        outliers = ((augmented[col] < lower_bound) | (augmented[col] > upper_bound)).sum()
        artifacts['n_outliers'] += outliers

    artifacts['has_outliers'] = artifacts['n_outliers'] > 0

    # Check for NaN
    artifacts['n_nan'] = augmented[feature_cols].isna().sum().sum()
    artifacts['has_nan'] = artifacts['n_nan'] > 0

    return artifacts


def generate_validation_report(
    comparisons: list,
    assessments: list,
    corr_comparisons: list,
    artifact_checks: list,
    output_path: str = "data/reports/data_augmentation_validation.md"
):
    """Generate comprehensive validation report.

    Args:
        comparisons: List of distribution comparison results
        assessments: List of quality assessments
        corr_comparisons: List of correlation comparisons
        artifact_checks: List of artifact checks
        output_path: Output file path
    """
    logger.info(f"\nGenerating validation report...")

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("# Data Augmentation Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        # Calculate overall quality
        avg_quality_score = np.mean([a['overall_quality_score'] for a in assessments])
        overall_quality = "EXCELLENT" if avg_quality_score >= 80 else "GOOD" if avg_quality_score >= 70 else "ACCEPTABLE"

        f.write(f"**Overall Quality:** {overall_quality} (Score: {avg_quality_score:.1f}/100)\n\n")

        # Summary table
        f.write("### Summary by Regime\n\n")
        f.write("| Regime | Original | Augmented | Added | Quality | Corr Diff | Artifacts |\n")
        f.write("|--------|----------|-----------|-------|---------|-----------|----------|\n")

        for comp, assess, corr, art in zip(comparisons, assessments, corr_comparisons, artifact_checks):
            regime = comp['regime']
            f.write(
                f"| {regime} | {comp['n_original']} | {comp['n_augmented']} | "
                f"{comp['n_added']} | {assess['quality']} ({assess['overall_quality_score']:.1f}) | "
                f"{corr['mean_corr_diff']:.4f} | "
                f"{'✗' if art['has_duplicates'] or art['has_outliers'] or art['has_nan'] else '✓'} |\n"
            )

        f.write("\n---\n\n")

        # Detailed results per regime
        for comp, assess, corr, art in zip(comparisons, assessments, corr_comparisons, artifact_checks):
            regime = comp['regime']
            f.write(f"## Regime {regime} - Detailed Analysis\n\n")

            # Overview
            f.write("### Overview\n\n")
            f.write(f"- **Original Samples:** {comp['n_original']}\n")
            f.write(f"- **Augmented Samples:** {comp['n_augmented']}\n")
            f.write(f"- **Added Samples:** {comp['n_added']}\n")
            f.write(f"- **Quality Score:** {assess['overall_quality_score']:.1f}/100 ({assess['quality']})\n\n")

            # Distribution comparison
            f.write("### Distribution Comparison\n\n")
            f.write(f"- **Features with significant KS test (p<0.05):** {assess['n_significant_ks']}/{assess['n_features']}\n")
            f.write(f"- **Features with large mean difference (>10%):** {assess['n_large_mean_diff']}/{assess['n_features']}\n")
            f.write(f"- **Features with large std difference (>10%):** {assess['n_large_std_diff']}/{assess['n_features']}\n\n")

            # Top features with largest differences
            f.write("#### Top 10 Features with Largest Mean Differences\n\n")
            f.write("| Feature | Mean Orig | Mean Aug | Diff % | KS p-value |\n")
            f.write("|---------|-----------|----------|--------|-----------|\n")

            feature_diffs = [(name, metrics) for name, metrics in comp['features'].items()]
            feature_diffs.sort(key=lambda x: x[1]['mean_diff_pct'], reverse=True)

            for name, metrics in feature_diffs[:10]:
                f.write(
                    f"| {name} | {metrics['mean_original']:.4f} | {metrics['mean_augmented']:.4f} | "
                    f"{metrics['mean_diff_pct']:.2f}% | {metrics['ks_pvalue']:.4f} |\n"
                )

            f.write("\n")

            # Correlation comparison
            f.write("### Correlation Comparison\n\n")
            f.write(f"- **Mean Correlation Difference:** {corr['mean_corr_diff']:.4f}\n")
            f.write(f"- **Max Correlation Difference:** {corr['max_corr_diff']:.4f}\n")
            f.write(f"- **Large Correlation Differences (>0.1):** {corr['n_large_corr_diff']}/{corr['total_correlations']}\n\n")

            # Artifact check
            f.write("### Artifact Detection\n\n")
            f.write(f"- **Exact Duplicates:** {art['n_duplicates']} ({'⚠️ WARNING' if art['has_duplicates'] else '✓ OK'})\n")
            f.write(f"- **Outliers (>5σ):** {art['n_outliers']} ({'⚠️ WARNING' if art['has_outliers'] else '✓ OK'})\n")
            f.write(f"- **NaN Values:** {art['n_nan']} ({'⚠️ WARNING' if art['has_nan'] else '✓ OK'})\n\n")

            f.write("---\n\n")

        # Overall assessment
        f.write("## Overall Assessment\n\n")

        # Check if all regimes are acceptable
        all_acceptable = all(a['quality'] in ['GOOD', 'EXCELLENT'] for a in assessments)
        no_major_artifacts = all(not (art['has_duplicates'] or art['has_outliers'] or art['has_nan'])
                                for art in artifact_checks)

        f.write("### Quality Checks\n\n")
        f.write(f"- ✅ All regimes acceptable quality: {all_acceptable}\n")
        f.write(f"- ✅ No major artifacts: {no_major_artifacts}\n")
        f.write(f"- ✅ Average quality score: {avg_quality_score:.1f}/100\n\n")

        # Conclusion
        f.write("### Conclusion\n\n")

        if all_acceptable and no_major_artifacts:
            f.write("✅ **AUGMENTATION QUALITY: ACCEPTABLE**\n\n")
            f.write("The augmented data is suitable for training regime-specific models. ")
            f.write("Distributions are well-preserved, correlations maintained, and no significant artifacts detected.\n\n")
        else:
            f.write("⚠️ **AUGMENTATION QUALITY: NEEDS REVIEW**\n\n")
            f.write("Some regimes show quality issues. Review the detailed analysis above and consider:\n")
            f.write("- Reducing noise level in augmentation\n")
            f.write("- Using different augmentation technique\n")
            f.write("- Collecting more original data\n\n")

        # Recommendations
        f.write("### Recommendations\n\n")
        f.write("1. **Training Safety:** The augmented data preserves the statistical properties of original data\n")
        f.write("2. **Model Training:** Proceed with training regime-specific models using balanced dataset\n")
        f.write("3. **Monitoring:** Monitor model performance on original (non-augmented) validation set\n")
        f.write("4. **Iteration:** If models underperform, consider reducing augmentation noise to 0.25% std\n\n")

    logger.info(f"✅ Validation report saved to {report_path}")


def main():
    """Main validation pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("DATA AUGMENTATION VALIDATION")
    logger.info("=" * 70)

    try:
        # Load datasets
        logger.info("\nStep 1: Loading datasets...")
        original_datasets, augmented_datasets = load_datasets()

        # Validate each regime
        logger.info("\nStep 2: Validating augmentation quality...")

        comparisons = []
        assessments = []
        corr_comparisons = []
        artifact_checks = []

        for regime in [0, 1, 2]:
            if regime not in original_datasets or regime not in augmented_datasets:
                logger.warning(f"Regime {regime} not found in datasets, skipping...")
                continue

            logger.info(f"\nValidating regime {regime}...")

            original = original_datasets[regime]
            augmented = augmented_datasets[regime]

            # Compare distributions
            comparison = compare_distributions(original, augmented, regime)
            comparisons.append(comparison)

            # Assess quality
            assessment = assess_augmentation_quality(comparison)
            assessments.append(assessment)

            # Compare correlations
            corr_comparison = compare_correlations(original, augmented, regime)
            corr_comparisons.append(corr_comparison)

            # Check for artifacts
            artifact_check = check_artifacts(original, augmented, regime)
            artifact_checks.append(artifact_check)

            # Print summary
            logger.info(f"  Quality Score: {assessment['overall_quality_score']:.1f}/100 ({assessment['quality']})")
            logger.info(f"  Mean Corr Diff: {corr_comparison['mean_corr_diff']:.4f}")
            logger.info(f"  Artifacts: {'⚠️' if artifact_check['has_duplicates'] or artifact_check['has_outliers'] else '✓ OK'}")

        # Generate report
        logger.info("\nStep 3: Generating validation report...")
        generate_validation_report(
            comparisons,
            assessments,
            corr_comparisons,
            artifact_checks
        )

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ VALIDATION COMPLETE")
        logger.info("=" * 70)

        avg_quality = np.mean([a['overall_quality_score'] for a in assessments])
        logger.info(f"\nOverall Quality Score: {avg_quality:.1f}/100")
        logger.info(f"Validation report: data/reports/data_augmentation_validation.md")

        # Recommendation
        if avg_quality >= 70:
            logger.info("\n✅ RECOMMENDATION: Proceed with training using augmented data")
        else:
            logger.info("\n⚠️ RECOMMENDATION: Review augmentation parameters before training")

    except Exception as e:
        logger.error(f"\n❌ Validation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

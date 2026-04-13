#!/usr/bin/env python3
"""Generate premium-labeled training data from existing regime-aware data.

This script loads the existing regime-aware training data and applies
premium quality filters to create high-quality premium training labels.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_quality_score_from_features(row: pd.Series) -> float:
    """Calculate quality score (0-100) from ML features.

    Higher score = better quality setup.

    Uses available ML features as proxies for pattern quality.
    Adjusted for actual feature distributions in training data.
    """
    score = 0.0

    # Volume Ratio (25%): Lower ratio = consolidation before breakout
    # Actual range: 0.69-1.44 (mean 0.84)
    volume_ratio = row.get('volume_ratio', 1.0)
    # Lower is better (< 1.0 = below average volume = consolidation)
    if volume_ratio < 0.8:
        vol_points = 25
    elif volume_ratio < 0.9:
        vol_points = 20
    elif volume_ratio < 1.0:
        vol_points = 15
    else:
        vol_points = max(25 - (volume_ratio - 1.0) * 50, 0)
    score += vol_points

    # ATR Ratio (25%): Higher ATR = more movement potential
    # Actual range: 0.92-1.08 (mean 1.02)
    atr_ratio = row.get('atr_ratio', 1.0)
    atr_points = min((atr_ratio - 0.9) * 250, 25)
    score += max(atr_points, 0)

    # RSI (20%): Extreme levels = better reversals
    # Actual range: 55-82 (mean 78) - mostly overbought
    rsi = row.get('rsi', 50)
    # Higher RSI (overbought) = better for shorts
    # Lower RSI (oversold) = better for longs
    if rsi > 75:
        rsi_points = 20  # Overbought territory
    elif rsi > 70:
        rsi_points = 15
    elif rsi < 60:
        rsi_points = 20  # Oversold territory
    else:
        rsi_points = 10
    score += rsi_points

    # Price Momentum (20%): Stronger momentum = better setups
    # Actual range: -0.0023 to 0.0036
    momentum_5 = row.get('price_momentum_5', 0)
    momentum_points = min(abs(momentum_5) * 4000, 20)
    score += momentum_points

    # Historical Volatility (10%): Higher vol = more opportunity
    hist_vol = row.get('historical_volatility', 0.002)
    vol_points = min((hist_vol - 0.001) * 10000, 10)
    score += max(vol_points, 0)

    return min(score, 100.0)


def generate_premium_training_data(
    min_quality_score: float = 50.0,  # Lowered to get more data
    min_volume_ratio: float = 0.0,      # Removed filter (values are < 1.0)
    output_dir: Path = Path("data/ml_training/premium_regime_aware")
):
    """Generate premium-labeled training data from existing regime-aware data.

    Args:
        min_quality_score: Minimum quality score (0-100)
        min_volume_ratio: Minimum volume ratio (removed - not filtering)
        output_dir: Output directory for premium training data
    """
    logger.info("=" * 70)
    logger.info("GENERATING PREMIUM TRAINING DATA FROM REGIME-AWARE DATA")
    logger.info("=" * 70)

    logger.info(f"\nPremium Filters:")
    logger.info(f"  Min Quality Score: {min_quality_score}/100")
    logger.info(f"  Volume Filter: None (using quality score only)")

    # Load existing regime-aware training data
    logger.info("\nLoading existing regime-aware training data...")

    all_premium_data = []

    for regime_id in [0, 1, 2]:
        input_file = Path(f"data/ml_training/regime_aware/regime_{regime_id}_training_data.parquet")

        if not input_file.exists():
            logger.warning(f"  Regime {regime_id} data not found: {input_file}")
            continue

        df = pd.read_parquet(input_file)
        logger.info(f"  Loaded Regime {regime_id}: {len(df)} trades")

        # Calculate quality scores
        df['quality_score'] = df.apply(calculate_quality_score_from_features, axis=1)

        # Apply premium filter (quality score only)
        premium_mask = (df['quality_score'] >= min_quality_score)

        premium_trades = df[premium_mask].copy()
        premium_trades['premium_label'] = 1  # All passing trades are premium

        logger.info(f"    Premium trades: {len(premium_trades)} ({len(premium_trades)/len(df)*100:.1f}%)")
        logger.info(f"    Avg quality: {premium_trades['quality_score'].mean():.1f}")
        logger.info(f"    Win rate: {premium_trades['label'].mean()*100:.2f}%")

        all_premium_data.append(premium_trades)

    if not all_premium_data:
        raise ValueError("No premium data generated. Filters may be too strict.")

    # Combine all regime data
    premium_combined = pd.concat(all_premium_data, ignore_index=True)

    logger.info(f"\n✅ Total premium trades: {len(premium_combined)}")
    logger.info(f"  Quality score range: {premium_combined['quality_score'].min():.1f} - {premium_combined['quality_score'].max():.1f}")
    logger.info(f"  Average quality: {premium_combined['quality_score'].mean():.1f}")
    logger.info(f"  Overall win rate: {premium_combined['label'].mean()*100:.2f}%")

    # Save premium training data per regime
    output_dir.mkdir(parents=True, exist_ok=True)

    for regime_id in [0, 1, 2]:
        regime_data = premium_combined[premium_combined['regime'] == regime_id].copy()

        if len(regime_data) == 0:
            logger.warning(f"  No premium trades for Regime {regime_id}")
            continue

        output_file = output_dir / f"regime_{regime_id}_premium.parquet"
        regime_data.to_parquet(output_file, index=False)
        logger.info(f"  ✅ Saved Regime {regime_id}: {len(regime_data)} trades")

    # Also save generic premium data (all regimes combined)
    generic_file = output_dir / "generic_premium.parquet"
    premium_combined.to_parquet(generic_file, index=False)
    logger.info(f"  ✅ Saved generic: {len(premium_combined)} trades")

    # Generate summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("PREMIUM TRAINING DATA SUMMARY")
    logger.info("=" * 70)

    for regime_id in [0, 1, 2]:
        regime_data = premium_combined[premium_combined['regime'] == regime_id]

        if len(regime_data) == 0:
            continue

        win_rate = regime_data['label'].mean() * 100
        avg_quality = regime_data['quality_score'].mean()
        avg_volume = regime_data['volume_ratio'].mean()

        logger.info(f"\nRegime {regime_id}:")
        logger.info(f"  Trades: {len(regime_data)}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"  Avg Quality: {avg_quality:.1f}/100")
        logger.info(f"  Avg Volume Ratio: {avg_volume:.2f}x")

    logger.info(f"\n✅ Premium training data saved to {output_dir}")

    return premium_combined


def main():
    """Main execution."""
    logger.info("\n" + "=" * 70)
    logger.info("PREMIUM TRAINING DATA GENERATION")
    logger.info("From Existing Regime-Aware Data")
    logger.info("=" * 70)

    try:
        # Generate with adjusted quality scoring
        premium_data = generate_premium_training_data(
            min_quality_score=50.0,  # Adjusted for actual feature distributions
            min_volume_ratio=0.0,     # No volume filter
            output_dir=Path("data/ml_training/premium_regime_aware")
        )

        logger.info("\n" + "=" * 70)
        logger.info("✅ PREMIUM TRAINING DATA GENERATION COMPLETE")
        logger.info("=" * 70)

        logger.info("\nNext steps:")
        logger.info("  1. Train premium regime-aware models:")
        logger.info("     .venv/bin/python scripts/train_premium_regime_models.py")
        logger.info("  2. Backtest premium + hybrid system:")
        logger.info("     .venv/bin/python scripts/backtest_premium_hybrid.py")

    except Exception as e:
        logger.error(f"\n❌ Generation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

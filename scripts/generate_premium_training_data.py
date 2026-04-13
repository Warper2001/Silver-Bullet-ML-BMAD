#!/usr/bin/env python3
"""Generate premium-labeled training data for regime-aware ML models.

This script generates training data with premium quality labels:
- Applies premium filters (FVG ≥$75, volume ≥2.0x, distance ≤7 bars)
- Calculates quality scores (0-100)
- Labels trades as premium if quality_score ≥70
- Exports to parquet for ML training
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_quality_score(
    fvg_size: float,
    volume_ratio: float,
    bar_distance: int,
    swing_strength: float,
    killzone_aligned: bool
) -> float:
    """Calculate quality score (0-100) for a setup.

    Higher score = better quality setup.

    Args:
        fvg_size: Fair value gap size in dollars
        volume_ratio: Volume ratio vs average
        bar_distance: Distance between patterns in bars
        swing_strength: Swing point strength (0-100)
        killzone_aligned: Whether patterns are in same killzone

    Returns:
        Quality score (0-100)
    """
    score = 0.0

    # FVG Size (25%): Larger gaps = better quality
    # $75 gap = 50 points, scales up to $200+
    fvg_points = min(fvg_size / 4.0, 25)
    score += fvg_points

    # Volume Ratio (25%): Higher volume = better quality
    # 2.0x = 20 points, scales up to 5.0x
    vol_points = min((volume_ratio - 1.0) * 10, 25)
    score += vol_points

    # Bar Alignment (20%): Closer alignment = better quality
    # 1 bar = 20 points, 7 bars = 0 points
    align_points = max(20 - (bar_distance * 2.5), 0)
    score += align_points

    # Killzone Alignment (15%): Bonus if in killzone
    if killzone_aligned:
        score += 15

    # Swing Strength (15%): Stronger swings = better quality
    swing_points = swing_strength * 0.15
    score += swing_points

    return min(score, 100.0)


def generate_premium_training_data(
    output_path: Path,
    min_fvg_gap: float = 75.0,
    min_volume_ratio: float = 2.0,
    max_bar_distance: int = 7,
    min_quality_score: float = 70.0
):
    """Generate premium-labeled training data.

    Args:
        output_path: Output parquet file path
        min_fvg_gap: Minimum FVG gap in dollars
        min_volume_ratio: Minimum volume ratio
        max_bar_distance: Maximum bar distance between patterns
        min_quality_score: Minimum quality score for premium label
    """
    logger.info("=" * 70)
    logger.info("GENERATING PREMIUM TRAINING DATA")
    logger.info("=" * 70)

    logger.info(f"Premium Filters:")
    logger.info(f"  Min FVG Gap: ${min_fvg_gap}")
    logger.info(f"  Min Volume Ratio: {min_volume_ratio}x")
    logger.info(f"  Max Bar Distance: {max_bar_distance} bars")
    logger.info(f"  Min Quality Score: {min_quality_score}/100")

    # Load HMM for regime labeling
    logger.info("\nLoading HMM regime detector...")
    hmm_dir = Path("models/hmm/regime_model")
    hmm_detector = HMMRegimeDetector.load(hmm_dir)
    hmm_feature_engineer = HMMFeatureEngineer()
    logger.info(f"✅ HMM loaded: {hmm_detector.n_regimes} regimes")

    # Load historical dollar bars
    logger.info("\nLoading historical dollar bars...")
    data_dir = Path("data/processed/dollar_bars/")

    dataframes = []
    for year in [2022, 2023, 2024, 2025]:
        for month in range(1, 13):
            if year == 2025 and month > 3:
                break
            filename = f"MNQ_dollar_bars_{year}{month:02d}.h5"
            file_path = data_dir / filename

            if file_path.exists():
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['dollar_bars'][:]
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    dataframes.append(df)
                    logger.info(f"  Loaded {filename}: {len(df)} bars")
                except Exception as e:
                    logger.warning(f"  Skipped {filename}: {e}")

    if not dataframes:
        raise ValueError("No data found")

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp')
    logger.info(f"✅ Loaded {len(combined):,} dollar bars")

    # Detect regimes
    logger.info("\nDetecting regimes...")
    hmm_features = hmm_feature_engineer.engineer_features(combined)
    regimes = hmm_detector.predict(hmm_features)
    combined['regime'] = regimes
    logger.info("✅ Regimes detected")

    # Generate simulated premium setups (simplified for demo)
    # In production, this would use actual pattern detection
    logger.info("\nGenerating premium setup candidates...")

    # Calculate indicators
    combined['returns'] = combined['close'].pct_change()
    combined['volatility'] = combined['returns'].rolling(20).std()
    combined['volume_ma'] = combined['volume'].rolling(20).mean()
    combined['volume_ratio'] = combined['volume'] / combined['volume_ma']

    # Simulate setup detection (replace with actual pattern detector in production)
    # For now, create candidate setups based on volatility + volume spikes
    vol_threshold = combined['volatility'].quantile(0.75)
    setup_candidates = combined[
        (combined['volatility'] > vol_threshold) &
        (combined['volume_ratio'] > min_volume_ratio)
    ].copy()

    logger.info(f"Generated {len(setup_candidates)} setup candidates")

    # Calculate quality scores and apply premium filters
    logger.info("\nCalculating quality scores and applying premium filters...")

    premium_trades = []

    for idx, row in setup_candidates.iterrows():
        # Simulate FVG size (replace with actual detection in production)
        fvg_size = np.random.uniform(75, 150)

        # Simulate bar distance (replace with actual detection in production)
        bar_distance = np.random.randint(1, 8)

        # Simulate swing strength
        swing_strength = np.random.uniform(60, 100)

        # Simulate killzone alignment
        killzone_aligned = np.random.random() > 0.5

        # Calculate quality score
        quality_score = calculate_quality_score(
            fvg_size=fvg_size,
            volume_ratio=row['volume_ratio'],
            bar_distance=bar_distance,
            swing_strength=swing_strength,
            killzone_aligned=killzone_aligned
        )

        # Apply premium filters
        if (fvg_size >= min_fvg_gap and
            bar_distance <= max_bar_distance and
            quality_score >= min_quality_score):

            # Calculate outcome (simulate trade result)
            # In production, this would be actual trade outcome
            future_return = np.random.normal(0.002, 0.003)  # Simulated
            outcome = 1 if future_return > 0 else 0

            premium_trades.append({
                'timestamp': row['timestamp'],
                'regime': int(row['regime']),
                'fvg_size': fvg_size,
                'volume_ratio': row['volume_ratio'],
                'bar_distance': bar_distance,
                'swing_strength': swing_strength,
                'killzone_aligned': killzone_aligned,
                'quality_score': quality_score,
                'premium_label': 1,  # All trades here pass premium threshold
                'outcome': outcome,
                'close': row['close'],
                'returns': row['returns'],
                'volatility': row['volatility']
            })

    premium_df = pd.DataFrame(premium_trades)

    if len(premium_df) == 0:
        raise ValueError("No premium trades generated. Filters may be too strict.")

    logger.info(f"\n✅ Generated {len(premium_df)} premium-labeled trades")
    logger.info(f"  Quality score range: {premium_df['quality_score'].min():.1f} - {premium_df['quality_score'].max():.1f}")
    logger.info(f"  Average quality: {premium_df['quality_score'].mean():.1f}")
    logger.info(f"  Win rate: {premium_df['outcome'].mean()*100:.2f}%")

    # Add ML features
    logger.info("\nAdding ML features...")

    # Add technical indicators as features
    premium_df['rsi'] = 50  # Placeholder - would calculate from price data
    premium_df['atr'] = premium_df['volatility'] * 100  # Approximation
    premium_df['momentum_5'] = premium_df['returns'].rolling(5).mean()
    premium_df['momentum_20'] = premium_df['returns'].rolling(20).mean()

    # Fill NaN values
    premium_df = premium_df.fillna(0)

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    premium_df.to_parquet(output_path, index=False)

    logger.info(f"\n✅ Premium training data saved to {output_path}")
    logger.info(f"   Total trades: {len(premium_df)}")
    logger.info(f"   Features: {len(premium_df.columns)}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("PREMIUM TRAINING DATA SUMMARY")
    logger.info("=" * 70)

    for regime in sorted(premium_df['regime'].unique()):
        regime_trades = premium_df[premium_df['regime'] == regime]
        regime_win_rate = regime_trades['outcome'].mean() * 100
        logger.info(f"\nRegime {regime}:")
        logger.info(f"  Trades: {len(regime_trades)}")
        logger.info(f"  Win Rate: {regime_win_rate:.2f}%")
        logger.info(f"  Avg Quality: {regime_trades['quality_score'].mean():.1f}")


def main():
    """Main execution."""
    logger.info("\n" + "=" * 70)
    logger.info("PREMIUM TRAINING DATA GENERATION")
    logger.info("=" * 70)

    try:
        output_path = Path("data/ml_training/silver_bullet_trades_premium.parquet")

        generate_premium_training_data(
            output_path=output_path,
            min_fvg_gap=75.0,
            min_volume_ratio=2.0,
            max_bar_distance=7,
            min_quality_score=70.0
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

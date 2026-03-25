"""Meta-Labeling Training Data Builder.

This module builds complete training datasets for meta-labeling by combining
label mapping and feature extraction.
"""

import logging
import pandas as pd
from src.ml.label_mapper import map_signals_to_outcomes
from src.ml.signal_feature_extractor import SignalFeatureExtractor

logger = logging.getLogger(__name__)


class MetaLabelingDatasetBuilder:
    """Build complete training datasets for meta-labeling model.

    Combines signal-to-outcome mapping with feature extraction to create
    a training-ready dataset with features and binary labels.
    """

    def __init__(self, feature_extractor: SignalFeatureExtractor = None):
        """Initialize the dataset builder.

        Args:
            feature_extractor: SignalFeatureExtractor instance (creates new if None)
        """
        self.feature_extractor = feature_extractor or SignalFeatureExtractor()
        logger.info("MetaLabelingDatasetBuilder initialized")

    def build_dataset(
        self,
        signals_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Build complete training dataset with features and labels.

        Combines three steps:
        1. Map signals to trade outcomes (binary labels)
        2. Extract features for each signal
        3. Merge features with labels

        Args:
            signals_df: DataFrame with signal metadata
                Index: signal timestamps
                Columns: direction, confidence, mss_detected, fvg_detected, etc.
            trades_df: DataFrame with trade outcomes
                Columns: entry_time, direction, return_pct, exit_reason
            price_data: DataFrame with OHLCV data
                Index: timestamps
                Columns: open, high, low, close, volume
            verbose: Whether to print progress messages

        Returns:
            DataFrame with:
            - Index: signal timestamps
            - Columns: 40+ feature columns + 'label' column
        """
        logger.info("Building meta-labeling training dataset...")

        # Step 1: Map signals to outcomes
        if verbose:
            logger.info("Step 1: Mapping signals to outcomes...")
        labeled_signals = map_signals_to_outcomes(signals_df, trades_df)

        # Step 2: Extract features
        if verbose:
            logger.info("Step 2: Extracting features...")
        features_df = self.feature_extractor.extract_for_all_signals(
            signals_df=signals_df,
            price_data=price_data,
            verbose=verbose
        )

        # Step 3: Merge features with labels
        if verbose:
            logger.info("Step 3: Merging features with labels...")

        # Align indices (handle any timestamp mismatches)
        common_index = labeled_signals.index.intersection(features_df.index)
        if len(common_index) < len(labeled_signals):
            logger.warning(
                f"Index mismatch: {len(labeled_signals)} signals vs "
                f"{len(features_df)} features, using {len(common_index)} common"
            )

        # Use common index
        labeled_aligned = labeled_signals.loc[common_index]
        features_aligned = features_df.loc[common_index]

        # Merge
        dataset = features_aligned.copy()
        dataset['label'] = labeled_aligned['label'].values
        dataset['return_pct'] = labeled_aligned['return_pct'].values

        # Add signal direction if not already in features
        if 'direction' not in dataset.columns and 'direction' in labeled_aligned.columns:
            # Convert direction to numeric (bullish=1, bearish=0)
            dataset['signal_direction'] = (
                labeled_aligned['direction'].map({'bullish': 1, 'bearish': 0}).values
            )

        # Add confidence if not already in features
        if 'confidence' not in dataset.columns and 'confidence' in labeled_aligned.columns:
            dataset['signal_confidence'] = labeled_aligned['confidence'].values

        # Handle missing values
        dataset = self._handle_missing_values(dataset)

        logger.info(
            f"Dataset built: {len(dataset)} samples × {len(dataset.columns)} features"
        )
        logger.info(f"Label distribution:")
        logger.info(f"  Profitable (1): {(dataset['label'] == 1).sum()} ({(dataset['label'] == 1).sum() / len(dataset) * 100:.1f}%)")
        logger.info(f"  Unprofitable (0): {(dataset['label'] == 0).sum()} ({(dataset['label'] == 0).sum() / len(dataset) * 100:.1f}%)")

        return dataset

    def _handle_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.

        - Drops columns with >50% missing values
        - Fills remaining missing values with forward fill then backward fill

        Args:
            dataset: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        initial_cols = len(dataset.columns)

        # Drop columns with >50% missing
        missing_pct = dataset.isnull().sum() / len(dataset)
        cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} columns with >50% missing: {cols_to_drop[:5]}")
            dataset = dataset.drop(columns=cols_to_drop)

        # Fill remaining missing values
        # Forward fill first
        dataset = dataset.ffill()
        # Then backward fill for any remaining NaNs at the start
        dataset = dataset.bfill()

        # If still have NaNs (shouldn't happen), fill with 0
        dataset = dataset.fillna(0)

        if len(cols_to_drop) > 0:
            logger.info(f"Missing value handling: {initial_cols} → {len(dataset.columns)} columns")

        return dataset

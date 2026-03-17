"""Training Data Preparation for ML Meta-Labeling.

This module implements data preparation for XGBoost classifier training,
including label calculation, feature selection, data splitting, and
pipeline orchestration.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.models import DollarBar, SilverBulletSetup

logger = logging.getLogger(__name__)


# ============================================================================
# Label Calculation
# ============================================================================


def calculate_labels(
    setup: SilverBulletSetup,
    future_bars: list[DollarBar],
    time_horizons: list[int],
    target_ticks: int = 20,
    stop_ticks: int = 20,
) -> dict[int, int]:
    """Calculate binary labels for Silver Bullet setup.

    Labels are calculated by simulating forward price movement and checking
    whether target or stop loss is hit first.

    Args:
        setup: Silver Bullet setup to label
        future_bars: Future Dollar Bars after setup timestamp
        time_horizons: List of time horizons in minutes (e.g., [5, 15, 30, 60])
        target_ticks: Target profit in ticks (default: 20)
        stop_ticks: Stop loss in ticks (default: 20)

    Returns:
        Dictionary mapping time_horizon -> label (1 = profitable, 0 = unprofitable)
    """
    labels = {}

    # Convert ticks to points (MNQ: 0.25 points per tick)
    target_points = target_ticks * 0.25
    stop_points = stop_ticks * 0.25

    # Determine entry price (middle of entry zone)
    entry_price = (setup.entry_zone_top + setup.entry_zone_bottom) / 2

    # Calculate target and stop levels
    if setup.direction == "bullish":
        target_level = entry_price + target_points
        stop_level = entry_price - stop_points
    else:  # bearish
        target_level = entry_price - target_points
        stop_level = entry_price + stop_points

    # For each time horizon, calculate label
    for horizon_minutes in time_horizons:
        # Find bars within time horizon
        cutoff_time = setup.timestamp + timedelta(minutes=horizon_minutes)
        horizon_bars = [
            bar for bar in future_bars if setup.timestamp < bar.timestamp <= cutoff_time
        ]

        if not horizon_bars:
            labels[horizon_minutes] = 0
            continue

        # Check if target or stop hit first
        target_hit = False
        stop_hit = False

        for bar in horizon_bars:
            if setup.direction == "bullish":
                if bar.high >= target_level:
                    target_hit = True
                    break
                if bar.low <= stop_level:
                    stop_hit = True
                    break
            else:  # bearish
                if bar.low <= target_level:
                    target_hit = True
                    break
                if bar.high >= stop_level:
                    stop_hit = True
                    break

        # Assign label
        if target_hit and not stop_hit:
            labels[horizon_minutes] = 1  # Profitable
        else:
            labels[horizon_minutes] = 0  # Unprofitable (stop hit or neither)

    logger.debug(f"Calculated labels for setup at {setup.timestamp}: {labels}")
    return labels


# ============================================================================
# Feature Selection
# ============================================================================


def select_features(
    features_df: pd.DataFrame,
    max_correlation: float = 0.9,
    top_k: int = 20,
) -> pd.DataFrame:
    """Select and preprocess features for ML training.

    Performs:
    1. Removes highly correlated features
    2. Selects top K features
    3. Handles missing values (forward-fill, backward-fill)
    4. Standardizes features (z-score normalization)

    Args:
        features_df: DataFrame with engineered features
        max_correlation: Maximum correlation threshold (default: 0.9)
        top_k: Number of top features to select (default: 20)

    Returns:
        DataFrame with selected and preprocessed features
    """
    df = features_df.copy()

    # Separate label columns if present
    label_cols = [
        col
        for col in df.columns
        if col
        in ["label", "time_horizon", "timestamp", "signal_direction", "trading_session"]
    ]
    feature_cols = [col for col in df.columns if col not in label_cols]

    # Keep only numeric columns for correlation analysis
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols

    # Handle missing values
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Remove highly correlated features
    corr_matrix = df[feature_cols].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features to drop
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > max_correlation)
    ]

    logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
    df = df.drop(columns=to_drop)
    feature_cols = [col for col in feature_cols if col not in to_drop]

    # Select top K features (using variance as proxy for importance)
    # In production, this would use actual feature importance from trained model
    feature_variances = df[feature_cols].var().sort_values(ascending=False)
    top_features = feature_variances.head(top_k).index.tolist()

    # Keep only top features and label columns
    selected_cols = top_features + label_cols
    df = df[selected_cols]

    # Standardize features (z-score normalization)
    for col in top_features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

    logger.info(f"Selected {len(top_features)} features: {top_features}")
    return df


# ============================================================================
# Data Splitting
# ============================================================================


def split_data(
    data_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split data into train, validation, and test sets.

    Uses time-based splitting to prevent data leakage.

    Args:
        data_df: DataFrame with features and labels
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        stratify_by: Column to stratify by (e.g., "signal_direction")

    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    # Sort by timestamp to ensure time-based split
    df = data_df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split data
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    # Create metadata
    metadata = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_label_distribution": train["label"].value_counts().to_dict()
        if "label" in train.columns
        else {},
        "val_label_distribution": val["label"].value_counts().to_dict()
        if "label" in val.columns
        else {},
        "test_label_distribution": test["label"].value_counts().to_dict()
        if "label" in test.columns
        else {},
        "train_time_range": (train["timestamp"].min(), train["timestamp"].max()),
        "val_time_range": (val["timestamp"].min(), val["timestamp"].max()),
        "test_time_range": (test["timestamp"].min(), test["timestamp"].max()),
    }

    logger.info(f"Data split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test, metadata


# ============================================================================
# Training Data Pipeline
# ============================================================================


class TrainingDataPipeline:
    """Orchestrates training data preparation for ML meta-labeling.

    Pipeline Flow:
    1. Load Dollar Bars with features
    2. Load Silver Bullet signals
    3. Calculate labels for each signal
    4. Select and preprocess features
    5. Split data into train/val/test sets
    6. Store results in Parquet format

    Performance:
    - < 5 seconds for 6-month dataset
    - Memory efficient (handles 100K+ samples)
    """

    def __init__(self, output_dir: str | Path = "data/processed/training_data"):
        """Initialize training data pipeline.

        Args:
            output_dir: Directory to store processed training data
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TrainingDataPipeline initialized with output: {self._output_dir}")

    def prepare_training_data(
        self,
        features_df: pd.DataFrame,
        setups: list[SilverBulletSetup],
        time_horizons: list[int],
        target_ticks: int = 20,
        stop_ticks: int = 20,
        max_correlation: float = 0.9,
        top_k: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """Prepare training data for ML model.

        Args:
            features_df: DataFrame with engineered features (indexed by timestamp)
            setups: List of Silver Bullet setups
            time_horizons: List of time horizons for labeling
            target_ticks: Target profit in ticks (default: 20)
            stop_ticks: Stop loss in ticks (default: 20)
            max_correlation: Maximum correlation threshold (default: 0.9)
            top_k: Number of top features to select (default: 20)

        Returns:
            Dictionary with train, val, test DataFrames for each time horizon
        """
        logger.info("Starting training data preparation...")

        # Create labeled dataset for each time horizon
        horizon_datasets = {}

        for horizon in time_horizons:
            logger.info(f"Processing {horizon}-minute horizon...")

            # Calculate labels for all setups
            labeled_data = self._label_setups(
                features_df=features_df,
                setups=setups,
                time_horizon=horizon,
                target_ticks=target_ticks,
                stop_ticks=stop_ticks,
            )

            if labeled_data.empty:
                logger.warning(f"No labeled data for {horizon}-minute horizon")
                continue

            # Select and preprocess features
            selected_data = select_features(
                features_df=labeled_data,
                max_correlation=max_correlation,
                top_k=top_k,
            )

            # Split data
            train, val, test, metadata = split_data(selected_data)

            # Store metadata
            metadata["time_horizon"] = horizon
            metadata["target_ticks"] = target_ticks
            metadata["stop_ticks"] = stop_ticks
            metadata["created_at"] = datetime.now().isoformat()

            # Store datasets
            horizon_datasets[horizon] = {
                "train": train,
                "val": val,
                "test": test,
                "metadata": metadata,
            }

            # Save to Parquet
            self._save_horizon_data(horizon, train, val, test, metadata)

            logger.info(
                f"{horizon}-minute horizon: {len(train)} train, "
                f"{len(val)} val, {len(test)} test"
            )

        logger.info(
            f"Training data preparation complete for {len(time_horizons)} horizons"
        )
        return horizon_datasets

    def _label_setups(
        self,
        features_df: pd.DataFrame,
        setups: list[SilverBulletSetup],
        time_horizon: int,
        target_ticks: int,
        stop_ticks: int,
    ) -> pd.DataFrame:
        """Label setups and join with features.

        Args:
            features_df: DataFrame with features (must have 'timestamp' column)
            setups: List of Silver Bullet setups
            time_horizon: Time horizon in minutes
            target_ticks: Target profit in ticks
            stop_ticks: Stop loss in ticks

        Returns:
            DataFrame with features and labels
        """
        labeled_rows = []

        # Pre-sort features by timestamp for faster lookup
        features_df = features_df.sort_values("timestamp").reset_index(drop=True)

        # Convert ticks to points
        target_points = target_ticks * 0.25
        stop_points = stop_ticks * 0.25

        for setup in setups:
            # Determine entry price
            entry_price = (setup.entry_zone_top + setup.entry_zone_bottom) / 2

            # Calculate target and stop levels
            if setup.direction == "bullish":
                target_level = entry_price + target_points
                stop_level = entry_price - stop_points
            else:  # bearish
                target_level = entry_price - target_points
                stop_level = entry_price + stop_points

            # Find cutoff time for this horizon
            cutoff_time = setup.timestamp + timedelta(minutes=time_horizon)

            # Find future bars within horizon
            future_mask = (features_df["timestamp"] > setup.timestamp) & (
                features_df["timestamp"] <= cutoff_time
            )
            future_rows = features_df[future_mask]

            if future_rows.empty:
                # No future data, label as 0
                label = 0
            else:
                # Check if target or stop hit first
                target_hit = False
                stop_hit = False

                for _, row in future_rows.iterrows():
                    if setup.direction == "bullish":
                        if row["high"] >= target_level:
                            target_hit = True
                            break
                        if row["low"] <= stop_level:
                            stop_hit = True
                            break
                    else:  # bearish
                        if row["low"] <= target_level:
                            target_hit = True
                            break
                        if row["high"] >= stop_level:
                            stop_hit = True
                            break

                # Assign label
                label = 1 if target_hit and not stop_hit else 0

            # Get feature row at setup timestamp
            setup_features = features_df[features_df["timestamp"] == setup.timestamp]

            if not setup_features.empty:
                row = setup_features.iloc[0].to_dict()
                row["label"] = label
                row["time_horizon"] = time_horizon
                row["signal_direction"] = setup.direction
                labeled_rows.append(row)

        if not labeled_rows:
            return pd.DataFrame()

        return pd.DataFrame(labeled_rows)

    def _save_horizon_data(
        self,
        horizon: int,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        metadata: dict,
    ) -> None:
        """Save horizon data to Parquet format.

        Args:
            horizon: Time horizon in minutes
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            metadata: Metadata dictionary
        """
        # Create horizon directory
        horizon_dir = self._output_dir / f"{horizon}_minute"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Save datasets
        train.to_parquet(horizon_dir / "train.parquet", index=False)
        val.to_parquet(horizon_dir / "val.parquet", index=False)
        test.to_parquet(horizon_dir / "test.parquet", index=False)

        # Save metadata
        # Convert dict fields to JSON strings for Parquet compatibility
        metadata_copy = metadata.copy()
        for key in [
            "train_label_distribution",
            "val_label_distribution",
            "test_label_distribution",
        ]:
            if key in metadata_copy and isinstance(metadata_copy[key], dict):
                import json

                metadata_copy[key] = json.dumps(metadata_copy[key])

        metadata_df = pd.DataFrame([metadata_copy])
        metadata_df.to_parquet(horizon_dir / "metadata.parquet", index=False)

        logger.info(f"Saved {horizon}-minute horizon data to {horizon_dir}")

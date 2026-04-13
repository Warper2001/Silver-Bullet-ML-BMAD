#!/usr/bin/env python3
"""Train regime-specific XGBoost models.

This script trains separate XGBoost models for each market regime detected
by the HMM, enabling regime-aware predictions.

Usage:
    python scripts/train_regime_specific_models.py

Output:
    - models/xgboost/regime_aware/: Regime-specific models
    - data/reports/regime_model_comparison.md: Performance comparison
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np
import h5py
import yaml

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "regime_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_dollar_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load dollar bar data from HDF5 files."""
    logger.info(f"Loading dollar bars from {start_date} to {end_date}")

    data_dir = Path("data/processed/dollar_bars/")
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    current = start_dt.replace(day=1)
    files = []

    while current <= end_dt:
        filename = f"MNQ_dollar_bars_{current.strftime('%Y%m')}.h5"
        file_path = data_dir / filename
        if file_path.exists():
            files.append(file_path)
        current = current + pd.DateOffset(months=1)

    dataframes = []
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['dollar_bars'][:]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('timestamp').set_index('timestamp')
    combined = combined.loc[
        (combined.index >= start_dt) &
        (combined.index <= end_dt.replace(day=28, hour=23, minute=59))
    ]

    logger.info(f"Loaded {len(combined):,} dollar bars")

    return combined


def load_silver_bullet_labels(start_date: str, end_date: str) -> pd.DataFrame:
    """Load Silver Bullet signal labels for training.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        DataFrame with timestamps and labels
    """
    logger.info(f"Loading Silver Bullet labels from {start_date} to {end_date}")

    # Try to load from existing training data
    signals_path = Path("data/ml_training/silver_bullet_signals.parquet")

    if not signals_path.exists():
        logger.warning("No Silver Bullet signals found. Using synthetic labels for testing.")
        return None

    signals_df = pd.read_parquet(signals_path)

    # Filter by date range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    signals_df = signals_df.loc[
        (signals_df['timestamp'] >= start_dt) &
        (signals_df['timestamp'] <= end_dt)
    ]

    logger.info(f"Loaded {len(signals_df)} Silver Bullet signals")

    return signals_df


def predict_regimes_for_data(
    detector: HMMRegimeDetector,
    data: pd.DataFrame
) -> pd.Series:
    """Predict regimes for dollar bar data.

    Args:
        detector: Trained HMM detector
        data: OHLCV data

    Returns:
        Series of regime labels indexed by timestamp
    """
    logger.info("Predicting regimes for training data...")

    # Engineer features
    feature_engineer = HMMFeatureEngineer()
    features_df = feature_engineer.engineer_features(data)

    # Predict regimes
    regime_labels = detector.predict(features_df)

    # Create series with timestamp index
    regime_series = pd.Series(regime_labels, index=features_df.index)

    # Print distribution
    unique, counts = np.unique(regime_labels, return_counts=True)
    logger.info("Regime distribution:")
    for regime_idx, count in zip(unique, counts):
        regime_name = detector.metadata.regime_names[regime_idx]
        pct = count / len(regime_labels) * 100
        logger.info(f"  {regime_name}: {count:,} bars ({pct:.1f}%)")

    return regime_series


def create_synthetic_training_data(
    data: pd.DataFrame,
    regime_series: pd.Series
) -> pd.DataFrame:
    """Create synthetic training data for testing regime-specific models.

    In production, this would be replaced with actual Silver Bullet labels.
    For now, we create synthetic labels based on price action.

    Args:
        data: OHLCV data
        regime_series: Regime labels

    Returns:
        DataFrame with features and labels
    """
    logger.info("Creating synthetic training data...")

    # Align data with regime labels
    aligned_data = data.loc[regime_series.index]

    # Create synthetic features
    features = pd.DataFrame(index=aligned_data.index)

    # Price-based features
    features['returns_1'] = aligned_data['close'].pct_change(1)
    features['returns_5'] = aligned_data['close'].pct_change(5)
    features['returns_10'] = aligned_data['close'].pct_change(10)

    # Volatility features
    features['volatility_10'] = features['returns_1'].rolling(10).std()
    features['volatility_20'] = features['returns_1'].rolling(20).std()

    # Volume features
    features['volume_z'] = (
        (aligned_data['volume'] - aligned_data['volume'].rolling(20).mean()) /
        aligned_data['volume'].rolling(20).std()
    )

    # ATR
    true_range = pd.concat([
        aligned_data['high'] - aligned_data['low'],
        (aligned_data['high'] - aligned_data['close']).abs(),
        (aligned_data['low'] - aligned_data['close']).abs()
    ], axis=1).max(axis=1)

    features['atr_norm'] = (true_range.rolling(14).mean() / aligned_data['close'])

    # RSI
    delta = aligned_data['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    features['rsi'] = (100 - (100 / (1 + rs)))

    # Momentum
    features['momentum_5'] = aligned_data['close'].diff(5) / aligned_data['close'].shift(5)
    features['momentum_10'] = aligned_data['close'].diff(10) / aligned_data['close'].shift(10)

    # Trend strength
    features['trend_strength'] = (
        features['returns_1']
        .rolling(20)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=False)
    )

    # Fill NaN
    features = features.fillna(0)

    # Create synthetic labels based on future returns
    # In production, these would be actual Silver Bullet trade outcomes
    future_returns = aligned_data['close'].shift(-5) / aligned_data['close'] - 1

    # Label: 1 if future return > 0, else 0
    features['label'] = (future_returns > 0).astype(int)

    # Add regime labels
    features['regime'] = regime_series

    # Remove rows with NaN
    features = features.dropna()

    logger.info(f"Created {len(features)} training samples")

    return features


def train_regime_specific_model(
    train_data: pd.DataFrame,
    regime_name: str,
    regime_idx: int
) -> dict:
    """Train XGBoost model for specific regime.

    Args:
        train_data: Training data with features and labels
        regime_name: Name of regime
        regime_idx: Index of regime

    Returns:
        Training results
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import xgboost as xgb

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training model for regime: {regime_name}")
    logger.info(f"{'=' * 70}")

    # Subset data for this regime
    regime_data = train_data[train_data['regime'] == regime_idx].copy()

    if len(regime_data) < 100:
        logger.warning(f"Insufficient data for {regime_name}: {len(regime_data)} samples")
        return None

    logger.info(f"Regime data: {len(regime_data)} samples")

    # Prepare features and labels
    feature_cols = [col for col in regime_data.columns if col not in ['label', 'regime']]
    X = regime_data[feature_cols]
    y = regime_data['label']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_val, y_pred_proba)
    except:
        roc_auc = 0.5

    logger.info(f"\nMetrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"\nTop 10 Features:")
    for i, (feat, imp) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feat:20s} {imp:.4f}")

    return {
        'model': model,
        'regime_name': regime_name,
        'regime_idx': regime_idx,
        'n_samples': len(regime_data),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'feature_importance': importance,
        'feature_cols': feature_cols
    }


def train_generic_model(train_data: pd.DataFrame) -> dict:
    """Train generic XGBoost model (not regime-aware).

    Args:
        train_data: Training data with features and labels

    Returns:
        Training results
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import xgboost as xgb

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training GENERIC model (baseline)")
    logger.info(f"{'=' * 70}")

    # Prepare features and labels
    feature_cols = [col for col in train_data.columns if col not in ['label', 'regime']]
    X = train_data[feature_cols]
    y = train_data['label']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_val, y_pred_proba)
    except:
        roc_auc = 0.5

    logger.info(f"\nMetrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))

    return {
        'model': model,
        'regime_name': 'generic',
        'regime_idx': -1,
        'n_samples': len(train_data),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'feature_importance': importance,
        'feature_cols': feature_cols
    }


def save_regime_models(regime_models: list, output_dir: Path):
    """Save regime-specific models to disk.

    Args:
        regime_models: List of training results for each regime
        output_dir: Output directory
    """
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)

    for result in regime_models:
        if result is None:
            continue

        regime_name = result['regime_name']
        model_path = output_dir / f"model_{regime_name}.joblib"

        joblib.dump(result['model'], model_path)

        logger.info(f"Saved model to {model_path}")


def generate_comparison_report(
    generic_result: dict,
    regime_results: list[dict],
    output_path: str = "data/reports/regime_model_comparison.md"
):
    """Generate comparison report between generic and regime-specific models.

    Args:
        generic_result: Training results for generic model
        regime_results: Training results for regime-specific models
        output_path: Output file path
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("# Regime-Specific Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("This report compares the performance of a generic XGBoost model vs. ")
        f.write("regime-specific XGBoost models trained on data subset by HMM-detected market regime.\n\n")

        # Generic model performance
        f.write("## Generic Model (Baseline)\n\n")
        f.write(f"- **Samples:** {generic_result['n_samples']:,}\n")
        f.write(f"- **Accuracy:** {generic_result['accuracy']:.4f}\n")
        f.write(f"- **Precision:** {generic_result['precision']:.4f}\n")
        f.write(f"- **Recall:** {generic_result['recall']:.4f}\n")
        f.write(f"- **F1 Score:** {generic_result['f1']:.4f}\n")
        f.write(f"- **ROC-AUC:** {generic_result['roc_auc']:.4f}\n\n")

        # Regime-specific models
        f.write("## Regime-Specific Models\n\n")
        f.write("| Regime | Samples | Accuracy | Precision | Recall | F1 | ROC-AUC | Improvement |\n")
        f.write("|--------|---------|----------|-----------|--------|-----|---------|-------------|\n")

        for result in regime_results:
            if result is None:
                continue

            regime_name = result['regime_name']
            samples = result['n_samples']
            accuracy = result['accuracy']
            precision = result['precision']
            recall = result['recall']
            f1 = result['f1']
            roc_auc = result['roc_auc']

            # Calculate improvement vs generic
            acc_improvement = (accuracy - generic_result['accuracy']) / generic_result['accuracy'] * 100

            f.write(f"| {regime_name} | {samples:,} | {accuracy:.4f} | {precision:.4f} | "
                   f"{recall:.4f} | {f1:.4f} | {roc_auc:.4f} | {acc_improvement:+.1f}% |\n")

        f.write("\n")

        # Top features per regime
        f.write("## Top Features by Regime\n\n")

        for result in regime_results:
            if result is None:
                continue

            regime_name = result['regime_name']
            importance = result['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

            f.write(f"### {regime_name}\n\n")
            for i, (feat, imp) in enumerate(top_features, 1):
                f.write(f"{i}. **{feat}**: {imp:.4f}\n")
            f.write("\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        # Calculate average improvement
        valid_results = [r for r in regime_results if r is not None]
        if valid_results:
            avg_improvement = np.mean([
                (r['accuracy'] - generic_result['accuracy']) / generic_result['accuracy'] * 100
                for r in valid_results
            ])

            f.write(f"**Average Accuracy Improvement:** {avg_improvement:+.1f}%\n\n")

            if avg_improvement > 0:
                f.write("✅ Regime-specific models show **improved performance** over the generic model.\n\n")
            else:
                f.write("⚠️ Regime-specific models show **mixed performance** compared to generic model.\n\n")
                f.write("This may be due to:\n")
                f.write("- Limited training data per regime\n")
                f.write("- Synthetic labels (not actual Silver Bullet signals)\n")
                f.write("- Need for feature engineering specific to each regime\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. Use actual Silver Bullet signal labels instead of synthetic labels\n")
        f.write("2. Increase training data size per regime\n")
        f.write("3. Tune hyperparameters separately for each regime\n")
        f.write("4. Test regime-aware predictions in backtesting\n\n")

    logger.info(f"✅ Comparison report saved to {report_path}")


def main():
    """Main training pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("REGIME-SPECIFIC XGBOOST MODEL TRAINING")
    logger.info("=" * 70)

    try:
        # Step 1: Load HMM model
        logger.info("\n[Step 1/6] Loading HMM model...")
        model_dir = Path("models/hmm/regime_model")

        if not model_dir.exists():
            logger.error(f"HMM model not found: {model_dir}")
            logger.info("Run: python scripts/train_hmm_regime_detector.py")
            return

        detector = HMMRegimeDetector.load(model_dir)
        logger.info(f"✅ HMM model loaded: {detector.n_regimes} regimes")

        # Step 2: Load training data
        logger.info("\n[Step 2/6] Loading training data...")
        train_data = load_dollar_bars("2024-01-01", "2024-12-31")

        # Step 3: Predict regimes for training data
        logger.info("\n[Step 3/6] Predicting regimes for training data...")
        regime_series = predict_regimes_for_data(detector, train_data)

        # Step 4: Create training dataset
        logger.info("\n[Step 4/6] Creating training dataset...")
        training_data = create_synthetic_training_data(train_data, regime_series)

        # Step 5: Train models
        logger.info("\n[Step 5/6] Training models...")

        # Train generic model (baseline)
        generic_result = train_generic_model(training_data)

        # Train regime-specific models
        regime_results = []
        for regime_idx, regime_name in enumerate(detector.metadata.regime_names):
            result = train_regime_specific_model(
                training_data,
                regime_name,
                regime_idx
            )
            regime_results.append(result)

        # Step 6: Save models and generate report
        logger.info("\n[Step 6/6] Saving models and generating report...")

        output_dir = Path("models/xgboost/regime_aware")
        save_regime_models(regime_results, output_dir)

        # Save generic model too
        import joblib
        generic_path = output_dir / "model_generic.joblib"
        joblib.dump(generic_result['model'], generic_path)
        logger.info(f"Saved generic model to {generic_path}")

        # Generate comparison report
        generate_comparison_report(generic_result, regime_results)

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nModels saved to: {output_dir}")
        logger.info(f"Comparison report: data/reports/regime_model_comparison.md")
        logger.info(f"Training logs: logs/regime_model_training.log")

        logger.info("\nNext steps:")
        logger.info("1. Review comparison report")
        logger.info("2. Integrate with MLInference for regime-aware predictions (Story 5.3.3)")
        logger.info("3. Backtest regime-aware strategy")

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

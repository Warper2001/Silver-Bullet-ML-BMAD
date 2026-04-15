#!/usr/bin/env python3
"""
Train Regime-Specific XGBoost Models with Tier 1 Features

This script trains machine learning models that combine:
1. Tier 1 Features (order flow, volatility, microstructure)
2. Triple-Barrier Labels (real trading outcomes)
3. Regime-Specific Training (separate models for each regime)

Expected Performance:
- Win rate: 55-65%
- Sharpe ratio: 2.0+
- Trade frequency: 10-25/day
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Tier1ModelTrainer:
    """Trains regime-specific XGBoost models with Tier 1 features."""

    # Tier 1 feature names
    TIER1_FEATURES = [
        'volume_imbalance_3',
        'volume_imbalance_5',
        'volume_imbalance_10',
        'cumulative_delta_20',
        'cumulative_delta_50',
        'cumulative_delta_100',
        'realized_vol_15',
        'realized_vol_30',
        'realized_vol_60',
        'vwap_deviation_5',
        'vwap_deviation_10',
        'vwap_deviation_20',
        'bid_ask_bounce',
        'noise_adj_momentum_5',
        'noise_adj_momentum_10',
        'noise_adj_momentum_20',
    ]

    def __init__(
        self,
        data_dir: Path,
        model_dir: Path,
        test_size: float = 0.3,
        random_state: int = 42
    ):
        """Initialize model trainer.

        Args:
            data_dir: Directory containing Tier 1 feature data
            model_dir: Directory to save trained models
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.test_size = test_size
        self.random_state = random_state

        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Training results
        self.results = {}

    def load_regime_data(self, regime_id: int) -> Optional[pd.DataFrame]:
        """Load Tier 1 feature data for a specific regime.

        Args:
            regime_id: Regime ID (0, 1, or 2)

        Returns:
            DataFrame with features and labels, or None if file not found
        """
        file_path = self.data_dir / f"regime_{regime_id}_tier1_features.csv"

        if not file_path.exists():
            print(f"WARNING: Data file not found for Regime {regime_id}: {file_path}")
            return None

        df = pd.read_csv(file_path)
        print(f"Loaded Regime {regime_id}: {len(df):,} bars")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training.

        Args:
            df: DataFrame with Tier 1 features and labels

        Returns:
            Tuple of (features, labels)
        """
        # Get available Tier 1 features
        available_features = [f for f in self.TIER1_FEATURES if f in df.columns]

        if not available_features:
            raise ValueError("No Tier 1 features found in DataFrame")

        print(f"Using {len(available_features)} Tier 1 features")

        # Extract features
        X = df[available_features].values

        # Extract labels
        if 'label' not in df.columns:
            raise ValueError("Label column 'label' not found in DataFrame")

        y = df['label'].values

        # Convert to binary classification: 1 (win) vs 0 (not win)
        # -1 (SL hit) → 0 (loss)
        # 0 (time exit) → 0 (neutral)
        # 1 (TP hit) → 1 (win)
        y_binary = (y == 1).astype(int)
        y = y_binary

        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"Valid samples after removing NaN: {len(X):,}")

        return X, y

    def train_regime_model(
        self,
        regime_id: int,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """Train XGBoost model for a specific regime.

        Args:
            regime_id: Regime ID (0, 1, or 2)
            X: Feature matrix
            y: Label vector

        Returns:
            Dictionary with training results
        """
        print(f"\n{'=' * 80}")
        print(f"Training Regime {regime_id} Model")
        print(f"{'=' * 80}")

        # Split data (temporal split - not random!)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

        # Calculate class weights for binary classification
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)

        print(f"\nClass distribution in training set:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {int(cls)} ({'Win' if cls == 1 else 'Loss'}): {count:,} samples ({count/total_samples:.1%})")

        # Calculate scale_pos_weight for binary classification
        # scale_pos_weight = sum(negative instances) / sum(positive instances)
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

        print(f"\nTraining XGBoost model...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics (binary classification)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Precision, recall, F1 for binary classification
        precision = precision_score(y_test, y_pred_test, average='binary', pos_label=1)
        recall = recall_score(y_test, y_pred_test, average='binary', pos_label=1)
        f1 = f1_score(y_test, y_pred_test, average='binary', pos_label=1)

        # Feature importance
        feature_importance = dict(zip(
            [f for f in self.TIER1_FEATURES if f in pd.DataFrame(X).columns],
            model.feature_importances_
        ))

        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        # Results
        results = {
            'regime_id': regime_id,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'feature_importance': feature_importance,
            'model': model
        }

        # Print results
        print(f"\n{'=' * 80}")
        print(f"TRAINING RESULTS - Regime {regime_id}")
        print(f"{'=' * 80}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision (Class 1): {precision:.4f}")
        print(f"Recall (Class 1): {recall:.4f}")
        print(f"F1 Score (Class 1): {f1:.4f}")

        print(f"\n=== Top 10 Feature Importance ===")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
            print(f"{i:2d}. {feature:<30} {importance:.4f}")

        return results

    def save_model(self, results: Dict, regime_id: int) -> Path:
        """Save trained model and metadata.

        Args:
            results: Training results dictionary
            regime_id: Regime ID

        Returns:
            Path to saved model file
        """
        # Save model
        model_file = self.model_dir / f"xgboost_regime_{regime_id}_tier1.joblib"
        joblib.dump(results['model'], model_file)
        print(f"Saved model to: {model_file}")

        # Save metadata
        metadata = {
            'regime_id': regime_id,
            'model_type': 'XGBoost',
            'features': 'Tier 1 (order flow, volatility, microstructure)',
            'feature_count': len([f for f in self.TIER1_FEATURES if f in results['feature_importance']]),
            'training_date': datetime.now().isoformat(),
            'performance': {
                'train_accuracy': results['train_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
            },
            'feature_importance': results['feature_importance'],
        }

        metadata_file = self.model_dir / f"xgboost_regime_{regime_id}_tier1_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_file}")

        return model_file

    def train_all_regimes(self) -> Dict[int, Dict]:
        """Train models for all regimes.

        Returns:
            Dictionary mapping regime_id to training results
        """
        print("=" * 80)
        print("TRAINING REGIME-SPECIFIC MODELS WITH TIER 1 FEATURES")
        print("=" * 80)

        all_results = {}

        for regime_id in [0, 1, 2]:
            # Load data
            df = self.load_regime_data(regime_id)

            if df is None:
                continue

            # Prepare features
            X, y = self.prepare_features(df)

            if len(X) == 0:
                print(f"WARNING: No valid data for Regime {regime_id}")
                continue

            # Train model
            results = self.train_regime_model(regime_id, X, y)

            # Save model
            self.save_model(results, regime_id)

            all_results[regime_id] = results

        return all_results


def main():
    """Main execution function."""
    print("=" * 80)
    print("Tier 1 Model Training")
    print("=" * 80)

    # Configuration
    DATA_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_tier1_features"
    MODEL_DIR = project_root / "models" / "xgboost" / "regime_aware_tier1"

    # Create trainer
    trainer = Tier1ModelTrainer(
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        test_size=0.3,
        random_state=42
    )

    # Train all regimes
    all_results = trainer.train_all_regimes()

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    for regime_id, results in all_results.items():
        print(f"\nRegime {regime_id}:")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")

    print(f"\n✅ Models saved to: {MODEL_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

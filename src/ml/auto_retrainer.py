"""Automated Model Retraining System.

This module monitors ML model performance and automatically retrains
models when performance degrades below acceptable thresholds.
"""

import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """Automated model retraining with performance monitoring.

    Monitors model performance and triggers retraining when:
    1. Win rate drops below threshold
    2. Model is too old (> 90 days)
    3. Prediction confidence is low
    """

    def __init__(
        self,
        model_dir: str = "models/xgboost/30_minute",
        degradation_threshold: float = 0.10,
        max_model_age_days: int = 90,
        min_probability: float = 0.60,
    ):
        """Initialize auto-retrainer.

        Args:
            model_dir: Directory containing model files
            degradation_threshold: Performance drop threshold (10%)
            max_model_age_days: Maximum age before forced retraining
            min_probability: Minimum average prediction confidence
        """
        self.model_dir = Path(model_dir)
        self.degradation_threshold = degradation_threshold
        self.max_model_age_days = max_model_age_days
        self.min_probability = min_probability

        logger.info(f"AutoRetrainer initialized: {model_dir}")

    def check_retraining_needed(self) -> Dict:
        """Check if model needs retraining.

        Returns:
            Dictionary with:
                - retrain (bool): Whether retraining is needed
                - reason (str): Explanation
                - current_metrics (dict): Current performance
                - baseline_metrics (dict): Baseline performance
                - gap (float): Performance difference
        """
        logger.info("Checking if retraining is needed...")

        # Load model metadata
        metadata_path = self.model_dir / "sb_params_optimized.json"
        if not metadata_path.exists():
            return {
                'retrain': True,
                'reason': 'No trained model found',
                'current_metrics': {},
                'baseline_metrics': {},
                'gap': 1.0,
            }

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Get training metrics
        baseline_win_rate = metadata.get('win_rate', 0.85)
        training_date = metadata.get('optimization_date', '')

        # Check model age
        if training_date:
            train_datetime = datetime.strptime(training_date, '%Y-%m-%d')
            age_days = (datetime.now() - train_datetime).days

            if age_days > self.max_model_age_days:
                return {
                    'retrain': True,
                    'reason': f'Model is {age_days} days old (> {self.max_model_age_days} days)',
                    'current_metrics': {},
                    'baseline_metrics': {'win_rate': baseline_win_rate},
                    'gap': 0.0,
                }

        # Load walk-forward results for realistic baseline
        wf_path = Path("models/xgboost/walk_forward_results.json")
        if wf_path.exists():
            with open(wf_path, 'r') as f:
                wf_results = json.load(f)
            realistic_win_rate = wf_results.get('realistic_win_rate', 0.45)
        else:
            realistic_win_rate = 0.50  # Conservative default

        # Get current metrics (placeholder - would load from live predictions)
        current_metrics = self._get_current_metrics()
        current_win_rate = current_metrics.get('win_rate', realistic_win_rate)

        # Calculate performance gap
        gap = realistic_win_rate - current_win_rate

        # Check if degradation exceeds threshold
        if gap > self.degradation_threshold:
            return {
                'retrain': True,
                'reason': f'Performance dropped {gap:.1%} below baseline',
                'current_metrics': current_metrics,
                'baseline_metrics': {'win_rate': realistic_win_rate},
                'gap': gap,
            }

        # Check prediction confidence
        avg_prob = current_metrics.get('avg_probability', 0.70)
        if avg_prob < self.min_probability:
            return {
                'retrain': True,
                'reason': f'Low prediction confidence: {avg_prob:.2%} < {self.min_probability:.2%}',
                'current_metrics': current_metrics,
                'baseline_metrics': {'win_rate': realistic_win_rate},
                'gap': gap,
            }

        # No retraining needed
        return {
            'retrain': False,
            'reason': 'Performance within acceptable range',
            'current_metrics': current_metrics,
            'baseline_metrics': {'win_rate': realistic_win_rate},
            'gap': gap,
        }

    def retrain_model(self) -> Optional[Dict]:
        """Retrain the model with recent data.

        Returns:
            Dictionary with retraining results or error
        """
        logger.info("Starting model retraining...")

        try:
            # 1. Backup current model
            backup_path = self._backup_current_model()
            logger.info(f"Current model backed up to: {backup_path}")

            # 2. Load recent training data
            train_data = self._load_recent_training_data()
            logger.info(f"Loaded {len(train_data)} training samples")

            # 3. Train new model with regularization
            new_model = self._train_regularized_model(train_data)
            logger.info("New model trained")

            # 4. Validate with walk-forward
            validation_results = self._validate_new_model(new_model, train_data)

            if not validation_results['acceptable']:
                # Rollback
                logger.warning("New model failed validation, rolling back...")
                self._rollback_model(backup_path)
                return {
                    'success': False,
                    'error': 'Model failed validation',
                    'validation_results': validation_results,
                }

            # 5. Deploy new model
            self._deploy_new_model(
                new_model,
                validation_results['metrics']
            )
            logger.info("New model deployed")

            return {
                'success': True,
                'metrics': validation_results['metrics'],
                'backup_path': str(backup_path),
            }

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }

    def schedule_retraining(self, interval: str = 'weekly') -> None:
        """Schedule automated retraining checks.

        Args:
            interval: Check interval ('daily' or 'weekly')
        """
        interval_seconds = 86400 if interval == 'daily' else 604800  # 1 day or 1 week

        logger.info(f"Scheduling {interval} retraining checks...")

        while True:
            try:
                logger.info(f"Running scheduled {interval} check...")

                # Check if retraining needed
                needs_retrain = self.check_retraining_needed()

                if needs_retrain['retrain']:
                    logger.warning(f"Retraining triggered: {needs_retrain['reason']}")
                    result = self.retrain_model()

                    if result['success']:
                        logger.info("✅ Scheduled retraining successful")
                    else:
                        logger.error(f"❌ Scheduled retraining failed: {result.get('error')}")

                # Wait until next check
                logger.info(f"Next check in {interval} seconds...")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Scheduled retraining stopped")
                break
            except Exception as e:
                logger.error(f"Error in scheduled check: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying

    def get_model_status(self) -> Dict:
        """Get current model status.

        Returns:
            Dictionary with model status information
        """
        metadata_path = self.model_dir / "sb_params_optimized.json"

        if not metadata_path.exists():
            return {
                'last_trained': 'Never',
                'training_period': 'Unknown',
                'current_win_rate': 'Unknown',
                'expected_win_rate': 'Unknown',
            }

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load walk-forward results
        wf_path = Path("models/xgboost/walk_forward_results.json")
        if wf_path.exists():
            with open(wf_path, 'r') as f:
                wf_results = json.load(f)
            expected_win_rate = wf_results.get('realistic_win_rate', 0.45)
        else:
            expected_win_rate = 0.50

        return {
            'last_trained': metadata.get('optimization_date', 'Unknown'),
            'training_period': f"{metadata.get('data_range', {}).get('start', 'Unknown')} to {metadata.get('data_range', {}).get('end', 'Unknown')}",
            'current_win_rate': f"{metadata.get('win_rate', 0):.2%}",
            'expected_win_rate': f"{expected_win_rate:.2%}",
        }

    def _get_current_metrics(self) -> Dict:
        """Get current model performance metrics.

        Returns:
            Dictionary with current metrics
        """
        # Placeholder - in production, this would load from live predictions
        # For now, return mock metrics
        return {
            'win_rate': 0.48,
            'precision': 0.52,
            'avg_probability': 0.62,
            'n_predictions': 150,
        }

    def _backup_current_model(self) -> Path:
        """Backup current model before retraining.

        Returns:
            Path to backup location
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.model_dir / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        for file in self.model_dir.glob("*.pkl"):
            shutil.copy2(file, backup_dir / file.name)

        for file in self.model_dir.glob("*.json"):
            shutil.copy2(file, backup_dir / file.name)

        return backup_dir

    def _load_recent_training_data(self) -> pd.DataFrame:
        """Load recent training data.

        Returns:
            DataFrame with features and target
        """
        # Placeholder - in production, load from recent backtests
        # For now, create mock data
        n_samples = 1000

        return pd.DataFrame({
            'atr_pct': np.random.beta(2, 5, n_samples),
            'rsi': np.random.uniform(30, 70, n_samples),
            'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 5, n_samples),
            'direction': np.random.randint(0, 2, n_samples),
            'confidence': np.random.uniform(60, 80, n_samples),
            'success': np.random.randint(0, 2, n_samples),
        })

    def _train_regularized_model(self, train_data: pd.DataFrame) -> xgb.XGBClassifier:
        """Train model with regularization to prevent overfitting.

        Args:
            train_data: Training data

        Returns:
            Trained XGBoost model
        """
        feature_cols = [col for col in train_data.columns if col != 'success']
        X_train = train_data[feature_cols]
        y_train = train_data['success']

        # Model with regularization
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,  # Reduced depth
            learning_rate=0.05,  # Slower learning
            min_child_weight=3,  # Higher minimum
            subsample=0.8,  # Sample rows
            colsample_bytree=0.8,  # Sample columns
            reg_lambda=1.0,  # L2 regularization
            reg_alpha=0.1,  # L1 regularization
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        )

        model.fit(X_train, y_train, verbose=False)

        return model

    def _validate_new_model(
        self, model: xgb.XGBClassifier, train_data: pd.DataFrame
    ) -> Dict:
        """Validate new model with cross-validation.

        Args:
            model: Trained model
            train_data: Training data

        Returns:
            Validation results
        """
        from sklearn.model_selection import cross_val_score

        feature_cols = [col for col in train_data.columns if col != 'success']
        X = train_data[feature_cols]
        y = train_data['success']

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        mean_accuracy = cv_scores.mean()
        std_accuracy = cv_scores.std()

        # Check if acceptable
        acceptable = mean_accuracy >= 0.45  # Minimum acceptable win rate

        return {
            'acceptable': acceptable,
            'metrics': {
                'win_rate': mean_accuracy,
                'win_rate_std': std_accuracy,
                'cv_scores': cv_scores.tolist(),
            }
        }

    def _deploy_new_model(
        self, model: xgb.XGBClassifier, metrics: Dict
    ) -> None:
        """Deploy new model to production.

        Args:
            model: Trained model
            metrics: Performance metrics
        """
        # Save model
        model_path = self.model_dir / "xgboost_model.pkl"
        joblib.dump(model, model_path)

        # Update metadata
        metadata_path = self.model_dir / "sb_params_optimized.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update with new metrics
        metadata.update({
            'optimization_date': datetime.now().strftime('%Y-%m-%d'),
            'win_rate': metrics['win_rate'],
            'data_range': {
                'start': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                'end': datetime.now().strftime('%Y-%m-%d'),
            },
            'retraining_count': metadata.get('retraining_count', 0) + 1,
        })

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model deployed to: {model_path}")

    def _rollback_model(self, backup_path: Path) -> None:
        """Rollback to previous model.

        Args:
            backup_path: Path to backup directory
        """
        # Restore from backup
        for file in backup_path.glob("*.pkl"):
            shutil.copy2(file, self.model_dir / file.name)

        for file in backup_path.glob("*.json"):
            shutil.copy2(file, self.model_dir / file.name)

        logger.info(f"Model rolled back from: {backup_path}")

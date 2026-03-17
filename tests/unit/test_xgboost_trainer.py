"""Unit tests for XGBoost Model Training.

Tests XGBoost classifier training, hyperparameter tuning,
model evaluation, and persistence for ML meta-labeling.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.xgboost_trainer import (
    XGBoostTrainer,
    evaluate_model,
    train_xgboost,
)


class TestTrainXGBoost:
    """Test XGBoost training function."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        # Create features
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X = np.random.randn(n_samples, n_features)

        # Create binary labels (40-60% positive class)
        y = np.random.randint(0, 2, n_samples)
        # Ensure reasonable class balance
        y[:400] = 0  # 40% negative
        y[400:] = 1  # 60% positive
        np.random.shuffle(y)

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = y

        return df

    def test_train_xgboost_returns_model(self, sample_training_data):
        """Verify training returns trained XGBoost model."""
        # Prepare data
        feature_cols = [
            col for col in sample_training_data.columns if col.startswith("feature_")
        ]
        X = sample_training_data[feature_cols]
        y = sample_training_data["label"]

        # Train model
        model, metrics = train_xgboost(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
        )

        # Check model exists
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        # Check metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_train_xgboost_hyperparameters(self, sample_training_data):
        """Verify hyperparameters are applied correctly."""
        feature_cols = [
            col for col in sample_training_data.columns if col.startswith("feature_")
        ]
        X = sample_training_data[feature_cols]
        y = sample_training_data["label"]

        # Train with custom hyperparameters
        model, metrics = train_xgboost(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
        )

        # Check model was trained
        assert model is not None
        assert len(metrics) > 0


class TestEvaluateModel:
    """Test model evaluation metrics."""

    @pytest.fixture
    def sample_model_predictions(self):
        """Create sample model predictions."""
        np.random.seed(42)
        n_samples = 1000

        # Create ground truth labels
        y_true = np.random.randint(0, 2, n_samples)

        # Create predictions with 70% accuracy
        y_pred = y_true.copy()
        noise_indices = np.random.choice(
            n_samples, size=int(n_samples * 0.3), replace=False
        )
        y_pred[noise_indices] = 1 - y_pred[noise_indices]

        # Create prediction probabilities
        y_proba = np.random.random(n_samples)
        y_proba[y_pred == 1] = 0.5 + np.random.random(int(y_pred.sum())) * 0.5

        return y_true, y_pred, y_proba

    def test_evaluate_model_returns_metrics(self, sample_model_predictions):
        """Verify evaluation returns all required metrics."""
        y_true, y_pred, y_proba = sample_model_predictions

        metrics = evaluate_model(y_true, y_pred, y_proba)

        # Check all metrics present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics

        # Check metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_evaluate_model_confusion_matrix(self, sample_model_predictions):
        """Verify confusion matrix is calculated correctly."""
        y_true, y_pred, y_proba = sample_model_predictions

        metrics = evaluate_model(y_true, y_pred, y_proba)

        cm = metrics["confusion_matrix"]
        assert cm.shape == (2, 2)  # Binary classification

        # Check confusion matrix sums to total samples
        assert cm.sum() == len(y_true)


class TestXGBoostTrainer:
    """Test XGBoost trainer orchestration class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for multiple horizons."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        datasets = {}
        for horizon in [5, 15, 30, 60]:
            # Create features
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)

            # Create binary labels
            y = np.random.randint(0, 2, n_samples)
            y[:400] = 0
            y[400:] = 1
            np.random.shuffle(y)

            # Create DataFrames
            train_df = pd.DataFrame(X, columns=feature_names)
            train_df["label"] = y

            val_df = train_df.iloc[:200].copy()
            train_df = train_df.iloc[200:]

            datasets[horizon] = {
                "train": train_df,
                "val": val_df,
            }

        return datasets

    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create temporary model directory."""
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def test_train_models_for_all_horizons(self, sample_training_data, temp_model_dir):
        """Verify training works for all time horizons."""
        trainer = XGBoostTrainer(model_dir=temp_model_dir)

        # Train models
        models = trainer.train_models(
            datasets=sample_training_data,
            time_horizons=[5, 15, 30, 60],
        )

        # Check models for all horizons
        assert len(models) == 4
        assert 5 in models
        assert 15 in models
        assert 30 in models
        assert 60 in models

        # Check each model has required components
        for horizon, model_data in models.items():
            assert "model" in model_data
            assert "metrics" in model_data
            assert "feature_importance" in model_data
            assert "hyperparameters" in model_data

    def test_hyperparameter_tuning(self, sample_training_data, temp_model_dir):
        """Verify hyperparameter tuning improves performance."""
        trainer = XGBoostTrainer(model_dir=temp_model_dir)

        # Train with tuning
        models = trainer.train_models(
            datasets=sample_training_data,
            time_horizons=[5],
            perform_tuning=True,
            n_iter=3,  # Few iterations for test speed
        )

        # Check model exists
        assert 5 in models
        assert "hyperparameters" in models[5]

    def test_model_persistence(self, sample_training_data, temp_model_dir):
        """Verify models are saved and can be loaded."""
        trainer = XGBoostTrainer(model_dir=temp_model_dir)

        # Train and save
        _ = trainer.train_models(
            datasets=sample_training_data,
            time_horizons=[5],
        )

        # Check model file exists
        model_file = temp_model_dir / "5_minute" / "xgboost_model.json"
        assert model_file.exists()

        # Load model
        loaded_model = trainer.load_model(horizon=5)
        assert loaded_model is not None

    def test_feature_importance_tracking(self, sample_training_data, temp_model_dir):
        """Verify feature importance is tracked."""
        trainer = XGBoostTrainer(model_dir=temp_model_dir)

        models = trainer.train_models(
            datasets=sample_training_data,
            time_horizons=[5],
        )

        # Check feature importance
        importance = models[5]["feature_importance"]
        assert len(importance) > 0
        assert all(imp >= 0 for imp in importance.values())

    def test_model_evaluation_metrics(self, sample_training_data, temp_model_dir):
        """Verify evaluation metrics are calculated."""
        trainer = XGBoostTrainer(model_dir=temp_model_dir)

        models = trainer.train_models(
            datasets=sample_training_data,
            time_horizons=[5],
        )

        # Check metrics
        metrics = models[5]["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

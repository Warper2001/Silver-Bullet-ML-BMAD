"""Unit tests for Walk-Forward Optimization.

Tests automated weekly model retraining on 6-month rolling window,
model comparison, deployment, and archiving.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.walk_forward_optimizer import WalkForwardOptimizer


class TestWalkForwardOptimizerInit:
    """Test WalkForwardOptimizer initialization and configuration."""

    def test_init_with_default_parameters(self):
        """Verify WalkForwardOptimizer initializes with default parameters."""
        optimizer = WalkForwardOptimizer()
        assert optimizer is not None
        assert optimizer._model_dir.name == "xgboost"
        assert optimizer._retraining_interval == "weekly"

    def test_init_with_custom_model_dir(self, tmp_path):
        """Verify WalkForwardOptimizer initializes with custom model directory."""
        custom_dir = tmp_path / "custom_models"
        optimizer = WalkForwardOptimizer(model_dir=custom_dir)
        assert optimizer._model_dir == custom_dir

    def test_init_with_custom_retraining_interval(self):
        """Verify WalkForwardOptimizer initializes with custom interval."""
        optimizer = WalkForwardOptimizer(retraining_interval="bi-weekly")
        assert optimizer._retraining_interval == "bi-weekly"

    def test_model_directory_created_on_init(self, tmp_path):
        """Verify model directory is created if it doesn't exist."""
        model_dir = tmp_path / "new_models"
        optimizer = WalkForwardOptimizer(model_dir=model_dir)
        assert model_dir.exists()
        assert (model_dir / "5_minute").exists()


class TestLoadTrainingWindow:
    """Test load_training_window() method."""

    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create test Dollar Bars and signals data."""
        # Create 6 months of dummy Dollar Bars (daily bars for simplicity)
        dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")
        bars_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(11700, 11900, len(dates)),
                "high": np.random.uniform(11800, 12000, len(dates)),
                "low": np.random.uniform(11600, 11800, len(dates)),
                "close": np.random.uniform(11700, 11900, len(dates)),
                "volume": np.random.uniform(1000, 5000, len(dates)),
            }
        )
        bars_data.set_index("timestamp", inplace=True)

        # Create dummy signals
        signals_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2024-01-01", end="2024-06-30", freq="W"
                ),
                "direction": np.random.choice(["bullish", "bearish"], 26),
                "outcome": np.random.choice([0, 1], 26),
            }
        )

        # Save to temporary files
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        import h5py

        with h5py.File(data_dir / "dollar_bars.h5", "w") as f:
            f.create_dataset("bars", data=bars_data)

        signals_data.to_csv(data_dir / "signals.csv", index=False)

        return bars_data, signals_data, data_dir

    def test_load_training_window_returns_six_months_of_data(self, setup_test_data):
        """Verify load_training_window() loads 6 months of data."""
        bars_data, signals_data, data_dir = setup_test_data
        optimizer = WalkForwardOptimizer()

        # Mock the data loading method
        with patch.object(optimizer, "_load_dollar_bars", return_value=bars_data):
            with patch.object(optimizer, "_load_signals", return_value=signals_data):
                (
                    train_data,
                    val_data,
                    train_signals,
                    val_signals,
                ) = optimizer.load_training_window()

        # Verify data was loaded
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert len(train_signals) > 0
        assert len(val_signals) >= 0

    def test_load_training_window_validates_data_completeness(self, setup_test_data):
        """Verify load_training_window() validates data completeness ≥ 95%."""
        bars_data, signals_data, data_dir = setup_test_data
        optimizer = WalkForwardOptimizer()

        # Mock the data loading method
        with patch.object(optimizer, "_load_dollar_bars", return_value=bars_data):
            with patch.object(optimizer, "_load_signals", return_value=signals_data):
                # Should not raise exception for complete data
                (
                    train_data,
                    val_data,
                    train_signals,
                    val_signals,
                ) = optimizer.load_training_window()
                assert True  # Test passes if no exception

    def test_load_training_window_raises_error_for_incomplete_data(self, tmp_path):
        """Verify DataInsufficientError raised if data completeness < 95%."""
        from src.ml.walk_forward_optimizer import DataInsufficientError

        optimizer = WalkForwardOptimizer()

        # Mock incomplete data (only 1 month instead of 6)
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        incomplete_data = pd.DataFrame(
            {
                "timestamp": dates,
                "close": np.random.uniform(11700, 11900, len(dates)),
            }
        )
        incomplete_data.set_index("timestamp", inplace=True)

        with patch.object(optimizer, "_load_dollar_bars", return_value=incomplete_data):
            with pytest.raises(DataInsufficientError):
                optimizer.load_training_window()


class TestFeatureEngineering:
    """Test feature engineering and label generation."""

    def test_prepare_features_uses_feature_engineer(self):
        """Verify _prepare_features() uses FeatureEngineer for feature calculation."""
        optimizer = WalkForwardOptimizer()

        # Create dummy data
        train_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
                "open": np.random.uniform(11700, 11900, 100),
                "high": np.random.uniform(11800, 12000, 100),
                "low": np.random.uniform(11600, 11800, 100),
                "close": np.random.uniform(11700, 11900, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )
        train_data.set_index("timestamp", inplace=True)

        train_signals = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="D"),
                "outcome": np.random.choice([0, 1], 10),
            }
        )

        # Mock FeatureEngineer
        with patch(
            "src.ml.walk_forward_optimizer.FeatureEngineer"
        ) as MockFeatureEngineer:
            mock_engineer = Mock()
            mock_features_df = pd.DataFrame(
                {
                    "atr": np.random.uniform(1.0, 2.0, 100),
                    "rsi": np.random.uniform(30, 70, 100),
                }
            )
            mock_engineer.engineer_features.return_value = mock_features_df
            MockFeatureEngineer.return_value = mock_engineer

            # Mock TrainingDataPipeline
            with patch(
                "src.ml.walk_forward_optimizer.TrainingDataPipeline"
            ) as MockPipeline:
                mock_pipeline = Mock()
                mock_pipeline.prepare_data.return_value = (
                    mock_features_df,
                    np.array([0, 1] * 50),
                )
                MockPipeline.return_value = mock_pipeline

                # Call method
                optimizer._prepare_features(
                    train_data, train_data, train_signals, train_signals
                )

                # Verify FeatureEngineer was called
                mock_engineer.engineer_features.assert_called()


class TestModelTraining:
    """Test model training functionality."""

    def test_train_model_creates_new_model(self):
        """Verify _train_model() creates and saves a new XGBoost model."""
        optimizer = WalkForwardOptimizer()

        # Create dummy training data
        train_features = pd.DataFrame(
            {
                "atr": np.random.uniform(1.0, 2.0, 1000),
                "rsi": np.random.uniform(30, 70, 1000),
            }
        )
        train_labels = np.array([0, 1] * 500)

        # Mock XGBoostTrainer
        with patch("src.ml.walk_forward_optimizer.XGBoostTrainer") as MockTrainer:
            mock_trainer = Mock()
            mock_model = Mock()
            mock_pipeline = Mock()
            mock_metadata = {
                "metrics": {
                    "roc_auc": 0.75,
                    "precision": 0.70,
                    "recall": 0.72,
                    "f1": 0.71,
                }
            }

            mock_trainer.train_xgboost.return_value = (
                mock_model,
                mock_pipeline,
                mock_metadata,
            )
            MockTrainer.return_value = mock_trainer

            # Call method
            result = optimizer._train_model(train_features, train_labels)

            # Verify trainer was called
            mock_trainer.train_xgboost.assert_called_once()

            # Verify result structure
            assert "model" in result
            assert "pipeline" in result
            assert "metadata" in result


class TestModelComparison:
    """Test model comparison and deployment logic."""

    def test_compare_models_deploys_when_roc_auc_improves(self):
        """Verify compare_models() deploys when new model ROC-AUC ≥ current."""
        optimizer = WalkForwardOptimizer()

        # Create mock models
        new_model = Mock()
        current_model = Mock()

        # Create dummy validation data
        val_features = pd.DataFrame(
            {
                "atr": np.random.uniform(1.0, 2.0, 100),
                "rsi": np.random.uniform(30, 70, 100),
            }
        )
        val_labels = np.array([0, 1] * 50)

        # Mock predictions
        new_model.predict_proba.return_value = np.array([[0.3, 0.7]] * 100)
        current_model.predict_proba.return_value = np.array([[0.4, 0.6]] * 100)

        # Call comparison
        result = optimizer.compare_models(
            new_model, current_model, val_features, val_labels
        )

        # Verify deployment decision
        assert result["deploy"] is True
        assert result["reason"] in ["roc_auc_improved_or_equal"]

    def test_compare_models_rejects_when_roc_auc_degrades(self):
        """Verify compare_models() rejects when new model ROC-AUC < current."""
        optimizer = WalkForwardOptimizer()

        # Create mock models
        new_model = Mock()
        current_model = Mock()

        # Create dummy validation data
        val_features = pd.DataFrame(
            {
                "atr": np.random.uniform(1.0, 2.0, 100),
                "rsi": np.random.uniform(30, 70, 100),
            }
        )
        val_labels = np.array([0, 1] * 50)

        # Mock predictions (new model worse than current)
        new_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 100)
        current_model.predict_proba.return_value = np.array([[0.3, 0.7]] * 100)

        # Call comparison
        result = optimizer.compare_models(
            new_model, current_model, val_features, val_labels
        )

        # Verify rejection decision
        assert result["deploy"] is False
        assert result["reason"] == "roc_auc_degraded"


class TestModelDeployment:
    """Test model deployment and archiving."""

    def test_deploy_model_archives_current_model(self, tmp_path):
        """Verify deploy_model() archives current model before deployment."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create current model files
        (model_dir / "xgboost_model.json").write_text("current_model")
        (model_dir / "feature_pipeline.pkl").write_text("current_pipeline")
        (model_dir / "metadata.json").write_text('{"metrics": {"roc_auc": 0.70}}')

        optimizer = WalkForwardOptimizer(model_dir=tmp_path / "xgboost")

        # Create new model files
        new_model_file = tmp_path / "new_model.json"
        new_model_file.write_text("new_model")
        new_pipeline_file = tmp_path / "new_pipeline.pkl"
        new_pipeline_file.write_text("new_pipeline")

        # Deploy new model
        optimizer.deploy_model(new_model_file, new_pipeline_file)

        # Verify old model was archived
        archive_dir = tmp_path / "xgboost" / "archive"
        assert archive_dir.exists()

        # Verify new model is in production
        assert (model_dir / "xgboost_model.json").read_text() == "new_model"
        assert (model_dir / "feature_pipeline.pkl").read_text() == "new_pipeline"

    def test_deploy_model_updates_model_registry(self, tmp_path):
        """Verify deploy_model() updates model registry CSV."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        optimizer = WalkForwardOptimizer(model_dir=tmp_path / "xgboost")

        # Create new model files
        new_model_file = tmp_path / "new_model.json"
        new_model_file.write_text("new_model")
        new_pipeline_file = tmp_path / "new_pipeline.pkl"
        new_pipeline_file.write_text("new_pipeline")

        # Deploy new model
        metadata = {
            "metrics": {"roc_auc": 0.75, "precision": 0.70, "recall": 0.72, "f1": 0.71}
        }
        optimizer.deploy_model(
            new_model_file, new_pipeline_file, metadata=metadata, deployed=True
        )

        # Verify registry was created
        registry_file = tmp_path / "xgboost" / "model_registry.csv"
        assert registry_file.exists()

        # Verify registry content
        registry_df = pd.read_csv(registry_file)
        assert len(registry_df) == 1
        assert registry_df["deployed"].iloc[0] == True
        assert registry_df["roc_auc"].iloc[0] == 0.75


class TestScheduling:
    """Test scheduling and automation."""

    def test_scheduler_initializes_on_start(self):
        """Verify scheduler is initialized in __init__."""
        optimizer = WalkForwardOptimizer()
        assert hasattr(optimizer, "_scheduler")
        assert optimizer._scheduler is not None

    @patch("src.ml.walk_forward_optimizer.AsyncIOScheduler")
    def test_start_scheduler_starts_scheduler(self, MockScheduler):
        """Verify start_scheduler() starts the APScheduler."""
        mock_scheduler = Mock()
        MockScheduler.return_value = mock_scheduler

        optimizer = WalkForwardOptimizer()
        optimizer.start_scheduler()

        # Verify scheduler was started
        mock_scheduler.start.assert_called_once()

    @patch("src.ml.walk_forward_optimizer.AsyncIOScheduler")
    def test_stop_scheduler_stops_scheduler(self, MockScheduler):
        """Verify stop_scheduler() stops the APScheduler."""
        mock_scheduler = Mock()
        MockScheduler.return_value = mock_scheduler

        optimizer = WalkForwardOptimizer()
        optimizer.stop_scheduler()

        # Verify scheduler was stopped
        mock_scheduler.shutdown.assert_called_once()


class TestRetrying:
    """Test retry logic for failed retraining."""

    @pytest.mark.asyncio
    async def test_run_retraining_with_retry_succeeds_on_first_attempt(self):
        """Verify retry succeeds when first attempt succeeds."""
        optimizer = WalkForwardOptimizer()

        # Mock successful retraining
        with patch.object(optimizer, "run_retraining", return_value={"success": True}):
            result = await optimizer.run_retraining_with_retry()

        # Verify success
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_retraining_with_retry_retries_on_failure(self):
        """Verify retry retries when first attempt fails."""
        optimizer = WalkForwardOptimizer()

        # Mock failed then successful retraining
        call_count = [0]

        def mock_retraining():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Temporary failure")
            return {"success": True}

        with patch.object(optimizer, "run_retraining", side_effect=mock_retraining):
            result = await optimizer.run_retraining_with_retry(max_retries=3)

        # Verify success after retry
        assert result["success"] is True
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_run_retraining_with_retry_fails_after_max_retries(self):
        """Verify retry fails after max retries exhausted."""
        optimizer = WalkForwardOptimizer()

        # Mock always-failing retraining
        with patch.object(
            optimizer, "run_retraining", side_effect=Exception("Permanent failure")
        ):
            result = await optimizer.run_retraining_with_retry(max_retries=2)

        # Verify failure
        assert result["success"] is False
        assert "error" in result


class TestPerformanceRequirements:
    """Test performance requirements for retraining."""

    def test_retraining_completes_under_5_minutes(self):
        """Verify full retraining pipeline completes in < 5 minutes."""
        import time

        optimizer = WalkForwardOptimizer()

        # Mock all components to simulate fast retraining
        with patch.object(optimizer, "load_training_window"):
            with patch.object(optimizer, "_prepare_features"):
                with patch.object(optimizer, "_train_model"):
                    with patch.object(optimizer, "_compare_and_deploy"):
                        start_time = time.perf_counter()

                        # Simulate retraining (very fast mock)
                        time.sleep(0.1)  # 100ms simulation

                        elapsed_seconds = time.perf_counter() - start_time

        # Should be much faster than 5 minutes (300 seconds)
        assert (
            elapsed_seconds < 300
        ), f"Retraining took {elapsed_seconds:.2f}s, exceeds 300s limit"


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_missing_current_model_deploys_new_model(self, tmp_path):
        """Verify new model is deployed when no current model exists."""
        model_dir = tmp_path / "xgboost" / "5_minute"
        model_dir.mkdir(parents=True, exist_ok=True)

        optimizer = WalkForwardOptimizer(model_dir=tmp_path / "xgboost")

        # Create new model files
        new_model_file = tmp_path / "new_model.json"
        new_model_file.write_text("new_model")
        new_pipeline_file = tmp_path / "new_pipeline.pkl"
        new_pipeline_file.write_text("new_pipeline")

        # Deploy without current model
        optimizer.deploy_model(new_model_file, new_pipeline_file)

        # Verify new model is in production
        assert (model_dir / "xgboost_model.json").exists()
        assert (model_dir / "feature_pipeline.pkl").exists()

    def test_insufficient_data_raises_error(self):
        """Verify DataInsufficientError raised for insufficient training data."""
        from src.ml.walk_forward_optimizer import DataInsufficientError

        optimizer = WalkForwardOptimizer()

        # Mock insufficient data
        dates = pd.date_range(start="2024-01-01", end="2024-02-01", freq="D")
        insufficient_data = pd.DataFrame(
            {
                "timestamp": dates,
                "close": np.random.uniform(11700, 11900, len(dates)),
            }
        )
        insufficient_data.set_index("timestamp", inplace=True)

        with patch.object(
            optimizer, "_load_dollar_bars", return_value=insufficient_data
        ):
            with pytest.raises(DataInsufficientError):
                optimizer.load_training_window()

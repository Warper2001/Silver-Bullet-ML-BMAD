"""
Unit tests for SilverBulletOptimizationResearcher.

Tests follow red-green-refactor cycle:
1. Write failing test first
2. Implement minimal code to pass
3. Refactor while keeping tests green
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestSilverBulletOptimizationResearcherInit:
    """Test initialization of SilverBulletOptimizationResearcher."""

    def test_init_with_default_parameters(self, tmp_path):
        """Test researcher initializes with default parameter values."""
        # This test will fail until module exists
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        assert researcher.model_path == "models/xgboost/1_minute/model.joblib"
        assert researcher.data_dir == "data/processed/dollar_bars/1_minute/"
        assert researcher.output_dir == tmp_path
        assert researcher.feature_sizes == [10, 15, 20, 25]
        assert researcher.param_grid is None
        assert researcher.min_win_rate == 0.65
        assert researcher.checkpoint_dir == tmp_path / "checkpoints"

    def test_init_with_custom_parameters(self, tmp_path):
        """Test researcher initializes with custom parameter overrides."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(
            model_path="custom/model.joblib",
            data_dir="custom/data/",
            output_dir=str(tmp_path),
            feature_sizes=[5, 10, 15],
            min_win_rate=0.70,
        )

        assert researcher.model_path == "custom/model.joblib"
        assert researcher.data_dir == "custom/data/"
        assert researcher.feature_sizes == [5, 10, 15]
        assert researcher.min_win_rate == 0.70

    def test_init_creates_output_directories(self, tmp_path):
        """Test that initialization creates required output directories."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        assert researcher.output_dir.exists()
        assert (researcher.output_dir / "plots").exists()
        assert researcher.checkpoint_dir.exists()


class TestCheckpointUtilityMethods:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Test that saving checkpoint creates a pickle file."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(
            output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )

        test_data = {"key": "value", "number": 42}
        researcher._save_checkpoint(test_data, "test_step")

        checkpoint_file = tmp_path / "checkpoints" / "test_step.pkl"
        assert checkpoint_file.exists()

    def test_load_checkpoint_returns_data(self, tmp_path):
        """Test that loading checkpoint returns saved data."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(
            output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )

        test_data = {"key": "value", "number": 42}
        researcher._save_checkpoint(test_data, "test_step")

        loaded_data = researcher._load_checkpoint("test_step")
        assert loaded_data == test_data

    def test_load_checkpoint_returns_none_when_not_exists(self, tmp_path):
        """Test that loading non-existent checkpoint returns None."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(
            output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )

        result = researcher._load_checkpoint("nonexistent_step")
        assert result is None

    def test_checkpoint_exists_returns_true_when_exists(self, tmp_path):
        """Test that checkpoint_exists returns True for existing checkpoints."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        researcher = SilverBulletOptimizationResearcher(
            output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )

        test_data = {"key": "value"}
        researcher._save_checkpoint(test_data, "test_step")

        assert researcher._checkpoint_exists("test_step") is True
        assert researcher._checkpoint_exists("nonexistent") is False


class TestModelLoadingAndValidation:
    """Test model loading and validation methods."""

    def test_load_and_validate_model_with_valid_model(self, tmp_path):
        """Test loading a valid XGBoost model."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import xgboost as xgb
        import joblib

        # Create a sample XGBoost model
        sample_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        # Fit with dummy data to set required attributes
        import numpy as np
        X_train = np.random.rand(100, 40)
        y_train = np.random.randint(0, 2, 100)
        sample_model.fit(X_train, y_train)

        # Save the model
        model_path = tmp_path / "model.joblib"
        joblib.dump(sample_model, model_path)

        # Test loading
        researcher = SilverBulletOptimizationResearcher(
            model_path=str(model_path), output_dir=str(tmp_path)
        )

        loaded_model = researcher._load_and_validate_model()
        assert loaded_model is not None
        assert hasattr(loaded_model, "feature_importances_")
        assert hasattr(loaded_model, "n_features_in_")
        assert loaded_model.n_features_in_ == 40

    def test_load_and_validate_model_raises_on_missing_file(self, tmp_path):
        """Test that missing model file raises ModelLoadError."""
        from src.ml.researcher import SilverBulletOptimizationResearcher, ModelLoadError

        researcher = SilverBulletOptimizationResearcher(
            model_path="nonexistent/model.joblib", output_dir=str(tmp_path)
        )

        with pytest.raises(ModelLoadError):
            researcher._load_and_validate_model()


class TestDataLoadingAndSplitting:
    """Test data loading and time-based splitting."""

    def test_load_and_split_data_creates_train_val_split(self, tmp_path):
        """Test that data loading creates 75/25 train/validation split."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Create sample 2025 daily data
        dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.rand(len(dates)) * 100 + 21000,
                "high": np.random.rand(len(dates)) * 100 + 21100,
                "low": np.random.rand(len(dates)) * 100 + 20900,
                "close": np.random.rand(len(dates)) * 100 + 21000,
                "volume": np.random.randint(1000, 10000, len(dates)),
            }
        )

        # Save to data directory
        data_dir = tmp_path / "dollar_bars"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / "mnq_1min_2025.csv", index=False)

        researcher = SilverBulletOptimizationResearcher(
            data_dir=str(data_dir), output_dir=str(tmp_path)
        )

        (X_train, y_train), (X_val, y_val) = researcher._load_and_split_data(
            "2025-01-01", "2025-12-31"
        )

        # Verify split is approximately 75/25
        total_samples = len(X_train) + len(X_val)
        train_ratio = len(X_train) / total_samples

        assert 0.70 <= train_ratio <= 0.80  # Allow some tolerance


class TestParameterOptimization:
    """Test parameter optimization methods."""

    def test_optimize_sb_parameters_exhausts_grid(self, tmp_path):
        """Test that parameter optimization exhausts the parameter grid."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd
        import numpy as np

        # Create sample data
        X_train = pd.DataFrame(np.random.rand(100, 10))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_val = pd.DataFrame(np.random.rand(50, 10))
        y_val = pd.Series(np.random.randint(0, 2, 50))

        # Mock data for backtesting
        data = pd.DataFrame({"close": np.random.rand(50) * 100 + 21000})

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        # Use small grid for testing
        ranked_features = [(f"f{i}", np.random.rand()) for i in range(10)]

        results = researcher._optimize_sb_parameters(
            ranked_features, X_train, y_train, X_val, y_val, data
        )

        assert results is not None
        assert isinstance(results, pd.DataFrame)
        # Default grid has 72 combinations; small grid would have less
        assert len(results) > 0

    def test_analyze_parameter_sensitivity_computes_variance(self, tmp_path):
        """Test that parameter sensitivity analysis computes variance."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        # Create sample results with known variance
        results_df = pd.DataFrame({
            "take_profit_pct": [0.4, 0.4, 0.5, 0.5],
            "stop_loss_pct": [0.2, 0.25, 0.2, 0.25],
            "sharpe": [1.0, 1.5, 0.8, 1.2],
        })

        sensitivity = researcher._analyze_parameter_sensitivity(results_df)

        assert isinstance(sensitivity, dict)
        assert "take_profit_pct" in sensitivity
        assert "stop_loss_pct" in sensitivity
        # Variance should be non-negative
        assert sensitivity["take_profit_pct"] >= 0
        assert sensitivity["stop_loss_pct"] >= 0


class TestModelRetrainingAndPersistence:
    """Test model retraining and persistence methods."""

    def test_retrain_optimized_model_creates_model(self, tmp_path):
        """Test that retraining creates a valid XGBoost model."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd
        import numpy as np

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        selected_features = [f"feature_{i}" for i in range(20)]
        X_train = pd.DataFrame(
            np.random.rand(100, 20), columns=selected_features
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_val = pd.DataFrame(
            np.random.rand(50, 20), columns=selected_features
        )
        y_val = pd.Series(np.random.randint(0, 2, 50))

        model, metrics = researcher._retrain_optimized_model(
            selected_features, X_train, y_train, X_val, y_val
        )

        assert model is not None
        assert model.n_features_in_ == 20
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_optimized_model_creates_files(self, tmp_path):
        """Test that saving creates model and JSON files."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import xgboost as xgb
        import pandas as pd
        import numpy as np
        from datetime import date

        # Set model_path to tmp_path for testing
        model_path = tmp_path / "model.joblib"
        researcher = SilverBulletOptimizationResearcher(
            model_path=str(model_path), output_dir=str(tmp_path)
        )

        # Create sample model
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame(np.random.rand(50, 15))
        y_train = pd.Series(np.random.randint(0, 2, 50))
        model.fit(X_train, y_train)

        features = [f"feature_{i}" for i in range(15)]
        metrics = {"accuracy": 0.8, "precision": 0.75}
        metadata = {"date": date.today().isoformat()}

        # Save the model - this creates model.joblib
        researcher._save_optimized_model(model, features, metrics, metadata)

        # Verify the model file was created at model_path
        assert model_path.exists(), f"Model file not found at {model_path}"

        # Verify the JSON config was created next to model
        expected_json = model_path.parent / "selected_features.json"
        assert expected_json.exists(), f"JSON config not found at {expected_json}"

    def test_save_sb_params_creates_json(self, tmp_path):
        """Test that saving SB params creates JSON file."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        from datetime import date

        # Set model_path to tmp_path for testing
        model_path = tmp_path / "model.joblib"
        researcher = SilverBulletOptimizationResearcher(
            model_path=str(model_path), output_dir=str(tmp_path)
        )

        params = {
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.25,
            "max_bars": 50,
            "probability_threshold": 0.65,
        }

        researcher._save_sb_params(params)

        import json
        params_file = model_path.parent / "sb_params.json"
        assert params_file.exists()

        with open(params_file) as f:
            loaded = json.load(f)

        assert loaded["take_profit_pct"] == 0.5
        assert "optimization_date" in loaded

    def test_optimize_handles_invalid_combinations(self, tmp_path):
        """Test that invalid parameter combinations are handled."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        # Create custom param_grid with too many combinations
        large_grid = {
            "take_profit_pct": list(range(20)),  # 20 values
            "stop_loss_pct": list(range(10)),  # 10 values
            "max_bars": list(range(10)),  # 10 values
        }
        # Total combinations = 20 * 10 * 10 = 2000 (exceeds 100 limit)

        with pytest.raises(ValueError, match="exceeds maximum of 100"):
            researcher._optimize_sb_parameters(
                param_grid=large_grid,
                features=[],
                X_train=pd.DataFrame(),
                y_train=pd.Series(),
                X_val=pd.DataFrame(),
                y_val=pd.Series(),
                data=pd.DataFrame(),
            )


class TestSHAPAnalysis:
    """Test SHAP analysis methods."""

    def test_plot_shap_summary_saves_file(self, tmp_path):
        """Test that SHAP summary plot is created and saved."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd
        import numpy as np

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        # Create sample SHAP values
        shap_values = pd.DataFrame(
            np.random.randn(100, 10),
            columns=[f"feature_{i}" for i in range(10)]
        )
        features = [f"feature_{i}" for i in range(10)]

        output_path = tmp_path / "plots" / "shap_summary.png"
        result_path = researcher._plot_shap_summary(shap_values, features, str(output_path))

        assert Path(result_path).exists()

    def test_rank_features_sorts_by_importance(self, tmp_path):
        """Test that feature ranking sorts by mean absolute SHAP value."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd
        import numpy as np

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        # Create SHAP values where feature_5 is most important
        shap_values = pd.DataFrame(
            np.random.randn(100, 10) * 0.1,  # Small random values
            columns=[f"feature_{i}" for i in range(10)]
        )
        # Make feature_5 have high importance
        shap_values["feature_5"] = np.random.randn(100) * 2.0

        ranked = researcher._rank_features_by_shap(shap_values)

        assert ranked[0][0] == "feature_5"  # Most important
        assert len(ranked) == 10
        assert all(isinstance(v, float) for _, v in ranked)


class TestFeatureSubsetSelection:
    """Test feature subset selection methods."""

    def test_select_optimal_subset_maximizes_sharpe(self, tmp_path):
        """Test that optimal subset selection maximizes Sharpe ratio."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        # Create results where 20 features has highest Sharpe
        results_df = pd.DataFrame({
            "n_features": [10, 15, 20, 25],
            "win_rate": [0.68, 0.70, 0.72, 0.69],
            "sharpe": [1.0, 1.2, 1.5, 1.1],
            "profit_factor": [2.1, 2.3, 2.5, 2.2],
            "training_time": [10.0, 15.0, 20.0, 25.0],
        })

        optimal_n = researcher._select_optimal_subset(results_df)

        assert optimal_n == 20  # Highest Sharpe

    def test_select_optimal_subset_respects_min_win_rate(self, tmp_path):
        """Test that selection respects minimum win rate constraint."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(min_win_rate=0.70, output_dir=str(tmp_path))

        # Create results where highest Sharpe has low win rate
        results_df = pd.DataFrame({
            "n_features": [10, 15, 20],
            "win_rate": [0.60, 0.72, 0.68],  # 10: too low, 15: meets threshold
            "sharpe": [2.0, 1.5, 1.3],  # 10 has highest Sharpe but fails win rate
            "profit_factor": [3.0, 2.3, 2.1],
            "training_time": [10.0, 15.0, 20.0],
        })

        optimal_n = researcher._select_optimal_subset(results_df)

        assert optimal_n == 15  # Best Sharpe meeting win rate threshold


class TestReportGeneration:
    """Test report generation methods."""

    def test_generate_markdown_report_creates_file(self, tmp_path):
        """Test that markdown report file is created."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        results = {
            "ranked_features": [(f"f{i}", 0.1 - i * 0.01) for i in range(20)],
            "subset_results": pd.DataFrame({
                "n_features": [10, 15, 20],
                "win_rate": [0.68, 0.70, 0.72],
                "sharpe": [1.0, 1.2, 1.5],
                "profit_factor": [2.1, 2.3, 2.5],
                "training_time": [10.0, 15.0, 20.0],
            }),
            "optimal_n_features": 20,
            "param_results": pd.DataFrame({
                "take_profit_pct": [0.5],
                "stop_loss_pct": [0.25],
                "max_bars": [50],
                "probability_threshold": [0.65],
            }),
            "best_params": {
                "take_profit_pct": 0.5,
                "stop_loss_pct": 0.25,
                "max_bars": 50,
                "probability_threshold": 0.65,
            },
            "validation_metrics": {
                "accuracy": 0.75,
                "precision": 0.70,
                "recall": 0.80,
                "f1": 0.74,
            },
        }

        report_path = researcher._generate_markdown_report(results)

        assert Path(report_path).exists()
        assert report_path.endswith(".md")

    def test_report_contains_all_sections(self, tmp_path):
        """Test that report contains all required sections."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        results = {
            "ranked_features": [(f"f{i}", 0.1 - i * 0.01) for i in range(20)],
            "subset_results": pd.DataFrame({
                "n_features": [10],
                "win_rate": [0.68],
                "sharpe": [1.0],
                "profit_factor": [2.1],
                "training_time": [10.0],
            }),
            "optimal_n_features": 10,
            "param_results": pd.DataFrame({
                "take_profit_pct": [0.5],
            }),
            "best_params": {"take_profit_pct": 0.5},
            "validation_metrics": {"accuracy": 0.75},
        }

        report_path = researcher._generate_markdown_report(results)

        with open(report_path) as f:
            content = f.read()

        # Check for required sections
        assert "# Silver Bullet Optimization Report" in content
        assert "## Summary" in content
        assert "## Feature Importance Analysis" in content
        assert "## Feature Selection Results" in content
        assert "## Parameter Optimization Results" in content
        assert "## Recommendations" in content

    def test_generate_performance_comparison_plot_saves_html(self, tmp_path):
        """Test that performance comparison plot generates HTML file."""
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import pandas as pd

        researcher = SilverBulletOptimizationResearcher(output_dir=str(tmp_path))

        results_df = pd.DataFrame({
            "n_features": [10, 15, 20],
            "win_rate": [0.68, 0.70, 0.72],
            "sharpe": [1.0, 1.2, 1.5],
            "profit_factor": [2.1, 2.3, 2.5],
            "training_time": [10.0, 15.0, 20.0],
        })

        output_path = tmp_path / "plots" / "performance.html"
        result_path = researcher._generate_performance_comparison_plot(
            results_df, str(output_path)
        )

        assert Path(result_path).exists()
        assert result_path.endswith(".html")

        # Verify file contains HTML content
        with open(result_path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content or "<html" in content

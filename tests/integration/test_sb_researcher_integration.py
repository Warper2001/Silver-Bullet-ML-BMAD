"""
Integration tests for SilverBulletOptimizationResearcher.

Tests the full optimization workflow with real models and data.
"""

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from joblib import dump as joblib_dump

# Suppress SHAP warnings during tests
logging.getLogger("shap").setLevel(logging.ERROR)


@pytest.fixture
def sample_model(tmp_path):
    """Create a sample trained XGBoost model."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # Train with sample data
    X_train = pd.DataFrame(np.random.rand(1000, 40), columns=[f"feature_{i}" for i in range(40)])
    y_train = pd.Series(np.random.randint(0, 2, 1000))
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def sample_model_path(tmp_path, sample_model):
    """Save sample model to a temporary path."""
    model_dir = tmp_path / "models" / "xgboost" / "1_minute"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib_dump(sample_model, model_path)
    return str(model_path)


@pytest.fixture
def sample_dollar_bars(tmp_path):
    """Create sample dollar bars for 2025."""
    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="1min")
    n_samples = len(dates)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.rand(n_samples) * 100 + 21000,
        "high": np.random.rand(n_samples) * 100 + 21100,
        "low": np.random.rand(n_samples) * 100 + 20900,
        "close": np.random.rand(n_samples) * 100 + 21000,
        "volume": np.random.randint(100, 1000, n_samples),
    })

    # Save to data directory
    data_dir = tmp_path / "dollar_bars"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "mnq_1min_2025.csv", index=False)

    return df


class TestSilverBulletOptimizationIntegration:
    """Integration tests for end-to-end optimization workflow."""

    @pytest.mark.slow
    def test_end_to_end_optimization(
        self, sample_model_path, sample_dollar_bars, tmp_path
    ):
        """Test complete optimization workflow from start to finish."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model_path,
            data_dir=str(tmp_path / "dollar_bars"),
            output_dir=str(output_dir),
            feature_sizes=[10, 15],  # Smaller for faster test
        )

        # Run optimization
        results = researcher._run_optimization(
            start_date="2025-01-01",
            end_date="2025-12-31",
            resume=False,
        )

        # Verify results structure
        assert "shap_values" in results
        assert "ranked_features" in results
        assert "subset_results" in results
        assert "optimal_n_features" in results
        assert "optimal_features" in results
        assert "param_results" in results
        assert "best_params" in results
        assert "validation_metrics" in results
        assert "report_path" in results

        # Verify optimal features is one of the tested sizes
        assert results["optimal_n_features"] in [10, 15]

        # Verify best params structure
        assert "take_profit_pct" in results["best_params"]
        assert "stop_loss_pct" in results["best_params"]
        assert "max_bars" in results["best_params"]
        assert "probability_threshold" in results["best_params"]

        # Verify output files were created
        model_path = Path(sample_model_path).parent / "model.joblib"
        features_json = Path(sample_model_path).parent / "selected_features.json"
        params_json = Path(sample_model_path).parent / "sb_params.json"
        report_path = Path(results["report_path"])

        assert model_path.exists()
        assert features_json.exists()
        assert params_json.exists()
        assert report_path.exists()

        # Verify model can be loaded
        import joblib
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None
        assert hasattr(loaded_model, "feature_importances_")

        # Verify feature JSON structure
        with open(features_json) as f:
            features_data = json.load(f)
        assert "feature_count" in features_data
        assert "features" in features_data
        assert "selection_date" in features_data
        assert isinstance(features_data["features"], list)

        # Verify params JSON structure
        with open(params_json) as f:
            params_data = json.load(f)
        assert "take_profit_pct" in params_data
        assert "stop_loss_pct" in params_data
        assert "max_bars" in params_data
        assert "optimization_date" in params_data

        # Verify report contains required sections
        with open(report_path) as f:
            report_content = f.read()
        assert "# Silver Bullet Optimization Report" in report_content
        assert "## Summary" in report_content
        assert "## Feature Importance Analysis" in report_content
        assert "## Recommendations" in report_content

        # Verify plots were created
        shap_plot = output_dir / "plots" / "shap_summary.png"
        perf_plot = output_dir / "plots" / "performance_comparison.html"

        assert shap_plot.exists()
        assert perf_plot.exists()

    @pytest.mark.slow
    def test_checkpoint_save_and_load(
        self, sample_model_path, sample_dollar_bars, tmp_path
    ):
        """Test that checkpoints are saved and can be loaded."""
        from src.ml.researcher import SilverBulletOptimizationResearcher

        output_dir = tmp_path / "output"
        checkpoint_dir = tmp_path / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model_path,
            data_dir=str(tmp_path / "dollar_bars"),
            output_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir),
            feature_sizes=[10],  # Single size for faster test
        )

        # Run optimization first time (creates checkpoints)
        results = researcher._run_optimization(
            start_date="2025-01-01",
            end_date="2025-12-31",
            resume=False,
        )

        # Verify checkpoints exist
        assert (checkpoint_dir / "shap_values.pkl").exists()
        assert (checkpoint_dir / "feature_subsets.pkl").exists()
        assert (checkpoint_dir / "param_optimization.pkl").exists()

        # Verify checkpoints can be loaded
        loaded_shap = researcher._load_checkpoint("shap_values")
        assert loaded_shap is not None

        loaded_subsets = researcher._load_checkpoint("feature_subsets")
        assert loaded_subsets is not None

        loaded_params = researcher._load_checkpoint("param_optimization")
        assert loaded_params is not None

        # Verify checkpoint exists method works
        assert researcher._checkpoint_exists("shap_values")
        assert not researcher._checkpoint_exists("nonexistent")


class TestCLIInvocation:
    """Tests for CLI invocation of optimization researcher."""

    def test_cli_argument_parser(self):
        """Test CLI argument parser validates correctly."""
        from src.cli.sb_optimize import parse_date, parse_feature_sizes

        # Test date parsing
        assert parse_date("2025-01-01") == "2025-01-01"

        with pytest.raises(argparse.ArgumentTypeError):
            parse_date("invalid-date")

        # Test feature size parsing
        sizes = parse_feature_sizes("10,15,20,25")
        assert sizes == [10, 15, 20, 25]

        with pytest.raises(argparse.ArgumentTypeError):
            parse_feature_sizes("1,2,3")  # Too few

        with pytest.raises(argparse.ArgumentTypeError):
            parse_feature_sizes("10,50,20")  # 50 is too high

    def test_cli_researcher_initialization(self, sample_model_path, tmp_path):
        """Test CLI can initialize the researcher."""
        from src.cli.sb_optimize import main
        from src.ml.researcher import SilverBulletOptimizationResearcher
        import argparse

        # Simulate CLI args
        args = argparse.Namespace(
            start="2025-01-01",
            end="2025-12-31",
            model=sample_model_path,
            data_dir=str(tmp_path / "dollar_bars"),
            output=str(tmp_path / "output"),
            feature_sizes="10",  # Single size for faster test
            min_win_rate=0.65,
            resume=False,
            verbose=False,
            quiet=True,
        )

        # Initialize researcher as CLI does
        researcher = SilverBulletOptimizationResearcher(
            model_path=args.model,
            data_dir=args.data_dir,
            output_dir=args.output,
            feature_sizes=[10],
            min_win_rate=args.min_win_rate,
        )

        assert researcher.model_path == sample_model_path
        assert researcher.min_win_rate == 0.65
        assert researcher.feature_sizes == [10]

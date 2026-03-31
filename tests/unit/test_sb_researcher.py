"""Unit tests for Silver Bullet Optimization Researcher."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.ml.researcher import (
    InsufficientDataError,
    ModelLoadError,
    OptimizationStatistics,
    SilverBulletOptimizationResearcher,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        output_dir = tmpdir / "output"
        checkpoint_dir = tmpdir / "checkpoints"
        model_dir = tmpdir / "models"

        for dir_path in [data_dir, output_dir, checkpoint_dir, model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        yield {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "model_dir": str(model_dir),
        }


@pytest.fixture
def sample_model(temp_dirs):
    """Create a sample trained XGBoost model."""
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)

    # Train on sample data
    X_train = np.random.rand(100, 40)
    y_train = np.random.randint(0, 2, 100)
    model.fit(X_train, y_train)

    # Save model
    model_path = Path(temp_dirs["model_dir"]) / "model.joblib"
    import joblib

    joblib.dump(model, model_path)

    return str(model_path)


@pytest.fixture
def sample_data():
    """Create sample dollar bar data."""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="5min"),
            "open": np.random.rand(n_samples) * 100 + 15000,
            "high": np.random.rand(n_samples) * 100 + 15100,
            "low": np.random.rand(n_samples) * 100 + 14900,
            "close": np.random.rand(n_samples) * 100 + 15000,
            "volume": np.random.randint(100, 1000, n_samples),
            "dollar_volume": np.random.rand(n_samples) * 1000000,
        }
    )

    return data


@pytest.fixture
def sample_features():
    """Create sample feature matrix."""
    np.random.seed(42)
    n_samples = 500
    n_features = 40

    features = {}
    for i in range(n_features):
        features[f"feature_{i}"] = np.random.rand(n_samples)

    df = pd.DataFrame(features)
    return df


# ============================================================================
# Test OptimizationStatistics
# ============================================================================


class TestOptimizationStatistics:
    """Tests for OptimizationStatistics class."""

    def test_init_creates_zero_values(self):
        """Test initialization sets all values to zero."""
        stats = OptimizationStatistics()

        assert stats.start_time > 0
        assert stats.shap_computation_time == 0.0
        assert stats.feature_selection_time == 0.0
        assert stats.parameters_tested == 0
        assert stats.features_analyzed == 0

    def test_to_dict_returns_nested_structure(self):
        """Test to_dict returns properly nested dictionary."""
        stats = OptimizationStatistics()
        stats.features_analyzed = 40
        stats.features_selected = 20

        result = stats.to_dict()

        assert "timing" in result
        assert "features" in result
        assert "optimization" in result
        assert result["features"]["analyzed"] == 40
        assert result["features"]["selected"] == 20


# ============================================================================
# Test SilverBulletOptimizationResearcher Initialization
# ============================================================================


class TestSilverBulletOptimizationResearcherInit:
    """Tests for SilverBulletOptimizationResearcher initialization."""

    def test_init_with_default_parameters(self, temp_dirs, sample_model):
        """Test initialization with default parameters."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        assert researcher._min_win_rate == 0.65
        assert researcher._feature_sizes == [10, 15, 20, 25]
        assert len(researcher._param_grid) == 4

    def test_init_with_custom_parameters(self, temp_dirs, sample_model):
        """Test initialization with custom parameters."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            feature_sizes=[5, 10, 15],
            min_win_rate=0.70,
        )

        assert researcher._min_win_rate == 0.70
        assert researcher._feature_sizes == [5, 10, 15]

    def test_init_creates_output_directories(self, temp_dirs, sample_model):
        """Test initialization creates output directories."""
        output_dir = Path(temp_dirs["output_dir"])

        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=str(output_dir),
        )

        assert output_dir.exists()
        assert (output_dir / "plots").exists()
        assert researcher._plots_dir.exists()


# ============================================================================
# Test Checkpoint Methods
# ============================================================================


class TestCheckpointMethods:
    """Tests for checkpoint functionality."""

    def test_save_checkpoint_creates_file(self, temp_dirs, sample_model):
        """Test _save_checkpoint creates checkpoint file."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            checkpoint_dir=temp_dirs["checkpoint_dir"],
        )

        test_data = {"key": "value", "number": 42}
        researcher._save_checkpoint(test_data, "test_step")

        checkpoint_path = Path(temp_dirs["checkpoint_dir"]) / "test_step.pkl"
        assert checkpoint_path.exists()

    def test_load_checkpoint_returns_data(self, temp_dirs, sample_model):
        """Test _load_checkpoint returns saved data."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            checkpoint_dir=temp_dirs["checkpoint_dir"],
        )

        test_data = {"key": "value", "number": 42}
        researcher._save_checkpoint(test_data, "test_step")

        loaded_data = researcher._load_checkpoint("test_step")
        assert loaded_data == test_data

    def test_load_checkpoint_returns_none_for_missing(self, temp_dirs, sample_model):
        """Test _load_checkpoint returns None for non-existent checkpoint."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            checkpoint_dir=temp_dirs["checkpoint_dir"],
        )

        result = researcher._load_checkpoint("nonexistent")
        assert result is None

    def test_checkpoint_exists(self, temp_dirs, sample_model):
        """Test _checkpoint_exists checks file existence."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            checkpoint_dir=temp_dirs["checkpoint_dir"],
        )

        assert not researcher._checkpoint_exists("test_step")

        researcher._save_checkpoint({"data": 1}, "test_step")
        assert researcher._checkpoint_exists("test_step")


# ============================================================================
# Test Model Loading and Validation
# ============================================================================


class TestModelLoading:
    """Tests for model loading functionality."""

    def test_load_and_validate_model_succeeds(self, temp_dirs, sample_model):
        """Test _load_and_validate_model succeeds with valid model."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        model = researcher._load_and_validate_model()
        assert isinstance(model, XGBClassifier)

    def test_load_and_validate_model_raises_for_missing_file(self, temp_dirs):
        """Test _load_and_validate_model raises ModelLoadError for missing file."""
        researcher = SilverBulletOptimizationResearcher(
            model_path="nonexistent_model.joblib",
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        with pytest.raises(ModelLoadError, match="Model file not found"):
            researcher._load_and_validate_model()


# ============================================================================
# Test SHAP Methods
# ============================================================================


class TestSHAPAnalysis:
    """Tests for SHAP analysis methods."""

    @patch("src.ml.researcher.shap.TreeExplainer")
    def test_rank_features_sorts_by_importance(
        self, mock_explainer, temp_dirs, sample_model
    ):
        """Test _rank_features_by_shap sorts correctly."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        # Create mock SHAP values
        shap_df = pd.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3],
                "feature_b": [0.5, 0.6, 0.7],
                "feature_c": [0.05, 0.1, 0.15],
            }
        )

        ranked = researcher._rank_features_by_shap(shap_df)

        assert ranked[0][0] == "feature_b"  # Highest importance
        assert ranked[-1][0] == "feature_c"  # Lowest importance

    @patch("src.ml.researcher.plt")
    def test_plot_shap_summary_saves_file(self, mock_plt, temp_dirs, sample_model):
        """Test _plot_shap_summary saves plot file."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        shap_df = pd.DataFrame({"feature_a": [0.1, 0.2, 0.3]})
        output_path = researcher._plots_dir / "test_shap.png"

        result = researcher._plot_shap_summary(shap_df, output_path)

        assert result == output_path
        mock_plt.savefig.assert_called_once()


# ============================================================================
# Test Feature Subset Selection
# ============================================================================


class TestFeatureSubsetSelection:
    """Tests for feature subset selection methods."""

    def test_select_optimal_subset_maximizes_sharpe(self, temp_dirs, sample_model):
        """Test _select_optimal_subset maximizes Sharpe ratio."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            min_win_rate=0.60,
        )

        # Create mock results
        results_df = pd.DataFrame(
            {
                "n_features": [10, 15, 20, 25],
                "win_rate": [0.62, 0.65, 0.68, 0.64],
                "precision": [0.60, 0.63, 0.66, 0.62],
                "recall": [0.70, 0.72, 0.75, 0.71],
                "f1": [0.64, 0.67, 0.70, 0.66],
                "sharpe_proxy": [1.2, 1.5, 1.8, 1.4],
            }
        )

        optimal_n, _ = researcher._select_optimal_subset(results_df)

        # Should select 20 features (highest sharpe_proxy meeting win rate)
        assert optimal_n == 20

    def test_select_optimal_subset_respects_min_win_rate(self, temp_dirs, sample_model):
        """Test _select_optimal_subset respects minimum win rate."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            min_win_rate=0.70,
        )

        # Create mock results where only one meets threshold
        results_df = pd.DataFrame(
            {
                "n_features": [10, 15, 20, 25],
                "win_rate": [0.62, 0.68, 0.72, 0.64],
                "precision": [0.60, 0.65, 0.70, 0.62],
                "recall": [0.70, 0.74, 0.78, 0.71],
                "f1": [0.64, 0.69, 0.74, 0.66],
                "sharpe_proxy": [1.2, 1.5, 1.8, 1.4],
            }
        )

        optimal_n, _ = researcher._select_optimal_subset(results_df)

        # Should select 20 features (only one meeting 0.70 threshold)
        assert optimal_n == 20


# ============================================================================
# Test Parameter Optimization
# ============================================================================


class TestParameterOptimization:
    """Tests for parameter optimization methods."""

    def test_analyze_parameter_sensitivity_computes_variance(
        self, temp_dirs, sample_model
    ):
        """Test _analyze_parameter_sensitivity computes variance."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        # Create mock results
        results_df = pd.DataFrame(
            {
                "take_profit_pct": [0.5, 0.5, 0.6, 0.6],
                "stop_loss_pct": [0.2, 0.3, 0.2, 0.3],
                "max_bars": [40, 50, 40, 50],
                "probability_threshold": [0.65, 0.65, 0.70, 0.70],
                "win_rate": [0.60, 0.65, 0.62, 0.68],
                "n_trades": [100, 120, 90, 110],
            }
        )

        sensitivity = researcher._analyze_parameter_sensitivity(results_df)

        assert "take_profit_pct" in sensitivity
        assert "stop_loss_pct" in sensitivity
        assert "max_bars" in sensitivity
        assert "probability_threshold" in sensitivity

        # All sensitivities should be non-negative
        for value in sensitivity.values():
            assert value >= 0


# ============================================================================
# Test Report Generation
# ============================================================================


class TestReportGeneration:
    """Tests for report generation methods."""

    def test_generate_markdown_report_creates_file(self, temp_dirs, sample_model):
        """Test _generate_markdown_report creates markdown file."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        results = {
            "feature_rankings": [("feature_a", 0.5), ("feature_b", 0.3)],
            "feature_subset_results": pd.DataFrame(
                {
                    "n_features": [10, 20],
                    "win_rate": [0.60, 0.65],
                    "precision": [0.58, 0.63],
                    "f1": [0.62, 0.67],
                    "sharpe_proxy": [1.2, 1.5],
                }
            ),
            "param_results": pd.DataFrame(
                {
                    "take_profit_pct": [0.5],
                    "stop_loss_pct": [0.25],
                    "max_bars": [50],
                    "probability_threshold": [0.65],
                    "win_rate": [0.68],
                    "n_trades": [100],
                }
            ),
        }

        report_path = researcher._generate_markdown_report(results)

        assert report_path.exists()
        assert report_path.suffix == ".md"

        # Read and verify content
        content = report_path.read_text()
        assert "Silver Bullet Optimization Report" in content
        assert "## Summary" in content
        assert "## Feature Importance" in content

    def test_report_contains_all_sections(self, temp_dirs, sample_model):
        """Test generated report contains all required sections."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        results = {
            "feature_rankings": [("feature_a", 0.5)],
            "feature_subset_results": pd.DataFrame(
                {
                    "n_features": [10],
                    "win_rate": [0.60],
                    "precision": [0.58],
                    "f1": [0.62],
                    "sharpe_proxy": [1.2],
                }
            ),
            "param_results": pd.DataFrame(
                {
                    "take_profit_pct": [0.5],
                    "stop_loss_pct": [0.25],
                    "max_bars": [50],
                    "probability_threshold": [0.65],
                    "win_rate": [0.68],
                    "n_trades": [100],
                }
            ),
        }

        report_path = researcher._generate_markdown_report(results)
        content = report_path.read_text()

        required_sections = [
            "## Summary",
            "## Feature Importance",
            "## Feature Selection Results",
            "## Parameter Optimization",
            "## Recommendations",
        ]

        for section in required_sections:
            assert section in content

    def test_generate_performance_comparison_plot(self, temp_dirs, sample_model):
        """Test _generate_performance_comparison_plot creates plot."""
        researcher = SilverBulletOptimizationResearcher(
            model_path=sample_model,
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
        )

        results_df = pd.DataFrame(
            {
                "n_features": [10, 15, 20],
                "win_rate": [0.60, 0.65, 0.68],
                "sharpe_proxy": [1.2, 1.5, 1.8],
            }
        )

        plot_path = researcher._generate_performance_comparison_plot(results_df)

        assert plot_path.parent.exists()
        assert "performance_comparison" in str(plot_path)
        assert plot_path.suffix == ".html"

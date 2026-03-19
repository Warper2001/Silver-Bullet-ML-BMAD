"""Unit tests for FeatureImportanceAnalyzer.

Tests feature importance analysis including importance extraction,
ranking, cumulative importance calculation, visualization, and export.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.research.feature_importance_analyzer import (
    FeatureImportanceAnalyzer
)


class TestFeatureImportanceAnalyzerInit:
    """Test FeatureImportanceAnalyzer initialization."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_init_with_default_parameters(self, mock_pickle_load, mock_open):
        """Verify initialization with default parameters."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_pickle_load.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        assert analyzer._model_path == Path("data/models/xgboost_classifier.pkl")
        assert analyzer._importance_type == "gain"
        assert analyzer._top_n == 20

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_init_with_custom_model_path(self, mock_pickle_load, mock_open):
        """Verify initialization with custom model path."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_pickle_load.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(
            model_path="custom/model.pkl"
        )

        assert analyzer._model_path == Path("custom/model.pkl")

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_init_with_custom_importance_type(self, mock_pickle_load, mock_open):
        """Verify initialization with custom importance type."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_pickle_load.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(
            importance_type="weight"
        )

        assert analyzer._importance_type == "weight"

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_init_with_custom_top_n(self, mock_pickle_load, mock_open):
        """Verify initialization with custom top_n value."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_pickle_load.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(
            top_n=15
        )

        assert analyzer._top_n == 15


class TestLoadXGBoostModel:
    """Test XGBoost model loading."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_load_model_from_pickle_file(self, mock_pickle, mock_file):
        """Verify model loads from pickle file."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()
        model = analyzer._model

        assert model == mock_model
        assert mock_pickle.called

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_model_handles_missing_file(self, mock_open):
        """Verify loading handles missing model file."""
        with pytest.raises(FileNotFoundError):
            FeatureImportanceAnalyzer()

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_validate_model_has_feature_importances(self, mock_pickle, mock_file):
        """Verify model validation checks feature_importances."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        assert hasattr(analyzer._model, 'feature_importances_')


class TestExtractImportanceScores:
    """Test importance score extraction."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_extract_gain_importance_scores(self, mock_pickle, mock_file):
        """Verify extraction of gain importance scores."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(importance_type="gain")

        scores = analyzer.extract_importance_scores()

        assert 'gain' in scores
        assert len(scores['gain']) == 3

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_extract_weight_importance_scores(self, mock_pickle, mock_file):
        """Verify extraction of weight importance scores."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(importance_type="weight")

        scores = analyzer.extract_importance_scores()

        assert 'weight' in scores

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_normalize_scores_to_percentages(self, mock_pickle, mock_file):
        """Verify scores normalized to percentages (sum = 100%)."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        scores = analyzer.extract_importance_scores()

        # Sum should be 100%
        gain_scores = list(scores.values())[0]
        assert abs(sum(gain_scores) - 100.0) < 0.01


class TestRankFeatures:
    """Test feature ranking by importance."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_rank_features_by_importance_descending(self, mock_pickle, mock_file):
        """Verify features ranked by importance (descending)."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([10, 30, 20])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = analyzer.rank_features(
            np.array([10, 30, 20]),
            ['feature_0', 'feature_1', 'feature_2']
        )

        # Should be sorted: feature_1 (30), feature_2 (20), feature_0 (10)
        assert ranked_df.iloc[0]['feature_name'] == 'feature_1'
        assert ranked_df.iloc[1]['feature_name'] == 'feature_2'
        assert ranked_df.iloc[2]['feature_name'] == 'feature_0'

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_assign_correct_ranks(self, mock_pickle, mock_file):
        """Verify correct rank assignment."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([10, 30, 20])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = analyzer.rank_features(
            np.array([10, 30, 20]),
            ['feature_0', 'feature_1', 'feature_2']
        )

        assert ranked_df.iloc[0]['rank'] == 1
        assert ranked_df.iloc[1]['rank'] == 2
        assert ranked_df.iloc[2]['rank'] == 3

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_handle_tie_scores(self, mock_pickle, mock_file):
        """Verify handling of tied importance scores."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([20, 20, 10])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = analyzer.rank_features(
            np.array([20, 20, 10]),
            ['feature_0', 'feature_1', 'feature_2']
        )

        # Both features with 20 should be rank 1 or 2
        top_two = ranked_df.iloc[:2]
        assert all(rank in [1, 2] for rank in top_two['rank'])


class TestCalculateCumulativeImportance:
    """Test cumulative importance calculation."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_calculate_cumulative_importance_correctly(self, mock_pickle, mock_file):
        """Verify cumulative importance calculated correctly."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = pd.DataFrame({
            'feature_name': ['f2', 'f0', 'f1'],
            'importance_gain': [50.0, 30.0, 20.0]
        })

        result = analyzer.calculate_cumulative_importance(ranked_df)

        # Cumulative: 50%, 80%, 100%
        assert result.iloc[0]['cumulative_pct'] == 50.0
        assert result.iloc[1]['cumulative_pct'] == 80.0
        assert result.iloc[2]['cumulative_pct'] == 100.0

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_top_features_have_high_cumulative_pct(self, mock_pickle, mock_file):
        """Verify top features have high cumulative %."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = pd.DataFrame({
            'feature_name': ['f2', 'f0', 'f1'],
            'importance_gain': [50.0, 30.0, 20.0]
        })

        result = analyzer.calculate_cumulative_importance(ranked_df)

        # Top feature should have significant cumulative %
        assert result.iloc[0]['cumulative_pct'] >= 40.0

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_last_feature_cumulative_equals_100(self, mock_pickle, mock_file):
        """Verify last feature cumulative = 100%."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = pd.DataFrame({
            'feature_name': ['f2', 'f0', 'f1'],
            'importance_gain': [50.0, 30.0, 20.0]
        })

        result = analyzer.calculate_cumulative_importance(ranked_df)

        # Last feature cumulative = 100%
        assert abs(result.iloc[-1]['cumulative_pct'] - 100.0) < 0.01


class TestGenerateBarChart:
    """Test bar chart generation."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_create_horizontal_bar_chart(self, mock_pickle, mock_file):
        """Verify horizontal bar chart creation."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        importance_df = pd.DataFrame({
            'feature_name': ['f2', 'f0', 'f1'],
            'importance_gain': [50.0, 30.0, 20.0],
            'rank': [1, 2, 3]
        })

        fig = analyzer.generate_bar_chart(importance_df)

        assert fig is not None
        plt.close(fig)

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_show_correct_number_of_features(self, mock_pickle, mock_file):
        """Verify chart shows top_n features."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([1, 2, 3, 4, 5])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(top_n=3)

        importance_df = pd.DataFrame({
            'feature_name': ['f4', 'f3', 'f2', 'f1', 'f0'],
            'importance_gain': [36.0, 28.0, 20.0, 12.0, 4.0],
            'rank': [1, 2, 3, 4, 5]
        })

        fig = analyzer.generate_bar_chart(importance_df)

        # Should show top 3
        ax = fig.gca()
        bars = ax.containers[0]
        assert len(bars) == 3
        plt.close(fig)

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_add_value_labels_on_bars(self, mock_pickle, mock_file):
        """Verify value labels added to bars."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        importance_df = pd.DataFrame({
            'feature_name': ['f2', 'f0'],
            'importance_gain': [60.0, 40.0],
            'rank': [1, 2]
        })

        fig = analyzer.generate_bar_chart(importance_df)

        # Check for text annotations (value labels)
        ax = fig.gca()
        texts = ax.texts
        assert len(texts) > 0
        plt.close(fig)

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_set_proper_labels_and_title(self, mock_pickle, mock_file):
        """Verify proper axis labels and title."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        importance_df = pd.DataFrame({
            'feature_name': ['f2'],
            'importance_gain': [100.0],
            'rank': [1]
        })

        fig = analyzer.generate_bar_chart(importance_df)

        ax = fig.gca()
        assert ax.get_xlabel() == "Importance Gain (%)"
        assert ax.get_ylabel() == "Feature"
        assert "Feature Importance" in ax.get_title()
        plt.close(fig)


class TestCreateImportanceTable:
    """Test importance table creation."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_create_table_with_correct_columns(self, mock_pickle, mock_file):
        """Verify table has correct columns."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = pd.DataFrame({
            'feature_name': ['f2', 'f0', 'f1'],
            'importance_gain': [50.0, 30.0, 20.0],
            'cumulative_pct': [50.0, 80.0, 100.0]
        })

        table = analyzer.create_importance_table(ranked_df)

        assert 'rank' in table.columns
        assert 'feature_name' in table.columns
        assert 'importance_gain' in table.columns
        assert 'cumulative_pct' in table.columns

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_sort_by_rank_ascending(self, mock_pickle, mock_file):
        """Verify table sorted by rank."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = pd.DataFrame({
            'feature_name': ['f2', 'f0', 'f1'],
            'importance_gain': [50.0, 30.0, 20.0],
            'cumulative_pct': [50.0, 80.0, 100.0],
            'rank': [1, 2, 3]
        })

        table = analyzer.create_importance_table(ranked_df)

        # Should be sorted by rank ascending
        assert table.iloc[0]['rank'] == 1
        assert table.iloc[1]['rank'] == 2
        assert table.iloc[2]['rank'] == 3

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_format_percentages_correctly(self, mock_pickle, mock_file):
        """Verify percentages formatted to 2 decimal places."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        ranked_df = pd.DataFrame({
            'feature_name': ['f2'],
            'importance_gain': [50.123],
            'cumulative_pct': [50.456]
        })

        table = analyzer.create_importance_table(ranked_df)

        # Should be rounded to 2 decimal places
        assert abs(table.iloc[0]['importance_gain'] - 50.12) < 0.01
        assert abs(table.iloc[0]['cumulative_pct'] - 50.46) < 0.01


class TestSaveResults:
    """Test chart and table saving."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    @patch('matplotlib.figure.Figure.savefig')
    def test_save_png_chart_to_correct_location(
        self,
        mock_savefig,
        mock_pickle,
        mock_file
    ):
        """Verify PNG saved to correct location."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        fig = plt.Figure()
        table = pd.DataFrame({'test': [1]})

        png_path, csv_path = analyzer.save_results(fig, table)

        assert "feature_importance_" in png_path
        assert png_path.endswith(".png")

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_table_to_correct_location(
        self, mock_to_csv, mock_pickle, mock_file, tmp_path
    ):
        """Verify CSV saved to correct location."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer(
            output_directory=str(tmp_path)
        )

        ranked_df = pd.DataFrame({
            'feature_name': ['f2'],
            'importance_gain': [100.0],
            'cumulative_pct': [100.0]
        })

        table = analyzer.create_importance_table(ranked_df)

        fig = plt.Figure()

        png_path, csv_path = analyzer.save_results(fig, table)

        assert csv_path.startswith(str(tmp_path))
        assert csv_path.endswith(".csv")
        assert "feature_importance_data_" in csv_path

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    @patch('matplotlib.figure.Figure.savefig')
    def test_filenames_include_timestamp(self, mock_savefig, mock_pickle, mock_file):
        """Verify filenames include timestamp."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        fig = plt.Figure()
        table = pd.DataFrame({'test': [1]})

        png_path, csv_path = analyzer.save_results(fig, table)

        # Check for date pattern (YYYY-MM-DD)
        assert "feature_importance_20" in png_path
        assert "feature_importance_data_20" in csv_path

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    def test_create_directory_if_needed(self, mock_pickle, mock_file, tmp_path):
        """Verify directory created if it doesn't exist."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([30, 20, 50])
        mock_pickle.return_value = mock_model

        output_dir = tmp_path / "new_reports"
        analyzer = FeatureImportanceAnalyzer(
            output_directory=str(output_dir)
        )

        fig = plt.Figure()
        table = pd.DataFrame({'test': [1]})

        png_path, csv_path = analyzer.save_results(fig, table)

        # Directory should be created
        assert output_dir.exists()


class TestAnalyzeFeatureImportance:
    """Test end-to-end feature importance analysis."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    @patch('matplotlib.figure.Figure.savefig')
    def test_end_to_end_analysis_pipeline(self, mock_savefig, mock_pickle, mock_file):
        """Verify complete analysis pipeline."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([10, 30, 20, 15, 25])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        # Create mock feature names
        feature_names = ['returns_5', 'volatility_10', 'volume_ratio',
                         'direction', 'confidence']

        result = analyzer.analyze_feature_importance(feature_names)

        # Should return DataFrame with results
        assert 'importance_df' in result
        assert 'png_path' in result
        assert 'csv_path' in result

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    @patch('matplotlib.figure.Figure.savefig')
    def test_all_components_integrated(
        self,
        mock_savefig,
        mock_pickle,
        mock_file
    ):
        """Verify all components integrated correctly."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([20, 30, 50])
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        feature_names = ['f1', 'f2', 'f3']
        result = analyzer.analyze_feature_importance(feature_names)

        importance_df = result['importance_df']

        # Check columns
        assert 'rank' in importance_df.columns
        assert 'feature_name' in importance_df.columns
        assert 'importance_gain' in importance_df.columns
        assert 'cumulative_pct' in importance_df.columns

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.research.feature_importance_analyzer.pickle.load')
    @patch('matplotlib.figure.Figure.savefig')
    def test_performance_requirement_under_10_seconds(
        self,
        mock_savefig,
        mock_pickle,
        mock_file
    ):
        """Verify analysis completes in < 10 seconds."""
        mock_model = MagicMock()
        # Create model with 100 features
        mock_model.feature_importances_ = np.random.rand(100)
        mock_pickle.return_value = mock_model

        analyzer = FeatureImportanceAnalyzer()

        feature_names = [f'feature_{i}' for i in range(100)]

        start_time = time.time()
        analyzer.analyze_feature_importance(feature_names)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0, \
            f"Analysis took {elapsed_time:.2f} seconds"

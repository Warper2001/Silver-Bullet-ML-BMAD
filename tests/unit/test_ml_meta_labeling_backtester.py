"""Unit tests for MLMetaLabelingBacktester.

Tests ML meta-labeling backtesting including model loading,
feature engineering, probability scoring, and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from src.research.ml_meta_labeling_backtester import (
    MLMetaLabelingBacktester
)


class TestMLMetaLabelingBacktesterInit:
    """Test MLMetaLabelingBacktester initialization."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_with_default_parameters(self, mock_file, mock_pickle):
        """Verify initialization with default parameters."""
        # Mock model and pipeline loading
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        assert backtester._probability_threshold == 0.65
        assert backtester._model_path == Path("data/models/xgboost_classifier.pkl")
        assert backtester._pipeline_path == Path("data/models/feature_pipeline.pkl")
        assert backtester._triple_barrier_config['take_profit_pct'] == 0.5
        assert backtester._triple_barrier_config['stop_loss_pct'] == 0.25
        assert backtester._triple_barrier_config['max_bars'] == 50

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_with_custom_model_path(self, mock_file, mock_pickle):
        """Verify initialization with custom model path."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester(
            model_path="custom/model.pkl"
        )

        assert backtester._model_path == Path("custom/model.pkl")

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_with_custom_probability_threshold(self, mock_file, mock_pickle):
        """Verify initialization with custom probability threshold."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester(
            probability_threshold=0.70
        )

        assert backtester._probability_threshold == 0.70

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_with_custom_triple_barrier_config(self, mock_file, mock_pickle):
        """Verify initialization with custom triple-barrier config."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        custom_config = {
            'take_profit_pct': 0.75,
            'stop_loss_pct': 0.30,
            'max_bars': 40
        }

        backtester = MLMetaLabelingBacktester(
            triple_barrier_config=custom_config
        )

        assert backtester._triple_barrier_config == custom_config


class TestModelAndPipelineLoading:
    """Test model and pipeline loading."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_model_from_pickle(self, mock_file, mock_pickle):
        """Verify model loads from pickle file."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        assert backtester._model == mock_model
        assert mock_pickle.called

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_pipeline_from_pickle(self, mock_file, mock_pickle):
        """Verify pipeline loads from pickle file."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        assert backtester._pipeline == mock_pipeline

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_model_handles_missing_file(self, mock_open):
        """Verify loading handles missing model file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            MLMetaLabelingBacktester()

        assert "Model file not found" in str(exc_info.value)


class TestFeatureEngineering:
    """Test feature engineering for historical signals."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_engineer_features_for_single_signal(self, mock_file, mock_pickle):
        """Verify feature engineering for single signal."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.transform = MagicMock(return_value=np.array([[1, 2, 3]]))
        mock_pipeline.get_feature_names_out = MagicMock(return_value=['f1', 'f2', 'f3'])
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        # Create signal and price data
        signal = pd.Series({
            'direction': 'bullish',
            'confidence': 80,
            'mss_detected': True,
            'fvg_detected': True,
            'sweep_detected': False
        }, name=pd.Timestamp('2024-03-01 10:00:00'))

        signals_df = pd.DataFrame([signal])
        signals_df.index.name = 'timestamp'

        price_data = pd.DataFrame({
            'open': np.full(100, 2100.0),
            'high': np.full(100, 2105.0),
            'low': np.full(100, 2098.0),
            'close': np.full(100, 2102.0),
            'volume': np.full(100, 1000)
        }, index=pd.date_range('2024-03-01 09:00:00', periods=100, freq='5min'))

        features_df = backtester.engineer_features(signals_df, price_data)

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 1

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_engineer_features_extract_pattern_features(self, mock_file, mock_pickle):
        """Verify pattern features extracted correctly."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        # Mock pipeline to return features
        mock_pipeline.transform = MagicMock(
            return_value=np.array([[1, 2, 3, 4, 5]])
        )
        mock_pipeline.get_feature_names_out = MagicMock(
            return_value=['f1', 'f2', 'f3', 'f4', 'f5']
        )

        backtester = MLMetaLabelingBacktester()

        signal = pd.Series({
            'direction': 'bullish',
            'confidence': 75,
            'mss_detected': True,
            'fvg_detected': True,
            'sweep_detected': True
        }, name=pd.Timestamp('2024-03-01 12:00:00'))  # Later in day

        signals_df = pd.DataFrame([signal])

        price_data = pd.DataFrame({
            'open': np.full(100, 2100.0),
            'high': np.full(100, 2105.0),
            'low': np.full(100, 2098.0),
            'close': np.full(100, 2102.0),
            'volume': np.full(100, 1000)
        }, index=pd.date_range('2024-03-01 08:00:00', periods=100, freq='5min'))

        features_df = backtester.engineer_features(signals_df, price_data)

        # Check that features were extracted (pipeline used for this test)
        assert not features_df.empty
        assert len(features_df) == 1


class TestProbabilityScoreGeneration:
    """Test probability score generation."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_generate_probability_scores(self, mock_file, mock_pickle):
        """Verify probability score generation."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        # Mock model predictions
        mock_model.predict_proba = MagicMock(
            return_value=np.array([[0.3, 0.7], [0.6, 0.4]])
        )

        backtester = MLMetaLabelingBacktester()

        features_df = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
        probabilities = backtester.generate_probability_scores(features_df)

        assert len(probabilities) == 2
        assert probabilities[0] == 0.7  # Class 1 probability
        assert probabilities[1] == 0.4

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_probability_scores_between_0_and_1(self, mock_file, mock_pickle):
        """Verify probability scores are between 0 and 1."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        mock_model.predict_proba = MagicMock(
            return_value=np.array([[0.2, 0.8], [0.9, 0.1]])
        )

        backtester = MLMetaLabelingBacktester()

        features_df = pd.DataFrame({'f1': [1, 2]})
        probabilities = backtester.generate_probability_scores(features_df)

        assert all(probabilities >= 0) and all(probabilities <= 1)


class TestProbabilityFiltering:
    """Test probability-based signal filtering."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_filter_by_probability_threshold(self, mock_file, mock_pickle):
        """Verify filtering by probability threshold."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester(probability_threshold=0.65)

        signals_df = pd.DataFrame({
            'direction': ['bullish', 'bearish', 'bullish', 'bearish'],
            'confidence': [80, 70, 60, 50]
        }, index=pd.date_range('2024-03-01', periods=4, freq='5min'))

        probabilities = np.array([0.70, 0.80, 0.50, 0.60])

        filtered = backtester.filter_by_probability(signals_df, probabilities)

        # Only signals with P >= 0.65 should pass
        assert len(filtered) == 2
        assert all(filtered['probability'] >= 0.65)

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_filter_counts_filtered_vs_total(self, mock_file, mock_pickle):
        """Verify filtering counts signals correctly."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester(probability_threshold=0.70)

        signals_df = pd.DataFrame({
            'direction': ['bullish'] * 10
        }, index=pd.date_range('2024-03-01', periods=10, freq='5min'))

        probabilities = np.array(
            [0.8, 0.75, 0.72, 0.6, 0.5, 0.85, 0.9, 0.70, 0.55, 0.45]
        )

        filtered = backtester.filter_by_probability(signals_df, probabilities)

        # 6 signals should pass (>= 0.70)
        assert len(filtered) == 6
        assert len(signals_df) == 10


class TestTripleBarrierLabeling:
    """Test triple-barrier outcome labeling."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_label_take_profit_outcome(self, mock_file, mock_pickle):
        """Verify labeling of take profit outcome."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        # Create bullish signal
        signals_df = pd.DataFrame({
            'direction': ['bullish']
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        # Entry at 2100, take profit at 2110.5 (2100 * 1.005)
        price_data = pd.DataFrame({
            'open': [2100.0, 2100.0, 2100.0, 2100.0],
            'high': [2100.0, 2102.0, 2111.0, 2104.0],  # 3rd bar: 2111 >= 2110.5
            'low': [2099.0, 2099.0, 2099.0, 2099.0],
            'close': [2100.0, 2100.0, 2100.0, 2100.0]
        }, index=pd.date_range('2024-03-01 10:00:00', periods=4, freq='5min'))

        labels = backtester.label_triple_barrier_outcomes(signals_df, price_data)

        # 3rd bar (10:10) has high 2111 which exceeds take profit (2110.5)
        assert labels.iloc[0] == 1

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_label_stop_loss_outcome(self, mock_file, mock_pickle):
        """Verify labeling of stop loss outcome."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        signals_df = pd.DataFrame({
            'direction': ['bullish']
        }, index=[pd.Timestamp('2024-03-01 10:00:00')])

        # Create price data with stop loss hit
        price_data = pd.DataFrame({
            'open': [2100.0] + [2099.0] * 5,
            'high': [2100.0] + [2100.0] * 5,
            'low': [2099.0] + list(range(2098, 2093, -1)),
            'close': [2100.0] + [2099.0] * 5
        }, index=pd.date_range('2024-03-01 10:00:00', periods=6, freq='5min'))

        labels = backtester.label_triple_barrier_outcomes(signals_df, price_data)

        # Stop loss hit
        assert labels.iloc[0] == 0

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_label_multiple_signals(self, mock_file, mock_pickle):
        """Verify labeling of multiple signals."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        signals_df = pd.DataFrame({
            'direction': ['bullish', 'bearish', 'bullish']
        }, index=pd.date_range('2024-03-01', periods=3, freq='5min'))

        price_data = pd.DataFrame({
            'open': [2100.0] * 10,
            'high': [2105.0] * 10,
            'low': [2095.0] * 10,
            'close': [2100.0] * 10
        }, index=pd.date_range('2024-03-01', periods=10, freq='5min'))

        labels = backtester.label_triple_barrier_outcomes(signals_df, price_data)

        assert len(labels) == 3
        assert all((labels == 0) | (labels == 1))


class TestPerformanceCalculation:
    """Test performance metrics calculation."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_calculate_win_rate_all_signals(self, mock_file, mock_pickle):
        """Verify win rate calculation for all signals."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        signals_df = pd.DataFrame({
            'direction': ['bullish'] * 10
        })

        labels = pd.Series([1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

        metrics = backtester.calculate_performance_metrics(signals_df, labels)

        assert metrics['total_signals'] == 10
        assert metrics['wins'] == 6
        assert metrics['losses'] == 4
        assert metrics['win_rate'] == 60.0

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_calculate_win_rate_filtered_signals(self, mock_file, mock_pickle):
        """Verify win rate calculation for filtered signals."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        signals_df = pd.DataFrame({
            'direction': ['bullish'] * 5
        })

        labels = pd.Series([1, 1, 1, 0, 1])

        metrics = backtester.calculate_performance_metrics(signals_df, labels)

        assert metrics['total_signals'] == 5
        assert metrics['win_rate'] == 80.0

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_calculate_improvement_percentage(self, mock_file, mock_pickle):
        """Verify improvement percentage calculation."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        backtester = MLMetaLabelingBacktester()

        metrics_all = {
            'total_signals': 100,
            'wins': 45,
            'losses': 55,
            'win_rate': 45.0
        }

        metrics_filtered = {
            'total_signals': 20,
            'wins': 13,
            'losses': 7,
            'win_rate': 65.0
        }

        comparison = backtester.compare_results(metrics_all, metrics_filtered)

        assert comparison['total_signals'] == 100
        assert comparison['filtered_signals'] == 20
        assert comparison['win_rate_all'] == 45.0
        assert comparison['win_rate_filtered'] == 65.0
        assert comparison['improvement'] == 20.0
        assert abs(comparison['improvement_pct'] - 44.44) < 0.1


class TestEndToEndBacktest:
    """Test end-to-end ML backtest pipeline."""

    @patch('src.research.ml_meta_labeling_backtester.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_ml_backtest_complete_pipeline(self, mock_file, mock_pickle):
        """Verify complete backtest pipeline execution."""
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_pipeline]

        # Mock model predictions
        mock_model.predict_proba = MagicMock(
            return_value=np.array([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
        )

        backtester = MLMetaLabelingBacktester()

        # Create signals
        signals_df = pd.DataFrame({
            'direction': ['bullish', 'bearish', 'bullish'],
            'confidence': [80, 70, 60],
            'mss_detected': [True, True, False],
            'fvg_detected': [True, False, True],
            'sweep_detected': [True, False, False]
        }, index=pd.date_range('2024-03-01', periods=3, freq='5min'))

        # Create price data
        price_data = pd.DataFrame({
            'open': [2100.0] * 20,
            'high': [2105.0] * 20,
            'low': [2095.0] * 20,
            'close': [2100.0] * 20,
            'volume': [1000] * 20
        }, index=pd.date_range('2024-03-01', periods=20, freq='5min'))

        results = backtester.run_ml_backtest(signals_df, price_data)

        assert 'total_signals' in results
        assert 'filtered_signals' in results
        assert 'win_rate_all' in results
        assert 'win_rate_filtered' in results
        assert 'improvement_pct' in results

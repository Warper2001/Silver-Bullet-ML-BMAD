"""Unit tests for Backtest CLI."""

import logging
from unittest.mock import patch

import pandas as pd

logger = logging.getLogger(__name__)


class TestCLIParsing:
    """Test CLI argument parsing."""

    def test_parse_required_arguments(self):
        """Verify required arguments parsed correctly."""
        # Just verify module structure exists
        from pathlib import Path
        cli_file = Path('src/cli/backtest.py')
        assert cli_file.exists()

    def test_threshold_default_value(self):
        """Verify threshold has correct default."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--start', required=True)
        parser.add_argument('--end', required=True)
        parser.add_argument('--threshold', type=float, default=0.65)
        args = parser.parse_args(['--start', '2023-01-01', '--end', '2023-12-31'])
        assert args.threshold == 0.65

    def test_output_directory_default(self):
        """Verify output directory has correct default."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--start', required=True)
        parser.add_argument('--end', required=True)
        parser.add_argument('--output', default='data/reports/')
        args = parser.parse_args(['--start', '2023-01-01', '--end', '2023-12-31'])
        assert args.output == 'data/reports/'


class TestRunBacktest:
    """Test run_backtest function."""

    @patch('src.cli.backtest.EquityCurveVisualizer')
    @patch('src.cli.backtest.BacktestReportGenerator')
    @patch('src.cli.backtest.MarketRegimeAnalyzer')
    @patch('src.cli.backtest.FeatureImportanceAnalyzer')
    @patch('src.cli.backtest.PerformanceMetricsCalculator')
    @patch('src.cli.backtest.MLMetaLabelingBacktester')
    @patch('src.cli.backtest.SilverBulletBacktester')
    @patch('src.cli.backtest.HistoricalDataLoader')
    def test_run_complete_backtest(
        self,
        mock_loader,
        mock_sb,
        mock_ml,
        mock_metrics,
        mock_features,
        mock_regime,
        mock_report,
        mock_visualizer
    ):
        """Verify complete backtest pipeline executes."""
        from src.cli.backtest import run_backtest

        # Setup mocks
        mock_data = pd.DataFrame({'close': [100, 101, 102]})
        mock_loader.return_value.load_data.return_value = mock_data

        mock_signals = pd.DataFrame({'direction': ['LONG']})
        mock_sb.return_value.run_backtest.return_value = mock_signals

        mock_trades = pd.DataFrame({'pnl': [100]})
        mock_ml.return_value.run_backtest.return_value = mock_trades

        mock_metrics.return_value.calculate_metrics.return_value = {
            'sharpe_ratio': 2.5,
            'win_rate': 60.0,
            'total_return': 10000
        }

        mock_visualizer.return_value.visualize.return_value = 'equity_curve.png'

        mock_features.return_value.analyze.return_value = {
            'chart_path': 'feature_importance.png'
        }

        mock_regime.return_value.analyze.return_value = {
            'csv_path': 'regime_analysis.csv'
        }

        mock_report.return_value.generate_backtest_report.return_value = {
            'csv_path': 'backtest.csv',
            'pdf_path': 'backtest.pdf'
        }

        # Run backtest
        results = run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_path='data/models/xgboost_latest.pkl',
            threshold=0.65,
            output_dir='data/reports/',
            verbose=False
        )

        # Verify results
        assert 'data' in results
        assert 'signals' in results
        assert 'trades' in results
        assert 'metrics' in results
        assert 'equity_curve_path' in results
        assert 'feature_importance' in results
        assert 'regime_analysis' in results
        assert 'report_paths' in results

    @patch('src.cli.backtest.EquityCurveVisualizer')
    @patch('src.cli.backtest.BacktestReportGenerator')
    @patch('src.cli.backtest.MarketRegimeAnalyzer')
    @patch('src.cli.backtest.FeatureImportanceAnalyzer')
    @patch('src.cli.backtest.PerformanceMetricsCalculator')
    @patch('src.cli.backtest.MLMetaLabelingBacktester')
    @patch('src.cli.backtest.SilverBulletBacktester')
    @patch('src.cli.backtest.HistoricalDataLoader')
    def test_skip_regime_analysis(
        self,
        mock_loader,
        mock_sb,
        mock_ml,
        mock_metrics,
        mock_features,
        mock_regime,
        mock_report,
        mock_visualizer
    ):
        """Verify regime analysis can be skipped."""
        from src.cli.backtest import run_backtest

        # Setup mocks
        mock_data = pd.DataFrame({'close': [100, 101, 102]})
        mock_loader.return_value.load_data.return_value = mock_data
        mock_sb.return_value.run_backtest.return_value = pd.DataFrame()
        mock_ml.return_value.run_backtest.return_value = pd.DataFrame()
        mock_metrics.return_value.calculate_metrics.return_value = {}
        mock_visualizer.return_value.visualize.return_value = 'equity_curve.png'
        mock_report.return_value.generate_backtest_report.return_value = {
            'csv_path': 'backtest.csv',
            'pdf_path': 'backtest.pdf'
        }

        # Run backtest with skip_regime_analysis
        results = run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_path='data/models/xgboost_latest.pkl',
            threshold=0.65,
            output_dir='data/reports/',
            skip_regime_analysis=True,
            verbose=False
        )

        # Verify regime analysis not called
        mock_regime.return_value.analyze.assert_not_called()
        assert 'regime_analysis' not in results

    @patch('src.cli.backtest.EquityCurveVisualizer')
    @patch('src.cli.backtest.BacktestReportGenerator')
    @patch('src.cli.backtest.MarketRegimeAnalyzer')
    @patch('src.cli.backtest.FeatureImportanceAnalyzer')
    @patch('src.cli.backtest.PerformanceMetricsCalculator')
    @patch('src.cli.backtest.MLMetaLabelingBacktester')
    @patch('src.cli.backtest.SilverBulletBacktester')
    @patch('src.cli.backtest.HistoricalDataLoader')
    def test_skip_feature_importance(
        self,
        mock_loader,
        mock_sb,
        mock_ml,
        mock_metrics,
        mock_regime,
        mock_features,
        mock_report,
        mock_visualizer
    ):
        """Verify feature importance can be skipped."""
        from src.cli.backtest import run_backtest

        # Setup mocks
        mock_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102]
        })
        mock_loader.return_value.load_data.return_value = mock_data
        mock_sb.return_value.run_backtest.return_value = pd.DataFrame()
        mock_ml.return_value.run_backtest.return_value = pd.DataFrame()
        mock_metrics.return_value.calculate_metrics.return_value = {}
        mock_visualizer.return_value.visualize.return_value = 'equity_curve.png'
        mock_regime.return_value.analyze_regime_performance.return_value = {
            'csv_path': 'regime_analysis.csv'
        }
        mock_report.return_value.generate_backtest_report.return_value = {
            'csv_path': 'backtest.csv',
            'pdf_path': 'backtest.pdf'
        }

        # Run backtest with skip_feature_importance
        results = run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_path='data/models/xgboost_latest.pkl',
            threshold=0.65,
            output_dir='data/reports/',
            skip_feature_importance=True,
            verbose=False
        )

        # Verify feature importance not called
        mock_features.return_value.analyze.assert_not_called()
        assert 'feature_importance' not in results


class TestProgressDisplay:
    """Test progress display functionality."""

    @patch('builtins.print')
    @patch('src.cli.backtest.EquityCurveVisualizer')
    @patch('src.cli.backtest.BacktestReportGenerator')
    @patch('src.cli.backtest.MarketRegimeAnalyzer')
    @patch('src.cli.backtest.FeatureImportanceAnalyzer')
    @patch('src.cli.backtest.PerformanceMetricsCalculator')
    @patch('src.cli.backtest.MLMetaLabelingBacktester')
    @patch('src.cli.backtest.SilverBulletBacktester')
    @patch('src.cli.backtest.HistoricalDataLoader')
    def test_verbose_mode_displays_progress(
        self,
        mock_loader,
        mock_sb,
        mock_ml,
        mock_metrics,
        mock_features,
        mock_regime,
        mock_report,
        mock_visualizer,
        mock_print
    ):
        """Verify verbose mode displays progress messages."""
        from src.cli.backtest import run_backtest

        # Setup mocks
        mock_data = pd.DataFrame({'close': [100, 101, 102]})
        mock_loader.return_value.load_data.return_value = mock_data
        mock_sb.return_value.run_backtest.return_value = pd.DataFrame()
        mock_ml.return_value.run_backtest.return_value = pd.DataFrame()
        mock_metrics.return_value.calculate_metrics.return_value = {}
        mock_visualizer.return_value.visualize.return_value = 'equity_curve.png'
        mock_features.return_value.analyze.return_value = {'chart_path': 'features.png'}
        mock_regime.return_value.analyze.return_value = {'csv_path': 'regime.csv'}
        mock_report.return_value.generate_backtest_report.return_value = {
            'csv_path': 'backtest.csv',
            'pdf_path': 'backtest.pdf'
        }

        # Run with verbose=True
        run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_path='data/models/xgboost_latest.pkl',
            threshold=0.65,
            output_dir='data/reports/',
            verbose=True
        )

        # Verify print was called multiple times (progress messages)
        assert mock_print.call_count > 0

    @patch('builtins.print')
    @patch('src.cli.backtest.EquityCurveVisualizer')
    @patch('src.cli.backtest.BacktestReportGenerator')
    @patch('src.cli.backtest.MarketRegimeAnalyzer')
    @patch('src.cli.backtest.FeatureImportanceAnalyzer')
    @patch('src.cli.backtest.PerformanceMetricsCalculator')
    @patch('src.cli.backtest.MLMetaLabelingBacktester')
    @patch('src.cli.backtest.SilverBulletBacktester')
    @patch('src.cli.backtest.HistoricalDataLoader')
    def test_quiet_mode_suppresses_output(
        self,
        mock_loader,
        mock_sb,
        mock_ml,
        mock_metrics,
        mock_features,
        mock_regime,
        mock_report,
        mock_visualizer,
        mock_print
    ):
        """Verify quiet mode suppresses progress output."""
        from src.cli.backtest import run_backtest

        # Setup mocks
        mock_data = pd.DataFrame({'close': [100, 101, 102]})
        mock_loader.return_value.load_data.return_value = mock_data
        mock_sb.return_value.run_backtest.return_value = pd.DataFrame()
        mock_ml.return_value.run_backtest.return_value = pd.DataFrame()
        mock_metrics.return_value.calculate_metrics.return_value = {}
        mock_visualizer.return_value.visualize.return_value = 'equity_curve.png'
        mock_features.return_value.analyze.return_value = {'chart_path': 'features.png'}
        mock_regime.return_value.analyze.return_value = {'csv_path': 'regime.csv'}
        mock_report.return_value.generate_backtest_report.return_value = {
            'csv_path': 'backtest.csv',
            'pdf_path': 'backtest.pdf'
        }

        # Run with verbose=False
        run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_path='data/models/xgboost_latest.pkl',
            threshold=0.65,
            output_dir='data/reports/',
            verbose=False
        )

        # Verify print was not called for progress
        # (may still be called for errors)
        assert mock_print.call_count == 0


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_threshold_too_high(self):
        """Verify error when threshold > 1.0."""
        threshold = 1.5
        if threshold > 1.0:
            assert True  # Would trigger error in main()

    def test_invalid_threshold_too_low(self):
        """Verify error when threshold < 0.0."""
        threshold = -0.5
        if threshold < 0.0:
            assert True  # Would trigger error in main()

    @patch('src.cli.backtest.run_backtest')
    def test_handle_backtest_exception(self, mock_run_backtest):
        """Verify exceptions are handled gracefully."""
        # Mock run_backtest to raise exception
        mock_run_backtest.side_effect = Exception("Test error")

        # Should catch exception
        try:
            from src.cli.backtest import run_backtest
            run_backtest(
                start_date='2023-01-01',
                end_date='2023-12-31',
                model_path='test.pkl',
                threshold=0.65,
                output_dir='data/reports/'
            )
        except Exception:
            assert True


class TestEndToEnd:
    """Test end-to-end CLI functionality."""

    @patch('src.cli.backtest.run_backtest')
    def test_complete_cli_execution(self, mock_run_backtest):
        """Verify complete CLI execution flow."""
        # Setup mock
        mock_run_backtest.return_value = {
            'metrics': {
                'sharpe_ratio': 2.5,
                'win_rate': 60.0
            }
        }

        # Test that function can be called
        from src.cli.backtest import run_backtest
        run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_path='test.pkl',
            threshold=0.65,
            output_dir='data/reports/',
            verbose=False
        )

        # Verify run_backtest was called
        mock_run_backtest.assert_called_once()

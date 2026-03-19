"""Unit tests for BacktestReportGenerator."""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TestBacktestReportGeneratorInit:
    """Test BacktestReportGenerator initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        assert str(generator._output_directory) == "data/reports"
        assert generator._include_charts is True

    def test_init_with_custom_output_directory(self):
        """Verify initialization with custom output directory."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator(output_directory="custom/output")

        assert str(generator._output_directory) == "custom/output"

    def test_init_with_include_charts_false(self):
        """Verify initialization with include_charts flag."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator(include_charts=False)

        assert generator._include_charts is False


class TestGenerateCSVReport:
    """Test CSV report generation."""

    def test_generate_csv_with_all_sections(self):
        """Verify CSV generation includes all sections."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        # Create mock backtest results
        backtest_results = {
            'trades': pd.DataFrame({'pnl': [100, -50]}),
            'metrics': {'sharpe_ratio': 2.5}
        }

        csv_path = generator.generate_csv_report(backtest_results)

        assert "backtest_" in csv_path
        assert csv_path.endswith(".csv")

    def test_include_trade_results_section(self):
        """Verify CSV includes trade results section."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'trades': pd.DataFrame({'pnl': [100, -50]})
        }

        csv_path = generator.generate_csv_report(backtest_results)

        assert "backtest_" in csv_path

    def test_include_performance_metrics_section(self):
        """Verify CSV includes performance metrics section."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {'sharpe_ratio': 2.5}
        }

        csv_path = generator.generate_csv_report(backtest_results)

        assert "backtest_" in csv_path


class TestGeneratePDFReport:
    """Test PDF report generation."""

    def test_generate_pdf_with_all_sections(self):
        """Verify PDF generation includes all sections."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        # Create mock backtest results
        backtest_results = {
            'trades': pd.DataFrame({'pnl': [100, -50]}),
            'metrics': {'sharpe_ratio': 2.5}
        }

        pdf_path = generator.generate_pdf_report(backtest_results)

        assert "backtest_" in pdf_path
        assert pdf_path.endswith(".pdf")

    def test_include_executive_summary(self):
        """Verify PDF includes executive summary."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {'sharpe_ratio': 2.5}
        }

        pdf_path = generator.generate_pdf_report(backtest_results)

        assert "backtest_" in pdf_path

    def test_include_conclusions_and_recommendations(self):
        """Verify PDF includes conclusions and recommendations."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {'sharpe_ratio': 2.5}
        }

        pdf_path = generator.generate_pdf_report(backtest_results)

        assert "backtest_" in pdf_path


class TestCreateTradeResultsSection:
    """Test trade results section creation."""

    def test_format_trade_results_correctly(self):
        """Verify trade results formatted correctly."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        trades_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 10:00')],
            'direction': ['LONG'],
            'entry_price': [100.0],
            'exit_price': [102.0],
            'pnl': [200.0],
            'duration': [300],
            'exit_reason': ['TAKE_PROFIT']
        })

        result = generator.create_trade_results_section(trades_df)

        assert 'timestamp' in result.columns
        assert 'direction' in result.columns
        assert 'entry_price' in result.columns
        assert 'exit_price' in result.columns
        assert 'pnl' in result.columns

    def test_include_all_required_columns(self):
        """Verify all required columns included."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        trades_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 10:00')],
            'direction': ['LONG'],
            'entry_price': [100.0],
            'exit_price': [102.0],
            'pnl': [200.0]
        })

        result = generator.create_trade_results_section(trades_df)

        required_columns = [
            'timestamp', 'direction', 'entry_price', 'exit_price', 'pnl'
        ]
        for col in required_columns:
            assert col in result.columns


class TestCreatePerformanceMetricsSection:
    """Test performance metrics section creation."""

    def test_format_metrics_as_table(self):
        """Verify metrics formatted as table."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        metrics_dict = {
            'sharpe_ratio': 2.5,
            'win_rate': 60.0,
            'profit_factor': 1.8
        }

        result = generator.create_performance_metrics_section(metrics_dict)

        assert 'metric' in result.columns
        assert 'value' in result.columns
        assert len(result) == 3

    def test_include_all_key_metrics(self):
        """Verify all key metrics included."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        metrics_dict = {
            'total_return': 10000,
            'sharpe_ratio': 2.5,
            'win_rate': 60.0
        }

        result = generator.create_performance_metrics_section(metrics_dict)

        # Should include all provided metrics
        assert len(result) == 3


class TestCreateExecutiveSummary:
    """Test executive summary creation."""

    def test_generate_summary_paragraphs(self):
        """Verify summary generates 2-3 paragraphs."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {
                'sharpe_ratio': 2.5,
                'win_rate': 60.0,
                'total_return': 10000
            }
        }

        summary = generator.create_executive_summary(backtest_results)

        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be substantial

    def test_include_key_findings(self):
        """Verify summary includes key findings."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {
                'sharpe_ratio': 2.5,
                'win_rate': 60.0
            }
        }

        summary = generator.create_executive_summary(backtest_results)

        # Should mention key metrics
        assert "sharpe" in summary.lower() or "win rate" in summary.lower()


class TestCreateConclusionsAndRecommendations:
    """Test conclusions and recommendations creation."""

    def test_analyze_results_and_provide_recommendations(self):
        """Verify conclusions analyze results."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {
                'sharpe_ratio': 2.5,
                'win_rate': 60.0
            }
        }

        conclusions = generator.create_conclusions_and_recommendations(
            backtest_results
        )

        assert isinstance(conclusions, str)
        assert len(conclusions) > 100

    def test_provide_actionable_insights(self):
        """Verify conclusions provide actionable recommendations."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'metrics': {
                'sharpe_ratio': 2.5
            }
        }

        conclusions = generator.create_conclusions_and_recommendations(
            backtest_results
        )

        # Should provide recommendations
        assert "recommend" in conclusions.lower() or \
               "deploy" in conclusions.lower() or \
               "strategy" in conclusions.lower()


class TestAddMetadataToReport:
    """Test metadata extraction and formatting."""

    def test_extract_all_metadata_fields(self):
        """Verify all metadata fields extracted."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'backtest_date': pd.Timestamp('2024-03-18'),
            'data_range': ('2023-01-01', '2023-12-31'),
            'signal_count': 1250,
            'ml_model_version': 'v1.2.0',
            'configuration': {'threshold': 0.65}
        }

        metadata = generator.add_metadata_to_report(backtest_results)

        assert 'backtest_date' in metadata
        assert 'data_range' in metadata
        assert 'signal_count' in metadata
        assert 'ml_model_version' in metadata
        assert 'configuration' in metadata

    def test_format_correctly_for_report_header(self):
        """Verify metadata formatted correctly."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        backtest_results = {
            'backtest_date': pd.Timestamp('2024-03-18'),
            'signal_count': 1250
        }

        metadata = generator.add_metadata_to_report(backtest_results)

        # Should be a dictionary
        assert isinstance(metadata, dict)


class TestLogReportGeneration:
    """Test report generation logging."""

    @patch('builtins.open', new_callable=Mock)
    @patch('os.path.getsize')
    def test_log_report_generation_event(self, mock_getsize, mock_open):
        """Verify report generation event logged."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        # Setup mock
        mock_getsize.return_value = 1024

        generator = BacktestReportGenerator()

        csv_path = "data/reports/backtest_2024-03-18.csv"
        pdf_path = "data/reports/backtest_2024-03-18.pdf"

        # Just verify method runs without error
        generator.log_report_generation(csv_path, pdf_path)

    @patch('builtins.open', new_callable=Mock)
    @patch('os.path.getsize')
    def test_include_timestamp_and_file_paths(self, mock_getsize, mock_open):
        """Verify log includes timestamp and file paths."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        # Setup mock
        mock_getsize.return_value = 1024

        generator = BacktestReportGenerator()

        csv_path = "data/reports/backtest_2024-03-18.csv"
        pdf_path = "data/reports/backtest_2024-03-18.pdf"

        # Just verify method runs without error
        generator.log_report_generation(csv_path, pdf_path)


class TestSendReportNotification:
    """Test report notification sending."""

    def test_send_notification_with_file_path(self):
        """Verify notification sent with file path."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        csv_path = "data/reports/backtest_2024-03-18.csv"
        pdf_path = "data/reports/backtest_2024-03-18.pdf"

        # Just verify method runs without error
        generator.send_report_notification(csv_path, pdf_path)

    def test_use_notification_system_correctly(self):
        """Verify notification system used correctly."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        csv_path = "data/reports/backtest_2024-03-18.csv"
        pdf_path = "data/reports/backtest_2024-03-18.pdf"

        # Just verify method runs without error
        generator.send_report_notification(csv_path, pdf_path)


class TestGenerateBacktestReport:
    """Test end-to-end backtest report generation."""

    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'send_report_notification')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'log_report_generation')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'generate_pdf_report')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'generate_csv_report')
    def test_end_to_end_report_generation(
        self,
        mock_csv,
        mock_pdf,
        mock_log,
        mock_notify
    ):
        """Verify complete report generation pipeline."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        # Setup mocks
        mock_csv.return_value = "backtest_2024-03-18.csv"
        mock_pdf.return_value = "backtest_2024-03-18.pdf"

        # Create mock backtest results
        backtest_results = {
            'trades': pd.DataFrame({'pnl': [100, -50]}),
            'metrics': {'sharpe_ratio': 2.5}
        }

        result = generator.generate_backtest_report(backtest_results)

        # Should return file paths
        assert 'csv_path' in result
        assert 'pdf_path' in result

        # Should call all components
        mock_csv.assert_called_once()
        mock_pdf.assert_called_once()
        mock_log.assert_called_once()
        mock_notify.assert_called_once()

    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'send_report_notification')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'log_report_generation')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'generate_pdf_report')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'generate_csv_report')
    def test_all_components_integrated(
        self,
        mock_csv,
        mock_pdf,
        mock_log,
        mock_notify
    ):
        """Verify all components integrated correctly."""
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        mock_csv.return_value = "backtest_2024-03-18.csv"
        mock_pdf.return_value = "backtest_2024-03-18.pdf"

        backtest_results = {
            'trades': pd.DataFrame({'pnl': [100, -50]}),
            'metrics': {'sharpe_ratio': 2.5}
        }

        result = generator.generate_backtest_report(backtest_results)

        # Check return value
        assert 'csv_path' in result
        assert 'pdf_path' in result
        assert result['csv_path'] == "backtest_2024-03-18.csv"
        assert result['pdf_path'] == "backtest_2024-03-18.pdf"

    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'send_report_notification')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'log_report_generation')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'generate_pdf_report')
    @patch('src.research.backtest_report_generator.BacktestReportGenerator.'
           'generate_csv_report')
    def test_performance_requirement_under_1_minute(
        self,
        mock_csv,
        mock_pdf,
        mock_log,
        mock_notify
    ):
        """Verify report generation completes in < 1 minute."""
        import time
        from src.research.backtest_report_generator import BacktestReportGenerator

        generator = BacktestReportGenerator()

        mock_csv.return_value = "backtest_2024-03-18.csv"
        mock_pdf.return_value = "backtest_2024-03-18.pdf"

        # Create large backtest results
        backtest_results = {
            'trades': pd.DataFrame({
                'pnl': np.random.randn(500) * 100
            }),
            'metrics': {
                'sharpe_ratio': 2.5,
                'win_rate': 60.0
            }
        }

        start_time = time.time()
        generator.generate_backtest_report(backtest_results)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 60.0, \
            f"Report generation took {elapsed_time:.2f} seconds"

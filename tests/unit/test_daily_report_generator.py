"""Unit tests for DailyReportGenerator.

Tests daily performance report generation including metric calculations,
PDF generation, CSV export, and scheduling.
"""

from unittest.mock import MagicMock, patch, mock_open

from src.monitoring.daily_report_generator import DailyReportGenerator


class TestDailyReportGeneratorInit:
    """Test DailyReportGenerator initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        generator = DailyReportGenerator()

        assert generator._output_directory == "data/reports"
        assert generator._timezone == "America/New_York"
        assert generator._audit_trail is None
        assert generator._notification_manager is None

    def test_init_with_custom_output_directory(self):
        """Verify initialization with custom output directory."""
        generator = DailyReportGenerator(output_directory="custom/reports")

        assert generator._output_directory == "custom/reports"

    def test_init_with_custom_timezone(self):
        """Verify initialization with custom timezone."""
        generator = DailyReportGenerator(timezone_str="US/Pacific")

        assert generator._timezone == "US/Pacific"

    def test_init_with_audit_trail(self):
        """Verify initialization with audit trail."""
        audit_trail = MagicMock()
        generator = DailyReportGenerator(audit_trail=audit_trail)

        assert generator._audit_trail == audit_trail

    def test_init_with_notification_manager(self):
        """Verify initialization with notification manager."""
        notification_manager = MagicMock()
        generator = DailyReportGenerator(
            notification_manager=notification_manager
        )

        assert generator._notification_manager == notification_manager


class TestCalculateSummaryMetrics:
    """Test _calculate_summary_metrics functionality."""

    def test_calculate_summary_metrics_with_winning_trades(self):
        """Verify calculation with winning trades only."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 100.00},
            {"pnl": 200.00},
            {"pnl": 150.00}
        ]

        summary = generator._calculate_summary_metrics(trades)

        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 3
        assert summary["losing_trades"] == 0
        assert summary["win_rate"] == 1.0
        assert summary["gross_profit"] == 450.00
        assert summary["gross_loss"] == 0.00
        assert summary["net_pnl"] == 450.00
        assert summary["average_win"] == 150.00
        assert summary["average_loss"] == 0.00

    def test_calculate_summary_metrics_with_losing_trades(self):
        """Verify calculation with losing trades only."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": -100.00},
            {"pnl": -200.00},
            {"pnl": -150.00}
        ]

        summary = generator._calculate_summary_metrics(trades)

        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 0
        assert summary["losing_trades"] == 3
        assert summary["win_rate"] == 0.0
        assert summary["gross_profit"] == 0.00
        assert summary["gross_loss"] == -450.00
        assert summary["net_pnl"] == -450.00
        assert summary["average_win"] == 0.00
        assert summary["average_loss"] == -150.00

    def test_calculate_summary_metrics_with_mixed_trades(self):
        """Verify calculation with mixed winning and losing trades."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 100.00},
            {"pnl": -50.00},
            {"pnl": 200.00},
            {"pnl": -75.00},
            {"pnl": 150.00}
        ]

        summary = generator._calculate_summary_metrics(trades)

        assert summary["total_trades"] == 5
        assert summary["winning_trades"] == 3
        assert summary["losing_trades"] == 2
        assert summary["win_rate"] == 0.6
        assert summary["gross_profit"] == 450.00
        assert summary["gross_loss"] == -125.00
        assert summary["net_pnl"] == 325.00
        assert summary["average_win"] == 150.00
        assert summary["average_loss"] == -62.50

    def test_calculate_summary_metrics_empty_list(self):
        """Verify calculation with empty trade list."""
        generator = DailyReportGenerator()

        summary = generator._calculate_summary_metrics([])

        assert summary["total_trades"] == 0
        assert summary["winning_trades"] == 0
        assert summary["losing_trades"] == 0
        assert summary["win_rate"] == 0.0
        assert summary["net_pnl"] == 0.00

    def test_calculate_profit_factor(self):
        """Verify profit factor calculation."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 200.00},
            {"pnl": -100.00},
            {"pnl": 150.00},
            {"pnl": -50.00}
        ]

        summary = generator._calculate_summary_metrics(trades)

        # Profit factor = gross_profit / abs(gross_loss)
        # = 350 / 150 = 2.33
        assert abs(summary["profit_factor"] - 2.33) < 0.01

    def test_calculate_max_drawdown(self):
        """Verify max drawdown calculation."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 100.00},  # Cumulative: 100
            {"pnl": 200.00},  # Cumulative: 300 (peak)
            {"pnl": -150.00},  # Cumulative: 150 (drawdown: -150)
            {"pnl": -200.00},  # Cumulative: -50 (drawdown: -350)
            {"pnl": 50.00}     # Cumulative: 0 (drawdown still -350)
        ]

        summary = generator._calculate_summary_metrics(trades)

        # Max drawdown should be -350
        assert summary["max_drawdown"] == -350.00


class TestCalculateAdvancedMetrics:
    """Test _calculate_advanced_metrics functionality."""

    def test_calculate_sharpe_ratio(self):
        """Verify Sharpe ratio calculation."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 100.00},
            {"pnl": -50.00},
            {"pnl": 200.00},
            {"pnl": -75.00}
        ]

        advanced = generator._calculate_advanced_metrics(trades)

        # Should calculate Sharpe ratio
        assert "sharpe_ratio" in advanced
        assert isinstance(advanced["sharpe_ratio"], float)

    def test_calculate_sortino_ratio(self):
        """Verify Sortino ratio calculation."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 100.00},
            {"pnl": -50.00},
            {"pnl": 200.00},
            {"pnl": -75.00}
        ]

        advanced = generator._calculate_advanced_metrics(trades)

        # Should calculate Sortino ratio
        assert "sortino_ratio" in advanced
        assert isinstance(advanced["sortino_ratio"], float)

    def test_calculate_win_rate_by_confidence(self):
        """Verify win rate grouped by confidence."""
        generator = DailyReportGenerator()

        trades = [
            {"pnl": 100.00, "signal_confidence": "HIGH"},
            {"pnl": -50.00, "signal_confidence": "HIGH"},
            {"pnl": 200.00, "signal_confidence": "HIGH"},
            {"pnl": -75.00, "signal_confidence": "MEDIUM"},
            {"pnl": 150.00, "signal_confidence": "MEDIUM"}
        ]

        advanced = generator._calculate_advanced_metrics(trades)

        # HIGH confidence: 2 wins out of 3 = 0.67
        assert advanced["win_rate_by_confidence"]["HIGH"] == 2/3

        # MEDIUM confidence: 1 win out of 2 = 0.50
        assert advanced["win_rate_by_confidence"]["MEDIUM"] == 0.50

    def test_calculate_win_rate_by_time_window(self):
        """Verify win rate grouped by time window."""
        generator = DailyReportGenerator()

        trades = [
            {
                "pnl": 100.00,
                "entry_time": "2026-03-18T10:00:00Z"  # 10:00 AM (morning)
            },
            {
                "pnl": -50.00,
                "entry_time": "2026-03-18T10:30:00Z"  # 10:30 AM (morning)
            },
            {
                "pnl": 200.00,
                "entry_time": "2026-03-18T14:00:00Z"  # 2:00 PM (afternoon)
            }
        ]

        advanced = generator._calculate_advanced_metrics(trades)

        # Morning window: 1 win out of 2 = 0.50
        assert "09:30-11:00" in advanced["win_rate_by_time_window"]
        assert advanced["win_rate_by_time_window"]["09:30-11:00"] == 0.50

        # Afternoon window: 1 win out of 1 = 1.00
        assert "13:00-16:00" in advanced["win_rate_by_time_window"]
        assert advanced["win_rate_by_time_window"]["13:00-16:00"] == 1.00


class TestGenerateCsvExport:
    """Test _generate_csv_export functionality."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join")
    def test_generate_csv_creates_file(self, mock_join, mock_file):
        """Verify CSV file creation."""
        mock_join.return_value = "data/reports/daily_2026-03-18.csv"

        generator = DailyReportGenerator()

        trades = [
            {
                "trade_id": "TRD-001",
                "entry_time": "2026-03-18T10:00:00Z",
                "exit_time": "2026-03-18T10:30:00Z",
                "direction": "LONG",
                "entry_price": 2185.25,
                "exit_price": 2188.50,
                "quantity": 1,
                "pnl": 325.00,
                "exit_reason": "UPPER_BAR_HIT",
                "signal_confidence": "HIGH",
                "ml_probability": 0.72,
                "hold_time_minutes": 30
            }
        ]

        generator._generate_csv_export("2026-03-18", trades)

        # Verify file was created
        mock_file.assert_called_once()
        mock_join.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join")
    def test_generate_csv_with_multiple_trades(self, mock_join, mock_file):
        """Verify CSV export with multiple trades."""
        mock_join.return_value = "data/reports/daily_2026-03-18.csv"

        generator = DailyReportGenerator()

        trades = [
            {
                "trade_id": "TRD-001",
                "entry_time": "2026-03-18T10:00:00Z",
                "exit_time": "2026-03-18T10:30:00Z",
                "direction": "LONG",
                "entry_price": 2185.25,
                "exit_price": 2188.50,
                "quantity": 1,
                "pnl": 325.00,
                "exit_reason": "UPPER_BAR_HIT",
                "signal_confidence": "HIGH",
                "ml_probability": 0.72,
                "hold_time_minutes": 30
            },
            {
                "trade_id": "TRD-002",
                "entry_time": "2026-03-18T11:00:00Z",
                "exit_time": "2026-03-18T11:20:00Z",
                "direction": "SHORT",
                "entry_price": 2190.00,
                "exit_price": 2187.75,
                "quantity": 1,
                "pnl": 225.00,
                "exit_reason": "LOWER_BAR_HIT",
                "signal_confidence": "MEDIUM",
                "ml_probability": 0.65,
                "hold_time_minutes": 20
            }
        ]

        generator._generate_csv_export("2026-03-18", trades)

        # Verify file was opened for writing
        mock_file.assert_called_once_with(
            "data/reports/daily_2026-03-18.csv",
            'w',
            newline=''
        )


class TestGenerateReport:
    """Test generate_report functionality."""

    @patch.object(DailyReportGenerator, '_generate_csv_export')
    @patch.object(DailyReportGenerator, '_generate_pdf_report')
    @patch.object(DailyReportGenerator, '_calculate_advanced_metrics')
    @patch.object(DailyReportGenerator, '_calculate_summary_metrics')
    @patch.object(DailyReportGenerator, '_load_trades_for_date')
    def test_generate_report_returns_metadata(
        self,
        mock_load_trades,
        mock_summary,
        mock_advanced,
        mock_pdf,
        mock_csv
    ):
        """Verify generate_report returns correct metadata."""
        mock_load_trades.return_value = [
            {"pnl": 100.00, "signal_confidence": "HIGH"}
        ]
        mock_summary.return_value = {
            "total_trades": 1,
            "net_pnl": 100.00
        }
        mock_advanced.return_value = {
            "sharpe_ratio": 1.5
        }
        mock_pdf.return_value = "data/reports/daily_2026-03-18.pdf"
        mock_csv.return_value = "data/reports/daily_2026-03-18.csv"

        generator = DailyReportGenerator()

        result = generator.generate_report("2026-03-18")

        assert result["trade_date"] == "2026-03-18"
        assert result["pdf_path"] == "data/reports/daily_2026-03-18.pdf"
        assert result["csv_path"] == "data/reports/daily_2026-03-18.csv"
        assert "summary_metrics" in result
        assert "advanced_metrics" in result

    @patch.object(DailyReportGenerator, '_generate_csv_export')
    @patch.object(DailyReportGenerator, '_generate_pdf_report')
    @patch.object(DailyReportGenerator, '_calculate_advanced_metrics')
    @patch.object(DailyReportGenerator, '_calculate_summary_metrics')
    @patch.object(DailyReportGenerator, '_load_trades_for_date')
    def test_generate_report_logs_to_audit_trail(
        self,
        mock_load_trades,
        mock_summary,
        mock_advanced,
        mock_pdf,
        mock_csv
    ):
        """Verify generate_report logs to audit trail."""
        mock_load_trades.return_value = [{"pnl": 100.00}]
        mock_summary.return_value = {"total_trades": 1, "net_pnl": 100.00}
        mock_advanced.return_value = {"sharpe_ratio": 1.5}
        mock_pdf.return_value = "data/reports/daily_2026-03-18.pdf"
        mock_csv.return_value = "data/reports/daily_2026-03-18.csv"

        audit_trail = MagicMock()
        generator = DailyReportGenerator(audit_trail=audit_trail)

        generator.generate_report("2026-03-18")

        # Verify audit trail was called
        audit_trail.log_action.assert_called_once()

    @patch.object(DailyReportGenerator, '_generate_csv_export')
    @patch.object(DailyReportGenerator, '_generate_pdf_report')
    @patch.object(DailyReportGenerator, '_calculate_advanced_metrics')
    @patch.object(DailyReportGenerator, '_calculate_summary_metrics')
    @patch.object(DailyReportGenerator, '_load_trades_for_date')
    def test_generate_report_sends_notification(
        self,
        mock_load_trades,
        mock_summary,
        mock_advanced,
        mock_pdf,
        mock_csv
    ):
        """Verify generate_report sends notification."""
        mock_load_trades.return_value = [{"pnl": 100.00}]
        mock_summary.return_value = {"total_trades": 1, "net_pnl": 100.00}
        mock_advanced.return_value = {"sharpe_ratio": 1.5}
        mock_pdf.return_value = "data/reports/daily_2026-03-18.pdf"
        mock_csv.return_value = "data/reports/daily_2026-03-18.csv"

        notification_manager = MagicMock()
        generator = DailyReportGenerator(
            notification_manager=notification_manager
        )

        generator.generate_report("2026-03-18")

        # Verify notification was sent
        notification_manager.send_notification.assert_called_once()


class TestPerformance:
    """Test performance requirements."""

    def test_calculate_summary_metrics_performance(self):
        """Verify summary metrics calculation is fast."""
        generator = DailyReportGenerator()

        # Create 100 trades
        trades = [{"pnl": 100.00 if i % 2 == 0 else -50.00} for i in range(100)]

        import time

        # Measure time to calculate metrics
        start = time.perf_counter()
        for _ in range(10):
            generator._calculate_summary_metrics(trades)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 10

        # Should be < 10ms per calculation
        assert elapsed_ms < 10.0

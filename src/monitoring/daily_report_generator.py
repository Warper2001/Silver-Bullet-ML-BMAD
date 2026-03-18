"""Daily Report Generator for trading performance.

Generates comprehensive daily performance reports with summary metrics,
trade-by-trade breakdown, and advanced analytics.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class DailyReportGenerator:
    """Generate daily performance reports.

    Generates comprehensive daily trading performance reports
    with summary metrics, trade breakdown, and advanced analytics.
    """

    def __init__(
        self,
        output_directory: str = "data/reports",
        timezone_str: str = "America/New_York",
        audit_trail=None,
        notification_manager=None
    ):
        """Initialize daily report generator.

        Args:
            output_directory: Directory to save reports
            timezone_str: Timezone for schedule (default EST)
            audit_trail: Optional audit trail for logging
            notification_manager: Optional notification manager
        """
        self._output_directory = output_directory
        self._timezone = timezone_str
        self._audit_trail = audit_trail
        self._notification_manager = notification_manager
        self._logger = logging.getLogger(__name__)

    def generate_report(self, trade_date: str) -> Dict:
        """Generate daily performance report for given date.

        Args:
            trade_date: Date string in YYYY-MM-DD format

        Returns:
            Report metadata with file paths and metrics
        """
        # Load trades for date
        trades = self._load_trades_for_date(trade_date)

        # Calculate summary metrics
        summary = self._calculate_summary_metrics(trades)

        # Calculate advanced metrics
        advanced = self._calculate_advanced_metrics(trades)

        # Generate PDF report
        pdf_path = self._generate_pdf_report(
            trade_date,
            trades,
            summary,
            advanced
        )

        # Generate CSV export
        csv_path = self._generate_csv_export(trade_date, trades)

        # Log to audit trail
        if self._audit_trail:
            self._audit_trail.log_action(
                "DAILY_REPORT_GENERATED",
                "daily_report_generator",
                "reporting",
                {
                    "trade_date": trade_date,
                    "pdf_path": pdf_path,
                    "csv_path": csv_path,
                    "total_trades": summary["total_trades"],
                    "net_pnl": summary["net_pnl"]
                }
            )

        # Send notification
        if self._notification_manager:
            self._notification_manager.send_notification(
                severity="INFO",
                title="Daily Report Available",
                message="Daily report available: {}".format(pdf_path),
                notification_type="DAILY_REPORT_GENERATED"
            )

        return {
            "trade_date": trade_date,
            "pdf_path": pdf_path,
            "csv_path": csv_path,
            "summary_metrics": summary,
            "advanced_metrics": advanced
        }

    def _load_trades_for_date(self, trade_date: str) -> List[Dict]:
        """Load all trades for given date.

        Args:
            trade_date: Date string in YYYY-MM-DD format

        Returns:
            List of trade dictionaries
        """
        # For now, return empty list
        # In production, this would query audit trail or trade database
        self._logger.info(
            "Loading trades for date: {}".format(trade_date)
        )
        return []

    def _calculate_summary_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate summary metrics from trades.

        Args:
            trades: List of trade dictionaries

        Returns:
            Summary metrics dictionary
        """
        if not trades:
            return self._empty_summary_metrics()

        # Count wins/losses
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        # Calculate P&L
        gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
        gross_loss = sum(t.get("pnl", 0) for t in losing_trades)
        net_pnl = sum(t.get("pnl", 0) for t in trades)

        # Calculate win rate
        win_rate = len(winning_trades) / len(trades)

        # Calculate average win/loss
        average_win = (
            gross_profit / len(winning_trades)
            if winning_trades else 0
        )
        average_loss = (
            gross_loss / len(losing_trades)
            if losing_trades else 0
        )

        # Calculate profit factor
        profit_factor = (
            gross_profit / abs(gross_loss)
            if gross_loss != 0 else float('inf')
        )

        # Calculate drawdown
        max_drawdown = self._calculate_max_drawdown(trades)

        # Calculate excursions (use 0 if not provided)
        max_favorable_excursion = max(
            t.get("max_favorable_excursion", 0) for t in trades
        ) if trades else 0

        max_adverse_excursion = min(
            t.get("max_adverse_excursion", 0) for t in trades
        ) if trades else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": net_pnl,
            "max_drawdown": max_drawdown,
            "max_favorable_excursion": max_favorable_excursion,
            "max_adverse_excursion": max_adverse_excursion,
            "average_win": average_win,
            "average_loss": average_loss,
            "profit_factor": profit_factor
        }

    def _calculate_advanced_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate advanced performance metrics.

        Args:
            trades: List of trade dictionaries

        Returns:
            Advanced metrics dictionary
        """
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(trades)

        # Calculate Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(trades)

        # Calculate win rate by confidence
        win_rate_by_confidence = self._calculate_win_rate_by_confidence(
            trades
        )

        # Calculate win rate by time window
        win_rate_by_time_window = self._calculate_win_rate_by_time_window(
            trades
        )

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate_by_confidence": win_rate_by_confidence,
            "win_rate_by_time_window": win_rate_by_time_window
        }

    def _generate_pdf_report(
        self,
        trade_date: str,
        trades: List[Dict],
        summary: Dict,
        advanced: Dict
    ) -> str:
        """Generate PDF report.

        Args:
            trade_date: Trade date
            trades: List of trades
            summary: Summary metrics
            advanced: Advanced metrics

        Returns:
            Path to generated PDF file
        """
        # Create output directory if needed
        Path(self._output_directory).mkdir(parents=True, exist_ok=True)

        # Generate PDF filename
        pdf_filename = "daily_{}.pdf".format(trade_date)
        pdf_path = os.path.join(self._output_directory, pdf_filename)

        # For now, just create an empty file
        # In production, this would use ReportLab to generate PDF
        with open(pdf_path, 'w') as f:
            f.write("# Daily Report: {}\n".format(trade_date))
            f.write("Total Trades: {}\n".format(summary["total_trades"]))
            f.write("Net P&L: ${:.2f}\n".format(summary["net_pnl"]))

        self._logger.info("Generated PDF report: {}".format(pdf_path))

        return pdf_path

    def _generate_csv_export(
        self,
        trade_date: str,
        trades: List[Dict]
    ) -> str:
        """Generate CSV export of trades.

        Args:
            trade_date: Trade date
            trades: List of trades

        Returns:
            Path to generated CSV file
        """
        # Create output directory if needed
        Path(self._output_directory).mkdir(parents=True, exist_ok=True)

        # Generate CSV filename
        csv_filename = "daily_{}.csv".format(trade_date)
        csv_path = os.path.join(self._output_directory, csv_filename)

        # Write CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "trade_id", "entry_time", "exit_time", "direction",
                "entry_price", "exit_price", "quantity", "pnl",
                "exit_reason", "signal_confidence", "ml_probability",
                "hold_time_minutes"
            ])

            # Write trades
            for trade in trades:
                writer.writerow([
                    trade.get("trade_id", ""),
                    trade.get("entry_time", ""),
                    trade.get("exit_time", ""),
                    trade.get("direction", ""),
                    trade.get("entry_price", ""),
                    trade.get("exit_price", ""),
                    trade.get("quantity", ""),
                    trade.get("pnl", ""),
                    trade.get("exit_reason", ""),
                    trade.get("signal_confidence", ""),
                    trade.get("ml_probability", ""),
                    trade.get("hold_time_minutes", "")
                ])

        self._logger.info("Generated CSV export: {}".format(csv_path))

        return csv_path

    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades.

        Args:
            trades: List of trades in chronological order

        Returns:
            Maximum drawdown value
        """
        if not trades:
            return 0.0

        # Calculate cumulative P&L
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        for trade in trades:
            cumulative_pnl += trade.get("pnl", 0)

            # Update peak
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl

            # Calculate drawdown
            drawdown = cumulative_pnl - peak_pnl

            # Update max drawdown
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trades.

        Args:
            trades: List of trades

        Returns:
            Sharpe ratio value
        """
        if len(trades) < 2:
            return 0.0

        # Calculate returns for each trade
        returns = [t.get("pnl", 0) for t in trades]

        # Calculate mean and std dev
        mean_return = sum(returns) / len(returns)
        std_return = (
            sum((r - mean_return) ** 2 for r in returns) / len(returns)
        ) ** 0.5

        # Calculate Sharpe ratio (assume risk-free rate = 0)
        if std_return == 0:
            return 0.0

        return mean_return / std_return

    def _calculate_sortino_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sortino ratio from trades.

        Args:
            trades: List of trades

        Returns:
            Sortino ratio value
        """
        if len(trades) < 2:
            return 0.0

        # Calculate returns for each trade
        returns = [t.get("pnl", 0) for t in trades]

        # Calculate mean return
        mean_return = sum(returns) / len(returns)

        # Calculate downside deviation (only losing trades)
        losing_returns = [r for r in returns if r < 0]

        if not losing_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_dev = (
            sum(r ** 2 for r in losing_returns) / len(returns)
        ) ** 0.5

        # Calculate Sortino ratio
        if downside_dev == 0:
            return 0.0

        return mean_return / downside_dev

    def _calculate_win_rate_by_confidence(
        self,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate win rate grouped by confidence level.

        Args:
            trades: List of trades

        Returns:
            Dictionary mapping confidence level to win rate
        """
        confidence_groups = {
            "HIGH": [],
            "MEDIUM": [],
            "LOW": []
        }

        # Group trades by confidence
        for trade in trades:
            confidence = trade.get("signal_confidence", "MEDIUM")
            if confidence not in confidence_groups:
                confidence = "MEDIUM"
            confidence_groups[confidence].append(trade)

        # Calculate win rate for each group
        win_rates = {}
        for confidence, group_trades in confidence_groups.items():
            if not group_trades:
                win_rates[confidence] = 0.0
                continue

            winning_trades = sum(
                1 for t in group_trades if t.get("pnl", 0) > 0
            )
            win_rates[confidence] = winning_trades / len(group_trades)

        return win_rates

    def _calculate_win_rate_by_time_window(
        self,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate win rate grouped by time window.

        Args:
            trades: List of trades

        Returns:
            Dictionary mapping time window to win rate
        """
        # Define time windows
        time_windows = {
            "09:30-11:00": [],  # Morning
            "11:00-13:00": [],  # Midday
            "13:00-16:00": []   # Afternoon
        }

        # Group trades by entry time window
        for trade in trades:
            entry_time_str = trade.get("entry_time", "")
            if not entry_time_str:
                continue

            try:
                entry_time = datetime.fromisoformat(entry_time_str)
                hour = entry_time.hour
                minute = entry_time.minute
                time_decimal = hour + minute / 60

                if 9.5 <= time_decimal < 11.0:
                    time_windows["09:30-11:00"].append(trade)
                elif 11.0 <= time_decimal < 13.0:
                    time_windows["11:00-13:00"].append(trade)
                elif 13.0 <= time_decimal < 16.0:
                    time_windows["13:00-16:00"].append(trade)
            except (ValueError, AttributeError):
                # Skip trades with invalid time
                continue

        # Calculate win rate for each window
        win_rates = {}
        for window, window_trades in time_windows.items():
            if not window_trades:
                win_rates[window] = 0.0
                continue

            winning_trades = sum(
                1 for t in window_trades if t.get("pnl", 0) > 0
            )
            win_rates[window] = winning_trades / len(window_trades)

        return win_rates

    def _empty_summary_metrics(self) -> Dict:
        """Return empty summary metrics.

        Returns:
            Empty summary metrics dictionary
        """
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_favorable_excursion": 0.0,
            "max_adverse_excursion": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0
        }

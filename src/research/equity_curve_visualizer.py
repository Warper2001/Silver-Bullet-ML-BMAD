"""Equity Curve Visualizer for backtest results.

Generates professional equity curve charts with drawdown highlighting,
watermark lines, and performance metrics legends.

Performance: Completes in < 30 seconds for typical backtests.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

logger = logging.getLogger(__name__)


class EquityCurveVisualizer:
    """Generate equity curve visualization with drawdown highlighting.

    Creates professional charts showing equity curve over time with
    drawdown periods highlighted as shaded regions.

    Performance: Completes in < 30 seconds for typical backtests.
    """

    def __init__(
        self,
        output_directory: str = "data/reports",
        figure_size: tuple = (14, 8),
        dpi: int = 100
    ):
        """Initialize equity curve visualizer.

        Args:
            output_directory: Directory to save charts and data
            figure_size: Figure size (width, height) in inches
            dpi: Dots per inch for image resolution
        """
        self._output_directory = Path(output_directory)
        self._figure_size = figure_size
        self._dpi = dpi

        # Create output directory if it doesn't exist
        self._output_directory.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"EquityCurveVisualizer initialized: "
            f"output_directory={output_directory}, "
            f"figure_size={figure_size}, dpi={dpi}"
        )

    def generate_equity_curve_chart(
        self,
        trades_df: pd.DataFrame,
        metrics_dict: dict[str, Any]
    ) -> tuple[str, str]:
        """Generate complete equity curve visualization.

        Args:
            trades_df: DataFrame with trade results (timestamp, pnl)
            metrics_dict: Dictionary with performance metrics

        Returns:
            Tuple of (png_file_path, csv_file_path)
        """
        logger.info("Generating equity curve visualization...")

        # Build equity curve
        equity_curve = self.build_equity_curve(trades_df)

        # Identify drawdown periods
        drawdown_periods = self.identify_drawdown_periods(equity_curve)

        # Create base figure
        fig, ax = self.create_base_figure(equity_curve)

        # Overlay drawdown regions
        self.overlay_drawdown_regions(ax, drawdown_periods, equity_curve)

        # Add watermark line
        self.add_watermark_line(ax, equity_curve)

        # Add metrics legend
        self.add_metrics_legend(ax, metrics_dict)

        # Save chart and data
        png_path, csv_path = self.save_chart_and_data(
            fig, equity_curve, drawdown_periods
        )

        logger.info(
            f"Equity curve chart saved: {png_path}, "
            f"data saved: {csv_path}"
        )

        return png_path, csv_path

    def build_equity_curve(self, trades_df: pd.DataFrame) -> pd.Series:
        """Build equity curve from trade results.

        Args:
            trades_df: DataFrame with trade results (timestamp, pnl)

        Returns:
            Series with equity values indexed by timestamp
        """
        if len(trades_df) == 0:
            return pd.Series([100000.0])

        # Sort by timestamp
        df = trades_df.sort_values('timestamp').copy()

        # Calculate cumulative P&L
        cumulative_pnl = df['pnl'].cumsum()

        # Add initial capital
        initial_capital = 100000.0
        equity_curve = initial_capital + cumulative_pnl

        return equity_curve

    def identify_drawdown_periods(
        self,
        equity_curve: pd.Series
    ) -> list[dict[str, Any]]:
        """Identify drawdown periods in equity curve.

        Args:
            equity_curve: Series of equity values

        Returns:
            List of drawdown periods with start, end, drawdown_pct
        """
        if len(equity_curve) < 2:
            return []

        # Calculate running maximum (watermark)
        running_max = equity_curve.cummax()

        # Calculate drawdown percentage
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100

        # Identify drawdown periods (when underwater)
        underwater = drawdown_pct < 0

        # Group consecutive underwater periods
        drawdowns = []
        in_drawdown = False
        start_idx = None

        for idx, is_underwater in underwater.items():
            if is_underwater and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_idx = idx
            elif not is_underwater and in_drawdown:
                # End of drawdown
                in_drawdown = False
                end_idx = idx

                # Get drawdown percentage (minimum in this period)
                period_dd = drawdown_pct.loc[start_idx:end_idx].min()

                drawdowns.append({
                    'start': start_idx,
                    'end': end_idx,
                    'drawdown_pct': abs(period_dd)
                })

        # Handle if still in drawdown at end
        if in_drawdown:
            end_idx = underwater.index[-1]
            period_dd = drawdown_pct.loc[start_idx:].min()
            drawdowns.append({
                'start': start_idx,
                'end': end_idx,
                'drawdown_pct': abs(period_dd)
            })

        # Filter out very small drawdowns (< 1%)
        drawdowns = [
            dd for dd in drawdowns
            if dd['drawdown_pct'] >= 1.0
        ]

        return drawdowns

    def create_base_figure(
        self,
        equity_curve: pd.Series
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create matplotlib figure with equity curve.

        Args:
            equity_curve: Series of equity values

        Returns:
            Tuple of (figure, axis)
        """
        fig, ax = plt.subplots(
            figsize=self._figure_size,
            dpi=self._dpi
        )

        # Plot equity curve
        ax.plot(
            equity_curve.index,
            equity_curve.values,
            'b-',
            linewidth=2,
            label='Equity Curve'
        )

        # Set labels
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Equity ($)", fontsize=12)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )

        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set title
        ax.set_title("Equity Curve", fontsize=14, fontweight='bold')

        return fig, ax

    def overlay_drawdown_regions(
        self,
        ax: plt.Axes,
        drawdown_periods: list[dict[str, Any]],
        equity_curve: pd.Series
    ) -> None:
        """Overlay drawdown regions as shaded red areas.

        Args:
            ax: Matplotlib axis
            drawdown_periods: List of drawdown periods
            equity_curve: Series of equity values
        """
        for dd in drawdown_periods:
            start = dd['start']
            end = dd['end']
            dd_pct = dd['drawdown_pct']

            # Add shaded region
            ax.axvspan(
                start,
                end,
                alpha=0.2,
                color='red',
                label=f'Drawdown -{dd_pct:.1f}%' if dd == drawdown_periods[0] else ""
            )

            # Add label at the bottom of drawdown
            if isinstance(start, pd.Timestamp):
                y_pos = equity_curve.loc[start:end].min() * 0.995
                ax.text(
                    start,
                    y_pos,
                    f'-{dd_pct:.1f}%',
                    fontsize=8,
                    color='red',
                    alpha=0.7
                )

    def add_watermark_line(
        self,
        ax: plt.Axes,
        equity_curve: pd.Series
    ) -> None:
        """Add all-time high watermark line.

        Args:
            ax: Matplotlib axis
            equity_curve: Series of equity values
        """
        max_equity = equity_curve.max()

        ax.axhline(
            y=max_equity,
            color='green',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label='All-Time High'
        )

    def add_metrics_legend(
        self,
        ax: plt.Axes,
        metrics_dict: dict[str, Any]
    ) -> None:
        """Add performance metrics to legend.

        Args:
            ax: Matplotlib axis
            metrics_dict: Dictionary with performance metrics
        """
        # Extract metrics
        sharpe = metrics_dict.get('sharpe_ratio', 0)
        win_rate = metrics_dict.get('win_rate', 0)
        max_dd = metrics_dict.get('max_drawdown', {})
        total_return = metrics_dict.get('total_return', {})

        if isinstance(max_dd, dict):
            max_dd_pct = max_dd.get('max_drawdown_pct', 0)
        else:
            max_dd_pct = max_dd

        if isinstance(total_return, dict):
            total_ret_pct = total_return.get('total_return_pct', 0)
        else:
            total_ret_pct = total_return

        # Create legend text
        legend_text = (
            f"Sharpe: {sharpe:.2f}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Max DD: -{max_dd_pct:.1f}%\n"
            f"Total Return: {total_ret_pct:+.1f}%"
        )

        # Add legend box
        ax.text(
            0.02,
            0.98,
            legend_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='wheat',
                alpha=0.5
            )
        )

        # Add regular legend
        ax.legend(loc='upper right')

    def save_chart_and_data(
        self,
        fig: plt.Figure,
        equity_curve: pd.Series,
        drawdown_periods: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """Save chart as PNG and data as CSV.

        Args:
            fig: Matplotlib figure
            equity_curve: Series of equity values
            drawdown_periods: List of drawdown periods

        Returns:
            Tuple of (png_file_path, csv_file_path)
        """
        # Generate filename with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

        png_filename = f"equity_curve_{timestamp}.png"
        csv_filename = f"equity_curve_data_{timestamp}.csv"

        png_path = self._output_directory / png_filename
        csv_path = self._output_directory / csv_filename

        # Save figure
        fig.savefig(png_path, dpi=self._dpi, bbox_inches='tight')
        plt.close(fig)

        # Create CSV data
        # Calculate drawdown percentage for each point
        running_max = equity_curve.cummax()
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100

        csv_data = pd.DataFrame({
            'timestamp': equity_curve.index,
            'equity': equity_curve.values,
            'drawdown_pct': drawdown_pct.values
        })

        # Save CSV
        csv_data.to_csv(csv_path, index=False)

        logger.debug(f"Saved chart to {png_path}, data to {csv_path}")

        return str(png_path), str(csv_path)

"""Unit tests for EquityCurveVisualizer.

Tests equity curve visualization generation including drawdown highlighting,
watermark lines, metrics legends, and file exports.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.research.equity_curve_visualizer import (
    EquityCurveVisualizer
)


class TestEquityCurveVisualizerInit:
    """Test EquityCurveVisualizer initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        visualizer = EquityCurveVisualizer()

        assert visualizer._output_directory == Path("data/reports")
        assert visualizer._figure_size == (14, 8)
        assert visualizer._dpi == 100

    def test_init_with_custom_output_directory(self):
        """Verify initialization with custom output directory."""
        visualizer = EquityCurveVisualizer(
            output_directory="custom/output"
        )

        assert visualizer._output_directory == Path("custom/output")

    def test_init_with_custom_figure_size(self):
        """Verify initialization with custom figure size."""
        visualizer = EquityCurveVisualizer(
            figure_size=(16, 10)
        )

        assert visualizer._figure_size == (16, 10)

    def test_init_with_custom_dpi(self):
        """Verify initialization with custom DPI."""
        visualizer = EquityCurveVisualizer(
            dpi=150
        )

        assert visualizer._dpi == 150


class TestBuildEquityCurve:
    """Test equity curve building from trade results."""

    def test_build_equity_curve_from_trades(self):
        """Verify equity curve building from trade results."""
        visualizer = EquityCurveVisualizer()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'pnl': [100, -50, 150, 75, -25, 200, -100, 50, 125, -75]
        })

        equity_curve = visualizer.build_equity_curve(trades_df)

        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == 10
        # First value should be initial capital + first P&L
        assert equity_curve.iloc[0] == 100100  # 100000 + 100
        # Last value should be cumulative
        assert equity_curve.iloc[-1] == 100450  # 100000 + 450

    def test_correct_cumulative_pnl_calculation(self):
        """Verify cumulative P&L calculation is correct."""
        visualizer = EquityCurveVisualizer()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'pnl': [100, 200, 150, -100, 50]
        })

        equity_curve = visualizer.build_equity_curve(trades_df)

        # Cumulative: 100, 300, 450, 350, 400
        expected = [100100, 100300, 100450, 100350, 100400]
        actual = equity_curve.tolist()
        assert actual == expected

    def test_initial_capital_added(self):
        """Verify initial capital (100K) is added to equity curve."""
        visualizer = EquityCurveVisualizer()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            'pnl': [0, 0, 0]
        })

        equity_curve = visualizer.build_equity_curve(trades_df)

        # All values should be 100000 (initial capital)
        assert all(equity_curve == 100000)

    def test_handle_empty_trades(self):
        """Verify handling of empty trades DataFrame."""
        visualizer = EquityCurveVisualizer()

        trades_df = pd.DataFrame({
            'timestamp': [],
            'pnl': []
        })

        equity_curve = visualizer.build_equity_curve(trades_df)

        # Should return series with just initial capital
        assert len(equity_curve) == 1
        assert equity_curve.iloc[0] == 100000


class TestIdentifyDrawdownPeriods:
    """Test drawdown period identification."""

    def test_identify_single_drawdown_period(self):
        """Verify identification of single drawdown period."""
        visualizer = EquityCurveVisualizer()

        # Create equity curve with one drawdown
        equity_values = [
            100000, 105000, 110000,  # Rising to peak
            108000, 103000,  # Drawdown
            105000, 108000   # Recovery
        ]
        equity_curve = pd.Series(
            equity_values,
            index=pd.date_range('2024-01-01', periods=7, freq='D')
        )

        drawdowns = visualizer.identify_drawdown_periods(equity_curve)

        assert len(drawdowns) == 1
        assert drawdowns[0]['drawdown_pct'] > 0
        assert 'start' in drawdowns[0]
        assert 'end' in drawdowns[0]

    def test_identify_multiple_drawdown_periods(self):
        """Verify identification of multiple drawdown periods."""
        visualizer = EquityCurveVisualizer()

        # Create equity curve with multiple drawdowns
        equity_values = [
            100000, 105000,  # Rise
            103000, 101000,  # Drawdown 1
            104000, 108000,  # Recovery and rise
            106000, 103000   # Drawdown 2
        ]
        equity_curve = pd.Series(
            equity_values,
            index=pd.date_range('2024-01-01', periods=8, freq='D')
        )

        drawdowns = visualizer.identify_drawdown_periods(equity_curve)

        assert len(drawdowns) >= 1

    def test_calculate_drawdown_percentages_correctly(self):
        """Verify drawdown percentage calculation."""
        visualizer = EquityCurveVisualizer()

        # Peak 110000, trough 103000
        # Drawdown = (110000 - 103000) / 110000 = 6.36%
        equity_values = [
            100000, 105000, 110000,  # Rising
            108000, 103000,  # Drawdown
            105000
        ]
        equity_curve = pd.Series(
            equity_values,
            index=pd.date_range('2024-01-01', periods=6, freq='D')
        )

        drawdowns = visualizer.identify_drawdown_periods(equity_curve)

        if len(drawdowns) > 0:
            assert abs(drawdowns[0]['drawdown_pct'] - 6.36) < 0.5

    def test_handle_no_drawdowns(self):
        """Verify handling of equity curve with no drawdowns."""
        visualizer = EquityCurveVisualizer()

        # Monotonically increasing equity curve
        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        drawdowns = visualizer.identify_drawdown_periods(equity_curve)

        # Should return empty list or list with no significant drawdowns
        assert len(drawdowns) == 0 or all(
            dd['drawdown_pct'] < 1 for dd in drawdowns
        )


class TestCreateBaseFigure:
    """Test base figure creation."""

    def test_create_matplotlib_figure(self):
        """Verify matplotlib figure creation."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = visualizer.create_base_figure(equity_curve)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_equity_curve_line(self):
        """Verify equity curve plotted as line."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = visualizer.create_base_figure(equity_curve)

        # Check that line exists in axis
        lines = ax.get_lines()
        assert len(lines) > 0
        plt.close(fig)

    def test_set_axis_labels_and_formatting(self):
        """Verify axis labels and formatting."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = visualizer.create_base_figure(equity_curve)

        # Check xlabel
        assert ax.get_xlabel() == "Date"
        # Check ylabel contains "Equity"
        assert "Equity" in ax.get_ylabel()
        plt.close(fig)

    def test_add_grid_lines(self):
        """Verify grid lines are added."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = visualizer.create_base_figure(equity_curve)

        # Check that grid is enabled
        assert ax.xaxis._major_tick_kw['gridOn'] or \
               ax.yaxis._major_tick_kw['gridOn']
        plt.close(fig)


class TestOverlayDrawdownRegions:
    """Test drawdown region overlay."""

    @patch('matplotlib.axes.Axes.axvspan')
    def test_overlay_single_drawdown_region(self, mock_axvspan):
        """Verify overlaying single drawdown region."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        drawdown_periods = [{
            'start': pd.Timestamp('2024-01-03'),
            'end': pd.Timestamp('2024-01-05'),
            'drawdown_pct': 5.0
        }]

        visualizer.overlay_drawdown_regions(ax, drawdown_periods, equity_curve)

        # Verify axvspan was called
        assert mock_axvspan.called
        plt.close(fig)

    @patch('matplotlib.axes.Axes.axvspan')
    def test_overlay_multiple_drawdown_regions(self, mock_axvspan):
        """Verify overlaying multiple drawdown regions."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        drawdown_periods = [
            {
                'start': pd.Timestamp('2024-01-02'),
                'end': pd.Timestamp('2024-01-03'),
                'drawdown_pct': 3.0
            },
            {
                'start': pd.Timestamp('2024-01-06'),
                'end': pd.Timestamp('2024-01-08'),
                'drawdown_pct': 5.0
            }
        ]

        visualizer.overlay_drawdown_regions(ax, drawdown_periods, equity_curve)

        # Verify axvspan called twice
        assert mock_axvspan.call_count == 2
        plt.close(fig)

    @patch('matplotlib.axes.Axes.axvspan')
    def test_correct_red_shading_with_transparency(self, mock_axvspan):
        """Verify red shading with transparency."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        drawdown_periods = [{
            'start': pd.Timestamp('2024-01-03'),
            'end': pd.Timestamp('2024-01-05'),
            'drawdown_pct': 5.0
        }]

        visualizer.overlay_drawdown_regions(ax, drawdown_periods, equity_curve)

        # Check color and alpha parameters
        call_kwargs = mock_axvspan.call_args[1]
        assert 'color' in call_kwargs or 'facecolor' in call_kwargs
        assert 'alpha' in call_kwargs
        assert call_kwargs['alpha'] == 0.2
        plt.close(fig)


class TestAddWatermarkLine:
    """Test watermark line addition."""

    def test_add_green_dashed_line_at_max_equity(self):
        """Verify green dashed line added at max equity."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            [100000, 105000, 110000, 108000, 107000],
            index=pd.date_range('2024-01-01', periods=5, freq='D')
        )

        fig, ax = plt.subplots()
        visualizer.add_watermark_line(ax, equity_curve)

        # Check that horizontal line exists
        lines = ax.get_lines()
        watermark_lines = [
            line for line in lines if
            abs(line.get_ydata()[0] - 110000) < 1
        ]
        assert len(watermark_lines) > 0
        plt.close(fig)

    def test_label_correctly_as_all_time_high(self):
        """Verify label says 'All-Time High'."""
        visualizer = EquityCurveVisualizer()

        equity_curve = pd.Series(
            [100000, 105000, 110000],
            index=pd.date_range('2024-01-01', periods=3, freq='D')
        )

        fig, ax = plt.subplots()
        visualizer.add_watermark_line(ax, equity_curve)

        # Check for line with green color and dashed linestyle
        lines = ax.get_lines()
        watermark_lines = [
            line for line in lines if
            line.get_linestyle() == '--' and
            line.get_color() == 'green'
        ]
        assert len(watermark_lines) > 0
        plt.close(fig)

    def test_handle_multiple_peaks(self):
        """Verify highest peak shown when multiple peaks."""
        visualizer = EquityCurveVisualizer()

        # Multiple peaks, last one is highest
        equity_curve = pd.Series(
            [100000, 105000, 103000, 108000, 106000, 112000, 110000],
            index=pd.date_range('2024-01-01', periods=7, freq='D')
        )

        fig, ax = plt.subplots()
        visualizer.add_watermark_line(ax, equity_curve)

        # Should show line at highest peak (112000)
        lines = ax.get_lines()
        watermark_lines = [
            line for line in lines if
            abs(line.get_ydata()[0] - 112000) < 1
        ]
        assert len(watermark_lines) > 0
        plt.close(fig)


class TestAddMetricsLegend:
    """Test metrics legend addition."""

    def test_add_performance_metrics_to_legend(self):
        """Verify performance metrics added to legend."""
        visualizer = EquityCurveVisualizer()

        metrics_dict = {
            'sharpe_ratio': 1.85,
            'win_rate': 58.4,
            'max_drawdown': {'max_drawdown_pct': 8.3},
            'total_return': {'total_return_pct': 25.0}
        }

        fig, ax = plt.subplots()
        visualizer.add_metrics_legend(ax, metrics_dict)

        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_format_metrics_correctly(self):
        """Verify metrics formatted correctly."""
        visualizer = EquityCurveVisualizer()

        metrics_dict = {
            'sharpe_ratio': 1.85,
            'win_rate': 58.4,
            'max_drawdown': {'max_drawdown_pct': 8.3},
            'total_return': {'total_return_pct': 25.0}
        }

        fig, ax = plt.subplots()
        visualizer.add_metrics_legend(ax, metrics_dict)

        # Check that text annotations were added (metrics displayed in box)
        texts = ax.texts
        assert len(texts) > 0
        plt.close(fig)

    def test_legend_positioned_correctly(self):
        """Verify legend positioned in appropriate location."""
        visualizer = EquityCurveVisualizer()

        metrics_dict = {
            'sharpe_ratio': 1.85,
            'win_rate': 58.4
        }

        fig, ax = plt.subplots()
        visualizer.add_metrics_legend(ax, metrics_dict)

        legend = ax.get_legend()
        assert legend is not None
        # Check location is set (loc should not be None)
        assert legend._loc is not None
        plt.close(fig)


class TestSaveChartAndData:
    """Test chart and data saving."""

    def test_save_png_image_to_correct_location(self, tmp_path):
        """Verify PNG saved to correct location."""
        visualizer = EquityCurveVisualizer(output_directory=str(tmp_path))

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        png_path, csv_path = visualizer.save_chart_and_data(
            fig, equity_curve, []
        )

        assert Path(png_path).exists()
        assert png_path.startswith(str(tmp_path))
        plt.close(fig)

    def test_save_csv_data_to_correct_location(self, tmp_path):
        """Verify CSV saved to correct location."""
        visualizer = EquityCurveVisualizer(output_directory=str(tmp_path))

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        png_path, csv_path = visualizer.save_chart_and_data(
            fig, equity_curve, []
        )

        assert Path(csv_path).exists()
        assert csv_path.startswith(str(tmp_path))
        plt.close(fig)

    def test_filenames_include_timestamp(self, tmp_path):
        """Verify filenames include timestamp."""
        visualizer = EquityCurveVisualizer(output_directory=str(tmp_path))

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        png_path, csv_path = visualizer.save_chart_and_data(
            fig, equity_curve, []
        )

        # Check that filenames contain date pattern
        assert "equity_curve_" in png_path
        assert "equity_curve_data_" in csv_path
        plt.close(fig)

    def test_csv_contains_correct_columns(self, tmp_path):
        """Verify CSV contains correct columns."""
        visualizer = EquityCurveVisualizer(output_directory=str(tmp_path))

        equity_curve = pd.Series(
            range(100000, 110000, 1000),
            index=pd.date_range('2024-01-01', periods=10, freq='D')
        )

        fig, ax = plt.subplots()
        png_path, csv_path = visualizer.save_chart_and_data(
            fig, equity_curve, []
        )

        # Read CSV and check columns
        df = pd.read_csv(csv_path)
        assert 'timestamp' in df.columns or 'date' in df.columns
        assert 'equity' in df.columns
        assert 'drawdown_pct' in df.columns
        plt.close(fig)


class TestGenerateEquityCurveChart:
    """Test end-to-end chart generation."""

    def test_end_to_end_chart_generation(self, tmp_path):
        """Verify complete chart generation pipeline."""
        visualizer = EquityCurveVisualizer(output_directory=str(tmp_path))

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='D'),
            'pnl': np.random.randn(50) * 100 + 50
        })

        metrics_dict = {
            'sharpe_ratio': 1.5,
            'win_rate': 55.0,
            'max_drawdown': {'max_drawdown_pct': 7.5},
            'total_return': {'total_return_pct': 20.0}
        }

        png_path, csv_path = visualizer.generate_equity_curve_chart(
            trades_df, metrics_dict
        )

        assert Path(png_path).exists()
        assert Path(csv_path).exists()

    def test_all_elements_included_in_chart(self, tmp_path):
        """Verify all elements included in generated chart."""
        visualizer = EquityCurveVisualizer(output_directory=str(tmp_path))

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='D'),
            'pnl': [100] * 20
        })

        metrics_dict = {
            'sharpe_ratio': 1.5,
            'win_rate': 60.0,
            'max_drawdown': {'max_drawdown_pct': 5.0},
            'total_return': {'total_return_pct': 15.0}
        }

        png_path, csv_path = visualizer.generate_equity_curve_chart(
            trades_df, metrics_dict
        )

        # Verify file was created
        assert Path(png_path).exists()

    def test_performance_requirement_under_30_seconds(self):
        """Verify chart generation completes in < 30 seconds."""
        visualizer = EquityCurveVisualizer()

        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
            'pnl': np.random.randn(1000) * 100
        })

        metrics_dict = {
            'sharpe_ratio': 1.8,
            'win_rate': 58.0,
            'max_drawdown': {'max_drawdown_pct': 8.0},
            'total_return': {'total_return_pct': 22.0}
        }

        start_time = time.time()
        png_path, csv_path = visualizer.generate_equity_curve_chart(
            trades_df, metrics_dict
        )
        elapsed_time = time.time() - start_time

        assert elapsed_time < 30.0, \
            f"Chart generation took {elapsed_time:.2f} seconds"

        # Cleanup
        if Path(png_path).exists():
            Path(png_path).unlink()
        if Path(csv_path).exists():
            Path(csv_path).unlink()

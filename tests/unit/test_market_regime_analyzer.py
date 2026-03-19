"""Unit tests for MarketRegimeAnalyzer."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TestMarketRegimeAnalyzerInit:
    """Test MarketRegimeAnalyzer initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        assert analyzer._adx_trending_threshold == 25.0
        assert analyzer._atr_volatile_threshold == 20.0
        assert analyzer._adx_period == 14
        assert analyzer._atr_period == 14
        assert str(analyzer._output_directory) == "data/reports"

    def test_init_with_custom_adx_threshold(self):
        """Verify initialization with custom ADX threshold."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(adx_trending_threshold=30.0)

        assert analyzer._adx_trending_threshold == 30.0
        assert analyzer._atr_volatile_threshold == 20.0

    def test_init_with_custom_atr_threshold(self):
        """Verify initialization with custom ATR threshold."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(atr_volatile_threshold=25.0)

        assert analyzer._adx_trending_threshold == 25.0
        assert analyzer._atr_volatile_threshold == 25.0

    def test_init_with_custom_adx_atr_periods(self):
        """Verify initialization with custom ADX/ATR periods."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(adx_period=20, atr_period=10)

        assert analyzer._adx_period == 20
        assert analyzer._atr_period == 10

    def test_init_with_custom_output_directory(self):
        """Verify initialization with custom output directory."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(output_directory="custom/output")

        assert str(analyzer._output_directory) == "custom/output"


class TestCalculateADX:
    """Test ADX calculation."""

    def test_calculate_adx_for_trending_market(self):
        """Verify ADX calculation for trending market."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create trending market data (consistent upward movement)
        np.random.seed(42)
        n = 100
        price_data = pd.DataFrame({
            'high': np.linspace(100, 120, n) + np.random.randn(n) * 0.5,
            'low': np.linspace(99, 119, n) + np.random.randn(n) * 0.5,
            'close': np.linspace(99.5, 119.5, n) + np.random.randn(n) * 0.5
        })

        adx = analyzer.calculate_adx(price_data, period=14)

        # ADX should be calculated and non-NaN after warmup period
        assert adx is not None
        assert len(adx) == len(price_data)
        # Trending market should have ADX > 25
        assert adx.iloc[-1] > 25

    def test_calculate_adx_for_ranging_market(self):
        """Verify ADX calculation for ranging market."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create ranging market data (oscillating around mean)
        np.random.seed(42)
        n = 100
        price_data = pd.DataFrame({
            'high': (100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2 +
                     np.random.randn(n) * 0.5),
            'low': (99 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2 +
                    np.random.randn(n) * 0.5),
            'close': (99.5 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2 +
                      np.random.randn(n) * 0.5)
        })

        adx = analyzer.calculate_adx(price_data, period=14)

        # ADX should be calculated
        assert adx is not None
        assert len(adx) == len(price_data)
        # Ranging market should have ADX < 25
        assert adx.iloc[-1] < 25

    def test_calculate_adx_handles_insufficient_data(self):
        """Verify ADX calculation handles insufficient data."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create insufficient data (less than period)
        price_data = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [99.5, 100.5, 101.5]
        })

        adx = analyzer.calculate_adx(price_data, period=14)

        # Should still calculate but with limited data
        assert adx is not None
        assert len(adx) == len(price_data)


class TestCalculateATR:
    """Test ATR calculation."""

    def test_calculate_atr_for_volatile_market(self):
        """Verify ATR calculation for volatile market."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create volatile market data (large price swings)
        np.random.seed(42)
        n = 100
        price_data = pd.DataFrame({
            'high': 100 + np.random.randn(n) * 5,
            'low': 100 + np.random.randn(n) * 5,
            'close': 100 + np.random.randn(n) * 5
        })
        price_data['low'] = price_data[['high', 'low', 'close']].min(axis=1)
        price_data['high'] = price_data[['high', 'low', 'close']].max(axis=1)

        atr = analyzer.calculate_atr(price_data, period=14)

        # ATR should be calculated and high for volatile market
        assert atr is not None
        assert len(atr) == len(price_data)
        # Volatile market should have higher ATR
        assert atr.iloc[-1] > 2

    def test_calculate_atr_for_quiet_market(self):
        """Verify ATR calculation for quiet market."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create quiet market data (small price movements)
        np.random.seed(42)
        n = 100
        price_data = pd.DataFrame({
            'high': 100 + np.random.randn(n) * 0.5,
            'low': 100 + np.random.randn(n) * 0.5,
            'close': 100 + np.random.randn(n) * 0.5
        })
        price_data['low'] = price_data[['high', 'low', 'close']].min(axis=1)
        price_data['high'] = price_data[['high', 'low', 'close']].max(axis=1)

        atr = analyzer.calculate_atr(price_data, period=14)

        # ATR should be calculated and low for quiet market
        assert atr is not None
        assert len(atr) == len(price_data)
        # Quiet market should have lower ATR
        assert atr.iloc[-1] < 1

    def test_calculate_atr_handles_insufficient_data(self):
        """Verify ATR calculation handles insufficient data."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create insufficient data (less than period)
        price_data = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [99.5, 100.5, 101.5]
        })

        atr = analyzer.calculate_atr(price_data, period=14)

        # Should return NaN for insufficient data
        assert atr is not None
        assert pd.isna(atr.iloc[-1])


class TestClassifyMarketRegimes:
    """Test market regime classification."""

    def test_classify_trending_volatile(self):
        """Verify classification of trending volatile regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(
            adx_trending_threshold=25.0,
            atr_volatile_threshold=20.0
        )

        # Create ADX > 25, ATR > 20
        adx = pd.Series([30, 35, 40])
        atr = pd.Series([25, 30, 35])

        regimes = analyzer.classify_market_regimes(adx, atr)

        assert all(regimes == "Trending Volatile")

    def test_classify_trending_quiet(self):
        """Verify classification of trending quiet regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(
            adx_trending_threshold=25.0,
            atr_volatile_threshold=20.0
        )

        # Create ADX > 25, ATR <= 20
        adx = pd.Series([30, 35, 40])
        atr = pd.Series([15, 18, 20])

        regimes = analyzer.classify_market_regimes(adx, atr)

        assert all(regimes == "Trending Quiet")

    def test_classify_ranging_volatile(self):
        """Verify classification of ranging volatile regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(
            adx_trending_threshold=25.0,
            atr_volatile_threshold=20.0
        )

        # Create ADX <= 25, ATR > 20
        adx = pd.Series([20, 22, 25])
        atr = pd.Series([25, 30, 35])

        regimes = analyzer.classify_market_regimes(adx, atr)

        assert all(regimes == "Ranging Volatile")

    def test_classify_ranging_quiet(self):
        """Verify classification of ranging quiet regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer(
            adx_trending_threshold=25.0,
            atr_volatile_threshold=20.0
        )

        # Create ADX <= 25, ATR <= 20
        adx = pd.Series([20, 22, 25])
        atr = pd.Series([15, 18, 20])

        regimes = analyzer.classify_market_regimes(adx, atr)

        assert all(regimes == "Ranging Quiet")


class TestAssignTradesToRegimes:
    """Test trade-to-regime assignment."""

    def test_map_trade_dates_to_correct_regimes(self):
        """Verify mapping trade dates to correct regimes."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime series
        dates = pd.date_range('2024-01-01', periods=5)
        regime_series = pd.Series(
            ['Trending Volatile', 'Trending Quiet',
             'Ranging Volatile', 'Ranging Quiet', 'Trending Volatile'],
            index=dates
        )

        # Create trades
        trades_df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2024-01-01 10:00'),
                pd.Timestamp('2024-01-02 14:30'),
                pd.Timestamp('2024-01-03 09:15')
            ],
            'pnl': [100, -50, 75]
        })

        result = analyzer.assign_trades_to_regimes(trades_df, regime_series)

        assert 'regime' in result.columns
        assert result.iloc[0]['regime'] == 'Trending Volatile'
        assert result.iloc[1]['regime'] == 'Trending Quiet'
        assert result.iloc[2]['regime'] == 'Ranging Volatile'

    def test_handle_trades_at_regime_boundaries(self):
        """Verify handling trades at regime boundaries."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime series
        dates = pd.date_range('2024-01-01', periods=3)
        regime_series = pd.Series(
            ['Trending Volatile', 'Trending Quiet', 'Ranging Volatile'],
            index=dates
        )

        # Create trade at exact boundary (midnight)
        trades_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-02 00:00')],
            'pnl': [100]
        })

        result = analyzer.assign_trades_to_regimes(trades_df, regime_series)

        assert 'regime' in result.columns
        # Should map to the day's regime
        assert result.iloc[0]['regime'] == 'Trending Quiet'

    def test_handle_missing_regime_data(self):
        """Verify handling trades with missing regime data."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime series with gap
        dates = pd.date_range('2024-01-01', periods=3)
        regime_series = pd.Series(
            ['Trending Volatile', 'Trending Quiet', 'Ranging Volatile'],
            index=dates
        )

        # Create trade outside regime range
        trades_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-05 10:00')],
            'pnl': [100]
        })

        result = analyzer.assign_trades_to_regimes(trades_df, regime_series)

        assert 'regime' in result.columns
        # Should handle missing regime gracefully
        assert pd.isna(result.iloc[0]['regime']) or \
               result.iloc[0]['regime'] == 'Unknown'


class TestCalculateRegimeMetrics:
    """Test regime-specific metrics calculation."""

    def test_calculate_win_rate_per_regime(self):
        """Verify win rate calculation per regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create trades by regime
        trades_by_regime = {
            'Trending Volatile': pd.DataFrame({
                'pnl': [100, -50, 75, -25, 150]  # 3 wins, 2 losses = 60%
            }),
            'Ranging Quiet': pd.DataFrame({
                'pnl': [50, -75, -30, 25]  # 2 wins, 2 losses = 50%
            })
        }

        metrics = analyzer.calculate_regime_metrics(trades_by_regime)

        assert 'Trending Volatile' in metrics.index
        assert 'Ranging Quiet' in metrics.index
        assert metrics.loc['Trending Volatile', 'win_rate'] == 60.0
        assert metrics.loc['Ranging Quiet', 'win_rate'] == 50.0

    def test_calculate_profit_factor_per_regime(self):
        """Verify profit factor calculation per regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create trades by regime
        trades_by_regime = {
            'Trending Volatile': pd.DataFrame({
                'pnl': [100, -50, 75, -25, 150]  # gross wins: 325, gross losses: 75
            })
        }

        metrics = analyzer.calculate_regime_metrics(trades_by_regime)

        profit_factor = metrics.loc['Trending Volatile', 'profit_factor']
        expected_pf = 325 / 75  # 4.33
        assert abs(profit_factor - expected_pf) < 0.1

    def test_calculate_sharpe_ratio_per_regime(self):
        """Verify Sharpe ratio calculation per regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create trades by regime
        trades_by_regime = {
            'Trending Volatile': pd.DataFrame({
                'pnl': [100, -50, 75, -25, 150],
                'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
            })
        }

        metrics = analyzer.calculate_regime_metrics(trades_by_regime)

        assert 'sharpe_ratio' in metrics.columns
        sharpe = metrics.loc['Trending Volatile', 'sharpe_ratio']
        # Should have positive Sharpe for profitable regime
        assert sharpe > 0

    def test_calculate_max_drawdown_per_regime(self):
        """Verify max drawdown calculation per regime."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create trades by regime
        trades_by_regime = {
            'Trending Volatile': pd.DataFrame({
                'pnl': [100, 50, -75, -25, 150],
                'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
            })
        }

        metrics = analyzer.calculate_regime_metrics(trades_by_regime)

        assert 'max_drawdown_pct' in metrics.columns
        dd = metrics.loc['Trending Volatile', 'max_drawdown_pct']
        # Should have drawdown < 0 (negative)
        assert dd < 0

    def test_handle_regimes_with_few_trades(self):
        """Verify handling regimes with insufficient trades."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime with only 1 trade
        trades_by_regime = {
            'Trending Volatile': pd.DataFrame({
                'pnl': [100]
            })
        }

        metrics = analyzer.calculate_regime_metrics(trades_by_regime)

        # Should still calculate metrics but with warnings
        assert 'Trending Volatile' in metrics.index


class TestGenerateComparisonTable:
    """Test comparison table generation."""

    def test_create_table_with_correct_columns(self):
        """Verify table has correct columns."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime metrics DataFrame
        metrics_df = pd.DataFrame({
            'trade_count': [100, 200, 150],
            'win_rate': [60.0, 55.0, 58.0],
            'profit_factor': [1.8, 1.5, 1.7],
            'sharpe_ratio': [2.1, 1.5, 1.9],
            'max_drawdown_pct': [-5.2, -8.1, -6.3]
        }, index=['Trending Volatile', 'Ranging Quiet', 'Trending Quiet'])

        table = analyzer.generate_comparison_table(metrics_df)

        assert 'regime_name' in table.columns
        assert 'trade_count' in table.columns
        assert 'win_rate_pct' in table.columns
        assert 'profit_factor' in table.columns
        assert 'sharpe_ratio' in table.columns
        assert 'max_drawdown_pct' in table.columns

    def test_format_percentages_correctly(self):
        """Verify percentage formatting."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime metrics DataFrame
        metrics_df = pd.DataFrame({
            'trade_count': [100],
            'win_rate': [60.123],
            'profit_factor': [1.8],
            'sharpe_ratio': [2.1],
            'max_drawdown_pct': [-5.234]
        }, index=['Trending Volatile'])

        table = analyzer.generate_comparison_table(metrics_df)

        # Should be rounded to 2 decimal places
        assert table['win_rate_pct'].iloc[0] == 60.12
        assert table['max_drawdown_pct'].iloc[0] == -5.23

    def test_sort_by_trade_count(self):
        """Verify sorting by trade count."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create regime metrics DataFrame (unsorted)
        metrics_df = pd.DataFrame({
            'trade_count': [150, 200, 100],
            'win_rate': [58.0, 55.0, 60.0],
            'profit_factor': [1.7, 1.5, 1.8],
            'sharpe_ratio': [1.9, 1.5, 2.1],
            'max_drawdown_pct': [-6.3, -8.1, -5.2]
        }, index=['Trending Quiet', 'Ranging Quiet', 'Trending Volatile'])

        table = analyzer.generate_comparison_table(metrics_df)

        # Should be sorted by trade_count descending
        assert table.iloc[0]['trade_count'] == 200
        assert table.iloc[2]['trade_count'] == 100


class TestGenerateRegimeCharts:
    """Test regime comparison chart generation."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    def test_create_win_rate_bar_chart(self, mock_subplots, mock_savefig):
        """Verify win rate bar chart creation."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        # Setup mock
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        analyzer = MarketRegimeAnalyzer()

        # Create regime metrics DataFrame
        metrics_df = pd.DataFrame({
            'trade_count': [100, 200, 150],
            'win_rate': [60.0, 55.0, 58.0],
            'profit_factor': [1.8, 1.5, 1.7],
            'sharpe_ratio': [2.1, 1.5, 1.9],
            'max_drawdown_pct': [-5.2, -8.1, -6.3]
        }, index=['Trending Volatile', 'Ranging Quiet', 'Trending Quiet'])

        fig = analyzer.generate_regime_charts(metrics_df)

        assert fig is not None
        mock_subplots.assert_called_once()
        # Should create subplots (2 charts)
        assert mock_subplots.call_args[0][0] == 1
        assert mock_subplots.call_args[0][1] == 2

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    def test_create_profit_factor_bar_chart(self, mock_subplots, mock_savefig):
        """Verify profit factor bar chart creation."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        # Setup mock
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        analyzer = MarketRegimeAnalyzer()

        # Create regime metrics DataFrame
        metrics_df = pd.DataFrame({
            'trade_count': [100, 200, 150],
            'win_rate': [60.0, 55.0, 58.0],
            'profit_factor': [1.8, 1.5, 1.7],
            'sharpe_ratio': [2.1, 1.5, 1.9],
            'max_drawdown_pct': [-5.2, -8.1, -6.3]
        }, index=['Trending Volatile', 'Ranging Quiet', 'Trending Quiet'])

        fig = analyzer.generate_regime_charts(metrics_df)

        assert fig is not None
        # Should create 2 subplots
        assert mock_subplots.call_args[0][1] == 2


class TestSaveResults:
    """Test saving results to files."""

    @patch('pandas.DataFrame.to_csv')
    @patch('matplotlib.figure.Figure.savefig')
    @patch('pathlib.Path.mkdir')
    def test_save_csv_to_correct_location(
        self,
        mock_mkdir,
        mock_savefig,
        mock_to_csv
    ):
        """Verify CSV saved to correct location."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create table and figure
        table = pd.DataFrame({'regime': ['Trending'], 'win_rate': [60.0]})
        fig = MagicMock()

        csv_path, png_path = analyzer.save_results(table, fig)

        assert "regime_analysis_" in csv_path
        assert csv_path.endswith(".csv")
        mock_to_csv.assert_called_once()

    @patch('pandas.DataFrame.to_csv')
    @patch('pathlib.Path.mkdir')
    def test_save_png_to_correct_location(self, mock_mkdir, mock_to_csv):
        """Verify PNG saved to correct location."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create table and figure
        table = pd.DataFrame({'regime': ['Trending'], 'win_rate': [60.0]})
        fig = MagicMock()

        csv_path, png_path = analyzer.save_results(table, fig)

        assert "regime_comparison_" in png_path
        assert png_path.endswith(".png")
        fig.savefig.assert_called_once()

    @patch('pandas.DataFrame.to_csv')
    @patch('matplotlib.figure.Figure.savefig')
    @patch('pathlib.Path.mkdir')
    def test_filenames_include_timestamp(
        self,
        mock_mkdir,
        mock_savefig,
        mock_to_csv
    ):
        """Verify filenames include timestamp."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create table and figure
        table = pd.DataFrame({'regime': ['Trending'], 'win_rate': [60.0]})
        fig = MagicMock()

        csv_path, png_path = analyzer.save_results(table, fig)

        # Should include date pattern (YYYY-MM-DD)
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        assert re.search(date_pattern, csv_path)
        assert re.search(date_pattern, png_path)

    @patch('pandas.DataFrame.to_csv')
    @patch('matplotlib.figure.Figure.savefig')
    @patch('pathlib.Path.mkdir')
    def test_create_directory_if_needed(
        self,
        mock_mkdir,
        mock_savefig,
        mock_to_csv
    ):
        """Verify directory created if it doesn't exist."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create table and figure
        table = pd.DataFrame({'regime': ['Trending'], 'win_rate': [60.0]})
        fig = MagicMock()

        csv_path, png_path = analyzer.save_results(table, fig)

        # mkdir should be called with parents=True, exist_ok=True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestAnalyzeRegimePerformance:
    """Test end-to-end regime performance analysis."""

    def test_end_to_end_analysis_pipeline(self):
        """Verify complete analysis pipeline."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create mock price data
        np.random.seed(42)
        n = 100
        price_data = pd.DataFrame({
            'high': np.linspace(100, 120, n) + np.random.randn(n) * 0.5,
            'low': np.linspace(99, 119, n) + np.random.randn(n) * 0.5,
            'close': np.linspace(99.5, 119.5, n) + np.random.randn(n) * 0.5
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))

        # Create mock trades
        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-15', periods=50, freq='D'),
            'pnl': np.random.randn(50) * 100
        })

        result = analyzer.analyze_regime_performance(price_data, trades_df)

        # Should return analysis results
        assert 'regime_metrics' in result
        assert 'comparison_table' in result
        assert 'csv_path' in result
        assert 'png_path' in result

    def test_all_components_integrated(self):
        """Verify all components integrated correctly."""
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create mock price data
        np.random.seed(42)
        n = 100
        price_data = pd.DataFrame({
            'high': np.linspace(100, 120, n) + np.random.randn(n) * 0.5,
            'low': np.linspace(99, 119, n) + np.random.randn(n) * 0.5,
            'close': np.linspace(99.5, 119.5, n) + np.random.randn(n) * 0.5
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))

        # Create mock trades
        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-15', periods=50, freq='D'),
            'pnl': np.random.randn(50) * 100
        })

        result = analyzer.analyze_regime_performance(price_data, trades_df)

        # Check regime_metrics DataFrame
        regime_metrics = result['regime_metrics']
        assert 'trade_count' in regime_metrics.columns
        assert 'win_rate' in regime_metrics.columns
        assert 'profit_factor' in regime_metrics.columns

    @patch('matplotlib.pyplot.savefig')
    @patch('pandas.DataFrame.to_csv')
    def test_performance_requirement_under_30_seconds(
        self,
        mock_to_csv,
        mock_savefig
    ):
        """Verify analysis completes in < 30 seconds."""
        import time
        from src.research.market_regime_analyzer import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer()

        # Create mock price data (large dataset)
        np.random.seed(42)
        n = 1000
        price_data = pd.DataFrame({
            'high': np.linspace(100, 120, n) + np.random.randn(n) * 0.5,
            'low': np.linspace(99, 119, n) + np.random.randn(n) * 0.5,
            'close': np.linspace(99.5, 119.5, n) + np.random.randn(n) * 0.5
        }, index=pd.date_range('2020-01-01', periods=n, freq='D'))

        # Create mock trades (500 trades)
        trades_df = pd.DataFrame({
            'timestamp': pd.date_range('2020-02-01', periods=500, freq='D'),
            'pnl': np.random.randn(500) * 100
        })

        start_time = time.time()
        analyzer.analyze_regime_performance(price_data, trades_df)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 30.0, \
            f"Analysis took {elapsed_time:.2f} seconds"

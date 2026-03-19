"""Market Regime Analyzer for strategy performance analysis.

Analyzes strategy performance across different market regimes (trending vs. ranging,
volatile vs. quiet) using ADX and ATR indicators.

Performance: Completes in < 30 seconds.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """Analyze strategy performance across different market regimes.

    Classifies trading days into regimes using ADX and ATR indicators,
    calculates regime-specific performance metrics, generates comparisons.

    Performance: Completes in < 30 seconds.
    """

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        atr_volatile_threshold: float = 20.0,
        adx_period: int = 14,
        atr_period: int = 14,
        output_directory: str = "data/reports"
    ):
        """Initialize market regime analyzer.

        Args:
            adx_trending_threshold: ADX value above which market is trending
            atr_volatile_threshold: ATR value above which market is volatile
            adx_period: ADX calculation period (default: 14)
            atr_period: ATR calculation period (default: 14)
            output_directory: Directory to save reports
        """
        self._adx_trending_threshold = adx_trending_threshold
        self._atr_volatile_threshold = atr_volatile_threshold
        self._adx_period = adx_period
        self._atr_period = atr_period
        self._output_directory = Path(output_directory)

        # Create output directory
        self._output_directory.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"MarketRegimeAnalyzer initialized: "
            f"adx_threshold={adx_trending_threshold}, "
            f"atr_threshold={atr_volatile_threshold}, "
            f"adx_period={adx_period}, atr_period={atr_period}"
        )

    def calculate_adx(
        self,
        price_data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index (ADX).

        Args:
            price_data: DataFrame with high, low, close columns
            period: ADX calculation period

        Returns:
            Series with ADX values
        """
        logger.debug("Calculating ADX...")

        # Extract prices
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate smoothed +DI and -DI
        alpha = 1 / period

        plus_di = 100 * (
            plus_dm.ewm(alpha=alpha, adjust=False).mean() /
            tr.ewm(alpha=alpha, adjust=False).mean()
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=alpha, adjust=False).mean() /
            tr.ewm(alpha=alpha, adjust=False).mean()
        )

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        logger.debug("ADX calculation complete")
        return adx

    def calculate_atr(
        self,
        price_data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range (ATR).

        Args:
            price_data: DataFrame with high, low, close columns
            period: ATR calculation period

        Returns:
            Series with ATR values
        """
        logger.debug("Calculating ATR...")

        # Extract prices
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR as rolling average
        atr = tr.rolling(window=period).mean()

        logger.debug("ATR calculation complete")
        return atr

    def classify_market_regimes(
        self,
        adx: pd.Series,
        atr: pd.Series
    ) -> pd.Series:
        """Classify each day into market regime.

        Args:
            adx: Series with ADX values
            atr: Series with ATR values

        Returns:
            Series with regime labels:
            - "Trending Volatile" (ADX > threshold, ATR > threshold)
            - "Trending Quiet" (ADX > threshold, ATR <= threshold)
            - "Ranging Volatile" (ADX <= threshold, ATR > threshold)
            - "Ranging Quiet" (ADX <= threshold, ATR <= threshold)
        """
        logger.debug("Classifying market regimes...")

        # Initialize regime series
        regimes = pd.Series(index=adx.index, dtype='object')

        # Classify based on ADX and ATR thresholds
        trending = adx > self._adx_trending_threshold
        volatile = atr > self._atr_volatile_threshold

        regimes.loc[trending & volatile] = "Trending Volatile"
        regimes.loc[trending & ~volatile] = "Trending Quiet"
        regimes.loc[~trending & volatile] = "Ranging Volatile"
        regimes.loc[~trending & ~volatile] = "Ranging Quiet"

        logger.debug("Market regime classification complete")
        return regimes

    def assign_trades_to_regimes(
        self,
        trades_df: pd.DataFrame,
        regime_series: pd.Series
    ) -> pd.DataFrame:
        """Assign each trade to its market regime.

        Args:
            trades_df: DataFrame with timestamp column
            regime_series: Series with regime labels indexed by date

        Returns:
            DataFrame with regime column added
        """
        logger.debug("Assigning trades to regimes...")

        # Extract date from timestamp
        trade_dates = trades_df['timestamp'].dt.date

        # Map trade dates to regimes
        trades_with_regime = trades_df.copy()
        trades_with_regime['regime'] = trade_dates.map(
            lambda d: regime_series.get(
                pd.Timestamp(d),
                'Unknown'
            )
        )

        logger.debug("Trade-to-regime assignment complete")
        return trades_with_regime

    def calculate_regime_metrics(
        self,
        trades_by_regime: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate performance metrics for each regime.

        Args:
            trades_by_regime: Dictionary mapping regime name to trades DataFrame

        Returns:
            DataFrame with regime metrics (trade_count, win_rate, profit_factor,
            sharpe_ratio, max_drawdown_pct)
        """
        logger.debug("Calculating regime-specific metrics...")

        metrics_list = []

        for regime, trades in trades_by_regime.items():
            if len(trades) == 0:
                continue

            # Trade count
            trade_count = len(trades)

            # Win rate
            wins = (trades['pnl'] > 0).sum()
            win_rate = 100 * wins / trade_count

            # Profit factor
            gross_wins = trades[trades['pnl'] > 0]['pnl'].sum()
            gross_losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.nan

            # Sharpe ratio (simplified)
            if 'timestamp' in trades.columns:
                # Calculate daily returns
                trades_daily = trades.set_index('timestamp')['pnl'].resample('D').sum()
                sharpe_ratio = (
                    trades_daily.mean() / trades_daily.std() * np.sqrt(252)
                    if trades_daily.std() > 0 else np.nan
                )
            else:
                sharpe_ratio = np.nan

            # Max drawdown
            cumulative_pnl = trades['pnl'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = (cumulative_pnl - running_max) / running_max * 100
            max_drawdown_pct = drawdown.min()

            metrics_list.append({
                'trade_count': trade_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown_pct
            })

        metrics_df = pd.DataFrame(metrics_list)
        # Filter out empty regimes
        metrics_df.index = [
            r for r in trades_by_regime.keys()
            if len(trades_by_regime[r]) > 0
        ]

        logger.debug("Regime metrics calculation complete")
        return metrics_df

    def generate_comparison_table(
        self,
        metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create formatted comparison table.

        Args:
            metrics_df: DataFrame with regime metrics

        Returns:
            Formatted DataFrame with regime_name, trade_count, win_rate_pct,
            profit_factor, sharpe_ratio, max_drawdown_pct columns
        """
        logger.debug("Generating comparison table...")

        # Reset index to make regime a column
        table = metrics_df.reset_index()
        table.rename(columns={'index': 'regime_name'}, inplace=True)

        # Format percentages
        table['win_rate_pct'] = table['win_rate'].round(2)
        table['max_drawdown_pct'] = table['max_drawdown_pct'].round(2)

        # Round other metrics
        table['profit_factor'] = table['profit_factor'].round(2)
        table['sharpe_ratio'] = table['sharpe_ratio'].round(2)

        # Sort by trade count (descending)
        table = table.sort_values('trade_count', ascending=False)

        # Select and order columns
        table = table[[
            'regime_name',
            'trade_count',
            'win_rate_pct',
            'profit_factor',
            'sharpe_ratio',
            'max_drawdown_pct'
        ]]

        logger.debug("Comparison table generated")
        return table

    def generate_regime_charts(
        self,
        metrics_df: pd.DataFrame
    ) -> plt.Figure:
        """Generate bar charts comparing metrics across regimes.

        Args:
            metrics_df: DataFrame with regime metrics

        Returns:
            Matplotlib figure with subplots
        """
        logger.debug("Generating regime comparison charts...")

        # Sort by trade count
        metrics_sorted = metrics_df.sort_values('trade_count', ascending=False)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

        # Win rate bar chart
        regimes = metrics_sorted.index
        win_rates = metrics_sorted['win_rate']

        bars1 = ax1.bar(range(len(regimes)), win_rates, color='steelblue',
                        edgecolor='black')
        ax1.set_xticks(range(len(regimes)))
        ax1.set_xticklabels(regimes, rotation=45, ha='right')
        ax1.set_ylabel('Win Rate (%)', fontsize=12)
        ax1.set_title('Win Rate by Market Regime', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'{value:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10
            )

        # Profit factor bar chart
        profit_factors = metrics_sorted['profit_factor']

        # Color code: green > 2.0, yellow 1.5-2.0, red < 1.5
        colors = []
        for pf in profit_factors:
            if pf >= 2.0:
                colors.append('green')
            elif pf >= 1.5:
                colors.append('orange')
            else:
                colors.append('red')

        bars2 = ax2.bar(
            range(len(regimes)),
            profit_factors,
            color=colors,
            edgecolor='black'
        )
        ax2.set_xticks(range(len(regimes)))
        ax2.set_xticklabels(regimes, rotation=45, ha='right')
        ax2.set_ylabel('Profit Factor', fontsize=12)
        ax2.set_title(
            'Profit Factor by Market Regime',
            fontsize=14,
            fontweight='bold'
        )
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Breakeven')

        # Add value labels on bars
        for bar, value in zip(bars2, profit_factors):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.05,
                f'{value:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.tight_layout()

        logger.debug("Regime charts generated")
        return fig

    def save_results(
        self,
        table: pd.DataFrame,
        fig: plt.Figure
    ) -> tuple[str, str]:
        """Save comparison table and charts to files.

        Args:
            table: Comparison table DataFrame
            fig: Matplotlib figure

        Returns:
            Tuple of (csv_path, png_path)
        """
        logger.debug("Saving results...")

        # Generate filename with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

        csv_filename = f"regime_analysis_{timestamp}.csv"
        png_filename = f"regime_comparison_{timestamp}.png"

        csv_path = self._output_directory / csv_filename
        png_path = self._output_directory / png_filename

        # Save CSV
        table.to_csv(csv_path, index=False)

        # Save PNG
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        logger.debug(f"Saved table to {csv_path}, chart to {png_path}")

        return str(csv_path), str(png_path)

    def analyze_regime_performance(
        self,
        price_data: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Perform complete regime performance analysis.

        Args:
            price_data: DataFrame with OHLC price data (indexed by date)
            trades_df: DataFrame with timestamp and pnl columns

        Returns:
            Dictionary with:
                - regime_metrics: DataFrame with regime-specific metrics
                - comparison_table: Formatted comparison table
                - csv_path: Path to saved CSV table
                - png_path: Path to saved PNG chart
        """
        logger.info("Starting regime performance analysis...")

        # Calculate indicators
        adx = self.calculate_adx(price_data, self._adx_period)
        atr = self.calculate_atr(price_data, self._atr_period)

        # Classify regimes
        regimes = self.classify_market_regimes(adx, atr)

        # Assign trades to regimes
        trades_with_regime = self.assign_trades_to_regimes(trades_df, regimes)

        # Group trades by regime
        trades_by_regime = {}
        for regime in regimes.unique():
            if regime != 'Unknown':
                regime_trades = trades_with_regime[
                    trades_with_regime['regime'] == regime
                ]
                trades_by_regime[regime] = regime_trades

        # Calculate regime metrics
        regime_metrics = self.calculate_regime_metrics(trades_by_regime)

        # Generate comparison table
        comparison_table = self.generate_comparison_table(regime_metrics)

        # Generate charts
        fig = self.generate_regime_charts(regime_metrics)

        # Save results
        csv_path, png_path = self.save_results(comparison_table, fig)

        # Log summary
        best_regime = regime_metrics['win_rate'].idxmax()
        best_win_rate = regime_metrics.loc[best_regime, 'win_rate']

        logger.info(
            f"Regime analysis complete: "
            f"Best regime: {best_regime} "
            f"(Win rate: {best_win_rate:.1f}%), "
            f"Regimes analyzed: {len(regime_metrics)}"
        )

        return {
            'regime_metrics': regime_metrics,
            'comparison_table': comparison_table,
            'csv_path': csv_path,
            'png_path': png_path
        }

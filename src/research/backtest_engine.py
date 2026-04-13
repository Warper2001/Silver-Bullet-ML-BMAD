"""Backtest Engine for strategy testing.

This module provides a backtesting engine to test trading strategies
on historical data and track trade performance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade.

    Attributes:
        entry_time: When the trade was entered
        exit_time: When the trade was exited
        direction: "long" or "short"
        entry_price: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        take_profit: Take profit price
        pnl: Profit/loss in dollars
        bars_held: Number of bars held
    """

    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    bars_held: int


class BacktestEngine:
    """Backtesting engine for strategy testing.

    Attributes:
        initial_capital: Starting capital for backtest
        trades: List of completed trades
        current_capital: Current capital after trades
    """

    MNQ_TICK_VALUE = 0.25  # $0.25 per tick for MNQ
    CONTRACT_MULTIPLIER = 20  # MNQ contract multiplier

    def __init__(self, initial_capital: float = 100000.0) -> None:
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital (default $100,000)
        """
        self.initial_capital = initial_capital
        self.trades: list[Trade] = []
        self.current_capital = initial_capital

    def add_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        bars_held: int,
    ) -> None:
        """Add a completed trade to the backtest.

        Args:
            entry_time: Trade entry time
            exit_time: Trade exit time
            direction: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            bars_held: Number of bars held
        """
        # Calculate P&L
        if direction == "long":
            price_diff = exit_price - entry_price
        else:  # short
            price_diff = entry_price - exit_price

        # MNQ: $0.25 per tick * 20 ticks per point = $5 per point
        pnl = price_diff * 5.0  # $5 per point for MNQ

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl,
            bars_held=bars_held,
        )

        self.trades.append(trade)
        self.current_capital += pnl

        logger.debug(
            f"Trade added: {direction} {entry_price:.2f} -> {exit_price:.2f}, "
            f"P&L: ${pnl:.2f}"
        )

    def get_total_pnl(self) -> float:
        """Get total P&L from all trades.

        Returns:
            Total P&L in dollars
        """
        return sum(trade.pnl for trade in self.trades)

    def get_win_count(self) -> int:
        """Get number of winning trades.

        Returns:
            Count of winning trades
        """
        return sum(1 for trade in self.trades if trade.pnl > 0)

    def get_loss_count(self) -> int:
        """Get number of losing trades.

        Returns:
            Count of losing trades
        """
        return sum(1 for trade in self.trades if trade.pnl < 0)

    def get_total_trades(self) -> int:
        """Get total number of trades.

        Returns:
            Total trade count
        """
        return len(self.trades)

    def get_all_trades(self) -> list[Trade]:
        """Get all completed trades.

        Returns:
            List of all trades
        """
        return self.trades

    def reset(self) -> None:
        """Reset the backtest engine."""
        self.trades = []
        self.current_capital = self.initial_capital
        logger.debug("Backtest engine reset")


class CalibrationComparison:
    """Side-by-side comparison of uncalibrated vs calibrated models.

    This class runs backtests with both uncalibrated and calibrated models
    and generates comparison metrics to validate calibration improvements.
    """

    def __init__(
        self,
        uncalibrated_model,
        calibrated_model,
        calibration,
        data_path: str = "data/processed/dollar_bars/1_minute",
    ) -> None:
        """Initialize calibration comparison.

        Args:
            uncalibrated_model: XGBoost model without calibration
            calibrated_model: XGBoost model with calibration applied
            calibration: ProbabilityCalibration instance
            data_path: Path to dollar bar data directory
        """
        self.uncalibrated_model = uncalibrated_model
        self.calibrated_model = calibrated_model
        self.calibration = calibration
        self.data_path = Path(data_path)
        logger.info(f"CalibrationComparison initialized with data_path: {data_path}")

    def run_uncalibrated_backtest(
        self, start_date: str, end_date: str
    ) -> dict[str, float]:
        """Run backtest with uncalibrated model.

        Args:
            start_date: Start date for backtest period
            end_date: End date for backtest period

        Returns:
            Dictionary with backtest metrics (win_rate, mean_predicted_probability,
            brier_score, trade_count)
        """
        logger.info(f"Running uncalibrated backtest: {start_date} to {end_date}")

        # Load data
        features, labels = self._load_data(start_date, end_date)

        if len(features) == 0:
            logger.warning("No data available for uncalibrated backtest")
            return {
                "win_rate": 0.0,
                "mean_predicted_probability": 0.0,
                "brier_score": 1.0,
                "trade_count": 0,
            }

        # Get predictions
        proba_result = self.uncalibrated_model.predict_proba(features.values)
        proba_array = np.array(proba_result)
        y_proba = proba_array[:, 1] if len(proba_array.shape) == 2 else proba_array

        # Calculate metrics
        win_rate = float(labels.mean())
        mean_prob = float(np.mean(y_proba))
        brier = float(brier_score_loss(labels.values, y_proba))
        trade_count = len(labels)

        logger.info(
            f"Uncalibrated results: win_rate={win_rate:.2%}, "
            f"mean_prob={mean_prob:.2%}, brier={brier:.4f}"
        )

        return {
            "win_rate": win_rate,
            "mean_predicted_probability": mean_prob,
            "brier_score": brier,
            "trade_count": trade_count,
        }

    def run_calibrated_backtest(
        self, start_date: str, end_date: str
    ) -> dict[str, float]:
        """Run backtest with calibrated model.

        Args:
            start_date: Start date for backtest period
            end_date: End date for backtest period

        Returns:
            Dictionary with backtest metrics (win_rate, mean_predicted_probability,
            brier_score, trade_count)
        """
        logger.info(f"Running calibrated backtest: {start_date} to {end_date}")

        # Load data
        features, labels = self._load_data(start_date, end_date)

        if len(features) == 0:
            logger.warning("No data available for calibrated backtest")
            return {
                "win_rate": 0.0,
                "mean_predicted_probability": 0.0,
                "brier_score": 1.0,
                "trade_count": 0,
            }

        # Get calibrated predictions
        y_proba = np.array([
            self.calibration.predict_proba(features.iloc[i].values.reshape(1, -1))[0]
            for i in range(len(features))
        ])

        # Calculate metrics
        win_rate = float(labels.mean())
        mean_prob = float(np.mean(y_proba))
        brier = float(brier_score_loss(labels.values, y_proba))
        trade_count = len(labels)

        logger.info(
            f"Calibrated results: win_rate={win_rate:.2%}, "
            f"mean_prob={mean_prob:.2%}, brier={brier:.4f}"
        )

        return {
            "win_rate": win_rate,
            "mean_predicted_probability": mean_prob,
            "brier_score": brier,
            "trade_count": trade_count,
        }

    def generate_comparison_metrics(
        self, uncalibrated_result: dict, calibrated_result: dict
    ) -> dict[str, float]:
        """Generate comparison metrics between uncalibrated and calibrated results.

        Args:
            uncalibrated_result: Metrics from uncalibrated backtest
            calibrated_result: Metrics from calibrated backtest

        Returns:
            Dictionary with improvement metrics
        """
        logger.info("Generating comparison metrics")

        win_rate_improvement = (
            calibrated_result["win_rate"] - uncalibrated_result["win_rate"]
        )
        brier_improvement = (
            uncalibrated_result["brier_score"] - calibrated_result["brier_score"]
        )
        prob_match_uncalibrated = abs(
            uncalibrated_result["mean_predicted_probability"]
            - uncalibrated_result["win_rate"]
        )
        prob_match_calibrated = abs(
            calibrated_result["mean_predicted_probability"] - calibrated_result["win_rate"]
        )
        prob_match_improvement = prob_match_uncalibrated - prob_match_calibrated

        return {
            "win_rate_improvement": win_rate_improvement,
            "brier_score_improvement": brier_improvement,
            "probability_match_improvement": prob_match_improvement,
            "uncalibrated_probability_match": prob_match_uncalibrated,
            "calibrated_probability_match": prob_match_calibrated,
        }

    def run_side_by_side_comparison(
        self, start_date: str, end_date: str
    ) -> dict:
        """Run complete side-by-side comparison.

        Args:
            start_date: Start date for comparison period
            end_date: End date for comparison period

        Returns:
            Dictionary with uncalibrated, calibrated, and comparison metrics
        """
        logger.info(f"Running side-by-side comparison: {start_date} to {end_date}")

        # Run both backtests
        uncalibrated_result = self.run_uncalibrated_backtest(start_date, end_date)
        calibrated_result = self.run_calibrated_backtest(start_date, end_date)

        # Generate comparison metrics
        comparison = self.generate_comparison_metrics(
            uncalibrated_result, calibrated_result
        )

        return {
            "uncalibrated": uncalibrated_result,
            "calibrated": calibrated_result,
            "comparison": comparison,
        }

    def _load_data(self, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.Series]:
        """Load data for backtest period.

        Args:
            start_date: Start date string
            end_date: End date string

        Returns:
            (features_df, labels_series)
        """
        from src.ml.features import FeatureEngineer

        # Find CSV files in data path
        csv_files = list(self.data_path.glob("mnq_1min_*.csv"))

        if not csv_files:
            logger.warning(f"No MNQ data files found in {self.data_path}")
            return pd.DataFrame(), pd.Series()

        # Load and concatenate all CSV files
        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            all_data.append(df)

        if not all_data:
            return pd.DataFrame(), pd.Series()

        df = pd.concat(all_data, ignore_index=True)

        # Filter by date range - handle timezone-aware timestamps
        if df["timestamp"].dt.tz is not None:
            start = pd.Timestamp(start_date, tz="UTC")
            end = pd.Timestamp(end_date, tz="UTC")
        else:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)

        df_filtered = df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        if len(df_filtered) == 0:
            return pd.DataFrame(), pd.Series()

        # Reset index
        df_filtered = df_filtered.reset_index(drop=True)

        # Extract features
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(df_filtered)

        # Drop non-numeric columns
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]

        # Remove rows with NaN values
        features_df = features_df.dropna()

        # Generate labels based on 5-minute forward returns
        close_prices = df_filtered["close"].iloc[len(df_filtered) - len(features_df) :]
        forward_returns = close_prices.pct_change(5).shift(-5)

        # Create labels
        labels_series = (forward_returns > 0).astype(int)

        # Remove last 5 rows
        labels_series = labels_series.iloc[:-5]
        features_df = features_df.iloc[: len(labels_series)]

        # Ensure alignment
        min_length = min(len(features_df), len(labels_series))
        features_df = features_df.iloc[:min_length]
        labels_series = labels_series.iloc[:min_length]

        logger.info(f"Loaded {len(features_df)} samples for backtest")

        return features_df, labels_series

    def _calculate_brier_score(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> float:
        """Calculate Brier score for predictions.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            Brier score (lower is better, 0 is perfect)
        """
        return float(brier_score_loss(y_true, y_prob))

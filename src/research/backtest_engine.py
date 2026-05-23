"""Backtest Engine for strategy testing.

This module provides:
- ``BacktestEngine`` — Tier-2 deterministic simulation engine using pure
  ``strategy_core`` functions.  Byte-identical across runs (NFR15).
- ``LegacyTradeLedger`` — Legacy trade ledger used by Tier-1/ML-calibration scripts.
- ``CalibrationComparison`` — Side-by-side calibrated/uncalibrated model analysis.
- ``PARITYError`` — Raised when BacktestEngine diverges from the reference backtest.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.research.strategy_core import (
    POINT_VALUE_USD,
    Direction,
    EntryDecision,
    ExitDecision,
    StrategyConfig,
    SweepSignal,
    calc_atr,
    check_exit,
    check_m15_confirmation,
    detect_fvg,
    detect_liquidity_sweep,
    kill_zone_filter,
    make_entry_decision,
    resample_to_h1,
    resample_to_m15,
    volatility_regime_filter,
)

logger = logging.getLogger(__name__)

# Kept for legacy importers; StrategyConfig.commission_per_roundtrip is now the
# canonical value.  BacktestEngine reads it from config (AR13).
COMMISSION_PER_ROUNDTRIP: float = 4.0


class PARITYError(Exception):
    """Raised when BacktestEngine output diverges from the reference backtest."""


@dataclass(frozen=True)
class TradeRecord:
    """One completed Tier-2 trade — fixed column order matches the PRD spec (AR10/AR11).

    ``direction`` and ``exit_reason`` hold ``.value`` strings so the CSV is
    human-readable without an enum lookup.  ``m15_confirmed`` is always
    ``False`` until Story 2.3 adds M15 CHoCH detection to strategy_core.
    """

    timestamp_entry: pd.Timestamp
    timestamp_exit: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    gap_size: float
    pnl_usd: float
    exit_reason: str
    h1_sweep_bars_ago: int
    m15_confirmed: bool
    kill_zone_active: bool
    vol_regime_pct: float
    contracts: int


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


class LegacyTradeLedger:
    """Trade ledger for legacy Tier-1 / ML-calibration backtests.

    Attributes:
        initial_capital: Starting capital for backtest
        trades: List of completed trades
        current_capital: Current capital after trades
    """

    MNQ_TICK_VALUE = 0.25  # $0.25 per tick for MNQ
    CONTRACT_MULTIPLIER = 20  # MNQ contract multiplier

    def __init__(self, initial_capital: float = 100000.0) -> None:
        """Initialize trade ledger.

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

    def __init__(  # type: ignore[no-untyped-def]
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
        y_proba = np.array(
            [
                self.calibration.predict_proba(features.iloc[i].values.reshape(1, -1))[  # type: ignore[union-attr]  # noqa: E501
                    0
                ]
                for i in range(len(features))
            ]
        )

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
        self, uncalibrated_result: dict[str, float], calibrated_result: dict[str, float]
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
            calibrated_result["mean_predicted_probability"]
            - calibrated_result["win_rate"]
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
    ) -> dict[str, object]:
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

    def _load_data(
        self, start_date: str, end_date: str
    ) -> tuple[pd.DataFrame, pd.Series]:
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

    def _calculate_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:  # type: ignore[type-arg]  # noqa: E501
        """Calculate Brier score for predictions.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            Brier score (lower is better, 0 is perfect)
        """
        return float(brier_score_loss(y_true, y_prob))


# ---------------------------------------------------------------------------
# Tier-2 deterministic simulation engine (Story 1.4)
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path("data/reports")
_LOGS_DIR = Path("logs")
# H1 buffer: must cover vol_regime_lookback (120) + h1_sweep_lookback (6) + margin H1 bars.
# 125 H1 bars × 60 min = 7,500 M1 bars.  3000 was too small (only 50 H1 bars).
_H1_BUFFER_BARS: int = 7500


class BacktestEngine:
    """Deterministic Tier-2 simulation engine.

    Loads 1-min OHLCV bars from a CSV file and replays them through the pure
    ``strategy_core`` functions to produce a list of ``TradeRecord`` objects.
    Identical inputs always produce identical outputs (NFR15).

    Parameters
    ----------
    csv_path:
        Path to a 1-min OHLCV CSV with America/New_York or UTC timestamps and
        columns ``timestamp,open,high,low,close,volume[,notional]``.
    config:
        Strategy parameters.  Defaults to ``StrategyConfig()`` which matches
        the live-system constants.
    """

    def __init__(
        self,
        csv_path: str,
        config: StrategyConfig | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.config = config or StrategyConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _load_bars(self, path: str) -> pd.DataFrame:
        """Load and validate 1-min bars from *path* (AC #1, AR9, AR19).

        Returns a tz-aware ``DatetimeIndex`` DataFrame in America/New_York.
        Conversion from UTC is performed exactly once here; ``strategy_core``
        never converts timezone (AR19).
        """
        df = pd.read_csv(path, parse_dates=["timestamp"])

        # Ensure timezone-awareness — CSV stores UTC offsets (+00:00)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Ingest boundary: single tz conversion here, never inside strategy_core (AR19)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        # Canonical schema (AR9)
        df = df.set_index("timestamp")
        df.index.name = "timestamp"
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        df = df.drop(columns=["notional"], errors="ignore")

        # Sort ascending
        df = df.sort_index()

        # Drop exact-duplicate timestamps (keep first)
        dup = df.index.duplicated(keep="first")
        if dup.any():
            logger.warning(
                "_load_bars: dropped %d duplicate timestamps in %s", dup.sum(), path
            )
            df = df[~dup]

        # DST-fold duplicates: after dedup the index should be monotonic
        # — log any residual
        if not df.index.is_monotonic_increasing:
            logger.warning("_load_bars: index not monotonic after dedup; forcing sort")
            df = df.sort_index()

        # Gap detection (NFR13 — never forward-fill)
        if len(df) > 1:
            diffs = df.index.to_series().diff().dropna()
            gap_threshold = pd.Timedelta("5min")
            gaps = diffs[diffs > gap_threshold]
            if not gaps.empty:
                logger.info(
                    "_load_bars: %d data gaps (>5 min) detected in %s", len(gaps), path
                )
                for ts, gap in gaps.items():
                    logger.debug("  gap at %s: %s", ts, gap)

        return df

    def run(self) -> list[TradeRecord]:
        """Run the deterministic simulation (AC #2, #4, #5).

        Returns
        -------
        list[TradeRecord]
            One record per completed trade, in chronological order.
        """
        bars = self._load_bars(self.csv_path)
        n = len(bars)
        config = self.config
        trades: list[TradeRecord] = []

        # Pre-compute full H1 once (O(n)) and slice at each boundary (O(log n))
        # rather than re-running resample_to_h1 on a rolling 3000-bar buffer each hour.
        full_h1 = resample_to_h1(bars)
        full_m15 = resample_to_m15(bars)
        col_low: int = cast(int, bars.columns.get_loc("low"))
        col_high: int = cast(int, bars.columns.get_loc("high"))

        # ── Active-trade state ──────────────────────────────────────────
        active: EntryDecision | None = None
        active_ts: pd.Timestamp | None = None
        active_gap: float = 0.0
        active_sweep_ago: int = 0
        active_kz: bool = False
        active_m15: bool = False
        active_vol_pct: float = 0.0
        pending: bool = False
        pending_bars: int = 0
        bars_held: int = 0

        # ── Daily circuit-breaker state ──────────────────────────────────
        daily_pnl: float = 0.0
        daily_halted: bool = False
        last_date: date | None = None

        # ── H1 structure state (refreshed once per H1 boundary) ──────────
        h1_bars: pd.DataFrame | None = None
        h1_atr: float = 0.0
        vol_ok_cached: bool = True
        sweep_cached: SweepSignal | None = None
        last_h1_ts: pd.Timestamp | None = None

        for i in range(n):
            if i > 0 and i % 1000 == 0:
                logger.info(
                    "Backtest progress: %d / %d bars (%.1f%%)",
                    i,
                    n,
                    100.0 * i / n,
                )

            bar_ts: pd.Timestamp = bars.index[i]

            # ── H1 boundary: slice pre-computed H1 and refresh cached state ──
            h1_boundary = bar_ts.replace(minute=0, second=0, microsecond=0)
            if h1_boundary != last_h1_ts:
                last_h1_ts = h1_boundary
                # All completed H1 bars before the current forming hour
                h1_idx = int(full_h1.index.searchsorted(h1_boundary))
                h1_start = max(0, h1_idx - _H1_BUFFER_BARS // 60)
                h1_slice = full_h1.iloc[h1_start:h1_idx]
                h1_bars = h1_slice if len(h1_slice) > 0 else None
                h1_atr = self._compute_h1_atr(h1_bars) if h1_bars is not None else 0.0
                # Cache vol-regime and sweep verdicts (invariant within the hour)
                if h1_bars is not None and len(h1_bars) >= 20:
                    try:
                        vol_ok_cached = volatility_regime_filter(h1_bars, config)
                    except ValueError:
                        vol_ok_cached = True
                else:
                    vol_ok_cached = True
                min_rows = config.h1_sweep_lookback + 5
                if h1_bars is not None and len(h1_bars) >= min_rows:
                    try:
                        sweep_cached = detect_liquidity_sweep(h1_bars, config)
                    except ValueError:
                        sweep_cached = None
                else:
                    sweep_cached = None

            # ── Step 1: Advance active trade ─────────────────────────────
            bar = bars.iloc[i]
            if active is not None:
                if pending:
                    pending_bars += 1
                    # Limit-fill: bar range must cross entry_price
                    if active.direction == Direction.BEARISH:
                        filled = float(bar["high"]) >= active.entry_price
                    else:
                        filled = float(bar["low"]) <= active.entry_price

                    if filled:
                        pending = False
                        bars_held = 0
                        # Check TP/SL on fill bar (0 bars held → time-stop impossible)
                        exit_dec = check_exit(bar, active, 0, config)
                        if exit_dec is not None:
                            self._append_trade(
                                trades,
                                active,
                                active_ts,
                                bar_ts,
                                exit_dec,
                                active_gap,
                                active_sweep_ago,
                                active_kz,
                                active_m15,
                                active_vol_pct,
                            )
                            daily_pnl += trades[-1].pnl_usd
                            active = None
                            bars_held = 0
                        # Either way: do NOT enter a new trade this bar
                    elif pending_bars >= config.max_pending_bars:
                        active = None
                        pending = False
                        pending_bars = 0
                        # Fall through to entry detection
                    else:
                        continue  # still pending

                if active is not None and not pending:
                    # Active (filled): check TP/SL/time-stop
                    bars_held += 1
                    exit_dec = check_exit(bar, active, bars_held, config)
                    if exit_dec is not None:
                        self._append_trade(
                            trades,
                            active,
                            active_ts,
                            bar_ts,
                            exit_dec,
                            active_gap,
                            active_sweep_ago,
                            active_kz,
                            active_m15,
                            active_vol_pct,
                        )
                        daily_pnl += trades[-1].pnl_usd
                        active = None
                        bars_held = 0
                        # Fall through to entry detection (mirrors reference behavior)
                    else:
                        continue  # still active

            # ── Step 2: Entry detection ──────────────────────────────────
            if active is not None:
                continue  # filled but not closed this bar

            if i < 20:
                continue  # need ≥ 20 bars for ATR

            if bar_ts.weekday() == 1:  # Tuesday filter
                continue

            # Daily circuit-breaker
            bar_date = bar_ts.date()
            if last_date != bar_date:
                daily_pnl = 0.0
                daily_halted = False
                last_date = bar_date
            if daily_halted:
                continue
            if daily_pnl <= config.max_daily_loss:
                daily_halted = True
                continue

            # Volatility regime gate (cached per H1)
            if not vol_ok_cached:
                continue

            # H1 sweep gate (cached per H1)
            sweep = sweep_cached
            if sweep is None:
                continue

            # Bearish-only gate
            if config.bearish_only and sweep.direction != Direction.BEARISH:
                continue

            # Fast FVG pre-check: skip bars where no FVG of the required direction
            # can possibly exist.  Only applied when the sweep direction is known.
            # Guard is direction-aware so it stays correct when bearish_only=False
            # (Epic 2) and a bullish sweep is active.
            if i < 2:
                continue
            c1_low: float = cast(float, bars.iat[i - 2, col_low])
            c3_high: float = cast(float, bars.iat[i, col_high])
            if sweep.direction == Direction.BEARISH and c1_low <= c3_high:
                continue  # no bearish FVG possible
            if sweep.direction == Direction.BULLISH:
                c1_high: float = cast(float, bars.iat[i - 2, col_high])
                c3_low: float = cast(float, bars.iat[i, col_low])
                if c3_low <= c1_high:
                    continue  # no bullish FVG possible

            # Full FVG detection (last 20 bars — enough for _atr's 20-bar window)
            m1_buf = bars.iloc[max(0, i - 19) : i + 1]
            try:
                fvg = detect_fvg(m1_buf, config, h1_atr)
            except ValueError:
                continue
            if fvg is None:
                continue

            kz = kill_zone_filter(bar_ts, config)
            if config.enable_kill_zone_filter and not kz:
                continue  # outside kill zone — skip this entry candidate

            # M15 confirmation gate
            m15_ok = True
            if config.m15_confirmation:
                m15_idx = int(full_m15.index.searchsorted(bar_ts))
                m15_slice = full_m15.iloc[:m15_idx]
                if len(m15_slice) >= 1:
                    m15_ok = check_m15_confirmation(sweep, m15_slice).confirmed
                else:
                    m15_ok = False

            # Entry decision
            entry = make_entry_decision(
                sweep, fvg, config, vol_ok=vol_ok_cached, m15_conf=m15_ok
            )
            if entry is None:
                continue

            # Arm pending trade
            active = entry
            active_ts = bar_ts
            active_gap = fvg.gap_size
            active_sweep_ago = sweep.bars_ago
            active_kz = kz
            active_m15 = m15_ok
            active_vol_pct = (
                self._compute_vol_pct(h1_bars, config) if h1_bars is not None else 0.0
            )
            pending = True
            pending_bars = 0

        logger.info("Backtest complete: %d trades from %d bars", len(trades), n)
        return trades

    def save_outputs(self, trades: list[TradeRecord]) -> tuple[Path, Path]:
        """Write trade log and equity curve to ``data/reports/`` (AC #3, NFR17).

        Returns
        -------
        tuple[Path, Path]
            ``(trade_csv_path, equity_csv_path)``
        """
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # wall-clock for filenames only

        # Trade log
        trade_path = _REPORTS_DIR / f"backtest_{stamp}.csv"
        fields = [
            "timestamp_entry",
            "timestamp_exit",
            "direction",
            "entry_price",
            "exit_price",
            "tp_price",
            "sl_price",
            "gap_size",
            "pnl_usd",
            "exit_reason",
            "h1_sweep_bars_ago",
            "m15_confirmed",
            "kill_zone_active",
            "vol_regime_pct",
            "contracts",
        ]
        with open(trade_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for t in trades:
                row = {
                    "timestamp_entry": t.timestamp_entry.isoformat(),
                    "timestamp_exit": t.timestamp_exit.isoformat(),
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "tp_price": t.tp_price,
                    "sl_price": t.sl_price,
                    "gap_size": t.gap_size,
                    "pnl_usd": t.pnl_usd,
                    "exit_reason": t.exit_reason,
                    "h1_sweep_bars_ago": t.h1_sweep_bars_ago,
                    "m15_confirmed": t.m15_confirmed,
                    "kill_zone_active": t.kill_zone_active,
                    "vol_regime_pct": round(t.vol_regime_pct, 4),
                    "contracts": t.contracts,
                }
                w.writerow(row)
        logger.info("Trade log → %s (%d trades)", trade_path, len(trades))

        # Equity curve: daily cumulative P&L
        eq_path = _REPORTS_DIR / f"equity_curve_{stamp}.csv"
        by_day: dict[str, float] = {}
        for t in trades:
            day = t.timestamp_entry.strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0.0) + t.pnl_usd
        cum = 0.0
        eq_rows: list[dict[str, object]] = []
        for day in sorted(by_day):
            cum += by_day[day]
            eq_rows.append(
                {
                    "date": day,
                    "daily_pnl": round(by_day[day], 2),
                    "cumulative_pnl": round(cum, 2),
                }
            )
        if eq_rows:
            with open(eq_path, "w", newline="") as fh:
                w2 = csv.DictWriter(
                    fh, fieldnames=["date", "daily_pnl", "cumulative_pnl"]
                )
                w2.writeheader()
                w2.writerows(eq_rows)
        logger.info("Equity curve → %s", eq_path)

        return trade_path, eq_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_trade(
        self,
        trades: list[TradeRecord],
        active: EntryDecision,
        ts_entry: pd.Timestamp | None,
        ts_exit: pd.Timestamp,
        exit_dec: ExitDecision,
        gap: float,
        sweep_ago: int,
        kz: bool,
        m15_ok: bool,
        vol_pct: float,
    ) -> None:
        assert ts_entry is not None, "_append_trade called with null ts_entry"
        ep = exit_dec.exit_price
        if active.direction == Direction.BEARISH:
            points = active.entry_price - ep
        else:
            points = ep - active.entry_price
        commission = self.config.commission_per_roundtrip
        pnl = points * POINT_VALUE_USD * active.contracts - commission
        trades.append(
            TradeRecord(
                timestamp_entry=ts_entry,
                timestamp_exit=ts_exit,
                direction=active.direction.value,
                entry_price=active.entry_price,
                exit_price=ep,
                tp_price=active.tp_price,
                sl_price=active.sl_price,
                gap_size=gap,
                pnl_usd=round(pnl, 2),
                exit_reason=exit_dec.reason.value,
                h1_sweep_bars_ago=sweep_ago,
                m15_confirmed=m15_ok,
                kill_zone_active=kz,
                vol_regime_pct=vol_pct,
                contracts=active.contracts,
            )
        )

    @staticmethod
    def _compute_h1_atr(h1_bars: pd.DataFrame) -> float:
        """20-bar mean ATR from H1 bars via strategy_core.calc_atr."""
        if h1_bars is None or len(h1_bars) < 2:
            return 0.0
        return calc_atr(h1_bars)

    @staticmethod
    def _compute_vol_pct(h1_bars: pd.DataFrame, config: StrategyConfig) -> float:
        """ATR percentile rank from H1 bars — recomputed via the same logic as
        volatility_regime_filter so TradeRecord.vol_regime_pct is consistent with
        the filter decision.  Returns 0.0 if history is insufficient."""
        if h1_bars is None or len(h1_bars) < 2:
            return 0.0
        h = h1_bars["high"].to_numpy(dtype=float)
        lo = h1_bars["low"].to_numpy(dtype=float)
        c = h1_bars["close"].to_numpy(dtype=float)
        prev_c = np.roll(c, 1).astype(float)
        prev_c[0] = np.nan
        tr = np.where(
            np.isnan(prev_c),
            h - lo,
            np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c))),
        )
        atr_series = pd.Series(tr).rolling(20, min_periods=5).mean()
        atr_history = [v for v in atr_series.dropna() if v > 0]
        atr_history = atr_history[-config.vol_regime_lookback :]
        if len(atr_history) < 20:
            return 0.0
        current_atr = atr_history[-1]
        return sum(1 for v in atr_history if v < current_atr) / len(atr_history)

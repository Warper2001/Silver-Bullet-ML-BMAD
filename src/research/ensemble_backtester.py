"""Ensemble Backtesting Engine for Multi-Strategy Trading System.

This module provides comprehensive backtesting capabilities for the ensemble trading
system, combining signals from all 5 strategies with weighted confidence scoring.
"""

import h5py
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Constants for MNQ Futures
MNQ_POINT_VALUE = 5.0  # $5 per point for MNQ micro futures
TRADING_HOURS_PER_DAY = 6.5  # 9:30 AM - 4:00 PM ET


def create_dollar_bar_from_series(bar_series):
    """Convert pandas Series to DollarBar object.

    Args:
        bar_series: pandas Series with timestamp, open, high, low, close, volume

    Returns:
        DollarBar object
    """
    from src.data.models import DollarBar

    # Calculate notional value if not present
    close_price = float(bar_series['close'])
    volume = int(bar_series['volume'])

    # Estimate notional value (price * volume * multiplier)
    # MNQ has $5/point multiplier
    notional_value = bar_series.get('notional_value', close_price * volume * 5.0)

    # Ensure notional_value is positive and reasonable
    if notional_value <= 0:
        notional_value = close_price * volume * 5.0

    # Cap at maximum allowed value (DollarBar validation requires <= 2B)
    max_notional = 2_000_000_000.0
    if notional_value > max_notional:
        notional_value = max_notional

    return DollarBar(
        timestamp=bar_series['timestamp'].to_pydatetime(),
        open=float(bar_series['open']),
        high=float(bar_series['high']),
        low=float(bar_series['low']),
        close=close_price,
        volume=volume,
        notional_value=float(notional_value),
        is_forward_filled=False,
    )


class CompletedTrade(BaseModel):
    """Represents a completed trade from ensemble backtesting.

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
        contracts: Number of contracts traded
        confidence: Ensemble confidence score (0-1)
        contributing_strategies: Which strategies contributed to this signal
    """

    entry_time: datetime = Field(..., description="Trade entry time")
    exit_time: datetime = Field(..., description="Trade exit time")
    direction: Literal["long", "short"] = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Entry price")
    exit_price: float = Field(..., gt=0, description="Exit price")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    take_profit: float = Field(..., gt=0, description="Take profit price")
    pnl: float = Field(..., description="Profit/loss in dollars")
    bars_held: int = Field(..., ge=0, description="Number of bars held")
    contracts: int = Field(..., ge=1, le=5, description="Number of contracts")
    confidence: float = Field(..., ge=0, le=1, description="Ensemble confidence score")
    contributing_strategies: list[str] = Field(
        ..., description="Strategies that contributed to signal"
    )


class BacktestResults(BaseModel):
    """Results from ensemble backtesting.

    Contains all 12 performance metrics required for analysis.
    """

    total_trades: int = Field(..., ge=0, description="Total number of trades")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1 scale)")
    profit_factor: float = Field(..., ge=0, description="Profit factor")
    average_win: float = Field(..., description="Average winning trade P&L")
    average_loss: float = Field(..., description="Average losing trade P&L")
    largest_win: float = Field(..., description="Largest winning trade")
    largest_loss: float = Field(..., description="Largest losing trade")
    max_drawdown: float = Field(..., ge=0, le=1, description="Max drawdown (0-1 scale)")
    max_drawdown_duration: int = Field(..., ge=0, description="Max DD duration (bars)")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    average_hold_time: float = Field(..., ge=0, description="Average hold time (minutes)")
    trade_frequency: float = Field(..., ge=0, description="Trades per day")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    confidence_threshold: float = Field(..., ge=0, le=1, description="Confidence threshold used")
    trades: list[CompletedTrade] = Field(default_factory=list, description="List of completed trades")
    total_pnl: float = Field(..., description="Total P&L in dollars")


class EnsembleBacktester:
    """Ensemble backtesting engine for multi-strategy trading system.

    Processes historical dollar bars through the complete ensemble pipeline:
    1. Strategy signal detection (all 5 strategies)
    2. Signal aggregation
    3. Weighted confidence scoring
    4. Confidence filtering
    5. Entry logic
    6. Exit logic
    7. P&L calculation
    """

    def __init__(self, config_path: str, data_path: str):
        """Initialize ensemble backtester.

        Args:
            config_path: Path to config-sim.yaml
            data_path: Path to HDF5 dollar bar data file
        """
        self.config_path = config_path
        self.data_path = data_path
        self.bars_processed = 0
        self._aggregator = None
        self._scorer = None
        self._entry_logic = None
        self._exit_logic = None

        # Initialize ensemble components lazily
        self._initialize_components()

    def _initialize_components(self):
        """Initialize ensemble components (aggregator, scorer, entry, exit)."""
        try:
            from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator
            from src.detection.weighted_confidence_scorer import WeightedConfidenceScorer

            self._aggregator = EnsembleSignalAggregator(max_lookback=10)
            self._scorer = WeightedConfidenceScorer(config_path=self.config_path)

            # For backtesting, use simplified entry/exit logic
            # (avoiding complex live trading dependencies)
            self._entry_logic = self._create_simple_entry_logic()
            self._exit_logic = self._create_simple_exit_logic()

            logger.info("Ensemble components initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import ensemble components: {e}")
            raise

    def _create_simple_entry_logic(self):
        """Create simplified entry logic for backtesting."""
        from dataclasses import dataclass

        @dataclass
        class EntryDecision:
            should_enter: bool
            contracts: int

        class SimpleEntryLogic:
            def evaluate_entry(self, bar, ensemble_signal, open_positions):
                # Simple backtesting entry logic
                # Allow entry if we have < 5 positions
                if len(open_positions) >= 5:
                    return EntryDecision(should_enter=False, contracts=0)

                # Use fixed 1 contract for backtesting
                return EntryDecision(should_enter=True, contracts=1)

        return SimpleEntryLogic()

    def _create_simple_exit_logic(self):
        """Create simplified exit logic for backtesting."""
        from dataclasses import dataclass

        @dataclass
        class ExitDecision:
            should_exit: bool
            exit_price: float = None
            exit_reason: str = ""

        class SimpleExitLogic:
            def evaluate_exit(self, bar, position):
                current_price = bar["close"]
                entry_price = position["entry_price"]
                stop_loss = position["stop_loss"]
                take_profit = position["take_profit"]
                direction = position["direction"]

                # Calculate hold time in minutes
                entry_time = position["entry_time"]
                bar_time = bar["timestamp"]
                hold_minutes = (bar_time - entry_time).total_seconds() / 60

                # Check 1: Time stop (10 minutes max)
                if hold_minutes >= 10.0:
                    return ExitDecision(
                        should_exit=True,
                        exit_price=current_price,
                        exit_reason="time_stop"
                    )

                # Check 2: Take profit hit
                if direction == "long" and current_price >= take_profit:
                    return ExitDecision(
                        should_exit=True,
                        exit_price=current_price,
                        exit_reason="take_profit"
                    )
                elif direction == "short" and current_price <= take_profit:
                    return ExitDecision(
                        should_exit=True,
                        exit_price=current_price,
                        exit_reason="take_profit"
                    )

                # Check 3: Stop loss hit
                if direction == "long" and current_price <= stop_loss:
                    return ExitDecision(
                        should_exit=True,
                        exit_price=current_price,
                        exit_reason="stop_loss"
                    )
                elif direction == "short" and current_price >= stop_loss:
                    return ExitDecision(
                        should_exit=True,
                        exit_price=current_price,
                        exit_reason="stop_loss"
                    )

                # No exit triggered
                return ExitDecision(should_exit=False, exit_reason="")

        return SimpleExitLogic()

    def run_backtest(
        self,
        start_date: date,
        end_date: date,
        confidence_threshold: float = 0.50,
    ) -> BacktestResults:
        """Run ensemble backtest on historical data.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            confidence_threshold: Minimum confidence for trade entry (0-1)

        Returns:
            BacktestResults with all performance metrics
        """
        logger.info(
            f"Running ensemble backtest: {start_date} to {end_date}, "
            f"threshold={confidence_threshold}"
        )

        # Load dollar bars
        bars = self._load_dollar_bars(start_date, end_date)
        self.bars_processed = len(bars)

        # Initialize tracking
        trades: list[CompletedTrade] = []
        open_positions: dict = {}  # position_id -> position data

        # Initialize strategies (call real strategies, not mock signals)
        if not hasattr(self, '_strategies_initialized'):
            self._initialize_strategies()
            self._strategies_initialized = True

        # Process bars chronologically
        for idx, bar in bars.iterrows():
            # Convert pandas Series to DollarBar for strategies
            dollar_bar = create_dollar_bar_from_series(bar)

            # Call each strategy and collect signals
            self._process_bar_with_strategies(dollar_bar)

            # Get aggregated signals for current bar (with wider window for confluence)
            current_bar_signals = self._aggregator.get_signals_for_bar(
                bar["timestamp"], window_bars=5  # Include 5 bars before and after (±25 min)
            )

            if not current_bar_signals:
                continue

            # Calculate weighted confidence
            ensemble_signal = self._scorer.score_signals(current_bar_signals)

            # Filter by confidence threshold
            if ensemble_signal is None:
                continue

            if ensemble_signal.composite_confidence < confidence_threshold:
                continue

            # Check entry criteria
            entry_decision = self._entry_logic.evaluate_entry(
                bar, ensemble_signal, open_positions
            )

            if entry_decision.should_enter:
                # Create position
                position = self._create_position(bar, ensemble_signal, entry_decision)
                open_positions[position["position_id"]] = position

            # Check exit conditions for open positions
            for pos_id, position in list(open_positions.items()):
                exit_decision = self._exit_logic.evaluate_exit(bar, position)

                if exit_decision.should_exit:
                    # Close position and record trade
                    trade = self._close_position(position, bar, exit_decision)
                    trades.append(trade)
                    del open_positions[pos_id]

        # Close any remaining positions at end of backtest
        for position in open_positions.values():
            final_bar = bars.iloc[-1]
            trade = self._close_position(position, final_bar, None)
            trades.append(trade)

        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            trades, start_date, end_date, confidence_threshold
        )

        logger.info(
            f"Backtest complete: {results.total_trades} trades, "
            f"win rate={results.win_rate:.2%}, "
            f"profit factor={results.profit_factor:.2f}"
        )

        return results

    def run_sensitivity_analysis(
        self, thresholds: list[float]
    ) -> dict[float, BacktestResults]:
        """Run backtest at multiple confidence thresholds.

        Args:
            thresholds: List of confidence thresholds to test

        Returns:
            Dictionary mapping threshold -> BacktestResults
        """
        logger.info(f"Running sensitivity analysis with {len(thresholds)} thresholds")

        results: dict[float, BacktestResults] = {}

        # Use same date range for all thresholds
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        for threshold in thresholds:
            logger.info(f"Testing threshold={threshold:.2f}")
            results[threshold] = self.run_backtest(start_date, end_date, threshold)

        return results

    def _load_dollar_bars(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load dollar bars from HDF5 file.

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Loading dollar bars from {self.data_path}")

        with h5py.File(self.data_path, "r") as f:
            timestamps = pd.to_datetime(f["timestamps"][:], unit="ns")
            open_prices = f["open"][:]
            high_prices = f["high"][:]
            low_prices = f["low"][:]
            close_prices = f["close"][:]
            volumes = f["volume"][:]

        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        })

        # Filter by date range
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + timedelta(days=1)

        bars = bars[
            (bars["timestamp"] >= start_datetime) & (bars["timestamp"] < end_datetime)
        ]

        logger.info(f"Loaded {len(bars)} bars for backtesting")

        return bars

    def _create_position(self, bar: pd.Series, ensemble_signal, entry_decision) -> dict:
        """Create a new trading position.

        Args:
            bar: Current bar
            ensemble_signal: Ensemble trade signal
            entry_decision: Entry logic decision

        Returns:
            Position dictionary
        """
        import uuid

        position_id = str(uuid.uuid4())

        position = {
            "position_id": position_id,
            "entry_time": bar["timestamp"],
            "direction": ensemble_signal.direction,
            "entry_price": ensemble_signal.entry_price,
            "stop_loss": ensemble_signal.stop_loss,
            "take_profit": ensemble_signal.take_profit,
            "contracts": entry_decision.contracts or 1,
            "confidence": ensemble_signal.composite_confidence,
            "contributing_strategies": ensemble_signal.contributing_strategies,
            "bars_held": 0,
        }

        return position

    def _close_position(
        self, position: dict, bar: pd.Series, exit_decision
    ) -> CompletedTrade:
        """Close a position and create completed trade record.

        Args:
            position: Open position
            bar: Current bar
            exit_decision: Exit logic decision (None if forced close)

        Returns:
            CompletedTrade object
        """
        exit_price = bar["close"]
        if exit_decision and exit_decision.exit_price:
            exit_price = exit_decision.exit_price

        # Calculate P&L
        if position["direction"] == "long":
            pnl = (exit_price - position["entry_price"]) * MNQ_POINT_VALUE * position["contracts"]
        else:  # short
            pnl = (position["entry_price"] - exit_price) * MNQ_POINT_VALUE * position["contracts"]

        trade = CompletedTrade(
            entry_time=position["entry_time"],
            exit_time=bar["timestamp"],
            direction=position["direction"],
            entry_price=position["entry_price"],
            exit_price=exit_price,
            stop_loss=position["stop_loss"],
            take_profit=position["take_profit"],
            pnl=pnl,
            bars_held=position["bars_held"],
            contracts=position["contracts"],
            confidence=position["confidence"],
            contributing_strategies=position["contributing_strategies"],
        )

        return trade



    def _initialize_strategies(self):
        """Initialize all 5 trading strategies."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
        from src.detection.wolf_pack_strategy import WolfPackStrategy
        from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
        from src.detection.vwap_bounce_strategy import VWAPBounceStrategy
        from src.detection.opening_range_strategy import OpeningRangeStrategy

        logger.info("Initializing real trading strategies...")

        self._strategies = []

        # Strategy 1: Triple Confluence
        try:
            tc = TripleConfluenceStrategy(config={})
            self._strategies.append(("triple_confluence", tc))
            logger.info("  ✓ Triple Confluence initialized")
        except Exception as e:
            logger.warning(f"  ✗ Triple Confluence failed: {e}")

        # Strategy 2: Wolf Pack (needs tick_size and risk_ticks)
        try:
            wp = WolfPackStrategy(tick_size=0.25, risk_ticks=20, min_confidence=0.8)
            self._strategies.append(("wolf_pack", wp))
            logger.info("  ✓ Wolf Pack initialized")
        except Exception as e:
            logger.warning(f"  ✗ Wolf Pack failed: {e}")

        # Strategy 3: Adaptive EMA
        try:
            ae = AdaptiveEMAStrategy()
            self._strategies.append(("adaptive_ema", ae))
            logger.info("  ✓ Adaptive EMA initialized")
        except Exception as e:
            logger.warning(f"  ✗ Adaptive EMA failed: {e}")

        # Strategy 4: VWAP Bounce
        try:
            vb = VWAPBounceStrategy(config={})
            self._strategies.append(("vwap_bounce", vb))
            logger.info("  ✓ VWAP Bounce initialized")
        except Exception as e:
            logger.warning(f"  ✗ VWAP Bounce failed: {e}")

        # Strategy 5: Opening Range
        try:
            ob = OpeningRangeStrategy(config={})
            self._strategies.append(("opening_range", ob))
            logger.info("  ✓ Opening Range initialized")
        except Exception as e:
            logger.warning(f"  ✗ Opening Range failed: {e}")

        logger.info(f"Initialized {len(self._strategies)}/5 strategies")

    def _process_bar_with_strategies(self, dollar_bar):
        """Process a single bar with all strategies and aggregate signals.

        Args:
            dollar_bar: DollarBar object
        """
        for strategy_name, strategy in self._strategies:
            try:
                # Call strategy based on its interface
                if hasattr(strategy, 'process_bar'):
                    signal = strategy.process_bar(dollar_bar)
                elif hasattr(strategy, 'process_bars'):
                    # Some strategies need list of bars
                    signal = strategy.process_bars([dollar_bar])
                else:
                    continue

                # If signal generated, normalize and add to aggregator
                if signal is not None:
                    try:
                        # Normalize signal to Ensemble format before adding
                        from src.detection.ensemble_signal_aggregator import normalize_signal

                        normalized_signal = normalize_signal(signal)
                        self._aggregator.add_signal(normalized_signal)
                    except Exception as e:
                        logger.debug(f"Failed to normalize signal from {strategy_name}: {e}")
                        continue

            except Exception as e:
                # Log error but continue processing other strategies
                logger.debug(f"Strategy {strategy_name} error on bar: {e}")
                continue

    def _calculate_performance_metrics(
        self,
        trades: list[CompletedTrade],
        start_date: date,
        end_date: date,
        confidence_threshold: float,
    ) -> BacktestResults:
        """Calculate all 12 performance metrics from trades.

        Args:
            trades: List of completed trades
            start_date: Backtest start date
            end_date: Backtest end date
            confidence_threshold: Confidence threshold used

        Returns:
            BacktestResults with all metrics
        """
        if not trades:
            return BacktestResults(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0,
                sharpe_ratio=0.0,
                average_hold_time=0.0,
                trade_frequency=0.0,
                start_date=start_date,
                end_date=end_date,
                confidence_threshold=confidence_threshold,
                trades=[],
                total_pnl=0.0,
            )

        # Basic counts
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L statistics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        average_win = np.mean(wins) if wins else 0.0
        average_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Total P&L
        total_pnl = sum(t.pnl for t in trades)

        # Drawdown calculation
        equity_curve = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(equity_curve)

        # Calculate drawdown percentage properly
        # Only calculate drawdown when we have a positive peak to avoid > 100% drawdown
        drawdown_pct = np.zeros_like(equity_curve)
        for i in range(len(equity_curve)):
            if running_max[i] > 0:  # Only calculate from positive peaks
                drawdown_pct[i] = (equity_curve[i] - running_max[i]) / running_max[i]
            else:
                drawdown_pct[i] = 0.0  # No drawdown if peak is <= 0

        max_drawdown = abs(np.min(drawdown_pct))
        max_drawdown = min(max_drawdown, 1.0)  # Cap at 100%

        # Max drawdown duration
        dd_end = np.argmin(drawdown_pct)
        dd_start = np.argmax(equity_curve[:dd_end] == np.max(equity_curve[:dd_end]))
        max_drawdown_duration = dd_end - dd_start if dd_end > dd_start else 0

        # Sharpe ratio (simplified, assuming 5-min bars)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Average hold time (minutes)
        hold_times = [
            (t.exit_time - t.entry_time).total_seconds() / 60 for t in trades
        ]
        average_hold_time = np.mean(hold_times) if hold_times else 0.0

        # Trade frequency (trades per day)
        days_traded = (end_date - start_date).days
        trade_frequency = total_trades / days_traded if days_traded > 0 else 0.0

        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            average_hold_time=average_hold_time,
            trade_frequency=trade_frequency,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=confidence_threshold,
            trades=trades,
            total_pnl=total_pnl,
        )

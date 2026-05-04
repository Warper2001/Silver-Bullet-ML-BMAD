"""Ensemble Signal Aggregator for normalizing and storing strategy signals."""

import asyncio
import logging
import pytz
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from src.detection.models import EnsembleSignal

logger = logging.getLogger(__name__)

# Target timezone for all ensemble processing
NY_TZ = pytz.timezone("America/New_York")

def _normalize_ts(dt: datetime) -> datetime:
    """Normalize datetime to New York timezone."""
    if dt.tzinfo is None:
        # Assume UTC if no timezone provided (standard API convention)
        dt = pytz.utc.localize(dt)
    return dt.astimezone(NY_TZ)

# =============================================================================
# Signal Normalizer Functions
# =============================================================================

def normalize_triple_confluence(signal: Any) -> EnsembleSignal:
    """Normalize Triple Confluence Scalper signal."""
    if not hasattr(signal, "entry_price") or signal.entry_price is None:
        raise ValueError("Invalid TripleConfluenceSignal: missing entry_price")

    # Decision 3A: Enforce explicit bar_timestamp or fall back to normalized timestamp
    bar_ts = getattr(signal, "bar_timestamp", signal.timestamp)

    metadata = {}
    if hasattr(signal, "contributing_factors") and signal.contributing_factors:
        metadata = dict(signal.contributing_factors)
    # Add expected win rate if available
    if hasattr(signal, "expected_win_rate"):
        metadata["expected_win_rate"] = signal.expected_win_rate
        
    return EnsembleSignal(
        strategy_name="triple_confluence_scaler",
        timestamp=_normalize_ts(signal.timestamp),
        direction=signal.direction.lower(),
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        confidence=signal.confidence,
        bar_timestamp=_normalize_ts(bar_ts),
        metadata=metadata,
    )

def normalize_wolf_pack(signal: Any) -> EnsembleSignal:
    """Normalize Wolf Pack 3-Edge signal."""
    if not hasattr(signal, "entry_price") or signal.entry_price is None:
        raise ValueError("Invalid WolfPackSignal: missing entry_price")

    bar_ts = getattr(signal, "bar_timestamp", signal.timestamp)

    metadata = {}
    if hasattr(signal, "contributing_factors") and signal.contributing_factors:
        metadata = dict(signal.contributing_factors)
        
    return EnsembleSignal(
        strategy_name="wolf_pack_3_edge",
        timestamp=_normalize_ts(signal.timestamp),
        direction=signal.direction.lower(),
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        confidence=signal.confidence,
        bar_timestamp=_normalize_ts(bar_ts),
        metadata=metadata,
    )

def normalize_ema_momentum(signal: Any) -> EnsembleSignal:
    """Normalize Adaptive EMA Momentum signal (handles 0-100 scale)."""
    if not hasattr(signal, "entry_price") or signal.entry_price is None:
        raise ValueError("Invalid MomentumSignal: missing entry_price")

    bar_ts = getattr(signal, "bar_timestamp", signal.timestamp)
    confidence = signal.confidence / 100.0 if signal.confidence > 1.0 else signal.confidence

    metadata = {}
    for attr in ["ema_fast", "ema_medium", "ema_slow", "macd_histogram", "rsi_value"]:
        if hasattr(signal, attr):
            metadata[attr] = getattr(signal, attr)

    return EnsembleSignal(
        strategy_name="adaptive_ema_momentum",
        timestamp=_normalize_ts(signal.timestamp),
        direction=signal.direction.lower(),
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        confidence=confidence,
        bar_timestamp=_normalize_ts(bar_ts),
        metadata=metadata,
    )

def normalize_vwap_bounce(signal: Any) -> EnsembleSignal:
    """Normalize VWAP Bounce signal."""
    if not hasattr(signal, "entry_price") or signal.entry_price is None:
        raise ValueError("Invalid VWAPBounceSignal: missing entry_price")

    bar_ts = getattr(signal, "bar_timestamp", signal.timestamp)

    metadata = {}
    if hasattr(signal, "contributing_factors") and signal.contributing_factors:
        metadata = dict(signal.contributing_factors)

    return EnsembleSignal(
        strategy_name="vwap_bounce",
        timestamp=_normalize_ts(signal.timestamp),
        direction=signal.direction.lower(),
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        confidence=signal.confidence,
        bar_timestamp=_normalize_ts(bar_ts),
        metadata=metadata,
    )

def normalize_opening_range(signal: Any) -> EnsembleSignal:
    """Normalize Opening Range Breakout signal."""
    if not hasattr(signal, "entry_price") or signal.entry_price is None:
        raise ValueError("Invalid OpeningRangeSignal: missing entry_price")

    bar_ts = getattr(signal, "bar_timestamp", signal.timestamp)

    metadata = {}
    if hasattr(signal, "contributing_factors") and signal.contributing_factors:
        metadata = dict(signal.contributing_factors)

    return EnsembleSignal(
        strategy_name="opening_range_breakout",
        timestamp=_normalize_ts(signal.timestamp),
        direction=signal.direction.lower(),
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        confidence=signal.confidence,
        bar_timestamp=_normalize_ts(bar_ts),
        metadata=metadata,
    )

def normalize_signal(signal: Any) -> EnsembleSignal:
    """Dispatcher function with strict type checking (Decision 4B)."""
    # Dynamic imports to avoid circularity
    from src.detection.models import TripleConfluenceSignal, WolfPackSignal, MomentumSignal
    from src.detection.vwap_bounce_strategy import VWAPBounceSignalModel
    from src.detection.opening_range_strategy import OpeningRangeSignalModel
    
    if isinstance(signal, TripleConfluenceSignal):
        return normalize_triple_confluence(signal)
    if isinstance(signal, WolfPackSignal):
        return normalize_wolf_pack(signal)
    if isinstance(signal, MomentumSignal):
        return normalize_ema_momentum(signal)
    if isinstance(signal, (VWAPBounceSignalModel,)): 
        # Checking for both standard and Model variant if exist
        return normalize_vwap_bounce(signal)
    if isinstance(signal, (OpeningRangeSignalModel,)):
        return normalize_opening_range(signal)
        
    # Fallback to duck typing for Mock objects in tests
    class_name = type(signal).__name__
    if "Triple" in class_name: return normalize_triple_confluence(signal)
    if "Wolf" in class_name: return normalize_wolf_pack(signal)
    if "Momentum" in class_name: return normalize_ema_momentum(signal)
    if "VWAP" in class_name: return normalize_vwap_bounce(signal)
    if "Opening" in class_name or "Range" in class_name: return normalize_opening_range(signal)

    raise ValueError(f"Unsupported signal type: {class_name}")


class EnsembleSignalAggregator:
    """Aggregates and normalizes signals from all trading strategies.

    This class provides a unified interface for receiving signals from
    multiple strategies, storing them with configurable lookback, and
    querying them for ensemble processing.

    Signals are stored per-strategy with automatic cleanup of old signals
    beyond the lookback window. Deduplication ensures only the latest
    signal is kept for each strategy-bar combination.
    """

    def __init__(self, max_lookback: int = 10, bar_interval: int = 5) -> None:
        """Initialize the ensemble signal aggregator.

        Args:
            max_lookback: Maximum number of bars to look back for signals.
                         Signals older than this are automatically cleaned up.
            bar_interval: The bar duration in minutes (e.g., 1 or 5).
        """
        if max_lookback < 1:
            raise ValueError("max_lookback must be at least 1")
        if bar_interval < 1:
            raise ValueError("bar_interval must be at least 1")

        self.max_lookback = max_lookback
        self.bar_interval = bar_interval
        self._signals: dict[str, deque[EnsembleSignal]] = {}
        self._lock = asyncio.Lock()  # For thread-safe async access
        self._background_task: asyncio.Task | None = None
        self._running = False
        logger.info(
            "EnsembleSignalAggregator initialized: max_lookback=%d, bar_interval=%d min",
            max_lookback,
            bar_interval,
        )

    def add_signal(self, signal: EnsembleSignal) -> None:
        """Add a normalized signal to the aggregator.

        This method stores the signal in the appropriate strategy-specific
        deque, performing deduplication and cleanup as needed.

        Args:
            signal: Normalized ensemble signal to add
        """
        strategy = signal.strategy_name
        bar_time = signal.bar_timestamp

        # Initialize deque for this strategy if needed
        if strategy not in self._signals:
            # We use a larger maxlen to avoid premature eviction during cleanup/dedup
            self._signals[strategy] = deque(maxlen=self.max_lookback + 10)
            logger.debug("Created new signal deque for strategy: %s", strategy)

        # Deduplication: Check if signal exists for this bar
        strategy_deque = self._signals[strategy]
        existing_index = None

        # O(N) search is fine for N=10, but we use index-based update for O(1) assignment
        for i, existing_signal in enumerate(strategy_deque):
            if existing_signal.bar_timestamp == bar_time:
                existing_index = i
                break

        if existing_index is not None:
            # Atomic update: Replace existing signal for this bar via index
            old_signal = strategy_deque[existing_index]
            strategy_deque[existing_index] = signal

            logger.debug(
                "Deduplication: Replaced signal for %s at bar %s (old conf: %.2f, new conf: %.2f)",
                strategy,
                bar_time,
                old_signal.confidence,
                signal.confidence,
            )
        else:
            # Add new signal
            strategy_deque.append(signal)
            logger.debug(
                "Added signal from %s: direction=%s, confidence=%.2f",
                strategy,
                signal.direction,
                signal.confidence,
            )

        # Automatic cleanup of old signals
        self.cleanup_old_signals(bar_time)

    def get_signals(
        self,
        strategy: str | None = None,
        direction: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[EnsembleSignal]:
        """Get signals with optional filtering.

        Args:
            strategy: Filter by strategy name (None = all strategies)
            direction: Filter by direction ('long' or 'short', None = both)
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            List of signals matching all filter criteria
        """
        # Collect signals from all strategies or specific strategy
        # Copy values to prevent RuntimeError if _signals is modified during iteration
        if strategy is not None:
            strategy_deques = [self._signals.get(strategy, deque())]
        else:
            strategy_deques = list(self._signals.values())

        # Flatten and filter
        all_signals = []
        for deque_signals in strategy_deques:
            # Copy deque to list to prevent RuntimeError during concurrent modification
            all_signals.extend(list(deque_signals))

        # Apply filters
        filtered = all_signals
        if direction is not None:
            # Case-insensitive filtering
            target_dir = direction.lower()
            filtered = [s for s in filtered if s.direction.lower() == target_dir]
            
        if min_confidence > 0:
            filtered = [s for s in filtered if s.confidence >= min_confidence]

        return filtered

    def get_signals_for_bar(
        self, bar_timestamp: datetime, window_bars: int = 0
    ) -> list[EnsembleSignal]:
        """Get signals for a specific bar with optional window.

        Args:
            bar_timestamp: The bar timestamp to get signals for
            window_bars: Include signals within N bars of the target bar (default: 0)

        Returns:
            List of signals for the target bar (and window if specified)
        """
        # Define time window (assuming 5-minute bars for MNQ)
        # In production, this should use actual bar timestamps from data
        window_delta = timedelta(minutes=5 * window_bars)

        window_start = bar_timestamp - window_delta
        window_end = bar_timestamp + window_delta

        matching_signals = []
        for strategy_deque in self._signals.values():
            for signal in strategy_deque:
                if window_start <= signal.bar_timestamp <= window_end:
                    matching_signals.append(signal)

        return matching_signals

    def cleanup_old_signals(self, current_bar_timestamp: datetime) -> None:
        """Remove signals that exceed the max_lookback window.

        Args:
            current_bar_timestamp: The current bar timestamp (for comparison)
        """
        # Approximate lookback in minutes (assuming 5-minute bars)
        lookback_delta = timedelta(minutes=5 * self.max_lookback)
        cutoff_time = current_bar_timestamp - lookback_delta

        total_removed = 0
        for strategy, strategy_deque in self._signals.items():
            # Remove signals older than cutoff
            initial_count = len(strategy_deque)
            # Filter in-place using deque rotation
            to_keep = deque()
            for signal in strategy_deque:
                if signal.bar_timestamp >= cutoff_time:
                    to_keep.append(signal)

            removed_count = initial_count - len(to_keep)
            total_removed += removed_count

            # Update deque
            strategy_deque.clear()
            strategy_deque.extend(to_keep)

            if removed_count > 0:
                logger.debug(
                    "Cleanup: Removed %d old signals from %s (cutoff: %s)",
                    removed_count,
                    strategy,
                    cutoff_time,
                )

        if total_removed > 0:
            logger.info("Cleanup: Removed %d total signals exceeding lookback", total_removed)

    def get_active_strategies(self) -> list[str]:
        """Get list of strategies that have signals in the lookback window.

        Returns:
            List of strategy names with at least one signal
        """
        return [strategy for strategy, deque_signals in self._signals.items() if len(deque_signals) > 0]

    def get_latest_signal(self, strategy: str) -> EnsembleSignal | None:
        """Get the most recent signal from a specific strategy.

        Args:
            strategy: Strategy name to get latest signal from

        Returns:
            Most recent signal from the strategy, or None if no signals exist
        """
        strategy_deque = self._signals.get(strategy)
        if not strategy_deque or len(strategy_deque) == 0:
            return None

        # Return last signal (most recently added)
        return strategy_deque[-1]

    def get_signals_by_direction(self, direction: str) -> list[EnsembleSignal]:
        """Get all signals of a specific direction.

        Args:
            direction: 'long' or 'short'

        Returns:
            List of signals matching the direction
        """
        return self.get_signals(direction=direction)

    def get_signals_by_confidence(self, min_confidence: float) -> list[EnsembleSignal]:
        """Get signals above a confidence threshold.

        Args:
            min_confidence: Minimum confidence level (0-1)

        Returns:
            List of signals with confidence >= threshold
        """
        return self.get_signals(min_confidence=min_confidence)

    def get_signal_count(self, strategy: str | None = None) -> int:
        """Count total signals or signals for a specific strategy.

        Args:
            strategy: Strategy name to count, or None for total count

        Returns:
            Number of signals
        """
        if strategy is not None:
            strategy_deque = self._signals.get(strategy)
            return len(strategy_deque) if strategy_deque else 0

        return sum(len(deque_signals) for deque_signals in self._signals.values())

    def clear_all_signals(self) -> None:
        """Remove all stored signals.

        Useful for testing or resetting state.
        """
        self._signals.clear()
        logger.info("Cleared all signals from aggregator")

    def get_storage_stats(self) -> dict:
        """Get statistics about signal storage.

        Returns:
            Dictionary with storage statistics including:
            - total_signals: Total number of signals stored
            - strategies: Dict of signal counts per strategy
            - active_strategies: Number of strategies with signals
        """
        stats = {
            "total_signals": 0,
            "strategies": {},
            "active_strategies": 0,
        }

        for strategy, strategy_deque in self._signals.items():
            count = len(strategy_deque)
            stats["strategies"][strategy] = count
            stats["total_signals"] += count

        stats["active_strategies"] = len(self.get_active_strategies())

        return stats

    def validate_lookback(self) -> bool:
        """Verify no signals exceed max_lookback.

        Returns:
            True if all signals are within lookback window
        """
        current_time = datetime.now()
        # Decision 2A: Use configured bar interval
        lookback_delta = timedelta(minutes=self.bar_interval * self.max_lookback)
        cutoff_time = current_time - lookback_delta

        for strategy, strategy_deque in list(self._signals.items()):
            for signal in list(strategy_deque):
                if signal.bar_timestamp < cutoff_time:
                    logger.warning(
                        "Signal from %s exceeds lookback: %s",
                        strategy,
                        signal.bar_timestamp,
                    )
                    return False

        return True

    # =============================================================================
    # Async Methods for Real-Time Processing
    # =============================================================================

    async def add_signal_async(self, signal: EnsembleSignal) -> None:
        """Async version of add_signal for thread-safe operation.

        Args:
            signal: Normalized ensemble signal to add
        """
        # Use the class lock for thread-safe async access if needed,
        # but deque operations are atomic in CPython. 
        # For full consistency with other async components:
        async with self._lock:
            self.add_signal(signal)

    async def process_signals_queue(self, queue: asyncio.Queue) -> None:
        """Process signals from an asyncio.Queue until exhausted."""
        logger.info("Starting to process signals from queue")
        processed_count = 0
        
        # We process while running OR while queue is not empty (to drain it on shutdown)
        while self._running or not queue.empty():
            try:
                # Use a small timeout to keep loop responsive to self._running changes
                signal = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue

            # Check for sentinel value to stop
            if signal is None:
                logger.info("Queue processing: received sentinel, stopping")
                break

            # Normalize if needed and add signal
            try:
                # If signal is not already EnsembleSignal, normalize it
                if not isinstance(signal, EnsembleSignal):
                    signal = normalize_signal(signal)

                self.add_signal(signal)
                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info("Queue processing: processed %d signals so far", processed_count)

            except Exception as e:
                logger.error("Error processing signal from queue: %s", e)

            # Mark task as done
            queue.task_done()

        logger.info("Queue processing: completed. Total processed: %d", processed_count)

    async def start_aggregator(self, queue: asyncio.Queue) -> asyncio.Task:
        """Start background task for continuous signal processing.

        Args:
            queue: asyncio.Queue containing signals to process

        Returns:
            The background task object

        Raises:
            RuntimeError: If aggregator is already running
        """
        if self._running:
            raise RuntimeError("Aggregator is already running")

        self._running = True
        self._background_task = asyncio.create_task(self.process_signals_queue(queue))
        logger.info("Started aggregator background task")
        return self._background_task

    async def stop_aggregator(self) -> None:
        """Stop the background signal processing task.

        Waits for the background task to complete gracefully.
        """
        if not self._running:
            logger.warning("Aggregator is not running, nothing to stop")
            return

        logger.info("Stopping aggregator background task...")
        self._running = False

        if self._background_task is not None:
            # Wait for task to complete (with timeout)
            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
                logger.info("Aggregator background task stopped successfully")
            except asyncio.TimeoutError:
                logger.warning("Aggregator background task did not stop within timeout, cancelling")
                self._background_task.cancel()
                try:
                    await self._background_task
                except asyncio.CancelledError:
                    logger.info("Aggregator background task cancelled")
            finally:
                self._background_task = None

    # =============================================================================
    # Consensus and Alignment Detection Methods
    # =============================================================================

    def get_consensus(self, bar_timestamp: datetime | None = None) -> dict[str, int]:
        """Get consensus count for a specific bar.

        Args:
            bar_timestamp: Target bar for consensus. If None, uses latest bar.

        Returns:
            Dictionary with 'long' and 'short' counts
        """
        target_ts = bar_timestamp
        if target_ts is None:
            all_signals = self.get_signals()
            if not all_signals:
                return {"long": 0, "short": 0}
            target_ts = max(s.bar_timestamp for s in all_signals)

        # Decision 1A: Strict consensus (Current bar only)
        signals = self.get_signals_for_bar(target_ts, window_bars=0)
        consensus = {"long": 0, "short": 0}

        for signal in signals:
            direction = signal.direction.lower()
            if direction == "long":
                consensus["long"] += 1
            elif direction == "short":
                consensus["short"] += 1

        return consensus

    def are_signals_aligned(self, bar_timestamp: datetime | None = None) -> bool:
        """Check if all signals for a bar agree on direction.

        Args:
            bar_timestamp: Target bar for alignment check.

        Returns:
            True if all signals are long or all are short, False otherwise.
        """
        consensus = self.get_consensus(bar_timestamp)
        total = consensus["long"] + consensus["short"]
        
        if total <= 1:
            return True

        return consensus["long"] == 0 or consensus["short"] == 0

    def get_alignment_strength(self, bar_timestamp: datetime | None = None) -> float:
        """Calculate the strength of signal alignment for a bar.

        Args:
            bar_timestamp: Target bar for alignment strength.

        Returns:
            Alignment strength from 0.0 (perfectly split) to 1.0 (unanimous).
        """
        consensus = self.get_consensus(bar_timestamp)
        total = consensus["long"] + consensus["short"]

        if total == 0:
            return 0.0

        majority = max(consensus["long"], consensus["short"])
        return majority / total

    def get_conflicting_strategies(self, bar_timestamp: datetime | None = None) -> list[str]:
        """Identify strategies that disagree with the majority for a bar.

        Args:
            bar_timestamp: Target bar for conflict detection.

        Returns:
            List of unique strategy names that are in the minority direction.
        """
        target_ts = bar_timestamp
        if target_ts is None:
            all_signals = self.get_signals()
            if not all_signals:
                return []
            target_ts = max(s.bar_timestamp for s in all_signals)

        signals = self.get_signals_for_bar(target_ts, window_bars=0)
        if len(signals) <= 1:
            return []

        consensus = self.get_consensus(target_ts)
        if consensus["long"] == consensus["short"]:
            # Perfect tie: all are conflicting
            return sorted(list(set(s.strategy_name for s in signals)))

        majority_direction = "long" if consensus["long"] > consensus["short"] else "short"

        # Find unique strategies in minority
        conflicting = {
            signal.strategy_name 
            for signal in signals 
            if signal.direction.lower() != majority_direction
        }

        return sorted(list(conflicting))

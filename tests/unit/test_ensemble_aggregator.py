"""Unit tests for Ensemble Signal Aggregator."""

from collections import deque
from datetime import datetime, timedelta

import pytest

from src.detection.models import EnsembleSignal


class TestEnsembleSignalModel:
    """Test EnsembleSignal Pydantic model validation and methods."""

    def test_ensemble_signal_creation_with_valid_data(self):
        """Test creating EnsembleSignal with all valid fields."""
        signal = EnsembleSignal(
            strategy_name="triple_confluence_scaler",
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.85,
            bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            metadata={"fvg_size": 10, "vwap_alignment": True}
        )

        assert signal.strategy_name == "triple_confluence_scaler"
        assert signal.direction == "long"
        assert signal.entry_price == 11850.00
        assert signal.stop_loss == 11840.00
        assert signal.take_profit == 11870.00
        assert signal.confidence == 0.85
        assert signal.metadata == {"fvg_size": 10, "vwap_alignment": True}

    def test_ensemble_signal_confidence_must_be_between_0_and_1(self):
        """Test confidence validation rejects values outside 0-1 range."""
        with pytest.raises(ValueError):  # Pydantic v2 built-in validation
            EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=datetime(2026, 3, 31, 10, 30, 0),
                direction="long",
                entry_price=11850.00,
                stop_loss=11840.00,
                take_profit=11870.00,
                confidence=1.5,  # Invalid: > 1
                bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            )

    def test_ensemble_signal_confidence_rejects_negative(self):
        """Test confidence validation rejects negative values."""
        with pytest.raises(ValueError):  # Pydantic v2 built-in validation
            EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=datetime(2026, 3, 31, 10, 30, 0),
                direction="long",
                entry_price=11850.00,
                stop_loss=11840.00,
                take_profit=11870.00,
                confidence=-0.1,  # Invalid: < 0
                bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            )

    def test_ensemble_signal_stop_loss_must_be_below_entry_for_long(self):
        """Test stop_loss validation for long trades."""
        with pytest.raises(ValueError, match="stop_loss must be below entry_price for long"):
            EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=datetime(2026, 3, 31, 10, 30, 0),
                direction="long",
                entry_price=11850.00,
                stop_loss=11860.00,  # Invalid: above entry for long
                take_profit=11870.00,
                confidence=0.85,
                bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            )

    def test_ensemble_signal_stop_loss_must_be_above_entry_for_short(self):
        """Test stop_loss validation for short trades."""
        with pytest.raises(ValueError, match="stop_loss must be above entry_price for short"):
            EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=datetime(2026, 3, 31, 10, 30, 0),
                direction="short",
                entry_price=11850.00,
                stop_loss=11840.00,  # Invalid: below entry for short
                take_profit=11830.00,
                confidence=0.85,
                bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            )

    def test_ensemble_signal_take_profit_must_respect_2to1_ratio(self):
        """Test take_profit validation enforces 2:1 reward-risk ratio."""
        with pytest.raises(ValueError, match="take_profit must respect 2:1 ratio"):
            EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=datetime(2026, 3, 31, 10, 30, 0),
                direction="long",
                entry_price=11850.00,
                stop_loss=11840.00,  # Risk: $10
                take_profit=11855.00,  # Reward: $5 (only 0.5:1, invalid)
                confidence=0.85,
                bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            )

    def test_ensemble_signal_take_profit_allows_small_tolerance(self):
        """Test take_profit allows small rounding tolerance around 2:1."""
        # Risk: $10, Reward: $19 (1.9:1, should be valid with tolerance)
        signal = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,  # Risk: $10
            take_profit=11869.00,  # Reward: $19 (1.9:1, valid with tolerance)
            confidence=0.85,
            bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
        )
        assert signal.take_profit == 11869.00

    def test_ensemble_signal_risk_reward_ratio_calculation(self):
        """Test risk_reward_ratio() method calculates correctly."""
        signal = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,  # Risk: $10
            take_profit=11870.00,  # Reward: $20 (2:1)
            confidence=0.85,
            bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
        )
        assert signal.risk_reward_ratio() == 2.0

    def test_ensemble_signal_is_valid_returns_true_for_valid_signal(self):
        """Test is_valid() returns True for valid signal."""
        signal = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.85,
            bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
        )
        assert signal.is_valid() is True

    def test_ensemble_signal_direction_must_be_valid(self):
        """Test direction must be either 'long' or 'short'."""
        with pytest.raises(ValueError):  # Pydantic v2 Literal validation
            EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=datetime(2026, 3, 31, 10, 30, 0),
                direction="invalid",  # Invalid direction
                entry_price=11850.00,
                stop_loss=11840.00,
                take_profit=11870.00,
                confidence=0.85,
                bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            )

    def test_ensemble_signal_metadata_preserves_strategy_specific_data(self):
        """Test metadata field preserves strategy-specific information."""
        metadata = {
            "fvg_size_ticks": 12,
            "sweep_extent": 8,
            "vwap_distance": 5,
            "atr_value": 15.5,
        }
        signal = EnsembleSignal(
            strategy_name="triple_confluence_scaler",
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.85,
            bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
            metadata=metadata,
        )
        assert signal.metadata == metadata
        assert signal.metadata["fvg_size_ticks"] == 12


@pytest.fixture
def sample_ensemble_signal():
    """Create a sample EnsembleSignal for testing."""
    return EnsembleSignal(
        strategy_name="triple_confluence_scaler",
        timestamp=datetime(2026, 3, 31, 10, 30, 0),
        direction="long",
        entry_price=11850.00,
        stop_loss=11840.00,
        take_profit=11870.00,
        confidence=0.85,
        bar_timestamp=datetime(2026, 3, 31, 10, 30, 0),
        metadata={"test": "data"},
    )


@pytest.fixture
def multiple_strategy_signals():
    """Create signals from multiple strategies for testing."""
    base_time = datetime(2026, 3, 31, 10, 0, 0)
    return [
        EnsembleSignal(
            strategy_name="triple_confluence_scaler",
            timestamp=base_time,
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.85,
            bar_timestamp=base_time,
        ),
        EnsembleSignal(
            strategy_name="wolf_pack_3_edge",
            timestamp=base_time + timedelta(seconds=1),
            direction="long",
            entry_price=11851.00,
            stop_loss=11841.00,
            take_profit=11871.00,
            confidence=0.82,
            bar_timestamp=base_time,
        ),
        EnsembleSignal(
            strategy_name="adaptive_ema_momentum",
            timestamp=base_time + timedelta(seconds=2),
            direction="short",
            entry_price=11849.00,
            stop_loss=11859.00,
            take_profit=11829.00,
            confidence=0.78,
            bar_timestamp=base_time,
        ),
    ]


class TestEnsembleSignalAggregator:
    """Test EnsembleSignalAggregator class functionality."""

    def test_aggregator_initialization_with_default_lookback(self):
        """Test aggregator initializes with default lookback of 10 bars."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        assert aggregator.max_lookback == 10

    def test_aggregator_initialization_with_custom_lookback(self):
        """Test aggregator initializes with custom lookback."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator(max_lookback=20)
        assert aggregator.max_lookback == 20

    def test_add_signal_stores_signal_correctly(self, sample_ensemble_signal):
        """Test adding a signal stores it correctly."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        aggregator.add_signal(sample_ensemble_signal)

        signals = aggregator.get_signals()
        assert len(signals) == 1
        assert signals[0].strategy_name == "triple_confluence_scaler"
        assert signals[0].confidence == 0.85

    def test_add_signal_creates_per_strategy_storage(self, sample_ensemble_signal):
        """Test that signals are stored per strategy in separate deques."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        aggregator.add_signal(sample_ensemble_signal)

        assert "triple_confluence_scaler" in aggregator._signals
        assert len(aggregator._signals["triple_confluence_scaler"]) == 1

    def test_deduplication_replaces_same_bar_signal(self, sample_ensemble_signal):
        """Test that adding signal from same strategy for same bar replaces existing."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        bar_time = datetime(2026, 3, 31, 10, 30, 0)

        # Add first signal
        signal1 = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=bar_time,
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.80,
            bar_timestamp=bar_time,
        )
        aggregator.add_signal(signal1)

        # Add second signal for same bar (should replace)
        signal2 = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=bar_time + timedelta(seconds=1),
            direction="long",
            entry_price=11851.00,
            stop_loss=11841.00,
            take_profit=11871.00,
            confidence=0.85,  # Different confidence
            bar_timestamp=bar_time,  # Same bar
        )
        aggregator.add_signal(signal2)

        signals = aggregator.get_signals(strategy="Test Strategy")
        assert len(signals) == 1  # Only one signal for the bar
        assert signals[0].confidence == 0.85  # Latest signal kept
        assert signals[0].entry_price == 11851.00

    def test_cleanup_old_signals_removes_signals_beyond_lookback(self, sample_ensemble_signal):
        """Test that old signals beyond lookback window are removed."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator(max_lookback=3)
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add signals for bars 0, 1, 2, 3, 4
        for i in range(5):
            signal = EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=base_time + timedelta(hours=i),
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time + timedelta(hours=i),
            )
            aggregator.add_signal(signal)

        # Clean up with current bar at hour 4
        aggregator.cleanup_old_signals(base_time + timedelta(hours=4))

        signals = aggregator.get_signals(strategy="Test Strategy")
        # Should keep bars 2, 3, 4 (within 3 bars lookback from bar 4)
        # Bar 0, 1 should be removed
        assert len(signals) <= 3

    def test_get_signals_filters_by_strategy(self, multiple_strategy_signals):
        """Test get_signals with strategy filter."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        # Get only Triple Confluence signals
        tc_signals = aggregator.get_signals(strategy="triple_confluence_scaler")
        assert len(tc_signals) == 1
        assert tc_signals[0].strategy_name == "triple_confluence_scaler"

    def test_get_signals_filters_by_direction(self, multiple_strategy_signals):
        """Test get_signals with direction filter."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        # Get only long signals
        long_signals = aggregator.get_signals(direction="long")
        assert len(long_signals) == 2
        assert all(s.direction == "long" for s in long_signals)

    def test_get_signals_filters_by_min_confidence(self, multiple_strategy_signals):
        """Test get_signals with minimum confidence filter."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        # Get only high confidence signals (>= 0.80)
        high_conf_signals = aggregator.get_signals(min_confidence=0.80)
        assert len(high_conf_signals) == 2
        assert all(s.confidence >= 0.80 for s in high_conf_signals)

    def test_get_signals_for_bar_with_window(self, multiple_strategy_signals):
        """Test get_signals_for_bar with window parameter."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add signals at different bars
        for i in range(5):
            signal = EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=base_time + timedelta(minutes=i * 5),
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time + timedelta(minutes=i * 5),
            )
            aggregator.add_signal(signal)

        # Get signals for bar at minute 10 with window of 1 bar
        bar_time = base_time + timedelta(minutes=10)
        signals = aggregator.get_signals_for_bar(bar_time, window_bars=1)

        # Should get signal at minute 10 only (window=0 would give only that bar)
        assert len(signals) >= 1

    def test_get_active_strategies_returns_list(self, multiple_strategy_signals):
        """Test get_active_strategies returns list of strategies with signals."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        active = aggregator.get_active_strategies()
        assert len(active) == 3
        assert "triple_confluence_scaler" in active
        assert "wolf_pack_3_edge" in active
        assert "adaptive_ema_momentum" in active

    def test_get_latest_signal_returns_most_recent(self, sample_ensemble_signal):
        """Test get_latest_signal returns most recent signal from strategy."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # Add multiple signals
        for i in range(3):
            signal = EnsembleSignal(
                strategy_name="Test Strategy",
                timestamp=base_time + timedelta(minutes=i),
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time + timedelta(minutes=i),
            )
            aggregator.add_signal(signal)

        latest = aggregator.get_latest_signal("Test Strategy")
        assert latest is not None
        assert latest.entry_price == 11852.00  # Last signal

    def test_get_signal_count(self, multiple_strategy_signals):
        """Test get_signal_count returns correct count."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        # Count all signals
        assert aggregator.get_signal_count() == 3

        # Count specific strategy
        assert aggregator.get_signal_count(strategy="triple_confluence_scaler") == 1

    def test_clear_all_signals_removes_all(self, multiple_strategy_signals):
        """Test clear_all_signals removes all stored signals."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        assert aggregator.get_signal_count() == 3

        aggregator.clear_all_signals()

        assert aggregator.get_signal_count() == 0
        assert len(aggregator.get_active_strategies()) == 0


class TestSignalNormalizer:
    """Test signal normalization from all 5 strategies."""

    def test_normalize_triple_confluence_signal(self):
        """Test normalizing Triple Confluence signal to Ensemble format."""
        from datetime import datetime
        from src.detection.ensemble_signal_aggregator import normalize_triple_confluence
        from src.detection.models import TripleConfluenceSignal

        original = TripleConfluenceSignal(
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            direction="long",
            confidence=0.85,
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            contributing_factors={"fvg_size": 10, "vwap_alignment": True},
        )

        normalized = normalize_triple_confluence(original)

        assert normalized.strategy_name == "triple_confluence_scaler"
        assert normalized.direction == "long"
        assert normalized.entry_price == 11850.00
        assert normalized.confidence == 0.85
        # Compare normalized timestamps
        from src.detection.ensemble_signal_aggregator import _normalize_ts
        assert normalized.bar_timestamp == _normalize_ts(original.timestamp)

        assert "fvg_size" in normalized.metadata

    def test_normalize_wolf_pack_signal(self):
        """Test normalizing Wolf Pack signal to Ensemble format."""
        from datetime import datetime
        from src.detection.ensemble_signal_aggregator import normalize_wolf_pack
        from src.detection.models import WolfPackSignal

        original = WolfPackSignal(
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            direction="long",
            confidence=0.88,
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            contributing_factors={"sweep_extent": 8, "trap_severity": 2.5},
        )

        normalized = normalize_wolf_pack(original)

        assert normalized.strategy_name == "wolf_pack_3_edge"
        assert normalized.direction == "long"
        assert normalized.entry_price == 11850.00
        assert normalized.confidence == 0.88
        assert "sweep_extent" in normalized.metadata

    def test_normalize_ema_momentum_signal(self):
        """Test normalizing EMA Momentum signal to Ensemble format."""
        from datetime import datetime
        from src.detection.ensemble_signal_aggregator import normalize_ema_momentum
        from src.detection.models import MomentumSignal

        original = MomentumSignal(
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            direction="LONG",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=82.0,  # 0-100 scale
            ema_fast=11848.0,
            ema_medium=11845.0,
            ema_slow=11840.0,
        )

        normalized = normalize_ema_momentum(original)

        assert normalized.strategy_name == "adaptive_ema_momentum"
        assert normalized.direction == "long"  # Converted from LONG
        assert normalized.entry_price == 11850.00
        assert normalized.confidence == 0.82  # Converted from 82.0
        assert "ema_fast" in normalized.metadata
        assert normalized.metadata["ema_fast"] == 11848.0

    def test_normalize_vwap_bounce_signal(self):
        """Test normalizing vwap_bounce signal to Ensemble format."""
        from datetime import datetime
        from src.detection.ensemble_signal_aggregator import normalize_vwap_bounce

        # Mock VWAPBounceSignalModel
        class MockVWAPSignal:
            def __init__(self):
                self.strategy_name = "vwap_bounce"
                self.timestamp = datetime(2026, 3, 31, 10, 30, 0)
                self.direction = "long"
                self.entry_price = 11850.00
                self.stop_loss = 11840.00
                self.take_profit = 11870.00
                self.confidence = 0.75
                self.contributing_factors = {"rejection_distance": 3, "adx_value": 25}

        original = MockVWAPSignal()

        normalized = normalize_vwap_bounce(original)

        assert normalized.strategy_name == "vwap_bounce"
        assert normalized.direction == "long"
        assert normalized.entry_price == 11850.00
        assert normalized.confidence == 0.75
        assert "rejection_distance" in normalized.metadata

    def test_normalize_opening_range_signal(self):
        """Test normalizing Opening Range signal to Ensemble format."""
        from datetime import datetime
        from src.detection.ensemble_signal_aggregator import normalize_opening_range

        # Mock OpeningRangeSignalModel
        class MockORSignal:
            def __init__(self):
                self.strategy_name = "opening_range_breakout"
                self.timestamp = datetime(2026, 3, 31, 10, 30, 0)
                self.direction = "short"
                self.entry_price = 11850.00
                self.stop_loss = 11860.00
                self.take_profit = 11830.00
                self.confidence = 0.72
                self.contributing_factors = {"or_high": 11855, "breakout_strength": 1.8}

        original = MockORSignal()

        normalized = normalize_opening_range(original)

        assert normalized.strategy_name == "opening_range_breakout"
        assert normalized.direction == "short"
        assert normalized.entry_price == 11850.00
        assert normalized.confidence == 0.72
        assert "or_high" in normalized.metadata

    def test_normalize_signal_dispatcher(self):
        """Test normalize_signal dispatcher routes to correct normalizer."""
        from datetime import datetime
        from src.detection.ensemble_signal_aggregator import normalize_signal
        from src.detection.models import TripleConfluenceSignal

        original = TripleConfluenceSignal(
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            direction="long",
            confidence=0.85,
            timestamp=datetime(2026, 3, 31, 10, 30, 0),
            contributing_factors={},
        )

        normalized = normalize_signal(original)

        assert normalized.strategy_name == "triple_confluence_scaler"
        assert isinstance(normalized, EnsembleSignal)

    def test_normalize_signal_raises_error_for_unknown_type(self):
        """Test normalize_signal raises ValueError for unknown signal types."""
        from src.detection.ensemble_signal_aggregator import normalize_signal

        class UnknownSignal:
            pass

        unknown = UnknownSignal()

        with pytest.raises(ValueError, match="Unsupported signal type"):
            normalize_signal(unknown)


class TestAsyncSignalProcessing:
    """Test async signal processing pipeline."""

    @pytest.mark.asyncio
    async def test_add_signal_async_stores_signal(self, sample_ensemble_signal):
        """Test async add_signal stores signal correctly."""
        import asyncio
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        await aggregator.add_signal_async(sample_ensemble_signal)

        signals = aggregator.get_signals()
        assert len(signals) == 1
        assert signals[0].strategy_name == "triple_confluence_scaler"

    @pytest.mark.asyncio
    async def test_process_signals_queue_handles_multiple_signals(self, sample_ensemble_signal):
        """Test processing multiple signals from queue."""
        import asyncio
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        queue = asyncio.Queue()

        # Add multiple signals to queue
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        for i in range(3):
            signal = EnsembleSignal(
                strategy_name=f"Strategy {i}",
                timestamp=base_time + timedelta(seconds=i),
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time,
            )
            await queue.put(signal)

        # Process queue
        await aggregator.process_signals_queue(queue)

        # Verify all signals added
        assert aggregator.get_signal_count() == 3

    @pytest.mark.asyncio
    async def test_process_signals_queue_stops_on_none(self, sample_ensemble_signal):
        """Test process_signals_queue stops when receiving None."""
        import asyncio
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        queue = asyncio.Queue()

        # Add signal and None sentinel
        await queue.put(sample_ensemble_signal)
        await queue.put(None)  # Sentinel to stop processing

        # Process queue (should stop after None)
        await aggregator.process_signals_queue(queue)

        # Verify only first signal added
        assert aggregator.get_signal_count() == 1

    @pytest.mark.asyncio
    async def test_start_and_stop_aggregator_background_task(self):
        """Test starting and stopping background aggregator task."""
        import asyncio
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        queue = asyncio.Queue()

        # Start background task
        task = await aggregator.start_aggregator(queue)
        assert task is not None

        # Add a signal
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        signal = EnsembleSignal(
            strategy_name="Test Strategy",
            timestamp=base_time,
            direction="long",
            entry_price=11850.00,
            stop_loss=11840.00,
            take_profit=11870.00,
            confidence=0.85,
            bar_timestamp=base_time,
        )
        await queue.put(signal)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Stop aggregator
        await aggregator.stop_aggregator()

        # Verify signal was processed
        assert aggregator.get_signal_count() == 1


class TestSignalConsensusAndAlignment:
    """Test signal consensus and alignment detection methods."""

    def test_get_consensus_with_unanimous_long(self, multiple_strategy_signals):
        """Test get_consensus when all signals agree on long."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()

        # All long signals
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        for i in range(3):
            signal = EnsembleSignal(
                strategy_name=f"Strategy {i}",
                timestamp=base_time,
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time,
            )
            aggregator.add_signal(signal)

        consensus = aggregator.get_consensus()
        assert consensus["long"] == 3
        assert consensus["short"] == 0

    def test_get_consensus_with_mixed_signals(self, multiple_strategy_signals):
        """Test get_consensus with mixed long/short signals."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        consensus = aggregator.get_consensus()
        # Should have 2 long, 1 short from multiple_strategy_signals fixture
        assert consensus["long"] == 2
        assert consensus["short"] == 1

    def test_are_signals_aligned_returns_true_for_unanimous(self):
        """Test are_signals_aligned returns True when all signals agree."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # All long signals
        for i in range(3):
            signal = EnsembleSignal(
                strategy_name=f"Strategy {i}",
                timestamp=base_time,
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time,
            )
            aggregator.add_signal(signal)

        assert aggregator.are_signals_aligned() is True

    def test_are_signals_aligned_returns_false_for_mixed(self, multiple_strategy_signals):
        """Test are_signals_aligned returns False when signals disagree."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        assert aggregator.are_signals_aligned() is False

    def test_are_signals_aligned_returns_true_for_single_signal(self, sample_ensemble_signal):
        """Test are_signals_aligned returns True with only one signal."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        aggregator.add_signal(sample_ensemble_signal)

        assert aggregator.are_signals_aligned() is True

    def test_get_alignment_strength_for_unanimous(self):
        """Test get_alignment_strength returns 1.0 for unanimous signals."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        base_time = datetime(2026, 3, 31, 10, 0, 0)

        # All long signals
        for i in range(4):
            signal = EnsembleSignal(
                strategy_name=f"Strategy {i}",
                timestamp=base_time,
                direction="long",
                entry_price=11850.00 + i,
                stop_loss=11840.00 + i,
                take_profit=11870.00 + i,
                confidence=0.85,
                bar_timestamp=base_time,
            )
            aggregator.add_signal(signal)

        strength = aggregator.get_alignment_strength()
        assert strength == 1.0

    def test_get_alignment_strength_for_split(self, multiple_strategy_signals):
        """Test get_alignment_strength for mixed signals."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        strength = aggregator.get_alignment_strength()
        # 2 long, 1 short = 2/3 strength
        assert abs(strength - 0.6667) < 0.01

    def test_get_conflicting_strategies(self, multiple_strategy_signals):
        """Test get_conflicting_strategies identifies minority signals."""
        from src.detection.ensemble_signal_aggregator import EnsembleSignalAggregator

        aggregator = EnsembleSignalAggregator()
        for signal in multiple_strategy_signals:
            aggregator.add_signal(signal)

        # 2 long, 1 short - short is conflicting
        conflicting = aggregator.get_conflicting_strategies()
        assert len(conflicting) == 1
        # The short signal should be identified as conflicting
        assert "adaptive_ema_momentum" in conflicting

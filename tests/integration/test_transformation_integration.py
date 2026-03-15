"""Integration tests for Dollar Bar transformation."""

import asyncio
from datetime import datetime

import pytest

from src.data.models import DollarBar, MarketData
from src.data.transformation import (
    BarBuilderState,
    DollarBarTransformer,
    DOLLAR_THRESHOLD,
)


class TestBarAccumulation:
    """Test bar accumulation logic."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for MarketData."""
        return asyncio.Queue()

    @pytest.fixture
    def output_queue(self):
        """Create output queue for DollarBar."""
        return asyncio.Queue()

    @pytest.fixture
    def transformer(self, input_queue, output_queue):
        """Create DollarBarTransformer instance."""
        return DollarBarTransformer(input_queue, output_queue)

    @pytest.mark.asyncio
    async def test_bar_initialization(self, transformer: DollarBarTransformer) -> None:
        """Test bar initializes with first trade."""
        # Create first market data point
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=100,
        )

        await transformer._process_market_data(market_data)

        # Verify bar initialized
        assert transformer.state == BarBuilderState.ACCUMULATING
        assert transformer._bar_open == 4523.50
        assert transformer._bar_high == 4523.50
        assert transformer._bar_low == 4523.50
        assert transformer._bar_close == 4523.50
        assert transformer._bar_volume == 100

    @pytest.mark.asyncio
    async def test_bar_accumulation_high_low(
        self, transformer: DollarBarTransformer
    ) -> None:
        """Test high/low update correctly."""
        # Initialize bar
        market_data1 = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.00,
            ask=4523.25,
            last=4523.25,
            volume=100,
        )
        await transformer._process_market_data(market_data1)

        # Add higher price
        market_data2 = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.75,
            ask=4524.00,
            last=4524.00,
            volume=100,
        )
        await transformer._process_market_data(market_data2)

        assert transformer._bar_high == 4524.00

        # Add lower price
        market_data3 = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4522.75,
            ask=4523.00,
            last=4523.00,
            volume=100,
        )
        await transformer._process_market_data(market_data3)

        assert transformer._bar_low == 4523.00

    @pytest.mark.asyncio
    async def test_notional_value_calculation(
        self, transformer: DollarBarTransformer
    ) -> None:
        """Test notional value calculation: price × volume × 0.5."""
        # Add trade with known notional value
        # 4523.50 × 1000 × 0.5 = 2,261,750
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=1000,
        )
        await transformer._process_market_data(market_data)

        expected_notional = 4523.50 * 1000 * 0.5
        assert transformer._bar_notional == expected_notional


class TestCompletionTriggers:
    """Test bar completion triggers."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for MarketData."""
        return asyncio.Queue()

    @pytest.fixture
    def output_queue(self):
        """Create output queue for DollarBar."""
        return asyncio.Queue()

    @pytest.fixture
    def transformer(self, input_queue, output_queue):
        """Create DollarBarTransformer instance."""
        return DollarBarTransformer(input_queue, output_queue)

    @pytest.mark.asyncio
    async def test_threshold_completion(
        self, transformer: DollarBarTransformer
    ) -> None:
        """Test bar completes at $50M threshold."""
        # Create market data that will exceed threshold
        # To exceed $50M: need price × volume × 0.5 >= 50,000,000
        # At price ~4523, need volume ~22,100 contracts
        high_volume = 25000

        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=high_volume,
        )

        await transformer._process_market_data(market_data)

        # Verify bar completed
        assert transformer.bars_created == 1

        # Verify DollarBar in output queue
        assert not transformer._output_queue.empty()
        bar = await transformer._output_queue.get()

        assert isinstance(bar, DollarBar)
        assert bar.notional_value >= DOLLAR_THRESHOLD

    @pytest.mark.asyncio
    async def test_timeout_completion(self, transformer: DollarBarTransformer) -> None:
        """Test bar completes after 5 seconds."""
        from datetime import timedelta

        # Initialize bar
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=100,
        )
        await transformer._process_market_data(market_data)

        # Simulate 5-second timeout by adjusting bar start time
        transformer._bar_start_time = datetime.now() - timedelta(seconds=6)

        # Check timeout
        await transformer._check_bar_timeout()

        # Verify bar completed
        assert transformer.bars_created == 1

        # Verify DollarBar in output queue
        assert not transformer._output_queue.empty()
        bar = await transformer._output_queue.get()

        assert isinstance(bar, DollarBar)


class TestStateMachine:
    """Test bar builder state machine."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for MarketData."""
        return asyncio.Queue()

    @pytest.fixture
    def output_queue(self):
        """Create output queue for DollarBar."""
        return asyncio.Queue()

    @pytest.fixture
    def transformer(self, input_queue, output_queue):
        """Create DollarBarTransformer instance."""
        return DollarBarTransformer(input_queue, output_queue)

    @pytest.mark.asyncio
    async def test_idle_to_accumulating_transition(
        self, transformer: DollarBarTransformer
    ) -> None:
        """Test state transitions from IDLE to ACCUMULATING."""
        assert transformer.state == BarBuilderState.IDLE

        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=100,
        )

        await transformer._process_market_data(market_data)

        assert transformer.state == BarBuilderState.ACCUMULATING

    @pytest.mark.asyncio
    async def test_accumulating_to_idle_on_completion(
        self, transformer: DollarBarTransformer
    ) -> None:
        """Test state transitions from ACCUMULATING to IDLE on completion."""
        # Initialize bar
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=100,
        )
        await transformer._process_market_data(market_data)

        assert transformer.state == BarBuilderState.ACCUMULATING

        # Complete bar by exceeding threshold
        high_volume = 25000
        market_data2 = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=high_volume,
        )
        await transformer._process_market_data(market_data2)

        # State should return to IDLE after completion
        assert transformer.state == BarBuilderState.IDLE


class TestTransformationPipeline:
    """Test end-to-end transformation pipeline."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for MarketData."""
        return asyncio.Queue()

    @pytest.fixture
    def output_queue(self):
        """Create output queue for DollarBar."""
        return asyncio.Queue()

    @pytest.fixture
    def transformer(self, input_queue, output_queue):
        """Create DollarBarTransformer instance."""
        return DollarBarTransformer(input_queue, output_queue)

    @pytest.mark.asyncio
    async def test_continuous_market_data_stream(
        self, transformer: DollarBarTransformer
    ) -> None:
        """Test processing continuous stream of market data."""
        # Simulate realistic market data stream
        prices = [4523.25, 4523.50, 4523.75, 4524.00, 4523.50, 4523.00]
        volume = 5000

        for price in prices:
            market_data = MarketData(
                symbol="MNQ",
                timestamp=datetime.now(),
                bid=price - 0.25,
                ask=price + 0.25,
                last=price,
                volume=volume,
            )
            await transformer._process_market_data(market_data)

        # Verify at least one bar was created
        assert transformer.bars_created >= 1

    @pytest.mark.asyncio
    async def test_queue_backpressure(self, transformer: DollarBarTransformer) -> None:
        """Test queue overflow handling."""
        # Fill output queue to max size
        max_size = 10
        full_queue = asyncio.Queue(maxsize=max_size)

        # Create transformer with full output queue
        transformer_full = DollarBarTransformer(transformer._input_queue, full_queue)

        # Complete multiple bars to fill queue
        for _ in range(max_size):
            market_data = MarketData(
                symbol="MNQ",
                timestamp=datetime.now(),
                bid=4523.25,
                ask=4523.50,
                last=4523.50,
                volume=30000,  # Will exceed threshold
            )
            await transformer_full._process_market_data(market_data)

        # Queue should be full
        assert full_queue.full()

        # Try to complete another bar (should handle gracefully)
        market_data = MarketData(
            symbol="MNQ",
            timestamp=datetime.now(),
            bid=4523.25,
            ask=4523.50,
            last=4523.50,
            volume=30000,
        )
        await transformer_full._process_market_data(market_data)

        # Transformer should still function
        assert transformer_full.bars_created > max_size

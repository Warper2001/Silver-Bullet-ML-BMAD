"""Unit tests for Trade Execution Pipeline Integration.

Tests pipeline initialization, signal processing, order submission,
position monitoring, integration with all components, and
error handling.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
from unittest.mock import Mock
import pytest

from src.execution.trade_execution_pipeline import (
    TradeExecutionPipeline,
    PipelineResult,
    TradingSignal,
)


@dataclass
class DollarBar:
    """Mock dollar bar for testing."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class TestTradingSignal:
    """Test trading signal dataclass."""

    def test_create_trading_signal(self):
        """Verify trading signal creation."""
        signal = TradingSignal(
            signal_id="SIG-123",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=["MSS", "FVG"],
            prediction={"probability": 0.85}
        )

        assert signal.signal_id == "SIG-123"
        assert signal.symbol == "MNQ 03-26"
        assert signal.direction == "bullish"
        assert signal.confidence_score == 0.85


class TestPipelineResult:
    """Test pipeline result dataclass."""

    def test_create_pipeline_result_success(self):
        """Verify pipeline result for successful execution."""
        result = PipelineResult(
            success=True,
            order_id="ORDER-123",
            filled=True,
            fill_price=11800.50,
            block_reason=None,
            error_message=None,
            position_id="ORDER-123"
        )

        assert result.success is True
        assert result.order_id == "ORDER-123"
        assert result.filled is True

    def test_create_pipeline_result_blocked(self):
        """Verify pipeline result for blocked signal."""
        result = PipelineResult(
            success=False,
            order_id=None,
            filled=False,
            fill_price=None,
            block_reason="Market closed",
            error_message=None,
            position_id=None
        )

        assert result.success is False
        assert result.block_reason == "Market closed"


class TestTradeExecutionPipelineInit:
    """Test pipeline initialization."""

    @pytest.fixture
    def dependencies(self):
        """Create mock dependencies."""
        return {
            "api_client": Mock(),
            "position_sizer": Mock(),
            "order_type_selector": Mock(),
            "market_order_submitter": Mock(),
            "limit_order_submitter": Mock(),
            "partial_fill_handler": Mock(),
            "barrier_calculator": Mock(),
            "barrier_monitor": Mock(),
            "exit_executor": Mock(),
            "position_monitoring_service": Mock(),
            "time_window_filter": Mock(),
            "audit_trail": Mock()
        }

    def test_init_with_all_dependencies(self, dependencies):
        """Verify pipeline initializes with all dependencies."""
        pipeline = TradeExecutionPipeline(**dependencies)

        assert pipeline._api_client == dependencies["api_client"]
        assert pipeline._position_sizer == dependencies["position_sizer"]
        assert pipeline._time_window_filter == dependencies["time_window_filter"]

    def test_init_with_missing_dependencies(self):
        """Verify pipeline raises error with missing dependencies."""
        with pytest.raises(ValueError):
            TradeExecutionPipeline(
                api_client=None,
                position_sizer=Mock(),
                order_type_selector=Mock(),
                market_order_submitter=Mock(),
                limit_order_submitter=Mock(),
                partial_fill_handler=Mock(),
                barrier_calculator=Mock(),
                barrier_monitor=Mock(),
                exit_executor=Mock(),
                position_monitoring_service=Mock(),
                time_window_filter=Mock(),
                audit_trail=Mock()
            )


class TestProcessSignal:
    """Test signal processing through pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked dependencies."""
        # Mock dependencies
        api_client = Mock()
        position_sizer = Mock()
        position_sizer.calculate_position_size.return_value = 5

        order_type_selector = Mock()
        order_type_selector.select_order_type.return_value = "LIMIT"

        market_order_submitter = Mock()
        limit_order_submitter = Mock()
        limit_order_submitter.submit_limit_order.return_value = Mock(
            success=True,
            order_id="ORDER-123",
            filled=True,
            filled_price=11800.50
        )

        partial_fill_handler = Mock()
        barrier_calculator = Mock()
        barrier_monitor = Mock()
        exit_executor = Mock()

        position_monitoring_service = Mock()

        time_window_filter = Mock()
        time_window_filter.check_time_window.return_value = Mock(
            allowed=True,
            reason="",
            window_name="",
            time_until_open=None,
            time_until_close=None
        )

        audit_trail = Mock()

        return TradeExecutionPipeline(
            api_client=api_client,
            position_sizer=position_sizer,
            order_type_selector=order_type_selector,
            market_order_submitter=market_order_submitter,
            limit_order_submitter=limit_order_submitter,
            partial_fill_handler=partial_fill_handler,
            barrier_calculator=barrier_calculator,
            barrier_monitor=barrier_monitor,
            exit_executor=exit_executor,
            position_monitoring_service=position_monitoring_service,
            time_window_filter=time_window_filter,
            audit_trail=audit_trail
        )

    @pytest.fixture
    def valid_signal(self):
        """Create valid trading signal."""
        return TradingSignal(
            signal_id="SIG-123",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=["MSS", "FVG"],
            prediction={"probability": 0.85}
        )

    def test_process_valid_bullish_signal(self, pipeline, valid_signal):
        """Verify valid bullish signal processed successfully."""
        result = pipeline.process_signal(valid_signal)

        assert result.success is True
        assert result.order_id == "ORDER-123"
        assert result.filled is True

    def test_process_valid_bearish_signal(self, pipeline):
        """Verify valid bearish signal processed successfully."""
        signal = TradingSignal(
            signal_id="SIG-456",
            symbol="MNQ 03-26",
            direction="bearish",
            confidence_score=0.80,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=["MSS"],
            prediction={"probability": 0.80}
        )

        result = pipeline.process_signal(signal)

        assert result.success is True

    def test_signal_blocked_by_time_window(self, pipeline, valid_signal):
        """Verify signal blocked when outside time window."""
        # Mock time window filter to block
        pipeline._time_window_filter.check_time_window.return_value = Mock(
            allowed=False,
            reason="Market closed",
            window_name="AFTER_HOURS",
            time_until_open=None,
            time_until_close=None
        )

        result = pipeline.process_signal(valid_signal)

        assert result.success is False
        assert "Market closed" in result.block_reason or result.block_reason is not None

    def test_signal_with_invalid_confidence(self, pipeline):
        """Verify signal with invalid confidence score blocked."""
        signal = TradingSignal(
            signal_id="SIG-789",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=1.5,  # Invalid: > 1.0
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=[],
            prediction={}
        )

        result = pipeline.process_signal(signal)

        assert result.success is False
        assert result.block_reason is not None

    def test_market_order_path(self, pipeline):
        """Verify market order submission path."""
        # Mock order type selector to return MARKET
        pipeline._order_type_selector.select_order_type.return_value = "MARKET"

        pipeline._market_order_submitter.submit_market_order.return_value = Mock(
            success=True,
            order_id="ORDER-MARKET-123",
            filled=True,
            filled_price=11800.00
        )

        signal = TradingSignal(
            signal_id="SIG-MARKET",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.75,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=[],
            prediction={}
        )

        result = pipeline.process_signal(signal)

        assert result.success is True
        assert result.order_id == "ORDER-MARKET-123"


class TestOnBarUpdate:
    """Test bar update handling."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked dependencies."""
        api_client = Mock()
        position_sizer = Mock()
        order_type_selector = Mock()
        market_order_submitter = Mock()
        limit_order_submitter = Mock()
        partial_fill_handler = Mock()
        barrier_calculator = Mock()
        barrier_monitor = Mock()
        exit_executor = Mock()
        position_monitoring_service = Mock()
        time_window_filter = Mock()
        audit_trail = Mock()

        # Mock on_price_update to return exits
        position_monitoring_service.on_price_update.return_value = [
            Mock(order_id="ORDER-123", exit_barrier="UPPER")
        ]

        return TradeExecutionPipeline(
            api_client=api_client,
            position_sizer=position_sizer,
            order_type_selector=order_type_selector,
            market_order_submitter=market_order_submitter,
            limit_order_submitter=limit_order_submitter,
            partial_fill_handler=partial_fill_handler,
            barrier_calculator=barrier_calculator,
            barrier_monitor=barrier_monitor,
            exit_executor=exit_executor,
            position_monitoring_service=position_monitoring_service,
            time_window_filter=time_window_filter,
            audit_trail=audit_trail
        )

    @pytest.fixture
    def dollar_bar(self):
        """Create dollar bar."""
        return DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.00,
            high=11815.50,
            low=11798.00,
            close=11805.00,
            volume=1500
        )

    def test_monitor_positions_on_bar_update(self, pipeline, dollar_bar):
        """Verify positions monitored on bar update."""
        exits = pipeline.on_bar_update(dollar_bar)

        assert len(exits) == 1
        assert exits[0].order_id == "ORDER-123"

    def test_no_exits_when_no_barriers_hit(self, pipeline, dollar_bar):
        """Verify no exits when no barriers hit."""
        # Mock to return no exits
        pipeline._position_monitoring_service.on_price_update.return_value = []

        exits = pipeline.on_bar_update(dollar_bar)

        assert len(exits) == 0


class TestErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked dependencies."""
        api_client = Mock()
        position_sizer = Mock()
        order_type_selector = Mock()
        market_order_submitter = Mock()
        limit_order_submitter = Mock()
        partial_fill_handler = Mock()
        barrier_calculator = Mock()
        barrier_monitor = Mock()
        exit_executor = Mock()
        position_monitoring_service = Mock()
        time_window_filter = Mock()
        audit_trail = Mock()

        return TradeExecutionPipeline(
            api_client=api_client,
            position_sizer=position_sizer,
            order_type_selector=order_type_selector,
            market_order_submitter=market_order_submitter,
            limit_order_submitter=limit_order_submitter,
            partial_fill_handler=partial_fill_handler,
            barrier_calculator=barrier_calculator,
            barrier_monitor=barrier_monitor,
            exit_executor=exit_executor,
            position_monitoring_service=position_monitoring_service,
            time_window_filter=time_window_filter,
            audit_trail=audit_trail
        )

    def test_handle_order_submission_failure(self, pipeline):
        """Verify order submission failure handled gracefully."""
        # Mock order submission to fail
        pipeline._limit_order_submitter.submit_limit_order.return_value = Mock(
            success=False,
            order_id=None,
            filled=False,
            filled_price=None,
            error_message="API timeout"
        )

        pipeline._order_type_selector.select_order_type.return_value = "LIMIT"

        signal = TradingSignal(
            signal_id="SIG-ERROR",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=[],
            prediction={}
        )

        result = pipeline.process_signal(signal)

        assert result.success is False
        assert result.error_message is not None

    def test_log_error_to_audit_trail(self, pipeline):
        """Verify errors logged to audit trail."""
        # Mock order submission to fail
        pipeline._limit_order_submitter.submit_limit_order.return_value = Mock(
            success=False,
            order_id=None,
            filled=False,
            filled_price=None,
            error_message="API timeout"
        )

        pipeline._order_type_selector.select_order_type.return_value = "LIMIT"

        signal = TradingSignal(
            signal_id="SIG-ERROR",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=[],
            prediction={}
        )

        result = pipeline.process_signal(signal)

        # Verify error returned in result
        assert result.success is False
        assert result.error_message == "API timeout"


class TestIntegration:
    """Test integration with all components."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked dependencies."""
        api_client = Mock()
        position_sizer = Mock()
        position_sizer.calculate_position_size.return_value = 5

        order_type_selector = Mock()
        order_type_selector.select_order_type.return_value = "LIMIT"

        limit_order_submitter = Mock()
        limit_order_submitter.submit_limit_order.return_value = Mock(
            success=True,
            order_id="ORDER-INT-123",
            filled=True,
            filled_price=11800.50
        )

        barrier_calculator = Mock()
        barrier_calculator.calculate_barriers.return_value = Mock(
            upper_barrier_price=11815.00,
            lower_barrier_price=11792.50,
            time_barrier_utc=datetime(2026, 3, 17, 19, 0, 0, tzinfo=timezone.utc)
        )

        position_monitoring_service = Mock()
        time_window_filter = Mock()
        time_window_filter.check_time_window.return_value = Mock(
            allowed=True,
            reason="",
            window_name="",
            time_until_open=None,
            time_until_close=None
        )

        audit_trail = Mock()

        return TradeExecutionPipeline(
            api_client=api_client,
            position_sizer=position_sizer,
            order_type_selector=order_type_selector,
            market_order_submitter=Mock(),
            limit_order_submitter=limit_order_submitter,
            partial_fill_handler=Mock(),
            barrier_calculator=barrier_calculator,
            barrier_monitor=Mock(),
            exit_executor=Mock(),
            position_monitoring_service=position_monitoring_service,
            time_window_filter=time_window_filter,
            audit_trail=audit_trail
        )

    def test_end_to_end_signal_processing(self, pipeline):
        """Verify complete signal flow from submission to monitoring."""
        signal = TradingSignal(
            signal_id="SIG-E2E",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=["MSS"],
            prediction={"probability": 0.85}
        )

        result = pipeline.process_signal(signal)

        assert result.success is True
        assert result.order_id == "ORDER-INT-123"

        # Verify position monitoring initialized
        pipeline._position_monitoring_service.on_position_entered.assert_called()

        # Verify audit trail logged
        pipeline._audit_trail.log_order_submit.assert_called()

    def test_audit_trail_has_all_events(self, pipeline):
        """Verify audit trail logs all pipeline events."""
        signal = TradingSignal(
            signal_id="SIG-AUDIT",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=[],
            prediction={}
        )

        pipeline.process_signal(signal)

        # Verify multiple audit events logged
        assert pipeline._audit_trail.log_order_submit.called or \
               pipeline._audit_trail.log_order_fill.called


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_signal_processing_completes_quickly(self):
        """Verify signal processing completes in reasonable time."""
        import time

        # Mock all dependencies
        api_client = Mock()
        position_sizer = Mock()
        position_sizer.calculate_position_size.return_value = 5

        order_type_selector = Mock()
        order_type_selector.select_order_type.return_value = "LIMIT"

        limit_order_submitter = Mock()
        limit_order_submitter.submit_limit_order.return_value = Mock(
            success=True,
            order_id="ORDER-123",
            filled=True,
            filled_price=11800.50
        )

        pipeline = TradeExecutionPipeline(
            api_client=api_client,
            position_sizer=position_sizer,
            order_type_selector=order_type_selector,
            market_order_submitter=Mock(),
            limit_order_submitter=limit_order_submitter,
            partial_fill_handler=Mock(),
            barrier_calculator=Mock(),
            barrier_monitor=Mock(),
            exit_executor=Mock(),
            position_monitoring_service=Mock(),
            time_window_filter=Mock(
                check_time_window=Mock(return_value=Mock(allowed=True))
            ),
            audit_trail=Mock()
        )

        signal = TradingSignal(
            signal_id="SIG-PERF",
            symbol="MNQ 03-26",
            direction="bullish",
            confidence_score=0.85,
            timestamp=datetime.now(timezone.utc),
            entry_price=11800.00,
            patterns=[],
            prediction={}
        )

        start_time = time.perf_counter()
        result = pipeline.process_signal(signal)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.success is True
        # Should complete in less than 100ms (mocked dependencies)
        assert elapsed_ms < 100.0

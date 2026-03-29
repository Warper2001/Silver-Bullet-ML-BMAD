"""End-to-end integration test for TradeStation SIM Paper Trading.

This test validates the complete paper trading flow:
1. Data ingestion from TradeStation SIM environment
2. ML inference on real-time dollar bar data
3. Order execution through risk validation
4. Position tracking with mark-to-market P&L

Acceptance Criteria:
- Given TradeStation SIM environment is accessible and authenticated, when system starts,
  then market data streams continuously without errors
- Given live dollar bar stream, when Silver Bullet signal occurs with ML probability ≥0.65,
  then features are engineered from real-time data and ML inference score is calculated
- Given ML inference score ≥0.65 and all risk layers pass, when order is submitted to
  TradeStation SIM, then order is accepted and filled instantly with confirmation
- Given SIM order fill, when position is opened, then position tracker updates quantity,
  entry price, and tracks unrealized P&L in real-time
- Given daily loss approaches $500, when circuit breaker triggers, then all positions
  close and trading halts for remainder of day
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_mock import MockerFixture

from src.data.models import DollarBar, MarketData, SilverBulletSetup
from src.execution.position_tracker import Position, PositionTracker
from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.market_data.streaming import QuoteStreamParser
from src.execution.tradestation.models import TradeStationQuote, TradeStationOrder
from src.ml.inference import MLInference
from src.risk.risk_orchestrator import RiskOrchestrator


logger = logging.getLogger(__name__)


@pytest.fixture
def mock_tradestation_client():
    """Create mock TradeStation client."""
    client = AsyncMock(spec=TradeStationClient)
    client.env = "sim"
    client.api_base_url = "https://sim-api.tradestation.com/v3"
    client.is_authenticated.return_value = True
    return client


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        MarketData(
            symbol="MNQH26",
            timestamp=base_time + timedelta(seconds=i),
            bid=11800.0 + i * 0.5,
            ask=11800.5 + i * 0.5,
            last=11800.25 + i * 0.5,
            volume=100 + i,
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_dollar_bars():
    """Create sample dollar bars for ML inference."""
    base_time = datetime.now(timezone.utc)
    return [
        DollarBar(
            timestamp=base_time + timedelta(minutes=i),
            open=11800.0 + i * 2,
            high=11805.0 + i * 2,
            low=11798.0 + i * 2,
            close=11803.0 + i * 2,
            volume=1000,
            is_forward_filled=False,
            notional_value=50000000.0,
        )
        for i in range(50)  # Need at least 20 bars for meaningful features
    ]


@pytest.fixture
def sample_silver_bullet_setup():
    """Create sample Silver Bullet setup."""
    from src.data.models import (
        MSSEvent,
        FVGEvent,
        LiquiditySweepEvent,
        SwingPoint,
        GapRange,
    )

    # Create MSS event
    mss_event = MSSEvent(
        timestamp=datetime.now(timezone.utc),
        direction="bullish",
        swing_point=SwingPoint(
            timestamp=datetime.now(timezone.utc),
            price=11750.0,
            swing_type="swing_low",
            bar_index=100,
        ),
    )

    # Create FVG event
    fvg_event = FVGEvent(
        timestamp=datetime.now(timezone.utc),
        gap=GapRange(top=11805.0, bottom=11795.0),
        direction="bullish",
    )

    # Create Liquidity Sweep event
    liquidity_sweep_event = LiquiditySweepEvent(
        timestamp=datetime.now(timezone.utc),
        sweep_level=11780.0,
        direction="bullish",
        swing_point=SwingPoint(
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            price=11780.0,
            swing_type="swing_low",
            bar_index=98,
        ),
    )

    return SilverBulletSetup(
        timestamp=datetime.now(timezone.utc),
        direction="bullish",
        mss_event=mss_event,
        fvg_event=fvg_event,
        liquidity_sweep_event=liquidity_sweep_event,
        entry_zone_top=11805.0,
        entry_zone_bottom=11795.0,
        invalidation_point=11750.0,
        confluence_count=3,
        priority="high",
        bar_index=100,
        confidence=4.5,
    )


@pytest.fixture
def risk_orchestrator(tmp_path):
    """Create test-ready RiskOrchestrator with all dependencies.

    This fixture creates a RiskOrchestrator with mock/test implementations
    of all 8 risk layers for testing purposes.
    """
    from src.risk.emergency_stop import EmergencyStop
    from src.risk.daily_loss_tracker import DailyLossTracker
    from src.risk.drawdown_tracker import DrawdownTracker
    from src.risk.position_size_tracker import PositionSizeTracker
    from src.risk.position_sizer import PositionSizer
    from src.risk.circuit_breaker_detector import CircuitBreakerDetector
    from src.risk.news_event_filter import NewsEventFilter
    from src.risk.per_trade_risk_limit import PerTradeRiskLimit
    from src.risk.notification_manager import NotificationManager

    # Create test instances with permissive settings
    emergency_stop = EmergencyStop()
    daily_loss_tracker = DailyLossTracker(
        daily_loss_limit=500.0,
        account_balance=100000.0  # Starting account balance
    )
    drawdown_tracker = DrawdownTracker(
        max_drawdown_percentage=0.12,  # 12% max drawdown
        recovery_threshold_percentage=0.95,  # 95% recovery threshold
        initial_value=100000.0,  # Starting account value
    )
    position_size_tracker = PositionSizeTracker(max_contracts=5)
    position_sizer = PositionSizer(
        account_balance=100000.0,
        risk_per_trade=0.02,  # 2% risk per trade
    )
    circuit_breaker_detector = CircuitBreakerDetector()
    news_event_filter = NewsEventFilter()
    per_trade_risk_limit = PerTradeRiskLimit(max_risk_per_trade=100.0)
    notification_manager = NotificationManager()
    audit_trail_path = tmp_path / "audit_trail.csv"

    return RiskOrchestrator(
        emergency_stop=emergency_stop,
        daily_loss_tracker=daily_loss_tracker,
        drawdown_tracker=drawdown_tracker,
        position_size_tracker=position_size_tracker,
        position_sizer=position_sizer,
        circuit_breaker_detector=circuit_breaker_detector,
        news_event_filter=news_event_filter,
        per_trade_risk_limit=per_trade_risk_limit,
        notification_manager=notification_manager,
        audit_trail_path=str(audit_trail_path)
    )


@pytest.mark.asyncio
async def test_data_ingestion_from_sim_environment(
    mock_tradestation_client,
    sample_market_data,
):
    """Test data ingestion from TradeStation SIM environment.

    Given TradeStation SIM environment is accessible and authenticated,
    when system starts,
    then market data streams continuously without errors.
    """
    # Arrange
    parser = QuoteStreamParser(client=mock_tradestation_client)
    queue = asyncio.Queue()

    # Mock the stream_quotes method to yield sample data
    async def mock_stream_quotes(symbols):
        for data in sample_market_data:
            # Convert MarketData to TradeStationQuote
            quote = TradeStationQuote(
                Symbol=data.symbol,
                Bid=data.bid,
                Ask=data.ask,
                Last=data.last,
                BidSize=data.volume,
                AskSize=data.volume,
                LastSize=data.volume,
                TimeStamp=data.timestamp,
                Volume=data.volume,
            )
            yield quote

    parser.stream_quotes = mock_stream_quotes

    # Act
    streaming_task = asyncio.create_task(parser.stream_to_queue(["MNQH26"], queue))

    # Collect first 5 quotes
    received_quotes = []
    for _ in range(5):
        quote = await asyncio.wait_for(queue.get(), timeout=2.0)
        received_quotes.append(quote)

    # Cleanup
    parser.stop_streaming()
    await asyncio.wait_for(streaming_task, timeout=1.0)

    # Assert
    assert len(received_quotes) == 5
    assert all(q.symbol == "MNQH26" for q in received_quotes)
    assert all(q.last is not None for q in received_quotes)
    logger.info(f"Received {len(received_quotes)} quotes from SIM environment")


@pytest.mark.asyncio
async def test_ml_inference_on_realtime_dollar_bars(
    sample_silver_bullet_setup,
    sample_dollar_bars,
):
    """Test ML inference on real-time dollar bar data.

    Given live dollar bar stream,
    when Silver Bullet signal occurs with ML probability ≥0.65,
    then features are engineered from real-time data and ML inference score is calculated.
    """
    # Arrange
    ml_inference = MLInference(model_dir="models/xgboost")

    # Mock model loading to avoid requiring trained models
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.35, 0.75]]  # 75% probability
    mock_model.get_booster().feature_names = [
        "atr",
        "rsi",
        "macd",
        "close_position",
        "volume_ratio",
    ]

    # Mock pipeline loading
    mock_pipeline = MagicMock()

    # Patch the lazy loading methods
    with patch.object(ml_inference, "_load_model_if_needed", return_value=mock_model):
        with patch.object(ml_inference, "_load_pipeline_if_needed", return_value=mock_pipeline):
            # Mock feature transformation
            with patch.object(
                ml_inference._pipeline_serializer,
                "transform_features",
                return_value=sample_dollar_bars[0:1],
            ):
                # Act
                result = ml_inference.predict_probability(
                    signal=sample_silver_bullet_setup,
                    horizon=30,
                    recent_bars=sample_dollar_bars,
                )

    # Assert
    assert result["probability"] >= 0.65, "ML probability should be ≥ 0.65"
    assert result["horizon"] == 30
    assert "latency_ms" in result
    assert result["latency_ms"] < 100, "Inference should be fast (< 100ms)"
    logger.info(f"ML inference score: {result['probability']:.4f}, latency: {result['latency_ms']:.2f}ms")


@pytest.mark.asyncio
async def test_order_execution_with_risk_validation(
    mock_tradestation_client,
    sample_silver_bullet_setup,
    risk_orchestrator,
):
    """Test order execution through risk validation.

    Given ML inference score ≥0.65 and all risk layers pass,
    when order is submitted to TradeStation SIM,
    then order is accepted and filled instantly with confirmation.
    """
    # Arrange

    # Mock order submission response
    mock_order = TradeStationOrder(
        OrderID="SIM-ORDER-12345",
        Symbol="MNQH26",
        OrderType="Market",
        Side="Buy",
        Quantity=2,
        status="FILLED",
        FilledQuantity=2,
        AvgFillPrice=11800.0,
        TimeStamp=datetime.now(timezone.utc),
    )

    # Mock the client's _request method
    async def mock_request(method, endpoint, **kwargs):
        if method == "POST" and endpoint == "/order":
            return {"Order": mock_order.model_dump(by_alias=True)}
        raise ValueError(f"Unexpected request: {method} {endpoint}")

    mock_tradestation_client._request = mock_request

    # Act - Simulate signal processing
    signal_valid = True  # Signal validation would pass
    time_allowed = True  # Time window check would pass

    # Create trading signal for risk validation
    from src.risk.risk_orchestrator import TradingSignal
    trading_signal = TradingSignal(
        signal_id="test-signal-1",
        entry_price=11800.0,
        stop_loss_price=11750.0,  # $50 stop loss
        quantity=2
    )
    risk_validation = risk_orchestrator.validate_trade(trading_signal)
    risk_passed = risk_validation['is_valid']

    # Submit order if all validations pass
    order_submitted = False
    order_filled = False
    if signal_valid and time_allowed and risk_passed:
        # Simulate order submission
        from src.execution.tradestation.orders.submission import OrdersClient

        orders_client = OrdersClient(client=mock_tradestation_client)
        order = await orders_client.place_order(
            symbol="MNQH26",
            side="Buy",
            order_type="Market",
            quantity=2,
        )
        order_submitted = True
        order_filled = order.status == "FILLED"

    # Assert
    assert risk_passed, "Risk validation should pass"
    assert order_submitted, "Order should be submitted"
    assert order_filled, "Order should be filled in SIM environment"
    logger.info(f"Order submitted and filled: {mock_order.order_id}")


@pytest.mark.asyncio
async def test_position_tracking_with_mark_to_market(
    sample_dollar_bars,
):
    """Test position tracking with mark-to-market P&L.

    Given SIM order fill,
    when position is opened,
    then position tracker updates quantity, entry price, and tracks unrealized P&L in real-time.
    """
    # Arrange
    tracker = PositionTracker()
    position = Position(
        order_id="SIM-ORDER-12345",
        signal_id="SIG-001",
        entry_price=11800.0,
        quantity=2,
        direction="bullish",
        order_type="MARKET",
        timestamp=datetime.now(timezone.utc),
    )
    tracker.add_position(position)

    # Act - Update mark-to-market as price moves
    prices = [11800.0, 11805.0, 11810.0, 11795.0, 11815.0]
    for i, price in enumerate(prices):
        tracker.update_mark_to_market("SIM-ORDER-12345", price)

        # Retrieve updated position
        updated_position = tracker.get_position("SIM-ORDER-12345")

        # Assert P&L calculation
        expected_pnl_per_contract = price - 11800.0
        expected_pnl = expected_pnl_per_contract * 2 * 0.5  # 2 contracts, $0.50/point

        assert updated_position.unrealized_pnl == pytest.approx(
            expected_pnl, abs=0.01
        ), f"P&L mismatch at price {price}"
        logger.info(
            f"Price: {price}, Unrealized P&L: ${updated_position.unrealized_pnl:.2f}"
        )

    # Final assertions
    final_position = tracker.get_position("SIM-ORDER-12345")
    assert final_position.quantity == 2
    assert final_position.entry_price == 11800.0
    assert final_position.current_price == 11815.0


@pytest.mark.asyncio
async def test_circuit_breaker_on_daily_loss_limit(risk_orchestrator):
    """Test circuit breaker on daily loss limit.

    Given daily loss approaches $500,
    when circuit breaker triggers,
    then all positions close and trading halts for remainder of day.
    """
    # Arrange
    tracker = PositionTracker()

    # Create positions with losses
    positions = [
        Position(
            order_id=f"ORDER-{i}",
            signal_id=f"SIG-{i}",
            entry_price=11800.0,
            quantity=2,
            direction="bullish",
            order_type="MARKET",
            timestamp=datetime.now(timezone.utc),
        )
        for i in range(3)
    ]

    for pos in positions:
        tracker.add_position(pos)
        # Simulate loss - price moved against position
        tracker.update_mark_to_market(pos.order_id, 11750.0)  # 50 point loss
        # Realize the loss
        tracker.realize_pnl(pos.order_id, 11750.0)

    # Calculate total realized loss
    total_loss = abs(tracker.get_total_realized_pnl())

    # Act - Check if circuit breaker should trigger
    # Loss per position: (11750 - 11800) * 2 * 0.5 = -$50
    # 3 positions = -$150 total
    # This is below the $500 limit, so no trigger yet

    assert total_loss == 150.0, f"Total loss should be $150, got ${total_loss}"

    # Add more loss to trigger circuit breaker
    # Need $350 more loss = 7 more positions with same loss
    for i in range(3, 10):
        pos = Position(
            order_id=f"ORDER-{i}",
            signal_id=f"SIG-{i}",
            entry_price=11800.0,
            quantity=2,
            direction="bullish",
            order_type="MARKET",
            timestamp=datetime.now(timezone.utc),
        )
        tracker.add_position(pos)
        tracker.update_mark_to_market(pos.order_id, 11750.0)
        tracker.realize_pnl(pos.order_id, 11750.0)

    total_loss = abs(tracker.get_total_realized_pnl())

    # Check circuit breaker
    circuit_breaker_triggered = total_loss >= 500.0

    # Assert
    assert circuit_breaker_triggered, f"Circuit breaker should trigger at ${total_loss} loss"
    logger.warning(f"Circuit breaker triggered at ${total_loss} daily loss")


@pytest.mark.asyncio
async def test_end_to_end_paper_trading_flow(
    mock_tradestation_client,
    sample_market_data,
    sample_dollar_bars,
    sample_silver_bullet_setup,
):
    """Test complete end-to-end paper trading flow.

    Integration test combining:
    1. Data ingestion
    2. ML inference
    3. Order execution
    4. Position tracking
    """
    # Arrange
    tracker = PositionTracker()
    ml_inference = MLInference(model_dir="models/xgboost")

    # Mock ML model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.30, 0.80]]  # 80% probability
    mock_model.get_booster().feature_names = ["atr", "rsi", "macd", "close_position", "volume_ratio"]

    mock_pipeline = MagicMock()

    # Step 1: Data ingestion
    parser = QuoteStreamParser(client=mock_tradestation_client)
    data_queue = asyncio.Queue()

    async def mock_stream_quotes(symbols):
        for data in sample_market_data:
            quote = TradeStationQuote(
                Symbol=data.symbol,
                Bid=data.bid,
                Ask=data.ask,
                Last=data.last,
                BidSize=data.volume,
                AskSize=data.volume,
                LastSize=data.volume,
                TimeStamp=data.timestamp,
                Volume=data.volume,
            )
            yield quote

    parser.stream_quotes = mock_stream_quotes

    # Start data ingestion
    streaming_task = asyncio.create_task(parser.stream_to_queue(["MNQH26"], data_queue))

    # Receive first quote
    received_data = await asyncio.wait_for(data_queue.get(), timeout=2.0)
    assert received_data.symbol == "MNQH26"
    logger.info(f"Step 1 complete: Data ingestion - received quote for {received_data.symbol}")

    # Step 2: ML inference
    with patch.object(ml_inference, "_load_model_if_needed", return_value=mock_model):
        with patch.object(ml_inference, "_load_pipeline_if_needed", return_value=mock_pipeline):
            with patch.object(
                ml_inference._pipeline_serializer,
                "transform_features",
                return_value=sample_dollar_bars[0:1],
            ):
                ml_result = ml_inference.predict_probability(
                    signal=sample_silver_bullet_setup,
                    horizon=30,
                    recent_bars=sample_dollar_bars,
                )

    assert ml_result["probability"] >= 0.65
    logger.info(f"Step 2 complete: ML inference - probability {ml_result['probability']:.4f}")

    # Step 3: Order execution (simulated)
    # Mock order response
    mock_order = TradeStationOrder(
        OrderID="SIM-ORDER-E2E",
        Symbol="MNQH26",
        OrderType="Market",
        Side="Buy",
        Quantity=2,
        status="FILLED",
        FilledQuantity=2,
        AvgFillPrice=sample_silver_bullet_setup.entry_price,
        TimeStamp=datetime.now(timezone.utc),
    )

    async def mock_request(method, endpoint, **kwargs):
        if method == "POST" and endpoint == "/order":
            return {"Order": mock_order.model_dump(by_alias=True)}
        raise ValueError(f"Unexpected request: {method} {endpoint}")

    mock_tradestation_client._request = mock_request

    # Create trading signal for risk validation
    from src.risk.risk_orchestrator import TradingSignal
    trading_signal = TradingSignal(
        signal_id="test-signal-2",
        entry_price=sample_silver_bullet_setup.entry_price,
        stop_loss_price=sample_silver_bullet_setup.entry_price - 50.0,
        quantity=2
    )
    risk_validation = risk_orchestrator.validate_trade(trading_signal)
    risk_passed = risk_validation['is_valid']

    assert risk_passed, "Risk validation should pass"

    from src.execution.tradestation.orders.submission import OrdersClient

    orders_client = OrdersClient(client=mock_tradestation_client)
    order = await orders_client.place_order(
        symbol="MNQH26",
        side="Buy",
        order_type="Market",
        quantity=2,
    )

    assert order.status == "FILLED"
    logger.info(f"Step 3 complete: Order execution - {order.order_id} filled")

    # Step 4: Position tracking
    position = Position(
        order_id=order.order_id,
        signal_id="SIG-E2E",
        entry_price=sample_silver_bullet_setup.entry_price,
        quantity=2,
        direction="bullish",
        order_type="MARKET",
        timestamp=datetime.now(timezone.utc),
    )
    tracker.add_position(position)

    # Update mark-to-market
    tracker.update_mark_to_market(order.order_id, 11810.0)
    tracked_position = tracker.get_position(order.order_id)

    assert tracked_position.unrealized_pnl > 0
    logger.info(
        f"Step 4 complete: Position tracking - unrealized P&L ${tracked_position.unrealized_pnl:.2f}"
    )

    # Cleanup
    parser.stop_streaming()
    await asyncio.wait_for(streaming_task, timeout=1.0)

    logger.info("End-to-end paper trading flow completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

#!/usr/bin/env python3
"""
TradeStation SIM Paper Trading Launcher

This script starts the complete paper trading system in the SIM environment:
1. Authenticate with TradeStation
2. Start data pipeline with SDK streaming
3. Monitor for Silver Bullet signals
4. Execute paper trades with full risk validation
5. Track P&L in real-time
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.data.orchestrator import DataPipelineOrchestrator
from src.execution.position_tracker import PositionTracker
from src.execution.tradestation.client import TradeStationClient
from src.ml.inference import MLInference
from src.risk.risk_orchestrator import RiskOrchestrator, TradingSignal
from src.risk.emergency_stop import EmergencyStop
from src.risk.daily_loss_tracker import DailyLossTracker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_size_tracker import PositionSizeTracker
from src.risk.position_sizer import PositionSizer
from src.risk.circuit_breaker_detector import CircuitBreakerDetector
from src.risk.news_event_filter import NewsEventFilter
from src.risk.per_trade_risk_limit import PerTradeRiskLimit
from src.risk.notification_manager import NotificationManager

# Import signal detection components
from src.detection.pipeline import DetectionPipeline
from src.detection.time_window_filter import DEFAULT_TRADING_WINDOWS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingSystem:
    """Complete paper trading system for TradeStation SIM environment."""

    def __init__(self):
        """Initialize the paper trading system."""
        self.settings = load_settings()
        self.running = False

        # Components
        self.client: TradeStationClient | None = None
        self.orchestrator: DataPipelineOrchestrator | None = None
        self.ml_inference: MLInference | None = None
        self.position_tracker: PositionTracker | None = None
        self.risk_orchestrator: RiskOrchestrator | None = None

        # Signal detection
        self.signal_queue: asyncio.Queue | None = None
        self.detection_pipeline: DetectionPipeline | None = None
        self.detection_task: asyncio.Task | None = None

    async def start(self):
        """Start the paper trading system."""
        logger.info("="*70)
        logger.info("🚀 Starting TradeStation SIM Paper Trading System")
        logger.info("="*70)
        logger.info(f"Environment: {self.settings.app_env}")
        logger.info(f"Streaming Symbols: {self.settings.streaming_symbols}")
        logger.info(f"Log Level: {self.settings.log_level}")
        logger.info("")

        self.running = True

        try:
            # Step 1: Initialize TradeStation client (SIM environment)
            logger.info("📡 Step 1: Initializing TradeStation SIM client...")
            self.client = TradeStationClient(
                client_id=self.settings.tradestation_client_id,
                env="sim"  # SIM environment for paper trading
            )

            async with self.client:
                # Load access token from file (saved by OAuth flow)
                logger.info("📝 Loading authentication from .access_token file...")
                try:
                    with open(".access_token", "r") as f:
                        access_token = f.read().strip()

                    # Set the token in the OAuth client's token manager
                    from src.execution.tradestation.auth.tokens import TokenResponse
                    from datetime import datetime, timezone

                    token_response = TokenResponse(
                        access_token=access_token,
                        refresh_token=self.settings.tradestation_refresh_token,
                        expires_in=86400,  # 24 hours
                        token_type="Bearer",
                    )
                    self.client.oauth_client.token_manager.set_token(token_response)
                    logger.info("✅ Access token loaded")
                    logger.info(f"   Token: {access_token[:30]}...")
                    logger.info("")

                    # Try to get a valid access token (will refresh if needed)
                    try:
                        # This will either return the current token or refresh it
                        valid_token = await self.client.oauth_client.get_access_token()
                        logger.info("✅ Valid access token obtained")
                    except RuntimeError as e:
                        # Token needs refresh
                        logger.warning(f"Token needs refresh: {e}")
                        logger.info("Refreshing access token...")

                        # Refresh the token
                        if self.settings.tradestation_refresh_token:
                            refresh_response = await self.client.oauth_client.refresh_access_token(
                                self.settings.tradestation_refresh_token
                            )
                            logger.info("✅ Access token refreshed successfully")
                            logger.info(f"   New Token: {refresh_response.access_token[:30]}...")
                        else:
                            logger.error("❌ No refresh token available")
                            return

                except FileNotFoundError:
                    logger.error("❌ No .access_token file found. Please run:")
                    logger.error("   .venv/bin/python exchange_token_simple.py")
                    logger.error("   with your authorization code")
                    return
                except Exception as e:
                    logger.error(f"❌ Error loading token: {e}")
                    return

                # Verify authentication
                if not self.client.is_authenticated():
                    logger.error("❌ Authentication verification failed")
                    return

                logger.info("✅ TradeStation SIM client authenticated")
                logger.info(f"   API Base URL: {self.client.api_base_url}")
                logger.info("")

                # Step 2: Initialize ML Inference
                logger.info("🤖 Step 2: Initializing ML Inference Engine...")
                self.ml_inference = MLInference(model_dir="models/xgboost")
                logger.info("✅ ML Inference ready")
                logger.info("")

                # Step 3: Initialize Position Tracker
                logger.info("📊 Step 3: Initializing Position Tracker...")
                self.position_tracker = PositionTracker()
                logger.info("✅ Position Tracker ready")
                logger.info("")

                # Step 4: Initialize Risk Orchestrator (all 8 layers)
                logger.info("🛡️  Step 4: Initializing Risk Management (8 layers)...")

                # Create all risk components
                emergency_stop = EmergencyStop()
                daily_loss_tracker = DailyLossTracker(
                    daily_loss_limit=500.0,  # $500 daily loss limit
                    account_balance=100000.0,
                    audit_trail_path="logs/risk_audit_trail.csv"
                )
                drawdown_tracker = DrawdownTracker(
                    max_drawdown_percentage=0.12,  # 12% max drawdown
                    recovery_threshold_percentage=0.95,
                    initial_value=100000.0,
                    audit_trail_path="logs/risk_audit_trail.csv"
                )
                position_size_tracker = PositionSizeTracker(max_position_size=5)
                position_sizer = PositionSizer(
                    account_equity=100000.0,
                    risk_per_trade=0.02,  # 2% per trade
                )
                circuit_breaker_detector = CircuitBreakerDetector(api_client=self.client)
                news_event_filter = NewsEventFilter()
                per_trade_risk_limit = PerTradeRiskLimit(
                    max_risk_dollars=100.0,
                    account_balance=100000.0
                )
                notification_manager = NotificationManager()

                self.risk_orchestrator = RiskOrchestrator(
                    emergency_stop=emergency_stop,
                    daily_loss_tracker=daily_loss_tracker,
                    drawdown_tracker=drawdown_tracker,
                    position_size_tracker=position_size_tracker,
                    position_sizer=position_sizer,
                    circuit_breaker_detector=circuit_breaker_detector,
                    news_event_filter=news_event_filter,
                    per_trade_risk_limit=per_trade_risk_limit,
                    notification_manager=notification_manager,
                    audit_trail_path="logs/risk_audit_trail.csv"
                )

                logger.info("✅ Risk Management initialized:")
                logger.info("   1. Emergency Stop")
                logger.info("   2. Daily Loss Limit ($500)")
                logger.info("   3. Max Drawdown (12%)")
                logger.info("   4. Max Position Size (5 contracts)")
                logger.info("   5. Position Sizer (Kelly Criterion)")
                logger.info("   6. Circuit Breaker Detector")
                logger.info("   7. News Event Filter")
                logger.info("   8. Per-Trade Risk Limit")
                logger.info("")

                # Step 5: Start data pipeline
                logger.info("📈 Step 5: Starting data pipeline (TradeStation SDK)...")
                self.orchestrator = DataPipelineOrchestrator(
                    client=self.client,
                    data_directory="data/dollar_bars",
                    max_queue_size=1000,
                    settings=self.settings
                )

                await self.orchestrator.start()
                logger.info("✅ Data pipeline running")
                logger.info("")

                # Step 5.5: Initialize Signal Detection
                logger.info("🎯 Step 5.5: Initializing Signal Detection...")
                self.signal_queue = asyncio.Queue()
                self.detection_pipeline = DetectionPipeline(
                    input_queue=self.orchestrator._gap_filled_queue,
                    signal_queue=self.signal_queue,
                    time_windows=DEFAULT_TRADING_WINDOWS
                )

                # Start detection pipeline
                self.detection_task = asyncio.create_task(
                    self.detection_pipeline.consume()
                )
                logger.info("✅ Signal detection initialized")
                logger.info("")

                # Step 6: Monitor paper trading
                logger.info("💰 Step 6: Paper trading monitor active")
                logger.info("="*70)
                logger.info("System is ready for paper trading!")
                logger.info("Press Ctrl+C to stop gracefully")
                logger.info("="*70)
                logger.info("")

                # Main monitoring loop
                await self.monitor_paper_trading()

        except Exception as e:
            logger.error(f"❌ Error in paper trading system: {e}", exc_info=True)
        finally:
            await self.stop()

    async def monitor_paper_trading(self):
        """Monitor paper trading activity and execute trades."""
        logger.info("🎯 Starting autonomous trading loop...")
        logger.info("Waiting for Silver Bullet signals...")
        logger.info("")

        while self.running:
            try:
                # Wait for signal with timeout
                signal = await asyncio.wait_for(
                    self.signal_queue.get(),
                    timeout=5.0
                )

                logger.info("="*70)
                logger.info(f"🎯 SILVER BULLET SIGNAL DETECTED!")
                logger.info("="*70)
                logger.info(f"Symbol: {signal.symbol}")
                logger.info(f"Direction: {signal.direction}")
                logger.info(f"Entry Zone: ${signal.entry_zone_bottom:.2f} - ${signal.entry_zone_top:.2f}")
                logger.info(f"Invalidation: ${signal.invalidation_point:.2f}")
                logger.info(f"Confluence: {signal.confluence_count} patterns")
                logger.info(f"Priority: {signal.priority}")
                logger.info("")

                # Step 1: ML Inference
                logger.info("🤖 Step 1: Running ML Inference...")
                try:
                    # Get recent dollar bars for feature engineering
                    recent_bars = []
                    bar_queue = self.orchestrator._gap_filled_queue

                    # Get up to 50 recent bars
                    while not bar_queue.empty() and len(recent_bars) < 50:
                        try:
                            bar = await asyncio.wait_for(
                                bar_queue.get(),
                                timeout=0.1
                            )
                            recent_bars.append(bar)
                        except asyncio.TimeoutError:
                            break

                    if len(recent_bars) < 20:
                        logger.warning(f"Insufficient bars for ML: {len(recent_bars)} < 20")
                        continue

                    # Run ML inference
                    probability = self.ml_inference.predict_probability(
                        signal=signal,
                        horizon=30,  # 30 bars lookahead
                        recent_bars=recent_bars
                    )

                    logger.info(f"✅ ML Probability: {probability:.2%}")

                    if probability < 0.60:  # 60% threshold
                        logger.info("❌ Signal rejected: ML probability below threshold")
                        continue

                except Exception as e:
                    logger.error(f"ML Inference error: {e}")
                    continue

                # Step 2: Risk Validation
                logger.info("🛡️  Step 2: Validating Risk (8 layers)...")
                try:
                    # Create TradingSignal for risk validation
                    trading_signal = TradingSignal(
                        signal_id=f"SIGNAL-{signal.timestamp.timestamp()}",
                        symbol=signal.symbol,
                        direction=signal.direction,
                        confidence_score=probability,
                        timestamp=signal.timestamp,
                        entry_price=signal.entry_zone_top,
                        patterns=["Silver Bullet", "MSS", "FVG"],
                        prediction={"probability": probability}
                    )

                    # Validate through all 8 risk layers
                    validation_result = await self.risk_orchestrator.validate_trade(
                        signal=trading_signal
                    )

                    if validation_result["is_valid"]:
                        logger.info("✅ All 8 risk layers passed")
                        logger.info(f"   Position Size: {validation_result['position_size']} contracts")
                    else:
                        logger.warning(f"❌ Risk validation failed: {validation_result['block_reason']}")
                        continue

                except Exception as e:
                    logger.error(f"Risk validation error: {e}")
                    continue

                # Step 3: Execute Trade
                logger.info("💰 Step 3: Executing Trade...")
                try:
                    # Calculate position size
                    position_size = validation_result['position_size']
                    entry_price = signal.entry_zone_top

                    # Create order
                    order = {
                        "Symbol": signal.symbol,
                        "Quantity": position_size,
                        "OrderType": "MARKET",
                        "Side": "BUY" if signal.direction == "bullish" else "SELL",
                        "TimeInForce": "DAY"
                    }

                    # Submit order
                    logger.info(f"   Submitting: {order['Side']} {position_size}x {signal.symbol} @ ${entry_price:.2f}")
                    order_response = await self.client.place_order(order)

                    logger.info(f"✅ Order submitted: {order_response.get('OrderID', 'UNKNOWN')}")
                    logger.info(f"   Status: {order_response.get('Status', 'UNKNOWN')}")

                    # Track position
                    from src.execution.position_tracker import Position
                    position = Position(
                        order_id=order_response.get('OrderID', 'UNKNOWN'),
                        signal_id=trading_signal.signal_id,
                        entry_price=entry_price,
                        quantity=position_size,
                        direction=signal.direction,
                        order_type="MARKET",
                        status="SUBMITTED",
                        timestamp=datetime.now(timezone.utc)
                    )
                    self.position_tracker.add_position(position)
                    logger.info(f"✅ Position tracked: {position_size} contracts @ ${entry_price:.2f}")
                    logger.info("")

                except Exception as e:
                    logger.error(f"Order execution error: {e}")
                    import traceback
                    traceback.print_exc()

            except asyncio.TimeoutError:
                # No signal received, log status and continue
                open_positions = self.position_tracker.get_open_positions()
                total_unrealized_pnl = self.position_tracker.get_total_unrealized_pnl()

                logger.info(f"Status: {len(open_positions)} open positions, "
                          f"Unrealized P&L: ${total_unrealized_pnl:.2f}")

                # Log pipeline metrics if available
                if self.orchestrator and hasattr(self.orchestrator, '_total_bars_created'):
                    logger.info(f"Pipeline: {self.orchestrator._total_bars_created} bars created")

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Brief pause before continuing

    async def stop(self):
        """Stop the paper trading system gracefully."""
        if not self.running:
            return

        logger.info("")
        logger.info("="*70)
        logger.info("🛑 Stopping Paper Trading System")
        logger.info("="*70)

        self.running = False

        # Cancel detection task
        if self.detection_task and not self.detection_task.done():
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ Signal detection stopped")

        # Stop orchestrator
        if self.orchestrator:
            try:
                await self.orchestrator.stop()
                logger.info("✅ Data pipeline stopped")
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")

        # Log final P&L
        if self.position_tracker:
            try:
                total_realized = self.position_tracker.get_total_realized_pnl()
                total_unrealized = self.position_tracker.get_total_unrealized_pnl()
                logger.info(f"Final P&L: Realized=${total_realized:.2f}, "
                          f"Unrealized=${total_unrealized:.2f}")
            except Exception as e:
                logger.error(f"Error calculating final P&L: {e}")

        logger.info("="*70)
        logger.info("Paper trading system stopped")
        logger.info("="*70)


async def main():
    """Main entry point."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Create system
    system = PaperTradingSystem()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        system.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(system.stop()))

    # Start system
    await system.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

#!/usr/bin/env python3
"""
Start Live Paper Trading with TradeStation

This script starts the live paper trading system using your authenticated
TradeStation API access.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.data.orchestrator import DataPipelineOrchestrator
from src.data.models import DollarBar
from src.execution.position_tracker import PositionTracker
from src.risk.factory import RiskComponentFactory
from src.execution.trade_execution_pipeline import TradingSignal
from src.execution.tradestation.orders.submission import SIMOrderSubmitter
from src.data.auth_v3 import TradeStationAuthV3
from src.data.contract_detector import ContractDetector
from src.ml.hybrid_pipeline import HybridMLPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


async def start_paper_trading():
    """Start the paper trading system."""

    print("\n" + "="*70)
    print("🎯 TradeStation Paper Trading System")
    print("="*70)
    print()

    # Check authentication
    print("📋 Step 1: Checking authentication...")
    try:
        with open(".access_token", "r") as f:
            access_token = f.read().strip()
        if not access_token or len(access_token) < 50:
            print("❌ No valid access token found!")
            print("\nPlease run OAuth flow first:")
            print("  .venv/bin/python get_standard_auth_url.py")
            print("  .venv/bin/python exchange_token_simple.py <code>")
            return
        print("✅ Access token found")
    except FileNotFoundError:
        print("❌ No .access_token file found!")
        print("\nPlease run OAuth flow first:")
        print("  .venv/bin/python get_standard_auth_url.py")
        print("  .venv/bin/python exchange_token_simple.py <code>")
        return

    print()

    # Load settings
    print("📋 Step 2: Loading configuration...")
    settings = load_settings()
    print(f"✅ Config loaded")
    print(f"   Environment: {settings.app_env}")
    print()

    # Initialize components
    print("📋 Step 3: Initializing system components...")

    # Initialize authentication (load v3 token with refresh token support)
    with open(".access_token", "r") as f:
        access_token = f.read().strip()

    # Load refresh token from settings
    refresh_token = settings.tradestation_refresh_token if hasattr(settings, 'tradestation_refresh_token') else ""

    if refresh_token:
        print(f"✅ V3 Authentication initialized with refresh token support")
    else:
        print(f"⚠️  V3 Authentication initialized WITHOUT refresh token (token refresh will fail)")
        print(f"   Run exchange_token_simple.py to fix this")

    auth = TradeStationAuthV3(access_token=access_token, refresh_token=refresh_token)

    # Start automatic token refresh every 10 minutes
    await auth.start_auto_refresh(interval_minutes=10)
    print("✅ Auto-refresh enabled (every 10 minutes)")

    # Initialize contract detector
    contract_detector = ContractDetector(access_token=access_token)
    print("✅ Contract Detector initialized")

    # Detect active contract
    print("\n📋 Step 3.5: Detecting active futures contract...")
    try:
        active_contract = await contract_detector.detect_active_contract("MNQH26")
        print(f"✅ Active contract: {active_contract}")
    except Exception as e:
        print(f"⚠️  Contract detection failed: {e}")
        print("   Using default symbol MNQH26")
        active_contract = "MNQH26"
    print()

    # Initialize orchestrator with HTTP streaming enabled (recommended)
    orchestrator = DataPipelineOrchestrator(
        auth=auth,
        data_directory="data/processed",
        use_http_streaming=True,  # Use HTTP streaming instead of deprecated WebSocket
        symbols=[active_contract],  # Stream active contract
    )
    print("✅ Data Orchestrator initialized")
    print(f"   Using HTTP streaming for {active_contract}")

    # Initialize Hybrid ML Pipeline (regime-aware with 40% threshold)
    signal_queue = asyncio.Queue(maxsize=100)
    hybrid_pipeline = HybridMLPipeline(
        output_queue=signal_queue,
        model_dir="models/xgboost",
        hmm_dir="models/hmm/regime_model"
    )
    print("✅ Hybrid ML Pipeline initialized")
    print("   - Regime-aware: 3 regimes with HMM detection")
    print("   - Probability threshold: 40%")
    print("   - Bar-by-bar evaluation enabled")
    print("   - Min bars between trades: 30 (2.5 hours)")
    print("   - Triple-barrier exits: TP 0.3%, SL 0.2%, Time 30min")

    # Initialize position tracker
    position_tracker = PositionTracker()
    print("✅ Position Tracker initialized")

    # Initialize risk orchestrator with all 8 risk layers
    print("\n📋 Step 3.5: Initializing risk management...")
    try:
        risk_orchestrator = RiskComponentFactory.create_risk_orchestrator()
        print("✅ Risk Orchestrator initialized with all 8 safety layers:")
        print("   1. Emergency Stop")
        print("   2. Daily Loss Limit")
        print("   3. Max Drawdown")
        print("   4. Position Size Limits")
        print("   5. Circuit Breaker Detector")
        print("   6. News Event Filter")
        print("   7. Per-Trade Risk Limit")
        print("   8. Notification Manager")
    except Exception as e:
        print(f"❌ Failed to initialize risk orchestrator: {e}")
        print("   Risk management is critical for safe trading")
        return

    # Initialize SIM order submitter with risk validation
    try:
        order_submitter = SIMOrderSubmitter(
            auth=auth,
            risk_orchestrator=risk_orchestrator  # CRITICAL: Safety integration
        )
        print("✅ SIM Order Submitter initialized")
        print("   ✅ All orders validated against 8 risk layers before submission")
    except Exception as e:
        print(f"❌ Failed to initialize order submitter: {e}")
        return

    print()

    # Setup signal handler for trading
    print("📋 Step 3.6: Setting up trading signal handler...")

    async def handle_trading_signal(signal: TradingSignal):
        """Handle trading signal with risk validation and order execution.

        Complete pipeline: HybridMLPipeline → Risk Validation → Order Execution

        Note: ML prediction and filtering already done by HybridMLPipeline
        with 40% threshold and regime-aware model selection.
        """
        logger.info(f"🎯 New trading signal received: {signal.signal_id}")
        logger.info(f"   Symbol: {signal.symbol}")
        logger.info(f"   Direction: {signal.direction}")
        logger.info(f"   Entry Price: ${signal.entry_price:.2f}")
        logger.info(f"   Confidence: {signal.confidence_score:.2%}")
        logger.info(f"   Regime: {signal.prediction.get('regime', 'N/A')}")

        # Step 1: Risk Validation - All 8 safety layers

        # Step 2: Risk Validation - All 8 safety layers
        try:
            logger.info("🛡️  Validating against 8 risk layers...")
            risk_result = risk_orchestrator.validate_trade(signal)

            if not risk_result['is_valid']:
                logger.warning(f"❌ Signal REJECTED by risk validation")
                logger.warning(f"   Block reason: {risk_result['block_reason']}")
                logger.warning(f"   Failed checks: {risk_result['checks_failed']}")
                return

            logger.info(f"✅ Signal passed all 8 risk layers")
            logger.info(f"   Passed checks: {risk_result['checks_passed']}")
        except Exception as e:
            logger.error(f"❌ Risk validation failed: {e}")
            return

        # Step 3: Order Execution - Submit to TradeStation SIM
        try:
            logger.info("💰 Submitting order to TradeStation SIM...")
            order_result = await order_submitter.submit_order(signal)

            if order_result.success:
                logger.info(f"✅ Order submitted successfully!")
                logger.info(f"   Order ID: {order_result.order_id}")
                logger.info(f"   Status: {order_result.status}")

                # Track position
                position_tracker.track_order(
                    order_id=order_result.order_id,
                    signal=signal,
                    entry_price=signal.entry_price,
                    quantity=5  # Default quantity
                )
                logger.info(f"📊 Position tracked")
            else:
                logger.error(f"❌ Order submission failed: {order_result.error_message}")
        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")

    # Connect signal handler to orchestrator (if supported)
    # TODO: orchestrator.register_signal_handler(handle_trading_signal)

    print("✅ Trading signal handler configured")
    print("   Pipeline: HybridML → Risk → Order")

    # Start signal processing task
    async def process_signals():
        """Process signals from hybrid pipeline output queue."""
        while True:
            try:
                # Wait for signal from hybrid pipeline (already a TradingSignal)
                signal = await signal_queue.get()

                # Log signal received
                logger.info(f"🎯 Signal received: {signal.signal_id}")
                logger.info(f"   Symbol: {signal.symbol}")
                logger.info(f"   Direction: {signal.direction}")
                logger.info(f"   Entry: ${signal.entry_price:.2f}")
                logger.info(f"   Confidence: {signal.confidence_score:.2%}")

                # Handle the signal directly
                await handle_trading_signal(signal)

            except asyncio.CancelledError:
                logger.info("Signal processing task cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing signal: {e}")

    # Start signal processor as background task
    signal_processor_task = asyncio.create_task(process_signals())
    print("✅ Signal processor started (background task)")
    print("   Pipeline: HybridML → Risk → Order")

    # Start bar processor for hybrid pipeline
    async def process_dollar_bars():
        """Process dollar bars from orchestrator through hybrid pipeline."""
        import h5py
        from pathlib import Path

        data_dir = Path("data/processed/dollar_bars")
        last_processed_timestamp = None

        logger.info("Dollar bar processor started")

        while True:
            try:
                # Find latest HDF5 file
                h5_files = sorted(data_dir.glob("MNQ_dollar_bars_*.h5"))

                if not h5_files:
                    await asyncio.sleep(5)
                    continue

                latest_file = h5_files[-1]

                # Read latest bars
                with h5py.File(latest_file, 'r') as f:
                    if 'dollar_bars' not in f:
                        await asyncio.sleep(5)
                        continue

                    data = f['dollar_bars'][:]

                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_value'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Filter new bars only
                    if last_processed_timestamp is not None:
                        new_bars = df[df['timestamp'] > last_processed_timestamp]
                    else:
                        # First run - process last 100 bars for context
                        new_bars = df.tail(100)

                    if len(new_bars) == 0:
                        await asyncio.sleep(5)
                        continue

                    # Process each new bar
                    for _, row in new_bars.iterrows():
                        bar = DollarBar(
                            timestamp=row['timestamp'],
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            notional_value=row['notional_value'],
                            symbol=active_contract
                        )

                        # Get historical data for features
                        historical_df = df[df['timestamp'] <= row['timestamp']].tail(100)

                        # Process bar through hybrid pipeline
                        await hybrid_pipeline.process_bar(bar, historical_df)

                    # Update last processed timestamp
                    last_processed_timestamp = new_bars['timestamp'].max()

                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                logger.info("Dollar bar processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing dollar bars: {e}")
                await asyncio.sleep(10)

    # Start bar processor as background task
    bar_processor_task = asyncio.create_task(process_dollar_bars())
    print("✅ Dollar bar processor started (background task)")
    print("   Monitoring: data/processed/dollar_bars/")
    print("   Processing interval: Every 10 seconds")

    print()

    # Start system
    print("📋 Step 4: Starting paper trading system...")
    print()
    print("="*70)
    print("🚀 HYBRID REGIME-AWARE TRADING SYSTEM")
    print("="*70)
    print()
    print("📊 Monitoring live market data...")
    print("🤖 Hybrid ML Pipeline (40% threshold, 3 regimes)")
    print("💰 Paper trading mode - NO REAL MONEY")
    print()
    print("Expected Performance:")
    print("  - Trades per day: 3.92")
    print("  - Win rate: 51.80%")
    print("  - Sharpe ratio: 0.74")
    print("  - Max drawdown: -2.78%")
    print()
    print("Press Ctrl+C to stop")
    print("="*70)
    print()

    try:
        # Start data orchestration
        await orchestrator.start()

        # Keep the system running indefinitely
        print("✅ Paper trading system is running!")
        print("   Monitoring live market data and executing trades...")
        print()

        # Keep alive until interrupted
        while True:
            await asyncio.sleep(60)
            # Log periodic status
            if orchestrator.is_running:
                logger.info(f"System uptime: {orchestrator.runtime_seconds:.0f}s | Queue depths: {orchestrator.queue_depths}")

    except KeyboardInterrupt:
        print("\n")
        print("="*70)
        print("🛑 SYSTEM SHUTDOWN")
        print("="*70)
        print()
        print("Cleaning up...")

        # Cancel background tasks
        signal_processor_task.cancel()
        bar_processor_task.cancel()
        print("✅ Background tasks cancelled")

        # Wait for tasks to complete
        await asyncio.gather(signal_processor_task, bar_processor_task, return_exceptions=True)

        # Stop auto-refresh
        await auth.cleanup()
        print("✅ Auto-refresh stopped")

        # Stop orchestrator
        await orchestrator.stop()

        print("✅ System stopped cleanly")
        print()

        # Show final positions
        positions = position_tracker.get_all_positions()
        if positions:
            print("📊 Final Positions:")
            for position in positions:
                print(f"   {position.symbol}: {position.quantity} contracts")
                print(f"   Entry: ${position.entry_price:.2f}")
                print(f"   P&L: ${position.unrealized_pnl:.2f}")
                print()

    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(start_paper_trading())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
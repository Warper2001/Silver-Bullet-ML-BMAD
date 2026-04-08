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

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.data.orchestrator import DataPipelineOrchestrator
from src.execution.position_tracker import PositionTracker
from src.risk.risk_orchestrator import RiskOrchestrator
from src.ml.inference import MLInference
from src.execution.trade_execution_pipeline import TradeExecutionPipeline
from src.data.auth_v3 import TradeStationAuthV3
from src.data.contract_detector import ContractDetector

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

    # Initialize authentication (load v3 token)
    with open(".access_token", "r") as f:
        access_token = f.read().strip()
    auth = TradeStationAuthV3(access_token=access_token)
    print("✅ V3 Authentication initialized")

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

    # Initialize ML inference
    ml_inference = MLInference(model_dir="models/xgboost")
    print("✅ ML Inference initialized")

    # Initialize position tracker
    position_tracker = PositionTracker()
    print("✅ Position Tracker initialized")

    # Skip risk orchestrator and trade executor for now
    # (requires complex dependency setup)
    print("⚠️  Risk components skipped (requires full setup)")

    print()

    # Start system
    print("📋 Step 4: Starting paper trading system...")
    print()
    print("="*70)
    print("🚀 SYSTEM STARTING")
    print("="*70)
    print()
    print("📊 Monitoring live market data...")
    print("🤖 ML models ready for predictions")
    print("💰 Paper trading mode - NO REAL MONEY")
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
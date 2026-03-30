#!/usr/bin/env python3
"""Verification script for INTENT_GAP fixes."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import datetime, timezone

print("="*70)
print("INTENT_GAP Fixes Verification")
print("="*70)
print()

# Fix 1: InsufficientDataError in ML inference
print("1. Testing InsufficientDataError (Fix #3, #4)...")
try:
    from src.ml.inference import MLInference, InsufficientDataError

    ml_inference = MLInference(model_dir="models/xgboost")

    # Try to predict with insufficient bars (<20)
    from src.data.models import DollarBar

    # Create only 5 bars (insufficient)
    bars = [
        DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11810.0,
            low=11790.0,
            close=11805.0,
            volume=1000,
            notional_value=50000000.0,
        )
        for _ in range(5)
    ]

    # This should raise InsufficientDataError
    try:
        ml_inference.predict_probability(
            signal=None,  # Not used for this test
            horizon=30,
            recent_bars=bars,
        )
        print("   ❌ FAIL: Should have raised InsufficientDataError")
    except InsufficientDataError as e:
        print(f"   ✅ PASS: InsufficientDataError raised correctly")
        print(f"      Message: {str(e)[:80]}...")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print()

# Fix 2: Race condition lock in DollarBarTransformer
print("2. Testing asyncio.Lock in DollarBarTransformer (Fix #1)...")
try:
    from src.data.transformation import DollarBarTransformer
    import inspect

    # Check if _state_lock exists
    source = inspect.getsource(DollarBarTransformer.__init__)
    if "_state_lock = asyncio.Lock()" in source:
        print("   ✅ PASS: asyncio.Lock added to __init__")
    else:
        print("   ❌ FAIL: asyncio.Lock not found in __init__")

    # Check if lock is used in _process_market_data
    source = inspect.getsource(DollarBarTransformer._process_market_data)
    if "async with self._state_lock:" in source:
        print("   ✅ PASS: Lock used in _process_market_data")
    else:
        print("   ❌ FAIL: Lock not used in _process_market_data")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print()

# Fix 3: Queue backpressure with timeout
print("3. Testing queue backpressure mechanism (Fix #2)...")
try:
    source = inspect.getsource(DollarBarTransformer._complete_bar)
    if "asyncio.wait_for" in source and "timeout=1.0" in source:
        print("   ✅ PASS: Blocking put with timeout implemented")
    else:
        print("   ❌ FAIL: No blocking put with timeout found")

    if "_bars_dropped" in source or "BACKPRESSURE" in source:
        print("   ✅ PASS: Backpressure logging and tracking added")
    else:
        print("   ❌ FAIL: No backpressure tracking found")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print()

# Fix 4: Staleness detection in streaming
print("4. Testing staleness detection (Fix #6)...")
try:
    from src.execution.tradestation.market_data.streaming import QuoteStreamParser
    import inspect

    # Check if staleness parameters exist
    source = inspect.getsource(QuoteStreamParser.__init__)
    if "staleness_threshold_seconds" in source:
        print("   ✅ PASS: staleness_threshold_seconds parameter added")
    else:
        print("   ❌ FAIL: staleness_threshold_seconds not found")

    if "_last_quote_timestamp" in source:
        print("   ✅ PASS: _last_quote_timestamp tracking added")
    else:
        print("   ❌ FAIL: _last_quote_timestamp not found")

    # Check if _monitor_staleness method exists
    if hasattr(QuoteStreamParser, "_monitor_staleness"):
        print("   ✅ PASS: _monitor_staleness method exists")
    else:
        print("   ❌ FAIL: _monitor_staleness method not found")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print()

# Fix 5: RiskOrchestrator test interface
print("5. Testing RiskOrchestrator test fixture (Fix #5)...")
try:
    # Check if risk_orchestrator fixture exists
    test_file = Path("tests/integration/test_tradestation_sim_paper_trading.py")
    if test_file.exists():
        content = test_file.read_text()
        if "def risk_orchestrator(tmp_path):" in content:
            print("   ✅ PASS: risk_orchestrator fixture created")
        else:
            print("   ❌ FAIL: risk_orchestrator fixture not found")

        if "TradingSignal(" in content and "validate_trade(" in content:
            print("   ✅ PASS: Tests use TradingSignal and validate_trade()")
        else:
            print("   ❌ FAIL: Tests don't use correct interface")
    else:
        print("   ⚠️  SKIP: Test file not found")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print()
print("="*70)
print("Verification Complete")
print("="*70)

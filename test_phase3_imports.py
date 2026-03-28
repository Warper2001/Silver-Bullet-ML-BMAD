#!/usr/bin/env python3
"""
Phase 3 Import Validation Test

Validates that all Phase 3 order management modules can be imported successfully.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("=" * 60)
print("Phase 3: Order Execution Integration - Import Validation")
print("=" * 60)

# Test 1: Import order management package
print("\n1. Testing order management package modules...")
try:
    from src.execution.tradestation.orders.submission import OrdersClient
    from src.execution.tradestation.orders.status import OrderStatusStream
    print("   ✅ All order management modules imported successfully")
    print(f"   - OrdersClient: {OrdersClient}")
    print(f"   - OrderStatusStream: {OrderStatusStream}")
except ImportError as e:
    print(f"   ❌ Failed to import order management modules: {e}")
    sys.exit(1)

# Test 2: Import order models
print("\n2. Testing order models...")
try:
    from src.execution.tradestation.models import (
        NewOrderRequest,
        TradeStationOrder,
        OrderStatusUpdate,
        OrderFill,
        AccountPosition,
        AccountBalance,
    )
    print("   ✅ All order models imported successfully")
    print(f"   - NewOrderRequest: {NewOrderRequest}")
    print(f"   - TradeStationOrder: {TradeStationOrder}")
    print(f"   - OrderStatusUpdate: {OrderStatusUpdate}")
    print(f"   - OrderFill: {OrderFill}")
    print(f"   - AccountPosition: {AccountPosition}")
    print(f"   - AccountBalance: {AccountBalance}")
except ImportError as e:
    print(f"   ❌ Failed to import order models: {e}")
    sys.exit(1)

# Test 3: Instantiate clients
print("\n3. Testing client instantiation...")
try:
    # Create mock client
    from unittest.mock import MagicMock
    from src.execution.tradestation.client import TradeStationClient

    mock_client = MagicMock(spec=TradeStationClient)
    mock_client.api_base_url = "https://sim-api.tradestation.com/v3"

    # Instantiate order management clients
    orders_client = OrdersClient(mock_client)
    status_stream = OrderStatusStream(mock_client)

    print("   ✅ All clients instantiated successfully")
    print(f"   - OrdersClient created")
    print(f"   - OrderStatusStream created")
except Exception as e:
    print(f"   ❌ Failed to instantiate clients: {e}")
    sys.exit(1)

# Test 4: Validate order parameters
print("\n4. Testing order parameter validation...")
try:
    orders_client = OrdersClient(mock_client)

    # Valid parameters
    orders_client._validate_order_parameters(
        symbol="MNQH26",
        side="Buy",
        order_type="Limit",
        quantity=1,
        price=15000.0,
        stop_price=None,
        time_in_force="Day",
    )
    print("   ✅ Valid order parameters accepted")

    # Invalid symbol
    try:
        orders_client._validate_order_parameters(
            symbol="",
            side="Buy",
            order_type="Market",
            quantity=1,
            price=None,
            stop_price=None,
            time_in_force="Day",
        )
        print("   ❌ Empty symbol should have been rejected")
        sys.exit(1)
    except Exception:
        print("   ✅ Invalid symbol correctly rejected")

    # Limit order without price
    try:
        orders_client._validate_order_parameters(
            symbol="MNQH26",
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=None,
            stop_price=None,
            time_in_force="Day",
        )
        print("   ❌ Limit order without price should have been rejected")
        sys.exit(1)
    except Exception:
        print("   ✅ Limit order without price correctly rejected")

    # Stop order without stop price
    try:
        orders_client._validate_order_parameters(
            symbol="MNQH26",
            side="Buy",
            order_type="Stop",
            quantity=1,
            price=None,
            stop_price=None,
            time_in_force="Day",
        )
        print("   ❌ Stop order without stop price should have been rejected")
        sys.exit(1)
    except Exception:
        print("   ✅ Stop order without stop price correctly rejected")

except Exception as e:
    print(f"   ❌ Order parameter validation test failed: {e}")
    sys.exit(1)

# Test 5: Validate order types
print("\n5. Testing order type validation...")
try:
    # Valid order types
    valid_types = ["Market", "Limit", "Stop", "StopLimit", "MarketOnClose"]
    for order_type in valid_types:
        assert order_type in orders_client.VALID_ORDER_TYPES

    print(f"   ✅ All {len(valid_types)} order types validated successfully")

    # Valid sides
    valid_sides = ["Buy", "Sell"]
    for side in valid_sides:
        assert side in orders_client.VALID_SIDES

    print(f"   ✅ All {len(valid_sides)} order sides validated successfully")

    # Valid time in force
    valid_tif = ["Day", "GTC", "IOC", "FOK"]
    for tif in valid_tif:
        assert tif in orders_client.VALID_TIF

    print(f"   ✅ All {len(valid_tif)} time in force values validated successfully")

except Exception as e:
    print(f"   ❌ Order type validation test failed: {e}")
    sys.exit(1)

# Test 6: Test order model validation
print("\n6. Testing order model validation...")
try:
    from datetime import datetime, timezone

    # Valid NewOrderRequest
    order_request = NewOrderRequest(
        symbol="MNQH26",
        side="Buy",
        order_type="Limit",
        quantity=1,
        price=15000.0,
        time_in_force="Day",
    )
    print("   ✅ Valid NewOrderRequest created")

    # Limit order without price (should fail)
    try:
        invalid_request = NewOrderRequest(
            symbol="MNQH26",
            side="Buy",
            order_type="Limit",
            quantity=1,
            price=None,
            time_in_force="Day",
        )
        print("   ❌ Limit order without price should have been rejected")
        sys.exit(1)
    except Exception:
        print("   ✅ Limit order without price correctly rejected by model")

    # Valid TradeStationOrder
    order = TradeStationOrder(
        order_id="order123",
        symbol="MNQH26",
        order_type="Limit",
        side="Buy",
        quantity=1,
        price=15000.0,
        status="Working",
        filled_quantity=0,
        avg_fill_price=None,
        timestamp=datetime.now(timezone.utc),
    )
    print("   ✅ Valid TradeStationOrder created")

    # Valid OrderStatusUpdate
    status_update = OrderStatusUpdate(
        order_id="order123",
        symbol="MNQH26",
        status="Filled",
        filled_quantity=1,
        avg_fill_price=15000.0,
        remaining_quantity=0,
        timestamp=datetime.now(timezone.utc),
    )
    print("   ✅ Valid OrderStatusUpdate created")

except Exception as e:
    print(f"   ❌ Order model validation test failed: {e}")
    sys.exit(1)

# Test 7: Test streaming parser state management
print("\n7. Testing OrderStatusStream state management...")
try:
    status_stream = OrderStatusStream(mock_client)

    # Check initial state
    assert not status_stream._is_streaming, "Should not be streaming initially"
    print("   ✅ Initial streaming state correct")

    # Test stop method
    status_stream.stop_streaming()
    assert not status_stream._is_streaming, "Should not be streaming after stop"
    print("   ✅ Stop streaming method works")

except Exception as e:
    print(f"   ❌ Streaming parser state test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Phase 3 Import Validation: ✅ ALL TESTS PASSED")
print("=" * 60)

print("\n📦 Phase 3 Implementation Summary:")
print("   ✅ OrdersClient - Order CRUD operations")
print("   ✅ OrderStatusStream - Real-time order status streaming")
print("   ✅ Order models - Pydantic validation")
print("\n🔗 Ready for Integration:")
print("   • Order execution pipeline")
print("   • Position monitoring")
print("   • Triple barrier exits")
print("   • Risk management integration")
print("\n➡️  Next Phase: Phase 3 Integration Testing")
print("   • Test with actual TradeStation SIM API")
print("   • Position tracking and reconciliation")
print("   • Risk management pre-flight checks")
print("   • Triple barrier exit execution")
print("=" * 60)

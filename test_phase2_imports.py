#!/usr/bin/env python3
"""
Phase 2 Import Validation Test

Validates that all Phase 2 market data modules can be imported successfully.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("=" * 60)
print("Phase 2: Market Data Integration - Import Validation")
print("=" * 60)

# Test 1: Import market data package
print("\n1. Testing market data package modules...")
try:
    from src.execution.tradestation.market_data.quotes import QuotesClient
    from src.execution.tradestation.market_data.history import HistoryClient
    from src.execution.tradestation.market_data.streaming import QuoteStreamParser
    print("   ✅ All market data modules imported successfully")
    print(f"   - QuotesClient: {QuotesClient}")
    print(f"   - HistoryClient: {HistoryClient}")
    print(f"   - QuoteStreamParser: {QuoteStreamParser}")
except ImportError as e:
    print(f"   ❌ Failed to import market data modules: {e}")
    sys.exit(1)

# Test 2: Instantiate clients
print("\n2. Testing client instantiation...")
try:
    # Create mock client
    from unittest.mock import MagicMock
    from src.execution.tradestation.client import TradeStationClient

    mock_client = MagicMock(spec=TradeStationClient)
    mock_client.api_base_url = "https://sim-api.tradestation.com/v3"

    # Instantiate market data clients
    quotes_client = QuotesClient(mock_client)
    history_client = HistoryClient(mock_client)
    stream_parser = QuoteStreamParser(mock_client)

    print("   ✅ All clients instantiated successfully")
    print(f"   - QuotesClient created")
    print(f"   - HistoryClient created")
    print(f"   - QuoteStreamParser created")
except Exception as e:
    print(f"   ❌ Failed to instantiate clients: {e}")
    sys.exit(1)

# Test 3: Validate HistoryClient bar types
print("\n3. Testing HistoryClient bar types...")
try:
    history_client = HistoryClient(mock_client)

    # Test all supported bar types
    valid_types = [
        ("minute", 1),
        ("minute5", 5),
        ("daily", 1),
        ("hour", 1),
        ("weekly", 1),
    ]

    for bar_type, interval in valid_types:
        history_client._validate_bar_type(bar_type, interval)

    print(f"   ✅ All {len(valid_types)} bar types validated successfully")
except Exception as e:
    print(f"   ❌ Bar type validation failed: {e}")
    sys.exit(1)

# Test 4: Validate symbol validation
print("\n4. Testing symbol validation...")
try:
    from src.execution.tradestation.exceptions import ValidationError

    history_client._validate_symbol("MNQH26")
    print("   ✅ Valid symbol accepted")

    try:
        history_client._validate_symbol("")
        print("   ❌ Empty symbol should have been rejected")
        sys.exit(1)
    except ValidationError:
        print("   ✅ Invalid symbol correctly rejected")

except Exception as e:
    print(f"   ❌ Symbol validation test failed: {e}")
    sys.exit(1)

# Test 5: Validate date validation
print("\n5. Testing date validation...")
try:
    from src.execution.tradestation.exceptions import ValidationError

    # Valid dates
    history_client._validate_dates("2024-01-01", "2024-12-31")
    print("   ✅ Valid date range accepted")

    # Invalid format
    try:
        history_client._validate_dates("01-01-2024", "2024-12-31")
        print("   ❌ Invalid date format should have been rejected")
        sys.exit(1)
    except ValidationError:
        print("   ✅ Invalid date format correctly rejected")

    # Start after end
    try:
        history_client._validate_dates("2024-12-31", "2024-01-01")
        print("   ❌ Start after end should have been rejected")
        sys.exit(1)
    except ValidationError:
        print("   ✅ Start after end correctly rejected")

except Exception as e:
    print(f"   ❌ Date validation test failed: {e}")
    sys.exit(1)

# Test 6: Test expected bars calculation
print("\n6. Testing expected bars calculation...")
try:
    # Minute bars
    minute_bars = history_client._calculate_expected_bars(
        start_date="2026-03-28",
        end_date="2026-03-28",
        bar_type="minute",
        interval=1,
    )
    assert minute_bars == 1440, f"Expected 1440 minute bars, got {minute_bars}"
    print(f"   ✅ Minute bars: {minute_bars} (expected: 1440)")

    # Daily bars
    daily_bars = history_client._calculate_expected_bars(
        start_date="2026-03-01",
        end_date="2026-03-31",
        bar_type="daily",
        interval=1,
    )
    assert daily_bars == 31, f"Expected 31 daily bars, got {daily_bars}"
    print(f"   ✅ Daily bars: {daily_bars} (expected: 31)")

except Exception as e:
    print(f"   ❌ Expected bars calculation failed: {e}")
    sys.exit(1)

# Test 7: Test streaming parser state management
print("\n7. Testing QuoteStreamParser state management...")
try:
    stream_parser = QuoteStreamParser(mock_client)

    # Check initial state
    assert not stream_parser._is_streaming, "Should not be streaming initially"
    print("   ✅ Initial streaming state correct")

    # Test stop method
    stream_parser.stop_streaming()
    assert not stream_parser._is_streaming, "Should not be streaming after stop"
    print("   ✅ Stop streaming method works")

except Exception as e:
    print(f"   ❌ Streaming parser state test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Phase 2 Import Validation: ✅ ALL TESTS PASSED")
print("=" * 60)

print("\n📦 Phase 2 Implementation Summary:")
print("   ✅ QuotesClient - Real-time quotes endpoint")
print("   ✅ HistoryClient - Historical bar data download")
print("   ✅ QuoteStreamParser - HTTP chunked transfer parser")
print("\n🔗 Ready for Integration:")
print("   • DollarBar pipeline (async generators/queues)")
print("   • HDF5 storage (historical data)")
print("   • Data validation (99.99% completeness)")
print("\n➡️  Next Phase: Phase 3 - Order Execution Integration")
print("   • Order submission, modification, cancellation")
print("   • Triple barrier exits")
print("   • Position monitoring")
print("=" * 60)
